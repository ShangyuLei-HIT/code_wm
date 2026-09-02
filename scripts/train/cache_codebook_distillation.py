"""Build the only teacher-dependent artifact used by joint distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from stable_pretraining import data as dt
from torch.utils.data import DataLoader, Dataset

from stable_worldmodel.wm.vq_lewm.distillation import (
    load_codebook_weights,
    resolve_weights_path,
    sha256_file,
    squared_distances,
)
from stable_worldmodel.wm.utils import load_pretrained


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/vq_lewm_joint_distillation.yaml',
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group('nccl', device_id=device)
    return rank, world_size, local_rank, device


def workers_for_rank(total: int, rank: int, world_size: int) -> int:
    return total // world_size + int(rank < total % world_size)


def canonical_hash(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def image_preprocessor(img_size: int):
    to_image = dt.transforms.ToImage(
        **dt.dataset_stats.ImageNet,
        source='pixels',
        target='pixels',
    )
    resize = dt.transforms.Resize(
        img_size, source='pixels', target='pixels'
    )
    return spt.data.transforms.Compose(to_image, resize)


class CacheShard(Dataset):
    def __init__(
        self,
        dataset,
        split_indices: np.ndarray,
        rank: int,
        world_size: int,
    ) -> None:
        self.dataset = dataset
        self.split_indices = split_indices
        self.rank = rank
        self.world_size = world_size
        remaining = max(0, len(split_indices) - rank)
        self.length = (remaining + world_size - 1) // world_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cache_position = self.rank + index * self.world_size
        dataset_position = int(self.split_indices[cache_position])
        sample = self.dataset[dataset_position]
        return sample['pixels'], cache_position


def file_specs(cfg, train_count: int, val_count: int) -> dict:
    steps = int(cfg.data.num_steps)
    dim = int(cfg.codebook.embedding_dim)
    topk = int(cfg.codebook.topk)
    k = int(cfg.codebook.num_embeddings)
    index_dtype = 'uint16' if k <= np.iinfo(np.uint16).max else 'uint32'
    specs = {}
    for split, count in (('train', train_count), ('val', val_count)):
        specs[f'{split}_teacher_latents.npy'] = {
            'shape': [count, steps, dim],
            'dtype': 'float16',
        }
        prefix = f'k{k}/{split}'
        specs[f'{prefix}_hard_tokens.npy'] = {
            'shape': [count, steps],
            'dtype': index_dtype,
        }
        specs[f'{prefix}_topk_indices.npy'] = {
            'shape': [count, steps, topk],
            'dtype': index_dtype,
        }
        specs[f'{prefix}_topk_probs.npy'] = {
            'shape': [count, steps, topk],
            'dtype': 'float16',
        }
    return specs


def expected_metadata(
    cfg,
    dataset_length: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> dict:
    teacher_weights = resolve_weights_path(cfg.paths.teacher_checkpoint)
    codebook_weights = resolve_weights_path(cfg.paths.codebook_checkpoint)
    stats = {
        key: [float(item) for item in value]
        for key, value in dt.dataset_stats.ImageNet.items()
    }
    metadata = {
        'format_version': 1,
        'teacher_checkpoint': str(
            Path(cfg.paths.teacher_checkpoint).expanduser().resolve()
        ),
        'teacher_sha256': sha256_file(teacher_weights),
        'codebook_checkpoint': str(
            Path(cfg.paths.codebook_checkpoint).expanduser().resolve()
        ),
        'codebook_sha256': sha256_file(codebook_weights),
        'dataset': cfg.data.dataset,
        'dataset_cache': str(
            Path(cfg.paths.dataset_cache).expanduser().resolve()
        ),
        'dataset_length': dataset_length,
        'split_seed': int(cfg.data.split_seed),
        'train_split': float(cfg.data.train_split),
        'sample_indices': {
            'train': {
                'file': 'train_indices.npy',
                'count': len(train_indices),
                'sha256': array_hash(train_indices),
            },
            'val': {
                'file': 'val_indices.npy',
                'count': len(val_indices),
                'sha256': array_hash(val_indices),
            },
        },
        'normalization': {'name': 'ImageNet', **stats},
        'resize': int(cfg.data.img_size),
        'frameskip': int(cfg.data.frameskip),
        'num_steps': int(cfg.data.num_steps),
        'codebook_size': int(cfg.codebook.num_embeddings),
        'latent_dimension': int(cfg.codebook.embedding_dim),
        'topk': int(cfg.codebook.topk),
        'temperature': float(cfg.codebook.temperature),
        'files': file_specs(cfg, len(train_indices), len(val_indices)),
    }
    metadata['metadata_sha256'] = canonical_hash(metadata)
    return metadata


def validate_existing(cache_dir: Path, expected: dict) -> bool:
    metadata_path = cache_dir / 'metadata.json'
    if not metadata_path.exists():
        return False
    actual = json.loads(metadata_path.read_text())
    if actual != expected:
        expected_hash = expected['metadata_sha256']
        actual_hash = actual.get('metadata_sha256', '<missing>')
        raise RuntimeError(
            'distillation cache metadata mismatch: '
            f'expected {expected_hash}, found {actual_hash}'
        )
    for relative, spec in expected['files'].items():
        path = cache_dir / relative
        if not path.exists():
            raise RuntimeError(f'cache metadata references missing file: {path}')
        array = np.load(path, mmap_mode='r')
        if list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']:
            raise RuntimeError(f'cache array shape/dtype mismatch: {path}')
    for split in ('train', 'val'):
        item = expected['sample_indices'][split]
        array = np.load(cache_dir / item['file'])
        if array_hash(array) != item['sha256']:
            raise RuntimeError(f'{split} sample-index hash mismatch')
    return True


def initialize_arrays(cache_dir: Path, metadata: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for relative, spec in metadata['files'].items():
        path = cache_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(
            path,
            mode='w+',
            dtype=np.dtype(spec['dtype']),
            shape=tuple(spec['shape']),
        ).flush()


def cache_split(
    split: str,
    cfg,
    dataset,
    split_indices: np.ndarray,
    cache_dir: Path,
    teacher,
    codebook: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    shard = CacheShard(dataset, split_indices, rank, world_size)
    workers = workers_for_rank(
        int(cfg.data.cpu_workers_total), rank, world_size
    )
    loader_kwargs = {
        'batch_size': int(cfg.data.cache_batch_size_per_gpu),
        'shuffle': False,
        'num_workers': workers,
        'drop_last': False,
        'pin_memory': bool(cfg.data.pin_memory),
        'persistent_workers': bool(cfg.data.persistent_workers) and workers > 0,
    }
    if workers:
        loader_kwargs['prefetch_factor'] = int(cfg.data.prefetch_factor)
    loader = DataLoader(shard, **loader_kwargs)

    k = int(cfg.codebook.num_embeddings)
    latent_map = np.load(
        cache_dir / f'{split}_teacher_latents.npy', mmap_mode='r+'
    )
    hard_map = np.load(
        cache_dir / f'k{k}/{split}_hard_tokens.npy', mmap_mode='r+'
    )
    topk_index_map = np.load(
        cache_dir / f'k{k}/{split}_topk_indices.npy', mmap_mode='r+'
    )
    topk_prob_map = np.load(
        cache_dir / f'k{k}/{split}_topk_probs.npy', mmap_mode='r+'
    )

    with torch.inference_mode():
        for batch_index, (pixels, positions) in enumerate(loader):
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                latent = teacher.encode({'pixels': pixels})['emb']
            flat = latent.reshape(-1, latent.size(-1)).float()
            distances = squared_distances(flat, codebook)
            values, indices = distances.topk(
                int(cfg.codebook.topk), largest=False, dim=-1
            )
            probabilities = torch.softmax(
                -values / float(cfg.codebook.temperature), dim=-1
            )
            shape = (pixels.size(0), int(cfg.data.num_steps), -1)
            positions = positions.numpy()
            latent_map[positions] = latent.float().cpu().numpy().astype(np.float16)
            topk_index_map[positions] = (
                indices.view(shape).cpu().numpy().astype(topk_index_map.dtype)
            )
            topk_prob_map[positions] = (
                probabilities.view(shape).cpu().numpy().astype(np.float16)
            )
            hard_map[positions] = (
                indices[:, 0]
                .view(pixels.size(0), int(cfg.data.num_steps))
                .cpu()
                .numpy()
                .astype(hard_map.dtype)
            )
            if rank == 0 and batch_index % 50 == 0:
                done = min((batch_index + 1) * loader.batch_size, len(shard))
                print(
                    f'cache/{split}: rank0 {done:,}/{len(shard):,}',
                    flush=True,
                )
    for array in (latent_map, hard_map, topk_index_map, topk_prob_map):
        array.flush()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    rank, world_size, _, device = setup_distributed()
    random.seed(int(cfg.seed) + rank)
    np.random.seed(int(cfg.seed) + rank)
    torch.manual_seed(int(cfg.seed) + rank)

    os.environ['LOCAL_DATASET_DIR'] = str(cfg.paths.dataset_cache)
    os.environ['STABLEWM_HOME'] = str(cfg.paths.dataset_cache)
    dataset = swm.data.load_dataset(
        cfg.data.dataset,
        transform=None,
        num_steps=int(cfg.data.num_steps),
        frameskip=int(cfg.data.frameskip),
        keys_to_load=['pixels'],
    )
    dataset.transform = image_preprocessor(int(cfg.data.img_size))

    generator = torch.Generator().manual_seed(int(cfg.data.split_seed))
    permutation = torch.randperm(len(dataset), generator=generator).numpy()
    train_count = int(len(dataset) * float(cfg.data.train_split))
    train_indices = permutation[:train_count].astype(np.int64, copy=False)
    val_indices = permutation[train_count:].astype(np.int64, copy=False)
    if bool(cfg.smoke.enabled):
        train_indices = train_indices[: int(cfg.smoke.train_samples)]
        val_indices = val_indices[: int(cfg.smoke.val_samples)]

    cache_dir = Path(cfg.paths.cache_dir).expanduser().resolve()
    metadata = expected_metadata(
        cfg, len(dataset), train_indices, val_indices
    )
    status = [None]
    if rank == 0:
        try:
            if validate_existing(cache_dir, metadata):
                status[0] = 'ready'
            else:
                initialize_arrays(cache_dir, metadata)
                np.save(cache_dir / 'train_indices.npy', train_indices)
                np.save(cache_dir / 'val_indices.npy', val_indices)
                status[0] = 'build'
        except Exception as error:
            status[0] = f'error:{type(error).__name__}:{error}'
    if world_size > 1:
        dist.broadcast_object_list(status, src=0)
        dist.barrier()
    if status[0].startswith('error:'):
        raise RuntimeError(status[0])
    if status[0] == 'ready':
        if rank == 0:
            print(f'Validated existing teacher cache: {cache_dir}', flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return

    original_codebook_hash = metadata['codebook_sha256']
    codebook = load_codebook_weights(
        cfg.paths.codebook_checkpoint,
        weight_key=cfg.codebook.weight_key,
        expected_shape=(
            int(cfg.codebook.num_embeddings),
            int(cfg.codebook.embedding_dim),
        ),
    ).to(device)
    teacher = load_pretrained(cfg.paths.teacher_checkpoint).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    for split, indices in (('train', train_indices), ('val', val_indices)):
        cache_split(
            split,
            cfg,
            dataset,
            indices,
            cache_dir,
            teacher,
            codebook,
            rank,
            world_size,
            device,
        )
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        current_hash = sha256_file(
            resolve_weights_path(cfg.paths.codebook_checkpoint)
        )
        if current_hash != original_codebook_hash:
            raise RuntimeError('frozen codebook checkpoint changed during caching')
        (cache_dir / 'metadata.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + '\n'
        )
        print(
            f'Offline teacher cache complete: {cache_dir} '
            f'({metadata["metadata_sha256"]})',
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
