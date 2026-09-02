"""Create the complete teacher-dependent cache for two-task distillation."""

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

from stable_worldmodel.wm.utils import load_pretrained
from stable_worldmodel.wm.vq_lewm.alignment import (
    load_alignment_bundle,
)
from stable_worldmodel.wm.vq_lewm.distillation import (
    load_codebook_weights,
    resolve_weights_path,
    sha256_file,
    squared_distances,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/multitask_vq_lewm.yaml',
    )
    parser.add_argument('--cpu-workers-total', type=int, default=None)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group('nccl', device_id=device)
    return rank, world_size, device


def canonical_hash(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def image_preprocessor(img_size: int):
    return spt.data.transforms.Compose(
        dt.transforms.ToImage(
            **dt.dataset_stats.ImageNet,
            source='pixels',
            target='pixels',
        ),
        dt.transforms.Resize(img_size, source='pixels', target='pixels'),
    )


class CacheShard(Dataset):
    def __init__(self, dataset, indices, rank: int, world_size: int):
        self.dataset = dataset
        self.indices = indices
        self.positions = np.arange(rank, len(indices), world_size)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        position = int(self.positions[index])
        sample = self.dataset[int(self.indices[position])]
        return sample['pixels'], position


@torch.no_grad()
def nearest_topk(
    latent: torch.Tensor,
    codebook: torch.Tensor,
    *,
    k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    best_values = torch.full(
        (len(latent), k), float('inf'), device=latent.device
    )
    best_indices = torch.zeros(
        (len(latent), k), dtype=torch.long, device=latent.device
    )
    for start in range(0, len(codebook), chunk_size):
        distances = squared_distances(latent, codebook[start : start + chunk_size])
        local_k = min(k, distances.size(1))
        values, indices = distances.topk(local_k, largest=False, dim=-1)
        candidates = torch.cat((best_values, values), dim=1)
        candidate_indices = torch.cat(
            (best_indices, indices + start), dim=1
        )
        best_values, order = candidates.topk(k, largest=False, dim=1)
        best_indices = candidate_indices.gather(1, order)
    return best_values, best_indices


def split_indices(length: int, task_id: int, cfg) -> tuple[np.ndarray, np.ndarray]:
    generator = torch.Generator().manual_seed(
        int(cfg.data.split_seed) + task_id
    )
    indices = torch.randperm(length, generator=generator).numpy()
    train_count = int(length * float(cfg.data.train_fraction))
    train = indices[:train_count].astype(np.int64, copy=False)
    validation = indices[train_count:].astype(np.int64, copy=False)
    if bool(cfg.smoke.enabled):
        train = train[: int(cfg.smoke.train_samples_per_task)]
        validation = validation[: int(cfg.smoke.validation_samples_per_task)]
    return train, validation


def task_files(task_name: str, split: str, count: int, cfg, k: int) -> dict:
    prefix = f'tasks/{task_name}'
    steps = int(cfg.data.num_steps)
    dim = int(cfg.codebook.embedding_dim)
    topk = int(cfg.codebook.topk)
    index_dtype = 'uint16' if k <= np.iinfo(np.uint16).max else 'uint32'
    return {
        f'{prefix}/{split}_teacher_latents.npy': {
            'shape': [count, steps, dim], 'dtype': 'float16'
        },
        f'{prefix}/{split}_hard_tokens.npy': {
            'shape': [count, steps], 'dtype': index_dtype
        },
        f'{prefix}/{split}_topk_indices.npy': {
            'shape': [count, steps, topk], 'dtype': index_dtype
        },
        f'{prefix}/{split}_topk_probs.npy': {
            'shape': [count, steps, topk], 'dtype': 'float16'
        },
    }


def initialize_arrays(root: Path, metadata: dict) -> None:
    for relative, spec in metadata['files'].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(
            path,
            mode='w+',
            dtype=np.dtype(spec['dtype']),
            shape=tuple(spec['shape']),
        ).flush()


def validate_existing(root: Path, expected: dict) -> bool:
    path = root / 'metadata.json'
    if not path.exists():
        return False
    actual = json.loads(path.read_text())
    if actual != expected:
        raise RuntimeError(
            'multitask cache metadata mismatch: '
            f'{actual.get("metadata_sha256")} != '
            f'{expected.get("metadata_sha256")}'
        )
    for relative, spec in expected['files'].items():
        array = np.load(root / relative, mmap_mode='r')
        if list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']:
            raise RuntimeError(f'invalid cached array: {relative}')
    return True


def metadata_for(cfg, task_records: list[dict], codebook: torch.Tensor) -> dict:
    weights = resolve_weights_path(cfg.paths.fused_codebook_checkpoint)
    fusion_metadata_path = weights.with_name('metadata.json')
    fusion_metadata = (
        json.loads(fusion_metadata_path.read_text())
        if fusion_metadata_path.exists()
        else None
    )
    files = {}
    for record in task_records:
        files.update(
            task_files(
                record['name'], 'train', record['train_count'], cfg, len(codebook)
            )
        )
        files.update(
            task_files(
                record['name'],
                'validation',
                record['validation_count'],
                cfg,
                len(codebook),
            )
        )
    metadata = {
        'format_version': 1,
        'task_id_to_name': {
            str(index): record['name']
            for index, record in enumerate(task_records)
        },
        'tasks': task_records,
        'fused_codebook_checkpoint': str(weights),
        'fused_codebook_sha256': sha256_file(weights),
        'num_embeddings': len(codebook),
        'latent_dimension': codebook.size(1),
        'fusion_metadata': fusion_metadata,
        'alignment_checkpoint': str(
            Path(cfg.paths.alignment_checkpoint).expanduser().resolve()
        ),
        'alignment_sha256': sha256_file(cfg.paths.alignment_checkpoint),
        'resize': int(cfg.data.img_size),
        'num_steps': int(cfg.data.num_steps),
        'topk': int(cfg.codebook.topk),
        'temperature': float(cfg.codebook.temperature),
        'files': files,
    }
    metadata['metadata_sha256'] = canonical_hash(metadata)
    return metadata


def cache_task(
    cfg,
    task,
    task_id: int,
    dataset,
    splits: dict[str, np.ndarray],
    teacher,
    aligner,
    codebook,
    root,
    rank,
    world_size,
    device,
) -> None:
    workers = int(cfg.data.cpu_workers_total) // world_size
    for split, indices in splits.items():
        shard = CacheShard(dataset, indices, rank, world_size)
        loader = DataLoader(
            shard,
            batch_size=int(cfg.data.cache_batch_size_per_gpu),
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        prefix = root / 'tasks' / str(task.name)
        latent_map = np.load(
            prefix / f'{split}_teacher_latents.npy', mmap_mode='r+'
        )
        hard_map = np.load(
            prefix / f'{split}_hard_tokens.npy', mmap_mode='r+'
        )
        index_map = np.load(
            prefix / f'{split}_topk_indices.npy', mmap_mode='r+'
        )
        probability_map = np.load(
            prefix / f'{split}_topk_probs.npy', mmap_mode='r+'
        )
        for batch_index, (pixels, positions) in enumerate(loader):
            pixels = pixels.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                latent = teacher.encode({'pixels': pixels})['emb']
            latent = aligner(latent.float()) if aligner is not None else latent.float()
            flat = latent.reshape(-1, latent.size(-1))
            values, indices_batch = nearest_topk(
                flat,
                codebook,
                k=int(cfg.codebook.topk),
                chunk_size=int(cfg.codebook.distance_chunk_size),
            )
            probabilities = torch.softmax(
                -values / float(cfg.codebook.temperature), dim=-1
            )
            shape = (len(pixels), int(cfg.data.num_steps), -1)
            positions = positions.numpy()
            latent_map[positions] = latent.cpu().numpy().astype(np.float16)
            hard_map[positions] = (
                indices_batch[:, 0]
                .view(len(pixels), int(cfg.data.num_steps))
                .cpu()
                .numpy()
                .astype(hard_map.dtype)
            )
            index_map[positions] = (
                indices_batch.view(shape).cpu().numpy().astype(index_map.dtype)
            )
            probability_map[positions] = (
                probabilities.view(shape).cpu().numpy().astype(np.float16)
            )
            if rank == 0 and batch_index % 50 == 0:
                print(
                    f'cache/{task.name}/{split}: '
                    f'{min((batch_index + 1) * loader.batch_size, len(shard)):,}'
                    f'/{len(shard):,}',
                    flush=True,
                )
        for array in (latent_map, hard_map, index_map, probability_map):
            array.flush()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.cpu_workers_total is not None:
        cfg.data.cpu_workers_total = max(0, int(args.cpu_workers_total))
    rank, world_size, device = setup_distributed()
    seed = int(cfg.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['LOCAL_DATASET_DIR'] = str(cfg.paths.dataset_cache)
    os.environ['STABLEWM_HOME'] = str(cfg.paths.dataset_cache)
    codebook = load_codebook_weights(
        cfg.paths.fused_codebook_checkpoint
    ).to(device)
    alignments = load_alignment_bundle(
        cfg.paths.alignment_checkpoint,
        expected_dim=codebook.size(1),
    )
    alignments = {name: value.to(device) for name, value in alignments.items()}
    datasets = []
    task_records = []
    task_splits = []
    for task_id, task in enumerate(cfg.tasks):
        dataset = swm.data.load_dataset(
            task.dataset,
            transform=None,
            num_steps=int(cfg.data.num_steps),
            frameskip=int(task.frameskip),
            keys_to_load=['pixels'],
        )
        dataset.transform = image_preprocessor(int(cfg.data.img_size))
        train, validation = split_indices(len(dataset), task_id, cfg)
        task_root = Path(cfg.paths.multitask_cache_dir) / 'tasks' / str(task.name)
        task_root.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            np.save(task_root / 'train_indices.npy', train)
            np.save(task_root / 'validation_indices.npy', validation)
        datasets.append(dataset)
        task_splits.append({'train': train, 'validation': validation})
        task_records.append(
            {
                'task_id': task_id,
                'name': str(task.name),
                'dataset': str(task.dataset),
                'dataset_length': len(dataset),
                'frameskip': int(task.frameskip),
                'teacher_checkpoint': str(
                    Path(task.teacher_checkpoint).expanduser().resolve()
                ),
                'teacher_sha256': sha256_file(
                    resolve_weights_path(task.teacher_checkpoint)
                ),
                'original_codebook_checkpoint': str(
                    Path(task.codebook_checkpoint).expanduser().resolve()
                ),
                'original_codebook_sha256': sha256_file(
                    resolve_weights_path(task.codebook_checkpoint)
                ),
                'train_count': len(train),
                'validation_count': len(validation),
                'train_indices_sha256': array_hash(train),
                'validation_indices_sha256': array_hash(validation),
            }
        )
    root = Path(cfg.paths.multitask_cache_dir).expanduser().resolve()
    metadata = metadata_for(cfg, task_records, codebook)
    status = [None]
    if rank == 0:
        try:
            if validate_existing(root, metadata):
                status[0] = 'ready'
            else:
                initialize_arrays(root, metadata)
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
            print(f'Validated existing cache: {root}', flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return

    for task_id, (task, dataset, splits) in enumerate(
        zip(cfg.tasks, datasets, task_splits, strict=True)
    ):
        teacher = load_pretrained(task.teacher_checkpoint).to(device)
        teacher.requires_grad_(False)
        teacher.eval()
        cache_task(
            cfg,
            task,
            task_id,
            dataset,
            splits,
            teacher,
            (
                alignments.get(str(task.name))
                if bool(cfg.alignment.get('enabled', True))
                else None
            ),
            codebook,
            root,
            rank,
            world_size,
            device,
        )
        del teacher
        torch.cuda.empty_cache()
        if world_size > 1:
            dist.barrier()
    if rank == 0:
        (root / 'metadata.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + '\n'
        )
        print(f'Multitask cache complete: {root}', flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
