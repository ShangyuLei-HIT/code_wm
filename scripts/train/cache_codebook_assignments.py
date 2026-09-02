"""Build codebook assignments from an existing immutable teacher-latent cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from scripts.train.cache_codebook_distillation import (
    array_hash,
    canonical_hash,
    file_specs,
    validate_existing,
)
from stable_worldmodel.wm.vq_lewm.affine import load_rigid_latent_transform
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
        default='scripts/train/config/vq_lewm_joint_distillation.yaml',
    )
    return parser.parse_args()


def setup_distributed():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group('nccl', device_id=device)
    return rank, world_size, device


def relative_symlink(target: Path, link: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return
        raise RuntimeError(f'conflicting symlink: {link}')
    if link.exists():
        if link.samefile(target):
            return
        raise RuntimeError(f'conflicting cache file: {link}')
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent))


def transform_metadata(cfg) -> dict | None:
    section = cfg.get('latent_transform')
    if not section or not bool(section.get('enabled', False)):
        return None
    checkpoint = Path(section.checkpoint).expanduser().resolve()
    return {
        'checkpoint': str(checkpoint),
        'sha256': sha256_file(checkpoint),
        'initialize_adapter': bool(section.get('initialize_adapter', True)),
        'reuse_base_assignments': bool(
            section.get('reuse_base_assignments', False)
        ),
    }


def expected_metadata(cfg, source: dict) -> dict:
    base_dir = Path(cfg.paths.base_teacher_cache).expanduser().resolve()
    codebook_weights = resolve_weights_path(cfg.paths.codebook_checkpoint)
    counts = {
        split: int(source['sample_indices'][split]['count'])
        for split in ('train', 'val')
    }
    if bool(cfg.smoke.enabled):
        counts = {
            'train': min(counts['train'], int(cfg.smoke.train_samples)),
            'val': min(counts['val'], int(cfg.smoke.val_samples)),
        }
    train_count = counts['train']
    val_count = counts['val']
    metadata = {
        key: source[key]
        for key in (
            'format_version',
            'teacher_checkpoint',
            'teacher_sha256',
            'dataset',
            'dataset_cache',
            'dataset_length',
            'split_seed',
            'train_split',
            'normalization',
            'resize',
            'frameskip',
            'num_steps',
        )
    }
    metadata['sample_indices'] = {}
    for split in ('train', 'val'):
        source_spec = source['sample_indices'][split]
        indices = np.load(base_dir / source_spec['file'])[: counts[split]]
        metadata['sample_indices'][split] = {
            'file': source_spec['file'],
            'count': counts[split],
            'sha256': array_hash(indices),
        }
    metadata.update(
        {
            'base_teacher_cache': str(base_dir),
            'base_cache_metadata_sha256': source['metadata_sha256'],
            'codebook_checkpoint': str(
                Path(cfg.paths.codebook_checkpoint).expanduser().resolve()
            ),
            'codebook_sha256': sha256_file(codebook_weights),
            'codebook_size': int(cfg.codebook.num_embeddings),
            'latent_dimension': int(cfg.codebook.embedding_dim),
            'topk': int(cfg.codebook.topk),
            'temperature': float(cfg.codebook.temperature),
            'latent_transform': transform_metadata(cfg),
            'files': file_specs(cfg, train_count, val_count),
        }
    )
    metadata['metadata_sha256'] = canonical_hash(metadata)
    return metadata


def initialize_cache_links(
    cfg, source: dict, metadata: dict, cache_dir: Path
) -> None:
    base_dir = Path(cfg.paths.base_teacher_cache).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        count = int(metadata['sample_indices'][split]['count'])
        latent_target = cache_dir / f'{split}_teacher_latents.npy'
        index_target = cache_dir / metadata['sample_indices'][split]['file']
        if bool(cfg.smoke.enabled):
            if not latent_target.exists():
                source_latent = np.load(
                    base_dir / f'{split}_teacher_latents.npy',
                    mmap_mode='r',
                )
                np.save(latent_target, np.asarray(source_latent[:count]))
            if not index_target.exists():
                source_indices = np.load(
                    base_dir / source['sample_indices'][split]['file']
                )
                np.save(index_target, source_indices[:count])
        else:
            relative_symlink(
                base_dir / f'{split}_teacher_latents.npy',
                latent_target,
            )
            spec = source['sample_indices'][split]
            relative_symlink(
                base_dir / spec['file'], index_target
            )

    k = int(cfg.codebook.num_embeddings)
    reuse = bool(
        metadata.get('latent_transform')
        and metadata['latent_transform']['reuse_base_assignments']
    )
    for split in ('train', 'val'):
        for suffix in ('hard_tokens', 'topk_indices', 'topk_probs'):
            relative = f'k{k}/{split}_{suffix}.npy'
            path = cache_dir / relative
            if reuse:
                source_path = base_dir / relative
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                relative_symlink(source_path, path)
                continue
            spec = metadata['files'][relative]
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                array = np.load(path, mmap_mode='r')
                if (
                    list(array.shape) != spec['shape']
                    or str(array.dtype) != spec['dtype']
                ):
                    raise RuntimeError(f'incompatible partial cache: {path}')
                continue
            np.lib.format.open_memmap(
                path,
                mode='w+',
                dtype=np.dtype(spec['dtype']),
                shape=tuple(spec['shape']),
            ).flush()


def cache_split(cfg, metadata, split, rank, world_size, device) -> None:
    if (
        metadata.get('latent_transform')
        and metadata['latent_transform']['reuse_base_assignments']
    ):
        return
    cache_dir = Path(cfg.paths.cache_dir).expanduser().resolve()
    count = int(metadata['sample_indices'][split]['count'])
    if bool(cfg.smoke.enabled):
        count = min(
            count,
            int(
                cfg.smoke.train_samples
                if split == 'train'
                else cfg.smoke.val_samples
            ),
        )
    latent = np.load(
        cache_dir / f'{split}_teacher_latents.npy', mmap_mode='r'
    )
    k = int(cfg.codebook.num_embeddings)
    hard = np.load(
        cache_dir / f'k{k}/{split}_hard_tokens.npy', mmap_mode='r+'
    )
    indices_map = np.load(
        cache_dir / f'k{k}/{split}_topk_indices.npy', mmap_mode='r+'
    )
    probabilities_map = np.load(
        cache_dir / f'k{k}/{split}_topk_probs.npy', mmap_mode='r+'
    )
    codebook = load_codebook_weights(
        cfg.paths.codebook_checkpoint,
        weight_key=cfg.codebook.weight_key,
        expected_shape=(
            int(cfg.codebook.num_embeddings),
            int(cfg.codebook.embedding_dim),
        ),
    ).to(device)
    transform = None
    if metadata.get('latent_transform'):
        transform = load_rigid_latent_transform(
            metadata['latent_transform']['checkpoint'],
            expected_dim=int(cfg.codebook.embedding_dim),
        ).to(device)

    batch_size = int(cfg.data.cache_batch_size_per_gpu)
    positions = np.arange(rank, count, world_size, dtype=np.int64)
    with torch.inference_mode():
        for offset in range(0, len(positions), batch_size):
            batch_positions = positions[offset : offset + batch_size]
            batch = torch.from_numpy(
                np.asarray(latent[batch_positions], dtype=np.float32)
            ).to(device)
            if transform is not None:
                batch = transform(batch)
            flat = batch.reshape(-1, batch.size(-1))
            distances = squared_distances(flat, codebook)
            values, indices = distances.topk(
                int(cfg.codebook.topk), largest=False, dim=-1
            )
            probabilities = torch.softmax(
                -values / float(cfg.codebook.temperature), dim=-1
            )
            shape = (len(batch_positions), int(cfg.data.num_steps), -1)
            indices_map[batch_positions] = (
                indices.view(shape).cpu().numpy().astype(indices_map.dtype)
            )
            probabilities_map[batch_positions] = (
                probabilities.view(shape)
                .cpu()
                .numpy()
                .astype(probabilities_map.dtype)
            )
            hard[batch_positions] = (
                indices[:, 0]
                .view(len(batch_positions), int(cfg.data.num_steps))
                .cpu()
                .numpy()
                .astype(hard.dtype)
            )
            if rank == 0 and offset % (50 * batch_size) == 0:
                print(
                    f'assignments/{split}: rank0 '
                    f'{min(offset + batch_size, len(positions)):,}/'
                    f'{len(positions):,}',
                    flush=True,
                )
    hard.flush()
    indices_map.flush()
    probabilities_map.flush()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if not cfg.paths.get('base_teacher_cache'):
        raise ValueError('paths.base_teacher_cache is required')
    rank, world_size, device = setup_distributed()
    base_dir = Path(cfg.paths.base_teacher_cache).expanduser().resolve()
    source_path = base_dir / 'metadata.json'
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = json.loads(source_path.read_text())
    unhashed = dict(source)
    stored_hash = unhashed.pop('metadata_sha256', None)
    if canonical_hash(unhashed) != stored_hash:
        raise RuntimeError('base teacher cache metadata self-hash mismatch')
    if int(source['split_seed']) != int(cfg.data.split_seed):
        raise RuntimeError('base cache split seed differs from experiment')
    if int(source['latent_dimension']) != int(cfg.codebook.embedding_dim):
        raise RuntimeError('base cache latent dimension differs from experiment')

    metadata = expected_metadata(cfg, source)
    cache_dir = Path(cfg.paths.cache_dir).expanduser().resolve()
    status = [None]
    if rank == 0:
        try:
            if validate_existing(cache_dir, metadata):
                status[0] = 'ready'
            else:
                initialize_cache_links(cfg, source, metadata, cache_dir)
                status[0] = 'linked'
        except Exception as error:
            status[0] = f'error:{type(error).__name__}:{error}'
    if world_size > 1:
        dist.broadcast_object_list(status, src=0)
        dist.barrier()
    if status[0].startswith('error:'):
        raise RuntimeError(status[0])
    if status[0] == 'ready':
        if rank == 0:
            print(f'Validated existing assignment cache: {cache_dir}', flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return

    for split in ('train', 'val'):
        cache_split(cfg, metadata, split, rank, world_size, device)
        if world_size > 1:
            dist.barrier()
    if rank == 0:
        (cache_dir / 'metadata.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + '\n'
        )
        if not validate_existing(cache_dir, metadata):
            raise RuntimeError('completed assignment cache failed validation')
        print(
            f'Assignment cache complete: {cache_dir} '
            f'({metadata["metadata_sha256"]})',
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
