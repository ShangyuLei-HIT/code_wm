"""Create the complete teacher-dependent cache for two-task distillation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict
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
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='override a config key, e.g. --set codebook.temperature=2.0',
    )
    parser.add_argument(
        '--save-config',
        default=None,
        metavar='PATH',
        help='save the resolved config (after --set overrides) to this path',
    )
    return parser.parse_args()


def parse_override_value(raw: str):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def apply_overrides(cfg, overrides: list[str]):
    """Apply repeated ``--set key.subkey=value`` overrides to a config."""
    for item in overrides:
        key, separator, raw = item.partition('=')
        if not separator or not key:
            raise ValueError(f'--set expects KEY=VALUE, got {item!r}')
        with open_dict(cfg):
            OmegaConf.update(cfg, key, parse_override_value(raw), merge=True)
    return cfg


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


# Metadata fields added after the first cache revision. Their values are
# implied by alignment_checkpoint/alignment_sha256 (which older caches still
# record), so they are compared only when the existing cache declares them.
LEGACY_OPTIONAL_METADATA_FIELDS = ('alignment_mode', 'reference_task')


def metadata_hash_payload(
    metadata: dict, *, include_alignment_fields: bool
) -> dict:
    """The field set covered by ``metadata_sha256`` for one schema revision."""
    return {
        key: value
        for key, value in metadata.items()
        if key != 'metadata_sha256'
        and (
            include_alignment_fields
            or key not in LEGACY_OPTIONAL_METADATA_FIELDS
        )
    }


def validate_existing(root: Path, expected: dict) -> bool:
    path = root / 'metadata.json'
    if not path.exists():
        return False
    actual = json.loads(path.read_text())
    stored_sha = actual.get('metadata_sha256')
    if stored_sha not in (
        canonical_hash(
            metadata_hash_payload(actual, include_alignment_fields=True)
        ),
        canonical_hash(
            metadata_hash_payload(actual, include_alignment_fields=False)
        ),
    ):
        raise RuntimeError(
            'multitask cache metadata is corrupt: stored metadata_sha256 '
            f'{stored_sha!r} matches neither the current nor the legacy '
            f'schema field set of {path}'
        )
    differences = [
        f'{field}: {actual[field]!r} != {expected.get(field)!r}'
        for field in LEGACY_OPTIONAL_METADATA_FIELDS
        if field in actual and actual[field] != expected.get(field)
    ]
    legacy_actual = metadata_hash_payload(
        actual, include_alignment_fields=False
    )
    legacy_expected = metadata_hash_payload(
        expected, include_alignment_fields=False
    )
    if legacy_actual != legacy_expected:
        differing = sorted(
            str(key)
            for key in set(legacy_actual) | set(legacy_expected)
            if legacy_actual.get(key) != legacy_expected.get(key)
        )
        differences.append(
            'content-defining fields differ (' + ', '.join(differing) + ')'
        )
    if differences:
        raise RuntimeError(
            'multitask cache metadata mismatch: '
            f'{stored_sha} != {expected.get("metadata_sha256")} '
            f'({"; ".join(differences)})'
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
    alignment_mode = str(cfg.alignment.get('mode', 'similarity') or 'similarity')
    reference_task = str(cfg.tasks[int(cfg.alignment.reference_task_id)].name)
    declared_reference = cfg.alignment.get('reference_task')
    if declared_reference is not None and str(declared_reference) != reference_task:
        raise ValueError(
            f'alignment.reference_task {str(declared_reference)!r} conflicts with '
            f'alignment.reference_task_id '
            f'{int(cfg.alignment.reference_task_id)} (= {reference_task!r})'
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
        'alignment_mode': alignment_mode,
        'reference_task': reference_task,
        'resize': int(cfg.data.img_size),
        'num_steps': int(cfg.data.num_steps),
        'topk': int(cfg.codebook.topk),
        'temperature': float(cfg.codebook.temperature),
        'files': files,
    }
    metadata['metadata_sha256'] = canonical_hash(metadata)
    return metadata


def reduce_split_stats(pieces: list[dict]) -> dict:
    """Combine per-rank streaming statistics into exact split statistics."""
    frames = sum(int(piece['frames']) for piece in pieces)
    if frames <= 0:
        raise ValueError('split produced no frames; refusing empty statistics')
    norm_sum = sum(float(piece['norm_sum']) for piece in pieces)
    squared_norm_sum = sum(
        float(piece['squared_norm_sum']) for piece in pieces
    )
    top1 = np.concatenate(
        [
            np.asarray(piece['top1_sqdist'], dtype=np.float64)
            for piece in pieces
        ]
    )
    mean_norm = norm_sum / frames
    variance = max(squared_norm_sum / frames - mean_norm**2, 0.0)
    return {
        'frames': frames,
        'latent_norm_mean': mean_norm,
        'latent_norm_std_from_second_moment': math.sqrt(variance),
        'top1_sqdist_median': (
            float(np.median(top1)) if top1.size else float('nan')
        ),
        'top1_sqdist_mean': (
            float(top1.mean()) if top1.size else float('nan')
        ),
    }


def build_cache_stats(cfg, codebook: torch.Tensor, task_stats: dict) -> dict:
    temperature = float(cfg.codebook.temperature)
    codebook_norms = codebook.detach().float().norm(dim=-1).double()
    tasks_payload = {}
    for name, splits in task_stats.items():
        tasks_payload[str(name)] = {}
        for split, entry in splits.items():
            median = float(entry['top1_sqdist_median'])
            mean = float(entry['top1_sqdist_mean'])
            tasks_payload[str(name)][str(split)] = {
                'frames': int(entry['frames']),
                'latent_norm_mean': float(entry['latent_norm_mean']),
                'latent_norm_std_from_second_moment': float(
                    entry['latent_norm_std_from_second_moment']
                ),
                'top1_sqdist_median': median,
                'top1_sqdist_mean': mean,
                # Per-frame (sum over latent dims) squared L2 — NOT dim
                # normalized like fused_codebook.quantization_mse.
                'quantization_mse_per_frame': mean,
                'temperature_effective': temperature / max(median, 1e-12),
                'task_loss_scale_reference': mean,
            }
    return {
        'temperature': temperature,
        'codebook_norm_mean': float(codebook_norms.mean()),
        'codebook_norm_std': float(codebook_norms.std(unbiased=False)),
        'tasks': tasks_payload,
    }


def atomic_write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    )
    os.replace(temporary, path)


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
) -> dict[str, dict | None]:
    """Fill one task's arrays and return per-split reduced statistics.

    Every rank accumulates streaming statistics for its deterministic shard
    (order is fixed because sharding and loader order are deterministic).
    The top-1 squared distances are gathered in full so rank 0 can compute
    the exact median/mean; non-zero ranks receive None placeholders.
    """
    workers = int(cfg.data.cpu_workers_total) // world_size
    reduced_stats: dict[str, dict | None] = {}
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
        frames = 0
        norm_sum = torch.zeros((), dtype=torch.float64)
        squared_norm_sum = torch.zeros((), dtype=torch.float64)
        top1_chunks: list[np.ndarray] = []
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
            norms = flat.norm(dim=-1)
            frames += int(norms.numel())
            norm_sum += norms.double().sum().cpu()
            squared_norm_sum += norms.square().double().sum().cpu()
            top1_chunks.append(
                values[:, 0].detach().cpu().numpy().astype(np.float64)
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
        local_stats = {
            'frames': frames,
            'norm_sum': float(norm_sum),
            'squared_norm_sum': float(squared_norm_sum),
            'top1_sqdist': (
                np.concatenate(top1_chunks)
                if top1_chunks
                else np.zeros(0, dtype=np.float64)
            ),
        }
        if world_size > 1:
            gathered: list = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_stats)
            reduced_stats[split] = (
                reduce_split_stats(gathered) if rank == 0 else None
            )
        else:
            reduced_stats[split] = reduce_split_stats([local_stats])
    return reduced_stats


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args.set)
    if args.cpu_workers_total is not None:
        cfg.data.cpu_workers_total = max(0, int(args.cpu_workers_total))
    rank, world_size, device = setup_distributed()
    if args.save_config is not None and rank == 0:
        target = Path(args.save_config).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, target)
        print(f'Saved resolved config to {target}', flush=True)
    if world_size > 1:
        dist.barrier()
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
            existing_metadata = json.loads(
                (root / 'metadata.json').read_text()
            )
            legacy_fields = [
                field
                for field in LEGACY_OPTIONAL_METADATA_FIELDS
                if field not in existing_metadata
            ]
            if legacy_fields:
                print(
                    f'NOTE: {root}/metadata.json predates '
                    f'{", ".join(legacy_fields)}; content-defining fields '
                    'matched, so the cache is reused unchanged',
                    flush=True,
                )
            if not (root / 'cache_stats.json').exists():
                print(
                    f'WARNING: {root}/cache_stats.json is absent (cache built '
                    'by an older script revision or interrupted); statistics '
                    'were NOT recomputed for the validated cache',
                    flush=True,
                )
        if world_size > 1:
            dist.destroy_process_group()
        return

    task_stats: dict[str, dict[str, dict]] = {}
    for task_id, (task, dataset, splits) in enumerate(
        zip(cfg.tasks, datasets, task_splits, strict=True)
    ):
        teacher = load_pretrained(task.teacher_checkpoint).to(device)
        teacher.requires_grad_(False)
        teacher.eval()
        reduced = cache_task(
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
        if rank == 0:
            for split, entry in reduced.items():
                if entry is None:
                    raise RuntimeError(
                        f'rank 0 received no reduced statistics for '
                        f'{task.name}/{split}'
                    )
            task_stats[str(task.name)] = reduced
        del teacher
        torch.cuda.empty_cache()
        if world_size > 1:
            dist.barrier()
    if rank == 0:
        # cache_stats.json is written before metadata.json so that any cache
        # declared complete by metadata.json always carries its statistics.
        stats = build_cache_stats(cfg, codebook, task_stats)
        atomic_write_json(stats, root / 'cache_stats.json')
        print(f'Wrote cache statistics to {root}/cache_stats.json', flush=True)
        (root / 'metadata.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + '\n'
        )
        print(f'Multitask cache complete: {root}', flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
