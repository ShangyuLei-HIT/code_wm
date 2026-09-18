"""Fit multiple task-specific similarity transforms into one reference space."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from torch.utils.data import DataLoader, Subset

from stable_worldmodel.wm.utils import load_pretrained
from stable_worldmodel.wm.vq_lewm.alignment import (
    SimilarityAlignment,
    alignment_metrics,
    fit_similarity_procrustes,
)
from stable_worldmodel.wm.vq_lewm.distillation import (
    load_codebook_weights,
    nearest_code_indices,
    resolve_weights_path,
    sha256_file,
)


ALIGNMENT_MODES = ('similarity', 'identity', 'center_norm_match')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/multitask_vq_lewm_three_tasks.yaml',
    )
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='override a config key, e.g. --set alignment.mode=identity',
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


def save_config(cfg, path: str | None) -> None:
    if path is None:
        return
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, target)
    print(f'Saved resolved config to {target}', flush=True)


def alignment_mode(cfg) -> str:
    mode = cfg.alignment.get('mode', 'similarity')
    if mode is None:
        mode = 'similarity'
    mode = str(mode)
    if mode not in ALIGNMENT_MODES:
        raise ValueError(
            f'unknown alignment.mode {mode!r}; expected one of {ALIGNMENT_MODES}'
        )
    return mode


def resolve_task_ids(cfg) -> tuple[int, list[int]]:
    """Resolve the reference task id and the ordered list of source ids."""
    names = [str(task.name) for task in cfg.tasks]
    if len(set(names)) != len(names):
        raise ValueError(f'task names must be unique, got {names}')
    if cfg.alignment.get('reference_task') is not None:
        reference_name = str(cfg.alignment.reference_task)
        if reference_name not in names:
            raise ValueError(
                f'alignment.reference_task {reference_name!r} is not one of '
                f'the configured tasks: {names}'
            )
        reference_id = names.index(reference_name)
        source_ids = [
            index for index in range(len(cfg.tasks)) if index != reference_id
        ]
        if not source_ids:
            raise ValueError(
                'alignment.reference_task resolution leaves no source tasks; '
                'at least two tasks are required'
            )
        legacy_reference = cfg.alignment.get('reference_task_id')
        if legacy_reference is not None and int(legacy_reference) != reference_id:
            raise ValueError(
                f'alignment.reference_task_id {int(legacy_reference)} conflicts '
                f'with alignment.reference_task {reference_name!r} '
                f'(resolved index {reference_id})'
            )
        legacy_sources = cfg.alignment.get('source_task_ids')
        if legacy_sources is not None:
            legacy = sorted(int(value) for value in legacy_sources)
            if legacy != sorted(source_ids):
                raise ValueError(
                    f'alignment.source_task_ids {legacy} conflicts with '
                    f'alignment.reference_task {reference_name!r} '
                    f'(resolved sources {sorted(source_ids)})'
                )
        return reference_id, source_ids
    reference_id = int(cfg.alignment.reference_task_id)
    source_ids = [int(value) for value in cfg.alignment.source_task_ids]
    if not source_ids or reference_id in source_ids:
        raise ValueError('source_task_ids must be non-empty and exclude reference')
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('source_task_ids contains duplicates')
    return reference_id, source_ids


def image_preprocessor(img_size: int):
    return spt.data.transforms.Compose(
        dt.transforms.ToImage(
            **dt.dataset_stats.ImageNet,
            source='pixels',
            target='pixels',
        ),
        dt.transforms.Resize(img_size, source='pixels', target='pixels'),
    )


@torch.no_grad()
def collect_anchors(cfg, reference_teacher, source_teacher, device):
    source_chunks = []
    reference_chunks = []
    selection = {}
    per_task = int(cfg.alignment.anchors_per_task)
    for task_id, task in enumerate(cfg.tasks):
        dataset = swm.data.load_dataset(
            task.dataset,
            transform=None,
            num_steps=int(cfg.alignment.num_steps),
            frameskip=int(task.frameskip),
            keys_to_load=['pixels'],
        )
        dataset.transform = image_preprocessor(int(cfg.data.img_size))
        count = min(per_task, len(dataset))
        generator = torch.Generator().manual_seed(
            int(cfg.seed) + 1009 * task_id
        )
        indices = torch.randperm(len(dataset), generator=generator)[:count]
        selection[str(task.name)] = {
            'dataset_length': len(dataset),
            'count': count,
            'indices': indices.tolist(),
        }
        loader = DataLoader(
            Subset(dataset, indices.tolist()),
            batch_size=int(cfg.alignment.batch_size),
            shuffle=False,
            num_workers=int(cfg.alignment.num_workers),
            pin_memory=True,
        )
        for batch in loader:
            pixels = batch['pixels'].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == 'cuda',
            ):
                source = source_teacher.encode({'pixels': pixels})['emb']
                reference = reference_teacher.encode({'pixels': pixels})['emb']
            source_chunks.append(source.reshape(-1, source.size(-1)).cpu())
            reference_chunks.append(
                reference.reshape(-1, reference.size(-1)).cpu()
            )
    return (
        torch.cat(source_chunks).float(),
        torch.cat(reference_chunks).float(),
        selection,
    )


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rms_norm(vectors: torch.Tensor) -> torch.Tensor:
    """Root-mean-square of per-row L2 norms: sqrt(mean_i ||row_i||^2)."""
    flat = vectors.reshape(-1, vectors.size(-1)).double()
    return flat.square().sum(dim=-1).mean().sqrt()


def fit_transform(
    mode: str,
    source_train: torch.Tensor,
    reference_train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit (rotation, scale, bias) for the requested alignment mode."""
    dim = source_train.size(-1)
    if mode == 'similarity':
        return fit_similarity_procrustes(source_train, reference_train)
    if mode == 'identity':
        return torch.eye(dim), torch.tensor(1.0), torch.zeros(dim)
    if mode == 'center_norm_match':
        source = source_train.reshape(-1, dim).double()
        reference = reference_train.reshape(-1, dim).double()
        source_mean = source.mean(dim=0)
        reference_mean = reference.mean(dim=0)
        source_centered = source - source_mean
        reference_centered = reference - reference_mean
        source_rms = rms_norm(source_centered)
        reference_rms = rms_norm(reference_centered)
        if not torch.isfinite(source_rms) or source_rms <= 0:
            raise ValueError('source anchors have zero centered variance')
        scale = reference_rms / source_rms
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError('center_norm_match produced a non-positive scale')
        bias = reference_mean - scale * source_mean
        return torch.eye(dim), scale.float(), bias.float()
    raise ValueError(
        f'unknown alignment.mode {mode!r}; expected one of {ALIGNMENT_MODES}'
    )


def fit_one(cfg, reference_task, source_task, reference_teacher, device, mode):
    source_teacher = load_pretrained(source_task.teacher_checkpoint).to(device)
    source_teacher.requires_grad_(False)
    source_teacher.eval()
    source, reference, selection = collect_anchors(
        cfg, reference_teacher, source_teacher, device
    )
    generator = torch.Generator().manual_seed(int(cfg.alignment.split_seed))
    permutation = torch.randperm(len(source), generator=generator)
    train_count = int(len(source) * float(cfg.alignment.train_fraction))
    train = permutation[:train_count]
    validation = permutation[train_count:]
    source_train = source[train].float()
    reference_train = reference[train].float()
    rotation, scale, bias = fit_transform(mode, source_train, reference_train)
    alignment = SimilarityAlignment(rotation, scale, bias)
    report = {
        'train': alignment_metrics(source[train], reference[train], alignment),
        'validation': alignment_metrics(
            source[validation], reference[validation], alignment
        ),
        'identity_validation': alignment_metrics(
            source[validation], reference[validation]
        ),
        'roundtrip_max_abs_error': float(
            (
                alignment.inverse(alignment(source[validation]))
                - source[validation]
            )
            .abs()
            .max()
        ),
    }
    source_centered = source_train - source_train.mean(dim=0, keepdim=True)
    reference_centered = reference_train - reference_train.mean(
        dim=0, keepdim=True
    )
    report.update(
        {
            'source_mean_norm': float(source_train.norm(dim=-1).mean()),
            'reference_mean_norm': float(reference_train.norm(dim=-1).mean()),
            'source_centered_rms_norm': float(rms_norm(source_centered)),
            'reference_centered_rms_norm': float(rms_norm(reference_centered)),
            'mapped_mean_norm': float(
                alignment(source_train).norm(dim=-1).mean()
            ),
        }
    )
    if source_task.get('codebook_checkpoint'):
        codebook = load_codebook_weights(source_task.codebook_checkpoint)
        original = nearest_code_indices(
            source[validation], codebook
        ).squeeze(-1)
        transformed = nearest_code_indices(
            alignment(source[validation]), alignment(codebook)
        ).squeeze(-1)
        report['source_token_preservation'] = float(
            (original == transformed).float().mean()
        )
    entry = {
        'source_task': str(source_task.name),
        'reference_task': str(reference_task.name),
        'mode': str(mode),
        'rotation': rotation.cpu(),
        'scale': scale.cpu(),
        'bias': bias.cpu(),
        'source_teacher_sha256': sha256_file(
            resolve_weights_path(source_task.teacher_checkpoint)
        ),
        'reference_teacher_sha256': sha256_file(
            resolve_weights_path(reference_task.teacher_checkpoint)
        ),
        'calibration_metadata': {
            'train_count': len(train),
            'validation_count': len(validation),
            'split_seed': int(cfg.alignment.split_seed),
            'task_selection': selection,
        },
        'metrics': report,
    }
    del source_teacher, source, reference
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return entry, report


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args.set)
    save_config(cfg, args.save_config)
    mode = alignment_mode(cfg)
    torch.manual_seed(int(cfg.seed))
    device = torch.device(str(cfg.alignment.device))
    os.environ['LOCAL_DATASET_DIR'] = str(cfg.paths.dataset_cache)
    os.environ['STABLEWM_HOME'] = str(cfg.paths.dataset_cache)

    reference_id, source_ids = resolve_task_ids(cfg)
    reference_task = cfg.tasks[reference_id]
    reference_teacher = load_pretrained(
        reference_task.teacher_checkpoint
    ).to(device)
    reference_teacher.requires_grad_(False)
    reference_teacher.eval()

    entries = {}
    reports = {}
    for source_id in source_ids:
        source_task = cfg.tasks[source_id]
        entry, report = fit_one(
            cfg, reference_task, source_task, reference_teacher, device, mode
        )
        entries[str(source_task.name)] = entry
        reports[str(source_task.name)] = report

    payload = {
        'format_version': 2,
        'mode': str(mode),
        'reference_task': str(reference_task.name),
        'reference_task_id': reference_id,
        'reference_teacher_sha256': sha256_file(
            resolve_weights_path(reference_task.teacher_checkpoint)
        ),
        'alignments': entries,
        'metrics': reports,
    }
    output = Path(cfg.paths.alignment_checkpoint).expanduser().resolve()
    atomic_save(payload, output)
    output.with_suffix('.json').write_text(
        json.dumps(reports, indent=2, ensure_ascii=False) + '\n'
    )
    print(json.dumps(reports, indent=2), flush=True)
    print(f'Saved {len(entries)} alignments to {output}', flush=True)


if __name__ == '__main__':
    main()
