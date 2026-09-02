"""Fit multiple task-specific similarity transforms into one reference space."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/multitask_vq_lewm_three_tasks.yaml',
    )
    return parser.parse_args()


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


def fit_one(cfg, reference_task, source_task, reference_teacher, device):
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
    rotation, scale, bias = fit_similarity_procrustes(
        source[train], reference[train]
    )
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
    cfg = OmegaConf.load(parse_args().config)
    torch.manual_seed(int(cfg.seed))
    device = torch.device(str(cfg.alignment.device))
    os.environ['LOCAL_DATASET_DIR'] = str(cfg.paths.dataset_cache)
    os.environ['STABLEWM_HOME'] = str(cfg.paths.dataset_cache)

    reference_id = int(cfg.alignment.reference_task_id)
    source_ids = [int(value) for value in cfg.alignment.source_task_ids]
    if not source_ids or reference_id in source_ids:
        raise ValueError('source_task_ids must be non-empty and exclude reference')
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('source_task_ids contains duplicates')
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
            cfg, reference_task, source_task, reference_teacher, device
        )
        entries[str(source_task.name)] = entry
        reports[str(source_task.name)] = report

    payload = {
        'format_version': 2,
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
