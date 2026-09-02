"""Discretize a frozen continuous LeWM latent space with a codebook."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from stable_pretraining import data as dt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from stable_worldmodel.wm.codebook_evaluation import (
    ERROR_DEFINITIONS,
    collect_quantization_errors,
    save_quantization_violin_plot,
    save_training_loss_curve,
    summarize_quantization_errors,
)
from stable_worldmodel.wm.latent_codebook import TeacherStudentCodebook
from stable_worldmodel.wm.utils import load_pretrained, save_pretrained


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


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


def encoder_fingerprint(checkpoint: str) -> dict:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    weights_path = (
        checkpoint_path / 'weights.pt'
        if checkpoint_path.is_dir()
        else checkpoint_path
    )
    stat = weights_path.stat()
    return {
        'checkpoint': str(checkpoint_path),
        'weights_size': stat.st_size,
        'weights_mtime_ns': stat.st_mtime_ns,
    }


def latent_cache_metadata(cfg, dataset_length: int) -> dict:
    return {
        **encoder_fingerprint(cfg.encoder_checkpoint),
        'dataset': cfg.data.dataset,
        'dataset_length': dataset_length,
        'max_latents': min(cfg.data.max_latents, dataset_length),
        'img_size': cfg.img_size,
        'seed': cfg.seed,
        'latent_key': 'emb',
    }


def load_frozen_encoder(checkpoint: str, device: torch.device):
    model = load_pretrained(checkpoint).to(device)
    model.requires_grad_(False)
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable:
        raise RuntimeError(f'frozen encoder still has {trainable} parameters')
    return model


def extract_latents(cfg, cache_path: Path, device: torch.device):
    dataset = swm.data.load_dataset(
        cfg.data.dataset,
        transform=None,
        num_steps=cfg.data.num_steps,
        frameskip=cfg.data.frameskip,
        keys_to_load=['pixels'],
    )
    dataset.transform = image_preprocessor(cfg.img_size)
    metadata = latent_cache_metadata(cfg, len(dataset))

    if cache_path.exists():
        cached = torch.load(
            cache_path, map_location='cpu', weights_only=False
        )
        if cached.get('metadata') != metadata:
            raise RuntimeError(
                f'latent cache metadata mismatch at {cache_path}; '
                'choose a new output_model_name or remove the stale cache'
            )
        print(
            f'Loaded {len(cached["latents"]):,} cached frozen latents '
            f'from {cache_path}',
            flush=True,
        )
        return cached['latents'], metadata

    sample_count = metadata['max_latents']
    generator = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(len(dataset), generator=generator)[
        :sample_count
    ].tolist()
    loader_kwargs = {
        'batch_size': cfg.data.extraction_batch_size,
        'shuffle': False,
        'num_workers': cfg.data.num_workers,
        'drop_last': False,
        'persistent_workers': cfg.data.num_workers > 0,
        'pin_memory': cfg.data.pin_memory,
    }
    if cfg.data.num_workers > 0:
        loader_kwargs['prefetch_factor'] = cfg.data.prefetch_factor
    loader = DataLoader(Subset(dataset, indices), **loader_kwargs)

    print(
        f'Extracting {sample_count:,} frozen latents with '
        f'{cfg.encoder_checkpoint}',
        flush=True,
    )
    encoder = load_frozen_encoder(cfg.encoder_checkpoint, device)
    chunks = []
    use_amp = device.type == 'cuda'
    with torch.inference_mode():
        for batch in tqdm(loader, desc='frozen encoder', dynamic_ncols=True):
            pixels = batch['pixels'].to(
                device, non_blocking=cfg.data.pin_memory
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                encoded = encoder.encode({'pixels': pixels})['emb']
            chunks.append(
                encoded.reshape(-1, encoded.size(-1)).half().cpu()
            )

    latents = torch.cat(chunks, dim=0)[:sample_count].contiguous()
    if latents.size(1) != cfg.codebook.embedding_dim:
        raise RuntimeError(
            f'encoder produced latent dim {latents.size(1)}, but codebook '
            f'expects {cfg.codebook.embedding_dim}'
        )
    atomic_torch_save(
        {'metadata': metadata, 'latents': latents}, cache_path
    )
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f'Saved latent cache to {cache_path}', flush=True)
    return latents, metadata


def test_latent_cache_metadata(cfg, dataset_length: int) -> dict:
    selection_offset = min(cfg.data.max_latents, dataset_length)
    available = max(0, dataset_length - selection_offset)
    sample_count = min(cfg.data.test_latents, available)
    if sample_count < 1:
        raise ValueError(
            'no disjoint test latents remain after the training cache'
        )
    return {
        **encoder_fingerprint(cfg.encoder_checkpoint),
        'dataset': cfg.data.dataset,
        'dataset_length': dataset_length,
        'test_latents': sample_count,
        'selection_offset': selection_offset,
        'img_size': cfg.img_size,
        'seed': cfg.seed,
        'latent_key': 'emb',
        'split': 'test',
    }


def extract_test_latents(cfg, cache_path: Path, device: torch.device):
    """Extract a test cache disjoint from all training/validation latents."""
    dataset = swm.data.load_dataset(
        cfg.data.dataset,
        transform=None,
        num_steps=cfg.data.num_steps,
        frameskip=cfg.data.frameskip,
        keys_to_load=['pixels'],
    )
    dataset.transform = image_preprocessor(cfg.img_size)
    metadata = test_latent_cache_metadata(cfg, len(dataset))

    if cache_path.exists():
        cached = torch.load(
            cache_path, map_location='cpu', weights_only=False
        )
        if cached.get('metadata') != metadata:
            raise RuntimeError(
                f'test latent cache metadata mismatch at {cache_path}; '
                'choose a new output_model_name or remove the stale cache'
            )
        print(
            f'Loaded {len(cached["latents"]):,} cached test latents '
            f'from {cache_path}',
            flush=True,
        )
        return cached['latents'], metadata

    generator = torch.Generator().manual_seed(cfg.seed)
    selection_offset = metadata['selection_offset']
    sample_count = metadata['test_latents']
    indices = torch.randperm(len(dataset), generator=generator)[
        selection_offset : selection_offset + sample_count
    ].tolist()
    loader_kwargs = {
        'batch_size': cfg.data.extraction_batch_size,
        'shuffle': False,
        'num_workers': cfg.data.num_workers,
        'drop_last': False,
        'persistent_workers': cfg.data.num_workers > 0,
        'pin_memory': cfg.data.pin_memory,
    }
    if cfg.data.num_workers > 0:
        loader_kwargs['prefetch_factor'] = cfg.data.prefetch_factor
    loader = DataLoader(Subset(dataset, indices), **loader_kwargs)

    print(
        f'Extracting {sample_count:,} disjoint frozen test latents',
        flush=True,
    )
    encoder = load_frozen_encoder(cfg.encoder_checkpoint, device)
    chunks = []
    use_amp = device.type == 'cuda'
    with torch.inference_mode():
        for batch in tqdm(
            loader, desc='test frozen encoder', dynamic_ncols=True
        ):
            pixels = batch['pixels'].to(
                device, non_blocking=cfg.data.pin_memory
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                encoded = encoder.encode({'pixels': pixels})['emb']
            chunks.append(
                encoded.reshape(-1, encoded.size(-1)).half().cpu()
            )

    latents = torch.cat(chunks, dim=0)[:sample_count].contiguous()
    if latents.size(1) != cfg.codebook.embedding_dim:
        raise RuntimeError(
            f'encoder produced test latent dim {latents.size(1)}, '
            f'but codebook expects {cfg.codebook.embedding_dim}'
        )
    atomic_torch_save(
        {'metadata': metadata, 'latents': latents}, cache_path
    )
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f'Saved test latent cache to {cache_path}', flush=True)
    return latents, metadata


@torch.no_grad()
def kmeans_plus_plus(
    latents: torch.Tensor,
    num_embeddings: int,
    sample_count: int,
    seed: int,
) -> torch.Tensor:
    """Choose data-dependent centers without updating the encoder."""
    if len(latents) < num_embeddings:
        raise ValueError('fewer initialization latents than codebook entries')
    generator = torch.Generator(device=latents.device).manual_seed(seed)
    count = min(sample_count, len(latents))
    sample_indices = torch.randperm(
        len(latents), generator=generator, device=latents.device
    )[:count]
    samples = latents[sample_indices].float()
    centers = torch.empty(
        num_embeddings,
        samples.size(1),
        device=samples.device,
        dtype=samples.dtype,
    )
    first = torch.randint(
        count, (), generator=generator, device=samples.device
    )
    centers[0] = samples[first]
    closest = (samples - centers[0]).square().sum(dim=1)

    for index in tqdm(
        range(1, num_embeddings),
        desc='k-means++ init',
        dynamic_ncols=True,
    ):
        closest.clamp_min_(0)
        total = closest.sum()
        if not torch.isfinite(total) or total <= 0:
            selected = torch.randint(
                count, (), generator=generator, device=samples.device
            )
        else:
            selected = torch.multinomial(
                closest / total, 1, generator=generator
            ).squeeze(0)
        centers[index] = samples[selected]
        distance = (samples - centers[index]).square().sum(dim=1)
        closest = torch.minimum(closest, distance)
    return centers


def usage_metrics(counts: torch.Tensor) -> dict[str, float]:
    counts = counts.float().cpu()
    probabilities = counts / counts.sum().clamp_min(1.0)
    entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
    return {
        'perplexity': float(entropy.exp()),
        'active_codes': int((counts > 0).sum()),
        'active_fraction': float((counts > 0).float().mean()),
        'max_code_fraction': float(probabilities.max()),
    }


@torch.no_grad()
def evaluate(
    model: TeacherStudentCodebook,
    latents: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    total_student = 0.0
    total_teacher = 0.0
    counts = torch.zeros(model.num_embeddings, device=latents.device)
    for start in range(0, len(latents), batch_size):
        batch = latents[start : start + batch_size]
        output = model(batch)
        size = len(batch)
        total_student += float(output['codebook_loss']) * size
        total_teacher += float(output['teacher_l2']) * size
        counts += output['counts']
    metrics = {
        'student_l2': total_student / len(latents),
        'teacher_l2': total_teacher / len(latents),
    }
    metrics.update(usage_metrics(counts))
    return metrics


def append_metrics(path: Path, row: dict) -> None:
    exists = path.exists()
    with open(path, 'a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def codebook_config(cfg) -> dict:
    return {
        '_target_': (
            'stable_worldmodel.wm.latent_codebook.'
            'TeacherStudentCodebook'
        ),
        'num_embeddings': cfg.codebook.num_embeddings,
        'embedding_dim': cfg.codebook.embedding_dim,
        'teacher_momentum': cfg.codebook.teacher_momentum,
        'source_checkpoint': str(
            Path(cfg.encoder_checkpoint).expanduser().resolve()
        ),
        'latent_key': 'emb',
    }


def save_loadable_codebook(model, cfg) -> None:
    save_pretrained(
        model,
        run_name=cfg.output_model_name,
        config=codebook_config(cfg),
        filename='weights.pt',
    )


def save_quantization_summary_csv(path: Path, summary: dict) -> None:
    fieldnames = [
        'split',
        'metric',
        'num_vectors',
        'mean',
        'std',
        'min',
        'median',
        'p90',
        'p95',
        'p99',
        'max',
    ]
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for split, split_summary in summary.items():
            for metric in ERROR_DEFINITIONS:
                writer.writerow(
                    {
                        'split': split,
                        'metric': metric,
                        'num_vectors': split_summary['num_vectors'],
                        **split_summary[metric],
                    }
                )


def print_quantization_evaluation(summary: dict) -> None:
    print(
        '\nPost-training quantization evaluation '
        '(EMA-teacher quantized vectors)',
        flush=True,
    )
    print(
        f'  absolute_l2 = {ERROR_DEFINITIONS["absolute_l2"]}',
        flush=True,
    )
    print(
        f'  relative_l2 = {ERROR_DEFINITIONS["relative_l2"]}',
        flush=True,
    )
    for split in ('train', 'validation', 'test'):
        split_summary = summary[split]
        print(
            f'[{split}] n={split_summary["num_vectors"]:,}',
            flush=True,
        )
        for metric in ERROR_DEFINITIONS:
            stats = split_summary[metric]
            scale = 100.0 if metric == 'relative_l2' else 1.0
            suffix = '%' if metric == 'relative_l2' else ''
            print(
                f'  {metric}: '
                f'mean={stats["mean"] * scale:.6f}{suffix}, '
                f'std={stats["std"] * scale:.6f}{suffix}, '
                f'median={stats["median"] * scale:.6f}{suffix}, '
                f'p95={stats["p95"] * scale:.6f}{suffix}, '
                f'p99={stats["p99"] * scale:.6f}{suffix}, '
                f'max={stats["max"] * scale:.6f}{suffix}',
                flush=True,
            )


def run_post_training_evaluation(
    cfg,
    model: TeacherStudentCodebook,
    splits: dict[str, torch.Tensor],
    run_dir: Path,
) -> dict:
    errors_by_split = {}
    for split in ('train', 'validation', 'test'):
        print(f'Evaluating quantization error on {split} split...', flush=True)
        errors_by_split[split] = collect_quantization_errors(
            model,
            splits[split],
            batch_size=cfg.evaluation.batch_size,
            relative_epsilon=cfg.evaluation.relative_epsilon,
        )

    summary = summarize_quantization_errors(errors_by_split)
    checkpoint_path = (
        swm.data.utils.get_cache_dir(sub_folder='checkpoints')
        / cfg.output_model_name
    )
    payload = {
        'checkpoint': str(checkpoint_path),
        'quantized_codebook': 'ema_teacher',
        'relative_epsilon': cfg.evaluation.relative_epsilon,
        'definitions': ERROR_DEFINITIONS,
        'splits': summary,
    }
    with open(run_dir / 'quantization_evaluation.json', 'w') as stream:
        json.dump(payload, stream, indent=2)
    save_quantization_summary_csv(
        run_dir / 'quantization_evaluation.csv', summary
    )
    np.savez_compressed(
        run_dir / 'quantization_errors.npz',
        **{
            f'{split}_{metric}': values.numpy()
            for split, errors in errors_by_split.items()
            for metric, values in errors.items()
        },
    )
    save_quantization_violin_plot(
        errors_by_split,
        run_dir / 'quantization_error_violin.png',
        max_points_per_split=(
            cfg.evaluation.violin_max_points_per_split
        ),
        seed=cfg.seed,
        dpi=cfg.evaluation.violin_dpi,
    )
    print_quantization_evaluation(summary)
    print(
        f'Saved evaluation and violin plot to {run_dir}',
        flush=True,
    )
    return payload


def train_codebook(cfg, latents: torch.Tensor, run_dir: Path, device):
    permutation = torch.randperm(
        len(latents), generator=torch.Generator().manual_seed(cfg.seed + 1)
    )
    train_size = int(len(latents) * cfg.data.train_split)
    train_latents = latents[permutation[:train_size]].float().to(device)
    val_latents = latents[permutation[train_size:]].float().to(device)
    del latents

    model = TeacherStudentCodebook(
        num_embeddings=cfg.codebook.num_embeddings,
        embedding_dim=cfg.codebook.embedding_dim,
        teacher_momentum=cfg.codebook.teacher_momentum,
        source_checkpoint=str(cfg.encoder_checkpoint),
        latent_key='emb',
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.student.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.optimizer.epochs,
        eta_min=cfg.optimizer.min_lr,
    )

    state_path = run_dir / 'training_state.pt'
    start_epoch = 0
    best_teacher_l2 = math.inf
    best_epoch = 0
    if cfg.checkpoint.resume and state_path.exists():
        state = torch.load(
            state_path, map_location=device, weights_only=False
        )
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch']
        best_teacher_l2 = state['best_teacher_l2']
        best_epoch = state['best_epoch']
        print(f'Resuming codebook at epoch {start_epoch}', flush=True)
    else:
        centers = kmeans_plus_plus(
            train_latents,
            cfg.codebook.num_embeddings,
            cfg.codebook.kmeans_init_samples,
            cfg.seed + 2,
        )
        model.initialize(centers)

    metrics_path = run_dir / 'metrics.csv'
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 3)
    if start_epoch:
        generator.manual_seed(cfg.seed + 3 + start_epoch)
    for epoch in range(start_epoch, cfg.optimizer.epochs):
        model.train()
        order = torch.randperm(
            len(train_latents), generator=generator, device=device
        )
        total_loss = 0.0
        train_counts = torch.zeros(
            model.num_embeddings, device=device, dtype=torch.long
        )
        progress = tqdm(
            range(0, len(train_latents), cfg.optimizer.batch_size),
            desc=f'epoch {epoch + 1}/{cfg.optimizer.epochs}',
            dynamic_ncols=True,
        )
        for start in progress:
            batch_indices = order[start : start + cfg.optimizer.batch_size]
            batch = train_latents[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = output['codebook_loss']
            loss.backward()
            optimizer.step()
            model.update_teacher()

            size = len(batch)
            total_loss += float(loss.detach()) * size
            train_counts += output['counts']
            progress.set_postfix(loss=f'{float(loss.detach()):.4f}')

        scheduler.step()
        train_metrics = {
            'student_l2': total_loss / len(train_latents),
            **usage_metrics(train_counts),
        }
        val_metrics = evaluate(
            model, val_latents, cfg.optimizer.batch_size
        )
        row = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'train_student_l2': train_metrics['student_l2'],
            'train_perplexity': train_metrics['perplexity'],
            'train_active_codes': train_metrics['active_codes'],
            'train_active_fraction': train_metrics['active_fraction'],
            'train_max_code_fraction': train_metrics['max_code_fraction'],
            'val_student_l2': val_metrics['student_l2'],
            'val_teacher_l2': val_metrics['teacher_l2'],
            'val_perplexity': val_metrics['perplexity'],
            'val_active_codes': val_metrics['active_codes'],
            'val_active_fraction': val_metrics['active_fraction'],
            'val_max_code_fraction': val_metrics['max_code_fraction'],
        }
        append_metrics(metrics_path, row)
        print(json.dumps(row, sort_keys=True), flush=True)

        if val_metrics['teacher_l2'] < best_teacher_l2:
            best_teacher_l2 = val_metrics['teacher_l2']
            best_epoch = epoch + 1
            save_loadable_codebook(model, cfg)

        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_teacher_l2': best_teacher_l2,
            'best_epoch': best_epoch,
        }
        atomic_torch_save(state, state_path)
        if (epoch + 1) % cfg.checkpoint.interval == 0:
            atomic_torch_save(
                state, run_dir / f'training_state_epoch_{epoch + 1}.pt'
            )

    loss_curve_path = run_dir / 'training_loss_curve.png'
    save_training_loss_curve(
        metrics_path,
        loss_curve_path,
        dpi=cfg.evaluation.violin_dpi,
    )
    print(f'Saved training loss curve to {loss_curve_path}', flush=True)

    summary = {
        'best_epoch': best_epoch,
        'best_val_teacher_l2': best_teacher_l2,
        'completed_epochs': cfg.optimizer.epochs,
        'loadable_checkpoint': str(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints')
            / cfg.output_model_name
        ),
    }
    with open(run_dir / 'summary.json', 'w') as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return train_latents, val_latents


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='latent_codebook',
)
def run(cfg) -> None:
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device(
        cfg.device if torch.cuda.is_available() else 'cpu'
    )
    run_dir = (
        swm.data.utils.get_cache_dir(sub_folder='codebook_runs')
        / cfg.output_model_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as stream:
        OmegaConf.save(cfg, stream)

    latents, metadata = extract_latents(
        cfg, run_dir / 'latents.pt', device
    )
    with open(run_dir / 'latent_metadata.json', 'w') as stream:
        json.dump(metadata, stream, indent=2)
    train_latents, val_latents = train_codebook(
        cfg, latents, run_dir, device
    )

    test_latents, test_metadata = extract_test_latents(
        cfg, run_dir / 'test_latents.pt', device
    )
    with open(run_dir / 'test_latent_metadata.json', 'w') as stream:
        json.dump(test_metadata, stream, indent=2)

    checkpoint_path = (
        swm.data.utils.get_cache_dir(sub_folder='checkpoints')
        / cfg.output_model_name
    )
    evaluation_model = load_pretrained(str(checkpoint_path)).to(device)
    evaluation_model.eval()
    run_post_training_evaluation(
        cfg,
        evaluation_model,
        {
            'train': train_latents,
            'validation': val_latents,
            'test': test_latents,
        },
        run_dir,
    )


if __name__ == '__main__':
    run()
