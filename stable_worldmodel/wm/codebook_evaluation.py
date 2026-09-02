"""Evaluation utilities for frozen-latent teacher/student codebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


ERROR_DEFINITIONS = {
    'absolute_l2': '||z_e - quantized||_2',
    'relative_l2': (
        '||z_e - quantized||_2 / max(||z_e||_2, epsilon)'
    ),
}


@torch.inference_mode()
def collect_quantization_errors(
    model,
    latents: torch.Tensor,
    batch_size: int = 8192,
    relative_epsilon: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Collect per-vector absolute and relative teacher-codebook errors."""
    if len(latents) == 0:
        raise ValueError('cannot evaluate an empty latent split')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if relative_epsilon <= 0:
        raise ValueError('relative_epsilon must be positive')

    model.eval()
    device = next(model.parameters()).device
    absolute_chunks = []
    relative_chunks = []
    for start in range(0, len(latents), batch_size):
        continuous = latents[start : start + batch_size].to(
            device=device, dtype=torch.float32
        )
        quantized = model(continuous)['quantized'].float()
        absolute = (continuous - quantized).norm(p=2, dim=-1)
        denominator = continuous.norm(p=2, dim=-1).clamp_min(
            relative_epsilon
        )
        absolute_chunks.append(absolute.cpu())
        relative_chunks.append((absolute / denominator).cpu())

    return {
        'absolute_l2': torch.cat(absolute_chunks),
        'relative_l2': torch.cat(relative_chunks),
    }


def distribution_statistics(values: torch.Tensor) -> dict[str, float]:
    """Return stable descriptive statistics for one error distribution."""
    values = values.detach().float().cpu()
    finite = values[torch.isfinite(values)]
    if len(finite) != len(values):
        raise ValueError('error distribution contains non-finite values')
    quantiles = torch.quantile(
        finite, torch.tensor([0.5, 0.9, 0.95, 0.99])
    )
    return {
        'mean': float(finite.mean()),
        'std': float(finite.std(unbiased=False)),
        'min': float(finite.min()),
        'median': float(quantiles[0]),
        'p90': float(quantiles[1]),
        'p95': float(quantiles[2]),
        'p99': float(quantiles[3]),
        'max': float(finite.max()),
    }


def summarize_quantization_errors(
    errors_by_split: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict]:
    """Summarize every metric for train, validation, and test splits."""
    return {
        split: {
            'num_vectors': len(errors['absolute_l2']),
            **{
                metric: distribution_statistics(values)
                for metric, values in errors.items()
            },
        }
        for split, errors in errors_by_split.items()
    }


def save_quantization_violin_plot(
    errors_by_split: dict[str, dict[str, torch.Tensor]],
    output_path: str | Path,
    max_points_per_split: int = 50000,
    seed: int = 0,
    dpi: int = 180,
) -> None:
    """Save absolute/relative quantization distributions as violin plots."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    if max_points_per_split < 2:
        raise ValueError('max_points_per_split must be at least two')

    split_order = ['train', 'validation', 'test']
    if dpi < 1:
        raise ValueError('dpi must be positive')
    missing = [name for name in split_order if name not in errors_by_split]
    if missing:
        raise KeyError(f'missing evaluation splits: {missing}')

    colors = ['#4C78A8', '#F58518', '#54A24B']
    sampled = {}
    for split_index, split in enumerate(split_order):
        sampled[split] = {}
        generator = np.random.default_rng(seed + split_index)
        for metric, tensor in errors_by_split[split].items():
            values = tensor.detach().float().cpu().numpy()
            if len(values) > max_points_per_split:
                indices = generator.choice(
                    len(values), max_points_per_split, replace=False
                )
                values = values[indices]
            sampled[split][metric] = values

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = [
        (
            'absolute_l2',
            'Absolute quantization error',
            'L2 distance',
        ),
        (
            'relative_l2',
            'Relative quantization error',
            'Relative L2 error',
        ),
    ]
    positions = np.arange(1, len(split_order) + 1)
    labels = ['Train', 'Validation', 'Test']

    for axis, (metric, title, ylabel) in zip(axes, panels):
        datasets = [sampled[split][metric] for split in split_order]
        parts = axis.violinplot(
            datasets,
            positions=positions,
            widths=0.82,
            showmeans=True,
            showmedians=True,
            showextrema=True,
            points=200,
        )
        for body, color in zip(parts['bodies'], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.72)
        for key in ('cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars'):
            parts[key].set_color('#202124')
            parts[key].set_linewidth(0.9)

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels)
        axis.grid(axis='y', alpha=0.25)
        axis.set_axisbelow(True)
        if metric == 'relative_l2':
            axis.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    figure.suptitle(
        'Frozen encoder latent vs. EMA-teacher quantized vector',
        fontsize=13,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def save_training_loss_curve(
    metrics_path: str | Path,
    output_path: str | Path,
    dpi: int = 180,
) -> None:
    """Plot student and EMA-teacher codebook losses over training."""
    import csv

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if dpi < 1:
        raise ValueError('dpi must be positive')
    metrics_path = Path(metrics_path)
    with open(metrics_path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'no training metrics found in {metrics_path}')

    series = [
        ('train_student_l2', 'Train student', '#4C78A8', '-'),
        ('val_student_l2', 'Validation student', '#F58518', '--'),
        ('val_teacher_l2', 'Validation EMA teacher', '#54A24B', '-'),
    ]
    required = {'epoch', *(name for name, *_ in series)}
    missing = required.difference(rows[0])
    if missing:
        raise KeyError(f'missing training metric columns: {sorted(missing)}')

    epochs = np.asarray([int(row['epoch']) for row in rows])
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for name, label, color, linestyle in series:
        values = np.asarray([float(row[name]) for row in rows])
        axis.plot(
            epochs,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )

    axis.set_title('Codebook training loss')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Squared L2 loss per latent vector')
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


__all__ = [
    'ERROR_DEFINITIONS',
    'collect_quantization_errors',
    'distribution_statistics',
    'save_quantization_violin_plot',
    'save_training_loss_curve',
    'summarize_quantization_errors',
]
