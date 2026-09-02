"""Plot the K512--K8192 latent-codebook quality comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / ".stablewm" / "codebook_runs"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "codebook_size_comparison"

RUNS = {
    512: RUN_ROOT / "official_lewm_pusht_compat_codebook",
    1024: RUN_ROOT / "official_lewm_pusht_compat_codebook_k1024",
    2048: RUN_ROOT / "official_lewm_pusht_compat_codebook_k2048",
    4096: RUN_ROOT / "official_lewm_pusht_compat_codebook_k4096",
    8192: RUN_ROOT / "official_lewm_pusht_compat_codebook_k8192",
}

COLORS = {
    512: "#4C78A8",
    1024: "#F58518",
    2048: "#54A24B",
    4096: "#E45756",
    8192: "#7A5195",
}


def read_run(path: Path) -> dict:
    with (path / "quantization_evaluation.json").open() as stream:
        evaluation = json.load(stream)
    with (path / "metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    return {"evaluation": evaluation, "metrics": metrics}


def style_axis(axis) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_quantization_quality(data: dict[int, dict]) -> None:
    sizes = np.asarray(list(data))
    positions = np.arange(len(sizes))
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    for metric, axis, scale, ylabel in (
        ("absolute_l2", axes[0], 1.0, "Absolute L2 error"),
        ("relative_l2", axes[1], 100.0, "Relative L2 error (%)"),
    ):
        for statistic, marker, linestyle in (
            ("mean", "o", "-"),
            ("median", "s", "-"),
            ("p90", "^", "--"),
            ("p99", "D", ":"),
        ):
            values = [
                data[int(size)]["evaluation"]["splits"]["test"][metric][statistic]
                * scale
                for size in sizes
            ]
            axis.plot(
                positions,
                values,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=6,
                label=statistic,
            )
        axis.set_xticks(positions, [f"K{size}" for size in sizes])
        axis.set_xlabel("Codebook size")
        axis.set_ylabel(ylabel)
        style_axis(axis)
        axis.legend(frameon=False, ncol=2)

    axes[0].set_title("Held-out test quantization error")
    axes[1].set_title("Held-out test relative distortion")
    figure.suptitle("PushT latent codebook quality versus codebook size", fontsize=14)
    figure.savefig(OUTPUT_DIR / "codebook_size_quantization_quality.png", dpi=180)
    plt.close(figure)


def plot_training_and_utilization(data: dict[int, dict]) -> None:
    sizes = np.asarray(list(data))
    positions = np.arange(len(sizes))
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    for size in sizes:
        metrics = data[int(size)]["metrics"]
        epochs = [int(row["epoch"]) for row in metrics]
        losses = [float(row["val_teacher_l2"]) for row in metrics]
        axes[0].plot(
            epochs,
            losses,
            color=COLORS[int(size)],
            linewidth=2.0,
            label=f"K{size}",
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"Validation $E[\|z-q(z)\|_2^2]$")
    axes[0].set_yscale("log")
    axes[0].set_title("EMA-teacher validation objective")
    style_axis(axes[0])
    axes[0].legend(frameon=False, ncol=2)

    active_fractions = []
    normalized_perplexities = []
    for size in sizes:
        final = data[int(size)]["metrics"][-1]
        active_fractions.append(float(final["val_active_fraction"]) * 100.0)
        normalized_perplexities.append(float(final["val_perplexity"]) / int(size) * 100.0)
    width = 0.36
    active_bars = axes[1].bar(
        positions - width / 2,
        active_fractions,
        width,
        color="#4C78A8",
        label="Active codes / K",
    )
    perplexity_bars = axes[1].bar(
        positions + width / 2,
        normalized_perplexities,
        width,
        color="#F58518",
        label="Perplexity / K",
    )
    axes[1].set_xticks(positions, [f"K{size}" for size in sizes])
    axes[1].set_ylim(0, 110)
    axes[1].set_xlabel("Codebook size")
    axes[1].set_ylabel("Validation utilization (%)")
    axes[1].set_title("Final validation codebook utilization")
    style_axis(axes[1])
    axes[1].legend(frameon=False)
    for bars in (active_bars, perplexity_bars):
        axes[1].bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    figure.suptitle("Training convergence and codebook utilization", fontsize=14)
    figure.savefig(OUTPUT_DIR / "codebook_size_training_and_utilization.png", dpi=180)
    plt.close(figure)


def plot_capacity_tradeoff(data: dict[int, dict]) -> None:
    sizes = np.asarray(list(data))
    positions = np.arange(len(sizes))
    checkpoint_sizes = []
    test_means = []
    generalization_gaps = []
    for size in sizes:
        run = data[int(size)]
        summary_path = RUNS[int(size)] / "summary.json"
        with summary_path.open() as stream:
            checkpoint = Path(json.load(stream)["loadable_checkpoint"]) / "weights.pt"
        checkpoint_sizes.append(checkpoint.stat().st_size / (1024.0**2))
        splits = run["evaluation"]["splits"]
        train_mean = splits["train"]["absolute_l2"]["mean"]
        test_mean = splits["test"]["absolute_l2"]["mean"]
        test_means.append(test_mean)
        generalization_gaps.append((test_mean / train_mean - 1.0) * 100.0)

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)
    axes[0].plot(checkpoint_sizes, test_means, color="#4C78A8", marker="o", linewidth=2.2)
    for size, x_value, y_value in zip(sizes, checkpoint_sizes, test_means, strict=True):
        axes[0].annotate(
            f"K{size}",
            (x_value, y_value),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set_xlabel("Loadable checkpoint size (MiB)")
    axes[0].set_ylabel("Test mean absolute L2")
    axes[0].set_title("Storage-quality frontier")
    style_axis(axes[0])

    bars = axes[1].bar(
        positions,
        generalization_gaps,
        color=[COLORS[int(size)] for size in sizes],
    )
    axes[1].set_xticks(positions, [f"K{size}" for size in sizes])
    axes[1].set_xlabel("Codebook size")
    axes[1].set_ylabel("Test vs. train mean L2 gap (%)")
    axes[1].set_title("Train-to-test quantization gap")
    style_axis(axes[1])
    axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    figure.suptitle("Capacity, quality, and generalization trade-offs", fontsize=14)
    figure.savefig(OUTPUT_DIR / "codebook_size_capacity_tradeoff.png", dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {size: read_run(path) for size, path in RUNS.items()}
    plot_quantization_quality(data)
    plot_training_and_utilization(data)
    plot_capacity_tradeoff(data)
    print(f"Saved comparison plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
