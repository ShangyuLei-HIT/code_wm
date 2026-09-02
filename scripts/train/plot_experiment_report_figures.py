"""Generate figures used by the three detailed experiment reports."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STABLEWM_ROOT = PROJECT_ROOT / ".stablewm"
ASSET_ROOT = PROJECT_ROOT / "docs" / "assets"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#7A5195"
TEAL = "#2A9D8F"
GRAY = "#7F7F7F"
LIGHT_GRAY = "#D9D9D9"


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def extract_success_rate(path: Path) -> float:
    match = re.search(r"success_rate['\"]?:\s*([0-9.]+)", path.read_text())
    if match is None:
        raise ValueError(f"success_rate not found in {path}")
    return float(match.group(1))


def style_axis(axis, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def add_phase_spans(axis) -> None:
    spans = (
        (0.5, 4.5, "#E8F1FA", "Phase 1"),
        (4.5, 14.5, "#FDF0E3", "Phase 2"),
        (14.5, 16.5, "#E9F5ED", "Phase 3"),
    )
    for left, right, color, label in spans:
        axis.axvspan(left, right, color=color, alpha=0.65, zorder=0)
        axis.text(
            (left + right) / 2,
            1.01,
            label,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=GRAY,
        )


def annotate_bars(axis, bars, *, suffix: str = "", decimals: int = 1) -> None:
    fmt = f"%.{decimals}f{suffix}"
    axis.bar_label(bars, fmt=fmt, padding=3, fontsize=8)


def plot_k8192_joint_distillation() -> None:
    output = ASSET_ROOT / "k8192_joint_distillation"
    output.mkdir(parents=True, exist_ok=True)
    run = STABLEWM_ROOT / "joint_distillation" / "lewm_pusht_k8192_seed3072"

    official = extract_success_rate(
        STABLEWM_ROOT
        / "checkpoints"
        / "official_lewm_pusht_compat"
        / "pusht_results_official_seeded_50.txt"
    )
    encoder_only = extract_success_rate(
        STABLEWM_ROOT
        / "checkpoints"
        / "official_lewm_pusht_compat_codebook_k8192_encoder_only"
        / "pusht_results_official_k8192_encoder_only_seeded_50.txt"
    )
    recurrent = extract_success_rate(
        STABLEWM_ROOT
        / "checkpoints"
        / "official_lewm_pusht_compat_codebook_k8192_recurrent"
        / "pusht_results_official_k8192_recurrent_seeded_50.txt"
    )
    scratch = extract_success_rate(
        STABLEWM_ROOT
        / "checkpoints"
        / "lewm_scratch_baseline_seed3072"
        / "pusht_results_scratch_epoch10_seeded_50.txt"
    )
    task_summary = load_json(run / "task_evaluation" / "summary.json")
    stage_rates = {row["stage"]: row["success_rate"] for row in task_summary["stages"]}

    labels = [
        "Official continuous",
        "Scratch epoch 10",
        "Official + encoder-only quant.",
        "Official + recurrent quant.",
        "Joint distill. Phase 1",
        "Joint distill. Phase 2",
        "Joint distill. Final",
    ]
    values = [
        official,
        scratch,
        encoder_only,
        recurrent,
        stage_rates["phase1"],
        stage_rates["phase2"],
        stage_rates["final"],
    ]
    colors = [BLUE, BLUE, RED, RED, ORANGE, GREEN, ORANGE]
    figure, axis = plt.subplots(figsize=(10.8, 5.7), constrained_layout=True)
    positions = np.arange(len(labels))
    bars = axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("PushT success rate (%) — 50 fixed starts")
    axis.set_title("Continuous, forced-quantized, and jointly distilled models")
    style_axis(axis, grid_axis="x")
    axis.bar_label(bars, fmt="%.0f%%", padding=4, fontsize=9)
    figure.savefig(output / "pusht_success_rate_comparison.png", dpi=180)
    plt.close(figure)

    metrics = load_jsonl(run / "metrics.jsonl")
    epochs = [row["epoch"] for row in metrics]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    for axis in axes:
        add_phase_spans(axis)

    for key, label, color in (
        ("validate/latent_mse", "Latent MSE", BLUE),
        ("validate/pred_student_mse", "Student prediction MSE", ORANGE),
        ("validate/soft_kl", "Soft KL", PURPLE),
    ):
        axes[0].plot(
            epochs,
            [row[key] for row in metrics],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=color,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation metric (log scale)")
    axes[0].set_title("Internal error metrics")
    axes[0].set_xticks([1, 4, 8, 12, 14, 16])
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    for key, label, color in (
        ("validate/token_agreement", "Top-1 token agreement", BLUE),
        ("validate/top5_token_agreement", "Top-5 token agreement", GREEN),
        ("validate/perplexity_ratio", "Perplexity ratio", ORANGE),
        ("validate/effective_rank_ratio", "Effective-rank ratio", PURPLE),
    ):
        axes[1].plot(
            epochs,
            [row[key] * 100.0 for row in metrics],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=color,
            label=label,
        )
    axes[1].set_ylim(89, 100.4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Agreement / ratio (%)")
    axes[1].set_title("Codebook-distribution alignment")
    axes[1].set_xticks([1, 4, 8, 12, 14, 16])
    axes[1].legend(frameon=False, fontsize=8)
    style_axis(axes[1])
    figure.suptitle("K8192 joint-distillation training dynamics", fontsize=14)
    figure.savefig(output / "joint_distillation_internal_metrics.png", dpi=180)
    plt.close(figure)


def prediction_only_epoch_metrics() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = (
        STABLEWM_ROOT
        / "experiments"
        / "lewm_pusht_prediction_only_seed3072_40ep"
        / "spt"
        / "runs"
        / "20260827"
        / "125348"
        / "c0be241c8de2"
        / "metrics.csv"
    )
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    train: dict[int, list[float]] = {}
    validation: dict[int, float] = {}
    for row in rows:
        if row.get("epoch") and row.get("fit/pred_loss"):
            epoch = int(float(row["epoch"]))
            train.setdefault(epoch, []).append(float(row["fit/pred_loss"]))
        if row.get("epoch") and row.get("validate/pred_loss_epoch"):
            validation[int(float(row["epoch"]))] = float(row["validate/pred_loss_epoch"])
    epochs = np.asarray(sorted(train)) + 1
    train_means = np.asarray([fmean(train[int(epoch - 1)]) for epoch in epochs])
    validation_values = np.asarray([validation[int(epoch - 1)] for epoch in epochs])
    return epochs, train_means, validation_values


def aggregate_episode_successes(paths: list[Path]) -> float:
    outcomes: list[bool] = []
    for path in paths:
        outcomes.extend(load_json(path)["metrics"]["episode_successes"])
    return fmean(float(value) for value in outcomes) * 100.0


def plot_codebook_quality_and_rigid() -> None:
    output = ASSET_ROOT / "codebook_quality_rigid"
    output.mkdir(parents=True, exist_ok=True)
    experiment_root = STABLEWM_ROOT / "experiments" / "codebook_quality_rigid_v1"
    summary = load_json(experiment_root / "experiment_summary.json")

    conditions = ["k512_original", "k2048_original", "k8192_original"]
    labels = ["K512", "K2048", "K8192"]
    absolute = [summary["codebook_quality"][name]["absolute_l2_mean"] for name in conditions]
    relative = [
        summary["codebook_quality"][name]["relative_l2_mean"] * 100.0
        for name in conditions
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    positions = np.arange(len(labels))
    bars = axes[0].bar(positions, absolute, color=[BLUE, ORANGE, GREEN])
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Test mean absolute L2")
    axes[0].set_title("Absolute quantization error")
    annotate_bars(axes[0], bars, decimals=3)
    style_axis(axes[0])
    bars = axes[1].bar(positions, relative, color=[BLUE, ORANGE, GREEN])
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Test mean relative L2 (%)")
    axes[1].set_title("Scale-normalized quantization error")
    annotate_bars(axes[1], bars, suffix="%", decimals=2)
    style_axis(axes[1])
    figure.suptitle("Offline codebook quality improves monotonically with K", fontsize=14)
    figure.savefig(output / "codebook_quantization_quality.png", dpi=180)
    plt.close(figure)

    heldout = [summary["conditions"][name]["mean_heldout_success_rate"] for name in conditions]
    heldout.append(summary["conditions"]["k8192_rigid"]["mean_heldout_success_rate"])
    prediction_root = (
        STABLEWM_ROOT / "experiments" / "lewm_pusht_prediction_only_seed3072_40ep"
    )
    prediction_epoch10 = aggregate_episode_successes(
        sorted((prediction_root / "evaluation_epoch10" / "heldout_test").glob("shard*/*.json"))
    )
    heldout.append(prediction_epoch10)
    heldout_labels = ["K512", "K2048", "K8192", "K8192 rigid", "Prediction-only"]

    comparison_order = [
        "k512_original_vs_k8192_original",
        "k2048_original_vs_k8192_original",
        "k8192_rigid_vs_k8192_original",
    ]
    comparison_labels = ["K512 − K8192", "K2048 − K8192", "K8192 rigid − original"]
    differences = np.asarray(
        [summary["comparisons"][name]["difference_percentage_points"] for name in comparison_order]
    )
    intervals = np.asarray(
        [summary["comparisons"][name]["ci90_percentage_points"] for name in comparison_order]
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    positions = np.arange(len(heldout_labels))
    bars = axes[0].barh(
        positions,
        heldout,
        color=[BLUE, ORANGE, GREEN, PURPLE, RED],
    )
    axes[0].set_yticks(positions, heldout_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 90)
    axes[0].set_xlabel("Held-out success rate (%)")
    axes[0].set_title("200 fixed PushT tasks")
    axes[0].bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
    style_axis(axes[0], grid_axis="x")

    y_positions = np.arange(len(comparison_labels))
    axes[1].axvspan(-5, 5, color=GREEN, alpha=0.12, label="±5 pp equivalence margin")
    axes[1].axvline(0, color=GRAY, linewidth=1)
    axes[1].errorbar(
        differences,
        y_positions,
        xerr=np.vstack((differences - intervals[:, 0], intervals[:, 1] - differences)),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=5,
        markersize=7,
    )
    axes[1].set_yticks(y_positions, comparison_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-10, 12)
    axes[1].set_xlabel("Success-rate difference (percentage points), 90% CI")
    axes[1].set_title("Paired comparisons remain inconclusive")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(axes[1], grid_axis="x")
    figure.suptitle("Task performance and paired uncertainty", fontsize=14)
    figure.savefig(output / "heldout_success_and_paired_ci.png", dpi=180)
    plt.close(figure)

    epochs, train_loss, validation_loss = prediction_only_epoch_metrics()
    checkpoint_epochs = np.asarray([10, 20, 30, 40])
    selection_rates = []
    for epoch in checkpoint_epochs:
        payload = load_json(
            prediction_root
            / f"evaluation_epoch{epoch}"
            / "selection"
            / f"pusht_results_epoch{epoch}_selection_50.json"
        )
        selection_rates.append(payload["metrics"]["success_rate"])
    heldout_epochs = np.asarray([10, 20])
    heldout_rates = [
        aggregate_episode_successes(
            sorted(
                (prediction_root / f"evaluation_epoch{epoch}" / "heldout_test").glob(
                    "shard*/*.json"
                )
            )
        )
        for epoch in heldout_epochs
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    axes[0].plot(epochs, train_loss, color=BLUE, linewidth=2, label="Train prediction MSE")
    axes[0].plot(
        epochs,
        validation_loss,
        color=ORANGE,
        linewidth=2,
        label="Validation prediction MSE",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Prediction MSE (log scale)")
    axes[0].set_title("Prediction loss continues to collapse")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    axes[1].plot(
        checkpoint_epochs,
        selection_rates,
        marker="o",
        linewidth=2,
        color=RED,
        label="Selection (n=50)",
    )
    axes[1].plot(
        heldout_epochs,
        heldout_rates,
        marker="s",
        linewidth=2,
        color=PURPLE,
        label="Held-out (n=200)",
    )
    axes[1].set_xticks(checkpoint_epochs)
    axes[1].set_ylim(0, 8)
    axes[1].set_xlabel("Checkpoint epoch")
    axes[1].set_ylabel("PushT success rate (%)")
    axes[1].set_title("Task performance does not recover")
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    figure.suptitle("Prediction-only, no-SIGReg collapse diagnostic", fontsize=14)
    figure.savefig(output / "prediction_only_collapse.png", dpi=180)
    plt.close(figure)


def final_codebook_metrics(run_dir: Path) -> dict[str, float]:
    with (run_dir / "metrics.csv").open(newline="") as stream:
        row = list(csv.DictReader(stream))[-1]
    return {key: float(value) for key, value in row.items() if key != "epoch"}


def task_rates_from_multitask_summary(path: Path) -> dict[str, float]:
    payload = load_json(path)
    return {row["task"]: row["metrics"]["success_rate"] for row in payload["tasks"]}


def draw_box(axis, center, text, *, width=0.20, height=0.17, color=BLUE) -> None:
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color,
        alpha=0.13,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=10)


def draw_arrow(axis, start, end, *, color=GRAY) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=color,
        )
    )


def plot_pusht_tworoom_fusion() -> None:
    output = ASSET_ROOT / "pusht_tworoom_fusion"
    output.mkdir(parents=True, exist_ok=True)

    pusht_codebook = (
        STABLEWM_ROOT / "codebook_runs" / "official_lewm_pusht_compat_codebook_k8192"
    )
    tworoom_codebook = (
        STABLEWM_ROOT
        / "codebook_runs"
        / "official_lewm_tworooms_compat_codebook_k8192"
    )
    codebook_runs = [pusht_codebook, tworoom_codebook]
    codebook_labels = ["PushT", "Two-Room"]
    evaluations = [load_json(path / "quantization_evaluation.json") for path in codebook_runs]
    final_metrics = [final_codebook_metrics(path) for path in codebook_runs]

    relative_stats = ["mean", "median", "p95"]
    relative_values = np.asarray(
        [
            [evaluation["splits"]["test"]["relative_l2"][stat] * 100.0 for stat in relative_stats]
            for evaluation in evaluations
        ]
    )
    active = [row["val_active_fraction"] * 100.0 for row in final_metrics]
    normalized_perplexity = [
        row["val_perplexity"] / 8192.0 * 100.0 for row in final_metrics
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    positions = np.arange(len(codebook_labels))
    width = 0.23
    for index, (stat, color) in enumerate(zip(relative_stats, [BLUE, ORANGE, PURPLE], strict=True)):
        bars = axes[0].bar(
            positions + (index - 1) * width,
            relative_values[:, index],
            width,
            color=color,
            label=stat,
        )
        annotate_bars(axes[0], bars, decimals=1)
    axes[0].set_xticks(positions, codebook_labels)
    axes[0].set_ylabel("Test relative L2 (%)")
    axes[0].set_title("Held-out quantization distortion")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    bars_active = axes[1].bar(
        positions - 0.18,
        active,
        0.36,
        color=BLUE,
        label="Active codes / K",
    )
    bars_perplexity = axes[1].bar(
        positions + 0.18,
        normalized_perplexity,
        0.36,
        color=ORANGE,
        label="Perplexity / K",
    )
    axes[1].set_xticks(positions, codebook_labels)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Validation utilization (%)")
    axes[1].set_title("Final K8192 codebook utilization")
    axes[1].legend(frameon=False)
    annotate_bars(axes[1], bars_active, suffix="%", decimals=1)
    annotate_bars(axes[1], bars_perplexity, suffix="%", decimals=1)
    style_axis(axes[1])
    figure.suptitle("Single-task K8192 codebooks differ sharply in difficulty", fontsize=14)
    figure.savefig(output / "single_task_codebook_quality.png", dpi=180)
    plt.close(figure)

    alignment = load_json(STABLEWM_ROOT / "multitask" / "pusht_tworoom_alignment.json")
    before = alignment["identity_validation"]
    after = alignment["validation"]
    error_metrics = ["mse", "rmse", "normalized_rmse"]
    score_metrics = ["cosine_similarity", "r2"]
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    positions = np.arange(len(error_metrics))
    width = 0.35
    bars_before = axes[0].bar(
        positions - width / 2,
        [before[key] for key in error_metrics],
        width,
        color=GRAY,
        label="Identity",
    )
    bars_after = axes[0].bar(
        positions + width / 2,
        [after[key] for key in error_metrics],
        width,
        color=GREEN,
        label="Similarity aligned",
    )
    axes[0].set_xticks(positions, ["MSE", "RMSE", "Normalized RMSE"])
    axes[0].set_ylabel("Held-out error")
    axes[0].set_title(f"MSE improves {after['mse_improvement_ratio']:.2f}×")
    axes[0].legend(frameon=False)
    annotate_bars(axes[0], bars_before, decimals=2)
    annotate_bars(axes[0], bars_after, decimals=2)
    style_axis(axes[0])

    positions = np.arange(len(score_metrics))
    bars_before = axes[1].bar(
        positions - width / 2,
        [before[key] for key in score_metrics],
        width,
        color=GRAY,
        label="Identity",
    )
    bars_after = axes[1].bar(
        positions + width / 2,
        [after[key] for key in score_metrics],
        width,
        color=GREEN,
        label="Similarity aligned",
    )
    axes[1].axhline(0, color=GRAY, linewidth=1)
    axes[1].set_xticks(positions, ["Cosine similarity", "R²"])
    axes[1].set_ylabel("Held-out score")
    axes[1].set_title("Coordinate agreement improves, residual remains")
    axes[1].legend(frameon=False)
    annotate_bars(axes[1], bars_before, decimals=2)
    annotate_bars(axes[1], bars_after, decimals=2)
    style_axis(axes[1])
    figure.suptitle("Two-Room → PushT Similarity Procrustes alignment", fontsize=14)
    figure.savefig(output / "alignment_before_after.png", dpi=180)
    plt.close(figure)

    fusion = load_json(STABLEWM_ROOT / "checkpoints" / "pusht_tworoom_fused_uot" / "metadata.json")
    figure, axis = plt.subplots(figsize=(13.2, 5.4), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    draw_box(axis, (0.12, 0.72), "PushT codebook\nK=8192", color=BLUE)
    draw_box(axis, (0.12, 0.30), "Two-Room codebook\nK=8192", color=ORANGE)
    draw_box(
        axis,
        (0.41, 0.51),
        f"UOT transport\n{fusion['mutual_candidate_count']} mutual candidates",
        width=0.24,
        color=PURPLE,
    )
    draw_box(
        axis,
        (0.67, 0.51),
        f"Hard safety gates\n{fusion['num_merges']} accepted merges",
        width=0.22,
        color=RED,
    )
    draw_box(
        axis,
        (0.89, 0.51),
        f"Final codebook\nK={fusion['num_embeddings']} (concat)",
        width=0.19,
        color=GREEN,
    )
    draw_arrow(axis, (0.22, 0.68), (0.30, 0.57))
    draw_arrow(axis, (0.22, 0.34), (0.30, 0.45))
    draw_arrow(axis, (0.53, 0.51), (0.56, 0.51))
    draw_arrow(axis, (0.78, 0.51), (0.79, 0.51))
    axis.text(
        0.5,
        0.11,
        "Teacher-token support remains task-disjoint: Jaccard = 0,  I(token; task) = 1.000 bit",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F6F6F6", "edgecolor": LIGHT_GRAY},
    )
    axis.set_title("UOT converged numerically but produced no compact fusion", fontsize=14)
    figure.savefig(output / "uot_zero_merge_outcome.png", dpi=180)
    plt.close(figure)

    m2_root = STABLEWM_ROOT / "multitask_distillation" / "pusht_tworoom_uot_seed3072"
    m2_metrics = load_jsonl(m2_root / "metrics.jsonl")
    epochs = [row["epoch"] for row in m2_metrics]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)
    for axis in axes:
        add_phase_spans(axis)
    for key, label, color in (
        ("train/total_loss", "Total loss", BLUE),
        ("train/latent_mse", "Latent MSE", ORANGE),
        ("train/prediction_mse", "Prediction MSE", GREEN),
        ("train/token_kl", "Token KL", PURPLE),
    ):
        axes[0].plot(
            epochs,
            [row[key] for row in m2_metrics],
            linewidth=1.9,
            marker="o",
            markersize=3,
            color=color,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training metric (log scale)")
    axes[0].set_title("M2 optimization converges")
    axes[0].legend(frameon=False, fontsize=8)
    style_axis(axes[0])

    for task, color in (("pusht", BLUE), ("tworoom", ORANGE)):
        axes[1].plot(
            epochs,
            [row["validation"][task]["student_prediction_mse"] for row in m2_metrics],
            linewidth=1.9,
            marker="o",
            markersize=3,
            color=color,
            label=task,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation student prediction MSE")
    axes[1].set_title("Both task heads improve")
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    figure.suptitle("M2 aligned dual-codebook distillation training", fontsize=14)
    figure.savefig(output / "m2_training_convergence.png", dpi=180)
    plt.close(figure)

    m3_root = STABLEWM_ROOT / "multitask_baseline" / "pusht_tworoom_m3_seed3072"
    m3_metrics = load_jsonl(m3_root / "metrics.jsonl")
    epochs = [row["epoch"] for row in m3_metrics]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)
    axes[0].plot(
        epochs,
        [row["train/loss"] for row in m3_metrics],
        linewidth=2,
        color=BLUE,
        label="Total loss",
    )
    axes[0].plot(
        epochs,
        [row["train/prediction_mse"] for row in m3_metrics],
        linewidth=2,
        color=ORANGE,
        label="Prediction MSE",
    )
    axes[0].plot(
        epochs,
        [0.09 * row["train/sigreg"] for row in m3_metrics],
        linewidth=2,
        color=PURPLE,
        label="0.09 × SIGReg",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training objective (log scale)")
    axes[0].set_title("SIGReg reduction dominates the objective")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    for task, color in (("pusht", RED), ("tworoom", GREEN)):
        axes[1].plot(
            epochs,
            [row["validation"][task]["prediction_mse"] for row in m3_metrics],
            linewidth=2,
            marker="o",
            markersize=3,
            color=color,
            label=task,
        )
    axes[1].axvspan(7.5, 8.5, color=RED, alpha=0.12, label="Epoch 7→8 transition")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation prediction MSE (log scale)")
    axes[1].set_title("PushT representation fails abruptly")
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    figure.suptitle("M3 continuous multitask negative transfer", fontsize=14)
    figure.savefig(output / "m3_negative_transfer.png", dpi=180)
    plt.close(figure)

    p0 = extract_success_rate(
        STABLEWM_ROOT
        / "checkpoints"
        / "official_lewm_pusht_compat"
        / "pusht_results_official_seeded_50.txt"
    )
    p1 = load_json(
        STABLEWM_ROOT
        / "joint_distillation"
        / "lewm_pusht_k8192_seed3072"
        / "task_evaluation"
        / "summary.json"
    )["best_stage"]["success_rate"]
    r0_payload = load_json(
        STABLEWM_ROOT
        / "checkpoints"
        / "official_lewm_tworooms_compat"
        / "task_evaluation"
        / "tworoom_results_official_seed42_50.json"
    )
    r0 = r0_payload["metrics"]["success_rate"]
    r1 = load_json(
        STABLEWM_ROOT
        / "joint_distillation"
        / "lewm_tworooms_k8192_seed3072"
        / "task_evaluation"
        / "summary.json"
    )["best_stage"]["success_rate"]
    m0 = task_rates_from_multitask_summary(
        STABLEWM_ROOT
        / "multitask_distillation"
        / "pusht_tworoom_m0_unaligned_concat_seed3072"
        / "task_evaluation"
        / "summary.json"
    )
    m2 = task_rates_from_multitask_summary(m2_root / "task_evaluation" / "summary.json")
    m3 = task_rates_from_multitask_summary(m3_root / "task_evaluation" / "summary.json")

    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), constrained_layout=True)
    task_positions = np.arange(2)
    width = 0.34
    bars_continuous = axes[0].bar(
        task_positions - width / 2,
        [p0, r0],
        width,
        color=BLUE,
        label="Official continuous",
    )
    bars_vq = axes[0].bar(
        task_positions + width / 2,
        [p1, r1],
        width,
        color=ORANGE,
        label="Single-task VQ",
    )
    axes[0].set_xticks(task_positions, ["PushT", "Two-Room"])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Single-task controls")
    axes[0].legend(frameon=False)
    annotate_bars(axes[0], bars_continuous, suffix="%", decimals=0)
    annotate_bars(axes[0], bars_vq, suffix="%", decimals=0)
    style_axis(axes[0])

    model_labels = ["M3 continuous", "M0 unaligned", "M2 aligned"]
    model_rows = [m3, m0, m2]
    model_positions = np.arange(len(model_labels))
    push_rates = [row["pusht"] for row in model_rows]
    room_rates = [row["tworoom"] for row in model_rows]
    macro_rates = [(left + right) / 2 for left, right in zip(push_rates, room_rates, strict=True)]
    bars_push = axes[1].bar(
        model_positions - width / 2,
        push_rates,
        width,
        color=BLUE,
        label="PushT",
    )
    bars_room = axes[1].bar(
        model_positions + width / 2,
        room_rates,
        width,
        color=ORANGE,
        label="Two-Room",
    )
    axes[1].plot(
        model_positions,
        macro_rates,
        color="black",
        marker="D",
        linewidth=1.5,
        label="Macro average",
    )
    axes[1].set_xticks(model_positions, model_labels)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_title("Shared multitask models")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    annotate_bars(axes[1], bars_push, suffix="%", decimals=0)
    annotate_bars(axes[1], bars_room, suffix="%", decimals=0)
    style_axis(axes[1])
    figure.suptitle("Fixed-start MPC success-rate matrix", fontsize=14)
    figure.savefig(output / "mpc_success_rate_matrix.png", dpi=180)
    plt.close(figure)


def plot_teacher_representation_ablation() -> None:
    output = ASSET_ROOT / "pusht_tworoom_fusion"
    output.mkdir(parents=True, exist_ok=True)
    base = STABLEWM_ROOT / "multitask_distillation"
    m2 = task_rates_from_multitask_summary(
        base / "pusht_tworoom_uot_seed3072" / "task_evaluation" / "summary.json"
    )
    m4 = task_rates_from_multitask_summary(
        base
        / "pusht_tworoom_m4_continuous_seed3072"
        / "task_evaluation"
        / "summary.json"
    )
    m5 = task_rates_from_multitask_summary(
        base
        / "pusht_tworoom_m5_codebook_seed3072"
        / "task_evaluation"
        / "summary.json"
    )

    labels = ["M2 (mixed)", "M4 (all continuous)", "M5 (all codebook)"]
    model_rows = [m2, m4, m5]
    positions = np.arange(len(labels))
    width = 0.34
    push_rates = [row["pusht"] for row in model_rows]
    room_rates = [row["tworoom"] for row in model_rows]
    macro_rates = [
        (left + right) / 2
        for left, right in zip(push_rates, room_rates, strict=True)
    ]

    figure, axis = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    bars_push = axis.bar(
        positions - width / 2, push_rates, width, color=BLUE, label="PushT"
    )
    bars_room = axis.bar(
        positions + width / 2, room_rates, width, color=ORANGE, label="Two-Room"
    )
    axis.plot(
        positions,
        macro_rates,
        color="black",
        marker="D",
        linewidth=1.5,
        label="Macro average",
    )
    for x, value in zip(positions, macro_rates, strict=True):
        axis.annotate(
            f"{value:.1f}",
            (x, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Success rate (%)")
    axis.set_title("Teacher-representation ablation on the M2 framework")
    axis.legend(frameon=False, ncol=3, fontsize=9)
    annotate_bars(axis, bars_push, suffix="%", decimals=0)
    annotate_bars(axis, bars_room, suffix="%", decimals=0)
    style_axis(axis)
    figure.savefig(output / "teacher_representation_ablation.png", dpi=180)
    plt.close(figure)


def main() -> None:
    plot_k8192_joint_distillation()
    plot_codebook_quality_and_rigid()
    plot_pusht_tworoom_fusion()
    plot_teacher_representation_ablation()
    print("Generated figures for all three experiment reports.")


if __name__ == "__main__":
    main()
