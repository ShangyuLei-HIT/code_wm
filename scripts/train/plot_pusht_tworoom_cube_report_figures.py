"""Generate figures for the PushT x Two-Room x Cube three-task report.

All numbers are read from the real evaluation JSON / metadata / metrics files
under .stablewm; nothing is hard-coded. Figure text is English because the
container has no CJK font installed (matching the existing report figures).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STABLEWM_ROOT = PROJECT_ROOT / ".stablewm"
ASSET_ROOT = PROJECT_ROOT / "docs" / "assets" / "pusht_tworoom_cube_fusion"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#7A5195"
TEAL = "#2A9D8F"
GRAY = "#7F7F7F"
LIGHT_GRAY = "#D9D9D9"

TASK_COLORS = {"pusht": BLUE, "tworoom": ORANGE, "cube": TEAL}
TASK_LABELS = {"pusht": "PushT", "tworoom": "Two-Room", "cube": "Cube"}


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract_success_rate_txt(path: Path) -> float:
    match = re.search(r"success_rate['\"]?:\s*([0-9.]+)", path.read_text())
    if match is None:
        raise ValueError(f"success_rate not found in {path}")
    return float(match.group(1))


def style_axis(axis, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def annotate_bars(axis, bars, *, suffix: str = "", decimals: int = 1) -> None:
    axis.bar_label(bars, fmt=f"%.{decimals}f{suffix}", padding=3, fontsize=8)


def add_phase_spans(axis) -> None:
    # M2 schedule: 4 / 10 / 2 epochs -> phase 1 [1,4], phase 2 [5,14], phase 3 [15,16]
    spans = (
        (0.5, 4.5, "#E8F1FA", "Phase 1"),
        (4.5, 14.5, "#FDF0E3", "Phase 2"),
        (14.5, 16.5, "#E9F5ED", "Phase 3"),
    )
    for left, right, color, label in spans:
        axis.axvspan(left, right, color=color, alpha=0.65, zorder=0)
        axis.text(
            (left + right) / 2, 1.01, label,
            transform=axis.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color=GRAY,
        )


def codebook_relative_l2(name: str) -> dict:
    payload = load_json(
        STABLEWM_ROOT / "codebook_runs" / name / "quantization_evaluation.json"
    )
    return payload["splits"]["test"]["relative_l2"]


def codebook_final_row(name: str) -> dict:
    rows = list(csv.DictReader(
        (STABLEWM_ROOT / "codebook_runs" / name / "metrics.csv").open(newline="")
    ))
    return rows[-1]


def multitask_rates(path: Path) -> dict:
    payload = load_json(path)
    return {row["task"]: row["metrics"]["success_rate"] for row in payload["tasks"]}


CODEBOOKS = {
    "pusht": "official_lewm_pusht_compat_codebook_k8192",
    "tworoom": "official_lewm_tworooms_compat_codebook_k8192",
    "cube": "official_lewm_cube_compat_codebook_k8192",
}
TASK_ORDER = ["pusht", "tworoom", "cube"]


def fig_single_task_codebook_quality() -> None:
    labels = [TASK_LABELS[t] for t in TASK_ORDER]
    stats = ["mean", "median", "p95"]
    rel = np.asarray([
        [codebook_relative_l2(CODEBOOKS[t])[s] * 100.0 for s in stats]
        for t in TASK_ORDER
    ])
    final_rows = {t: codebook_final_row(CODEBOOKS[t]) for t in TASK_ORDER}
    active = [float(final_rows[t]["val_active_fraction"]) * 100.0 for t in TASK_ORDER]
    norm_ppl = [float(final_rows[t]["val_perplexity"]) / 8192.0 * 100.0 for t in TASK_ORDER]

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    positions = np.arange(len(labels))
    width = 0.25
    for index, (stat, color) in enumerate(zip(stats, [BLUE, ORANGE, PURPLE])):
        bars = axes[0].bar(positions + (index - 1) * width, rel[:, index], width,
                           color=color, label=stat)
        annotate_bars(axes[0], bars, decimals=1)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Test relative L2 (%)")
    axes[0].set_title("Held-out quantization distortion (K=8192)")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    bars_active = axes[1].bar(positions - 0.18, active, 0.36, color=BLUE,
                             label="Active codes / K")
    bars_ppl = axes[1].bar(positions + 0.18, norm_ppl, 0.36, color=ORANGE,
                          label="Perplexity / K")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Validation utilization (%)")
    axes[1].set_title("Final K8192 codebook utilization")
    axes[1].legend(frameon=False)
    annotate_bars(axes[1], bars_active, suffix="%", decimals=1)
    annotate_bars(axes[1], bars_ppl, suffix="%", decimals=1)
    style_axis(axes[1])
    figure.suptitle("Single-task K8192 codebooks: PushT and Cube quantize hardest", fontsize=14)
    figure.savefig(ASSET_ROOT / "single_task_codebook_quality.png", dpi=180)
    plt.close(figure)


def fig_alignment_before_after() -> None:
    alignment = load_json(STABLEWM_ROOT / "multitask" / "pusht_tworoom_cube_alignment.json")
    sources = ["tworoom", "cube"]
    error_metrics = ["mse", "rmse", "normalized_rmse"]
    error_titles = ["MSE", "RMSE", "Normalized RMSE"]
    score_metrics = ["cosine_similarity", "r2"]
    score_titles = ["Cosine similarity", "R²"]

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.2), constrained_layout=True)
    width = 0.35
    for row, source in enumerate(sources):
        before = alignment[source]["identity_validation"]
        after = alignment[source]["validation"]
        ratio = after["mse_improvement_ratio"]

        positions = np.arange(len(error_metrics))
        b0 = axes[row, 0].bar(positions - width / 2, [before[k] for k in error_metrics],
                              width, color=GRAY, label="Identity")
        a0 = axes[row, 0].bar(positions + width / 2, [after[k] for k in error_metrics],
                              width, color=TASK_COLORS[source], label="Similarity aligned")
        axes[row, 0].set_xticks(positions, error_titles)
        axes[row, 0].set_ylabel("Held-out error")
        axes[row, 0].set_title(f"{TASK_LABELS[source]} → PushT: MSE improves {ratio:.2f}×")
        axes[row, 0].legend(frameon=False)
        annotate_bars(axes[row, 0], b0, decimals=2)
        annotate_bars(axes[row, 0], a0, decimals=2)
        style_axis(axes[row, 0])

        positions = np.arange(len(score_metrics))
        b1 = axes[row, 1].bar(positions - width / 2, [before[k] for k in score_metrics],
                              width, color=GRAY, label="Identity")
        a1 = axes[row, 1].bar(positions + width / 2, [after[k] for k in score_metrics],
                              width, color=TASK_COLORS[source], label="Similarity aligned")
        axes[row, 1].axhline(0, color=GRAY, linewidth=1)
        axes[row, 1].set_xticks(positions, score_titles)
        axes[row, 1].set_ylabel("Held-out score")
        tok = alignment[source]["source_token_preservation"]
        axes[row, 1].set_title(f"Coordinate agreement improves (token preservation={tok:.0%})")
        axes[row, 1].legend(frameon=False)
        annotate_bars(axes[row, 1], b1, decimals=2)
        annotate_bars(axes[row, 1], a1, decimals=2)
        style_axis(axes[row, 1])
    figure.suptitle("Two-source Similarity Procrustes alignment to PushT space", fontsize=15)
    figure.savefig(ASSET_ROOT / "alignment_before_after.png", dpi=180)
    plt.close(figure)


def draw_box(axis, center, text, *, width=0.20, height=0.16, color=BLUE) -> None:
    x, y = center
    patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                           boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.13)
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=9.5)


def draw_arrow(axis, start, end, *, color=GRAY) -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                   mutation_scale=13, linewidth=1.4, color=color))


def fig_uot_zero_merge() -> None:
    fusion = load_json(
        STABLEWM_ROOT / "checkpoints" / "pusht_tworoom_cube_fused_uot" / "metadata.json"
    )
    stages = fusion["stages"]
    figure, axis = plt.subplots(figsize=(13.4, 5.8), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    draw_box(axis, (0.11, 0.80), "PushT K=8192", color=BLUE)
    draw_box(axis, (0.11, 0.50), "Two-Room K=8192", color=ORANGE)
    draw_box(axis, (0.11, 0.20), "Cube K=8192", color=TEAL)

    s1 = stages[0]
    draw_box(axis, (0.40, 0.65),
             f"Stage 1 UOT\nPushT + Two-Room\n{s1['mutual_candidate_count']} candidates → {s1['num_merges']} merges\nK={s1['num_codes_after']}",
             width=0.26, height=0.24, color=PURPLE)
    s2 = stages[1]
    draw_box(axis, (0.70, 0.45),
             f"Stage 2 UOT\n+ Cube\n{s2['mutual_candidate_count']} candidates → {s2['num_merges']} merges\nK={s2['num_codes_after']}",
             width=0.26, height=0.24, color=RED)
    draw_box(axis, (0.92, 0.45),
             f"Final\nK={fusion['num_embeddings']}\n(concat)",
             width=0.14, height=0.20, color=GREEN)

    draw_arrow(axis, (0.20, 0.78), (0.29, 0.70))
    draw_arrow(axis, (0.20, 0.52), (0.29, 0.62))
    draw_arrow(axis, (0.53, 0.62), (0.59, 0.50))
    draw_arrow(axis, (0.20, 0.22), (0.59, 0.40))
    draw_arrow(axis, (0.83, 0.45), (0.85, 0.45))

    axis.text(0.5, 0.055,
              f"Total accepted merges = {fusion['num_merges']} under the 2% per-task QE budget; "
              f"final K_shared = {fusion['num_embeddings']} (= 3 x 8192). "
              "Gain over M0 comes from alignment, not codebook compression.",
              ha="center", va="center", fontsize=10.5,
              bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F6F6F6", "edgecolor": LIGHT_GRAY})
    axis.set_title("Sequential UOT converged but accepted zero cross-task merges", fontsize=14)
    figure.savefig(ASSET_ROOT / "uot_zero_merge_outcome.png", dpi=180)
    plt.close(figure)


def fig_m2_training_convergence() -> None:
    root = STABLEWM_ROOT / "multitask_distillation" / "pusht_tworoom_cube_uot_seed3072"
    metrics = load_jsonl(root / "metrics.jsonl")
    epochs = [row["epoch"] for row in metrics]

    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.1), constrained_layout=True)
    for axis in axes:
        add_phase_spans(axis)
    for key, label, color in (
        ("train/total_loss", "Total loss", BLUE),
        ("train/latent_mse", "Latent MSE", ORANGE),
        ("train/prediction_mse", "Prediction MSE", GREEN),
        ("train/token_kl", "Token KL", PURPLE),
    ):
        axes[0].plot(epochs, [row[key] for row in metrics], linewidth=1.9,
                     marker="o", markersize=3, color=color, label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training metric (log scale)")
    axes[0].set_title("M2 optimization converges")
    axes[0].legend(frameon=False, fontsize=8)
    style_axis(axes[0])

    for task in TASK_ORDER:
        axes[1].plot(epochs,
                     [row["validation"][task]["student_prediction_mse"] for row in metrics],
                     linewidth=1.9, marker="o", markersize=3,
                     color=TASK_COLORS[task], label=TASK_LABELS[task])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation student prediction MSE")
    axes[1].set_title("All three task heads improve")
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    figure.suptitle("M2 aligned tri-codebook distillation training", fontsize=14)
    figure.savefig(ASSET_ROOT / "m2_training_convergence.png", dpi=180)
    plt.close(figure)


def fig_m3_negative_transfer() -> None:
    root = STABLEWM_ROOT / "multitask_baseline" / "pusht_tworoom_cube_m3_seed3072"
    metrics = load_jsonl(root / "metrics.jsonl")
    epochs = [row["epoch"] for row in metrics]
    sigreg_weight = 0.09

    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.1), constrained_layout=True)
    axes[0].plot(epochs, [row["train/loss"] for row in metrics],
                 linewidth=2, color=BLUE, label="Total loss")
    axes[0].plot(epochs, [row["train/prediction_mse"] for row in metrics],
                 linewidth=2, color=ORANGE, label="Prediction MSE")
    axes[0].plot(epochs, [sigreg_weight * row["train/sigreg"] for row in metrics],
                 linewidth=2, color=PURPLE, label=f"{sigreg_weight} x SIGReg")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training objective (log scale)")
    axes[0].set_title("SIGReg reduction dominates the objective")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    for task in TASK_ORDER:
        axes[1].plot(epochs,
                     [row["validation"][task]["prediction_mse"] for row in metrics],
                     linewidth=2, marker="o", markersize=3,
                     color=TASK_COLORS[task], label=TASK_LABELS[task])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation prediction MSE (log scale)")
    axes[1].set_title("PushT dynamics stays weakest under the shared objective")
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    figure.suptitle("M3 continuous tri-task baseline training", fontsize=14)
    figure.savefig(ASSET_ROOT / "m3_training_dynamics.png", dpi=180)
    plt.close(figure)


def fig_success_matrix() -> None:
    p0 = extract_success_rate_txt(
        STABLEWM_ROOT / "checkpoints" / "official_lewm_pusht_compat"
        / "pusht_results_official_seeded_50.txt"
    )
    p1 = load_json(
        STABLEWM_ROOT / "joint_distillation" / "lewm_pusht_k8192_seed3072"
        / "task_evaluation" / "summary.json"
    )["best_stage"]["success_rate"]
    r0 = load_json(
        STABLEWM_ROOT / "checkpoints" / "official_lewm_tworooms_compat"
        / "task_evaluation" / "tworoom_results_official_seed42_50.json"
    )["metrics"]["success_rate"]
    r1 = load_json(
        STABLEWM_ROOT / "joint_distillation" / "lewm_tworooms_k8192_seed3072"
        / "task_evaluation" / "summary.json"
    )["best_stage"]["success_rate"]
    c0 = load_json(
        STABLEWM_ROOT / "checkpoints" / "official_lewm_cube_compat"
        / "task_evaluation" / "cube_results_official_seed42_50.json"
    )["metrics"]["success_rate"]
    c1 = load_json(
        STABLEWM_ROOT / "joint_distillation" / "lewm_cube_k8192_seed3072"
        / "task_evaluation" / "summary.json"
    )["best_stage"]["success_rate"]

    base = STABLEWM_ROOT / "multitask_distillation"
    m0 = multitask_rates(base / "pusht_tworoom_cube_m0_unaligned_concat_seed3072"
                         / "task_evaluation" / "summary.json")
    m2 = multitask_rates(base / "pusht_tworoom_cube_uot_seed3072"
                         / "task_evaluation" / "summary.json")
    m3 = multitask_rates(STABLEWM_ROOT / "multitask_baseline"
                         / "pusht_tworoom_cube_m3_seed3072"
                         / "task_evaluation" / "summary.json")

    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), constrained_layout=True)

    task_positions = np.arange(3)
    width = 0.34
    cont = [p0, r0, c0]
    vq = [p1, r1, c1]
    bars_c = axes[0].bar(task_positions - width / 2, cont, width, color=BLUE,
                         label="Official continuous")
    bars_v = axes[0].bar(task_positions + width / 2, vq, width, color=ORANGE,
                         label="Single-task VQ (best stage)")
    axes[0].set_xticks(task_positions, [TASK_LABELS[t] for t in TASK_ORDER])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Single-task controls")
    axes[0].legend(frameon=False)
    annotate_bars(axes[0], bars_c, suffix="%", decimals=0)
    annotate_bars(axes[0], bars_v, suffix="%", decimals=0)
    style_axis(axes[0])

    model_labels = ["M3 continuous", "M0 unaligned", "M2 aligned"]
    model_rows = [m3, m0, m2]
    model_positions = np.arange(len(model_labels))
    width = 0.26
    offsets = {"pusht": -width, "tworoom": 0.0, "cube": width}
    for task in TASK_ORDER:
        vals = [row[task] for row in model_rows]
        bars = axes[1].bar(model_positions + offsets[task], vals, width,
                           color=TASK_COLORS[task], label=TASK_LABELS[task])
        annotate_bars(axes[1], bars, suffix="%", decimals=0)
    macro = [sum(row.values()) / 3.0 for row in model_rows]
    axes[1].plot(model_positions, macro, color="black", marker="D",
                 linewidth=1.5, label="Macro average")
    for x, m in zip(model_positions, macro):
        axes[1].annotate(f"{m:.1f}", (x, m), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
    axes[1].set_xticks(model_positions, model_labels)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_title("Shared tri-task models")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    style_axis(axes[1])
    figure.suptitle("Fixed-start MPC success-rate matrix (Seed=42, 50 ep/task)", fontsize=14)
    figure.savefig(ASSET_ROOT / "mpc_success_rate_matrix.png", dpi=180)
    plt.close(figure)


def fig_teacher_representation_ablation() -> None:
    base = STABLEWM_ROOT / "multitask_distillation"
    m2 = multitask_rates(base / "pusht_tworoom_cube_uot_seed3072"
                         / "task_evaluation" / "summary.json")
    m4 = multitask_rates(base / "pusht_tworoom_cube_m4_continuous_seed3072"
                         / "task_evaluation" / "summary.json")
    m5 = multitask_rates(base / "pusht_tworoom_cube_m5_codebook_seed3072"
                         / "task_evaluation" / "summary.json")

    labels = ["M2 (mixed)", "M4 (all continuous)", "M5 (all codebook)"]
    rows = [m2, m4, m5]
    positions = np.arange(len(labels))
    width = 0.26
    offsets = {"pusht": -width, "tworoom": 0.0, "cube": width}

    figure, axis = plt.subplots(figsize=(11.2, 5.6), constrained_layout=True)
    for task in TASK_ORDER:
        vals = [row[task] for row in rows]
        bars = axis.bar(positions + offsets[task], vals, width,
                        color=TASK_COLORS[task], label=TASK_LABELS[task])
        annotate_bars(axis, bars, suffix="%", decimals=0)
    macro = [sum(row.values()) / 3.0 for row in rows]
    axis.plot(positions, macro, color="black", marker="D", linewidth=1.5,
              label="Macro average")
    for x, m in zip(positions, macro):
        axis.annotate(f"{m:.1f}", (x, m), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Success rate (%)")
    axis.set_title("Teacher-representation ablation on the M2 framework")
    axis.legend(frameon=False, ncol=2, fontsize=9)
    style_axis(axis)
    figure.suptitle("Continuous teacher target matches M2; full discretization loses accuracy",
                    fontsize=13)
    figure.savefig(ASSET_ROOT / "teacher_representation_ablation.png", dpi=180)
    plt.close(figure)


def main() -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    fig_single_task_codebook_quality()
    fig_alignment_before_after()
    fig_uot_zero_merge()
    fig_m2_training_convergence()
    fig_m3_negative_transfer()
    fig_success_matrix()
    fig_teacher_representation_ablation()
    print("Generated all PushT x Two-Room x Cube report figures in", ASSET_ROOT)


if __name__ == "__main__":
    main()
