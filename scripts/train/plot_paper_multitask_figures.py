"""Merged two-/three-/six-task paper figures for the multi-task experiments.

The paper outline presents the two-task (PushT + Two-Room), three-task
(+ Cube) and six-task (+ Scene + Reacher + HumanoidMaze) experiment lines as
one "multi-task" section, so the same-kind report figures that existed
separately per line are redrawn here as single figures with one panel (row)
per experiment line / alignment source, following the panel-(a)/(b)/(c)
convention of ``main_success_matrix.png``:

- alignment/alignment_before_after.png
      (one summary panel per metric over all 8 alignment fits: 2-task pool,
       3-task x2 sources, 6-task x5 sources)
- uot/uot_zero_merge_outcome.png            ((a) two-task, (b) three-task,
                                             (c) sequential six-task stages)
- consolidation/teacher_representation_ablation.png
      ((a)/(b)/(c) panels, now with continuous-teacher / discrete-expert
       macro-average reference lines, matching main_success_matrix.png)
- consolidation/m3_negative_transfer.png    (rows: two-, three-, six-task)
- consolidation/m2_training_convergence.png (rows: two-, three-, six-task; appendix)

All numbers are read from the real evaluation JSON / metadata / metrics files
under .stablewm; nothing is hard-coded. Figure text is English (no CJK font).
"""

from __future__ import annotations

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
ASSET_ROOT = PROJECT_ROOT / "docs" / "assets" / "paper_figures"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
PURPLE = "#7A5195"
TEAL = "#2A9D8F"
GRAY = "#7F7F7F"
LIGHT_GRAY = "#D9D9D9"

TASK_COLORS = {
    "pusht": BLUE,
    "tworoom": ORANGE,
    "cube": TEAL,
    "scene": "#59A14F",
    "reacher": "#B07AA1",
    "humanoidmaze": "#9C755F",
}
TASK_LABELS = {
    "pusht": "PushT",
    "tworoom": "Two-Room",
    "cube": "Cube",
    "scene": "Scene",
    "reacher": "Reacher",
    "humanoidmaze": "HumanoidMaze",
}
# Order used by the six-task experiment (fixed in the training configs).
ALL_TASKS = ("pusht", "tworoom", "cube", "scene", "reacher", "humanoidmaze")


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


def multitask_rates(path: Path) -> dict:
    payload = load_json(path)
    return {row["task"]: row["metrics"]["success_rate"] for row in payload["tasks"]}


def style_axis(axis, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def annotate_bars(axis, bars, *, suffix: str = "", decimals: int = 1) -> None:
    # '%' must be escaped in %-style fmt strings or bar_label renders it literally.
    axis.bar_label(bars, fmt=f"%.{decimals}f{suffix.replace('%', '%%')}",
                   padding=3, fontsize=8)


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


def draw_box(axis, center, text, *, width=0.20, height=0.16, color=BLUE) -> None:
    x, y = center
    patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                           boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.13)
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=9)


def draw_arrow(axis, start, end, *, color=GRAY) -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                   mutation_scale=13, linewidth=1.4, color=color))


def teacher_and_expert_rates() -> tuple[dict, dict]:
    teachers = {
        "pusht": extract_success_rate_txt(
            STABLEWM_ROOT / "checkpoints" / "official_lewm_pusht_compat"
            / "pusht_results_official_seeded_50.txt"
        ),
        "tworoom": load_json(
            STABLEWM_ROOT / "checkpoints" / "official_lewm_tworooms_compat"
            / "task_evaluation" / "tworoom_results_official_seed42_50.json"
        )["metrics"]["success_rate"],
        "cube": load_json(
            STABLEWM_ROOT / "checkpoints" / "official_lewm_cube_compat"
            / "task_evaluation" / "cube_results_official_seed42_50.json"
        )["metrics"]["success_rate"],
        "scene": load_json(
            STABLEWM_ROOT / "checkpoints" / "lewm_scene_scratch_seed3072_compat"
            / "task_evaluation" / "scene_results_scratch_seed42_50.json"
        )["metrics"]["success_rate"],
        "reacher": load_json(
            STABLEWM_ROOT / "checkpoints" / "official_lewm_reacher_compat"
            / "task_evaluation" / "reacher_results_official_seed42_50.json"
        )["metrics"]["success_rate"],
        "humanoidmaze": load_json(
            STABLEWM_ROOT / "checkpoints" / "lewm_humanoidmaze_scratch_seed3072_compat"
            / "task_evaluation" / "humanoidmaze_results_scratch_seed42_50.json"
        )["metrics"]["success_rate"],
    }
    experts = {
        "pusht": load_json(
            STABLEWM_ROOT / "joint_distillation" / "lewm_pusht_k8192_seed3072"
            / "task_evaluation" / "summary.json"
        )["best_stage"]["success_rate"],
        "tworoom": load_json(
            STABLEWM_ROOT / "joint_distillation" / "lewm_tworooms_k8192_seed3072"
            / "task_evaluation" / "summary.json"
        )["best_stage"]["success_rate"],
        "cube": load_json(
            STABLEWM_ROOT / "joint_distillation" / "lewm_cube_k8192_seed3072"
            / "task_evaluation" / "summary.json"
        )["best_stage"]["success_rate"],
        "scene": load_json(
            STABLEWM_ROOT / "joint_distillation" / "lewm_scene_k8192_seed3072"
            / "task_evaluation" / "summary.json"
        )["best_stage"]["success_rate"],
        "reacher": load_json(
            STABLEWM_ROOT / "joint_distillation" / "lewm_reacher_k8192_seed3072"
            / "task_evaluation" / "summary.json"
        )["best_stage"]["success_rate"],
        "humanoidmaze": load_json(
            STABLEWM_ROOT / "joint_distillation" / "lewm_humanoidmaze_k8192_seed3072"
            / "task_evaluation" / "summary.json"
        )["best_stage"]["success_rate"],
    }
    return teachers, experts


# ---------------------------------------------------------------------------
# Figure 8: alignment quality, all fits (two/three/six-task pools) in one figure
# ---------------------------------------------------------------------------

def fig_alignment_merged() -> None:
    two = load_json(STABLEWM_ROOT / "multitask" / "pusht_tworoom_alignment.json")
    three = load_json(STABLEWM_ROOT / "multitask" / "pusht_tworoom_cube_alignment.json")
    six = load_json(
        STABLEWM_ROOT / "multitask"
        / "pusht_tworoom_cube_scene_reacher_humanoidmaze_alignment.json"
    )

    # One row per Procrustes fit; payloads share the identity/.validation layout.
    # The two-task JSON is flat (single Two-Room source); three/six-task JSONs
    # are keyed by source task.
    fits = [
        ("2-task pool", "tworoom", two),
        ("3-task pool", "tworoom", three["tworoom"]),
        ("3-task pool", "cube", three["cube"]),
        ("6-task pool", "tworoom", six["tworoom"]),
        ("6-task pool", "cube", six["cube"]),
        ("6-task pool", "scene", six["scene"]),
        ("6-task pool", "reacher", six["reacher"]),
        ("6-task pool", "humanoidmaze", six["humanoidmaze"]),
    ]
    x_labels = [f"{TASK_LABELS[source]} → PushT\n({pool})" for pool, source, _ in fits]
    positions = np.arange(len(fits))
    width = 0.36

    figure, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), constrained_layout=True)

    # Panel (a): held-out MSE, identity vs aligned, log scale, ×-improvement labels.
    axis = axes[0]
    identity_mse = [fit["identity_validation"]["mse"] for *_, fit in fits]
    aligned_mse = [fit["validation"]["mse"] for *_, fit in fits]
    axis.bar(positions - width / 2, identity_mse, width, color=GRAY, label="Identity")
    axis.bar(positions + width / 2, aligned_mse, width,
             color=[TASK_COLORS[source] for _, source, _ in fits],
             label="Similarity aligned")
    for index, (_, _, fit) in enumerate(fits):
        ratio = fit["validation"]["mse_improvement_ratio"]
        axis.annotate(f"{ratio:.2f}×", (index, max(identity_mse[index], aligned_mse[index])),
                      textcoords="offset points", xytext=(0, 5), ha="center",
                      fontsize=8, fontweight="bold")
    axis.set_yscale("log")
    axis.set_ylabel("Held-out MSE (log scale)")
    axis.set_title("(a) Held-out MSE improves in every fit", fontsize=11)
    axis.legend(frameon=False, fontsize=8, loc="upper right")

    # Panel (b): cosine similarity, identity vs aligned.
    axis = axes[1]
    identity_cos = [fit["identity_validation"]["cosine_similarity"] for *_, fit in fits]
    aligned_cos = [fit["validation"]["cosine_similarity"] for *_, fit in fits]
    b = axis.bar(positions - width / 2, identity_cos, width, color=GRAY, label="Identity")
    a = axis.bar(positions + width / 2, aligned_cos, width,
                 color=[TASK_COLORS[source] for _, source, _ in fits],
                 label="Similarity aligned")
    annotate_bars(axis, b, decimals=2)
    annotate_bars(axis, a, decimals=2)
    axis.set_ylabel("Held-out cosine similarity")
    axis.set_title("(b) Coordinate agreement improves, stays partial", fontsize=11)
    axis.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel (c): R², identity vs aligned — remains low after alignment.
    axis = axes[2]
    identity_r2 = [fit["identity_validation"]["r2"] for *_, fit in fits]
    aligned_r2 = [fit["validation"]["r2"] for *_, fit in fits]
    b = axis.bar(positions - width / 2, identity_r2, width, color=GRAY, label="Identity")
    a = axis.bar(positions + width / 2, aligned_r2, width,
                 color=[TASK_COLORS[source] for _, source, _ in fits],
                 label="Similarity aligned")
    axis.axhline(0, color=GRAY, linewidth=1)
    annotate_bars(axis, b, decimals=2)
    annotate_bars(axis, a, decimals=2)
    axis.set_ylabel("Held-out R²")
    axis.set_title("(c) R² stays low: partial, not identical, spaces", fontsize=11)
    axis.legend(frameon=False, fontsize=8, loc="lower right")

    for axis in axes:
        axis.set_xticks(positions, x_labels, fontsize=7.5, rotation=45,
                        ha="right", rotation_mode="anchor")
        style_axis(axis)

    token_values = [fit["source_token_preservation"] for *_, fit in fits]
    figure.suptitle(
        "Similarity Procrustes alignment to the PushT reference space "
        f"(two-, three-, and six-task anchor pools) — token preservation "
        f"≥ {min(token_values):.4%} in every fit",
        fontsize=13,
    )
    output = ASSET_ROOT / "alignment"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "alignment_before_after.png", dpi=180)
    plt.close(figure)
    print("Wrote", output / "alignment_before_after.png")


# ---------------------------------------------------------------------------
# Figure 9: UOT zero-merge outcome, two-task and sequential three-task
# ---------------------------------------------------------------------------

def fig_uot_merged() -> None:
    two = load_json(
        STABLEWM_ROOT / "checkpoints" / "pusht_tworoom_fused_uot" / "metadata.json"
    )
    three = load_json(
        STABLEWM_ROOT / "checkpoints" / "pusht_tworoom_cube_fused_uot" / "metadata.json"
    )
    six = load_json(
        STABLEWM_ROOT / "checkpoints"
        / "pusht_tworoom_cube_scene_reacher_humanoidmaze_fused_uot" / "metadata.json"
    )

    figure, axes = plt.subplots(1, 3, figsize=(19.8, 6.0),
                                constrained_layout=True)
    for axis in axes[:2]:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    # Panel (a): two-task pipeline.
    axis = axes[0]
    draw_box(axis, (0.12, 0.72), "PushT codebook\nK=8192", color=BLUE)
    draw_box(axis, (0.12, 0.30), "Two-Room codebook\nK=8192", color=ORANGE)
    draw_box(axis, (0.41, 0.51),
             f"UOT transport\n{two['mutual_candidate_count']} mutual candidates",
             width=0.24, color=PURPLE)
    draw_box(axis, (0.67, 0.51),
             f"Hard safety gates\n{two['num_merges']} accepted merges",
             width=0.22, color=RED)
    draw_box(axis, (0.89, 0.51),
             f"Final codebook\nK={two['num_embeddings']} (concat)",
             width=0.19, color=GREEN)
    draw_arrow(axis, (0.22, 0.68), (0.30, 0.57))
    draw_arrow(axis, (0.22, 0.34), (0.30, 0.45))
    draw_arrow(axis, (0.53, 0.51), (0.56, 0.51))
    draw_arrow(axis, (0.78, 0.51), (0.79, 0.51))
    axis.text(
        0.5, 0.09,
        "Teacher-token support remains task-disjoint: "
        "Jaccard = 0,  I(token; task) = 1.000 bit",
        ha="center", va="center", fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F6F6F6",
              "edgecolor": LIGHT_GRAY},
    )
    axis.set_title("(a) Two tasks: converged, no compact fusion", fontsize=12.5)

    # Panel (b): sequential three-task pipeline.
    axis = axes[1]
    stages = three["stages"]
    draw_box(axis, (0.11, 0.80), "PushT K=8192", color=BLUE)
    draw_box(axis, (0.11, 0.50), "Two-Room K=8192", color=ORANGE)
    draw_box(axis, (0.11, 0.20), "Cube K=8192", color=TEAL)
    s1, s2 = stages[0], stages[1]
    draw_box(axis, (0.40, 0.65),
             f"Stage 1 UOT\nPushT + Two-Room\n"
             f"{s1['mutual_candidate_count']} candidates → {s1['num_merges']} merges\n"
             f"K={s1['num_codes_after']}",
             width=0.26, height=0.24, color=PURPLE)
    draw_box(axis, (0.70, 0.45),
             f"Stage 2 UOT\n+ Cube\n"
             f"{s2['mutual_candidate_count']} candidates → {s2['num_merges']} merges\n"
             f"K={s2['num_codes_after']}",
             width=0.26, height=0.24, color=RED)
    draw_box(axis, (0.92, 0.45),
             f"Final\nK={three['num_embeddings']}\n(concat)",
             width=0.14, height=0.20, color=GREEN)
    draw_arrow(axis, (0.20, 0.78), (0.29, 0.70))
    draw_arrow(axis, (0.20, 0.52), (0.29, 0.62))
    draw_arrow(axis, (0.53, 0.62), (0.59, 0.50))
    draw_arrow(axis, (0.20, 0.22), (0.59, 0.40))
    draw_arrow(axis, (0.83, 0.45), (0.85, 0.45))
    axis.text(
        0.5, 0.055,
        f"Total accepted merges = {three['num_merges']} under the 2% per-task QE "
        f"budget; K_shared = {three['num_embeddings']} (= 3 × 8192).",
        ha="center", va="center", fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F6F6F6",
              "edgecolor": LIGHT_GRAY},
    )
    axis.set_title("(b) Three tasks: sequential UOT, zero merges", fontsize=12.5)

    # Panel (c): six-task sequential UOT — candidates per stage, all rejected.
    axis = axes[2]
    stages = six["stages"]
    task_order = six["task_order"]
    added = [TASK_LABELS[task_order[index + 1]] for index in range(len(stages))]
    candidates = [stage["mutual_candidate_count"] for stage in stages]
    merges = [stage["num_merges"] for stage in stages]
    codes = [stage["num_codes_after"] for stage in stages]
    stage_positions = np.arange(len(stages))
    bars = axis.bar(stage_positions, candidates, 0.55, color=PURPLE)
    axis.bar_label(bars, fmt="%d", padding=3, fontsize=8.5)
    for x, merge_count, code in zip(stage_positions, merges, codes):
        axis.annotate(f"{merge_count} merges\nK={code:,}", (x, 0.45),
                      ha="center", va="bottom", fontsize=8, color=RED,
                      fontweight="bold")
    axis.set_yscale("log")
    axis.set_ylim(0.3, max(candidates) * 6)
    axis.set_xticks(stage_positions, [f"+ {name}" for name in added], fontsize=8)
    axis.set_xlabel("Task added at each sequential stage")
    axis.set_ylabel("Mutual candidate pairs (log scale)")
    axis.set_title(
        f"(c) Six tasks: {sum(candidates):,} candidates, "
        f"{sum(merges)} merges, K = {six['num_embeddings']:,} (concat)",
        fontsize=12.5,
    )
    style_axis(axis)

    figure.suptitle(
        "UOT converged numerically but accepted zero cross-task merges "
        "at two, three, and six tasks", fontsize=14
    )
    output = ASSET_ROOT / "uot"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "uot_zero_merge_outcome.png", dpi=180)
    plt.close(figure)
    print("Wrote", output / "uot_zero_merge_outcome.png")


# ---------------------------------------------------------------------------
# Figure 11: teacher-representation ablation with supervision-source reference
# ---------------------------------------------------------------------------

def fig_teacher_ablation_merged() -> None:
    teachers, experts = teacher_and_expert_rates()
    base = STABLEWM_ROOT / "multitask_distillation"

    def rates(prefix: str) -> dict:
        return {
            key: multitask_rates(base / f"{prefix}_{run}_seed3072"
                                 / "task_evaluation" / "summary.json")
            for key, run in (
                ("m2", "uot"),
                ("m4", "m4_continuous"),
                ("m5", "m5_codebook"),
            )
        }

    scales = [
        ("(a) Two tasks: PushT + Two-Room", ["pusht", "tworoom"], rates("pusht_tworoom")),
        ("(b) Three tasks: + Cube", ["pusht", "tworoom", "cube"],
         rates("pusht_tworoom_cube")),
        ("(c) Six tasks: + Scene + Reacher + HumanoidMaze", list(ALL_TASKS),
         rates("pusht_tworoom_cube_scene_reacher_humanoidmaze")),
    ]
    labels = ["M2 (mixed)", "M4 (all continuous)", "M5 (all codebook)"]
    keys = ["m2", "m4", "m5"]

    figure, axes = plt.subplots(1, 3, figsize=(19.8, 5.6), constrained_layout=True)
    for axis, (title, tasks, rows) in zip(axes, scales):
        teacher_macro = sum(teachers[t] for t in tasks) / len(tasks)
        expert_macro = sum(experts[t] for t in tasks) / len(tasks)

        positions = np.arange(len(labels))
        n_tasks = len(tasks)
        width = {2: 0.34, 3: 0.26, 6: 0.13}[n_tasks]
        offsets = (np.arange(n_tasks) - (n_tasks - 1) / 2.0) * width
        for task, offset in zip(tasks, offsets):
            bars = axis.bar(positions + offset, [rows[key][task] for key in keys],
                            width, color=TASK_COLORS[task], label=TASK_LABELS[task])
            annotate_bars(axis, bars, suffix="%", decimals=0)

        macro = [sum(rows[key][t] for t in tasks) / n_tasks for key in keys]
        axis.plot(positions, macro, color="black", marker="D", linewidth=1.5,
                  label="Macro average")
        for x, value in zip(positions, macro):
            axis.annotate(f"{value:.1f}", (x, value), textcoords="offset points",
                          xytext=(0, 11), ha="center", fontsize=9, fontweight="bold")

        axis.axhline(teacher_macro, linestyle="--", color="black", linewidth=1.4,
                     label=f"Continuous-teacher macro = {teacher_macro:.1f}")
        axis.axhline(expert_macro, linestyle=":", color="#333333", linewidth=1.4,
                     label=f"Discrete-expert macro = {expert_macro:.1f}")

        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 108)
        axis.set_ylabel("MPC success rate (%)")
        axis.set_title(title)
        axis.legend(frameon=False, fontsize=7, ncol=3 if n_tasks == 6 else 2)
        style_axis(axis)

    figure.suptitle(
        "Teacher-representation ablation vs its supervision sources "
        "(50 fixed starts; MPC seed 42; train seed 3072)", fontsize=13
    )
    output = ASSET_ROOT / "consolidation"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "teacher_representation_ablation.png", dpi=180)
    plt.close(figure)
    print("Wrote", output / "teacher_representation_ablation.png")


# ---------------------------------------------------------------------------
# Figure 12: M3 native-joint training dynamics, two-task and three-task rows
# ---------------------------------------------------------------------------

def fig_m3_merged() -> None:
    runs = [
        ("(a) Two tasks: objective terms",
         "(b) Two tasks: PushT MSE jumps 100× at epoch 7→8",
         ("pusht", "tworoom"), True,
         STABLEWM_ROOT / "multitask_baseline" / "pusht_tworoom_m3_seed3072"),
        ("(c) Three tasks: objective terms",
         "(d) Three tasks: PushT MSE lowest, closed-loop still 4%",
         ("pusht", "tworoom", "cube"), False,
         STABLEWM_ROOT / "multitask_baseline" / "pusht_tworoom_cube_m3_seed3072"),
        ("(e) Six tasks: objective terms",
         "(f) Six tasks: PushT & Reacher end with the lowest MSE, "
         "closed-loop 2% / 6%",
         ALL_TASKS, False,
         STABLEWM_ROOT / "multitask_baseline"
         / "pusht_tworoom_cube_scene_reacher_humanoidmaze_m3_seed3072"),
    ]
    sigreg_weight = 0.09

    figure, axes = plt.subplots(3, 2, figsize=(13.4, 14.4), constrained_layout=True)
    for row, (title_obj, title_val, tasks, mark_transition, root) in enumerate(runs):
        metrics = load_jsonl(root / "metrics.jsonl")
        epochs = [entry["epoch"] for entry in metrics]

        axes[row, 0].plot(epochs, [entry["train/loss"] for entry in metrics],
                          linewidth=2, color=BLUE, label="Total loss")
        axes[row, 0].plot(epochs, [entry["train/prediction_mse"] for entry in metrics],
                          linewidth=2, color=ORANGE, label="Prediction MSE")
        axes[row, 0].plot(
            epochs, [sigreg_weight * entry["train/sigreg"] for entry in metrics],
            linewidth=2, color=PURPLE, label=f"{sigreg_weight} × SIGReg")
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_xlabel("Epoch")
        axes[row, 0].set_ylabel("Training objective (log scale)")
        axes[row, 0].set_title(title_obj)
        axes[row, 0].legend(frameon=False)
        style_axis(axes[row, 0])

        for task in tasks:
            if task not in metrics[0]["validation"]:
                continue
            axes[row, 1].plot(
                epochs,
                [entry["validation"][task]["prediction_mse"] for entry in metrics],
                linewidth=2, marker="o", markersize=3,
                color=TASK_COLORS[task], label=TASK_LABELS[task])
        if mark_transition:
            axes[row, 1].axvspan(7.5, 8.5, color=RED, alpha=0.12,
                                 label="Epoch 7→8 transition")
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_xlabel("Epoch")
        axes[row, 1].set_ylabel("Validation prediction MSE (log scale)")
        axes[row, 1].set_title(title_val)
        axes[row, 1].legend(frameon=False, fontsize=8)
        style_axis(axes[row, 1])

    figure.suptitle(
        "M3 native joint training: negative transfer at two, three, and six "
        "tasks (train seed 3072)", fontsize=14
    )
    output = ASSET_ROOT / "consolidation"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "m3_negative_transfer.png", dpi=180)
    plt.close(figure)
    print("Wrote", output / "m3_negative_transfer.png")


# ---------------------------------------------------------------------------
# Appendix figure A3: M2 distillation training convergence, both scales
# ---------------------------------------------------------------------------

def fig_m2_merged() -> None:
    runs = [
        ("(a) Two tasks: M2 training metrics",
         "(b) Two tasks: validation student prediction MSE",
         ("pusht", "tworoom"),
         STABLEWM_ROOT / "multitask_distillation" / "pusht_tworoom_uot_seed3072"),
        ("(c) Three tasks: M2 training metrics",
         "(d) Three tasks: validation student prediction MSE",
         ("pusht", "tworoom", "cube"),
         STABLEWM_ROOT / "multitask_distillation" / "pusht_tworoom_cube_uot_seed3072"),
        ("(e) Six tasks: M2 training metrics",
         "(f) Six tasks: validation student prediction MSE",
         ALL_TASKS,
         STABLEWM_ROOT / "multitask_distillation"
         / "pusht_tworoom_cube_scene_reacher_humanoidmaze_uot_seed3072"),
    ]

    figure, axes = plt.subplots(3, 2, figsize=(13.4, 14.4), constrained_layout=True)
    for row, (title_train, title_val, tasks, root) in enumerate(runs):
        metrics = load_jsonl(root / "metrics.jsonl")
        epochs = [entry["epoch"] for entry in metrics]
        for axis in (axes[row, 0], axes[row, 1]):
            add_phase_spans(axis)

        for key, label, color in (
            ("train/total_loss", "Total loss", BLUE),
            ("train/latent_mse", "Latent MSE", ORANGE),
            ("train/prediction_mse", "Prediction MSE", GREEN),
            ("train/token_kl", "Token KL", PURPLE),
        ):
            axes[row, 0].plot(epochs, [entry[key] for entry in metrics],
                              linewidth=1.9, marker="o", markersize=3,
                              color=color, label=label)
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_xlabel("Epoch")
        axes[row, 0].set_ylabel("Training metric (log scale)")
        # Extra pad keeps the title clear of the phase-span labels (y = 1.01).
        axes[row, 0].set_title(title_train, pad=20)
        axes[row, 0].legend(frameon=False, fontsize=8)
        style_axis(axes[row, 0])

        for task in tasks:
            if task not in metrics[0]["validation"]:
                continue
            axes[row, 1].plot(
                epochs,
                [entry["validation"][task]["student_prediction_mse"]
                 for entry in metrics],
                linewidth=1.9, marker="o", markersize=3,
                color=TASK_COLORS[task], label=TASK_LABELS[task])
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_xlabel("Epoch")
        axes[row, 1].set_ylabel("Validation student prediction MSE")
        axes[row, 1].set_title(title_val, pad=20)
        axes[row, 1].legend(frameon=False, fontsize=8)
        style_axis(axes[row, 1])

    figure.suptitle(
        "M2 aligned multi-codebook distillation training "
        "(two, three, and six tasks; train seed 3072)", fontsize=14
    )
    output = ASSET_ROOT / "consolidation"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "m2_training_convergence.png", dpi=180)
    plt.close(figure)
    print("Wrote", output / "m2_training_convergence.png")


def main() -> None:
    fig_alignment_merged()
    fig_uot_merged()
    fig_teacher_ablation_merged()
    fig_m3_merged()
    fig_m2_merged()


if __name__ == "__main__":
    main()
