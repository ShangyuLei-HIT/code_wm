"""Main-results figure for the paper: consolidation vs native joint training.

Redraws the per-experiment-line ``mpc_success_rate_matrix`` figures (two-task,
three-task, six-task) as one paper figure with per-task teacher /
discrete-expert reference lines, which the existing figures lack and which the
paper text relies on ("consolidation can exceed its specialists", "zero
consolidation loss").

All numbers are read from the real evaluation JSON / TXT files under .stablewm;
nothing is hard-coded. Figure text is English (no CJK font available).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STABLEWM_ROOT = PROJECT_ROOT / ".stablewm"
ASSET_ROOT = PROJECT_ROOT / "docs" / "assets" / "paper_figures" / "consolidation"

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

# Paper-facing model names (internal codenames in parentheses for traceability).
MODELS = [
    ("m3", "Native joint (M3)", RED),
    ("m0", "Unaligned concat (M0)", GRAY),
    ("m2", "Hybrid (M2)", PURPLE),
    ("m4", "Continuous (M4)", "#1f4e79"),
    ("m5", "Discrete (M5)", TEAL),
]


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


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


def model_rates(prefix: str) -> dict:
    distill = STABLEWM_ROOT / "multitask_distillation"
    paths = {
        "m3": STABLEWM_ROOT / "multitask_baseline" / f"{prefix}_m3_seed3072"
              / "task_evaluation" / "summary.json",
        "m0": distill / f"{prefix}_m0_unaligned_concat_seed3072"
              / "task_evaluation" / "summary.json",
        "m2": distill / f"{prefix}_uot_seed3072"
              / "task_evaluation" / "summary.json",
        "m4": distill / f"{prefix}_m4_continuous_seed3072"
              / "task_evaluation" / "summary.json",
        "m5": distill / f"{prefix}_m5_codebook_seed3072"
              / "task_evaluation" / "summary.json",
    }
    return {key: multitask_rates(path) for key, path in paths.items()}


def draw_panel(axis, tasks, rates, teachers, experts, *, title: str) -> None:
    task_positions = np.arange(len(tasks))
    n_models = len(MODELS)
    width = 0.15
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * (width + 0.005)

    for (key, label, color), offset in zip(MODELS, offsets):
        values = [rates[key][task] for task in tasks]
        bars = axis.bar(task_positions + offset, values, width,
                        color=color, label=label)
        axis.bar_label(bars, fmt="%.0f", padding=2, fontsize=7.5)

    # Per-task reference lines: continuous teacher and discrete expert.
    group_half = (n_models / 2.0) * (width + 0.005) + width / 2.0 + 0.02
    for position, task in zip(task_positions, tasks):
        axis.hlines(teachers[task], position - group_half, position + group_half,
                    linestyles="--", colors="black", linewidth=1.4, zorder=3)
        axis.hlines(experts[task], position - group_half, position + group_half,
                    linestyles=":", colors="#333333", linewidth=1.4, zorder=3)

    axis.set_xticks(task_positions, [TASK_LABELS[task] for task in tasks])
    axis.set_ylim(0, 108)
    axis.set_ylabel("MPC success rate (%)")
    axis.set_title(title)
    style_axis(axis)


def main() -> None:
    teachers, experts = teacher_and_expert_rates()
    two = model_rates("pusht_tworoom")
    three = model_rates("pusht_tworoom_cube")
    six = model_rates("pusht_tworoom_cube_scene_reacher_humanoidmaze")

    figure = plt.figure(figsize=(13.8, 10.6), constrained_layout=True)
    axes = figure.subplot_mosaic([["two", "three"], ["six", "six"]])
    draw_panel(axes["two"], ["pusht", "tworoom"], two, teachers, experts,
               title="(a) Two tasks: PushT + Two-Room")
    draw_panel(axes["three"], ["pusht", "tworoom", "cube"], three, teachers, experts,
               title="(b) Three tasks: + Cube")
    draw_panel(axes["six"], list(ALL_TASKS), six, teachers, experts,
               title="(c) Six tasks: + Scene + Reacher + HumanoidMaze "
                     "(heterogeneous action widths padded to 105)")

    handles, labels = axes["two"].get_legend_handles_labels()
    teacher_line = plt.Line2D([], [], color="black", linestyle="--", linewidth=1.4,
                              label="Single-task continuous teacher")
    expert_line = plt.Line2D([], [], color="#333333", linestyle=":", linewidth=1.4,
                             label="Single-task discrete expert")
    # "outside upper center" makes constrained_layout reserve room for the legend;
    # the previous bbox_inches="tight" + bbox_extra_artists=() save silently
    # dropped a legend anchored above the canvas.
    figure.legend(handles + [teacher_line, expert_line],
                  labels + [teacher_line.get_label(), expert_line.get_label()],
                  loc="outside upper center", ncol=4, frameon=False, fontsize=9)
    figure.suptitle("Shared multi-task models vs teachers / experts "
                    "(50 fixed starts per task; MPC seed 42; train seed 3072)",
                    fontsize=13)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASSET_ROOT / "main_success_matrix.png", dpi=180)
    plt.close(figure)
    print("Wrote", ASSET_ROOT / "main_success_matrix.png")


if __name__ == "__main__":
    main()
