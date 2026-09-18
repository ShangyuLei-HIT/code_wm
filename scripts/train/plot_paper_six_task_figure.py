"""Paper figure: six-task per-task success for the five shared models.

Grouped bars (M3/M0/M2/M4/M5) with per-task reference lines for the
original continuous specialists and the single-task hybrid students.
Drawn at the FINAL print size (5.5 x 2.3 in = ``\\linewidth``), so every
font is specified at its printed point size (floor 8 pt).

All numbers are read from the real evaluation JSON / TXT files under
.stablewm; nothing is hard-coded.  Figure text is English (no CJK font).
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
RED = "#E45756"
PURPLE = "#7A5195"
TEAL = "#2A9D8F"
GRAY = "#7F7F7F"
LIGHT_GRAY = "#D9D9D9"

TASK_LABELS = {
    "pusht": "PushT",
    "tworoom": "Two-Room",
    "cube": "Cube",
    "scene": "Scene",
    "reacher": "Reacher",
    "humanoidmaze": "HumanoidMaze",
}
ALL_TASKS = ("pusht", "tworoom", "cube", "scene", "reacher", "humanoidmaze")

# Paper-facing labels; M-codes are defined in the protocol section.
MODELS = [
    ("m3", "M3 native", RED),
    ("m0", "M0 unaligned", GRAY),
    ("m2", "M2 hybrid", PURPLE),
    ("m4", "M4 continuous", "#1f4e79"),
    ("m5", "M5 discrete", TEAL),
]

matplotlib.rcParams.update({"font.size": 8})


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
    return {row["task"]: row["metrics"]["success_rate"]
            for row in payload["tasks"]}


def reference_rates() -> tuple[dict, dict]:
    """Specialists (continuous teachers) and single-task hybrid students."""
    teachers = {
        "pusht": extract_success_rate_txt(
            STABLEWM_ROOT / "checkpoints" / "official_lewm_pusht_compat"
            / "pusht_results_official_seeded_50.txt"),
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
            STABLEWM_ROOT / "checkpoints"
            / "lewm_humanoidmaze_scratch_seed3072_compat"
            / "task_evaluation" / "humanoidmaze_results_scratch_seed42_50.json"
        )["metrics"]["success_rate"],
    }
    student_dirs = {
        "pusht": "lewm_pusht_k8192_seed3072",
        "tworoom": "lewm_tworooms_k8192_seed3072",
        "cube": "lewm_cube_k8192_seed3072",
        "scene": "lewm_scene_k8192_seed3072",
        "reacher": "lewm_reacher_k8192_seed3072",
        "humanoidmaze": "lewm_humanoidmaze_k8192_seed3072",
    }
    students = {
        task: load_json(
            STABLEWM_ROOT / "joint_distillation" / directory
            / "task_evaluation" / "summary.json")["best_stage"]["success_rate"]
        for task, directory in student_dirs.items()
    }
    return teachers, students


def shared_rates() -> dict:
    prefix = "pusht_tworoom_cube_scene_reacher_humanoidmaze"
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


def main() -> None:
    teachers, students = reference_rates()
    rates = shared_rates()

    figure, axis = plt.subplots(figsize=(5.5, 2.3),
                                constrained_layout=True)
    task_positions = np.arange(len(ALL_TASKS))
    n_models = len(MODELS)
    width = 0.13
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * (width + 0.012)

    for (key, label, color), offset in zip(MODELS, offsets):
        values = [rates[key][task] for task in ALL_TASKS]
        axis.bar(task_positions + offset, values, width,
                 color=color, label=label)

    # Per-task reference lines: specialists and single-task students.
    group_half = (n_models / 2.0) * (width + 0.012) + width / 2.0 + 0.02
    for position, task in zip(task_positions, ALL_TASKS):
        axis.hlines(teachers[task], position - group_half,
                    position + group_half, linestyles="--", colors="black",
                    linewidth=1.1, zorder=3)
        axis.hlines(students[task], position - group_half,
                    position + group_half, linestyles=":", colors="#333333",
                    linewidth=1.1, zorder=3)

    axis.set_xticks(task_positions,
                    [TASK_LABELS[task] for task in ALL_TASKS], fontsize=8)
    axis.tick_params(labelsize=8)
    axis.set_ylim(0, 132)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_ylabel("success (%)", fontsize=8)
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    handles, labels = axis.get_legend_handles_labels()
    teacher_line = plt.Line2D([], [], color="black", linestyle="--",
                              linewidth=1.1, label="specialists")
    student_line = plt.Line2D([], [], color="#333333", linestyle=":",
                              linewidth=1.1, label="1-task students")
    figure.legend(handles + [teacher_line, student_line],
                  labels + [teacher_line.get_label(),
                            student_line.get_label()],
                  loc="outside upper center", ncol=4, frameon=False,
                  fontsize=8, handlelength=1.4, columnspacing=1.0,
                  handletextpad=0.4)

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASSET_ROOT / "six_task_results.png", dpi=300)
    figure.savefig(ASSET_ROOT / "six_task_results.pdf")
    plt.close(figure)
    print("Wrote", ASSET_ROOT / "six_task_results.png")


if __name__ == "__main__":
    main()
