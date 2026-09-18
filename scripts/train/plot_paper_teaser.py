"""Teaser figure (Figure 1) for the paper overview — print-size version.

Left-to-right story required by the outline (§1.3), drawn without internal
model codenames (readers have not met them on page 1):

  three single-task experts (real rollout frames) -> Extract / Align /
  Consolidate -> one shared world model with per-task success bars vs its
  teachers.

Expert cards embed REAL evaluation rollout frames (``task_*.png``,
extracted from eval videos — see ``docs/assets/paper_figures/teaser/``).

Canvas is the FINAL print size (7.2 in wide -> ``\\linewidth`` in LaTeX
scales it to 5.5 in, a 0.764 factor), so every drawn font is specified at
its on-canvas point size and no drawn text goes below 8 pt.  Strings are
kept short; dense evidence panels (PCA clusters, validation-MSE curves)
live in the appendix assets instead of this figure.  Every number equals
the two/three-task main results — the per-task bars read the three-task
continuous instantiation (94/86/64) and the official teachers (90/86/68)
from the real evaluation files (same sources as
main_success_matrix.png).  Text is English (no CJK font).
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
ASSET_ROOT = PROJECT_ROOT / "docs" / "assets" / "paper_figures" / "teaser"
# Shared-model per-task bars: the three-task continuous instantiation
# (PushT 94 vs teacher 90 is the strongest student/teacher pair).
SHARED_MODEL_SUMMARY = (PROJECT_ROOT / ".stablewm" / "multitask_distillation"
                        / "pusht_tworoom_cube_m4_continuous_seed3072"
                        / "task_evaluation" / "summary.json")
TEACHER_EVAL_FILES = {
    "pusht": (PROJECT_ROOT / ".stablewm" / "checkpoints"
              / "official_lewm_pusht_compat"
              / "pusht_results_official_seeded_50.txt"),
    "tworoom": (PROJECT_ROOT / ".stablewm" / "checkpoints"
                / "official_lewm_tworooms_compat" / "task_evaluation"
                / "tworoom_results_official_seed42_50.json"),
    "cube": (PROJECT_ROOT / ".stablewm" / "checkpoints"
             / "official_lewm_cube_compat" / "task_evaluation"
             / "cube_results_official_seed42_50.json"),
}

BLUE = "#4C78A8"       # PushT
ORANGE = "#F58518"     # Two-Room
TEAL = "#2A9D8F"       # Cube
PURPLE = "#7A5195"     # shared model
GRAY = "#7F7F7F"
LIGHT_GRAY = "#D9D9D9"
DARK = "#333333"

# Final print canvas: 7.2 in wide, scaled to \\linewidth (5.5 in) in LaTeX.
FIG_W, FIG_H = 7.2, 2.4
# axes-fraction <-> inch helpers (main axis spans the whole figure)
def fx(inches: float) -> float:
    return inches / FIG_W

def fy(inches: float) -> float:
    return inches / FIG_H


def rounded_box(axis, cx, cy, width, height, *, edge, face_alpha=0.10,
                lw=1.2, rounding=0.010, zorder=2):
    axis.add_patch(FancyBboxPatch(
        (cx - width / 2, cy - height / 2), width, height,
        boxstyle=f"round,pad=0.004,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=edge, alpha=face_alpha,
        zorder=zorder))


def arrow(axis, start, end, *, color=GRAY, lw=1.3, rad=0.0, zorder=1,
          scale=9):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=zorder))


def place_image(figure, path, cx, cy, width_in, height_in, *, edge):
    """Insert a real image centered at (cx, cy) in figure fractions."""
    axis = figure.add_axes(
        [cx - fx(width_in) / 2, cy - fy(height_in) / 2,
         fx(width_in), fy(height_in)])
    axis.imshow(plt.imread(path))
    axis.set_xticks([])
    axis.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_color(edge)
        axis.spines[side].set_linewidth(1.0)
    axis.tick_params(length=0, pad=0)
    return axis


def draw_result_visual(figure) -> None:
    """Per-task success: shared model vs its single-task teachers.

    Reads the three-task continuous instantiation and the official teacher
    evaluations from the real .stablewm files — the same sources as
    main_success_matrix.png (no composite numbers)."""
    axis = figure.add_axes([fx(5.46), fy(0.16), fx(1.68), fy(1.04)])
    tasks = ["pusht", "tworoom", "cube"]
    groups = ["PushT", "Two-Room", "Cube"]
    summary = json.loads(SHARED_MODEL_SUMMARY.read_text())
    shared = {row["task"]: row["metrics"]["success_rate"]
              for row in summary["tasks"]}
    ours = [shared[task] for task in tasks]
    teachers = []
    for task in tasks:
        path = TEACHER_EVAL_FILES[task]
        if path.suffix == ".txt":
            match = re.search(r"success_rate['\"]?:\s*([0-9.]+)",
                              path.read_text())
            teachers.append(float(match.group(1)))
        else:
            teachers.append(
                json.loads(path.read_text())["metrics"]["success_rate"])
    positions = np.arange(3)
    bars_a = axis.bar(positions - 0.19, ours, 0.34, color=PURPLE,
                      label="ours")
    bars_b = axis.bar(positions + 0.19, teachers, 0.34, color="0.62",
                      hatch="///", edgecolor="white", label="teachers")
    axis.bar_label(bars_a, fmt="%.0f", padding=1.5, fontsize=8)
    axis.bar_label(bars_b, fmt="%.0f", padding=1.5, fontsize=8)
    axis.set_xticks(positions, groups, fontsize=8)
    axis.set_ylim(0, 130)
    axis.set_yticks([])
    style_spines(axis)
    axis.legend(frameon=False, fontsize=8, loc="upper center", ncol=2,
                handlelength=1.1, columnspacing=0.9, handletextpad=0.4,
                borderaxespad=0.1, bbox_to_anchor=(0.5, 1.16))
    axis.set_title("success per task", fontsize=8.5, color=DARK, pad=11)


def style_spines(axis, *, keep=("bottom", "left")) -> None:
    axis.set_axisbelow(True)
    axis.tick_params(length=1.5, pad=1.2, labelsize=8)
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(side in keep)
        axis.spines[side].set_color("#999999")
        axis.spines[side].set_linewidth(0.7)


def main() -> None:
    figure = plt.figure(figsize=(FIG_W, FIG_H))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    # ------------------------------------------------------------- left zone
    axis.text(fx(0.86), fy(2.27), "Specialist experts", ha="center",
              va="center", fontsize=10.5, fontweight="bold", color=DARK)
    experts = [
        ("PushT expert", "task_pusht.png", BLUE),
        ("Two-Room expert", "task_tworoom.png", ORANGE),
        ("Cube expert", "task_cube.png", TEAL),
    ]
    for index, (name, filename, color) in enumerate(experts):
        cy = 1.76 - index * 0.58
        rounded_box(axis, fx(0.86), fy(cy), fx(1.60), fy(0.48), edge=color,
                    face_alpha=0.10, lw=1.3)
        # real rollout frame on the left of the card, texts on the right
        place_image(figure, ASSET_ROOT / filename, fx(0.335), fy(cy),
                    0.36, 0.36, edge=color)
        axis.text(fx(0.56), fy(cy + 0.09), name, ha="left", va="center",
                  fontsize=8.5, fontweight="bold", color=DARK, zorder=5)
        axis.text(fx(0.56), fy(cy - 0.09), "frozen weights", ha="left",
                  va="center", fontsize=8, color=GRAY, zorder=5)

    # ---------------------------------------------------------- middle zone
    axis.text(fx(3.57), fy(2.27), "Consolidation (offline)", ha="center",
              va="center", fontsize=10.5, fontweight="bold", color=DARK)
    axis.text(fx(3.57), fy(2.07),
              "represent first, align second, distill last",
              ha="center", va="center", fontsize=8, color=GRAY)

    steps = [
        (2.47, "EXTRACT",
         "freeze experts,\nfit codebooks\n(8192 codes),\ncache latents\n+ soft targets"),
        (3.57, "ALIGN",
         "Similarity\nProcrustes into\none frame\n(assignments\nkept exactly)"),
        (4.67, "CONSOLIDATE",
         "phased teacher\n$\\to$ student mix;\nno teacher or\ncodebook at\ndeployment"),
    ]
    for cx, title, body in steps:
        rounded_box(axis, fx(cx), fy(1.28), fx(1.00), fy(1.30),
                    edge="#555555", face_alpha=0.06, lw=1.2)
        axis.text(fx(cx), fy(1.76), title, ha="center", va="center",
                  fontsize=9, fontweight="bold", color=DARK, zorder=5)
        axis.text(fx(cx), fy(1.21), body, ha="center", va="center",
                  fontsize=8, color=DARK, zorder=5, linespacing=1.35)
    for left, right in ((2.99, 3.05), (4.09, 4.15)):
        arrow(axis, (fx(left), fy(1.28)), (fx(right), fy(1.28)), lw=1.3,
              scale=8)
    for index, (_, _, color) in enumerate(experts):  # fan-in to step 1
        arrow(axis, (fx(1.68), fy(1.76 - index * 0.58)), (fx(1.94), fy(1.30)),
              color=color, lw=1.2, rad=0.10, scale=8)
    arrow(axis, (fx(5.19), fy(1.28)), (fx(5.42), fy(1.56)), color=PURPLE,
          lw=1.5, rad=-0.10, scale=10)

    # ----------------------------------------------------------- right zone
    axis.text(fx(6.30), fy(2.27), "Shared world model", ha="center",
              va="center", fontsize=10.5, fontweight="bold", color=PURPLE)
    axis.text(fx(6.30), fy(2.07),
              "teachers are\ntraining-time only", ha="center", va="center",
              fontsize=8, color=GRAY, linespacing=1.3)
    draw_result_visual(figure)

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASSET_ROOT / "teaser_overview.png", dpi=300,
                   bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    print("Wrote", ASSET_ROOT / "teaser_overview.png")


if __name__ == "__main__":
    main()
