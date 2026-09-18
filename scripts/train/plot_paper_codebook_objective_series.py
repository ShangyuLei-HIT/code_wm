"""Codebook-quality figure for the paper: fully-discrete vs mixed objectives.

Redraws ``codebook_quality_rigid/heldout_success_and_paired_ci.png`` without
the rigid-transform condition (out of scope for the paper) and adds the
mixed-objective comparison series plus the pure-codebook prediction-error
panel, which together carry the section's conclusion: under fully-discrete
teacher targets, codebook quality directly determines closed-loop performance,
while continuous targets buffer quantization error.

All numbers are read from the real evaluation / metrics files under .stablewm;
nothing is hard-coded. Figure text is English (no CJK font available).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STABLEWM_ROOT = PROJECT_ROOT / ".stablewm"
ASSET_ROOT = PROJECT_ROOT / "docs" / "assets" / "paper_figures" / "codebook_quality"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
GRAY = "#7F7F7F"
LIGHT_GRAY = "#D9D9D9"

# Fully-discrete series (second round): PushT held-out 200-task evaluations.
FULLY_DISCRETE_RUNS = {
    512: STABLEWM_ROOT / "experiments" / "fully_discrete_codebook_series_v1"
         / "runs" / "pusht" / "k512_fully_discrete",
    2048: STABLEWM_ROOT / "experiments" / "fully_discrete_codebook_series_v1"
          / "runs" / "pusht" / "k2048_fully_discrete",
    8192: STABLEWM_ROOT / "joint_distillation"
          / "lewm_pusht_k8192_fully_discrete_seed3072",
}

# Mixed-objective series (first round), original (non-rigid) conditions only.
MIXED_RUNS = {
    512: STABLEWM_ROOT / "experiments" / "codebook_quality_rigid_v1"
         / "evaluations" / "k512_original" / "seed3072",
    2048: STABLEWM_ROOT / "experiments" / "codebook_quality_rigid_v1"
          / "evaluations" / "k2048_original" / "seed3072",
    8192: STABLEWM_ROOT / "experiments" / "codebook_quality_rigid_v1"
          / "evaluations" / "k8192_original" / "seed3072",
}


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def heldout_success(path: Path) -> tuple[float, int, int]:
    summary = load_json(path / "summary.json")
    block = summary["heldout_test"]
    successes = int(block["successes"])
    episodes = int(block.get("episodes", len(block["episode_successes"])))
    return float(block["success_rate"]), successes, episodes


def final_pred_teacher_mse(run_dir: Path) -> float:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text()
            .splitlines() if line.strip()]
    return float(rows[-1]["validate/pred_teacher_mse"])


def wilson_interval(successes: int, episodes: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, in percent."""
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / episodes
                         + z * z / (4.0 * episodes * episodes)) / denominator
    return 100.0 * (center - half), 100.0 * (center + half)


def style_axis(axis, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color=LIGHT_GRAY, linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main() -> None:
    sizes = [512, 2048, 8192]

    discrete = [heldout_success(FULLY_DISCRETE_RUNS[k] / "task_evaluation_heldout")
                for k in sizes]
    mixed = [heldout_success(MIXED_RUNS[k]) for k in sizes]

    discrete_ci = [wilson_interval(s, n) for _, s, n in discrete]
    mixed_ci = [wilson_interval(s, n) for _, s, n in mixed]
    discrete_err = np.array([
        [rate - lo for (rate, _, _), (lo, _) in zip(discrete, discrete_ci)],
        [hi - rate for (rate, _, _), (_, hi) in zip(discrete, discrete_ci)],
    ])
    mixed_err = np.array([
        [rate - lo for (rate, _, _), (lo, _) in zip(mixed, mixed_ci)],
        [hi - rate for (rate, _, _), (_, hi) in zip(mixed, mixed_ci)],
    ])

    pred_mse = [final_pred_teacher_mse(FULLY_DISCRETE_RUNS[k]) for k in sizes]

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)

    positions = np.arange(len(sizes))
    width = 0.34
    bars_d = axes[0].bar(positions - width / 2, [r for r, _, _ in discrete], width,
                         yerr=discrete_err, capsize=4, color=BLUE,
                         error_kw={"elinewidth": 1.2},
                         label="Fully-discrete objective")
    bars_m = axes[0].bar(positions + width / 2, [r for r, _, _ in mixed], width,
                         yerr=mixed_err, capsize=4, color=ORANGE,
                         error_kw={"elinewidth": 1.2},
                         label="Mixed objective")
    axes[0].bar_label(bars_d, fmt="%.1f", padding=2, fontsize=9)
    axes[0].bar_label(bars_m, fmt="%.1f", padding=2, fontsize=9)
    axes[0].set_xticks(positions, [f"K={k}" for k in sizes])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("PushT held-out success rate (%)")
    axes[0].set_title("Closed-loop success: monotone in K only when the objective is fully discrete")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    axes[1].plot(positions, pred_mse, marker="o", markersize=7, linewidth=2,
                 color=GREEN)
    for position, value in zip(positions, pred_mse):
        axes[1].annotate(f"{value:.4f}", (position, value),
                         textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=9)
    axes[1].set_xticks(positions, [f"K={k}" for k in sizes])
    axes[1].set_ylim(0.05, 0.09)
    axes[1].set_ylabel("One-step prediction MSE (pure-codebook trajectories)")
    axes[1].set_title("Same ordering as closed-loop success (fully-discrete runs)")
    style_axis(axes[1])

    episodes = discrete[0][2]
    figure.suptitle(
        f"PushT codebook series ({episodes} held-out episodes, Wilson 95% CI): "
        "codebook quality determines closed-loop performance\nwhen every teacher "
        "target is a code vector; continuous targets in the mixed objective buffer "
        "the quantization error", fontsize=13)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASSET_ROOT / "pusht_objective_comparison.png", dpi=180)
    plt.close(figure)
    print("Wrote", ASSET_ROOT / "pusht_objective_comparison.png")


if __name__ == "__main__":
    main()
