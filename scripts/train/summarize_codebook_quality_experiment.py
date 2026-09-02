"""Aggregate the two-seed experiment and write the final JSON/Markdown report."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


COMPARISONS = (
    ('k512_original', 'k8192_original'),
    ('k2048_original', 'k8192_original'),
    ('k8192_rigid', 'k8192_original'),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/codebook_quality_rigid_experiment.yaml',
    )
    parser.add_argument(
        '--report',
        default='docs/codebook_quality_and_rigid_transform_experiment_report.md',
    )
    parser.add_argument('--bootstrap-samples', type=int, default=20000)
    parser.add_argument(
        '--seeds',
        default=None,
        help='Optional comma-separated completed training seeds.',
    )
    return parser.parse_args()


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def evaluation_summary(matrix, condition: str, seed: int) -> dict:
    path = (
        Path(matrix.experiment.root).expanduser().resolve()
        / 'evaluations'
        / condition
        / f'seed{seed}'
        / 'summary.json'
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get('status') != 'complete':
        raise RuntimeError(f'incomplete evaluation summary: {path}')
    payload['_path'] = str(path)
    return payload


def exact_mcnemar_p(left: np.ndarray, right: np.ndarray) -> dict:
    left_only = int(np.logical_and(left, np.logical_not(right)).sum())
    right_only = int(np.logical_and(np.logical_not(left), right).sum())
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(left_only, right_only)
        probability = sum(
            math.comb(discordant, value)
            for value in range(tail + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * probability)
    return {
        'left_only_successes': left_only,
        'right_only_successes': right_only,
        'discordant': discordant,
        'exact_two_sided_p_value': p_value,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values for one comparison family."""
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def paired_hierarchical_bootstrap(
    left_by_seed: list[np.ndarray],
    right_by_seed: list[np.ndarray],
    *,
    samples: int,
    seed: int = 20260826,
) -> dict:
    if len(left_by_seed) != len(right_by_seed) or not left_by_seed:
        raise ValueError('paired bootstrap requires matching non-empty seeds')
    differences = []
    for left, right in zip(left_by_seed, right_by_seed, strict=True):
        if left.shape != right.shape:
            raise ValueError('paired outcome shapes differ')
        differences.append(left.astype(np.float64) - right.astype(np.float64))
    point = float(np.mean([value.mean() for value in differences]) * 100.0)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    seed_count = len(differences)
    for index in range(samples):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        seed_means = []
        for seed_index in sampled_seeds:
            values = differences[int(seed_index)]
            positions = rng.integers(0, len(values), size=len(values))
            seed_means.append(float(values[positions].mean()))
        draws[index] = np.mean(seed_means) * 100.0
    return {
        'difference_percentage_points': point,
        'ci90_percentage_points': [
            float(np.quantile(draws, 0.05)),
            float(np.quantile(draws, 0.95)),
        ],
        'ci95_percentage_points': [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        'bootstrap_samples': samples,
        'bootstrap_seed': seed,
    }


def codebook_quality(project_root: Path) -> dict:
    paths = {
        'k512_original': (
            '.stablewm/codebook_runs/'
            'official_lewm_pusht_compat_codebook/'
            'quantization_evaluation.json'
        ),
        'k2048_original': (
            '.stablewm/codebook_runs/'
            'official_lewm_pusht_compat_codebook_k2048/'
            'quantization_evaluation.json'
        ),
        'k8192_original': (
            '.stablewm/codebook_runs/'
            'official_lewm_pusht_compat_codebook_k8192/'
            'quantization_evaluation.json'
        ),
    }
    result = {}
    for condition, relative in paths.items():
        payload = json.loads((project_root / relative).read_text())
        test = payload['splits']['test']
        result[condition] = {
            'absolute_l2_mean': test['absolute_l2']['mean'],
            'relative_l2_mean': test['relative_l2']['mean'],
        }
    return result


def collect(
    matrix,
    project_root: Path,
    bootstrap_samples: int,
    training_seeds: list[int] | None = None,
) -> dict:
    seeds = training_seeds or [int(value) for value in matrix.seeds]
    condition_names = list(matrix.conditions)
    summaries = {
        condition: {
            str(seed): evaluation_summary(matrix, condition, seed)
            for seed in seeds
        }
        for condition in condition_names
    }
    conditions = {}
    for condition, by_seed in summaries.items():
        seed_rows = {}
        rates = []
        for seed in seeds:
            summary = by_seed[str(seed)]
            heldout = summary['heldout_test']
            outcomes = np.asarray(
                heldout['episode_successes'], dtype=bool
            )
            rows = np.asarray(heldout['row_indices'], dtype=np.int64)
            rates.append(float(heldout['success_rate']))
            seed_rows[str(seed)] = {
                'selected_stage': summary['best_stage']['stage'],
                'selection_success_rate': summary['best_stage'][
                    'success_rate'
                ],
                'heldout_success_rate': heldout['success_rate'],
                'heldout_successes': heldout['successes'],
                'heldout_count': heldout['num_eval'],
                'episode_successes': outcomes.tolist(),
                'row_indices': rows.tolist(),
                'summary_path': summary['_path'],
            }
        conditions[condition] = {
            'mean_heldout_success_rate': float(np.mean(rates)),
            'std_heldout_success_rate': float(
                np.std(rates, ddof=1) if len(rates) > 1 else 0.0
            ),
            'seeds': seed_rows,
        }

    comparisons = {}
    margin = float(
        matrix.evaluation.equivalence_margin_percentage_points
    )
    for left, right in COMPARISONS:
        left_values = []
        right_values = []
        mcnemar = {}
        for seed in seeds:
            left_row = conditions[left]['seeds'][str(seed)]
            right_row = conditions[right]['seeds'][str(seed)]
            if left_row['row_indices'] != right_row['row_indices']:
                raise RuntimeError(
                    f'held-out rows differ: {left} vs {right}, seed={seed}'
                )
            left_outcomes = np.asarray(
                left_row['episode_successes'], dtype=bool
            )
            right_outcomes = np.asarray(
                right_row['episode_successes'], dtype=bool
            )
            left_values.append(left_outcomes)
            right_values.append(right_outcomes)
            mcnemar[str(seed)] = exact_mcnemar_p(
                left_outcomes, right_outcomes
            )
        pooled_mcnemar = exact_mcnemar_p(
            np.concatenate(left_values),
            np.concatenate(right_values),
        )
        bootstrap = paired_hierarchical_bootstrap(
            left_values,
            right_values,
            samples=bootstrap_samples,
        )
        ci90 = bootstrap['ci90_percentage_points']
        equivalent = ci90[0] >= -margin and ci90[1] <= margin
        comparisons[f'{left}_vs_{right}'] = {
            'left': left,
            'right': right,
            **bootstrap,
            'equivalence_margin_percentage_points': margin,
            'practically_equivalent': equivalent,
            'mcnemar_by_seed': mcnemar,
            'mcnemar_pooled': pooled_mcnemar,
        }

    adjusted = holm_adjust(
        {
            name: row['mcnemar_pooled']['exact_two_sided_p_value']
            for name, row in comparisons.items()
        }
    )
    for name, row in comparisons.items():
        row['mcnemar_pooled']['holm_adjusted_p_value'] = adjusted[name]
        ci90 = row['ci90_percentage_points']
        beyond_margin = ci90[0] > margin or ci90[1] < -margin
        significant = adjusted[name] < 0.05 and beyond_margin
        row['significant_difference'] = significant
        if row['practically_equivalent']:
            row['decision'] = 'practically_equivalent'
        elif significant:
            row['decision'] = 'significant_difference'
        else:
            row['decision'] = 'inconclusive'

    transform_manifest = (
        Path(matrix.transform.rigid_output).expanduser().resolve()
        / 'rigid_transform_manifest.json'
    )
    diagnostic_path = (
        Path(matrix.experiment.root).expanduser().resolve()
        / 'diagnostic_decision.json'
    )
    return {
        'status': 'complete',
        'experiment': str(
            Path(matrix.experiment.root).expanduser().resolve()
        ),
        'training_seeds': seeds,
        'split_seed': int(matrix.split_seed),
        'devices': [int(value) for value in matrix.compute.devices],
        'global_batch_size': int(matrix.compute.global_batch_size),
        'batch_size_per_gpu': int(matrix.compute.batch_size_per_gpu),
        'codebook_quality': codebook_quality(project_root),
        'conditions': conditions,
        'comparisons': comparisons,
        'rigid_transform': json.loads(transform_manifest.read_text()),
        'diagnostic_decision': (
            json.loads(diagnostic_path.read_text())
            if diagnostic_path.is_file()
            else None
        ),
    }


def markdown_report(summary: dict) -> str:
    quality = summary['codebook_quality']
    conditions = summary['conditions']
    comparisons = summary['comparisons']
    seeds = summary['training_seeds']
    status = (
        '已完成多训练 seed 正式确认'
        if len(seeds) > 1
        else '已完成 seed3072 受控主实验；结论限于单一训练 seed'
    )
    lines = [
        '# 码本质量与 K8192 刚体变换实验报告',
        '',
        f'> 实验状态：{status}  ',
        f'> 训练 seeds：{seeds}  ',
        '> 主要指标：200 个固定 held-out PushT 任务成功率',
        '',
        '## 1. 码本量化质量',
        '',
        '| 条件 | 测试绝对 L2 | 测试相对 L2 |',
        '|---|---:|---:|',
    ]
    for condition in (
        'k512_original',
        'k2048_original',
        'k8192_original',
    ):
        row = quality[condition]
        lines.append(
            f'| {condition} | {row["absolute_l2_mean"]:.4f} | '
            f'{row["relative_l2_mean"] * 100:.2f}% |'
        )
    lines.extend(
        [
            '',
            '## 2. Held-out 任务结果',
            '',
            '| 条件 | '
            + ' | '.join(f'Seed {seed}' for seed in seeds)
            + ' | 均值 ± 标准差 |',
            '|---|' + '---:|' * (len(seeds) + 1),
        ]
    )
    for condition, row in conditions.items():
        seed_rates = ' | '.join(
            f'{row["seeds"][str(seed)]["heldout_success_rate"]:.2f}%'
            for seed in seeds
        )
        lines.append(
            f'| {condition} | {seed_rates} | '
            f'{row["mean_heldout_success_rate"]:.2f}% ± '
            f'{row["std_heldout_success_rate"]:.2f} |'
        )
    lines.extend(
        [
            '',
            '## 3. 配对差值与实践等价性',
            '',
            '| 比较（左−右） | 差值 | 90% CI | 95% CI | Holm p | 判定 |',
            '|---|---:|---:|---:|---:|---|',
        ]
    )
    for row in comparisons.values():
        ci90 = row['ci90_percentage_points']
        ci95 = row['ci95_percentage_points']
        holm_p = row['mcnemar_pooled']['holm_adjusted_p_value']
        lines.append(
            f'| {row["left"]} − {row["right"]} | '
            f'{row["difference_percentage_points"]:+.2f}pp | '
            f'[{ci90[0]:+.2f}, {ci90[1]:+.2f}] | '
            f'[{ci95[0]:+.2f}, {ci95[1]:+.2f}] | '
            f'{holm_p:.4g} | {row["decision"]} |'
        )
    rigid = summary['rigid_transform']
    audit = rigid['audit']
    lines.extend(
        [
            '',
            '## 4. K8192 刚体变换审计',
            '',
            f'- 变换模式：{rigid["mode"]}',
            f'- 旋转 determinant：{rigid["rotation_determinant"]:.8f}',
            f'- 正交最大误差：{rigid["orthogonality_max_abs_error"]:.3e}',
            f'- 平移长度：{rigid["translation_norm"]:.6f}',
            f'- Top-1 token 一致率：{audit["top1_agreement"] * 100:.4f}%',
            f'- Top-32 有序一致率：'
            f'{audit["top32_ordered_agreement"] * 100:.4f}%',
            '',
            '## 5. 异常诊断决策',
            '',
        ]
    )
    diagnostic = summary.get('diagnostic_decision')
    if diagnostic is None:
        lines.append('- 尚无诊断决策文件。')
    elif diagnostic['triggered']:
        lines.append(
            f'- Seed3072 rigid/original 差值为 '
            f'{diagnostic["observed_seed3072_gap_percentage_points"]:.2f}pp，'
            f'超过 {diagnostic["trigger_threshold_percentage_points"]:.1f}pp，'
            '已触发 rotation-only 与 translation-only。'
        )
    else:
        lines.append(
            f'- Seed3072 rigid/original 差值为 '
            f'{diagnostic["observed_seed3072_gap_percentage_points"]:.2f}pp，'
            f'未超过 {diagnostic["trigger_threshold_percentage_points"]:.1f}pp，'
            '因此未运行额外诊断条件。'
        )
    lines.extend(
        [
            '',
            '## 6. 结论',
            '',
            (
                '最终结论以本报告的 held-out 成功率、配对置信区间和'
                '预注册 ±5pp 实践等价标准为准。'
                + ('已完成多 seed 正式确认，但其统计强度仍低于三 seed '
                   '或更多重复实验。' if len(seeds) > 1 else
                   '当前仅完成 seed3072，不能外推为正式多-seed 结论。')
            ),
            '',
        ]
    )
    return '\n'.join(lines)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    matrix = OmegaConf.load(Path(args.config).expanduser().resolve())
    OmegaConf.resolve(matrix)
    training_seeds = (
        [int(value) for value in args.seeds.split(',') if value.strip()]
        if args.seeds
        else None
    )
    summary = collect(
        matrix, project_root, args.bootstrap_samples, training_seeds
    )
    summary_path = (
        Path(matrix.experiment.root).expanduser().resolve()
        / 'experiment_summary.json'
    )
    atomic_write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        summary_path,
    )
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = project_root / report_path
    atomic_write_text(markdown_report(summary), report_path)
    print(f'Experiment summary: {summary_path}', flush=True)
    print(f'Experiment report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
