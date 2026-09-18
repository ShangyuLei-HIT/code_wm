"""Aggregate multi-seed multitask evaluation runs into statistics.

Reads ``task_evaluation/summary.json`` from every directory matched by the
``--runs label=glob`` patterns (literal globs, one seed run per directory,
seed parsed from the trailing ``_seed<digits>``), computes per-label,
per-task success statistics across seeds, macro / worst-task statistics,
teacher gaps from ``--teachers task=path.json``, and a per-seed paired
comparison between two labels (``--paired student=teacher``).

Every matched directory is accounted for in ``aggregation_manifest.json``:
failed, incomplete, unparsable, or duplicate runs are NEVER silently
dropped — they are recorded with a status and a reason, and simply do not
contribute to the statistics.

Statistics
    mean, std (ddof=1 when n>1), and a 95% CI
    ``t_{0.975,n-1} * std / sqrt(n)``. The t critical values are hardcoded
    for df = 1..30 and use the normal quantile 1.959963984540054 for
    df > 30 (documented divergence from scipy of at most ~0.4% at df=30;
    with 1.96 the CI is slightly conservative). Paired p-values are the
    exact two-sided Student-t p, computed from the regularized incomplete
    beta function via the modified Lentz continued fraction (std
    Numerical-Recipes implementation, no scipy).

Success values are reported in whatever unit the evaluations produced
(``metrics.success_rate`` — percent in this project).

CPU only; stdlib + numpy.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Two-sided 95% critical values t_{0.975, df} for df = 1..30; the normal
# quantile is used beyond df 30 (see module docstring).
T_CRITICAL_95 = {
    1: 12.706204736,
    2: 4.302652730,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
    11: 2.200985160,
    12: 2.178812830,
    13: 2.160368656,
    14: 2.144786688,
    15: 2.131449546,
    16: 2.119905299,
    17: 2.109815578,
    18: 2.100922040,
    19: 2.093024054,
    20: 2.085963447,
    21: 2.079613845,
    22: 2.073873068,
    23: 2.068657610,
    24: 2.063898562,
    25: 2.059538553,
    26: 2.055529466,
    27: 2.051830516,
    28: 2.048407142,
    29: 2.045229642,
    30: 2.042272456,
}
T_NORMAL_95 = 1.959963984540054
P_METHOD = (
    'two-sided Student-t p from the regularized incomplete beta function '
    '(modified Lentz continued fraction; no scipy)'
)
SEED_PATTERN = re.compile(r'_seed(\d+)$')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Aggregate multi-seed multitask evaluation summaries into '
            'aggregated.json / aggregated.md / aggregation_manifest.json.'
        )
    )
    parser.add_argument(
        '--runs',
        action='append',
        default=[],
        metavar='LABEL=GLOB',
        help=(
            "run label and a literal glob of its per-seed run directories, "
            "e.g. 'M2=/abs/path/pusht_tworoom_cube_uot_seed*' (repeatable; "
            'no environment-variable expansion)'
        ),
    )
    parser.add_argument(
        '--teachers',
        action='append',
        default=[],
        metavar='TASK=PATH.JSON',
        help=(
            'per-task teacher success json; accepted shapes: a bare number, '
            '{"success_rate": x}, or {"metrics": {"success_rate": x}} '
            '(repeatable)'
        ),
    )
    parser.add_argument(
        '--paired',
        action='append',
        default=None,
        metavar='STUDENT=TEACHER',
        help=(
            'label pair for the per-seed paired difference, e.g. M2=M0; '
            "default: M2=M0 when both labels appear in --runs"
        ),
    )
    parser.add_argument(
        '--summary-subdir',
        default='task_evaluation',
        help=(
            'subdirectory of each run holding summary.json '
            '(e.g. task_evaluation_manifest for manifest-based evals)'
        ),
    )
    parser.add_argument(
        '--out-dir',
        default='results/multiseed',
        help='output directory for the three report files',
    )
    return parser.parse_args()


def split_assignment(value: str, option: str) -> tuple[str, str]:
    label, separator, rest = value.partition('=')
    if not separator or not label or not rest:
        raise ValueError(
            f'invalid {option} value {value!r}: expected LABEL=VALUE'
        )
    return label, rest


def t_critical(df: int) -> float:
    if df < 1:
        raise ValueError(f'degrees of freedom must be >= 1, got {df}')
    return T_CRITICAL_95.get(df, T_NORMAL_95)


def summarize_values(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    stats = {
        'n_seeds': n,
        'mean': float(array.mean()) if n else None,
        'std': None,
        'ci95': None,
        't_critical': None,
        'ci_method': (
            't_{0.975,n-1} * std / sqrt(n); t table hardcoded for df<=30, '
            'normal 1.95996 beyond'
        ),
    }
    if n > 1:
        stats['std'] = float(array.std(ddof=1))
        stats['t_critical'] = t_critical(n - 1)
        stats['ci95'] = float(
            stats['t_critical'] * stats['std'] / math.sqrt(n)
        )
    return stats


# --------------------------------------------------------------------------
# Student-t two-sided p-value via the regularized incomplete beta function.
# --------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float, max_iterations: int = 300,
            epsilon: float = 3.0e-12) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    converged = False
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            converged = True
            break
    if not converged:
        raise ArithmeticError(
            f'incomplete-beta continued fraction did not converge for '
            f'a={a}, b={b}, x={x} within {max_iterations} iterations'
        )
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_statistic: float, df: int) -> float:
    if df < 1:
        raise ValueError(f'degrees of freedom must be >= 1, got {df}')
    if math.isinf(t_statistic):
        return 0.0
    if math.isnan(t_statistic):
        raise ValueError('t statistic is NaN')
    x = df / (df + t_statistic * t_statistic)
    return regularized_incomplete_beta(0.5 * df, 0.5, x)


def paired_stats(diffs: list[float]) -> dict:
    stats = summarize_values(diffs)
    n = stats['n_seeds']
    if n > 1 and stats['std'] == 0.0:
        # Zero variance: t diverges (p = 0) unless the mean is also zero,
        # in which case any t is consistent with the data (p = 1).
        stats['t_statistic'] = (
            math.copysign(math.inf, stats['mean']) if stats['mean'] != 0.0
            else None
        )
        stats['df'] = n - 1
        stats['p_two_sided'] = (
            student_t_two_sided_p(stats['t_statistic'], n - 1)
            if stats['t_statistic'] is not None
            else None
        )
    elif n > 1:
        stats['t_statistic'] = stats['mean'] / (
            stats['std'] / math.sqrt(n)
        )
        stats['df'] = n - 1
        stats['p_two_sided'] = student_t_two_sided_p(
            stats['t_statistic'], n - 1
        )
    else:
        stats['t_statistic'] = None
        stats['df'] = None
        stats['p_two_sided'] = None
    stats['p_method'] = P_METHOD
    return stats


# --------------------------------------------------------------------------
# Run-directory inspection
# --------------------------------------------------------------------------


def training_steps(run_dir: Path) -> int | None:
    path = run_dir / 'metrics.jsonl'
    if not path.is_file():
        return None
    last_line = None
    try:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    last_line = line
        if last_line is None:
            return None
        value = json.loads(last_line).get('global_step')
        return int(value) if isinstance(value, (int, float)) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def classify_run(run_dir: Path, summary_subdir: str = 'task_evaluation') -> dict:
    summary_path = run_dir / summary_subdir / 'summary.json'
    record = {
        'dir': str(run_dir),
        'status': 'ok',
        'reason': None,
        'per_task': {},
        'tasks_missing_success': [],
        'training_steps': training_steps(run_dir),
        'summary': None,
    }
    if not (run_dir / 'weights_final.pt').is_file():
        record['status'] = 'missing_weights_final'
        record['reason'] = (
            'weights_final.pt not found in the run directory '
            '(training incomplete or crashed)'
        )
        return record
    if not summary_path.is_file():
        record['status'] = 'missing_summary'
        record['reason'] = f'missing {summary_path}'
        return record
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        record['status'] = 'missing_summary'
        record['reason'] = f'unreadable summary.json: {exc}'
        return record
    record['summary'] = summary
    tasks = summary.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        record['status'] = 'eval_failed'
        record['reason'] = 'summary.json contains no task entries'
        return record
    for entry in tasks:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get('task'))
        metrics = entry.get('metrics')
        rate = (
            metrics.get('success_rate')
            if isinstance(metrics, dict)
            else None
        )
        if (
            isinstance(rate, (int, float))
            and not isinstance(rate, bool)
            and math.isfinite(float(rate))
        ):
            record['per_task'][name] = float(rate)
        else:
            record['tasks_missing_success'].append(name)
    if record['tasks_missing_success']:
        record['status'] = 'eval_failed'
        record['reason'] = (
            'missing success_rate for task(s): '
            f'{sorted(set(record["tasks_missing_success"]))}'
        )
    return record


def collect_label(label: str, pattern: str) -> dict:
    matches = sorted(glob.glob(pattern))
    directories = [Path(match) for match in matches if Path(match).is_dir()]
    ignored = [match for match in matches if not Path(match).is_dir()]
    records = []
    for directory in directories:
        match = SEED_PATTERN.search(directory.name)
        record = classify_run(directory, args.summary_subdir)
        if match is None:
            record['seed'] = None
            record['status'] = 'unparsed_seed'
            record['reason'] = (
                f'directory name {directory.name!r} does not end in '
                '_seed<digits>'
            )
        else:
            record['seed'] = int(match.group(1))
        records.append(record)
    by_seed: dict[int, list[dict]] = {}
    for record in records:
        if record['seed'] is not None:
            by_seed.setdefault(record['seed'], []).append(record)
    for seed, group in by_seed.items():
        if len(group) > 1:
            for record in group:
                record['status'] = 'duplicate_seed'
                record['reason'] = (
                    f'{len(group)} directories claim seed {seed}: '
                    f'{[entry["dir"] for entry in group]}'
                )
    for record in records:
        record['included_in_stats'] = record['status'] == 'ok'
        if record['status'] == 'ok':
            values = list(record['per_task'].values())
            record['macro'] = float(np.mean(values)) if values else None
            record['worst_task'] = float(min(values)) if values else None
    return {
        'label': label,
        'glob': pattern,
        'records': records,
        'ignored_matches': ignored,
        'note': (
            'no directories matched the glob'
            if not directories and not ignored
            else (
                'all glob matches were files, not directories'
                if not directories
                else None
            )
        ),
    }


# --------------------------------------------------------------------------
# Episode-count bookkeeping
# --------------------------------------------------------------------------


def config_num_eval(run_dir: Path) -> int | None:
    """Minimal targeted parse of the run's config.yaml for num_eval."""
    path = run_dir / 'config.yaml'
    if not path.is_file():
        return None
    in_evaluation = False
    try:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if not line[0].isspace():
                in_evaluation = line.startswith('evaluation:')
                continue
            if in_evaluation:
                match = re.match(r'\s*num_eval:\s*(\d+)', line)
                if match:
                    return int(match.group(1))
    except OSError:
        return None
    return None


def count_episodes(label_data: dict, task_name: str) -> tuple[int | None, str]:
    """Number of evaluated episodes for one label x task.

    Resolution order: the eval result json's ``episode_indices`` length,
    the manifest entry count recorded in summary.json (manifest-driven
    runs), ``episode_successes`` length, then the run config's
    ``evaluation.num_eval``.
    """
    for record in label_data['records']:
        if not record['included_in_stats']:
            continue
        summary = record['summary'] or {}
        entry = next(
            (
                item
                for item in summary.get('tasks', [])
                if isinstance(item, dict) and str(item.get('task')) == task_name
            ),
            None,
        )
        if entry is None:
            continue
        result_path = entry.get('result')
        if result_path and Path(result_path).is_file():
            try:
                payload = json.loads(Path(result_path).read_text())
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict) and isinstance(
                payload.get('episode_indices'), list
            ):
                return len(payload['episode_indices']), 'result_json'
        manifest = entry.get('manifest')
        if (
            isinstance(manifest, dict)
            and isinstance(manifest.get('num_entries'), int)
        ):
            return int(manifest['num_entries']), 'manifest_num_entries'
        metrics = entry.get('metrics')
        if isinstance(metrics, dict) and isinstance(
            metrics.get('episode_successes'), list
        ):
            return len(metrics['episode_successes']), 'episode_successes'
        fallback = config_num_eval(Path(record['dir']))
        if fallback is not None:
            return fallback, 'run_config_num_eval'
    return None, 'unavailable'


# --------------------------------------------------------------------------
# Teachers
# --------------------------------------------------------------------------


def load_teacher_success(path: str) -> float:
    teacher_path = Path(path).expanduser()
    if not teacher_path.is_file():
        raise ValueError(f'teacher success file not found: {teacher_path}')
    try:
        payload = json.loads(teacher_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f'unreadable teacher success json {teacher_path}: {exc}'
        ) from exc
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return float(payload)
    if isinstance(payload, dict):
        rate = payload.get('success_rate')
        if isinstance(rate, (int, float)) and not isinstance(rate, bool):
            return float(rate)
        metrics = payload.get('metrics')
        if isinstance(metrics, dict) and isinstance(
            metrics.get('success_rate'), (int, float)
        ) and not isinstance(metrics.get('success_rate'), bool):
            return float(metrics['success_rate'])
    raise ValueError(
        f'cannot read a success_rate from {teacher_path}: expected a bare '
        'number, {"success_rate": x}, or {"metrics": {"success_rate": x}}'
    )


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def build_label_stats(label_data: dict) -> dict:
    ok_records = [
        record for record in label_data['records'] if record['included_in_stats']
    ]
    tasks = sorted(
        {name for record in ok_records for name in record['per_task']}
    )
    task_sets = {
        tuple(sorted(record['per_task'])) for record in ok_records
    }
    per_task = {}
    for task_name in tasks:
        per_seed = [
            (record['seed'], record['per_task'][task_name])
            for record in ok_records
            if task_name in record['per_task']
        ]
        stats = summarize_values([value for _, value in per_seed])
        n_episodes, episodes_source = count_episodes(label_data, task_name)
        stats['per_seed'] = [
            {'seed': seed, 'success': value} for seed, value in per_seed
        ]
        stats['n_episodes'] = n_episodes
        stats['n_episodes_source'] = episodes_source
        per_task[task_name] = stats
    stats = {
        'n_dirs_matched': len(label_data['records']),
        'n_seeds_ok': len(ok_records),
        'tasks': per_task,
        'task_set_mismatch': (
            sorted({' | '.join(names) for names in task_sets})
            if len(task_sets) > 1
            else None
        ),
    }
    for key, field in (('macro', 'macro'), ('worst_task', 'worst_task')):
        values = [
            record[field]
            for record in ok_records
            if record[field] is not None
        ]
        stats[key] = summarize_values(values)
        stats[key]['per_seed'] = [
            {'seed': record['seed'], 'value': record[field]}
            for record in ok_records
            if record[field] is not None
        ]
    return stats


def build_teacher_gaps(labels: dict, teachers: dict) -> dict:
    gaps = {}
    for label, data in labels.items():
        per_task = {}
        for task_name, teacher_rate in teachers.items():
            stats = data['stats']['tasks'].get(task_name)
            if stats is None or stats['mean'] is None:
                per_task[task_name] = {
                    'teacher': teacher_rate,
                    'student_mean': None,
                    'gap': None,
                    'note': (
                        'no successful seeds for this task under this label'
                    ),
                }
                continue
            per_task[task_name] = {
                'teacher': teacher_rate,
                'student_mean': stats['mean'],
                'gap': teacher_rate - stats['mean'],
                'note': None,
            }
        skipped = sorted(set(data['stats']['tasks']) - set(teachers))
        gaps[label] = {'per_task': per_task, 'tasks_without_teacher': skipped}
    return gaps


def build_paired(
    labels: dict, student_label: str, teacher_label: str
) -> dict:
    if student_label not in labels:
        raise ValueError(
            f'paired student label {student_label!r} not among run labels '
            f'{sorted(labels)}'
        )
    if teacher_label not in labels:
        raise ValueError(
            f'paired reference label {teacher_label!r} not among run labels '
            f'{sorted(labels)}'
        )
    student_ok = {
        record['seed']: record
        for record in labels[student_label]['records']
        if record['included_in_stats']
    }
    reference_ok = {
        record['seed']: record
        for record in labels[teacher_label]['records']
        if record['included_in_stats']
    }
    common = sorted(set(student_ok) & set(reference_ok))
    tasks = sorted(
        {
            name
            for seed in common
            for name in student_ok[seed]['per_task']
            if name in reference_ok[seed]['per_task']
        }
    )
    per_task = {}
    for task_name in tasks:
        diffs = [
            (
                seed,
                student_ok[seed]['per_task'][task_name]
                - reference_ok[seed]['per_task'][task_name],
            )
            for seed in common
            if task_name in student_ok[seed]['per_task']
            and task_name in reference_ok[seed]['per_task']
        ]
        stats = paired_stats([value for _, value in diffs])
        stats['per_seed'] = [
            {'seed': seed, 'diff': value} for seed, value in diffs
        ]
        per_task[task_name] = stats
    return {
        'student_label': student_label,
        'reference_label': teacher_label,
        'common_seeds': common,
        'seeds_only_in_student': sorted(set(student_ok) - set(reference_ok)),
        'seeds_only_in_reference': sorted(set(reference_ok) - set(student_ok)),
        'per_task': per_task,
    }


def format_mean_ci(stats: dict) -> str:
    if stats['mean'] is None:
        return 'n/a'
    if stats['ci95'] is None:
        return f"{stats['mean']:.2f} (n=1, CI n/a)"
    return f"{stats['mean']:.2f} +/- {stats['ci95']:.2f}"


def render_markdown(
    aggregated: dict, pairs: list[tuple[str, str]]
) -> str:
    lines = [
        '# Multi-seed aggregation',
        '',
        f"Generated: {aggregated['generated_at']} (UTC ISO)",
        '',
        'Success values are `metrics.success_rate` as reported by the '
        'evaluations (percent). CI is the two-sided 95% interval '
        '`t_{0.975,n-1} * std / sqrt(n)` (t table hardcoded for df<=30, '
        'normal 1.95996 beyond); std uses ddof=1.',
        '',
        '## Per-task success (mean +/- 95% CI, n_seeds, n_episodes)',
        '',
        '| label | task | success | n_seeds | n_episodes |',
        '|---|---|---:|---:|---:|',
    ]
    for label, data in aggregated['labels'].items():
        for task_name, stats in data['tasks'].items():
            episodes = (
                'n/a'
                if stats['n_episodes'] is None
                else str(stats['n_episodes'])
            )
            lines.append(
                f"| {label} | {task_name} | {format_mean_ci(stats)} | "
                f"{stats['n_seeds']} | {episodes} |"
            )
        for key, title in (('macro', 'macro (mean over tasks)'),
                           ('worst_task', 'worst task')):
            stats = data[key]
            lines.append(
                f"| {label} | {title} | {format_mean_ci(stats)} | "
                f"{stats['n_seeds']} | n/a |"
            )
    lines += ['', '## Teacher gap (teacher minus student mean)', '',
              '| label | task | teacher | student mean | gap |',
              '|---|---|---:|---:|---:|']
    for label, gap in aggregated['teacher_gaps'].items():
        for task_name, row in gap['per_task'].items():
            student = (
                'n/a' if row['student_mean'] is None
                else f"{row['student_mean']:.2f}"
            )
            gap_text = 'n/a' if row['gap'] is None else f"{row['gap']:.2f}"
            lines.append(
                f"| {label} | {task_name} | {row['teacher']:.2f} | "
                f"{student} | {gap_text} |"
            )
    for student_label, reference_label in pairs:
        pair = aggregated['paired'][f'{student_label}={reference_label}']
        lines += [
            '',
            f'## Paired difference {student_label} minus {reference_label} '
            '(per seed, per task)',
            '',
            f"Common seeds: {pair['common_seeds'] or 'none'}; "
            f"only in {student_label}: {pair['seeds_only_in_student'] or 'none'}; "
            f"only in {reference_label}: {pair['seeds_only_in_reference'] or 'none'}.",
            '',
            'p-values: ' + P_METHOD + '.',
            '',
            '| task | diff (mean +/- 95% CI) | t | df | p (two-sided) '
            '| n_seeds |',
            '|---|---|---:|---:|---:|---:|',
        ]
        for task_name, stats in pair['per_task'].items():
            t_text = (
                'n/a' if stats['t_statistic'] is None
                else f"{stats['t_statistic']:.3f}"
            )
            df_text = 'n/a' if stats['df'] is None else str(stats['df'])
            p_text = (
                'n/a' if stats['p_two_sided'] is None
                else f"{stats['p_two_sided']:.4f}"
            )
            lines.append(
                f"| {task_name} | {format_mean_ci(stats)} | {t_text} | "
                f"{df_text} | {p_text} | {stats['n_seeds']} |"
            )
    return '\n'.join(lines) + '\n'


def build_manifest(
    labels: dict, paired_cache: dict, pairs: list[tuple[str, str]]
) -> dict:
    manifest = {
        'format_version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'labels': {},
        'pairs': {},
    }
    for label, data in labels.items():
        manifest['labels'][label] = {
            'glob': data['glob'],
            'n_matched_dirs': len(data['records']),
            'ignored_non_dir_matches': data['ignored_matches'],
            'note': data['note'],
            'seeds': [
                {
                    'dir': record['dir'],
                    'seed': record['seed'],
                    'status': record['status'],
                    'reason': record['reason'],
                    'included_in_stats': record['included_in_stats'],
                    'tasks_with_success': sorted(record['per_task']),
                    'tasks_missing_success': sorted(
                        set(record['tasks_missing_success'])
                    ),
                    'training_steps': record['training_steps'],
                }
                for record in data['records']
            ],
        }
    for student_label, reference_label in pairs:
        key = f'{student_label}={reference_label}'
        pair = paired_cache[key]
        manifest['pairs'][key] = {
            'student_label': student_label,
            'reference_label': reference_label,
            'common_seeds': pair['common_seeds'],
            'seeds_only_in_student': pair['seeds_only_in_student'],
            'seeds_only_in_reference': pair['seeds_only_in_reference'],
        }
    return manifest


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if not args.runs:
        raise ValueError('at least one --runs LABEL=GLOB is required')

    labels: dict[str, dict] = {}
    for value in args.runs:
        label, pattern = split_assignment(value, '--runs')
        if label in labels:
            raise ValueError(f'duplicate run label {label!r}')
        labels[label] = collect_label(label, pattern)

    teachers: dict[str, float] = {}
    for value in args.teachers:
        task_name, path = split_assignment(value, '--teachers')
        if task_name in teachers:
            raise ValueError(f'duplicate teacher task {task_name!r}')
        teachers[task_name] = load_teacher_success(path)

    pairs: list[tuple[str, str]] = []
    if args.paired is not None:
        for value in args.paired:
            pairs.append(split_assignment(value, '--paired'))
    elif 'M2' in labels and 'M0' in labels:
        pairs.append(('M2', 'M0'))

    paired_cache: dict[str, dict] = {}
    for student_label, reference_label in pairs:
        key = f'{student_label}={reference_label}'
        if key in paired_cache:
            raise ValueError(f'duplicate pair {key!r}')
        paired_cache[key] = build_paired(
            labels, student_label, reference_label
        )

    for label, data in labels.items():
        data['stats'] = build_label_stats(data)

    aggregated = {
        'format_version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'labels': {label: data['stats'] for label, data in labels.items()},
        'teachers': teachers,
        'teacher_gaps': build_teacher_gaps(labels, teachers),
        'paired': paired_cache,
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        out_dir / 'aggregated.json',
        json.dumps(aggregated, indent=2, sort_keys=True) + '\n',
    )
    atomic_write(
        out_dir / 'aggregated.md', render_markdown(aggregated, pairs)
    )
    manifest = build_manifest(labels, paired_cache, pairs)
    atomic_write(
        out_dir / 'aggregation_manifest.json',
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
    )
    for label, data in labels.items():
        ok = sum(1 for record in data['records'] if record['included_in_stats'])
        failed = len(data['records']) - ok
        if failed:
            print(
                f'[warn] label {label}: {failed}/{len(data["records"])} '
                'matched runs excluded from stats (see '
                'aggregation_manifest.json)',
                flush=True,
            )
    print(f'wrote aggregated reports to {out_dir}', flush=True)


if __name__ == '__main__':
    main()
