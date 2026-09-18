"""Decompose the supervision gap between teacher, experts, and students.

Collects per-task success rates from multitask distillation run directories
(``task_evaluation/summary.json``), single-task teacher/expert result JSONs
(``{"metrics": {"success_rate": ...}}``), and reports every gap of interest:
teacher - student, expert - student, and the paired continuous-minus-discrete
contrasts m4 - m2 / m4 - m5 (paired per training seed). Missing inputs are
recorded in the ``missing`` section and never dropped silently.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# (positive arm, negative arm, report key) for the paired contrasts; the arms
# refer to --student names m4 (continuous supervision) vs m2 (mixed) / m5
# (fully discrete supervision).
PAIRINGS = (
    ('m4', 'm2', 'm4_minus_m2'),
    ('m4', 'm5', 'm4_minus_m5'),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Supervision-gap decomposition across tasks and seeds.'
    )
    parser.add_argument(
        '--student',
        action='append',
        default=[],
        metavar='NAME=RUN_DIR',
        help='distilled multitask run directory holding '
        'task_evaluation/summary.json (repeatable, same NAME may appear '
        'once per seed, e.g. m2=..., m4=..., m5=..., factorial_cc_w0=...)',
    )
    parser.add_argument(
        '--teacher',
        action='append',
        default=[],
        metavar='TASK=JSON',
        help='per-task teacher result JSON with metrics.success_rate '
        '(repeatable, e.g. '
        'pusht=.stablewm/checkpoints/official_lewm_pusht_compat/'
        'task_evaluation/pusht_results_official_seed42_50.json)',
    )
    parser.add_argument(
        '--expert',
        action='append',
        default=[],
        metavar='TASK=JSON',
        help='optional per-task single-task distilled expert (C1) result '
        'JSON (repeatable)',
    )
    parser.add_argument(
        '--summary-subdir',
        default='task_evaluation',
        help=(
            'subdirectory of each student run holding summary.json '
            '(e.g. task_evaluation_manifest)'
        ),
    )
    parser.add_argument(
        '--out-dir',
        default='results/supervision_gap',
        metavar='DIR',
        help='output directory for decomposition.json / decomposition.md',
    )
    return parser.parse_args()


def parse_named(items: list[str], flag: str) -> list[tuple[str, str]]:
    parsed = []
    for item in items:
        name, separator, value = item.partition('=')
        if not separator or not name or not value:
            raise ValueError(f'{flag} expects NAME=VALUE, got {item!r}')
        parsed.append((name, value))
    return parsed


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def seed_label(seed: int | None) -> str:
    return 'unknown' if seed is None else str(seed)


def extract_success_rate(payload: object) -> float | None:
    metrics = payload.get('metrics') if isinstance(payload, dict) else None
    # Single-task C1 runs summarize as {'best_stage': {'success_rate': ...},
    # 'stages': [{'success_rate': ...}, ...]} instead of eval_wm's
    # {'metrics': {'success_rate': ...}}.
    best_stage = payload.get('best_stage') if isinstance(payload, dict) else None
    for candidate in (
        metrics.get('success_rate') if isinstance(metrics, dict) else None,
        payload.get('success_rate') if isinstance(payload, dict) else None,
        best_stage.get('success_rate')
        if isinstance(best_stage, dict)
        else None,
    ):
        if isinstance(candidate, (int, float)) and not isinstance(
            candidate, bool
        ):
            return float(candidate)
    return None


def read_rate_json(
    path: Path, missing: list[dict], kind: str, label: str
) -> float | None:
    if not path.is_file():
        missing.append(
            {
                'kind': kind,
                'name': label,
                'path': str(path),
                'reason': 'file not found',
            }
        )
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        missing.append(
            {
                'kind': kind,
                'name': label,
                'path': str(path),
                'reason': f'unreadable JSON: {error}',
            }
        )
        return None
    rate = extract_success_rate(payload)
    if rate is None:
        missing.append(
            {
                'kind': kind,
                'name': label,
                'path': str(path),
                'reason': 'no metrics.success_rate found',
            }
        )
    return rate


def detect_seed(run_dir: Path) -> int | None:
    """Best-effort train seed: run_manifest, saved config, then dir name."""
    manifest = run_dir / 'run_manifest.json'
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text())
            seed = payload.get('train_seed') if isinstance(payload, dict) else None
            if isinstance(seed, bool):
                seed = None
            if isinstance(seed, int):
                return int(seed)
            if isinstance(seed, str) and seed.isdigit():
                return int(seed)
        except (json.JSONDecodeError, OSError):
            pass
    config = run_dir / 'config.yaml'
    if config.is_file():
        try:
            match = re.search(
                r'(?m)^seed:\s*(-?\d+)\s*$', config.read_text()
            )
            if match:
                return int(match.group(1))
        except OSError:
            pass
    match = re.search(r'(?<![0-9])seed(\d+)(?![0-9])', run_dir.name)
    if match:
        return int(match.group(1))
    return None


def read_student_summary(
    run_dir: Path,
    missing: list[dict],
    name: str,
    summary_subdir: str = 'task_evaluation',
) -> dict[str, float] | None:
    summary_path = run_dir / summary_subdir / 'summary.json'
    if not summary_path.is_file():
        missing.append(
            {
                'kind': 'student_run',
                'name': name,
                'path': str(summary_path),
                'reason': 'task_evaluation/summary.json not found',
            }
        )
        return None
    try:
        payload = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        missing.append(
            {
                'kind': 'student_run',
                'name': name,
                'path': str(summary_path),
                'reason': f'unreadable JSON: {error}',
            }
        )
        return None
    tasks = payload.get('tasks') if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        missing.append(
            {
                'kind': 'student_run',
                'name': name,
                'path': str(summary_path),
                'reason': 'summary.json has no tasks list',
            }
        )
        return None
    per_task: dict[str, float] = {}
    for entry in tasks:
        if not isinstance(entry, dict):
            continue
        task = entry.get('task')
        rate = extract_success_rate(entry)
        if task is None or rate is None:
            missing.append(
                {
                    'kind': 'student_task',
                    'name': f'{name}/{task}',
                    'path': str(summary_path),
                    'reason': 'no metrics.success_rate for this task',
                }
            )
            continue
        per_task[str(task)] = rate
    return per_task


def collect_students(
    student_args: list[tuple[str, str]], missing: list[dict]
) -> dict[str, list[dict]]:
    students: dict[str, list[dict]] = {}
    for name, run_dir_text in student_args:
        run_dir = Path(run_dir_text).expanduser()
        per_task = read_student_summary(
            run_dir, missing, name, args.summary_subdir
        )
        entry = {
            'run_dir': str(run_dir),
            'seed': detect_seed(run_dir),
            'per_task': per_task or {},
        }
        if per_task is None:
            entry['unusable'] = True
        students.setdefault(name, []).append(entry)
    for name, runs in students.items():
        seen: set[str] = set()
        for run in runs:
            label = seed_label(run['seed'])
            if label in seen:
                raise ValueError(
                    f'student {name!r} has multiple runs with seed {label}; '
                    'seeds must be unique per student name'
                )
            seen.add(label)
    return students


def student_rates_by_seed(
    students: dict[str, list[dict]], name: str, task: str
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for run in students.get(name, []):
        if run.get('unusable'):
            continue
        if task in run['per_task']:
            rates[seed_label(run['seed'])] = run['per_task'][task]
    return rates


def gap_block(reference: float | None, rates: dict[str, float]) -> dict:
    """Reference-minus-student gap per seed (reference - student)."""
    if reference is None or not rates:
        return {'per_seed': {}, 'mean': None, 'n_seeds': 0}
    per_seed = {
        seed: reference - rate for seed, rate in rates.items()
    }
    return {
        'per_seed': per_seed,
        'mean': mean(list(per_seed.values())),
        'n_seeds': len(per_seed),
    }


def paired_gap(
    students: dict[str, list[dict]],
    positive: str,
    negative: str,
    task: str,
) -> dict:
    positive_rates = student_rates_by_seed(students, positive, task)
    negative_rates = student_rates_by_seed(students, negative, task)
    shared = sorted(set(positive_rates) & set(negative_rates))
    per_seed = {
        seed: positive_rates[seed] - negative_rates[seed] for seed in shared
    }
    return {
        'per_seed': per_seed,
        'mean': mean(list(per_seed.values())),
        'n_seeds': len(per_seed),
    }


def format_rate(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.2f}'


def format_gap(block: dict | None) -> str:
    if not block or block.get('mean') is None:
        return 'n/a'
    return f"{block['mean']:.2f} ({block['n_seeds']} seed(s))"


def build_report(
    students: dict[str, list[dict]],
    teachers: dict[str, dict],
    experts: dict[str, dict],
    missing: list[dict],
) -> tuple[dict, str]:
    tasks: list[str] = []
    for source in (teachers, experts):
        for task in source:
            if task not in tasks:
                tasks.append(task)
    for runs in students.values():
        for run in runs:
            for task in run['per_task']:
                if task not in tasks:
                    tasks.append(task)

    per_task: dict[str, dict] = {}
    for task in tasks:
        teacher_success = teachers.get(task, {}).get('success_rate')
        expert_success = experts.get(task, {}).get('success_rate')
        student_blocks = {}
        for name in students:
            rates = student_rates_by_seed(students, name, task)
            student_blocks[name] = {
                'per_seed': rates,
                'mean': mean(list(rates.values())),
                'n_seeds': len(rates),
            }
        gaps = {
            name: {
                'teacher_minus_student': gap_block(
                    teacher_success, block['per_seed']
                ),
                'expert_minus_student': gap_block(
                    expert_success, block['per_seed']
                ),
            }
            for name, block in student_blocks.items()
        }
        pair_blocks = {}
        for positive, negative, key in PAIRINGS:
            if positive in students and negative in students:
                pair_blocks[key] = paired_gap(
                    students, positive, negative, task
                )
        per_task[task] = {
            'teacher_success': teacher_success,
            'expert_success': expert_success,
            'students': student_blocks,
            'gaps': gaps,
            'continuous_minus_discrete': pair_blocks,
        }

    def macro_mean(task_key: str) -> float | None:
        return mean(
            [
                block[task_key]
                for block in per_task.values()
                if block[task_key] is not None
            ]
        )

    macro_students = {
        name: {
            'mean': mean(
                [
                    block['students'][name]['mean']
                    for block in per_task.values()
                    if block['students'][name]['mean'] is not None
                ]
            ),
            'n_seeds': max(
                (
                    block['students'][name]['n_seeds']
                    for block in per_task.values()
                ),
                default=0,
            ),
        }
        for name in students
    }
    macro_gaps = {
        name: {
            'teacher_minus_student': mean(
                [
                    block['gaps'][name]['teacher_minus_student']['mean']
                    for block in per_task.values()
                    if block['gaps'][name]['teacher_minus_student']['mean']
                    is not None
                ]
            ),
            'expert_minus_student': mean(
                [
                    block['gaps'][name]['expert_minus_student']['mean']
                    for block in per_task.values()
                    if block['gaps'][name]['expert_minus_student']['mean']
                    is not None
                ]
            ),
        }
        for name in students
    }
    macro_pairs = {}
    for _, _, key in PAIRINGS:
        if any(key in block['continuous_minus_discrete'] for block in per_task.values()):
            macro_pairs[key] = mean(
                [
                    block['continuous_minus_discrete'][key]['mean']
                    for block in per_task.values()
                    if key in block['continuous_minus_discrete']
                    and block['continuous_minus_discrete'][key]['mean']
                    is not None
                ]
            )

    for positive, negative, key in PAIRINGS:
        if positive not in students or negative not in students:
            missing.append(
                {
                    'kind': 'pairing',
                    'name': key,
                    'path': None,
                    'reason': (
                        f'student {positive!r} or {negative!r} was not '
                        'provided; paired contrast skipped'
                    ),
                }
            )
        else:
            positive_seeds = {
                seed_label(run['seed'])
                for run in students[positive]
                if not run.get('unusable')
            }
            negative_seeds = {
                seed_label(run['seed'])
                for run in students[negative]
                if not run.get('unusable')
            }
            if not (positive_seeds & negative_seeds):
                missing.append(
                    {
                        'kind': 'pairing',
                        'name': key,
                        'path': None,
                        'reason': (
                            f'no shared seed between {positive!r} '
                            f'({sorted(positive_seeds)}) and {negative!r} '
                            f'({sorted(negative_seeds)}); contrast empty'
                        ),
                    }
                )

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'pairings': [
            {'positive': positive, 'negative': negative, 'key': key}
            for positive, negative, key in PAIRINGS
        ],
        'tasks': tasks,
        'teachers': teachers,
        'experts': experts,
        'students': {
            name: [
                {
                    'run_dir': run['run_dir'],
                    'seed': run['seed'],
                    'per_task': run['per_task'],
                }
                for run in runs
            ]
            for name, runs in students.items()
        },
        'per_task': per_task,
        'macro': {
            'teacher_success': macro_mean('teacher_success'),
            'expert_success': macro_mean('expert_success'),
            'students': macro_students,
            'gaps': macro_gaps,
            'continuous_minus_discrete': macro_pairs,
        },
        'missing': missing,
    }

    lines: list[str] = [
        '# Supervision gap decomposition',
        '',
        f'Generated: {report["generated_at"]}',
        f'Tasks: {", ".join(tasks) if tasks else "(none)"}',
        '',
        '## Per-task results',
        '',
    ]
    for task in tasks:
        block = per_task[task]
        lines.append(f'### Task `{task}`')
        lines.append('')
        lines.append('| source | success rate (%) | seeds |')
        lines.append('| --- | --- | --- |')
        lines.append(
            f'| teacher | {format_rate(block["teacher_success"])} | - |'
        )
        lines.append(
            f'| expert | {format_rate(block["expert_success"])} | - |'
        )
        for name, student in block['students'].items():
            seeds = ', '.join(sorted(student['per_seed'])) or '-'
            lines.append(
                f'| student {name} | {format_rate(student["mean"])} '
                f'({student["n_seeds"]} seed(s)) | {seeds} |'
            )
        lines.append('')
        lines.append('| gap | mean (pp) | seeds |')
        lines.append('| --- | --- | --- |')
        for name, gap in block['gaps'].items():
            lines.append(
                f'| teacher - {name} | '
                f'{format_gap(gap["teacher_minus_student"])} | - |'
            )
            if block['expert_success'] is not None:
                lines.append(
                    f'| expert - {name} | '
                    f'{format_gap(gap["expert_minus_student"])} | - |'
                )
        for key, pair in block['continuous_minus_discrete'].items():
            seeds = ', '.join(sorted(pair['per_seed'])) or '-'
            lines.append(
                f'| {key} (paired) | {format_gap(pair)} | {seeds} |'
            )
        lines.append('')
    lines.append('## Macro (mean over tasks)')
    lines.append('')
    lines.append('| source | macro success rate (%) | seeds |')
    lines.append('| --- | --- | --- |')
    lines.append(
        f'| teacher | {format_rate(report["macro"]["teacher_success"])} | - |'
    )
    lines.append(
        f'| expert | {format_rate(report["macro"]["expert_success"])} | - |'
    )
    for name, student in macro_students.items():
        lines.append(
            f'| student {name} | {format_rate(student["mean"])} '
            f'({student["n_seeds"]} seed(s)) | - |'
        )
    lines.append('')
    lines.append('| gap | macro mean (pp) |')
    lines.append('| --- | --- |')
    for name, gap in macro_gaps.items():
        lines.append(
            f'| teacher - {name} | '
            f'{format_rate(gap["teacher_minus_student"])} |'
        )
        lines.append(
            f'| expert - {name} | '
            f'{format_rate(gap["expert_minus_student"])} |'
        )
    for key, value in macro_pairs.items():
        lines.append(f'| {key} (paired) | {format_rate(value)} |')
    lines.append('')
    lines.append('## Missing / skipped')
    lines.append('')
    if missing:
        for item in missing:
            lines.append(
                f'- kind={item["kind"]} name={item["name"]} '
                f'reason={item["reason"]}'
                + (f' path={item["path"]}' if item.get('path') else '')
            )
    else:
        lines.append('- none')
    lines.append('')
    return report, '\n'.join(lines)


def main() -> int:
    args = parse_args()
    student_args = parse_named(args.student, '--student')
    teacher_args = parse_named(args.teacher, '--teacher')
    expert_args = parse_named(args.expert, '--expert')
    missing: list[dict] = []

    teachers: dict[str, dict] = {}
    for task, path_text in teacher_args:
        path = Path(path_text).expanduser()
        rate = read_rate_json(path, missing, 'teacher', task)
        teachers[task] = {
            'path': str(path),
            'success_rate': rate,
        }

    experts: dict[str, dict] = {}
    for task, path_text in expert_args:
        path = Path(path_text).expanduser()
        rate = read_rate_json(path, missing, 'expert', task)
        experts[task] = {
            'path': str(path),
            'success_rate': rate,
        }

    students = collect_students(student_args, missing)
    report, markdown = build_report(students, teachers, experts, missing)
    out_dir = Path(args.out_dir).expanduser()
    atomic_write_text(
        out_dir / 'decomposition.json', json.dumps(report, indent=2) + '\n'
    )
    atomic_write_text(out_dir / 'decomposition.md', markdown)
    print(f'wrote {out_dir / "decomposition.json"}', flush=True)
    print(f'wrote {out_dir / "decomposition.md"}', flush=True)
    if missing:
        print(f'{len(missing)} missing/skipped entries recorded:', flush=True)
        for item in missing:
            print(f'  - {item}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
