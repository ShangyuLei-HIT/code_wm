"""Compare alignment-mode ablation runs (identity / center-norm / refswap / similarity).

Reads finished multitask distillation runs plus their distillation caches and
fused codebooks, and produces comparison.json + comparison.md with:

* assignment preservation against the ORIGINAL per-task codebooks,
* cross-mode token-change agreement between caches sharing train splits,
* held-out alignment fit metrics (from the fit report JSON),
* per-run / per-(mode, variant) evaluation success statistics.

CPU-only; every missing artifact is recorded under ``missing`` instead of
silently dropped.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from stable_worldmodel.wm.vq_lewm.alignment import (
    SimilarityAlignment,
    load_similarity_alignment,
)
from stable_worldmodel.wm.vq_lewm.distillation import (
    load_codebook_weights,
    nearest_code_indices,
    resolve_weights_path,
)

MAX_SAMPLE_ROWS = 20000
QUERY_CHUNK = 8192
CODEBOOK_CHUNK = 2048
TRAIN_SPLIT = 'train'

VARIANT_PATTERN = re.compile(r'm0_unaligned|m2|m4|m5|m3|uot')
ALIGNMENT_PATTERN = re.compile(r'identity|center_norm|refswap_tworoom')
SEED_PATTERN = re.compile(r'seed(\d+)')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare alignment-mode ablation runs (CPU only).'
    )
    parser.add_argument(
        '--runs',
        action='append',
        default=[],
        metavar='NAME=RUN_DIR',
        help='run label and directory, e.g. '
        '--runs m2_identity_seed3072=.stablewm/multitask_distillation/'
        'pusht_tworoom_cube_m2_identity_seed3072 (repeatable)',
    )
    parser.add_argument(
        '--reference-cache',
        default='.stablewm/distillation_cache/pusht_tworoom_cube_fused',
        help='similarity-mode cache used as the comparison anchor',
    )
    parser.add_argument(
        '--original-codebooks',
        action='append',
        default=[],
        metavar='TASK=PATH',
        help='override an original per-task codebook checkpoint '
        '(default: tasks[].codebook_checkpoint from each run config)',
    )
    parser.add_argument(
        '--summary-subdir',
        default='task_evaluation',
        help=(
            'subdirectory of each run holding summary.json '
            '(e.g. task_evaluation_manifest)'
        ),
    )
    parser.add_argument(
        '--out-dir',
        default='results/alignment_modes',
        help='directory for comparison.json / comparison.md',
    )
    return parser.parse_args()


def parse_run_name(name: str) -> dict:
    """Infer variant / alignment mode / seed tokens from a run label."""
    variant = VARIANT_PATTERN.search(name)
    alignment = ALIGNMENT_PATTERN.search(name)
    seed = SEED_PATTERN.search(name)
    variant_token = variant.group(0) if variant else None
    alignment_token = alignment.group(0) if alignment else None
    if variant_token == 'm0_unaligned':
        alignment_label = 'unaligned'
    elif alignment_token is None:
        alignment_label = 'similarity'
    else:
        alignment_label = alignment_token
    return {
        'variant': variant_token,
        'alignment_token': alignment_token,
        'alignment_label': alignment_label,
        'seed': int(seed.group(1)) if seed else None,
        'group': f'{variant_token}|{alignment_label}',
        'warnings': [
            note
            for note in (
                None
                if variant_token
                else f'no variant token matched (expected one of {VARIANT_PATTERN.pattern})',
                None
                if seed
                else 'no seed(\\d+) token matched',
            )
            if note
        ],
    }


def resolve_config_path(value, anchor: Path | None) -> Path:
    """Resolve a possibly relative path; fall back to ancestors of anchor."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    here = Path.cwd() / path
    if here.exists() or anchor is None:
        return here
    base = anchor.expanduser().resolve()
    for parent in (base, *base.parents):
        candidate = parent / path
        if candidate.exists():
            return candidate.resolve()
    return here


def identity_alignment(dim: int) -> SimilarityAlignment:
    return SimilarityAlignment(
        torch.eye(dim), torch.tensor(1.0), torch.zeros(dim)
    )


def sample_indices(count: int, limit: int = MAX_SAMPLE_ROWS) -> np.ndarray:
    """Evenly strided row sampling, deterministic for a fixed count."""
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, count - 1, limit)).astype(np.int64))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def nearest_original_codes(
    latent: torch.Tensor, codebook: torch.Tensor
) -> torch.Tensor:
    """Exact nearest original-code indices, chunked over queries and codes."""
    outputs = []
    for start in range(0, len(latent), QUERY_CHUNK):
        chunk = latent[start : start + QUERY_CHUNK]
        outputs.append(
            nearest_code_indices(
                chunk, codebook, k=1, codebook_chunk_size=CODEBOOK_CHUNK
            ).squeeze(-1)
        )
    return torch.cat(outputs)


def record(missing: list, artifact: str, reason: str) -> None:
    missing.append({'artifact': artifact, 'reason': reason})


def dedupe_missing(missing: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for item in missing:
        key = (item['artifact'], item['reason'])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def load_fused_task_inverse(fused_checkpoint, task: str, missing: list):
    """Return ``(fused->original inverse map, fused codebook size)``.

    build_multitask_fused_codebook.py stores ``task_token_maps`` as
    ORIGINAL-index -> fused-index for both fusion methods (concat stores
    ``arange(K) + offset``; uot stores each task's merge map into the fused
    codebook), so the comparison needs the per-task INVERSE. Fused codes with
    no original counterpart for this task (codes contributed by other tasks)
    map to -1.
    """
    try:
        path = resolve_weights_path(fused_checkpoint)
    except Exception as error:  # noqa: BLE001 - recorded, never silent
        record(missing, f'fused codebook {fused_checkpoint}', repr(error))
        return None, 0
    if not path.exists():
        record(missing, f'fused codebook {path}', 'file does not exist')
        return None, 0
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as error:  # noqa: BLE001
        record(missing, f'fused codebook {path}', f'load failed: {error!r}')
        return None, 0
    maps = payload.get('task_token_maps') if isinstance(payload, dict) else None
    if not isinstance(maps, dict) or task not in maps:
        record(
            missing,
            f'fused codebook {path}',
            f'task_token_maps has no entry for task {task!r}',
        )
        return None, 0
    task_map = maps[task].detach().long().reshape(-1)
    if len(torch.unique(task_map)) != len(task_map):
        record(
            missing,
            f'fused codebook {path}',
            f'task_token_maps[{task!r}] is not injective; cannot invert it '
            'into a fused->original map',
        )
        return None, 0
    weights = payload.get('teacher.weight')
    if torch.is_tensor(weights) and weights.dim() >= 1:
        fused_size = int(weights.size(0))
    else:
        # Fallback: for both fusion methods every fused code is the image of
        # some task's original code, so the union of the map values bounds
        # the fused codebook size.
        candidates = [
            value.detach().long().reshape(-1)
            for value in maps.values()
            if torch.is_tensor(value) and value.numel()
        ]
        if not candidates:
            record(
                missing,
                f'fused codebook {path}',
                'payload has no teacher.weight and no usable '
                'task_token_maps values; cannot infer fused codebook size',
            )
            return None, 0
        fused_size = max(int(tensor.max()) for tensor in candidates) + 1
    if len(task_map) and (
        int(task_map.min()) < 0 or int(task_map.max()) >= fused_size
    ):
        record(
            missing,
            f'fused codebook {path}',
            f'task_token_maps[{task!r}] holds original->fused indices '
            f'outside [0, {fused_size}): '
            f'[{int(task_map.min())}, {int(task_map.max())}]',
        )
        return None, 0
    inverse = torch.full((fused_size,), -1, dtype=torch.long)
    inverse[task_map] = torch.arange(len(task_map), dtype=torch.long)
    return inverse, fused_size


def task_alignment(
    alignment_checkpoint, alignment_applied: bool, task: str, dim: int
) -> tuple[SimilarityAlignment | None, str]:
    """Resolve the transform to invert cached latents back to task space."""
    if not alignment_applied or alignment_checkpoint is None:
        return identity_alignment(dim), 'identity (alignment not applied)'
    try:
        transform = load_similarity_alignment(
            alignment_checkpoint, expected_dim=dim, source_task=task
        )
        return transform, 'bundle'
    except KeyError as error:
        # The bundle stores no transform for its own reference task.
        return identity_alignment(dim), f'identity ({error})'
    except Exception as error:  # noqa: BLE001
        return None, (
            f'cannot load alignment {alignment_checkpoint} for {task!r}: '
            f'{error!r}'
        )


class CacheAnalysis:
    """Per-cache artifacts shared by preservation and token-change checks."""

    def __init__(self, label: str, cache_dir: Path, missing: list):
        self.label = label
        self.cache_dir = cache_dir
        self.missing = missing
        self.metadata: dict | None = None
        self.cache_stats: dict | None = None
        self.fused_checkpoint = None
        self.alignment_checkpoint = None
        self.tasks: dict[str, dict] = {}

    def load_metadata(self) -> None:
        path = self.cache_dir / 'metadata.json'
        if not path.exists():
            record(self.missing, str(path), 'file does not exist')
            return
        try:
            self.metadata = json.loads(path.read_text())
            self.fused_checkpoint = self.metadata.get(
                'fused_codebook_checkpoint'
            )
            self.alignment_checkpoint = self.metadata.get(
                'alignment_checkpoint'
            )
        except Exception as error:  # noqa: BLE001
            record(self.missing, str(path), f'parse failed: {error!r}')

    def load_cache_stats(self) -> None:
        path = self.cache_dir / 'cache_stats.json'
        if not path.exists():
            record(
                self.missing,
                str(path),
                'file does not exist (cache built by an older script revision '
                'or statistics interrupted)',
            )
            return
        try:
            self.cache_stats = json.loads(path.read_text())
        except Exception as error:  # noqa: BLE001
            record(self.missing, str(path), f'parse failed: {error!r}')

    def analyze_task(
        self,
        task: str,
        alignment_applied: bool | None,
        alignment_flags: dict[str, bool] | None,
        original_codebook: Path | None,
    ) -> None:
        entry: dict = {}
        self.tasks[task] = entry
        if original_codebook is None:
            entry['preservation'] = None
            return
        prefix = self.cache_dir / 'tasks' / task
        tokens_path = prefix / f'{TRAIN_SPLIT}_hard_tokens.npy'
        latents_path = prefix / f'{TRAIN_SPLIT}_teacher_latents.npy'
        indices_path = prefix / f'{TRAIN_SPLIT}_indices.npy'
        for path in (tokens_path, latents_path):
            if not path.exists():
                record(self.missing, str(path), 'file does not exist')
                entry['preservation'] = None
                return
        entry['train_indices_sha256'] = (
            file_sha256(indices_path) if indices_path.exists() else None
        )
        if entry['train_indices_sha256'] is None:
            record(
                self.missing,
                str(indices_path),
                'file does not exist; cross-mode token pairs will be skipped',
            )
        tokens = np.load(tokens_path, mmap_mode='r')
        rows = sample_indices(len(tokens))
        latents = np.load(latents_path, mmap_mode='r')
        inverse_map, fused_size = load_fused_task_inverse(
            self.fused_checkpoint, task, self.missing
        )
        if inverse_map is None:
            entry['preservation'] = None
            return
        try:
            codebook = load_codebook_weights(original_codebook)
        except Exception as error:  # noqa: BLE001
            record(
                self.missing,
                f'original codebook {original_codebook}',
                f'load failed: {error!r}',
            )
            entry['preservation'] = None
            return
        dim = int(codebook.size(1))
        applied = (
            alignment_flags.get(task, True)
            if alignment_flags is not None
            else alignment_applied
        )
        transform, note = task_alignment(
            self.alignment_checkpoint, bool(applied), task, dim
        )
        entry['alignment_source'] = note
        if transform is None:
            record(self.missing, f'{self.label}/{task}', note)
            entry['preservation'] = None
            return
        latent = torch.from_numpy(
            np.ascontiguousarray(latents[rows])
        ).float().reshape(-1, dim)
        fused_tokens = torch.from_numpy(
            np.ascontiguousarray(tokens[rows])
        ).long().reshape(-1)
        if len(fused_tokens) == 0:
            record(
                self.missing,
                f'{self.label}/{task}',
                f'{TRAIN_SPLIT}_hard_tokens.npy has no rows to sample',
            )
            entry['preservation'] = None
            return
        if int(fused_tokens.min()) < 0 or int(fused_tokens.max()) >= fused_size:
            record(
                self.missing,
                f'{self.label}/{task}',
                f'cached fused tokens outside [0, {fused_size}): '
                f'[{int(fused_tokens.min())}, {int(fused_tokens.max())}]; '
                'cache and fused codebook disagree',
            )
            entry['preservation'] = None
            return
        mapped = inverse_map[fused_tokens]
        original_latent = transform.inverse(latent)
        nearest = nearest_original_codes(original_latent, codebook)
        entry.update(
            {
                'sampled_rows': int(len(rows)),
                'frames': int(len(fused_tokens)),
                'fused_codebook_size': int(fused_size),
                'original_codebook_size': int(codebook.size(0)),
                'frames_without_original_counterpart': int(
                    (mapped < 0).sum()
                ),
                'counterpart_fraction': float(
                    (mapped >= 0).float().mean()
                ),
                # Frames whose cached fused code is the fused image of the
                # nearest original code. Fused codes with no original
                # counterpart for this task (mapped == -1, i.e. codes
                # contributed by other tasks) count as NOT preserved.
                'preservation': float((mapped == nearest).float().mean()),
            }
        )
        self.tasks[task]['_fused'] = fused_tokens.numpy()
        self.tasks[task]['_mapped'] = mapped.numpy()


def load_run(name: str, run_dir: Path, missing: list) -> dict | None:
    run: dict = {
        'name': name,
        'run_dir': str(run_dir),
        'parsed': parse_run_name(name),
    }
    config_path = run_dir / 'config.yaml'
    if not config_path.exists():
        record(missing, str(config_path), 'run config does not exist')
        return run
    try:
        cfg = OmegaConf.load(config_path)
        run['cache_dir'] = str(
            resolve_config_path(cfg.paths.multitask_cache_dir, run_dir)
        )
        run['fused_codebook_checkpoint'] = str(cfg.paths.fused_codebook_checkpoint)
        run['alignment_checkpoint'] = str(cfg.paths.alignment_checkpoint)
        run['alignment_enabled'] = bool(cfg.alignment.get('enabled', True))
        run['alignment_mode'] = str(
            cfg.alignment.get('mode', 'similarity') or 'similarity'
        )
        run['tasks'] = {
            str(task.name): {
                'codebook_checkpoint': str(
                    resolve_config_path(task.codebook_checkpoint, run_dir)
                )
            }
            for task in cfg.tasks
        }
    except Exception as error:  # noqa: BLE001
        record(
            missing,
            str(config_path),
            f'failed to read required keys: {error!r}',
        )
        return run
    return run


def load_held_out(alignment_checkpoint: str | None, missing: list, artifact: str):
    if alignment_checkpoint is None:
        record(missing, artifact, 'no alignment checkpoint recorded')
        return None
    report_path = Path(alignment_checkpoint).expanduser().with_suffix('.json')
    if not report_path.exists():
        record(missing, str(report_path), 'alignment fit report does not exist')
        return None
    try:
        reports = json.loads(report_path.read_text())
    except Exception as error:  # noqa: BLE001
        record(missing, str(report_path), f'parse failed: {error!r}')
        return None
    held: dict[str, dict] = {}
    for task, task_report in reports.items():
        validation = (
            task_report.get('validation', {}) if isinstance(task_report, dict) else {}
        )
        held[task] = {
            key: validation.get(key)
            for key in (
                'mse',
                'rmse',
                'normalized_rmse',
                'r2',
                'cosine_similarity',
                'mse_improvement_ratio',
            )
        }
        # Norm diagnostics live at the top level of each per-task report, not
        # inside 'validation' (see fit_multitask_latent_alignments.py).
        for key in (
            'source_mean_norm',
            'reference_mean_norm',
            'mapped_mean_norm',
        ):
            held[task][key] = (
                task_report.get(key) if isinstance(task_report, dict) else None
            )
    return held


def load_success(
    run_dir: Path,
    missing: list,
    name: str,
    summary_subdir: str = 'task_evaluation',
):
    path = run_dir / summary_subdir / 'summary.json'
    if not path.exists():
        record(
            missing,
            str(path),
            'evaluation summary does not exist (run not evaluated yet?)',
        )
        return None
    try:
        summary = json.loads(path.read_text())
    except Exception as error:  # noqa: BLE001
        record(missing, str(path), f'parse failed: {error!r}')
        return None
    per_task = {}
    for task in summary.get('tasks', []):
        metrics = task.get('metrics') or {}
        value = metrics.get('success_rate')
        if value is None:
            record(
                missing,
                f'{path}[{task.get("task")}]',
                'metrics.success_rate is absent',
            )
        per_task[str(task.get('task'))] = (
            None if value is None else float(value)
        )
    values = {name: value for name, value in per_task.items() if value is not None}
    worst = min(values.items(), key=lambda item: item[1]) if values else (None, None)
    return {
        'per_task': per_task,
        'macro_success': (
            float(np.mean(list(values.values()))) if values else None
        ),
        'worst_task': {'task': worst[0], 'success_rate': worst[1]},
        'num_tasks_with_metrics': len(values),
    }


def group_statistics(runs: list[dict]) -> dict:
    groups: dict[str, dict] = {}
    for run in runs:
        success = run.get('success') or {}
        macro = success.get('macro_success')
        parsed = run['parsed']
        group = groups.setdefault(
            parsed['group'],
            {'runs': [], 'seeds': [], 'macro': [], 'worst': []},
        )
        group['runs'].append(run['name'])
        seed = parsed['seed']
        if seed is None:
            # The label may omit the seed token; fall back to the run
            # directory name (always carries _seed<N> from apply_train_seed).
            directory_seed = SEED_PATTERN.search(run.get('run_dir', ''))
            seed = (
                int(directory_seed.group(1)) if directory_seed else None
            )
        if seed is not None:
            group['seeds'].append(seed)
        if macro is not None:
            group['macro'].append(macro)
        worst = success.get('worst_task') or {}
        if worst.get('success_rate') is not None:
            group['worst'].append(worst['success_rate'])
    payload = {}
    for key, group in groups.items():
        payload[key] = {
            'runs': group['runs'],
            'seeds': group['seeds'],
            'num_seeds': len(set(group['seeds'])),
            'macro_success_mean': (
                float(np.mean(group['macro'])) if group['macro'] else None
            ),
            'macro_success_std': (
                float(np.std(group['macro'])) if group['macro'] else None
            ),
            'worst_task_success_mean': (
                float(np.mean(group['worst'])) if group['worst'] else None
            ),
            'worst_task_success_std': (
                float(np.std(group['worst'])) if group['worst'] else None
            ),
        }
    return payload


def build_token_change(
    caches: dict[str, CacheAnalysis],
    reference: CacheAnalysis,
    all_tasks: list[str],
):
    """Pairwise token agreement between every distinct cache.

    The reference cache is skipped when it coincides with one of the run
    caches (same directory), because the pair would be trivially identical.
    """
    merged = dict(caches)
    if str(reference.cache_dir) not in merged:
        merged[str(reference.cache_dir)] = reference
    token_change: dict[str, dict] = {}
    skipped: list[dict] = []
    for task in all_tasks:
        pairs: dict[str, dict] = {}
        labels = sorted(merged)
        for first in range(len(labels)):
            for second in range(first + 1, len(labels)):
                one, two = labels[first], labels[second]
                one_entry = merged[one].tasks.get(task)
                two_entry = merged[two].tasks.get(task)
                pair = f'{merged[one].label}|{merged[two].label}'
                if not one_entry or not two_entry:
                    continue
                if '_fused' not in one_entry or '_fused' not in two_entry:
                    continue
                key_one = one_entry.get('train_indices_sha256')
                key_two = two_entry.get('train_indices_sha256')
                if key_one is None or key_two is None or key_one != key_two:
                    skipped.append(
                        {
                            'task': task,
                            'pair': pair,
                            'reason': 'train split indices differ or are '
                            'unavailable',
                        }
                    )
                    continue
                if one_entry.get('sampled_rows') != two_entry.get(
                    'sampled_rows'
                ):
                    skipped.append(
                        {
                            'task': task,
                            'pair': pair,
                            'reason': 'different sampled row counts',
                        }
                    )
                    continue
                fused_one = one_entry['_fused']
                fused_two = two_entry['_fused']
                mapped_one = one_entry['_mapped']
                mapped_two = two_entry['_mapped']
                pairs[pair] = {
                    'frames': int(len(fused_one)),
                    'rows': int(one_entry['sampled_rows']),
                    'direct_fused_token_agreement': float(
                        (fused_one == fused_two).mean()
                    ),
                    # -1 means the fused code has no original counterpart
                    # for this task, which is not an original code: agreement
                    # requires an actual counterpart on both sides.
                    'both_map_to_same_original_code': float(
                        (
                            (mapped_one == mapped_two) & (mapped_one >= 0)
                        ).mean()
                    ),
                    'both_no_original_counterpart': float(
                        ((mapped_one < 0) & (mapped_two < 0)).mean()
                    ),
                }
        token_change[task] = pairs
    return token_change, skipped


def percentage(value) -> str:
    if value is None:
        return 'n/a'
    return f'{100.0 * float(value):.1f}%'


def decimal(value, digits: int = 4) -> str:
    if value is None:
        return 'n/a'
    return f'{float(value):.{digits}f}'


def cell(text: str) -> str:
    """Escape pipe characters so they cannot break markdown tables."""
    return text.replace('|', '\\|')


def markdown_report(payload: dict) -> str:
    lines: list[str] = []
    lines.append('# Alignment-mode comparison')
    lines.append('')
    lines.append(f'Generated: {payload["generated_utc"]}')
    lines.append('')
    lines.append(
        'Reference cache: '
        f'`{payload["arguments"]["reference_cache"]}`'
    )
    lines.append('')

    lines.append('## Success by mode (macro / worst task)')
    lines.append('')
    lines.append(
        '| group (variant\\|alignment) | seeds | macro mean | macro std | '
        'worst-task mean | worst-task std |'
    )
    lines.append('|---|---|---|---|---|---|')
    for group, stats in sorted(payload['groups'].items()):
        lines.append(
            f'| {cell(group)} | {stats["num_seeds"]} '
            f'({",".join(str(s) for s in sorted(set(stats["seeds"]))) or "n/a"}) '
            f'| {decimal(stats["macro_success_mean"])} '
            f'| {decimal(stats["macro_success_std"], 3)} '
            f'| {decimal(stats["worst_task_success_mean"])} '
            f'| {decimal(stats["worst_task_success_std"], 3)} |'
        )
    lines.append('')

    lines.append('## Assignment preservation vs original codebooks')
    lines.append('')
    tasks = payload['task_order']
    header = '| cache / run | ' + ' | '.join(tasks) + ' |'
    lines.append(header)
    lines.append('|---' * (len(tasks) + 1) + '|')
    for cache in sorted(payload['caches'].values(), key=lambda c: c['label']):
        row = [f'`{cell(cache["label"])}`']
        for task in tasks:
            entry = cache.get('tasks', {}).get(task, {})
            row.append(percentage(entry.get('preservation')))
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')

    lines.append('## Cross-mode token change (train split)')
    lines.append('')
    for task in tasks:
        lines.append(f'### Task `{task}`')
        lines.append('')
        pairs = payload['token_change'].get(task, {})
        if not pairs:
            lines.append('_no comparable cache pairs_')
            lines.append('')
            continue
        lines.append(
            '| cache pair | direct fused-token agreement | '
            'both map to same original code | both no counterpart | frames |'
        )
        lines.append('|---|---|---|---|---|')
        for pair, values in sorted(pairs.items()):
            lines.append(
                f'| {cell(pair)} | '
                f'{percentage(values["direct_fused_token_agreement"])} '
                f'| {percentage(values["both_map_to_same_original_code"])} '
                f'| {percentage(values["both_no_original_counterpart"])} '
                f'| {values["frames"]:,} |'
            )
        lines.append('')

    lines.append('## Held-out alignment fit (validation anchors)')
    lines.append('')
    lines.append('| run | source task | mse | cosine | r2 | improvement |')
    lines.append('|---|---|---|---|---|---|')
    for name in sorted(payload['held_out']):
        held = payload['held_out'][name]
        if held is None:
            lines.append(f'| {cell(name)} | - | missing | - | - | - |')
            continue
        for task, values in sorted(held.items()):
            lines.append(
                f'| {cell(name)} | {task} | {decimal(values.get("mse"))} '
                f'| {decimal(values.get("cosine_similarity"))} '
                f'| {decimal(values.get("r2"))} '
                f'| {decimal(values.get("mse_improvement_ratio"))} |'
            )
    lines.append('')

    lines.append('## Cache statistics (train split)')
    lines.append('')
    lines.append('| cache | task | frames | top1 sqdist median | temperature effective | latent norm mean |')
    lines.append('|---|---|---|---|---|---|')
    for cache in sorted(payload['caches'].values(), key=lambda c: c['label']):
        stats = cache.get('cache_stats')
        label = cell(cache['label'])
        if not stats:
            lines.append(f'| `{label}` | - | - | missing | - | - |')
            continue
        for task in tasks:
            entry = (stats.get('tasks') or {}).get(task, {}).get(TRAIN_SPLIT)
            if entry is None:
                lines.append(f'| `{label}` | {task} | - | missing | - | - |')
                continue
            lines.append(
                f'| `{label}` | {task} | {entry["frames"]:,} '
                f'| {decimal(entry["top1_sqdist_median"])} '
                f'| {decimal(entry["temperature_effective"], 2)} '
                f'| {decimal(entry["latent_norm_mean"], 3)} |'
            )
    lines.append('')

    lines.append('## Missing artifacts')
    lines.append('')
    if not payload['missing']:
        lines.append('_none_')
    else:
        for item in payload['missing']:
            lines.append(
                f'- `{item["artifact"]}`: {item["reason"]}'
            )
    lines.append('')
    return '\n'.join(lines)


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    missing: list[dict] = []
    codebook_overrides: dict[str, str] = {}
    for item in args.original_codebooks:
        task, separator, path = item.partition('=')
        if not separator or not task or not path:
            raise ValueError(
                f'--original-codebooks expects TASK=PATH, got {item!r}'
            )
        codebook_overrides[task] = path

    runs: list[dict] = []
    for item in args.runs:
        name, separator, directory = item.partition('=')
        if not separator or not name or not directory:
            raise ValueError(f'--runs expects NAME=RUN_DIR, got {item!r}')
        runs.append(
            load_run(name, Path(directory).expanduser(), missing)
        )

    # The reference cache is analyzed as an anchor pseudo-run.
    reference_dir = Path(args.reference_cache).expanduser()
    reference_cache = CacheAnalysis('reference_cache', reference_dir, missing)
    if not reference_dir.exists():
        record(missing, str(reference_dir), 'reference cache does not exist')
    reference_cache.load_metadata()
    reference_cache.load_cache_stats()
    reference_alignment_flags: dict[str, bool] | None = None
    if reference_cache.metadata is not None:
        fusion = reference_cache.metadata.get('fusion_metadata') or {}
        fusion_tasks = fusion.get('tasks')
        if isinstance(fusion_tasks, list) and fusion_tasks:
            reference_alignment_flags = {
                str(task.get('name')): bool(task.get('alignment_applied'))
                for task in fusion_tasks
                if isinstance(task, dict) and task.get('name')
            }

    caches: dict[str, CacheAnalysis] = {}
    cache_labels: dict[str, list[str]] = {}

    for run in runs:
        name = run['name']
        cache_key = run.get('cache_dir')
        if cache_key is None:
            continue
        if cache_key not in caches:
            cache = CacheAnalysis(f'{name} [cache]', Path(cache_key), missing)
            if not Path(cache_key).exists():
                record(missing, cache_key, 'cache directory does not exist')
            cache.load_metadata()
            cache.load_cache_stats()
            if cache.fused_checkpoint is None:
                cache.fused_checkpoint = run.get('fused_codebook_checkpoint')
            if cache.alignment_checkpoint is None:
                cache.alignment_checkpoint = run.get('alignment_checkpoint')
            caches[cache_key] = cache
            cache_labels[cache_key] = []
        cache_labels[cache_key].append(name)
        cache = caches[cache_key]
        applied = bool(run.get('alignment_enabled', True))
        for task, info in (run.get('tasks') or {}).items():
            original = codebook_overrides.get(task, info['codebook_checkpoint'])
            if task in cache.tasks:
                continue
            cache.analyze_task(
                task,
                alignment_applied=applied,
                alignment_flags=None,
                original_codebook=Path(original)
                if original
                else None,
            )
            if original is None:
                record(
                    missing,
                    f'{name}/{task}',
                    'no original codebook checkpoint resolved',
                )

    if reference_dir.exists():
        reference_tasks: dict[str, str] = {}
        if reference_cache.metadata is not None:
            for task in reference_cache.metadata.get('tasks', []):
                checkpoint = task.get('original_codebook_checkpoint')
                if checkpoint:
                    reference_tasks[str(task['name'])] = checkpoint
        if not reference_tasks:
            record(
                missing,
                f'{reference_dir}/metadata.json',
                'no task original_codebook_checkpoint entries',
            )
        for task, checkpoint in reference_tasks.items():
            if task in codebook_overrides:
                checkpoint = codebook_overrides[task]
            reference_cache.analyze_task(
                task,
                alignment_applied=False,
                alignment_flags=reference_alignment_flags,
                original_codebook=Path(checkpoint)
                if checkpoint
                else None,
            )

    task_order: list[str] = []
    for run in runs:
        for task in run.get('tasks') or {}:
            if task not in task_order:
                task_order.append(task)
    for task in reference_cache.tasks:
        if task not in task_order:
            task_order.append(task)

    for run in runs:
        run['held_out'] = load_held_out(
            run.get('alignment_checkpoint'),
            missing,
            f'runs/{run["name"]}/alignment_report',
        )
        run['success'] = (
            load_success(
                Path(run['run_dir']),
                missing,
                run['name'],
                summary_subdir=args.summary_subdir,
            )
            if run.get('run_dir')
            else None
        )

    token_change, skipped_pairs = build_token_change(
        caches,
        reference_cache,
        task_order,
    )

    reference_held_out = load_held_out(
        reference_cache.alignment_checkpoint,
        missing,
        'reference_cache/alignment_report',
    )

    payload = {
        'generated_utc': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        'arguments': {
            'runs': [item for item in args.runs],
            'reference_cache': str(reference_dir),
            'original_codebooks': codebook_overrides,
            'out_dir': str(args.out_dir),
            'max_sample_rows': MAX_SAMPLE_ROWS,
            'query_chunk': QUERY_CHUNK,
            'codebook_chunk': CODEBOOK_CHUNK,
        },
        'task_order': task_order,
        'runs': {
            run['name']: {
                'run_dir': run.get('run_dir'),
                'parsed': run['parsed'],
                'cache_dir': run.get('cache_dir'),
                'alignment_checkpoint': run.get('alignment_checkpoint'),
                'alignment_mode': run.get('alignment_mode'),
                'alignment_enabled': run.get('alignment_enabled'),
                'fused_codebook_checkpoint': run.get(
                    'fused_codebook_checkpoint'
                ),
                'tasks': run.get('tasks'),
                'held_out': run.get('held_out'),
                'success': run.get('success'),
            }
            for run in runs
        },
        'caches': {
            cache_key: {
                'label': cache.label,
                'runs': cache_labels.get(cache_key, []),
                'metadata_present': cache.metadata is not None,
                'fused_codebook_checkpoint': cache.fused_checkpoint,
                'alignment_checkpoint': cache.alignment_checkpoint,
                'alignment_mode': (
                    cache.metadata.get('alignment_mode')
                    if cache.metadata
                    else None
                ),
                'reference_task': (
                    cache.metadata.get('reference_task')
                    if cache.metadata
                    else None
                ),
                'cache_stats': cache.cache_stats,
                'tasks': {
                    task: {
                        key: value
                        for key, value in entry.items()
                        if not key.startswith('_')
                    }
                    for task, entry in cache.tasks.items()
                },
            }
            for cache_key, cache in caches.items()
        },
        'reference_cache': {
            'cache_dir': str(reference_dir),
            'metadata_present': reference_cache.metadata is not None,
            'alignment_flags': reference_alignment_flags,
            'alignment_checkpoint': reference_cache.alignment_checkpoint,
            'held_out': reference_held_out,
            'cache_stats': reference_cache.cache_stats,
            'tasks': {
                task: {
                    key: value
                    for key, value in entry.items()
                    if not key.startswith('_')
                }
                for task, entry in reference_cache.tasks.items()
            },
        },
        'preservation': {
            f'{cache.label}::{task}': entry.get('preservation')
            for cache in caches.values()
            for task, entry in cache.tasks.items()
        },
        'token_change': token_change,
        'skipped_token_pairs': skipped_pairs,
        'held_out': {run['name']: run.get('held_out') for run in runs},
        'success': {run['name']: run.get('success') for run in runs},
        'groups': group_statistics(runs),
        'missing': dedupe_missing(missing),
    }
    payload['preservation'].update(
        {
            f'reference_cache::{task}': entry.get('preservation')
            for task, entry in reference_cache.tasks.items()
        }
    )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'comparison.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n'
    )
    (out_dir / 'comparison.md').write_text(markdown_report(payload))
    print(
        f'Wrote {out_dir}/comparison.json and {out_dir}/comparison.md '
        f'({len(payload["missing"])} missing artifacts recorded)',
        flush=True,
    )


if __name__ == '__main__':
    main()
