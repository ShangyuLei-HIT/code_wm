"""Create disjoint, immutable selection/test start manifests for every task.

Generalizes ``create_pusht_eval_manifests.py`` from the single PushT dataset
to all tasks of a multitask config. For each task it loads the evaluation
dataset exactly like the PushT creator (``swm.data.load_dataset`` with
``LOCAL_DATASET_DIR``/``STABLEWM_HOME`` pointing at the config's
``paths.dataset_cache``), computes the valid start rows with the same
formula ``scripts/plan/eval_wm.py`` uses
(``step_idx <= episode_len - goal_offset_steps - 1``), then samples two
disjoint row sets:

- selection manifest: seed ``--selection-seed`` (default 42), count
  ``--selection-count`` (default 50), rows sorted;
- held-out test manifest: seed ``--test-seed`` (default 4242), count
  ``--test-count`` (default 100), sampled from the valid rows *excluding*
  the selection rows, sorted.

Manifest entry format is identical to the PushT creator
(``row_index``/``episode_idx``/``start_step``), so ``eval_wm.py`` consumes
the files through ``eval.manifest_path`` unchanged. Unlike the PushT
creator the test set is written as a single unsharded file per task
(``<task>_test_seed<N>_n<M>.json``): the multitask evaluator pins
``eval.num_eval = len(entries)`` and therefore needs exactly one manifest
per task and split. ``--shard-size`` is kept (and validated) only for
interface parity with the PushT creator.

The whole procedure is deterministic: a pure function of the seeds and the
datasets. Existing files are never overwritten; a conflicting rewrite
raises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create per-task selection/test eval start manifests.'
    )
    parser.add_argument(
        '--config',
        default='scripts/train/config/multitask_vq_lewm_six_tasks.yaml',
        help=(
            'multitask config whose tasks[] provide name + dataset; '
            'dataset references may use ${paths.dataset_cache} tokens'
        ),
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help=(
            'output directory (default: '
            '<paths.dataset_cache>/evaluation_manifests/multitask_v1)'
        ),
    )
    parser.add_argument('--goal-offset-steps', type=int, default=25)
    parser.add_argument('--selection-seed', type=int, default=42)
    parser.add_argument('--selection-count', type=int, default=50)
    parser.add_argument('--test-seed', type=int, default=4242)
    parser.add_argument('--test-count', type=int, default=100)
    parser.add_argument(
        '--shard-size',
        type=int,
        default=50,
        help=(
            'retained for interface parity with the PushT creator; '
            'manifests are written unsharded (test_count must still be '
            'divisible by it)'
        ),
    )
    return parser.parse_args()


def episode_column(dataset) -> str:
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=np.int64).tobytes()
    ).hexdigest()


def entries(dataset, rows: np.ndarray, episode_key: str) -> list[dict]:
    episodes = dataset.get_col_data(episode_key)
    steps = dataset.get_col_data('step_idx')
    return [
        {
            'row_index': int(row),
            'episode_idx': int(episodes[row]),
            'start_step': int(steps[row]),
        }
        for row in rows
    ]


def atomic_json_dump(value: dict, path: Path) -> None:
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def manifest_text(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + '\n'


def write_or_validate(value: dict, path: Path) -> str:
    """Write the manifest (or verify an identical copy exists).

    Returns the sha256 of the canonical file content so callers can record
    it without re-reading the file.
    """
    text = manifest_text(value)
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if path.is_file():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f'conflicting evaluation manifest: {path}')
        return digest
    atomic_json_dump(value, path)
    return digest


def resolve_dataset_reference(raw: str, cache_root: Path) -> str:
    """Resolve ``${paths.dataset_cache}`` / ``${oc.env:...}`` tokens.

    Reading the value through OmegaConf already expands these for a single
    loaded file; this manual pass is a guarded fallback for leftovers
    (e.g. configs assembled from multiple sources) and never expands
    unknown tokens silently.
    """
    text = str(raw)
    env_pattern = re.compile(
        r'\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}'
    )
    for _ in range(8):
        updated = text.replace('${paths.dataset_cache}', str(cache_root))

        def substitute(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            value = os.environ.get(name)
            if value is None:
                return default if default is not None else match.group(0)
            return value

        updated = env_pattern.sub(substitute, updated)
        if updated == text:
            break
        text = updated
    if '${' in text:
        raise ValueError(
            f'cannot resolve dataset reference {raw!r}: {text!r} still '
            'contains ${...} tokens; set STABLEWM_HOME or use a config '
            'without unresolvable placeholders'
        )
    return text


def absolutize_local(name: str) -> str:
    """Absolutize relative local file paths (e.g. './.stablewm/...h5').

    ``swm.data.load_dataset`` resolves relative names against its own
    ``<cache_dir>/datasets`` directory, so a config-expanded relative path
    would be looked up in the wrong place. HF repo ids (never existing
    files) pass through unchanged.
    """
    candidate = Path(name).expanduser()
    if not candidate.is_absolute() and candidate.is_file():
        return str(candidate.resolve())
    return name


def make_manifest(
    *,
    kind: str,
    dataset_name: str,
    rows: np.ndarray,
    episode_key: str,
    dataset,
    seed: int,
    goal_offset_steps: int,
    parent_hash: str | None = None,
) -> dict:
    result = {
        'format_version': 1,
        'kind': kind,
        'dataset': dataset_name,
        'seed': seed,
        'goal_offset_steps': goal_offset_steps,
        'count': len(rows),
        'row_indices_sha256': array_hash(rows),
        'entries': entries(dataset, rows, episode_key),
    }
    if parent_hash is not None:
        result['parent_row_indices_sha256'] = parent_hash
    return result


def valid_start_rows(dataset, goal_offset_steps: int):
    """Same valid-row formula as ``scripts/plan/eval_wm.py``."""
    episode_key = episode_column(dataset)
    episode_ids = dataset.get_col_data(episode_key)
    step_ids = dataset.get_col_data('step_idx')
    unique_episodes = np.unique(episode_ids)
    maximum_starts = {}
    for episode in unique_episodes:
        maximum_starts[episode] = (
            int(step_ids[episode_ids == episode].max())
            + 1
            - goal_offset_steps
            - 1
        )
    max_start_per_row = np.asarray(
        [maximum_starts[episode] for episode in episode_ids]
    )
    valid_rows = np.flatnonzero(step_ids <= max_start_per_row)
    return episode_key, episode_ids, valid_rows


def process_task(
    *,
    task,
    cache_root: Path,
    goal_offset_steps: int,
    selection_seed: int,
    selection_count: int,
    test_seed: int,
    test_count: int,
) -> dict:
    name = str(task.name)
    dataset_name = absolutize_local(
        resolve_dataset_reference(str(task.dataset), cache_root)
    )
    # Load exactly like the PushT creator: env vars + swm.data.load_dataset.
    # No keys_to_cache — only the episode/step index columns are read and
    # per-task action/proprio/state columns differ across the six tasks.
    dataset = swm.data.load_dataset(dataset_name, keys_to_cache=[])
    episode_key, episode_ids, valid_rows = valid_start_rows(
        dataset, goal_offset_steps
    )
    if len(valid_rows) < selection_count:
        raise ValueError(
            f'task {name}: only {len(valid_rows)} valid start rows, need '
            f'{selection_count} for the selection manifest'
        )
    selection_rng = np.random.default_rng(selection_seed)
    selection_positions = selection_rng.choice(
        len(valid_rows), size=selection_count, replace=False
    )
    selection_rows = np.sort(valid_rows[selection_positions])

    remaining_rows = np.setdiff1d(
        valid_rows, selection_rows, assume_unique=True
    )
    if len(remaining_rows) < test_count:
        raise ValueError(
            f'task {name}: only {len(remaining_rows)} valid start rows left '
            f'after the selection draw, need {test_count} for the disjoint '
            'test manifest'
        )
    test_rng = np.random.default_rng(test_seed)
    test_positions = test_rng.choice(
        len(remaining_rows), size=test_count, replace=False
    )
    test_rows = np.sort(remaining_rows[test_positions])
    if np.intersect1d(selection_rows, test_rows).size:
        raise RuntimeError(
            f'task {name}: selection and test manifests overlap'
        )
    return {
        'name': name,
        'dataset': dataset_name,
        'episode_column': episode_key,
        'episode_ids': episode_ids,
        'selection_rows': selection_rows,
        'test_rows': test_rows,
        'valid_rows': valid_rows,
        'dataset_handle': dataset,
    }


def main():
    args = parse_args()
    if args.selection_count < 1 or args.test_count < 1:
        raise ValueError('manifest counts must be positive')
    if args.goal_offset_steps < 0:
        raise ValueError('goal offset steps must be non-negative')
    if args.shard_size < 1 or args.test_count % args.shard_size:
        raise ValueError('test count must be divisible by shard size')

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f'config not found: {config_path}')
    cfg = OmegaConf.load(config_path)
    try:
        raw_cache = str(cfg.paths.dataset_cache)
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f'{config_path} has no paths.dataset_cache entry'
        ) from exc
    cache_root = Path(resolve_dataset_reference(raw_cache, Path.cwd()))
    if not cache_root.is_absolute():
        cache_root = (Path.cwd() / cache_root).resolve()
    cache_root = cache_root.expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else cache_root / 'evaluation_manifests' / 'multitask_v1'
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ['LOCAL_DATASET_DIR'] = str(cache_root)
    os.environ['STABLEWM_HOME'] = str(cache_root)

    tasks_summary = {}
    for task in cfg.tasks:
        processed = process_task(
            task=task,
            cache_root=cache_root,
            goal_offset_steps=args.goal_offset_steps,
            selection_seed=args.selection_seed,
            selection_count=args.selection_count,
            test_seed=args.test_seed,
            test_count=args.test_count,
        )
        name = processed['name']
        dataset = processed['dataset_handle']
        selection = make_manifest(
            kind='selection',
            dataset_name=processed['dataset'],
            rows=processed['selection_rows'],
            episode_key=processed['episode_column'],
            dataset=dataset,
            seed=args.selection_seed,
            goal_offset_steps=args.goal_offset_steps,
        )
        selection_path = out_dir / (
            f'{name}_selection_seed{args.selection_seed}_'
            f'n{args.selection_count}.json'
        )
        selection_digest = write_or_validate(selection, selection_path)

        test = make_manifest(
            kind='heldout_test',
            dataset_name=processed['dataset'],
            rows=processed['test_rows'],
            episode_key=processed['episode_column'],
            dataset=dataset,
            seed=args.test_seed,
            goal_offset_steps=args.goal_offset_steps,
            parent_hash=array_hash(processed['test_rows']),
        )
        test_path = out_dir / (
            f'{name}_test_seed{args.test_seed}_n{args.test_count}.json'
        )
        test_digest = write_or_validate(test, test_path)

        tasks_summary[name] = {
            'dataset': processed['dataset'],
            'episode_column': processed['episode_column'],
            'dataset_rows': int(len(processed['episode_ids'])),
            'dataset_episodes': int(
                len(np.unique(processed['episode_ids']))
            ),
            'valid_start_count': int(len(processed['valid_rows'])),
            'selection': {
                'seed': args.selection_seed,
                'count': args.selection_count,
                'path': str(selection_path),
                'sha256': selection_digest,
                'row_indices_sha256': selection['row_indices_sha256'],
            },
            'test': {
                'seed': args.test_seed,
                'count': args.test_count,
                'path': str(test_path),
                'sha256': test_digest,
                'row_indices_sha256': test['row_indices_sha256'],
            },
            'disjoint': True,
        }
        print(
            f'{name}: valid={len(processed["valid_rows"])} '
            f'rows={len(processed["episode_ids"])} | selection '
            f'n{args.selection_count} -> {selection_path.name} | test '
            f'n{args.test_count} -> {test_path.name}',
            flush=True,
        )

    summary = {
        'format_version': 1,
        'config': str(config_path),
        'dataset_cache': str(cache_root),
        'goal_offset_steps': args.goal_offset_steps,
        'selection_seed': args.selection_seed,
        'selection_count': args.selection_count,
        'test_seed': args.test_seed,
        'test_count': args.test_count,
        'shard_size': args.shard_size,
        'sharded': False,
        'tasks': tasks_summary,
    }
    summary_path = out_dir / 'manifest_summary.json'
    write_or_validate(summary, summary_path)
    print(f'manifests written to {out_dir}', flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
