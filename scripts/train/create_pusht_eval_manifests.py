"""Create disjoint, immutable PushT selection and held-out manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import stable_worldmodel as swm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='galilai-group/lewm-pusht')
    parser.add_argument('--dataset-cache', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--goal-offset-steps', type=int, default=25)
    parser.add_argument('--selection-seed', type=int, default=42)
    parser.add_argument('--selection-count', type=int, default=50)
    parser.add_argument('--test-seed', type=int, default=4242)
    parser.add_argument('--test-count', type=int, default=200)
    parser.add_argument('--shard-size', type=int, default=50)
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


def write_or_validate(value: dict, path: Path) -> None:
    if path.is_file():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f'conflicting evaluation manifest: {path}')
        return
    atomic_json_dump(value, path)


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
    shard_index: int | None = None,
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
    if shard_index is not None:
        result['shard_index'] = shard_index
    return result


def main():
    args = parse_args()
    if args.selection_count < 1 or args.test_count < 1:
        raise ValueError('manifest counts must be positive')
    if args.shard_size < 1 or args.test_count % args.shard_size:
        raise ValueError('test count must be divisible by shard size')

    os.environ['LOCAL_DATASET_DIR'] = str(
        Path(args.dataset_cache).expanduser().resolve()
    )
    os.environ['STABLEWM_HOME'] = os.environ['LOCAL_DATASET_DIR']
    dataset = swm.data.load_dataset(
        args.dataset,
        cache_dir=args.dataset_cache,
        keys_to_cache=['action', 'proprio', 'state'],
    )
    episode_key = episode_column(dataset)
    episode_ids = dataset.get_col_data(episode_key)
    step_ids = dataset.get_col_data('step_idx')
    unique_episodes = np.unique(episode_ids)
    maximum_starts = {}
    for episode in unique_episodes:
        maximum_starts[episode] = (
            int(step_ids[episode_ids == episode].max())
            + 1
            - args.goal_offset_steps
            - 1
        )
    max_start_per_row = np.asarray(
        [maximum_starts[episode] for episode in episode_ids]
    )
    valid_rows = np.flatnonzero(step_ids <= max_start_per_row)

    selection_rng = np.random.default_rng(args.selection_seed)
    selection_positions = selection_rng.choice(
        len(valid_rows), size=args.selection_count, replace=False
    )
    selection_rows = np.sort(valid_rows[selection_positions])

    remaining_rows = np.setdiff1d(
        valid_rows, selection_rows, assume_unique=True
    )
    test_rng = np.random.default_rng(args.test_seed)
    test_positions = test_rng.choice(
        len(remaining_rows), size=args.test_count, replace=False
    )
    test_rows = np.sort(remaining_rows[test_positions])
    if np.intersect1d(selection_rows, test_rows).size:
        raise RuntimeError('selection and test manifests overlap')

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = make_manifest(
        kind='selection',
        dataset_name=args.dataset,
        rows=selection_rows,
        episode_key=episode_key,
        dataset=dataset,
        seed=args.selection_seed,
        goal_offset_steps=args.goal_offset_steps,
    )
    selection_path = output_dir / (
        f'pusht_selection_seed{args.selection_seed}_'
        f'n{args.selection_count}.json'
    )
    write_or_validate(selection, selection_path)

    test_parent_hash = array_hash(test_rows)
    shard_paths = []
    for shard_index, start in enumerate(
        range(0, len(test_rows), args.shard_size)
    ):
        shard_rows = test_rows[start : start + args.shard_size]
        shard = make_manifest(
            kind='heldout_test_shard',
            dataset_name=args.dataset,
            rows=shard_rows,
            episode_key=episode_key,
            dataset=dataset,
            seed=args.test_seed,
            goal_offset_steps=args.goal_offset_steps,
            parent_hash=test_parent_hash,
            shard_index=shard_index,
        )
        shard_path = output_dir / (
            f'pusht_test_seed{args.test_seed}_n{args.test_count}_'
            f'shard{shard_index:02d}.json'
        )
        write_or_validate(shard, shard_path)
        shard_paths.append(str(shard_path))

    summary = {
        'format_version': 1,
        'dataset': args.dataset,
        'valid_start_count': len(valid_rows),
        'selection_manifest': str(selection_path),
        'selection_row_indices_sha256': selection['row_indices_sha256'],
        'test_seed': args.test_seed,
        'test_count': args.test_count,
        'test_row_indices_sha256': test_parent_hash,
        'test_shards': shard_paths,
        'disjoint': True,
    }
    summary_path = output_dir / 'manifest_summary.json'
    write_or_validate(summary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
