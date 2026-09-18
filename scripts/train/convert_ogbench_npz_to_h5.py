"""Replay OGBench state npz datasets into the SWM HDF5 pixel schema.

OGBench ships scene / humanoidmaze datasets as flat ``.npz`` files
(``observations``, ``actions``, ``terminals``, ``qpos``, ``qvel`` and for
scene ``button_states``) without pixels and without per-step goal columns.
The six-task fusion pipeline consumes the same HDF5 layout as
``cube_single_expert.h5`` (per-step columns + ``ep_len``/``ep_offset``).

This script replays every ``qpos``/``qvel`` row through the matching SWM
environment (``set_state`` on ``env.unwrapped``) and renders 224x224 pixels,
deriving the goal columns the replay evaluation callables need:

- scene: ``privileged_block_0_pos`` (qpos[14:17]), ``privileged_block_0_quat``
  (qpos[17:21]), ``privileged_button_{0,1}_state`` (button_states),
  ``privileged_drawer_pos`` (qpos[23]), ``privileged_window_pos`` (qpos[24])
- humanoidmaze: ``xy`` (qpos[:2]) feeding ``set_goal(goal_xy=...)``

Episodes are recovered from ``terminals``. Rendering is embarrassingly
parallel across episodes, so the script renders an episode slice per
invocation (``--episode-start/--episode-end``) and ``--merge`` concatenates
the slice files in order. ``--benchmark`` renders a few frames to measure
render throughput.

Example (scene, 4 slices over 1000 episodes)::

    python scripts/train/convert_ogbench_npz_to_h5.py --task scene \
        --npz .stablewm/datasets/ogbench_mirror/scene-play-v0.npz \
        --out .stablewm/datasets/ogbench/scene_play_v0.h5 \
        --episode-start 0 --episode-end 250
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import h5py
import numpy as np

import stable_worldmodel.envs  # noqa: F401  (registers swm/* gym ids)
from stable_worldmodel.data.formats.hdf5 import HDF5Writer

SCENE_QPOS_CUBE_POS = slice(14, 17)
SCENE_QPOS_CUBE_QUAT = slice(17, 21)
SCENE_QPOS_BUTTON = (21, 22)
SCENE_QPOS_DRAWER = 23
SCENE_QPOS_WINDOW = 24

ENV_SPECS = {
    'scene': dict(
        env_id='swm/OGBScene-v0',
        env_kwargs=dict(ob_type='states', width=224, height=224),
    ),
    'humanoidmaze': dict(
        env_id='swm/OGBMaze-v0',
        env_kwargs=dict(
            loco_env_type='humanoid',
            maze_env_type='maze',
            maze_type='medium',
            ob_type='states',
            width=224,
            height=224,
        ),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=sorted(ENV_SPECS))
    parser.add_argument('--npz', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path, help='final h5 path')
    parser.add_argument('--episode-start', type=int, default=0)
    parser.add_argument('--episode-end', type=int, default=None)
    parser.add_argument(
        '--benchmark-frames',
        type=int,
        default=0,
        help='render N frames to measure throughput, then exit',
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='concatenate the finished slice files into --out and delete them',
    )
    parser.add_argument(
        '--num-slices',
        type=int,
        default=None,
        help='with --merge: merge slices 0..num_slices-1 (default: all present)',
    )
    return parser.parse_args()


def load_columns(npz_path: Path) -> tuple[dict, np.ndarray]:
    with np.load(npz_path) as data:
        # Decompress each entry exactly once; per-episode slicing then hits
        # in-memory arrays instead of re-inflating the zip member.
        columns = {
            key: np.asarray(data[key])
            for key in data.files
            if key != 'terminals'
        }
        terminals = np.asarray(data['terminals'])
    ends = np.where(terminals)[0] + 1
    bounds = np.concatenate([[0], ends]).astype(np.int64)
    print(
        f'loaded {len(bounds) - 1} episodes from {npz_path}', flush=True
    )
    return columns, bounds


def episodes_from_columns(
    columns: dict, bounds: np.ndarray, start: int, end: int
) -> list[dict]:
    episodes = []
    for ep_id in range(start, min(end, len(bounds) - 1)):
        lo, hi = int(bounds[ep_id]), int(bounds[ep_id + 1])
        episodes.append(
            {
                'ep_id': ep_id,
                'actions': columns['actions'][lo:hi].astype(np.float32),
                'observations': columns['observations'][lo:hi].astype(
                    np.float32
                ),
                'qpos': columns['qpos'][lo:hi].astype(np.float64),
                'qvel': columns['qvel'][lo:hi].astype(np.float64),
                **{
                    key: columns[key][lo:hi]
                    for key in ('button_states',)
                    if key in columns
                },
            }
        )
    return episodes


def make_env(task: str):
    spec = ENV_SPECS[task]
    env = gym.make(spec['env_id'], **spec['env_kwargs'])
    env.reset(seed=0)
    return env


def set_env_state(env, task: str, qpos, qvel, button_states=None) -> None:
    target = env.unwrapped
    if task == 'scene':
        if button_states is None:
            raise ValueError('scene replay requires per-step button_states')
        # swm SceneEnv.set_state requires the logical button states; the
        # options-based reset path drops them (scene_env.py reset).
        target.set_state(
            qpos,
            qvel,
            button_state_0=float(button_states[0]),
            button_state_1=float(button_states[1]),
        )
    else:
        target.set_state(qpos, qvel)


def derived_columns(task: str, episode: dict) -> dict:
    qpos = episode['qpos']
    columns = {}
    if task == 'scene':
        buttons = episode['button_states']
        columns = {
            'privileged_block_0_pos': qpos[:, SCENE_QPOS_CUBE_POS].copy(),
            'privileged_block_0_quat': qpos[:, SCENE_QPOS_CUBE_QUAT].copy(),
            # Integer dtype: compute_observation one-hot-indexes the current
            # button state via np.eye(...)[state], which rejects floats.
            'privileged_button_0_state': buttons[:, 0].astype(np.int64),
            'privileged_button_1_state': buttons[:, 1].astype(np.int64),
            'privileged_drawer_pos': qpos[:, SCENE_QPOS_DRAWER : SCENE_QPOS_DRAWER + 1].copy(),
            'privileged_window_pos': qpos[:, SCENE_QPOS_WINDOW : SCENE_QPOS_WINDOW + 1].copy(),
        }
    elif task == 'humanoidmaze':
        columns = {'xy': qpos[:, :2].copy()}
    return columns


def render_episode(env, task: str, episode: dict) -> dict:
    frames = np.empty((len(episode['qpos']), 224, 224, 3), dtype=np.uint8)
    for t in range(len(episode['qpos'])):
        set_env_state(
            env,
            task,
            episode['qpos'][t],
            episode['qvel'][t],
            episode['button_states'][t]
            if 'button_states' in episode
            else None,
        )
        frame = env.render()
        frame = np.asarray(frame)
        if frame.shape != (224, 224, 3):
            raise RuntimeError(
                f'{task}: unexpected render shape {frame.shape}'
            )
        frames[t] = frame
    length = len(episode['qpos'])
    columns = {
        'pixels': frames,
        'action': episode['actions'],
        'observation': episode['observations'],
        'qpos': episode['qpos'],
        'qvel': episode['qvel'],
        'ep_idx': np.full(length, episode['ep_id'], dtype=np.int32),
        'step_idx': np.arange(length, dtype=np.int64),
        **derived_columns(task, episode),
    }
    return columns


def slice_path(out: Path, index: int) -> Path:
    return out.with_name(f'{out.name}.slice{index:04d}.h5')


def render_slice(args, episodes, slice_index) -> Path:
    target = slice_path(args.out, slice_index)
    if target.exists():
        print(f'reusing slice {target}', flush=True)
        return target
    env = make_env(args.task)
    started = time.time()
    with HDF5Writer(target) as writer:
        for offset, episode in enumerate(episodes):
            writer.write_episode(render_episode(env, args.task, episode))
            if (offset + 1) % 10 == 0 or offset + 1 == len(episodes):
                done = offset + 1
                rate = sum(len(e['qpos']) for e in episodes[:done]) / max(
                    time.time() - started, 1e-9
                )
                print(
                    f'slice {slice_index}: {done}/{len(episodes)} episodes, '
                    f'{rate:.1f} frames/s',
                    flush=True,
                )
    env.close()
    return target


def merge_slices(args) -> None:
    if args.out.exists():
        print(f'final dataset already exists: {args.out}', flush=True)
        return
    if args.num_slices is not None:
        indices = list(range(args.num_slices))
    else:
        indices = sorted(
            int(p.name.rsplit('slice', 1)[1][: -len('.h5')])
            for p in args.out.parent.glob(f'{args.out.name}.slice*.h5')
        )
    if not indices:
        raise FileNotFoundError(f'no slice files next to {args.out}')
    temporary = args.out.with_name(f'.{args.out.name}.tmp.h5')
    frames = 0
    with HDF5Writer(temporary) as writer:
        for index in indices:
            source = slice_path(args.out, index)
            with h5py.File(source, 'r') as h5:
                ep_len = h5['ep_len'][:]
                ep_offset = h5['ep_offset'][:]
                for length, offset in zip(ep_len, ep_offset):
                    lo, hi = int(offset), int(offset + length)
                    writer.write_episode(
                        {
                            col: h5[col][lo:hi]
                            for col in h5.keys()
                            if col not in ('ep_len', 'ep_offset')
                        }
                    )
                    frames += int(length)
            print(f'merged {source}', flush=True)
    os.replace(temporary, args.out)
    print(f'merged {len(indices)} slices, {frames} frames -> {args.out}', flush=True)
    for index in indices:
        slice_file = slice_path(args.out, index)
        if slice_file.exists():
            os.remove(slice_file)


def benchmark(args, episodes) -> None:
    env = make_env(args.task)
    episode = episodes[0]
    started = time.time()
    for t in range(args.benchmark_frames):
        index = t % len(episode['qpos'])
        set_env_state(
            env,
            args.task,
            episode['qpos'][index],
            episode['qvel'][index],
            episode['button_states'][index]
            if 'button_states' in episode
            else None,
        )
        env.render()
    rate = args.benchmark_frames / max(time.time() - started, 1e-9)
    print(f'{args.task}: {rate:.1f} frames/s single process', flush=True)
    env.close()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_slices(args)
        return

    columns, bounds = load_columns(args.npz)
    end = len(bounds) - 1 if args.episode_end is None else args.episode_end

    if args.benchmark_frames:
        episodes = episodes_from_columns(columns, bounds, 0, 1)
        benchmark(args, episodes)
        return

    selected = episodes_from_columns(
        columns, bounds, args.episode_start, end
    )
    slice_index = args.episode_start
    render_slice(args, selected, slice_index)


if __name__ == '__main__':
    main()
