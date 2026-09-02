"""Launch offline caching, then the teacher-free three-GPU training job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/vq_lewm_joint_distillation.yaml',
    )
    parser.add_argument('--nproc-per-node', type=int, default=3)
    return parser.parse_args()


def run_distributed(script: Path, config: Path, nproc: int, env: dict):
    command = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nproc-per-node',
        str(nproc),
        str(script),
        '--config',
        str(config),
    ]
    print('Launching:', ' '.join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def run_task_evaluation(script: Path, config: Path, devices: list[str], env: dict):
    command = [
        sys.executable,
        str(script),
        '--config',
        str(config),
        '--devices',
        ','.join(devices),
    ]
    print('Launching automatic task evaluation:', ' '.join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(config)
    output_dir = Path(cfg.paths.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = [item for item in visible.split(',') if item.strip()]
    if len(devices) != args.nproc_per_node:
        raise RuntimeError(
            f'expected {args.nproc_per_node} visible GPUs, got {visible!r}'
        )
    env = os.environ.copy()
    env.update(
        {
            'STABLEWM_HOME': str(Path(cfg.paths.dataset_cache).resolve()),
            'LOCAL_DATASET_DIR': str(Path(cfg.paths.dataset_cache).resolve()),
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'PYTHONUNBUFFERED': '1',
            'TOKENIZERS_PARALLELISM': 'false',
        }
    )
    (output_dir / 'launcher.json').write_text(
        json.dumps(
            {
                'config': str(config),
                'cuda_visible_devices': visible,
                'nproc_per_node': args.nproc_per_node,
                'cpu_workers_total': int(cfg.data.cpu_workers_total),
                'python': sys.executable,
                'automatic_evaluation': {
                    'enabled': bool(cfg.evaluation.enabled),
                    'stages': list(cfg.evaluation.stages),
                    'num_eval': int(cfg.evaluation.num_eval),
                },
            },
            indent=2,
        )
        + '\n'
    )
    cache_script = (
        'cache_codebook_assignments.py'
        if cfg.paths.get('base_teacher_cache')
        else 'cache_codebook_distillation.py'
    )
    run_distributed(
        project_root / 'scripts/train' / cache_script,
        config,
        args.nproc_per_node,
        env,
    )
    print(
        'Offline teacher process exited; starting cache-only training.',
        flush=True,
    )
    run_distributed(
        project_root / 'scripts/train/vq_lewm_joint_distillation.py',
        config,
        args.nproc_per_node,
        env,
    )
    if bool(cfg.evaluation.enabled):
        print(
            'Training complete; exporting checkpoints and evaluating PushT.',
            flush=True,
        )
        run_task_evaluation(
            project_root / 'scripts/train/evaluate_joint_distillation.py',
            config,
            devices,
            env,
        )


if __name__ == '__main__':
    main()
