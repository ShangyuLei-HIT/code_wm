"""Run an isolated, tiny 0 -> transition -> 1 integration test."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / 'scripts/train/config/vq_lewm_joint_distillation.yaml'
    cfg = OmegaConf.load(source)
    cfg.smoke.enabled = True
    cfg.data.cache_batch_size_per_gpu = 128
    cfg.data.train_batch_size_per_gpu = 64
    cfg.data.cpu_workers_total = 6
    cfg.optimizer.weight_decay = 0.0
    for phase in ('phase1', 'phase2', 'phase3'):
        cfg.phases[phase].encoder_lr = [0.0, 0.0]
    cfg.paths.cache_dir = f'{cfg.paths.cache_dir}_smoke'
    cfg.paths.output_dir = f'{cfg.paths.output_dir}_smoke'
    generated = Path('/tmp/vq_lewm_joint_distillation_smoke.yaml')
    OmegaConf.save(cfg, generated)

    env = os.environ.copy()
    env['STABLEWM_HOME'] = str(Path(cfg.paths.dataset_cache).resolve())
    env['LOCAL_DATASET_DIR'] = str(Path(cfg.paths.dataset_cache).resolve())
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    for script in (
        'cache_codebook_distillation.py',
        'vq_lewm_joint_distillation.py',
    ):
        command = [
            sys.executable,
            '-m',
            'torch.distributed.run',
            '--standalone',
            '--nproc-per-node',
            '3',
            str(root / 'scripts/train' / script),
            '--config',
            str(generated),
        ]
        print('Smoke launching:', ' '.join(command), flush=True)
        subprocess.run(command, check=True, env=env)


if __name__ == '__main__':
    main()
