"""Prepare, smoke-test, and run the two-seed codebook experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf, open_dict


CORE_CONDITIONS = (
    'k512_original',
    'k2048_original',
    'k8192_original',
    'k8192_rigid',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/codebook_quality_rigid_experiment.yaml',
    )
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument(
        '--conditions',
        default=None,
        help='Optional comma-separated condition subset.',
    )
    parser.add_argument(
        '--seeds',
        default=None,
        help='Optional comma-separated training-seed subset.',
    )
    parser.add_argument('--skip-diagnostics', action='store_true')
    parser.add_argument(
        '--devices',
        default=None,
        help='Temporary comma-separated device override (useful for smoke).',
    )
    return parser.parse_args()


def run(command: list[str], env: dict, *, cwd: Path) -> None:
    print('Launching:', ' '.join(command), flush=True)
    subprocess.run(command, check=True, cwd=cwd, env=env)


def experiment_env(matrix, root: Path) -> dict:
    env = os.environ.copy()
    devices = ','.join(str(value) for value in matrix.compute.devices)
    env.update(
        {
            'CUDA_VISIBLE_DEVICES': devices,
            'STABLEWM_HOME': str(
                Path(matrix.experiment.base_teacher_cache).parent.parent
            ),
            'LOCAL_DATASET_DIR': str(
                Path(matrix.experiment.base_teacher_cache).parent.parent
            ),
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'PYTHONUNBUFFERED': '1',
            'TOKENIZERS_PARALLELISM': 'false',
        }
    )
    return env


def ensure_manifests(matrix, project_root: Path, env: dict) -> dict:
    output_dir = Path(matrix.evaluation.manifest_dir).expanduser().resolve()
    command = [
        sys.executable,
        str(project_root / 'scripts/train/create_pusht_eval_manifests.py'),
        '--dataset',
        'galilai-group/lewm-pusht',
        '--dataset-cache',
        str(Path(matrix.experiment.base_teacher_cache).parent.parent),
        '--output-dir',
        str(output_dir),
        '--selection-seed',
        str(matrix.evaluation.selection_seed),
        '--selection-count',
        str(matrix.evaluation.selection_count),
        '--test-seed',
        str(matrix.evaluation.test_seed),
        '--test-count',
        str(matrix.evaluation.test_count),
        '--shard-size',
        str(matrix.evaluation.shard_size),
    ]
    run(command, env, cwd=project_root)
    return json.loads((output_dir / 'manifest_summary.json').read_text())


def transform_output(matrix, mode: str) -> Path:
    key = {
        'rigid': 'rigid_output',
        'rotation_only': 'rotation_only_output',
        'translation_only': 'translation_only_output',
    }[mode]
    return Path(matrix.transform[key]).expanduser().resolve()


def ensure_transform(matrix, project_root: Path, env: dict, mode: str) -> Path:
    output = transform_output(matrix, mode)
    command = [
        sys.executable,
        str(project_root / 'scripts/train/create_rigid_codebook.py'),
        '--source-codebook',
        str(Path(matrix.transform.source_codebook).expanduser().resolve()),
        '--teacher-cache',
        str(Path(matrix.experiment.base_teacher_cache).expanduser().resolve()),
        '--output',
        str(output),
        '--seed',
        str(matrix.transform.seed),
        '--translation-scale',
        str(matrix.transform.translation_scale),
        '--audit-samples',
        str(matrix.transform.audit_samples),
        '--mode',
        mode,
    ]
    run(command, env, cwd=project_root)
    return output


def condition_cache_dir(matrix, condition: str, smoke: bool) -> Path:
    root = Path(matrix.experiment.root).expanduser().resolve()
    prefix = root / 'smoke' if smoke else root
    return prefix / 'caches' / condition


def condition_run_dir(
    matrix, condition: str, seed: int, smoke: bool
) -> Path:
    root = Path(matrix.experiment.root).expanduser().resolve()
    prefix = root / 'smoke' if smoke else root
    return prefix / 'runs' / condition / f'seed{seed}'


def condition_evaluation_dir(
    matrix, condition: str, seed: int, smoke: bool
) -> Path:
    root = Path(matrix.experiment.root).expanduser().resolve()
    prefix = root / 'smoke' if smoke else root
    return prefix / 'evaluations' / condition / f'seed{seed}'


def make_training_config(
    matrix,
    condition_name: str,
    condition,
    seed: int,
    manifests: dict,
    *,
    smoke: bool,
) -> tuple[Path, object]:
    cfg = OmegaConf.load(matrix.experiment.base_training_config)
    codebook_checkpoint = Path(
        condition.codebook_checkpoint
    ).expanduser().resolve()
    transform_mode = str(condition.transform_mode)
    with open_dict(cfg):
        cfg.seed = seed
        cfg.data.split_seed = int(matrix.split_seed)
        cfg.data.train_batch_size_per_gpu = int(
            matrix.compute.batch_size_per_gpu
        )
        cfg.data.cpu_workers_total = int(matrix.compute.cpu_workers_total)
        cfg.codebook.num_embeddings = int(condition.num_embeddings)
        cfg.paths.codebook_checkpoint = str(codebook_checkpoint)
        cfg.model.num_embeddings = int(condition.num_embeddings)
        cfg.model.codebook_checkpoint = str(codebook_checkpoint)
        cfg.gates.enforce = False
        cfg.evaluation.selection_manifest = manifests['selection_manifest']
        cfg.evaluation.test_manifests = list(manifests['test_shards'])
        cfg.evaluation.selection_video = bool(
            matrix.evaluation.selection_video
        )
        cfg.evaluation.heldout_video = bool(
            matrix.evaluation.heldout_video
        )
        cfg.evaluation.num_eval = int(matrix.evaluation.selection_count)
        cfg.evaluation.output_root = str(
            condition_evaluation_dir(
                matrix, condition_name, seed, smoke
            )
        )
        cfg.evaluation.equivalence_margin_percentage_points = float(
            matrix.evaluation.equivalence_margin_percentage_points
        )
        cfg.latent_transform = {
            'enabled': transform_mode != 'none',
            'checkpoint': (
                str(codebook_checkpoint / 'transform.pt')
                if transform_mode != 'none'
                else None
            ),
            'initialize_adapter': True,
            'reuse_base_assignments': (
                transform_mode != 'none' and not smoke
            ),
            'mode': transform_mode,
        }
        if condition_name == 'k8192_original':
            cfg.paths.cache_dir = str(
                Path(matrix.experiment.base_teacher_cache)
                .expanduser()
                .resolve()
            )
            cfg.paths.pop('base_teacher_cache', None)
        else:
            cfg.paths.cache_dir = str(
                condition_cache_dir(matrix, condition_name, smoke)
            )
            cfg.paths.base_teacher_cache = str(
                Path(matrix.experiment.base_teacher_cache)
                .expanduser()
                .resolve()
            )
        cfg.paths.output_dir = str(
            condition_run_dir(matrix, condition_name, seed, smoke)
        )
        if smoke:
            cfg.smoke.enabled = True
            cfg.smoke.train_samples = 1536
            cfg.smoke.val_samples = 768
            cfg.smoke.batches_per_epoch = 2
            cfg.data.cache_batch_size_per_gpu = 128
            cfg.data.train_batch_size_per_gpu = 64
            cfg.data.cpu_workers_total = 8
            cfg.evaluation.enabled = False
            cfg.optimizer.weight_decay = 0.0
    OmegaConf.resolve(cfg)
    root = Path(matrix.experiment.root).expanduser().resolve()
    prefix = root / 'smoke' if smoke else root
    config_path = (
        prefix
        / 'configs'
        / f'{condition_name}_seed{seed}.yaml'
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)
    return config_path, cfg


def summary_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get('status') == 'complete'
    except (json.JSONDecodeError, OSError):
        return False


def run_condition(
    matrix,
    project_root: Path,
    env: dict,
    condition_name: str,
    seed: int,
    config_path: Path,
    cfg,
    *,
    smoke: bool,
) -> None:
    evaluation_root = Path(cfg.evaluation.output_root)
    summary_path = evaluation_root / 'summary.json'
    if not smoke and summary_complete(summary_path):
        print(
            f'Skipping complete condition: {condition_name} seed={seed}',
            flush=True,
        )
        return

    reuse_official = (
        condition_name == 'k8192_original'
        and seed == 3072
        and not smoke
    )
    if reuse_official:
        source = Path(
            matrix.experiment.official_k8192_seed3072_output
        ).expanduser().resolve()
        cfg.paths.output_dir = str(source)
        OmegaConf.save(cfg, config_path)
        command = [
            sys.executable,
            str(project_root / 'scripts/train/evaluate_joint_distillation.py'),
            '--config',
            str(config_path),
            '--devices',
            ','.join(str(value) for value in matrix.compute.devices),
            '--evaluation-root',
            str(evaluation_root),
        ]
        run(command, env, cwd=project_root)
        return

    command = [
        sys.executable,
        str(project_root / 'scripts/train/run_joint_distillation.py'),
        '--config',
        str(config_path),
        '--nproc-per-node',
        str(matrix.compute.nproc_per_node),
    ]
    run(command, env, cwd=project_root)


def heldout_rate(matrix, condition: str, seed: int) -> float:
    path = (
        condition_evaluation_dir(matrix, condition, seed, False)
        / 'summary.json'
    )
    payload = json.loads(path.read_text())
    return float(payload['heldout_test']['success_rate'])


def diagnostic_triggered(matrix) -> tuple[bool, float]:
    original = heldout_rate(matrix, 'k8192_original', 3072)
    rigid = heldout_rate(matrix, 'k8192_rigid', 3072)
    gap = abs(rigid - original)
    return (
        gap > float(matrix.diagnostics.trigger_gap_percentage_points),
        gap,
    )


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    matrix = OmegaConf.load(Path(args.config).expanduser().resolve())
    OmegaConf.resolve(matrix)
    if args.devices:
        devices = [
            int(value.strip())
            for value in args.devices.split(',')
            if value.strip()
        ]
        if not devices:
            raise ValueError('device override is empty')
        if int(matrix.compute.global_batch_size) % len(devices):
            raise ValueError('global batch is not divisible by device count')
        with open_dict(matrix):
            matrix.compute.devices = devices
            matrix.compute.nproc_per_node = len(devices)
            matrix.compute.batch_size_per_gpu = (
                int(matrix.compute.global_batch_size) // len(devices)
            )
    root = Path(matrix.experiment.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = experiment_env(matrix, root)
    manifests = ensure_manifests(matrix, project_root, env)
    ensure_transform(matrix, project_root, env, 'rigid')

    if args.prepare_only:
        print(f'Experiment preparation complete: {root}', flush=True)
        return

    selected_conditions = list(CORE_CONDITIONS)
    if args.conditions:
        requested = {
            value.strip()
            for value in args.conditions.split(',')
            if value.strip()
        }
        unknown = requested.difference(CORE_CONDITIONS)
        if unknown:
            raise ValueError(f'unknown core conditions: {sorted(unknown)}')
        selected_conditions = [
            value for value in CORE_CONDITIONS if value in requested
        ]
    seeds = [int(value) for value in matrix.seeds]
    if args.seeds:
        seeds = [
            int(value.strip())
            for value in args.seeds.split(',')
            if value.strip()
        ]
    if args.smoke:
        seeds = [3072]

    for seed in seeds:
        for condition_name in selected_conditions:
            if args.smoke and condition_name == 'k8192_original':
                continue
            condition = matrix.conditions[condition_name]
            config_path, cfg = make_training_config(
                matrix,
                condition_name,
                condition,
                seed,
                manifests,
                smoke=args.smoke,
            )
            run_condition(
                matrix,
                project_root,
                env,
                condition_name,
                seed,
                config_path,
                cfg,
                smoke=args.smoke,
            )

    if args.smoke:
        print(f'All requested smoke conditions completed: {root}', flush=True)
        return
    if args.conditions:
        return

    triggered, gap = diagnostic_triggered(matrix)
    if args.skip_diagnostics:
        triggered = False
    diagnostic_record = {
        'trigger_threshold_percentage_points': float(
            matrix.diagnostics.trigger_gap_percentage_points
        ),
        'observed_seed3072_gap_percentage_points': gap,
        'triggered': triggered,
        'conditions': [],
    }
    if triggered and bool(matrix.diagnostics.enabled):
        for condition_name, condition in matrix.diagnostics.conditions.items():
            ensure_transform(
                matrix, project_root, env, str(condition.transform_mode)
            )
            config_path, cfg = make_training_config(
                matrix,
                condition_name,
                condition,
                int(matrix.diagnostics.seed),
                manifests,
                smoke=False,
            )
            run_condition(
                matrix,
                project_root,
                env,
                condition_name,
                int(matrix.diagnostics.seed),
                config_path,
                cfg,
                smoke=False,
            )
            diagnostic_record['conditions'].append(condition_name)
    record_path = root / 'diagnostic_decision.json'
    temporary = record_path.with_name(f'.{record_path.name}.tmp')
    temporary.write_text(
        json.dumps(diagnostic_record, indent=2, sort_keys=True) + '\n'
    )
    os.replace(temporary, record_path)
    print(json.dumps(diagnostic_record, indent=2, sort_keys=True), flush=True)
    run(
        [
            sys.executable,
            str(
                project_root
                / 'scripts/train/summarize_codebook_quality_experiment.py'
            ),
            '--config',
            str(Path(args.config).expanduser().resolve()),
            '--seeds',
            ','.join(str(seed) for seed in seeds),
        ],
        env,
        cwd=project_root,
    )


if __name__ == '__main__':
    main()
