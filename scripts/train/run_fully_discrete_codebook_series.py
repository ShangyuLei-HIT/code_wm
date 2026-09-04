"""Run the fully-discrete codebook series (k512/k2048/k8192_rigid) on PushT and Cube.

Extends the running K8192 fully-discrete single-task experiments with the rest
of the codebook-quality conditions retrained under the fully-discrete
(M5-style) loss. Waits for the current orchestrator to finish, prepares the
missing Cube codebooks, then trains and evaluates every condition sequentially
on all four GPUs. Idempotent: completed stages are validated and skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf, open_dict

PUSHT_TASK = 'pusht'
CUBE_TASK = 'cube'
CONDITION_ORDER = ('k512_fully_discrete', 'k2048_fully_discrete',
                   'k8192_rigid_fully_discrete')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/fully_discrete_codebook_series.yaml',
    )
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Generate configs and verify reusable assets, then exit.',
    )
    parser.add_argument(
        '--skip-wait',
        action='store_true',
        help='Do not wait for the previous orchestrator (tests only).',
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class SeriesLogger:
    def __init__(self, log_root: Path, status_path: Path):
        self.log_root = log_root
        self.status_path = status_path
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.stages = self.log_root / 'stages.log'

    def stage(self, name: str) -> None:
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {name}'
        print(line, flush=True)
        with self.stages.open('a') as stream:
            stream.write(line + '\n')

    def status(self, text: str) -> None:
        stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self.status_path.write_text(f'{stamp} {text}\n')

    def summary_complete(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            return json.loads(path.read_text()).get('status') == 'complete'
        except (json.JSONDecodeError, OSError):
            return False


def experiment_env(matrix) -> dict:
    env = os.environ.copy()
    devices = ','.join(str(value) for value in matrix.compute.devices)
    env.update(
        {
            'CUDA_VISIBLE_DEVICES': devices,
            'STABLEWM_HOME': str(project_root() / '.stablewm'),
            'LOCAL_DATASET_DIR': str(project_root() / '.stablewm'),
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'PYTHONUNBUFFERED': '1',
            'TOKENIZERS_PARALLELISM': 'false',
        }
    )
    return env


def wait_for_previous(logger: SeriesLogger, status_path: Path) -> None:
    logger.stage(f'wait_for_previous_orchestrator status={status_path}')
    while True:
        text = status_path.read_text() if status_path.is_file() else ''
        if 'complete' in text:
            logger.stage('previous_orchestrator_complete')
            return
        if 'failed' in text:
            raise RuntimeError(
                f'previous orchestrator failed; refusing to start: {text}'
            )
        time.sleep(60)


def condition_output_dir(matrix, task: str, condition: str) -> Path:
    root = Path(matrix.experiment.root).expanduser().resolve()
    return root / 'runs' / task / condition


def condition_cache_dir(matrix, task: str, condition: str) -> Path:
    spec = matrix[f'{task}']['conditions'][condition]
    configured = spec.get('cache_dir')
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(matrix.experiment.root).expanduser().resolve()
    return root / 'caches' / f'{task}_{condition}'


def latent_transform_block(codebook_checkpoint: Path) -> dict:
    return {
        'enabled': True,
        'checkpoint': str(codebook_checkpoint / 'transform.pt'),
        'initialize_adapter': True,
        'reuse_base_assignments': True,
    }


def make_training_config(matrix, task: str, condition: str) -> Path:
    spec = matrix[f'{task}']['conditions'][condition]
    base_path = Path(
        matrix.experiment[f'{task}_base_config']
    ).expanduser().resolve()
    cfg = OmegaConf.load(base_path)
    codebook_checkpoint = Path(spec.codebook_checkpoint).expanduser().resolve()
    base_teacher_cache = spec.get('base_teacher_cache')
    if base_teacher_cache is None:
        base_teacher_cache = matrix[f'{task}']['base_teacher_cache']
    base_teacher_cache = Path(base_teacher_cache).expanduser().resolve()
    output_dir = condition_output_dir(matrix, task, condition)
    baselines = dict(
        matrix.evaluation[f'{task}_eval_baselines']
        if task == CUBE_TASK
        else matrix.evaluation['pusht_eval_baselines']
    )
    with open_dict(cfg):
        cfg.seed = 3072
        cfg.paths.codebook_checkpoint = str(codebook_checkpoint)
        cfg.paths.base_teacher_cache = str(base_teacher_cache)
        cfg.paths.cache_dir = str(condition_cache_dir(matrix, task, condition))
        cfg.paths.output_dir = str(output_dir)
        cfg.codebook.num_embeddings = int(spec.num_embeddings)
        cfg.model.num_embeddings = int(spec.num_embeddings)
        cfg.data.train_batch_size_per_gpu = int(
            matrix.compute.batch_size_per_gpu
        )
        cfg.data.cpu_workers_total = int(matrix.compute.cpu_workers_total)
        cfg.gates.enforce = False
        if str(spec.transform_mode) == 'rigid':
            cfg.latent_transform = latent_transform_block(codebook_checkpoint)
        else:
            cfg.latent_transform = {'enabled': False}
        cfg.evaluation.baselines = baselines
        cfg.evaluation.num_eval = int(matrix.evaluation.num_eval)
        cfg.evaluation.seed = int(matrix.evaluation.seed)
        cfg.evaluation.stages = list(matrix.evaluation.stages)
    OmegaConf.resolve(cfg)
    root = Path(matrix.experiment.root).expanduser().resolve()
    config_path = root / 'configs' / f'{task}_{condition}_seed3072.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)
    return config_path


def make_heldout_config(matrix, task: str, condition: str) -> Path:
    training_path = make_training_config(matrix, task, condition)
    cfg = OmegaConf.load(training_path)
    manifest_dir = Path(
        matrix[f'{task}']['heldout_manifest_dir']
    ).expanduser().resolve()
    shards = sorted(manifest_dir.glob('pusht_test_seed4242_n200_shard*.json'))
    if len(shards) != 4:
        raise FileNotFoundError(f'expected 4 held-out shards in {manifest_dir}')
    with open_dict(cfg):
        cfg.evaluation.selection_manifest = str(
            manifest_dir / 'pusht_selection_seed42_n50.json'
        )
        cfg.evaluation.test_manifests = [str(path) for path in shards]
        cfg.evaluation.selection_video = False
        cfg.evaluation.heldout_video = False
        cfg.evaluation.output_root = str(
            condition_output_dir(matrix, task, condition)
            / 'task_evaluation_heldout'
        )
        cfg.evaluation.baselines = dict(
            matrix.evaluation['heldout_baselines']
        )
    OmegaConf.resolve(cfg)
    root = Path(matrix.experiment.root).expanduser().resolve()
    config_path = root / 'configs' / f'{task}_{condition}_heldout_seed3072.yaml'
    OmegaConf.save(cfg, config_path)
    return config_path


def run(command: list[str], env: dict, log_path: Path, device: str) -> None:
    env = dict(env)
    env['CUDA_VISIBLE_DEVICES'] = device
    print('Launching:', ' '.join(command), flush=True)
    with log_path.open('w') as stream:
        process = subprocess.run(
            command,
            cwd=project_root(),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if process.returncode:
        raise RuntimeError(
            f'stage failed ({process.returncode}); log={log_path}'
        )


def codebook_checkpoint_valid(path: Path, num_embeddings: int) -> bool:
    if not (path / 'weights.pt').is_file():
        return False
    config_path = path / 'config.json'
    if config_path.is_file():
        try:
            return int(
                json.loads(config_path.read_text())['num_embeddings']
            ) == num_embeddings
        except (json.JSONDecodeError, KeyError, ValueError):
            return False
    return False


def prepare_cube_codebooks(matrix, logger: SeriesLogger, env: dict) -> None:
    log_root = logger.log_root
    jobs = []
    for device_id, (name, spec) in enumerate(
        matrix.cube['codebook_prep'].items()
    ):
        output = (
            project_root()
            / '.stablewm'
            / 'checkpoints'
            / str(spec.output_model_name)
        )
        if codebook_checkpoint_valid(output, int(spec.num_embeddings)):
            logger.stage(f'reuse_cube_codebook_{name}')
            continue
        logger.stage(f'cube_codebook_{name}_gpu{device_id}')
        command = [
            sys.executable,
            str(project_root() / 'scripts/train/latent_codebook.py'),
            f'encoder_checkpoint={matrix.cube.teacher_checkpoint}',
            f'data.dataset={matrix.cube.dataset}',
            f'output_model_name={spec.output_model_name}',
            f'codebook.num_embeddings={int(spec.num_embeddings)}',
            f'data.num_workers={int(matrix.compute.codebook_workers)}',
        ]
        jobs.append((name, command, device_id))
    processes = []
    for name, command, device_id in jobs:
        stream = (log_root / f'cube_codebook_{name}.log').open('w')
        env_local = dict(env)
        env_local['CUDA_VISIBLE_DEVICES'] = str(device_id)
        print('Launching:', ' '.join(command), flush=True)
        processes.append(
            (
                name,
                subprocess.Popen(
                    command,
                    cwd=project_root(),
                    env=env_local,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                ),
                stream,
            )
        )
    errors = []
    for name, process, stream in processes:
        code = process.wait()
        stream.close()
        if code:
            errors.append(f'cube codebook {name} exited {code}')
    if errors:
        raise RuntimeError('; '.join(errors))
    for name, spec in matrix.cube['codebook_prep'].items():
        output = (
            project_root()
            / '.stablewm'
            / 'checkpoints'
            / str(spec.output_model_name)
        )
        if not codebook_checkpoint_valid(output, int(spec.num_embeddings)):
            raise RuntimeError(f'cube codebook missing after prep: {output}')


def prepare_cube_rigid(matrix, logger: SeriesLogger, env: dict) -> None:
    output = Path(matrix.cube.rigid.output).expanduser().resolve()
    required = [
        output / 'weights.pt',
        output / 'transform.pt',
        output / 'rigid_transform_manifest.json',
    ]
    if all(path.is_file() for path in required):
        logger.stage('reuse_cube_rigid_codebook')
        return
    logger.stage('cube_rigid_codebook_gpu0')
    run(
        [
            sys.executable,
            str(project_root() / 'scripts/train/create_rigid_codebook.py'),
            '--source-codebook',
            str(Path(matrix.cube.source_codebook_for_rigid)),
            '--teacher-cache',
            str(Path(matrix.cube.base_teacher_cache)),
            '--output',
            str(output),
            '--seed',
            str(matrix.cube.rigid.seed),
            '--translation-scale',
            str(matrix.cube.rigid.translation_scale),
            '--audit-samples',
            str(matrix.cube.rigid.audit_samples),
            '--mode',
            'rigid',
        ],
        env,
        logger.log_root / 'cube_rigid_codebook.log',
        '0',
    )


def verify_reusable_assets(matrix, logger: SeriesLogger) -> None:
    """Fail fast before training if a reused asset is missing or mismatched."""
    for condition in CONDITION_ORDER:
        spec = matrix[PUSHT_TASK]['conditions'][condition]
        codebook = Path(spec.codebook_checkpoint).expanduser().resolve()
        cache = Path(spec.cache_dir).expanduser().resolve()
        if not (codebook / 'weights.pt').is_file():
            raise FileNotFoundError(codebook / 'weights.pt')
        metadata = json.loads((cache / 'metadata.json').read_text())
        if metadata['codebook_checkpoint'] != str(codebook):
            raise RuntimeError(
                f'{condition}: cache expects codebook '
                f'{metadata["codebook_checkpoint"]}, config has {codebook}'
            )
        if int(metadata['codebook_size']) != int(spec.num_embeddings):
            raise RuntimeError(f'{condition}: cache codebook size mismatch')
        if str(spec.transform_mode) == 'rigid':
            transform = metadata['latent_transform']
            expected = {
                'checkpoint': str(codebook / 'transform.pt'),
                'initialize_adapter': True,
                'reuse_base_assignments': True,
            }
            for key, value in expected.items():
                if transform[key] != value:
                    raise RuntimeError(
                        f'{condition}: cache latent_transform.{key} mismatch'
                    )
            if sha256_file(codebook / 'transform.pt') != transform['sha256']:
                raise RuntimeError(
                    f'{condition}: transform.pt hash differs from cache'
                )
    logger.stage('reusable_assets_verified')


def train_condition(
    matrix, logger: SeriesLogger, env: dict, task: str, condition: str
) -> None:
    output = condition_output_dir(matrix, task, condition)
    if (output / 'weights_final.pt').is_file() and logger.summary_complete(
        output / 'task_evaluation' / 'summary.json'
    ):
        logger.stage(f'reuse_{task}_{condition}')
        return
    config_path = make_training_config(matrix, task, condition)
    logger.stage(f'{task}_{condition}_train_and_evaluation_gpu0123')
    run(
        [
            sys.executable,
            str(project_root() / 'scripts/train/run_joint_distillation.py'),
            '--config',
            str(config_path),
            '--nproc-per-node',
            str(matrix.compute.nproc_per_node),
        ],
        env,
        logger.log_root / f'{task}_{condition}_train.log',
        ','.join(str(value) for value in matrix.compute.devices),
    )


def evaluate_heldout(
    matrix, logger: SeriesLogger, env: dict, task: str, condition: str
) -> None:
    output = condition_output_dir(matrix, task, condition)
    summary = output / 'task_evaluation_heldout' / 'summary.json'
    if logger.summary_complete(summary):
        logger.stage(f'reuse_{task}_{condition}_heldout_evaluation')
        return
    config_path = make_heldout_config(matrix, task, condition)
    logger.stage(f'{task}_{condition}_heldout_evaluation_gpu0123')
    run(
        [
            sys.executable,
            str(
                project_root()
                / 'scripts/train/evaluate_joint_distillation.py'
            ),
            '--config',
            str(config_path),
            '--devices',
            ','.join(str(value) for value in matrix.compute.devices),
        ],
        env,
        logger.log_root / f'{task}_{condition}_heldout_evaluation.log',
        ','.join(str(value) for value in matrix.compute.devices),
    )


def main():
    args = parse_args()
    root = project_root()
    matrix = OmegaConf.load(Path(args.config).expanduser().resolve())
    OmegaConf.resolve(matrix)
    log_root = root / 'logs' / 'fully_discrete_codebook_series_gpu0123'
    logger = SeriesLogger(log_root, log_root / 'status.txt')
    logger.status('running')
    try:
        env = experiment_env(matrix)
        if not args.skip_wait:
            wait_for_previous(
                logger,
                Path(matrix.experiment.wait_status).expanduser().resolve(),
            )
        verify_reusable_assets(matrix, logger)
        if args.prepare_only:
            for task in (PUSHT_TASK, CUBE_TASK):
                for condition in CONDITION_ORDER:
                    make_training_config(matrix, task, condition)
                    if task == PUSHT_TASK:
                        make_heldout_config(matrix, task, condition)
            logger.stage('prepare_only_complete')
            logger.status('complete_prepare_only')
            return

        # Cube needs k512/k2048 codebooks and a rigid transform before its
        # conditions can run; prepare them first so failures surface early.
        prepare_cube_codebooks(matrix, logger, env)
        prepare_cube_rigid(matrix, logger, env)

        for task in (PUSHT_TASK, CUBE_TASK):
            for condition in CONDITION_ORDER:
                train_condition(matrix, logger, env, task, condition)
                if task == PUSHT_TASK:
                    evaluate_heldout(matrix, logger, env, task, condition)
        logger.stage('fully_discrete_codebook_series_complete')
        logger.status('complete')
    except Exception as error:
        logger.stage(f'error:{type(error).__name__}:{error}')
        logger.status('failed')
        raise


if __name__ == '__main__':
    main()
