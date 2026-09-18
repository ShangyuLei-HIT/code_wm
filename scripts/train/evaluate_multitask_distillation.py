"""Export one shared model with per-task defaults and run both MPC suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/multitask_vq_lewm.yaml',
    )
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--devices', default=None)
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument(
        '--manifest-dir',
        default=None,
        help=(
            'directory holding <task>_<split>_seed*_n*.json evaluation '
            'manifests; overrides cfg.evaluation.manifest_dir; when unset '
            'the evaluation samples episodes itself (previous behavior)'
        ),
    )
    parser.add_argument(
        '--eval-split',
        choices=['selection', 'test'],
        default=None,
        help=(
            'which manifest split to evaluate; effective default is '
            "cfg.evaluation.eval_split if present, else 'test'"
        ),
    )
    parser.add_argument(
        '--export-root',
        default=None,
        help=(
            'write the per-task exports and summary under this directory '
            'instead of <output_dir>/task_evaluation (useful to keep '
            'manifest-based re-evaluations separate from legacy ones)'
        ),
    )
    parser.add_argument(
        '--run-label',
        default=None,
        help='free-form label stamped into summary.json (e.g. M2/seed tag)',
    )
    return parser.parse_args()


def deployment_config(cfg, task_id: int) -> dict:
    model = OmegaConf.to_container(cfg.model, resolve=True)
    encoder = model.get('student_encoder', model.get('encoder'))
    return {
        '_target_': (
            'stable_worldmodel.wm.vq_lewm.multitask.'
            'MultiTaskDistilledLeWM'
        ),
        'encoder': encoder,
        'projector': model['projector'],
        'adapter': model['adapter'],
        'action_encoder': model['action_encoder'],
        'predictor': model['predictor'],
        'pred_proj': model['pred_proj'],
        'embedding_dim': int(model.get('embedding_dim', 192)),
        'num_tasks': int(model.get('num_tasks', len(cfg.tasks))),
        'default_task_id': task_id,
    }


def state_from_export(payload: dict) -> dict[str, torch.Tensor]:
    if 'state_dict' in payload:
        return payload['state_dict']
    modules = payload['modules']
    rename = {'student_encoder': 'encoder'}
    state = {}
    for name, component in modules.items():
        target = rename.get(name, name)
        for key, value in component.items():
            state[f'{target}.{key}'] = value
    return state


def atomic_torch_save(value, path: Path):
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def export_tasks(cfg, checkpoint: Path, root: Path) -> list[dict]:
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = state_from_export(payload)
    exports = []
    for task_id, task in enumerate(cfg.tasks):
        config = deployment_config(cfg, task_id)
        model = instantiate(config)
        model.load_state_dict(state, strict=True)
        del model
        directory = root / str(task.name)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(state, directory / 'weights.pt')
        (directory / 'config.json').write_text(
            json.dumps(config, indent=2) + '\n'
        )
        exports.append(
            {
                'task_id': task_id,
                'task': str(task.name),
                'directory': str(directory),
                'checkpoint': str(directory / 'weights.pt'),
            }
        )
    return exports


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str | None:
    try:
        finished = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return finished.stdout.strip() or None


def resolve_manifest_settings(cfg, args) -> tuple[str | None, str]:
    """CLI wins over cfg.evaluation; effective split default is 'test'."""
    manifest_dir = args.manifest_dir
    if manifest_dir is None:
        manifest_dir = cfg.evaluation.get('manifest_dir', None)
        if isinstance(manifest_dir, str) and not manifest_dir.strip():
            manifest_dir = None
    eval_split = args.eval_split
    if eval_split is None:
        eval_split = cfg.evaluation.get('eval_split', 'test')
    if eval_split not in ('selection', 'test'):
        raise ValueError(
            f'invalid eval split {eval_split!r}: expected selection or test'
        )
    return manifest_dir, eval_split


def find_task_manifest(
    manifest_dir: str, task_name: str, eval_split: str
) -> dict:
    directory = Path(manifest_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f'manifest directory does not exist: {directory}')
    pattern = f'{task_name}_{eval_split}_seed*_n*.json'
    matches = sorted(directory.glob(pattern))
    if not matches:
        available = sorted(path.name for path in directory.glob('*.json'))
        raise ValueError(
            f'no evaluation manifest for task {task_name!r} ({eval_split} '
            f'split): expected {directory / pattern}; available files: '
            f'{available}'
        )
    if len(matches) > 1:
        raise ValueError(
            f'ambiguous evaluation manifest for task {task_name!r} '
            f'({eval_split} split): matched {matches}; the directory must '
            'hold exactly one <task>_<split>_seed*_n*.json file per task'
        )
    manifest_path = matches[0]
    payload = json.loads(manifest_path.read_text())
    if 'entries' not in payload or not isinstance(payload['entries'], list):
        raise ValueError(f'manifest {manifest_path} has no entries list')
    return {
        'path': str(manifest_path),
        'num_entries': len(payload['entries']),
        'sha256': sha256_file(manifest_path),
    }


def evaluation_command(
    project_root: Path, cfg, task, export, manifest: dict | None = None
) -> list[str]:
    plan_config = str(task.get('plan_config', task.name))
    result_name = f'{task.name}_results.json'
    export['result'] = str(Path(export['directory']) / result_name)
    history_size = int(cfg.get('wm', {}).get('history_size', 3))
    num_eval = int(cfg.evaluation.num_eval)
    if manifest is not None:
        num_eval = int(manifest['num_entries'])
    command = [
        sys.executable,
        str(project_root / 'scripts/plan/eval_wm.py'),
        '--config-name',
        plan_config,
        f'policy={export["checkpoint"]}',
        f'eval.dataset_name={task.evaluation_dataset}',
        f'eval.num_eval={num_eval}',
        f'++plan_config.history_len={history_size}',
        f'++eval.video={str(bool(cfg.evaluation.get("video", False))).lower()}',
        f'seed={int(cfg.evaluation.seed)}',
        f'output.filename={Path(result_name).with_suffix(".txt")}',
        f'++output.directory={export["directory"]}',
        '++output.append=false',
    ]
    if manifest is not None:
        # '++' append prefix: most plan configs do not declare
        # eval.manifest_path, and a bare override fails hydra parsing there.
        command.append(f'++eval.manifest_path={manifest["path"]}')
    return command


def run_evaluations(project_root, cfg, exports, devices, manifests):
    processes = []
    zipped = zip(cfg.tasks, exports, devices, manifests, strict=True)
    for task, export, device, manifest in zipped:
        log_path = Path(export['directory']) / 'evaluation.log'
        stream = log_path.open('w')
        environment = os.environ.copy()
        environment.update(
            {
                'CUDA_VISIBLE_DEVICES': str(device),
                'STABLEWM_HOME': str(
                    Path(cfg.paths.dataset_cache).expanduser().resolve()
                ),
                'LOCAL_DATASET_DIR': str(
                    Path(cfg.paths.dataset_cache).expanduser().resolve()
                ),
                'PYTHONUNBUFFERED': '1',
            }
        )
        process = subprocess.Popen(
            evaluation_command(project_root, cfg, task, export, manifest),
            cwd=project_root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, stream, log_path, export))
    for process, stream, log_path, export in processes:
        return_code = process.wait()
        stream.close()
        if return_code:
            raise RuntimeError(
                f'{export["task"]} evaluation failed; see {log_path}'
            )
        result_path = Path(export['result'])
        result = json.loads(result_path.read_text())
        export['metrics'] = result['metrics']
        export['evaluation_seconds'] = result['evaluation_time_seconds']


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    manifest_dir, eval_split = resolve_manifest_settings(cfg, args)
    checkpoint = Path(
        args.checkpoint or Path(cfg.paths.output_dir) / 'weights_final.pt'
    ).expanduser().resolve()
    root = Path(cfg.paths.output_dir).expanduser().resolve() / 'task_evaluation'
    if args.export_root is not None:
        root = Path(args.export_root).expanduser().resolve()
    manifests: list[dict] | None = None
    if manifest_dir is not None:
        manifests = [
            find_task_manifest(manifest_dir, str(task.name), eval_split)
            for task in cfg.tasks
        ]
    exports = export_tasks(cfg, checkpoint, root)
    if manifests is not None:
        for export, manifest in zip(exports, manifests, strict=True):
            export['manifest'] = {
                'path': manifest['path'],
                'sha256': manifest['sha256'],
                'num_entries': manifest['num_entries'],
                'split': eval_split,
            }
    if not args.export_only:
        configured = args.devices or ','.join(
            str(value) for value in cfg.evaluation.devices
        )
        devices = [value.strip() for value in configured.split(',') if value.strip()]
        if len(devices) < len(exports):
            raise ValueError('one evaluation device per task is required')
        run_evaluations(
            Path(__file__).resolve().parents[2],
            cfg,
            exports,
            devices[: len(exports)],
            manifests if manifests is not None else [None] * len(exports),
        )
    summary = {
        'shared_source_checkpoint': str(checkpoint),
        'run_label': args.run_label,
        'eval_split': eval_split,
        'manifest_dir': (
            str(Path(manifest_dir).expanduser().resolve())
            if manifest_dir is not None
            else None
        ),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(Path(__file__).resolve().parents[2]),
        'config': str(Path(args.config)),
        'checkpoint_sha256': sha256_file(checkpoint),
        'tasks': exports,
    }
    (root / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n'
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
