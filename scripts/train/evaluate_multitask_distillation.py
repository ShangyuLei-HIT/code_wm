"""Export one shared model with per-task defaults and run both MPC suites."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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


def evaluation_command(project_root: Path, cfg, task, export) -> list[str]:
    plan_config = str(task.get('plan_config', task.name))
    result_name = f'{task.name}_results.json'
    export['result'] = str(Path(export['directory']) / result_name)
    history_size = int(cfg.get('wm', {}).get('history_size', 3))
    command = [
        sys.executable,
        str(project_root / 'scripts/plan/eval_wm.py'),
        '--config-name',
        plan_config,
        f'policy={export["checkpoint"]}',
        f'eval.dataset_name={task.evaluation_dataset}',
        f'eval.num_eval={int(cfg.evaluation.num_eval)}',
        f'++plan_config.history_len={history_size}',
        f'++eval.video={str(bool(cfg.evaluation.get("video", False))).lower()}',
        f'seed={int(cfg.evaluation.seed)}',
        f'output.filename={Path(result_name).with_suffix(".txt")}',
        f'++output.directory={export["directory"]}',
        '++output.append=false',
    ]
    return command


def run_evaluations(project_root, cfg, exports, devices):
    processes = []
    for task, export, device in zip(cfg.tasks, exports, devices, strict=True):
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
            evaluation_command(project_root, cfg, task, export),
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
    checkpoint = Path(
        args.checkpoint or Path(cfg.paths.output_dir) / 'weights_final.pt'
    ).expanduser().resolve()
    root = Path(cfg.paths.output_dir).expanduser().resolve() / 'task_evaluation'
    exports = export_tasks(cfg, checkpoint, root)
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
        )
    summary = {
        'shared_source_checkpoint': str(checkpoint),
        'tasks': exports,
    }
    (root / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n'
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
