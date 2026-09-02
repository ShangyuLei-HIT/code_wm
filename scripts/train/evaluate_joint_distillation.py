"""Export phase checkpoints and evaluate them with a configured MPC task."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


MODULE_NAMES = (
    'student_encoder',
    'projector',
    'adapter',
    'action_encoder',
    'predictor',
    'pred_proj',
)
DEPLOYMENT_NAMES = {
    'student_encoder': 'encoder',
    'projector': 'projector',
    'adapter': 'adapter',
    'action_encoder': 'action_encoder',
    'predictor': 'predictor',
    'pred_proj': 'pred_proj',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/vq_lewm_joint_distillation.yaml',
    )
    parser.add_argument(
        '--devices',
        default=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        help='Comma-separated physical GPU ids used in parallel.',
    )
    parser.add_argument(
        '--evaluation-root',
        default=None,
        help='Optional output root, separate from the training directory.',
    )
    return parser.parse_args()


def atomic_json_dump(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def build_deployment_config(cfg) -> dict:
    model = OmegaConf.to_container(cfg.model, resolve=True)
    return {
        '_target_': 'stable_worldmodel.wm.vq_lewm.deployment.DistilledLeWM',
        'encoder': model['student_encoder'],
        'projector': model['projector'],
        'adapter': model['adapter'],
        'action_encoder': model['action_encoder'],
        'predictor': model['predictor'],
        'pred_proj': model['pred_proj'],
    }


def modules_from_phase_checkpoint(payload: dict) -> dict:
    state = payload['model']
    modules = {}
    for name in MODULE_NAMES:
        prefix = f'model.{name}.'
        component = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not component:
            raise RuntimeError(f'{prefix} is absent from phase checkpoint')
        modules[name] = component
    return modules


def flatten_deployment_modules(modules: dict) -> dict:
    state = {}
    for source_name, target_name in DEPLOYMENT_NAMES.items():
        if source_name not in modules:
            raise RuntimeError(f'{source_name!r} is absent from export')
        for key, value in modules[source_name].items():
            state[f'{target_name}.{key}'] = value
    return state


def source_for_stage(output_dir: Path, stage: str) -> Path:
    if stage == 'final':
        return output_dir / 'weights_final.pt'
    match = re.fullmatch(r'phase([1-3])', stage)
    if match is None:
        raise ValueError(f'unsupported evaluation stage: {stage!r}')
    return output_dir / f'{stage}_last.ckpt'


def export_stage(cfg, stage: str, evaluation_root: Path) -> dict:
    output_dir = Path(cfg.paths.output_dir).expanduser().resolve()
    source = source_for_stage(output_dir, stage)
    if not source.is_file():
        raise FileNotFoundError(f'evaluation source is missing: {source}')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    modules = (
        payload['modules']
        if stage == 'final'
        else modules_from_phase_checkpoint(payload)
    )
    state = flatten_deployment_modules(modules)
    deployment_config = build_deployment_config(cfg)

    # Instantiate and strictly load before saving: malformed exports fail early.
    model = instantiate(deployment_config)
    model.load_state_dict(state, strict=True)
    del model

    stage_dir = evaluation_root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    weights_path = stage_dir / 'weights.pt'
    temporary_weights = stage_dir / '.weights.pt.tmp'
    torch.save(state, temporary_weights)
    os.replace(temporary_weights, weights_path)
    temporary_config = stage_dir / '.config.json.tmp'
    temporary_config.write_text(
        json.dumps(deployment_config, indent=2) + '\n'
    )
    os.replace(temporary_config, stage_dir / 'config.json')
    result = {
        'stage': stage,
        'source_checkpoint': str(source),
        'checkpoint': str(weights_path),
        'directory': str(stage_dir),
    }
    validation = payload.get('validation_metrics', {})
    if stage == 'final':
        final_metrics = output_dir / 'final_evaluation.json'
        if final_metrics.is_file():
            validation = json.loads(final_metrics.read_text())
    if 'validate/pred_mixed_mse' in validation:
        result['validation_pred_mixed_mse'] = float(
            validation['validate/pred_mixed_mse']
        )
    return result


def parse_task_result_details(path: Path) -> dict:
    json_path = path.with_suffix('.json')
    if json_path.is_file():
        payload = json.loads(json_path.read_text())
        metrics = payload['metrics']
        return {
            'success_rate': float(metrics['success_rate']),
            'evaluation_time_seconds': payload.get(
                'evaluation_time_seconds'
            ),
            'episode_successes': [
                bool(value)
                for value in metrics.get('episode_successes', [])
            ],
            'row_indices': [
                int(value) for value in payload.get('row_indices', [])
            ],
            'manifest_path': payload.get('manifest_path'),
            'json_result_file': str(json_path),
        }
    text = path.read_text()
    rates = re.findall(
        r"['\"]success_rate['\"]\s*:\s*([0-9.eE+-]+)", text
    )
    if not rates:
        raise RuntimeError(f'no success_rate found in {path}')
    times = re.findall(r'evaluation_time:\s*([0-9.eE+-]+)', text)
    return {
        'success_rate': float(rates[-1]),
        'evaluation_time_seconds': float(times[-1]) if times else None,
        'episode_successes': [],
        'row_indices': [],
        'manifest_path': None,
        'json_result_file': None,
    }


def parse_task_result(path: Path) -> tuple[float, float | None]:
    details = parse_task_result_details(path)
    return (
        details['success_rate'],
        details['evaluation_time_seconds'],
    )


def evaluation_command(project_root: Path, cfg, item: dict) -> list[str]:
    manifest_value = item.get('manifest')
    manifest = (
        Path(manifest_value).expanduser().resolve()
        if manifest_value
        else None
    )
    num_eval = int(cfg.evaluation.num_eval)
    history_size = int(cfg.wm.history_size)
    if manifest is not None:
        manifest_payload = json.loads(manifest.read_text())
        num_eval = int(manifest_payload['count'])
        item['manifest'] = str(manifest)
    plan_config = str(cfg.evaluation.get('plan_config', 'pusht'))
    result_prefix = str(cfg.evaluation.get('result_prefix', plan_config))
    filename = f'{result_prefix}_results_{item["stage"]}_{num_eval}.txt'
    item['result_file'] = str(Path(item['directory']) / filename)
    item['num_eval'] = num_eval
    command = [
        sys.executable,
        str(project_root / 'scripts/plan/eval_wm.py'),
        '--config-name',
        plan_config,
        f'policy={item["checkpoint"]}',
        f'eval.dataset_name={cfg.evaluation.dataset_name}',
        f'eval.num_eval={num_eval}',
        f'++plan_config.history_len={history_size}',
        f'++eval.video={str(bool(item.get("video", False))).lower()}',
        f'seed={int(cfg.evaluation.seed)}',
        f'output.filename={filename}',
        f'++output.directory={Path(item["directory"]).resolve()}',
        '++output.append=false',
    ]
    if manifest is not None:
        command.append(f'++eval.manifest_path={manifest}')
    return command


def evaluate_batch(project_root: Path, cfg, batch: list[dict], devices: list[str]):
    processes = []
    for item, device in zip(batch, devices, strict=True):
        stage_dir = Path(item['directory'])
        log_path = stage_dir / 'eval.log'
        log_stream = log_path.open('w')
        env = os.environ.copy()
        env.update(
            {
                'CUDA_VISIBLE_DEVICES': device,
                'STABLEWM_HOME': str(Path(cfg.paths.dataset_cache).resolve()),
                'LOCAL_DATASET_DIR': str(
                    Path(cfg.paths.dataset_cache).resolve()
                ),
                'PYTHONUNBUFFERED': '1',
            }
        )
        command = evaluation_command(project_root, cfg, item)
        print(
            f'[task-eval] stage={item["stage"]} gpu={device}: '
            + ' '.join(command),
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((item, process, log_stream, log_path, time.time()))

    errors = []
    for item, process, log_stream, log_path, started in processes:
        return_code = process.wait()
        log_stream.close()
        item['wall_time_seconds'] = time.time() - started
        item['log_file'] = str(log_path)
        if return_code:
            errors.append(
                f'{item["stage"]} exited with {return_code}; log={log_path}'
            )
            continue
        details = parse_task_result_details(Path(item['result_file']))
        item.update(details)
        item['successes'] = sum(details['episode_successes'])
        if not details['episode_successes']:
            item['successes'] = round(
                details['success_rate'] / 100.0 * int(item['num_eval'])
            )
        item['reported_evaluation_seconds'] = details[
            'evaluation_time_seconds'
        ]
        print(
            f'[task-eval] stage={item["stage"]} '
            f'success_rate={details["success_rate"]:.2f}%',
            flush=True,
        )
    if errors:
        raise RuntimeError('; '.join(errors))


def add_baseline_deltas(cfg, stages: list[dict]) -> None:
    configured = cfg.evaluation.get('baselines')
    if configured is None:
        baselines = {}
    elif OmegaConf.is_config(configured):
        baselines = OmegaConf.to_container(configured, resolve=True)
    else:
        baselines = dict(configured)
    for item in stages:
        if 'success_rate' not in item:
            continue
        item['baseline_delta_percentage_points'] = {
            name: item['success_rate'] - float(rate)
            for name, rate in baselines.items()
        }


def add_best_stage(summary: dict) -> None:
    completed = [
        item for item in summary['stages'] if 'success_rate' in item
    ]
    if not completed:
        return
    order = {'phase1': 0, 'phase2': 1, 'final': 2}
    best = max(
        completed,
        key=lambda item: (
            item['success_rate'],
            -float(item.get('validation_pred_mixed_mse', float('inf'))),
            -order.get(item['stage'], 99),
        ),
    )
    summary['best_stage'] = {
        'stage': best['stage'],
        'success_rate': best['success_rate'],
        'successes': best['successes'],
        'checkpoint': best['checkpoint'],
    }


def evaluate_heldout(
    project_root: Path,
    cfg,
    summary: dict,
    devices: list[str],
    evaluation_root: Path,
) -> None:
    manifests = [
        str(value)
        for value in cfg.evaluation.get('test_manifests', [])
        if value
    ]
    if not manifests:
        return
    if 'best_stage' not in summary:
        raise RuntimeError('held-out evaluation requires a selected stage')
    checkpoint = summary['best_stage']['checkpoint']
    items = []
    for index, manifest in enumerate(manifests):
        directory = evaluation_root / 'heldout_test' / f'shard{index:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        items.append(
            {
                'stage': (
                    f'{summary["best_stage"]["stage"]}_'
                    f'test_shard{index:02d}'
                ),
                'checkpoint': checkpoint,
                'directory': str(directory),
                'manifest': manifest,
                'video': bool(
                    cfg.evaluation.get('heldout_video', False)
                ),
            }
        )
    for offset in range(0, len(items), len(devices)):
        batch = items[offset : offset + len(devices)]
        evaluate_batch(project_root, cfg, batch, devices[: len(batch)])

    outcomes = [
        value
        for item in items
        for value in item.get('episode_successes', [])
    ]
    row_indices = [
        value for item in items for value in item.get('row_indices', [])
    ]
    if len(outcomes) != sum(int(item['num_eval']) for item in items):
        raise RuntimeError('held-out shards are missing per-episode outcomes')
    if len(set(row_indices)) != len(row_indices):
        raise RuntimeError('held-out shard row indices overlap')
    successes = sum(outcomes)
    summary['heldout_test'] = {
        'status': 'complete',
        'selected_stage': summary['best_stage']['stage'],
        'checkpoint': checkpoint,
        'num_eval': len(outcomes),
        'successes': successes,
        'success_rate': successes / len(outcomes) * 100.0,
        'episode_successes': outcomes,
        'row_indices': row_indices,
        'shards': items,
    }


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    if not bool(cfg.evaluation.enabled):
        print('[task-eval] disabled by configuration', flush=True)
        return

    devices = [item.strip() for item in args.devices.split(',') if item.strip()]
    if not devices:
        raise RuntimeError('automatic task evaluation requires at least one GPU')
    stages = [str(stage) for stage in cfg.evaluation.stages]
    configured_root = cfg.evaluation.get('output_root')
    evaluation_root = (
        Path(args.evaluation_root).expanduser().resolve()
        if args.evaluation_root
        else (
            Path(configured_root).expanduser().resolve()
            if configured_root
            else (
                Path(cfg.paths.output_dir).expanduser().resolve()
                / 'task_evaluation'
            )
        )
    )
    summary_path = evaluation_root / 'summary.json'
    summary = {
        'status': 'running',
        'config': str(config_path),
        'dataset_name': str(cfg.evaluation.dataset_name),
        'seed': int(cfg.evaluation.seed),
        'rng_seed_scope': ['python', 'numpy', 'torch', 'cuda'],
        'num_eval': int(cfg.evaluation.num_eval),
        'devices': devices,
        'stages': [],
    }
    atomic_json_dump(summary, summary_path)
    try:
        selection_manifest = cfg.evaluation.get('selection_manifest')
        for stage in stages:
            item = export_stage(cfg, stage, evaluation_root)
            item['manifest'] = (
                str(selection_manifest) if selection_manifest else None
            )
            item['video'] = bool(
                cfg.evaluation.get('selection_video', True)
            )
            summary['stages'].append(item)
            atomic_json_dump(summary, summary_path)
        for offset in range(0, len(summary['stages']), len(devices)):
            batch = summary['stages'][offset : offset + len(devices)]
            evaluate_batch(project_root, cfg, batch, devices[: len(batch)])
            atomic_json_dump(summary, summary_path)
        add_baseline_deltas(cfg, summary['stages'])
        add_best_stage(summary)
        atomic_json_dump(summary, summary_path)
        evaluate_heldout(
            project_root, cfg, summary, devices, evaluation_root
        )
        summary['status'] = 'complete'
        atomic_json_dump(summary, summary_path)
        print(f'[task-eval] summary={summary_path}', flush=True)
    except Exception as error:
        summary['status'] = 'failed'
        summary['error'] = f'{type(error).__name__}: {error}'
        atomic_json_dump(summary, summary_path)
        raise


if __name__ == '__main__':
    main()
