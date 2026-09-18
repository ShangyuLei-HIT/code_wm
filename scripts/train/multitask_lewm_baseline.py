"""Native continuous, balanced PushT/Two-Room LeWM baseline (M3)."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from stable_worldmodel.data import column_normalizer
from stable_worldmodel.wm.loss import SIGReg

SIGREG_MODES = ('shared', 'per_task', 'none')

MATCHED_INT_FIELDS = (
    'expected_optimizer_steps',
    'expected_per_task_batch',
    'expected_world_size',
    'expected_global_batch',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/multitask_lewm_baseline.yaml',
    )
    parser.add_argument(
        '--run_id',
        default=None,
        help='Run identifier recorded in run_manifest.json; defaults to the '
        'config filename stem.',
    )
    parser.add_argument(
        '--train_seed',
        type=int,
        default=None,
        help='Override cfg.seed and rewrite the seed<old> token inside '
        'cfg.paths.output_dir.',
    )
    parser.add_argument(
        '--max_optimizer_steps',
        type=int,
        default=None,
        help='Stop after this many optimizer steps; recorded as '
        'cfg.trainer.max_optimizer_steps.',
    )
    parser.add_argument(
        '--per_task_batch',
        type=int,
        default=None,
        help='Override cfg.data.batch_size_per_task_per_gpu.',
    )
    parser.add_argument(
        '--eval_manifest',
        default=None,
        help='Manifest path recorded as cfg.evaluation.eval_manifest.',
    )
    parser.add_argument(
        '--save_config',
        default=None,
        help='Save the fully resolved config to this path before training.',
    )
    parser.add_argument(
        '--sigreg_mode',
        choices=SIGREG_MODES,
        default=None,
        help='Override cfg.loss.sigreg.mode (shared/per_task/none).',
    )
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Override an arbitrary config key, e.g. '
        '--set trainer.epochs=8. Values are parsed as Python literals '
        'with a raw-string fallback.',
    )
    return parser.parse_args()


def apply_overrides(cfg, overrides):
    """Apply repeatable --set key.subkey=value overrides to the config."""
    for item in overrides:
        key, separator, raw = item.partition('=')
        if not separator or not key:
            raise ValueError(
                f'invalid --set item {item!r}: expected KEY=VALUE'
            )
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        with open_dict(cfg):
            OmegaConf.update(cfg, key, value)
    return cfg


def apply_train_seed(cfg, seed):
    """Set cfg.seed and rewrite the seed token in cfg.paths.output_dir.

    Returns the effective training seed used.
    """
    old_seed = int(cfg.seed)
    new_seed = int(seed)
    with open_dict(cfg):
        cfg.seed = new_seed
        output_dir = str(cfg.paths.output_dir)
        old_token = re.escape(f'seed{old_seed}')
        pattern = re.compile(rf'(?<![0-9]){old_token}(?![0-9])')
        if pattern.search(output_dir):
            output_dir = pattern.sub(f'seed{new_seed}', output_dir)
        else:
            output_dir = f'{output_dir}_seed{new_seed}'
        cfg.paths.output_dir = output_dir
    return int(cfg.seed)


def setup_distributed():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group('nccl', device_id=device)
    return rank, world_size, local_rank, device


def image_preprocessor(img_size: int):
    return spt.data.transforms.Compose(
        dt.transforms.ToImage(
            **dt.dataset_stats.ImageNet,
            source='pixels',
            target='pixels',
        ),
        dt.transforms.Resize(img_size, source='pixels', target='pixels'),
    )


class TaskSubset(Dataset):
    def __init__(self, base, indices, task_id: int):
        self.base = base
        self.indices = indices
        self.task_id = int(task_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.base[int(self.indices[index])]
        return {
            'pixels': sample['pixels'],
            'action': sample['action'],
            'task_id': self.task_id,
        }


class BalancedLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        self.length = min(map(len, loaders))

    def __len__(self):
        return self.length

    def __iter__(self):
        for batches in zip(*(iter(loader) for loader in self.loaders)):
            combined = {}
            for key in batches[0]:
                values = [batch[key] for batch in batches]
                if key == 'action':
                    width = max(value.size(-1) for value in values)
                    values = [
                        F.pad(value, (0, width - value.size(-1)))
                        for value in values
                    ]
                combined[key] = torch.cat(values, dim=0)
            yield combined


class NativeObjective(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.sigreg = SIGReg(**OmegaConf.to_container(cfg.loss.sigreg.kwargs))
        self.history = int(cfg.wm.history_size)
        self.sigreg_weight = float(cfg.loss.sigreg.weight)
        self.sigreg_mode = str(
            OmegaConf.select(cfg, 'loss.sigreg.mode', default='shared')
        )
        if self.sigreg_mode not in SIGREG_MODES:
            raise ValueError(
                f'loss.sigreg.mode must be one of {SIGREG_MODES}, '
                f'got {self.sigreg_mode!r}'
            )

    def sigreg_shared(self, embedding):
        return self.sigreg(embedding.transpose(0, 1))

    def sigreg_per_task(self, embedding, task_ids):
        per_task_losses = []
        for task_id in task_ids.unique().tolist():
            task_embedding = embedding[task_ids == task_id]
            if task_embedding.size(0) < 2:
                continue  # SIGReg needs at least 2 sequences per slice.
            per_task_losses.append(
                self.sigreg(task_embedding.transpose(0, 1))
            )
        if not per_task_losses:
            return embedding.new_zeros(())
        return torch.stack(per_task_losses).mean()

    def forward(self, batch):
        embedding = self.model.encode({'pixels': batch['pixels']})['emb']
        prediction = self.model.predict_actions(
            embedding[:, : self.history],
            batch['action'][:, : self.history],
            batch['task_id'],
        )
        target = embedding[:, 1 : self.history + 1]
        prediction_loss = F.mse_loss(prediction, target)
        if self.sigreg_mode == 'none':
            regularization = embedding.new_zeros(())
        elif self.sigreg_mode == 'per_task':
            regularization = self.sigreg_per_task(
                embedding, batch['task_id'].reshape(-1)
            )
        else:
            regularization = self.sigreg_shared(embedding)
        return {
            'prediction_loss': prediction_loss,
            'sigreg_loss': regularization,
            'loss': prediction_loss + self.sigreg_weight * regularization,
        }


def build_loaders(cfg, rank, world_size):
    cache_root = Path(cfg.paths.multitask_cache_dir).expanduser().resolve()
    train_loaders = []
    validation_loaders = []
    samplers = []
    action_dims = []
    workers = int(cfg.data.cpu_workers_total) // world_size // len(cfg.tasks)
    for task_id, task in enumerate(cfg.tasks):
        base = swm.data.load_dataset(
            task.dataset,
            transform=None,
            num_steps=int(cfg.data.num_steps),
            frameskip=int(task.frameskip),
            keys_to_load=['pixels', 'action'],
        )
        action_dims.append(int(task.frameskip) * int(base.get_dim('action')))
        base.transform = spt.data.transforms.Compose(
            image_preprocessor(int(cfg.data.img_size)),
            column_normalizer(base, 'action', 'action'),
        )
        task_root = cache_root / 'tasks' / str(task.name)
        if (task_root / 'train_indices.npy').exists():
            train_indices = np.load(task_root / 'train_indices.npy')
            validation_indices = np.load(
                task_root / 'validation_indices.npy'
            )
        else:
            generator = torch.Generator().manual_seed(
                int(cfg.data.split_seed) + task_id
            )
            indices = torch.randperm(len(base), generator=generator).numpy()
            train_count = int(len(base) * float(cfg.data.train_fraction))
            train_indices = indices[:train_count]
            validation_indices = indices[train_count:]
            if bool(cfg.smoke.enabled):
                train_indices = train_indices[
                    : int(cfg.smoke.train_samples_per_task)
                ]
                validation_indices = validation_indices[
                    : int(cfg.smoke.validation_samples_per_task)
                ]
        train_set = TaskSubset(base, train_indices, task_id)
        validation_set = TaskSubset(base, validation_indices, task_id)
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(cfg.seed),
            drop_last=True,
        )
        validation_sampler = DistributedSampler(
            validation_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        common = {
            'batch_size': int(cfg.data.batch_size_per_task_per_gpu),
            'num_workers': workers,
            'pin_memory': True,
            'persistent_workers': workers > 0,
        }
        train_loaders.append(
            DataLoader(train_set, sampler=train_sampler, drop_last=True, **common)
        )
        validation_loaders.append(
            DataLoader(
                validation_set,
                sampler=validation_sampler,
                drop_last=False,
                **common,
            )
        )
        samplers.append(train_sampler)
    return (
        BalancedLoader(train_loaders),
        validation_loaders,
        samplers,
        max(action_dims),
    )


def move_batch(batch, device):
    return {
        key: (
            torch.nan_to_num(value.to(device, non_blocking=True), 0.0)
            if key == 'action'
            else value.to(device, non_blocking=True)
        )
        for key, value in batch.items()
    }


def reduce_sum(value):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value)
    return value


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    history = int(cfg.wm.history_size)
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    max_batches = (
        int(cfg.smoke.batches_per_epoch) if bool(cfg.smoke.enabled) else None
    )
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            embedding = model.encode({'pixels': batch['pixels']})['emb']
            prediction = model.predict_actions(
                embedding[:, :history],
                batch['action'][:, :history],
                batch['task_id'],
            )
        batch_size = len(embedding)
        totals[0] += float(
            F.mse_loss(
                prediction.float(),
                embedding[:, 1 : history + 1].float(),
            )
        ) * batch_size
        totals[1] += batch_size
    reduce_sum(totals)
    return {'prediction_mse': float(totals[0] / totals[1].clamp_min(1))}


def cosine_lr(step, total_steps, base_lr, min_lr, warmup_fraction):
    warmup = int(total_steps * warmup_fraction)
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def freeze_projector_batchnorm(model):
    for module in model.projector.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def parameter_counts(model):
    names = (
        'encoder',
        'projector',
        'adapter',
        'action_encoder',
        'task_embedding',
        'predictor',
        'pred_proj',
    )
    result = {
        name: sum(parameter.numel() for parameter in getattr(model, name).parameters())
        for name in names
    }
    result['trainable_total'] = sum(result.values())
    return result


def atomic_save(payload, path):
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def git_commit():
    try:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def relative_diff(expected, actual) -> float:
    expected = float(expected)
    actual = float(actual)
    if expected == 0.0:
        return 0.0 if actual == 0.0 else float('inf')
    return abs(actual - expected) / abs(expected)


def validate_matched_block(cfg):
    """Fail fast on a malformed cfg.matched section (before training)."""
    matched_cfg = OmegaConf.select(cfg, 'matched', default=None)
    if matched_cfg is None:
        return
    reference = OmegaConf.select(matched_cfg, 'reference', default=None)
    if not isinstance(reference, str) or not reference:
        raise ValueError(
            f'cfg.matched.reference must be a non-empty string, '
            f'got {reference!r}'
        )
    for field in MATCHED_INT_FIELDS:
        value = OmegaConf.select(matched_cfg, field, default=None)
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = None
        if number is None or number != value or number < 1:
            raise ValueError(
                f'cfg.matched.{field} must be a positive int, '
                f'got {value!r}'
            )
    tolerance = OmegaConf.select(matched_cfg, 'tolerance', default=None)
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError):
        tolerance_value = -1.0
    if tolerance is None or tolerance_value != tolerance:
        raise ValueError(
            f'cfg.matched.tolerance must be a non-negative number, '
            f'got {tolerance!r}'
        )
    if tolerance_value < 0.0:
        raise ValueError(
            f'cfg.matched.tolerance must be >= 0, got {tolerance!r}'
        )
    results_dir = OmegaConf.select(matched_cfg, 'results_dir', default=None)
    if results_dir is not None and not isinstance(results_dir, str):
        raise ValueError(
            f'cfg.matched.results_dir must be a string or null, '
            f'got {results_dir!r}'
        )


def save_resolved_config(cfg, path):
    """Atomically save the fully resolved config to an arbitrary path."""
    path = Path(path).expanduser()
    if path.is_dir():
        raise ValueError(
            f'--save_config path {str(path)!r} is an existing directory'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    OmegaConf.save(cfg, temporary, resolve=True)
    os.replace(temporary, path)
    return path


def write_matched_summary(cfg, manifest, output_dir, metrics_path):
    """Compare achieved exposure against cfg.matched; never raises."""
    matched_cfg = OmegaConf.select(cfg, 'matched', default=None)
    if matched_cfg is None:
        return None
    num_tasks = len(cfg.tasks)
    per_task_batch = int(manifest['per_task_batch'])
    world_size = int(manifest['world_size'])
    expected = {
        'optimizer_steps': int(matched_cfg.expected_optimizer_steps),
        'per_task_batch': int(matched_cfg.expected_per_task_batch),
        'world_size': int(matched_cfg.expected_world_size),
        'global_batch': int(matched_cfg.expected_global_batch),
    }
    actual = {
        'optimizer_steps': int(manifest['optimizer_steps_completed']),
        'per_task_batch': per_task_batch,
        'world_size': world_size,
        'global_batch': per_task_batch * num_tasks * world_size,
    }
    tolerance = float(matched_cfg.tolerance)
    checks = {
        name: {
            'expected': expected[name],
            'actual': actual[name],
            'rel_diff': relative_diff(expected[name], actual[name]),
            'within_tolerance': bool(
                relative_diff(expected[name], actual[name]) <= tolerance
            ),
        }
        for name in expected
    }
    summary = {
        'run_id': manifest['run_id'],
        'reference': str(matched_cfg.reference),
        'matched': all(check['within_tolerance'] for check in checks.values()),
        'tolerance': tolerance,
        'checks': checks,
        'actuals': {
            'optimizer_steps': actual['optimizer_steps'],
            'epochs_completed': int(manifest['epochs_completed']),
            'per_task_batch': per_task_batch,
            'world_size': world_size,
            'num_tasks': num_tasks,
            'global_batch': actual['global_batch'],
            'max_optimizer_steps': manifest['max_optimizer_steps'],
            'samples_per_task': manifest['samples_per_task'],
        },
        'train_seed': manifest['train_seed'],
        'config_sha256': manifest['config_sha256'],
        'written_at': utc_iso(time.time()),
    }
    text = json.dumps(summary, indent=2) + '\n'
    (output_dir / 'summary.json').write_text(text)
    results_dir = OmegaConf.select(matched_cfg, 'results_dir', default=None)
    if isinstance(results_dir, str) and results_dir:
        try:
            results_path = Path(results_dir).expanduser()
            results_path.mkdir(parents=True, exist_ok=True)
            (results_path / 'summary.json').write_text(text)
            if metrics_path.exists():
                shutil.copy2(metrics_path, results_path / 'metrics.jsonl')
        except OSError as error:
            print(
                f'WARNING: could not write matched results to '
                f'{results_dir!r}: {error}',
                flush=True,
            )
    print(
        'matched exposure check: '
        + json.dumps(
            {
                name: {
                    'expected': check['expected'],
                    'actual': check['actual'],
                    'ok': check['within_tolerance'],
                }
                for name, check in checks.items()
            },
            sort_keys=True,
        )
        + f' -> matched={summary["matched"]}',
        flush=True,
    )
    return summary


def main():
    started_at = time.time()
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    run_id = args.run_id if args.run_id is not None else Path(args.config).stem
    # Apply CLI overrides before anything heavy (CUDA, datasets, models).
    if args.train_seed is not None:
        old_seed = int(cfg.seed)
        effective_seed = apply_train_seed(cfg, args.train_seed)
        results_dir = OmegaConf.select(
            cfg, 'matched.results_dir', default=None
        )
        if isinstance(results_dir, str) and f'seed{old_seed}' in results_dir:
            with open_dict(cfg):
                cfg.matched.results_dir = results_dir.replace(
                    f'seed{old_seed}', f'seed{effective_seed}'
                )
    if args.max_optimizer_steps is not None:
        if args.max_optimizer_steps < 1:
            raise ValueError(
                f'--max_optimizer_steps must be >= 1, '
                f'got {args.max_optimizer_steps}'
            )
        with open_dict(cfg):
            cfg.trainer.max_optimizer_steps = int(args.max_optimizer_steps)
    if args.per_task_batch is not None:
        if args.per_task_batch < 1:
            raise ValueError(
                f'--per_task_batch must be >= 1, got {args.per_task_batch}'
            )
        cfg.data.batch_size_per_task_per_gpu = int(args.per_task_batch)
    if args.sigreg_mode is not None:
        with open_dict(cfg):
            cfg.loss.sigreg.mode = args.sigreg_mode
    if args.eval_manifest is not None:
        with open_dict(cfg):
            cfg.evaluation.eval_manifest = args.eval_manifest
    apply_overrides(cfg, args.set)
    sigreg_mode = str(
        OmegaConf.select(cfg, 'loss.sigreg.mode', default='shared')
    )
    if sigreg_mode not in SIGREG_MODES:
        raise ValueError(
            f'loss.sigreg.mode must be one of {SIGREG_MODES}, '
            f'got {sigreg_mode!r}'
        )
    max_optimizer_steps = OmegaConf.select(
        cfg, 'trainer.max_optimizer_steps', default=None
    )
    if max_optimizer_steps is not None:
        max_optimizer_steps = int(max_optimizer_steps)
        if max_optimizer_steps < 1:
            raise ValueError(
                f'trainer.max_optimizer_steps must be >= 1, '
                f'got {max_optimizer_steps}'
            )
    eval_manifest = OmegaConf.select(
        cfg, 'evaluation.eval_manifest', default=None
    )
    validate_matched_block(cfg)
    rank, world_size, local_rank, device = setup_distributed()
    seed = int(cfg.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['LOCAL_DATASET_DIR'] = str(cfg.paths.dataset_cache)
    os.environ['STABLEWM_HOME'] = str(cfg.paths.dataset_cache)
    train_loader, validation_loaders, samplers, action_dim = build_loaders(
        cfg, rank, world_size
    )
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = action_dim
    model = hydra.utils.instantiate(cfg.model).to(device)
    objective = NativeObjective(model, cfg).to(device)
    ddp = (
        DDP(
            objective,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        if world_size > 1
        else objective
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
        betas=tuple(cfg.optimizer.betas),
    )
    output_dir = Path(cfg.paths.output_dir).expanduser().resolve()
    config_path = output_dir / 'config.yaml'
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_path)
        config_text = config_path.read_text()
        if args.save_config is not None:
            saved_config_path = save_resolved_config(cfg, args.save_config)
            config_text = saved_config_path.read_text()
        manifest = {
            'baseline': 'M3_native_continuous',
            'uses_teacher': False,
            'uses_alignment': False,
            'uses_codebook': False,
            'parameter_counts': parameter_counts(model),
            'world_size': world_size,
            'run_id': run_id,
            'git_commit': git_commit(),
            'config_sha256': hashlib.sha256(config_text.encode()).hexdigest(),
            'train_seed': int(cfg.seed),
            'sigreg_mode': sigreg_mode,
            'per_task_batch': int(cfg.data.batch_size_per_task_per_gpu),
            'max_optimizer_steps': max_optimizer_steps,
            'optimizer_steps_completed': None,
            'epochs_completed': None,
            'samples_per_task': None,
            'started_at': utc_iso(started_at),
            'finished_at': None,
            'gpu_hours': None,
            'evaluation_manifest': (
                str(eval_manifest) if eval_manifest is not None else None
            ),
        }
        (output_dir / 'run_manifest.json').write_text(
            json.dumps(manifest, indent=2) + '\n'
        )
    if world_size > 1:
        dist.barrier()
    steps_per_epoch = len(train_loader)
    if bool(cfg.smoke.enabled):
        steps_per_epoch = min(steps_per_epoch, int(cfg.smoke.batches_per_epoch))
    total_steps = int(cfg.trainer.epochs) * steps_per_epoch
    # max_optimizer_steps only STOPS training early; the LR schedule keeps its
    # natural horizon so a capped run never silently compresses the cosine
    # (same semantics as multitask_vq_lewm_distillation.py).
    global_step = 0
    epochs_completed = 0
    num_tasks = len(cfg.tasks)
    task_samples = torch.zeros(num_tasks, device=device, dtype=torch.int64)
    stop_training = False
    metrics_path = output_dir / 'metrics.jsonl'
    for epoch in range(int(cfg.trainer.epochs)):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        objective.train()
        freeze_projector_batchnorm(model)
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= steps_per_epoch:
                break
            batch = move_batch(batch, device)
            learning_rate = cosine_lr(
                global_step,
                total_steps,
                float(cfg.optimizer.lr),
                float(cfg.optimizer.min_lr),
                float(cfg.optimizer.warmup_fraction),
            )
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = ddp(batch)
            if not torch.isfinite(output['loss']):
                raise FloatingPointError('non-finite M3 loss')
            output['loss'].backward()
            torch.nn.utils.clip_grad_norm_(
                objective.parameters(), float(cfg.trainer.gradient_clip_val)
            )
            optimizer.step()
            counts = torch.bincount(
                batch['task_id'].reshape(-1).long(), minlength=num_tasks
            )
            if counts.numel() != num_tasks:
                raise ValueError(
                    f'batch task_id outside 0..{num_tasks - 1} '
                    f'(histogram length {counts.numel()})'
                )
            task_samples += counts
            totals += torch.tensor(
                [
                    float(output['loss'].detach()),
                    float(output['prediction_loss'].detach()),
                    float(output['sigreg_loss'].detach()),
                ],
                device=device,
                dtype=torch.float64,
            )
            global_step += 1
            if (
                max_optimizer_steps is not None
                and global_step >= max_optimizer_steps
            ):
                stop_training = True
                break
        if (
            max_optimizer_steps is not None
            and global_step >= max_optimizer_steps
        ):
            stop_training = True
        reduce_sum(totals)
        validation = {
            str(task.name): validate(
                model, validation_loaders[index], cfg, device
            )
            for index, task in enumerate(cfg.tasks)
        }
        epochs_completed = epoch + 1
        denominator = max(1, steps_per_epoch * world_size)
        row = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'train/loss': float(totals[0] / denominator),
            'train/prediction_mse': float(totals[1] / denominator),
            'train/sigreg': float(totals[2] / denominator),
            'train/epoch_seconds': time.perf_counter() - started,
            'validation': validation,
        }
        if rank == 0:
            with metrics_path.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            print(json.dumps(row, sort_keys=True), flush=True)
            atomic_save(
                {
                    'model': objective.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'validation': validation,
                },
                output_dir / 'last.ckpt',
            )
        if world_size > 1:
            dist.barrier()
        if stop_training:
            break
    reduce_sum(task_samples)
    finished_at = time.time()
    elapsed_seconds = finished_at - started_at
    samples_per_task = {
        str(task.name): int(count)
        for task, count in zip(cfg.tasks, task_samples.tolist())
    }
    if rank == 0:
        atomic_save(
            {
                'format_version': 1,
                'model_target': (
                    'stable_worldmodel.wm.vq_lewm.multitask.'
                    'MultiTaskDistilledLeWM'
                ),
                'state_dict': model.state_dict(),
                'num_tasks': len(cfg.tasks),
                'task_id_to_name': {
                    str(index): str(task.name)
                    for index, task in enumerate(cfg.tasks)
                },
                'parameter_counts': parameter_counts(model),
            },
            output_dir / 'weights_final.pt',
        )
        manifest.update(
            {
                'optimizer_steps_completed': int(global_step),
                'epochs_completed': int(epochs_completed),
                'samples_per_task': samples_per_task,
                'steps_per_epoch': int(steps_per_epoch),
                'finished_at': utc_iso(finished_at),
                'gpu_hours': elapsed_seconds * world_size / 3600.0,
            }
        )
        (output_dir / 'run_manifest.json').write_text(
            json.dumps(manifest, indent=2) + '\n'
        )
        write_matched_summary(cfg, manifest, output_dir, metrics_path)
        print(f'M3 training complete: {output_dir}', flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
