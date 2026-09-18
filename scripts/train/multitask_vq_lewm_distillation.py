"""Teacher-free balanced training with a shared aligned/fused codebook."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
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
from stable_worldmodel.wm.vq_lewm.distillation import (
    cosine_phase_lr,
    effective_rank_from_moments,
    nearest_code_indices,
    phase_for_epoch,
    resolve_weights_path,
    sequence_teacher_forcing,
    sha256_file,
    sparse_topk_kl,
    teacher_forcing_alpha,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/multitask_vq_lewm.yaml',
    )
    parser.add_argument(
        '--train_seed',
        type=int,
        default=None,
        metavar='N',
        help='set cfg.seed and rewrite the seed token in paths.output_dir',
    )
    parser.add_argument(
        '--save-config',
        default=None,
        metavar='PATH',
        help='write the resolved (post-override) config to this YAML path',
    )
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='override a config key (repeatable); VALUE uses ast.literal_eval '
        'with a raw-string fallback',
    )
    return parser.parse_args()


def apply_overrides(cfg, overrides: list[str]) -> None:
    """Apply --set key.subkey=value overrides to the loaded config."""
    for item in overrides:
        key, separator, value = item.partition('=')
        if not separator or not key:
            raise ValueError(f'--set expects KEY=VALUE, got {item!r}')
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        with open_dict(cfg):
            OmegaConf.update(cfg, key, parsed)


def apply_train_seed(cfg, seed: int) -> int:
    """Set cfg.seed and rewrite the seed token inside paths.output_dir.

    A `seed<old>` token (not embedded in a longer number) becomes `seed<new>`;
    otherwise `_seed<new>` is appended. Returns the effective seed.
    """
    seed = int(seed)
    old_seed = int(cfg.seed)
    with open_dict(cfg):
        cfg.seed = seed
        output_dir = str(cfg.paths.output_dir)
        token = re.escape(f'seed{old_seed}')
        pattern = re.compile(rf'(?<![0-9]){token}(?![0-9])')
        if pattern.search(output_dir):
            output_dir = pattern.sub(f'seed{seed}', output_dir)
        else:
            output_dir = f'{output_dir}_seed{seed}'
        cfg.paths.output_dir = output_dir
    return int(cfg.seed)


def save_resolved_config(cfg, path: Path) -> None:
    """Atomically write the effective config as YAML text."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(OmegaConf.to_yaml(cfg))
    os.replace(temporary, path)


def git_commit_hash(project_root: Path) -> str | None:
    """Best-effort HEAD commit; None when git is unavailable or fails."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group('nccl', device_id=device)
    return rank, world_size, local_rank, device


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
        ).encode()
    ).hexdigest()


def image_preprocessor(img_size: int):
    return spt.data.transforms.Compose(
        dt.transforms.ToImage(
            **dt.dataset_stats.ImageNet,
            source='pixels',
            target='pixels',
        ),
        dt.transforms.Resize(img_size, source='pixels', target='pixels'),
    )


class CachedTaskDataset(Dataset):
    def __init__(self, base, indices, cache_root: Path, task_name: str, split: str, task_id: int):
        self.base = base
        self.indices = indices
        root = cache_root / 'tasks' / task_name
        self.teacher_latents = np.load(
            root / f'{split}_teacher_latents.npy', mmap_mode='r'
        )
        self.hard_tokens = np.load(
            root / f'{split}_hard_tokens.npy', mmap_mode='r'
        )
        self.topk_indices = np.load(
            root / f'{split}_topk_indices.npy', mmap_mode='r'
        )
        self.topk_probs = np.load(
            root / f'{split}_topk_probs.npy', mmap_mode='r'
        )
        self.task_id = int(task_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.base[int(self.indices[index])]
        return {
            'pixels': sample['pixels'],
            'action': sample['action'],
            'task_id': self.task_id,
            'teacher_latent': torch.from_numpy(
                np.array(self.teacher_latents[index], copy=True)
            ),
            'hard_tokens': torch.from_numpy(
                np.array(self.hard_tokens[index], copy=True).astype(np.int64)
            ),
            'topk_indices': torch.from_numpy(
                np.array(self.topk_indices[index], copy=True).astype(np.int64)
            ),
            'topk_probs': torch.from_numpy(
                np.array(self.topk_probs[index], copy=True)
            ),
        }


class BalancedLoader:
    """Zip per-task loaders so every optimizer step has equal task counts."""

    def __init__(self, loaders: list[DataLoader]):
        if len(loaders) < 2:
            raise ValueError('balanced training requires at least two tasks')
        self.loaders = loaders
        self.length = min(map(len, loaders))

    def __len__(self):
        return self.length

    def __iter__(self):
        for batches in zip(*(iter(loader) for loader in self.loaders)):
            keys = batches[0].keys()
            combined = {}
            for key in keys:
                values = [batch[key] for batch in batches]
                if key == 'action':
                    width = max(value.size(-1) for value in values)
                    values = [
                        F.pad(value, (0, width - value.size(-1)))
                        for value in values
                    ]
                combined[key] = torch.cat(values, dim=0)
            yield combined


class MultiTaskObjective(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.history = int(cfg.wm.history_size)
        self.temperature = float(cfg.codebook.temperature)
        self.chunk_size = int(cfg.codebook.distance_chunk_size)
        # Teacher-representation ablation knobs (defaults reproduce M2 exactly):
        #   latent_target: 'continuous' -> MSE against z^T = E_T(o); 'codebook'
        #     -> MSE against the quantized code c_{y^T}.
        #   prediction_source: 'codebook' -> teacher-forcing mixes in c_{y^T};
        #     'continuous' -> teacher-forcing mixes in z^T.
        # Token loss is skipped entirely when token_weight == 0.
        self.latent_target = str(cfg.loss.get('latent_target', 'continuous'))
        self.prediction_source = str(
            cfg.loss.get('prediction_source', 'codebook')
        )
        self.token_weight = float(cfg.loss.get('token_weight', 0.0))
        if self.latent_target not in ('continuous', 'codebook'):
            raise ValueError(
                f'loss.latent_target must be continuous|codebook, '
                f'got {self.latent_target}'
            )
        if self.prediction_source not in ('continuous', 'codebook'):
            raise ValueError(
                f'loss.prediction_source must be continuous|codebook, '
                f'got {self.prediction_source}'
            )
        self.num_tasks = len(cfg.tasks)
        if self.num_tasks < 1:
            raise ValueError('at least one task is required')

    def forward(self, batch: dict[str, torch.Tensor], alpha: float):
        student = self.model.encode_student(batch['pixels'])
        teacher = batch['teacher_latent'].float()
        # The quantized teacher target is now always materialized: it feeds
        # both the loss path and the supervision-factor diagnostics below.
        teacher_code = self.model.lookup_teacher_codes(batch['hard_tokens'])
        continuous_recon_mse = F.mse_loss(student.float(), teacher)
        codebook_target_mse = F.mse_loss(student.float(), teacher_code.float())
        if self.latent_target == 'codebook':
            latent_loss = codebook_target_mse
        else:
            latent_loss = continuous_recon_mse
        if self.token_weight > 0.0:
            token_loss = sparse_topk_kl(
                student,
                self.model.codebook,
                batch['topk_indices'],
                batch['topk_probs'],
                temperature=self.temperature,
                codebook_chunk_size=self.chunk_size,
            )
        else:
            token_loss = student.new_zeros(())
        pred_teacher = (
            teacher_code if self.prediction_source == 'codebook' else teacher
        )
        mixed, mask = sequence_teacher_forcing(student, pred_teacher, alpha)
        prediction = self.model.predict(
            mixed[:, : self.history],
            batch['action'][:, : self.history],
            batch['task_id'],
        )
        target = mixed[:, 1 : self.history + 1]
        prediction_loss = F.mse_loss(prediction.float(), target.float())
        task_ids = batch['task_id'].reshape(-1).long()
        if task_ids.numel() != student.size(0):
            raise ValueError(
                f'batch carries {task_ids.numel()} task ids for '
                f'{student.size(0)} sequences'
            )
        if int(task_ids.min()) < 0 or int(task_ids.max()) >= self.num_tasks:
            raise ValueError(
                f'batch task ids must lie in [0, {self.num_tasks - 1}], got '
                f'[{int(task_ids.min())}, {int(task_ids.max())}]'
            )
        # Per-task prediction error sums (no grad): column 0 accumulates the
        # squared error summed over every element of each sequence assigned to
        # the task, column 1 the sequence count.
        per_row_sqerr = (
            (prediction.float() - target.float())
            .square()
            .sum(dim=(1, 2))
            .detach()
            .float()
        )
        per_task_pred_sqerr = torch.zeros(
            (self.num_tasks, 2), device=prediction.device, dtype=torch.float32
        )
        per_task_pred_sqerr[:, 0].index_add_(0, task_ids, per_row_sqerr)
        per_task_pred_sqerr[:, 1].index_add_(
            0, task_ids, torch.ones_like(per_row_sqerr)
        )
        return {
            'latent_loss': latent_loss,
            'token_loss': token_loss,
            'prediction_loss': prediction_loss,
            'student_fraction': mask.float().mean(),
            # Supervision diagnostics (all detached, so DDP never gradients
            # through them).
            'continuous_recon_mse': continuous_recon_mse.detach().float(),
            'codebook_target_mse': codebook_target_mse.detach().float(),
            'latent_norm_mean': student.detach().float().norm(dim=-1).mean(),
            'per_task_pred_sqerr': per_task_pred_sqerr,
            'student_detach': student.detach(),
        }


def load_cache(cfg) -> tuple[dict, list[dict[str, np.ndarray]]]:
    root = Path(cfg.paths.multitask_cache_dir).expanduser().resolve()
    metadata = json.loads((root / 'metadata.json').read_text())
    stored = metadata.get('metadata_sha256')
    unhashed = dict(metadata)
    unhashed.pop('metadata_sha256', None)
    if canonical_hash(unhashed) != stored:
        raise RuntimeError('multitask cache metadata self-hash mismatch')
    weights = resolve_weights_path(cfg.paths.fused_codebook_checkpoint)
    if metadata['fused_codebook_sha256'] != sha256_file(weights):
        raise RuntimeError('fused codebook and cache hashes differ')
    splits = []
    for task_id, task in enumerate(cfg.tasks):
        expected_name = metadata['task_id_to_name'][str(task_id)]
        if expected_name != str(task.name):
            raise RuntimeError('task ordering differs from the cache')
        task_root = root / 'tasks' / str(task.name)
        splits.append(
            {
                split: np.load(task_root / f'{split}_indices.npy')
                for split in ('train', 'validation')
            }
        )
    return metadata, splits


def build_loaders(cfg, splits, rank, world_size):
    cache_root = Path(cfg.paths.multitask_cache_dir).expanduser().resolve()
    train_loaders = []
    validation_loaders = []
    samplers = []
    action_dims = []
    workers = int(cfg.data.cpu_workers_total) // world_size // len(cfg.tasks)
    for task_id, (task, task_splits) in enumerate(
        zip(cfg.tasks, splits, strict=True)
    ):
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
        train_set = CachedTaskDataset(
            base,
            task_splits['train'],
            cache_root,
            str(task.name),
            'train',
            task_id,
        )
        validation_set = CachedTaskDataset(
            base,
            task_splits['validation'],
            cache_root,
            str(task.name),
            'validation',
            task_id,
        )
        sampler = DistributedSampler(
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
            DataLoader(train_set, sampler=sampler, drop_last=True, **common)
        )
        validation_loaders.append(
            DataLoader(
                validation_set,
                sampler=validation_sampler,
                drop_last=False,
                **common,
            )
        )
        samplers.append(sampler)
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
def validate_task(model, loader, cfg, device) -> dict[str, float | None]:
    model.eval()
    history = int(cfg.wm.history_size)
    k = model.codebook.size(0)
    latent_dim = model.codebook.size(1)
    sums = torch.zeros(9, device=device, dtype=torch.float64)
    counts = torch.zeros(k, device=device, dtype=torch.float64)
    # Second-moment accumulators for the student latents (float64).
    moment_count = torch.zeros((), device=device, dtype=torch.float64)
    vector_sum = torch.zeros(latent_dim, device=device, dtype=torch.float64)
    outer_sum = torch.zeros(
        (latent_dim, latent_dim), device=device, dtype=torch.float64
    )
    max_batches = (
        int(cfg.smoke.batches_per_epoch) if bool(cfg.smoke.enabled) else None
    )
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            student = model.encode_student(batch['pixels'])
            teacher_code = model.lookup_teacher_codes(batch['hard_tokens'])
            prediction = model.predict(
                student[:, :history],
                batch['action'][:, :history],
                batch['task_id'],
            )
        teacher = batch['teacher_latent'].float()
        nearest = nearest_code_indices(
            student,
            model.codebook,
            k=min(5, k),
            codebook_chunk_size=int(cfg.codebook.distance_chunk_size),
        )
        hard = batch['hard_tokens'].long()
        batch_size = len(student)
        flat_student = student.reshape(-1, latent_dim).double()
        moment_count += flat_student.size(0)
        vector_sum += flat_student.sum(dim=0)
        outer_sum += flat_student.t() @ flat_student
        batch_metrics = torch.tensor(
            [
                float(F.mse_loss(student.float(), teacher)),
                float((nearest[..., 0] == hard).float().mean()),
                float((nearest == hard[..., None]).any(-1).float().mean()),
                float(
                    F.mse_loss(
                        prediction.float(),
                        student[:, 1 : history + 1].float(),
                    )
                ),
                float(
                    F.mse_loss(
                        prediction.float(),
                        teacher_code[:, 1 : history + 1].float(),
                    )
                ),
                float(
                    F.cosine_similarity(student.float(), teacher, dim=-1).mean()
                ),
                float(student.float().norm(dim=-1).mean()),
                batch_size,
                student.numel() // student.size(-1),
            ],
            device=device,
            dtype=torch.float64,
        )
        batch_metrics[:7] *= batch_size
        sums += batch_metrics
        counts += torch.bincount(nearest[..., 0].reshape(-1), minlength=k)
    reduce_sum(sums)
    reduce_sum(counts)
    reduce_sum(moment_count)
    reduce_sum(vector_sum)
    reduce_sum(outer_sum)
    samples = sums[7].clamp_min(1)
    probabilities = counts / counts.sum().clamp_min(1)
    active = probabilities > 0
    perplexity = torch.exp(
        -(probabilities[active] * probabilities[active].log()).sum()
    )
    result = {
        'latent_mse': float(sums[0] / samples),
        'token_agreement': float(sums[1] / samples),
        'top5_token_agreement': float(sums[2] / samples),
        'student_prediction_mse': float(sums[3] / samples),
        'teacher_code_prediction_mse': float(sums[4] / samples),
        'teacher_student_cosine': float(sums[5] / samples),
        'latent_norm_mean': float(sums[6] / samples),
        'active_codes': int(active.sum()),
        'dead_code_fraction': float(1.0 - active.float().mean()),
        'perplexity': float(perplexity),
    }
    if float(moment_count) >= 2.0:
        mean = vector_sum / moment_count
        per_dim_variance = outer_sum / moment_count - mean * mean
        result['latent_variance'] = float(per_dim_variance.mean())
        result['effective_rank'] = float(
            effective_rank_from_moments(moment_count, vector_sum, outer_sum)
        )
    else:
        result['latent_variance'] = None
        result['effective_rank'] = None
    return result


def configure_optimizer(model, cfg):
    encoder_parameters = []
    for module in (model.student_encoder, model.projector, model.adapter):
        encoder_parameters.extend(module.parameters())
    predictor_parameters = []
    for module in (
        model.action_encoder,
        model.task_embedding,
        model.predictor,
        model.pred_proj,
    ):
        predictor_parameters.extend(module.parameters())
    return torch.optim.AdamW(
        [
            {'params': encoder_parameters, 'name': 'encoder'},
            {'params': predictor_parameters, 'name': 'predictor'},
        ],
        lr=1.0,
        weight_decay=float(cfg.optimizer.weight_decay),
        betas=tuple(cfg.optimizer.betas),
    )


def phase_lrs(cfg, phase, step, steps_per_epoch):
    phase_cfg = cfg.phases[f'phase{phase}']
    total = int(cfg.phases.epochs[phase - 1]) * steps_per_epoch
    return (
        cosine_phase_lr(
            step,
            total,
            float(phase_cfg.encoder_lr[0]),
            float(phase_cfg.encoder_lr[1]),
            float(phase_cfg.warmup_fraction),
        ),
        cosine_phase_lr(
            step,
            total,
            float(phase_cfg.predictor_lr[0]),
            float(phase_cfg.predictor_lr[1]),
            float(phase_cfg.warmup_fraction),
        ),
    )


ALPHA_SCHEDULES = ('gradual', 'fixed_teacher', 'fixed_student')


def resolve_alpha_schedule(cfg) -> str:
    """Return the validated trainer.alpha_schedule (default: 'gradual')."""
    schedule = OmegaConf.select(cfg, 'trainer.alpha_schedule')
    schedule = 'gradual' if schedule is None else str(schedule)
    if schedule not in ALPHA_SCHEDULES:
        raise ValueError(
            f'trainer.alpha_schedule must be one of {ALPHA_SCHEDULES}, '
            f'got {schedule!r}'
        )
    return schedule


def scheduled_alpha(cfg, global_step: int, steps_per_epoch: int) -> float:
    """Teacher-forcing mixing coefficient under the configured schedule."""
    schedule = resolve_alpha_schedule(cfg)
    if schedule == 'fixed_teacher':
        return 0.0
    if schedule == 'fixed_student':
        return 1.0
    return teacher_forcing_alpha(
        global_step, steps_per_epoch, cfg.phases.epochs
    )


def freeze_projector_batchnorm(model):
    for module in model.projector.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def atomic_save(payload, path: Path):
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parameter_counts(model) -> dict[str, int]:
    modules = {
        'encoder': model.student_encoder,
        'projector': model.projector,
        'adapter': model.adapter,
        'action_encoder': model.action_encoder,
        'task_embedding': model.task_embedding,
        'predictor': model.predictor,
        'prediction_head': model.pred_proj,
    }
    result = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in modules.items()
    }
    result['trainable_total'] = sum(result.values())
    result['frozen_codebook_storage'] = model.codebook.numel()
    return result


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args.set)
    if args.train_seed is not None:
        apply_train_seed(cfg, args.train_seed)
    rank, world_size, local_rank, device = setup_distributed()
    alpha_schedule_name = resolve_alpha_schedule(cfg)
    max_optimizer_steps = OmegaConf.select(cfg, 'trainer.max_optimizer_steps')
    if max_optimizer_steps is not None:
        max_optimizer_steps = int(max_optimizer_steps)
        if max_optimizer_steps < 1:
            raise ValueError(
                'trainer.max_optimizer_steps must be >= 1, got '
                f'{max_optimizer_steps}'
            )
    if rank == 0 and args.save_config:
        save_resolved_config(cfg, Path(args.save_config))
    run_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    seed = int(cfg.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['LOCAL_DATASET_DIR'] = str(cfg.paths.dataset_cache)
    os.environ['STABLEWM_HOME'] = str(cfg.paths.dataset_cache)
    metadata, splits = load_cache(cfg)
    train_loader, validation_loaders, samplers, action_dim = build_loaders(
        cfg, splits, rank, world_size
    )
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = action_dim
    model = hydra.utils.instantiate(cfg.model).to(device)
    if model.codebook.size(0) != int(metadata['num_embeddings']):
        raise RuntimeError('model inferred the wrong fused codebook size')
    num_tasks = len(cfg.tasks)
    task_names = [str(task.name) for task in cfg.tasks]
    latent_dim = int(cfg.codebook.embedding_dim)
    if model.codebook.size(1) != latent_dim:
        raise RuntimeError(
            f'model embedding dim {model.codebook.size(1)} differs from '
            f'codebook.embedding_dim={latent_dim}'
        )
    # Each prediction row covers history_size frames of embedding_dim channels.
    per_row_elements = int(cfg.wm.history_size) * latent_dim
    if per_row_elements < 1:
        raise ValueError(
            f'wm.history_size * codebook.embedding_dim must be positive, got '
            f'{per_row_elements}'
        )
    objective = MultiTaskObjective(model, cfg).to(device)
    optimizer = configure_optimizer(model, cfg)
    output_dir = Path(cfg.paths.output_dir).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / 'config.yaml')
        config_sha256 = hashlib.sha256(
            (output_dir / 'config.yaml').read_text().encode()
        ).hexdigest()
        manifest = {
            'cache_metadata_sha256': metadata['metadata_sha256'],
            'fused_codebook_sha256': metadata['fused_codebook_sha256'],
            'world_size': world_size,
            'balanced_samples_per_step': (
                int(cfg.data.batch_size_per_task_per_gpu)
                * len(cfg.tasks)
                * world_size
            ),
            'parameter_counts': parameter_counts(model),
            'git_commit': git_commit_hash(
                Path(__file__).resolve().parents[2]
            ),
            'config_sha256': config_sha256,
            'train_seed': int(cfg.seed),
            'alpha_schedule': alpha_schedule_name,
            'loss': {
                'latent_target': objective.latent_target,
                'prediction_source': objective.prediction_source,
                'token_weight': objective.token_weight,
            },
            'max_optimizer_steps': max_optimizer_steps,
            'started_at': started_at,
        }
        (output_dir / 'run_manifest.json').write_text(
            json.dumps(manifest, indent=2) + '\n'
        )
    if world_size > 1:
        dist.barrier()
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
    steps_per_epoch = len(train_loader)
    if bool(cfg.smoke.enabled):
        steps_per_epoch = min(steps_per_epoch, int(cfg.smoke.batches_per_epoch))
    total_epochs = sum(int(value) for value in cfg.phases.epochs)
    global_step = 0
    metrics_path = output_dir / 'metrics.jsonl'
    total_task_samples = torch.zeros(
        num_tasks, device=device, dtype=torch.float64
    )
    reached_max_steps = False
    for epoch in range(total_epochs):
        phase, _ = phase_for_epoch(epoch, cfg.phases.epochs)
        phase_start = sum(cfg.phases.epochs[: phase - 1]) * steps_per_epoch
        for sampler in samplers:
            sampler.set_epoch(epoch)
        objective.train()
        freeze_projector_batchnorm(model)
        epoch_sums = torch.zeros(8, device=device, dtype=torch.float64)
        per_task_sqerr = torch.zeros(
            (num_tasks, 2), device=device, dtype=torch.float64
        )
        # Per-task student-latent moment accumulators: count, vector_sum (D,),
        # outer_sum (D, D), and the row-norm sum for the norm mean.
        task_moment_count = torch.zeros(
            num_tasks, device=device, dtype=torch.float64
        )
        task_vector_sums = torch.zeros(
            (num_tasks, latent_dim), device=device, dtype=torch.float64
        )
        task_outer_sums = torch.zeros(
            (num_tasks, latent_dim, latent_dim), device=device,
            dtype=torch.float64,
        )
        task_norm_sums = torch.zeros(
            num_tasks, device=device, dtype=torch.float64
        )
        sample_count = 0
        epoch_steps = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= steps_per_epoch:
                break
            batch = move_batch(batch, device)
            alpha = scheduled_alpha(cfg, global_step, steps_per_epoch)
            encoder_lr, predictor_lr = phase_lrs(
                cfg, phase, global_step - phase_start, steps_per_epoch
            )
            optimizer.param_groups[0]['lr'] = encoder_lr
            optimizer.param_groups[1]['lr'] = predictor_lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = ddp(batch, alpha)
                loss = (
                    float(cfg.loss.latent_weight) * output['latent_loss']
                    + float(cfg.loss.token_weight) * output['token_loss']
                    + float(cfg.loss.prediction_weight)
                    * output['prediction_loss']
                )
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite multitask loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                objective.parameters(), float(cfg.trainer.gradient_clip_val)
            )
            optimizer.step()
            epoch_sums += torch.tensor(
                [
                    float(loss.detach()),
                    float(output['latent_loss'].detach()),
                    float(output['token_loss'].detach()),
                    float(output['prediction_loss'].detach()),
                    float(output['student_fraction'].detach()),
                    float(output['continuous_recon_mse'].detach()),
                    float(output['codebook_target_mse'].detach()),
                    float(output['latent_norm_mean'].detach()),
                ],
                device=device,
                dtype=torch.float64,
            )
            per_task_sqerr += output['per_task_pred_sqerr'].to(
                device=device, dtype=torch.float64
            )
            # Group the (detached) student rows by task id for the per-task
            # second-moment statistics; rows are (B*T, D) so this is
            # chunk-safe for any batch layout.
            student_rows = output['student_detach']
            frames_per_sequence = student_rows.size(1)
            flat_student = student_rows.reshape(
                -1, student_rows.size(-1)
            ).float()
            row_task_ids = (
                batch['task_id']
                .reshape(-1)
                .long()
                .repeat_interleave(frames_per_sequence)
            )
            row_norms = flat_student.norm(dim=-1).double()
            for task_index in range(num_tasks):
                selected = row_task_ids == task_index
                if bool(selected.any()):
                    rows = flat_student[selected].double()
                    task_moment_count[task_index] += rows.size(0)
                    task_vector_sums[task_index] += rows.sum(dim=0)
                    task_outer_sums[task_index] += rows.t() @ rows
                    task_norm_sums[task_index] += row_norms[selected].sum()
            sample_count += len(batch['pixels'])
            global_step += 1
            epoch_steps += 1
            if (
                max_optimizer_steps is not None
                and global_step >= max_optimizer_steps
            ):
                reached_max_steps = True
                break
        reduce_sum(epoch_sums)
        reduce_sum(per_task_sqerr)
        reduce_sum(task_moment_count)
        reduce_sum(task_vector_sums)
        reduce_sum(task_outer_sums)
        reduce_sum(task_norm_sums)
        total_task_samples += per_task_sqerr[:, 1]
        elapsed = time.perf_counter() - started
        validation = {
            str(task.name): validate_task(
                model, validation_loaders[index], cfg, device
            )
            for index, task in enumerate(cfg.tasks)
        }
        # epoch_steps equals steps_per_epoch unless an early stop (max
        # optimizer steps) truncated the epoch; using it keeps the per-step
        # averages correct in both cases.
        steps_denominator = max(1, epoch_steps * world_size)
        row = {
            'epoch': epoch + 1,
            'phase': phase,
            'global_step': global_step,
            'train/total_loss': float(epoch_sums[0] / steps_denominator),
            'train/latent_mse': float(epoch_sums[1] / steps_denominator),
            'train/token_kl': float(epoch_sums[2] / steps_denominator),
            'train/prediction_mse': float(epoch_sums[3] / steps_denominator),
            'train/student_fraction': float(
                epoch_sums[4] / steps_denominator
            ),
            'train/continuous_recon_mse': float(
                epoch_sums[5] / steps_denominator
            ),
            'train/codebook_target_mse': float(
                epoch_sums[6] / steps_denominator
            ),
            'train/latent_norm_mean': float(
                epoch_sums[7] / steps_denominator
            ),
            'train/samples_per_second_per_rank': sample_count / max(elapsed, 1e-12),
            'validation': validation,
        }
        overall_count = float(task_moment_count.sum())
        row['train/effective_rank'] = (
            float(
                effective_rank_from_moments(
                    task_moment_count.sum(),
                    task_vector_sums.sum(dim=0),
                    task_outer_sums.sum(dim=0),
                )
            )
            if overall_count >= 2.0
            else None
        )
        for task_index, task_name in enumerate(task_names):
            prefix = f'train/task_{task_name}'
            task_count = int(task_moment_count[task_index])
            row[f'{prefix}/count'] = task_count
            row[f'{prefix}/latent_norm_mean'] = (
                float(task_norm_sums[task_index] / task_count)
                if task_count >= 1
                else None
            )
            row[f'{prefix}/effective_rank'] = (
                float(
                    effective_rank_from_moments(
                        task_moment_count[task_index],
                        task_vector_sums[task_index],
                        task_outer_sums[task_index],
                    )
                )
                if task_count >= 2
                else None
            )
            task_sequences = float(per_task_sqerr[task_index, 1])
            row[f'{prefix}/prediction_mse'] = (
                float(
                    per_task_sqerr[task_index, 0]
                    / (task_sequences * per_row_elements)
                )
                if task_sequences >= 1.0
                else None
            )
        if rank == 0:
            with metrics_path.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            print(json.dumps(row, sort_keys=True), flush=True)
            payload = {
                'model': objective.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'cache_metadata_sha256': metadata['metadata_sha256'],
                'fused_codebook_sha256': metadata['fused_codebook_sha256'],
                'validation': validation,
            }
            atomic_save(payload, output_dir / f'phase{phase}_last.ckpt')
        if world_size > 1:
            dist.barrier()
        if reached_max_steps:
            if rank == 0:
                print(
                    f'reached trainer.max_optimizer_steps='
                    f'{max_optimizer_steps}; stopping after epoch '
                    f'{epoch + 1}',
                    flush=True,
                )
            break

    if rank == 0:
        export = {
            'format_version': 1,
            'model_target': (
                'stable_worldmodel.wm.vq_lewm.multitask.'
                'MultiTaskDistilledLeWM'
            ),
            'modules': model.deployment_state_dict(),
            'codebook': model.codebook.cpu(),
            'num_embeddings': model.codebook.size(0),
            'embedding_dim': model.codebook.size(1),
            'num_tasks': len(cfg.tasks),
            'task_id_to_name': metadata['task_id_to_name'],
            'parameter_counts': parameter_counts(model),
        }
        atomic_save(export, output_dir / 'weights_final.pt')
        manifest.update(
            {
                'optimizer_steps': global_step,
                'samples_per_task': {
                    name: int(total_task_samples[index])
                    for index, name in enumerate(task_names)
                },
                'stopped_at_max_optimizer_steps': reached_max_steps,
                'finished_at': datetime.now(timezone.utc).isoformat(),
                'gpu_hours': (
                    (time.perf_counter() - run_started) * world_size / 3600.0
                ),
            }
        )
        manifest_path = output_dir / 'run_manifest.json'
        manifest_temporary = manifest_path.with_name(
            f'.{manifest_path.name}.tmp'
        )
        manifest_temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        os.replace(manifest_temporary, manifest_path)
        print(f'Training complete: {output_dir}', flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
