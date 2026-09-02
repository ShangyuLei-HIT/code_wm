"""Teacher-free balanced training with a shared aligned/fused codebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
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
    return parser.parse_args()


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

    def forward(self, batch: dict[str, torch.Tensor], alpha: float):
        student = self.model.encode_student(batch['pixels'])
        teacher = batch['teacher_latent'].float()
        teacher_code = None
        if self.latent_target == 'codebook' or self.prediction_source == 'codebook':
            teacher_code = self.model.lookup_teacher_codes(batch['hard_tokens'])
        if self.latent_target == 'codebook':
            latent_loss = F.mse_loss(student.float(), teacher_code.float())
        else:
            latent_loss = F.mse_loss(student.float(), teacher)
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
        return {
            'latent_loss': latent_loss,
            'token_loss': token_loss,
            'prediction_loss': prediction_loss,
            'student_fraction': mask.float().mean(),
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
def validate_task(model, loader, cfg, device) -> dict[str, float]:
    model.eval()
    history = int(cfg.wm.history_size)
    k = model.codebook.size(0)
    sums = torch.zeros(7, device=device, dtype=torch.float64)
    counts = torch.zeros(k, device=device, dtype=torch.float64)
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
                batch_size,
                student.numel() // student.size(-1),
            ],
            device=device,
            dtype=torch.float64,
        )
        batch_metrics[:5] *= batch_size
        sums += batch_metrics
        counts += torch.bincount(nearest[..., 0].reshape(-1), minlength=k)
    reduce_sum(sums)
    reduce_sum(counts)
    samples = sums[5].clamp_min(1)
    probabilities = counts / counts.sum().clamp_min(1)
    active = probabilities > 0
    perplexity = torch.exp(
        -(probabilities[active] * probabilities[active].log()).sum()
    )
    return {
        'latent_mse': float(sums[0] / samples),
        'token_agreement': float(sums[1] / samples),
        'top5_token_agreement': float(sums[2] / samples),
        'student_prediction_mse': float(sums[3] / samples),
        'teacher_code_prediction_mse': float(sums[4] / samples),
        'active_codes': int(active.sum()),
        'dead_code_fraction': float(1.0 - active.float().mean()),
        'perplexity': float(perplexity),
    }


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
    cfg = OmegaConf.load(parse_args().config)
    rank, world_size, local_rank, device = setup_distributed()
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
    objective = MultiTaskObjective(model, cfg).to(device)
    optimizer = configure_optimizer(model, cfg)
    output_dir = Path(cfg.paths.output_dir).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / 'config.yaml')
        (output_dir / 'run_manifest.json').write_text(
            json.dumps(
                {
                    'cache_metadata_sha256': metadata['metadata_sha256'],
                    'fused_codebook_sha256': metadata[
                        'fused_codebook_sha256'
                    ],
                    'world_size': world_size,
                    'balanced_samples_per_step': (
                        int(cfg.data.batch_size_per_task_per_gpu)
                        * len(cfg.tasks)
                        * world_size
                    ),
                    'parameter_counts': parameter_counts(model),
                },
                indent=2,
            )
            + '\n'
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
    for epoch in range(total_epochs):
        phase, _ = phase_for_epoch(epoch, cfg.phases.epochs)
        phase_start = sum(cfg.phases.epochs[: phase - 1]) * steps_per_epoch
        for sampler in samplers:
            sampler.set_epoch(epoch)
        objective.train()
        freeze_projector_batchnorm(model)
        epoch_sums = torch.zeros(5, device=device, dtype=torch.float64)
        sample_count = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= steps_per_epoch:
                break
            batch = move_batch(batch, device)
            alpha = teacher_forcing_alpha(
                global_step, steps_per_epoch, cfg.phases.epochs
            )
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
                ],
                device=device,
                dtype=torch.float64,
            )
            sample_count += len(batch['pixels'])
            global_step += 1
        reduce_sum(epoch_sums)
        elapsed = time.perf_counter() - started
        validation = {
            str(task.name): validate_task(
                model, validation_loaders[index], cfg, device
            )
            for index, task in enumerate(cfg.tasks)
        }
        row = {
            'epoch': epoch + 1,
            'phase': phase,
            'global_step': global_step,
            'train/total_loss': float(
                epoch_sums[0] / max(1, steps_per_epoch * world_size)
            ),
            'train/latent_mse': float(
                epoch_sums[1] / max(1, steps_per_epoch * world_size)
            ),
            'train/token_kl': float(
                epoch_sums[2] / max(1, steps_per_epoch * world_size)
            ),
            'train/prediction_mse': float(
                epoch_sums[3] / max(1, steps_per_epoch * world_size)
            ),
            'train/student_fraction': float(
                epoch_sums[4] / max(1, steps_per_epoch * world_size)
            ),
            'train/samples_per_second_per_rank': sample_count / max(elapsed, 1e-12),
            'validation': validation,
        }
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
        print(f'Training complete: {output_dir}', flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
