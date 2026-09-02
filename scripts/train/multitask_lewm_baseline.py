"""Native continuous, balanced PushT/Two-Room LeWM baseline (M3)."""

from __future__ import annotations

import argparse
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
from stable_worldmodel.wm.loss import SIGReg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/multitask_lewm_baseline.yaml',
    )
    return parser.parse_args()


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

    def forward(self, batch):
        embedding = self.model.encode({'pixels': batch['pixels']})['emb']
        prediction = self.model.predict_actions(
            embedding[:, : self.history],
            batch['action'][:, : self.history],
            batch['task_id'],
        )
        target = embedding[:, 1 : self.history + 1]
        prediction_loss = F.mse_loss(prediction, target)
        regularization = self.sigreg(embedding.transpose(0, 1))
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
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / 'config.yaml')
        (output_dir / 'run_manifest.json').write_text(
            json.dumps(
                {
                    'baseline': 'M3_native_continuous',
                    'uses_teacher': False,
                    'uses_alignment': False,
                    'uses_codebook': False,
                    'parameter_counts': parameter_counts(model),
                    'world_size': world_size,
                },
                indent=2,
            )
            + '\n'
        )
    if world_size > 1:
        dist.barrier()
    steps_per_epoch = len(train_loader)
    if bool(cfg.smoke.enabled):
        steps_per_epoch = min(steps_per_epoch, int(cfg.smoke.batches_per_epoch))
    total_steps = int(cfg.trainer.epochs) * steps_per_epoch
    global_step = 0
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
        reduce_sum(totals)
        validation = {
            str(task.name): validate(
                model, validation_loaders[index], cfg, device
            )
            for index, task in enumerate(cfg.tasks)
        }
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
        print(f'M3 training complete: {output_dir}', flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
