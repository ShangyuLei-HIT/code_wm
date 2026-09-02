"""Three-phase DDP training in cached continuous/codebook latent space."""

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
from stable_worldmodel.wm.vq_lewm.affine import (
    RigidLatentTransform,
    initialize_adapter_from_transform,
    load_rigid_latent_transform,
)
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
        default='scripts/train/config/vq_lewm_joint_distillation.yaml',
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


def workers_for_rank(total: int, rank: int, world_size: int) -> int:
    return total // world_size + int(rank < total % world_size)


def canonical_hash(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def image_preprocessor(img_size: int):
    to_image = dt.transforms.ToImage(
        **dt.dataset_stats.ImageNet,
        source='pixels',
        target='pixels',
    )
    resize = dt.transforms.Resize(
        img_size, source='pixels', target='pixels'
    )
    return dt.transforms.Compose(to_image, resize)


def load_cache_metadata(cfg) -> tuple[dict, np.ndarray, np.ndarray]:
    cache_dir = Path(cfg.paths.cache_dir).expanduser().resolve()
    metadata_path = cache_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(
            f'offline teacher cache is missing: {metadata_path}'
        )
    metadata = json.loads(metadata_path.read_text())
    stored_hash = metadata.get('metadata_sha256')
    unhashed = dict(metadata)
    unhashed.pop('metadata_sha256', None)
    if canonical_hash(unhashed) != stored_hash:
        raise RuntimeError('cache metadata self-hash mismatch')

    expected = {
        'dataset': cfg.data.dataset,
        'dataset_cache': str(
            Path(cfg.paths.dataset_cache).expanduser().resolve()
        ),
        'split_seed': int(cfg.data.split_seed),
        'train_split': float(cfg.data.train_split),
        'resize': int(cfg.data.img_size),
        'frameskip': int(cfg.data.frameskip),
        'num_steps': int(cfg.data.num_steps),
        'codebook_size': int(cfg.codebook.num_embeddings),
        'latent_dimension': int(cfg.codebook.embedding_dim),
        'topk': int(cfg.codebook.topk),
        'temperature': float(cfg.codebook.temperature),
        'teacher_checkpoint': str(
            Path(cfg.paths.teacher_checkpoint).expanduser().resolve()
        ),
        'codebook_checkpoint': str(
            Path(cfg.paths.codebook_checkpoint).expanduser().resolve()
        ),
    }
    transform_cfg = cfg.get('latent_transform')
    if transform_cfg and bool(transform_cfg.get('enabled', False)):
        transform_path = Path(transform_cfg.checkpoint).expanduser().resolve()
        expected['latent_transform'] = {
            'checkpoint': str(transform_path),
            'sha256': sha256_file(transform_path),
            'initialize_adapter': bool(
                transform_cfg.get('initialize_adapter', True)
            ),
            'reuse_base_assignments': bool(
                transform_cfg.get('reuse_base_assignments', False)
            ),
        }
    elif metadata.get('latent_transform') is not None:
        expected['latent_transform'] = None
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    teacher_hash = sha256_file(
        resolve_weights_path(cfg.paths.teacher_checkpoint)
    )
    codebook_hash = sha256_file(
        resolve_weights_path(cfg.paths.codebook_checkpoint)
    )
    if metadata.get('teacher_sha256') != teacher_hash:
        mismatches['teacher_sha256'] = (
            metadata.get('teacher_sha256'),
            teacher_hash,
        )
    if metadata.get('codebook_sha256') != codebook_hash:
        mismatches['codebook_sha256'] = (
            metadata.get('codebook_sha256'),
            codebook_hash,
        )
    if mismatches:
        raise RuntimeError(f'cache/config metadata mismatch: {mismatches}')

    indices = []
    for split in ('train', 'val'):
        spec = metadata['sample_indices'][split]
        array = np.load(cache_dir / spec['file'])
        if len(array) != spec['count'] or array_hash(array) != spec['sha256']:
            raise RuntimeError(f'{split} cache sample indices are invalid')
        if bool(cfg.smoke.enabled):
            limit = int(
                cfg.smoke.train_samples
                if split == 'train'
                else cfg.smoke.val_samples
            )
            array = array[:limit]
        indices.append(array)
    for relative, spec in metadata['files'].items():
        array = np.load(cache_dir / relative, mmap_mode='r')
        if list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']:
            raise RuntimeError(f'cache array mismatch: {relative}')
    return metadata, indices[0], indices[1]


class CachedSplit(Dataset):
    def __init__(self, base, indices: np.ndarray, cache_dir: Path, split: str, k: int):
        self.base = base
        self.indices = indices
        self.teacher_latent = np.load(
            cache_dir / f'{split}_teacher_latents.npy', mmap_mode='r'
        )
        self.hard_tokens = np.load(
            cache_dir / f'k{k}/{split}_hard_tokens.npy', mmap_mode='r'
        )
        self.topk_indices = np.load(
            cache_dir / f'k{k}/{split}_topk_indices.npy', mmap_mode='r'
        )
        self.topk_probs = np.load(
            cache_dir / f'k{k}/{split}_topk_probs.npy', mmap_mode='r'
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.base[int(self.indices[index])]
        return {
            'pixels': sample['pixels'],
            'action': sample['action'],
            'teacher_latent': torch.from_numpy(
                np.array(self.teacher_latent[index], copy=True)
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


class JointObjective(nn.Module):
    """Make the complete loss one DDP forward without owning a teacher."""

    def __init__(
        self,
        model,
        cfg,
        latent_transform: RigidLatentTransform | None = None,
    ):
        super().__init__()
        self.model = model
        self.history_size = int(cfg.wm.history_size)
        self.temperature = float(cfg.codebook.temperature)
        self.chunk_size = int(cfg.codebook.distance_chunk_size)
        self.latent_transform = latent_transform

    def forward(self, batch: dict[str, torch.Tensor], alpha: float):
        student = self.model.encode_student(batch['pixels'])
        teacher_continuous = batch['teacher_latent'].float()
        if self.latent_transform is not None:
            teacher_continuous = self.latent_transform(teacher_continuous)
        latent_loss = F.mse_loss(student.float(), teacher_continuous)
        soft_kl = sparse_topk_kl(
            student,
            self.model.codebook,
            batch['topk_indices'],
            batch['topk_probs'],
            temperature=self.temperature,
            codebook_chunk_size=self.chunk_size,
        )
        teacher_code = self.model.lookup_teacher_codes(batch['hard_tokens'])
        mixed, mask = sequence_teacher_forcing(student, teacher_code, alpha)
        prediction = self.model.predict(
            mixed[:, : self.history_size],
            batch['action'][:, : self.history_size],
        )
        target = mixed[:, 1 : self.history_size + 1]
        prediction_loss = F.mse_loss(prediction.float(), target.float())
        return {
            'latent_loss': latent_loss,
            'soft_kl': soft_kl,
            'prediction_loss': prediction_loss,
            'student_fraction': mask.float().mean(),
        }


def make_loaders(cfg, base, train_indices, val_indices, rank, world_size):
    cache_dir = Path(cfg.paths.cache_dir).expanduser().resolve()
    k = int(cfg.codebook.num_embeddings)
    train_set = CachedSplit(base, train_indices, cache_dir, 'train', k)
    val_set = CachedSplit(base, val_indices, cache_dir, 'val', k)
    workers = workers_for_rank(
        int(cfg.data.cpu_workers_total), rank, world_size
    )
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(cfg.seed),
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    common = {
        'batch_size': int(cfg.data.train_batch_size_per_gpu),
        'num_workers': workers,
        'pin_memory': bool(cfg.data.pin_memory),
        'persistent_workers': bool(cfg.data.persistent_workers) and workers > 0,
    }
    if workers:
        common['prefetch_factor'] = int(cfg.data.prefetch_factor)
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_sampler


def initialize_student_from_state(model, checkpoint: str | None) -> None:
    if not checkpoint:
        return
    state = torch.load(
        resolve_weights_path(checkpoint), map_location='cpu', weights_only=True
    )
    for source, target in (
        ('encoder.', model.student_encoder),
        ('projector.', model.projector),
    ):
        component = {
            key[len(source) :]: value
            for key, value in state.items()
            if key.startswith(source)
        }
        if not component:
            raise RuntimeError(f'no {source} weights in {checkpoint}')
        target.load_state_dict(component, strict=True)


def move_batch(batch, device):
    result = {}
    for key, value in batch.items():
        value = value.to(device, non_blocking=True)
        result[key] = torch.nan_to_num(value, 0.0) if key == 'action' else value
    return result


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


def distributed_all_finite(tensor: torch.Tensor) -> bool:
    flag = torch.tensor(
        int(torch.isfinite(tensor).all()), device=tensor.device
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def configure_optimizer(model, cfg):
    encoder = []
    for module in (model.student_encoder, model.projector, model.adapter):
        encoder.extend(module.parameters())
    predictor = []
    for module in (model.predictor, model.action_encoder, model.pred_proj):
        predictor.extend(module.parameters())
    return torch.optim.AdamW(
        [
            {'params': encoder, 'name': 'encoder'},
            {'params': predictor, 'name': 'predictor'},
        ],
        lr=1.0,
        weight_decay=float(cfg.optimizer.weight_decay),
        betas=tuple(float(x) for x in cfg.optimizer.betas),
    )


def freeze_projector_batchnorm_statistics(model):
    """Match the eval-mode normalization used by the offline teacher cache."""
    for module in model.projector.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def phase_learning_rates(cfg, phase, step_in_phase, steps_per_epoch):
    phase_cfg = cfg.phases[f'phase{phase}']
    total = int(cfg.phases.epochs[phase - 1]) * steps_per_epoch
    encoder_lr = cosine_phase_lr(
        step_in_phase,
        total,
        float(phase_cfg.encoder_lr[0]),
        float(phase_cfg.encoder_lr[1]),
        float(phase_cfg.warmup_fraction),
    )
    predictor_lr = cosine_phase_lr(
        step_in_phase,
        total,
        float(phase_cfg.predictor_lr[0]),
        float(phase_cfg.predictor_lr[1]),
        float(phase_cfg.warmup_fraction),
    )
    return encoder_lr, predictor_lr


def atomic_torch_save(value, path: Path):
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def local_rng_state() -> dict:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state(),
    }


def gather_rng_states(rank: int, world_size: int) -> list[dict] | None:
    state = local_rng_state()
    if world_size > 1:
        gathered = (
            [None for _ in range(world_size)] if rank == 0 else None
        )
        dist.gather_object(state, gathered, dst=0)
        return gathered
    return [state]


def checkpoint_payload(
    wrapper,
    optimizer,
    epoch,
    global_step,
    phase,
    alpha,
    cache_hash,
    codebook_hash,
    best_values,
    metrics,
    rng_states,
):
    if rng_states is None:
        raise ValueError('rank0 checkpoint requires gathered RNG states')
    return {
        'model': wrapper.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'phase': phase,
        'teacher_forcing_alpha': alpha,
        'learning_rates': [group['lr'] for group in optimizer.param_groups],
        'cache_metadata_hash': cache_hash,
        'codebook_sha256': codebook_hash,
        'best_values': best_values,
        'validation_metrics': metrics,
        'checkpoint_format_version': 2,
        'rng_state_by_rank': rng_states,
    }


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = list(output_dir.glob('phase*_last.ckpt'))
    if not candidates:
        return None
    scored = []
    for path in candidates:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        scored.append((int(payload['epoch']), path))
    return max(scored)[1]


def restore_checkpoint(
    wrapper,
    optimizer,
    path,
    cache_hash,
    codebook_hash,
    rank,
    world_size,
):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['cache_metadata_hash'] != cache_hash:
        raise RuntimeError('resume checkpoint cache hash mismatch')
    if payload['codebook_sha256'] != codebook_hash:
        raise RuntimeError('resume checkpoint codebook hash mismatch')
    wrapper.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    states = payload.get('rng_state_by_rank')
    if states is not None:
        if len(states) != world_size:
            raise RuntimeError(
                'resume checkpoint world size differs from saved RNG states'
            )
        state = states[rank]
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state(state['cuda'])
    else:
        # Backward compatibility for completed v1 runs. New checkpoints always
        # use independent per-rank states above.
        random.setstate(payload['python_rng_state'])
        np.random.set_state(payload['numpy_rng_state'])
        torch.set_rng_state(payload['torch_rng_state'])
        torch.cuda.set_rng_state_all(payload['cuda_rng_state'])
    return (
        int(payload['epoch']) + 1,
        int(payload['global_step']),
        dict(payload.get('best_values', {})),
    )


def append_jsonl(path: Path, row: dict):
    with path.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')


def require_finite(tensor, rank, output_dir, epoch, global_step, label):
    if distributed_all_finite(tensor):
        return
    payload = {
        'error': f'non-finite {label}',
        'epoch': epoch + 1,
        'global_step': global_step,
    }
    if rank == 0:
        path = output_dir / 'HARD_ERROR_NONFINITE.json'
        path.write_text(json.dumps(payload, indent=2) + '\n')
    raise FloatingPointError(payload['error'])


@torch.no_grad()
def validate(model, loader, cfg, alpha, device, latent_transform=None):
    model.eval()
    dim = int(cfg.codebook.embedding_dim)
    k = int(cfg.codebook.num_embeddings)
    sums = torch.zeros(11, device=device, dtype=torch.float64)
    student_counts = torch.zeros(k, device=device, dtype=torch.float64)
    teacher_counts = torch.zeros(k, device=device, dtype=torch.float64)
    student_sum = torch.zeros(dim, device=device, dtype=torch.float64)
    teacher_sum = torch.zeros(dim, device=device, dtype=torch.float64)
    student_outer = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    teacher_outer = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    vector_count = torch.zeros((), device=device, dtype=torch.float64)
    history = int(cfg.wm.history_size)
    max_batches = int(cfg.smoke.batches_per_epoch) if cfg.smoke.enabled else None

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            student = model.encode_student(batch['pixels'])
            teacher_code = model.lookup_teacher_codes(batch['hard_tokens'])
            pred_teacher = model.predict(
                teacher_code[:, :history], batch['action'][:, :history]
            )
            pred_student = model.predict(
                student[:, :history], batch['action'][:, :history]
            )
            mixed, _ = sequence_teacher_forcing(student, teacher_code, alpha)
            pred_mixed = model.predict(
                mixed[:, :history], batch['action'][:, :history]
            )
        teacher_continuous = batch['teacher_latent'].float()
        if latent_transform is not None:
            teacher_continuous = latent_transform(teacher_continuous)
        latent_mse = F.mse_loss(student.float(), teacher_continuous)
        soft_kl = sparse_topk_kl(
            student,
            model.codebook,
            batch['topk_indices'],
            batch['topk_probs'],
            temperature=float(cfg.codebook.temperature),
            codebook_chunk_size=int(cfg.codebook.distance_chunk_size),
        )
        nearest = nearest_code_indices(
            student,
            model.codebook,
            k=5,
            codebook_chunk_size=int(cfg.codebook.distance_chunk_size),
        )
        student_hard = nearest[..., 0]
        teacher_hard = batch['hard_tokens'].long()
        agreement = (student_hard == teacher_hard).float().mean()
        top5 = (nearest == teacher_hard[..., None]).any(dim=-1).float().mean()
        target_teacher = teacher_code[:, 1 : history + 1]
        target_student = student[:, 1 : history + 1]
        target_mixed = mixed[:, 1 : history + 1]
        batch_size = student.size(0)
        sums += torch.tensor(
            [
                float(latent_mse),
                float(soft_kl),
                float(agreement),
                float(top5),
                float(F.mse_loss(pred_teacher.float(), target_teacher.float())),
                float(F.mse_loss(pred_student.float(), target_student.float())),
                float(F.mse_loss(pred_mixed.float(), target_mixed.float())),
                batch_size,
                batch_size,
                batch_size,
                batch_size,
            ],
            device=device,
            dtype=torch.float64,
        ) * torch.tensor(
            [batch_size] * 7 + [1, 1, 1, 1],
            device=device,
            dtype=torch.float64,
        )
        student_counts += torch.bincount(
            student_hard.reshape(-1), minlength=k
        ).double()
        teacher_counts += torch.bincount(
            teacher_hard.reshape(-1), minlength=k
        ).double()
        sflat = student.reshape(-1, dim).double()
        tflat = teacher_continuous.reshape(-1, dim).double()
        student_sum += sflat.sum(0)
        teacher_sum += tflat.sum(0)
        student_outer += sflat.t() @ sflat
        teacher_outer += tflat.t() @ tflat
        vector_count += len(sflat)

    for value in (
        sums,
        student_counts,
        teacher_counts,
        student_sum,
        teacher_sum,
        student_outer,
        teacher_outer,
        vector_count,
    ):
        all_reduce(value)
    sample_count = sums[7].clamp_min(1.0)
    averages = sums[:7] / sample_count

    def perplexity(counts):
        probs = counts / counts.sum().clamp_min(1.0)
        active = probs > 0
        return torch.exp(-(probs[active] * probs[active].log()).sum())

    student_perplexity = perplexity(student_counts)
    teacher_perplexity = perplexity(teacher_counts)
    student_rank = effective_rank_from_moments(
        vector_count, student_sum, student_outer
    )
    teacher_rank = effective_rank_from_moments(
        vector_count, teacher_sum, teacher_outer
    )
    teacher_pred = float(averages[4])
    student_pred = float(averages[5])
    return {
        'validate/latent_mse': float(averages[0]),
        'validate/soft_kl': float(averages[1]),
        'validate/token_agreement': float(averages[2]),
        'validate/top5_token_agreement': float(averages[3]),
        'validate/student_active_codes': int((student_counts > 0).sum()),
        'validate/student_active_code_fraction': float(
            (student_counts > 0).sum() / max(1, k)
        ),
        'validate/teacher_active_codes': int((teacher_counts > 0).sum()),
        'validate/teacher_active_code_fraction': float(
            (teacher_counts > 0).sum() / max(1, k)
        ),
        'validate/student_perplexity': float(student_perplexity),
        'validate/student_perplexity_fraction': float(
            student_perplexity / max(1, k)
        ),
        'validate/teacher_perplexity': float(teacher_perplexity),
        'validate/teacher_perplexity_fraction': float(
            teacher_perplexity / max(1, k)
        ),
        'validate/perplexity_ratio': float(
            student_perplexity / teacher_perplexity.clamp_min(1e-12)
        ),
        'validate/pred_teacher_mse': teacher_pred,
        'validate/pred_student_mse': student_pred,
        'validate/pred_mixed_mse': float(averages[6]),
        'validate/teacher_student_prediction_gap': student_pred - teacher_pred,
        'validate/prediction_gap_ratio': (
            (student_pred - teacher_pred) / max(teacher_pred, 1e-12)
        ),
        'validate/student_effective_rank': float(student_rank),
        'validate/teacher_effective_rank': float(teacher_rank),
        'validate/effective_rank_ratio': float(
            student_rank / teacher_rank.clamp_min(1e-12)
        ),
    }


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    rank, world_size, local_rank, device = setup_distributed()
    seed = int(cfg.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = bool(cfg.trainer.benchmark)

    os.environ['LOCAL_DATASET_DIR'] = str(cfg.paths.dataset_cache)
    os.environ['STABLEWM_HOME'] = str(cfg.paths.dataset_cache)
    metadata, train_indices, val_indices = load_cache_metadata(cfg)
    base = swm.data.load_dataset(
        cfg.data.dataset,
        transform=None,
        num_steps=int(cfg.data.num_steps),
        frameskip=int(cfg.data.frameskip),
        keys_to_load=['pixels', 'action'],
    )
    action_dim = int(cfg.data.frameskip) * int(base.get_dim('action'))
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = action_dim
    base.transform = spt.data.transforms.Compose(
        image_preprocessor(int(cfg.data.img_size)),
        column_normalizer(base, 'action', 'action'),
    )
    train_loader, val_loader, train_sampler = make_loaders(
        cfg, base, train_indices, val_indices, rank, world_size
    )
    steps_per_epoch = len(train_loader)
    if cfg.smoke.enabled:
        steps_per_epoch = min(steps_per_epoch, int(cfg.smoke.batches_per_epoch))

    latent_transform = None
    transform_cfg = cfg.get('latent_transform')
    if transform_cfg and bool(transform_cfg.get('enabled', False)):
        latent_transform = load_rigid_latent_transform(
            transform_cfg.checkpoint,
            expected_dim=int(cfg.codebook.embedding_dim),
        )
    model = hydra.utils.instantiate(cfg.model)
    initialize_student_from_state(model, cfg.paths.student_init_checkpoint)
    if (
        latent_transform is not None
        and bool(transform_cfg.get('initialize_adapter', True))
    ):
        initialize_adapter_from_transform(model.adapter, latent_transform)
    model.to(device)
    if latent_transform is not None:
        latent_transform.to(device)
    wrapper = JointObjective(model, cfg, latent_transform).to(device)
    optimizer = configure_optimizer(model, cfg)
    codebook_hash = sha256_file(
        resolve_weights_path(cfg.paths.codebook_checkpoint)
    )
    output_dir = Path(cfg.paths.output_dir).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / 'config.yaml')
        (output_dir / 'run_manifest.json').write_text(
            json.dumps(
                {
                    'data_cache': str(Path(cfg.paths.cache_dir).resolve()),
                    'cache_metadata_hash': metadata['metadata_sha256'],
                    'codebook_checkpoint': str(
                        Path(cfg.paths.codebook_checkpoint).resolve()
                    ),
                    'codebook_sha256': codebook_hash,
                    'student_init_checkpoint': str(
                        Path(cfg.paths.student_init_checkpoint).resolve()
                    ),
                    'gpus': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'world_size': world_size,
                    'cpu_workers_total': int(cfg.data.cpu_workers_total),
                    'batch_size_per_gpu': int(cfg.data.train_batch_size_per_gpu),
                    'global_batch_size': (
                        world_size * int(cfg.data.train_batch_size_per_gpu)
                    ),
                    'latent_transform': (
                        {
                            'checkpoint': str(
                                Path(transform_cfg.checkpoint)
                                .expanduser()
                                .resolve()
                            ),
                            'sha256': sha256_file(
                                Path(transform_cfg.checkpoint)
                                .expanduser()
                                .resolve()
                            ),
                            'initialize_adapter': bool(
                                transform_cfg.get('initialize_adapter', True)
                            ),
                        }
                        if latent_transform is not None
                        else None
                    ),
                },
                indent=2,
            )
            + '\n'
        )
    if world_size > 1:
        dist.barrier()

    start_epoch = 0
    global_step = 0
    best_values = {'1': math.inf, '2': math.inf, '3': math.inf}
    resume_path = latest_checkpoint(output_dir) if cfg.trainer.resume == 'auto' else None
    if resume_path is not None:
        start_epoch, global_step, best_values = restore_checkpoint(
            wrapper,
            optimizer,
            resume_path,
            metadata['metadata_sha256'],
            codebook_hash,
            rank,
            world_size,
        )
        if rank == 0:
            print(
                f'Resumed {resume_path} at epoch={start_epoch}, '
                f'step={global_step}, alpha='
                f'{teacher_forcing_alpha(global_step, steps_per_epoch, cfg.phases.epochs):.6f}',
                flush=True,
            )

    ddp = DDP(
        wrapper,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    ) if world_size > 1 else wrapper
    total_epochs = sum(int(value) for value in cfg.phases.epochs)
    metrics_path = output_dir / 'metrics.jsonl'
    previous_latent_mse = math.inf

    for epoch in range(start_epoch, total_epochs):
        phase, _ = phase_for_epoch(epoch, cfg.phases.epochs)
        phase_start_step = sum(
            int(value) for value in cfg.phases.epochs[: phase - 1]
        ) * steps_per_epoch
        train_sampler.set_epoch(epoch)
        wrapper.train()
        freeze_projector_batchnorm_statistics(model)
        torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        local_samples = 0
        epoch_sums = torch.zeros(5, device=device, dtype=torch.float64)
        max_batches = int(cfg.smoke.batches_per_epoch) if cfg.smoke.enabled else None
        for batch_index, batch in enumerate(train_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            alpha = teacher_forcing_alpha(
                global_step, steps_per_epoch, cfg.phases.epochs
            )
            encoder_lr, predictor_lr = phase_learning_rates(
                cfg,
                phase,
                global_step - phase_start_step,
                steps_per_epoch,
            )
            optimizer.param_groups[0]['lr'] = encoder_lr
            optimizer.param_groups[1]['lr'] = predictor_lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = ddp(batch, alpha)
                total_loss = (
                    float(cfg.loss.prediction_weight) * output['prediction_loss']
                    + float(cfg.loss.latent_weight) * output['latent_loss']
                    + float(cfg.loss.soft_kl_weight) * output['soft_kl']
                )
            require_finite(
                total_loss.detach(),
                rank, output_dir, epoch, global_step, 'loss',
            )
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                wrapper.parameters(), float(cfg.trainer.gradient_clip_val)
            )
            require_finite(
                grad_norm.detach(),
                rank, output_dir, epoch, global_step, 'gradient norm',
            )
            optimizer.step()
            epoch_sums += torch.tensor(
                [
                    float(total_loss.detach()),
                    float(output['latent_loss'].detach()),
                    float(output['soft_kl'].detach()),
                    float(output['prediction_loss'].detach()),
                    float(output['student_fraction'].detach()),
                ],
                device=device,
                dtype=torch.float64,
            )
            local_samples += int(batch['pixels'].size(0))
            global_step += 1
            if rank == 0 and global_step % int(cfg.trainer.log_every_n_steps) == 0:
                print(
                    f'epoch={epoch + 1}/{total_epochs} phase={phase} '
                    f'step={global_step} alpha={alpha:.4f} '
                    f'loss={float(total_loss.detach()):.6f} '
                    f'lr=({encoder_lr:.2e},{predictor_lr:.2e})',
                    flush=True,
                )
        torch.cuda.synchronize(device)
        epoch_seconds = torch.tensor(
            time.perf_counter() - epoch_started,
            device=device, dtype=torch.float64,
        )
        global_samples = torch.tensor(
            local_samples, device=device, dtype=torch.float64
        )
        peak_memory = torch.tensor(
            torch.cuda.max_memory_allocated(device),
            device=device, dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(epoch_seconds, op=dist.ReduceOp.MAX)
            dist.all_reduce(global_samples, op=dist.ReduceOp.SUM)
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        all_reduce(epoch_sums)
        denominator = max(1, steps_per_epoch * world_size)
        averages = epoch_sums / denominator
        current_alpha = teacher_forcing_alpha(
            global_step, steps_per_epoch, cfg.phases.epochs
        )
        validation = validate(
            model,
            val_loader,
            cfg,
            current_alpha,
            device,
            latent_transform,
        )
        validation_values = torch.tensor(
            [float(value) for value in validation.values()], device=device
        )
        require_finite(
            validation_values,
            rank, output_dir, epoch, global_step, 'validation metric',
        )
        train_metrics = {
            'train/total_loss': float(averages[0]),
            'train/latent_mse': float(averages[1]),
            'train/soft_kl': float(averages[2]),
            'train/prediction_mse': float(averages[3]),
            'train/actual_student_fraction': float(averages[4]),
            'train/teacher_forcing_alpha': current_alpha,
            'train/encoder_lr': optimizer.param_groups[0]['lr'],
            'train/predictor_lr': optimizer.param_groups[1]['lr'],
            'train/epoch_seconds': float(epoch_seconds),
            'train/global_samples': int(global_samples.item()),
            'train/samples_per_second': float(
                global_samples / epoch_seconds.clamp_min(1e-12)
            ),
            'train/peak_memory_bytes': int(peak_memory.item()),
        }
        row = {
            'epoch': epoch + 1,
            'phase': phase,
            'global_step': global_step,
            **train_metrics,
            **validation,
        }
        rng_states = gather_rng_states(rank, world_size)
        if rank == 0:
            append_jsonl(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
            monitor = validation['validate/pred_mixed_mse']
            payload = checkpoint_payload(
                wrapper,
                optimizer,
                epoch,
                global_step,
                phase,
                current_alpha,
                metadata['metadata_sha256'],
                codebook_hash,
                best_values,
                validation,
                rng_states,
            )
            atomic_torch_save(payload, output_dir / f'phase{phase}_last.ckpt')
            if monitor < float(best_values[str(phase)]):
                best_values[str(phase)] = monitor
                payload['best_values'] = best_values
                atomic_torch_save(payload, output_dir / f'phase{phase}_best.ckpt')
        if world_size > 1:
            dist.barrier()

        phase1_finished = epoch + 1 == int(cfg.phases.epochs[0])
        gate_enforced = bool(cfg.gates.get('enforce', True))
        if phase1_finished:
            gate_ok = (
                validation['validate/token_agreement']
                >= float(cfg.gates.phase1_token_agreement)
                and validation['validate/perplexity_ratio']
                >= float(cfg.gates.phase1_perplexity_ratio)
                and validation['validate/student_active_codes'] > 1
                and validation['validate/latent_mse'] <= previous_latent_mse
            )
            gate_tensor = torch.tensor(int(gate_ok), device=device)
            if world_size > 1:
                dist.broadcast(gate_tensor, src=0)
            if not bool(gate_tensor.item()):
                report_name = (
                    'STOPPED_PHASE1_GATE.json'
                    if gate_enforced
                    else 'PHASE1_GATE_WARNING.json'
                )
                if rank == 0:
                    (output_dir / report_name).write_text(
                        json.dumps(validation, indent=2) + '\n'
                    )
                    message = (
                        'Phase-1 gate failed; transition cancelled.'
                        if gate_enforced
                        else 'Phase-1 gate failed; experiment continues '
                        'because gates.enforce=false.'
                    )
                    print(message, flush=True)
                if gate_enforced:
                    if world_size > 1:
                        dist.destroy_process_group()
                    raise SystemExit(2)
        previous_latent_mse = validation['validate/latent_mse']

    if rank == 0:
        final_ok = (
            validation['validate/token_agreement']
            >= float(cfg.gates.final_token_agreement)
            and validation['validate/perplexity_ratio']
            >= float(cfg.gates.final_perplexity_ratio)
            and validation['validate/prediction_gap_ratio']
            <= float(cfg.gates.final_prediction_gap_ratio)
        )
        final_report = {'requirements_met': final_ok, **validation}
        (output_dir / 'final_evaluation.json').write_text(
            json.dumps(final_report, indent=2) + '\n'
        )
        export = {
            'format_version': 1,
            'model_target': (
                'stable_worldmodel.wm.vq_lewm.deployment.DistilledLeWM'
            ),
            'modules': model.deployment_state_dict(),
            'embedding_dim': int(cfg.codebook.embedding_dim),
            'history_size': int(cfg.wm.history_size),
            'action_dim': action_dim,
        }
        atomic_torch_save(export, output_dir / 'weights_final.pt')
        current_hash = sha256_file(
            resolve_weights_path(cfg.paths.codebook_checkpoint)
        )
        if current_hash != codebook_hash:
            raise RuntimeError('frozen codebook changed during training')
        print(
            f'Training complete; export={output_dir / "weights_final.pt"}; '
            f'requirements_met={final_ok}',
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
