"""Continuous-latent distillation with a frozen, offline-only codebook."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_weights_path(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    return checkpoint / 'weights.pt' if checkpoint.is_dir() else checkpoint


def load_codebook_weights(
    checkpoint: str | Path,
    *,
    weight_key: str = 'teacher.weight',
    expected_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Load only the immutable table from a generated codebook checkpoint."""
    weights_path = resolve_weights_path(checkpoint)
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    if weight_key not in state:
        raise KeyError(f'{weight_key!r} is absent from {weights_path}')
    table = state[weight_key].detach().float().contiguous()
    if expected_shape is not None and tuple(table.shape) != expected_shape:
        raise ValueError(
            f'expected codebook shape {expected_shape}, got {tuple(table.shape)}'
        )
    return table


def squared_distances(inputs: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    inputs = inputs.float()
    codebook = codebook.float()
    return (
        inputs.square().sum(dim=-1, keepdim=True)
        + codebook.square().sum(dim=-1)
        - 2.0 * inputs @ codebook.t()
    ).clamp_min_(0.0)


def sparse_topk_kl(
    student_latent: torch.Tensor,
    codebook: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    *,
    temperature: float = 1.0,
    codebook_chunk_size: int = 2048,
) -> torch.Tensor:
    """Compute tau^2 KL(p_teacher || p_student) without a dense teacher.

    The student denominator is exact over every code. Only the teacher's
    non-zero top-k support is materialized.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if codebook_chunk_size < 1:
        raise ValueError('codebook_chunk_size must be positive')
    latent_dim = student_latent.size(-1)
    flat_student = student_latent.reshape(-1, latent_dim).float()
    flat_indices = teacher_topk_indices.reshape(flat_student.size(0), -1).long()
    flat_probs = teacher_topk_probs.reshape(flat_student.size(0), -1).float()
    flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    log_normalizer = None
    for start in range(0, codebook.size(0), codebook_chunk_size):
        chunk = codebook[start : start + codebook_chunk_size]
        chunk_logits = -squared_distances(flat_student, chunk) / temperature
        chunk_lse = torch.logsumexp(chunk_logits, dim=-1)
        log_normalizer = (
            chunk_lse
            if log_normalizer is None
            else torch.logaddexp(log_normalizer, chunk_lse)
        )

    selected_codes = F.embedding(flat_indices, codebook.float())
    selected_logits = -(
        flat_student[:, None, :] - selected_codes
    ).square().sum(dim=-1) / temperature
    student_log_probs = selected_logits - log_normalizer[:, None]
    teacher_log_probs = flat_probs.clamp_min(1e-12).log()
    per_vector = (
        flat_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    return (temperature**2 * per_vector.mean()).clamp_min(0.0)


@torch.no_grad()
def nearest_code_indices(
    latent: torch.Tensor,
    codebook: torch.Tensor,
    *,
    k: int = 1,
    codebook_chunk_size: int = 2048,
) -> torch.Tensor:
    """Exact nearest code indices with bounded intermediate memory."""
    if not 1 <= k <= codebook.size(0):
        raise ValueError('k must be in [1, codebook_size]')
    flat = latent.reshape(-1, latent.size(-1)).float()
    best_values = torch.full(
        (flat.size(0), k), float('inf'), device=flat.device
    )
    best_indices = torch.zeros(
        (flat.size(0), k), dtype=torch.long, device=flat.device
    )
    for start in range(0, codebook.size(0), codebook_chunk_size):
        distances = squared_distances(
            flat, codebook[start : start + codebook_chunk_size]
        )
        local_k = min(k, distances.size(1))
        values, indices = distances.topk(local_k, largest=False, dim=-1)
        indices = indices + start
        candidates_v = torch.cat((best_values, values), dim=-1)
        candidates_i = torch.cat((best_indices, indices), dim=-1)
        best_values, order = candidates_v.topk(k, largest=False, dim=-1)
        best_indices = candidates_i.gather(-1, order)
    return best_indices.view(*latent.shape[:-1], k)


def sequence_teacher_forcing(
    student_latent: torch.Tensor,
    teacher_codebook_latent: torch.Tensor,
    alpha: float,
    *,
    mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one source for every frame and target in each sequence."""
    if student_latent.shape != teacher_codebook_latent.shape:
        raise ValueError('student and teacher latent shapes must match')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be in [0, 1]')
    if mask is None:
        random = torch.rand(
            student_latent.size(0),
            1,
            1,
            device=student_latent.device,
            generator=generator,
        )
        mask = random < alpha
    expected = (student_latent.size(0), 1, 1)
    if tuple(mask.shape) != expected:
        raise ValueError(f'sequence mask must have shape {expected}')
    mixed = torch.where(mask, student_latent, teacher_codebook_latent)
    return mixed, mask


def phase_for_epoch(epoch: int, phase_epochs: Iterable[int]) -> tuple[int, int]:
    """Return one-based phase and zero-based epoch within the phase."""
    cursor = 0
    for phase, length in enumerate(phase_epochs, start=1):
        if cursor <= epoch < cursor + length:
            return phase, epoch - cursor
        cursor += length
    raise ValueError(f'epoch {epoch} is outside the configured phases')


def teacher_forcing_alpha(
    global_step: int,
    steps_per_epoch: int,
    phase_epochs: Iterable[int] = (4, 10, 2),
) -> float:
    phase_epochs = tuple(phase_epochs)
    phase1_steps = phase_epochs[0] * steps_per_epoch
    phase2_steps = phase_epochs[1] * steps_per_epoch
    if global_step < phase1_steps:
        return 0.0
    if global_step >= phase1_steps + phase2_steps:
        return 1.0
    return min(1.0, (global_step - phase1_steps) / max(1, phase2_steps - 1))


def cosine_phase_lr(
    step_in_phase: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_fraction: float,
) -> float:
    """Warm up to base_lr, then cosine decay to min_lr within one phase."""
    warmup_steps = int(total_steps * warmup_fraction)
    if warmup_steps and step_in_phase < warmup_steps:
        return base_lr * float(step_in_phase + 1) / warmup_steps
    progress = (step_in_phase - warmup_steps) / max(
        1, total_steps - warmup_steps - 1
    )
    progress = min(1.0, max(0.0, progress))
    factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * factor


def effective_rank_from_moments(
    count: torch.Tensor, vector_sum: torch.Tensor, outer_sum: torch.Tensor
) -> torch.Tensor:
    count = count.float().clamp_min(2.0)
    mean = vector_sum.float() / count
    covariance = outer_sum.float() / count - mean[:, None] * mean[None, :]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    return torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())


class JointDistillationLeWM(nn.Module):
    """Student encoder and predictor; no teacher network is instantiated."""

    def __init__(
        self,
        student_encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module,
        pred_proj: nn.Module,
        *,
        embedding_dim: int = 192,
        adapter: nn.Module | None = None,
        codebook_checkpoint: str | Path | None = None,
        codebook_weights: torch.Tensor | None = None,
        codebook_weight_key: str = 'teacher.weight',
        num_embeddings: int | None = None,
    ) -> None:
        super().__init__()
        self.student_encoder = student_encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector
        self.pred_proj = pred_proj
        self.adapter = adapter or nn.Linear(embedding_dim, embedding_dim)
        if isinstance(self.adapter, nn.Linear):
            if self.adapter.in_features != self.adapter.out_features:
                raise ValueError('adapter must preserve the teacher dimension')
            nn.init.eye_(self.adapter.weight)
            if self.adapter.bias is not None:
                nn.init.zeros_(self.adapter.bias)

        if codebook_weights is None:
            if codebook_checkpoint is None:
                raise ValueError('codebook_checkpoint or codebook_weights is required')
            expected = (
                (num_embeddings, embedding_dim)
                if num_embeddings is not None
                else None
            )
            codebook_weights = load_codebook_weights(
                codebook_checkpoint,
                weight_key=codebook_weight_key,
                expected_shape=expected,
            )
        codebook_weights = codebook_weights.detach().float().contiguous()
        if codebook_weights.size(1) != embedding_dim:
            raise ValueError('codebook and student dimensions differ')
        self.register_buffer('codebook', codebook_weights, persistent=False)

    def encode_student(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 5:
            raise ValueError('pixels must have shape (B,T,C,H,W)')
        batch_size = pixels.size(0)
        pixels = pixels.to(next(self.student_encoder.parameters()).dtype)
        flat = rearrange(pixels, 'b t ... -> (b t) ...')
        features = self.student_encoder(
            flat, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]
        latent = self.adapter(self.projector(features))
        return rearrange(latent, '(b t) d -> b t d', b=batch_size)

    def lookup_teacher_codes(self, hard_tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(hard_tokens.long(), self.codebook)

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_embedding = self.action_encoder(action)
        prediction = self.predictor(latent, action_embedding)
        flat = rearrange(prediction, 'b t d -> (b t) d')
        projected = self.pred_proj(flat)
        return rearrange(projected, '(b t) d -> b t d', b=latent.size(0))

    def deployment_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Export only modules required by image-to-prediction deployment."""
        names = (
            'student_encoder',
            'projector',
            'adapter',
            'action_encoder',
            'predictor',
            'pred_proj',
        )
        return {name: getattr(self, name).state_dict() for name in names}


__all__ = [
    'JointDistillationLeWM',
    'cosine_phase_lr',
    'effective_rank_from_moments',
    'load_codebook_weights',
    'nearest_code_indices',
    'phase_for_epoch',
    'resolve_weights_path',
    'sequence_teacher_forcing',
    'sha256_file',
    'sparse_topk_kl',
    'squared_distances',
    'teacher_forcing_alpha',
]
