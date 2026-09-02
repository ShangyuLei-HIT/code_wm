"""Shared task-conditioned LeWM models for discrete and continuous training."""

from __future__ import annotations

from pathlib import Path

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from stable_worldmodel.wm.lewm.lewm import LeWM

from .distillation import JointDistillationLeWM


class PaddedActionEncoder(nn.Module):
    """Pad task-specific action blocks before a shared action encoder."""

    def __init__(self, encoder: nn.Module, input_dim: int | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        inferred = getattr(encoder, 'input_dim', None)
        if inferred is None and hasattr(encoder, 'patch_embed'):
            inferred = encoder.patch_embed.in_channels
        self.input_dim = int(input_dim if input_dim is not None else inferred)
        if self.input_dim < 1:
            raise ValueError('input_dim must be positive')
        if inferred is not None and int(inferred) != self.input_dim:
            raise ValueError(
                'wrapped encoder input dimension differs from padding target: '
                f'{inferred} != {self.input_dim}'
            )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        width = action.size(-1)
        if width > self.input_dim:
            raise ValueError(
                f'action width {width} exceeds shared width {self.input_dim}'
            )
        if width < self.input_dim:
            action = F.pad(action, (0, self.input_dim - width))
        return self.encoder(action)


def _task_condition(
    action_embedding: torch.Tensor,
    task_embedding: nn.Embedding,
    task_id: torch.Tensor | int,
) -> torch.Tensor:
    task_id = torch.as_tensor(task_id, device=action_embedding.device).long()
    if task_id.ndim == 0:
        task_id = task_id.expand(action_embedding.size(0))
    if tuple(task_id.shape) not in {
        (action_embedding.size(0),),
        (action_embedding.size(0), action_embedding.size(1)),
    }:
        raise ValueError('task_id must be scalar, (B,), or (B,T)')
    condition = task_embedding(task_id)
    if condition.ndim == 2:
        condition = condition[:, None, :]
    return action_embedding + condition


class MultiTaskJointDistillationLeWM(JointDistillationLeWM):
    """One student, frozen shared table, predictor, and task embedding."""

    def __init__(
        self,
        student_encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module,
        pred_proj: nn.Module,
        *,
        num_tasks: int = 2,
        embedding_dim: int = 192,
        adapter: nn.Module | None = None,
        codebook_checkpoint: str | Path | None = None,
        codebook_weights: torch.Tensor | None = None,
        codebook_weight_key: str = 'teacher.weight',
        num_embeddings: int | None = None,
    ) -> None:
        super().__init__(
            student_encoder=student_encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
            embedding_dim=embedding_dim,
            adapter=adapter,
            codebook_checkpoint=codebook_checkpoint,
            codebook_weights=codebook_weights,
            codebook_weight_key=codebook_weight_key,
            num_embeddings=num_embeddings,
        )
        self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
        nn.init.normal_(self.task_embedding.weight, std=0.02)

    def predict(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor | int,
    ) -> torch.Tensor:
        conditioning = _task_condition(
            self.action_encoder(action), self.task_embedding, task_id
        )
        prediction = self.predictor(latent, conditioning)
        flat = rearrange(prediction, 'b t d -> (b t) d')
        projected = self.pred_proj(flat)
        return rearrange(projected, '(b t) d -> b t d', b=latent.size(0))

    def deployment_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        state = super().deployment_state_dict()
        state['task_embedding'] = self.task_embedding.state_dict()
        return state


class MultiTaskContinuousLeWM(LeWM):
    """Native continuous M3 baseline with the same trainable components."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module,
        pred_proj: nn.Module,
        *,
        adapter: nn.Module | None = None,
        embedding_dim: int = 192,
        num_tasks: int = 2,
    ) -> None:
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        self.adapter = adapter or nn.Linear(embedding_dim, embedding_dim)
        if isinstance(self.adapter, nn.Linear):
            nn.init.eye_(self.adapter.weight)
            if self.adapter.bias is not None:
                nn.init.zeros_(self.adapter.bias)
        self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
        nn.init.normal_(self.task_embedding.weight, std=0.02)

    def encode(self, info: dict[str, torch.Tensor]):
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        batch_size = pixels.size(0)
        flat = rearrange(pixels, 'b t ... -> (b t) ...')
        features = self.encoder(
            flat, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]
        latent = self.adapter(self.projector(features))
        info['emb'] = rearrange(latent, '(b t) d -> b t d', b=batch_size)
        return info

    def predict_actions(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor | int,
    ) -> torch.Tensor:
        conditioning = _task_condition(
            self.action_encoder(action), self.task_embedding, task_id
        )
        prediction = self.predictor(latent, conditioning)
        flat = rearrange(prediction, 'b t d -> (b t) d')
        prediction = self.pred_proj(flat)
        return rearrange(
            prediction, '(b t) d -> b t d', b=latent.size(0)
        )


class MultiTaskDistilledLeWM(MultiTaskContinuousLeWM):
    """Deployment wrapper selecting one task while retaining shared weights."""

    def __init__(self, *args, default_task_id: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.default_task_id = int(default_task_id)

    def predict(self, latent: torch.Tensor, action_embedding: torch.Tensor):
        conditioning = _task_condition(
            action_embedding, self.task_embedding, self.default_task_id
        )
        prediction = self.predictor(latent, conditioning)
        flat = rearrange(prediction, 'b t d -> (b t) d')
        prediction = self.pred_proj(flat)
        return rearrange(
            prediction, '(b t) d -> b t d', b=latent.size(0)
        )


__all__ = [
    'PaddedActionEncoder',
    'MultiTaskContinuousLeWM',
    'MultiTaskDistilledLeWM',
    'MultiTaskJointDistillationLeWM',
]
