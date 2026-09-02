"""Inference-only LeWM wrapper that projects latents onto a frozen codebook."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from stable_worldmodel.wm.lewm.lewm import LeWM

from .distillation import load_codebook_weights, nearest_code_indices


class CodebookQuantizedLeWM(LeWM):
    """Run an existing continuous LeWM through a frozen nearest-neighbor codebook.

    Observation and goal latents are always quantized before planning. When
    ``quantize_rollout_predictions`` is true, every autoregressive prediction is
    also projected back onto the codebook before it is fed to the next step.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        *,
        codebook_checkpoint: str | Path | None = None,
        codebook_weights: torch.Tensor | None = None,
        codebook_weight_key: str = 'teacher.weight',
        num_embeddings: int | None = None,
        embedding_dim: int | None = None,
        codebook_chunk_size: int = 2048,
        quantize_rollout_predictions: bool = False,
    ) -> None:
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        if codebook_weights is None:
            if codebook_checkpoint is None:
                raise ValueError(
                    'codebook_checkpoint or codebook_weights is required'
                )
            expected_shape = (
                (num_embeddings, embedding_dim)
                if num_embeddings is not None and embedding_dim is not None
                else None
            )
            codebook_weights = load_codebook_weights(
                codebook_checkpoint,
                weight_key=codebook_weight_key,
                expected_shape=expected_shape,
            )
        codebook_weights = codebook_weights.detach().float().contiguous()
        if embedding_dim is not None and codebook_weights.size(1) != embedding_dim:
            raise ValueError('codebook embedding dimension does not match')
        self.register_buffer('codebook', codebook_weights, persistent=False)
        self.codebook_chunk_size = int(codebook_chunk_size)
        self.quantize_rollout_predictions = bool(
            quantize_rollout_predictions
        )

    def quantize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        indices = nearest_code_indices(
            latent,
            self.codebook,
            k=1,
            codebook_chunk_size=self.codebook_chunk_size,
        ).squeeze(-1)
        return F.embedding(indices, self.codebook)

    def encode(self, info: dict[str, torch.Tensor]):
        info = super().encode(info)
        info['emb'] = self.quantize_latent(info['emb'])
        return info

    def predict(
        self, latent: torch.Tensor, action_embedding: torch.Tensor
    ) -> torch.Tensor:
        prediction = super().predict(latent, action_embedding)
        if self.quantize_rollout_predictions:
            prediction = self.quantize_latent(prediction)
        return prediction


__all__ = ['CodebookQuantizedLeWM']
