"""Teacher-, cache-, and codebook-free deployment model."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from stable_worldmodel.wm.lewm.lewm import LeWM


class DistilledLeWM(LeWM):
    """The six trainable modules exported by joint distillation."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        adapter: nn.Module,
        action_encoder: nn.Module,
        predictor: nn.Module,
        pred_proj: nn.Module,
    ) -> None:
        super().__init__(
            encoder, predictor, action_encoder, projector, pred_proj
        )
        self.adapter = adapter

    def encode(self, info: dict[str, torch.Tensor]):
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        batch_size = pixels.size(0)
        flat = rearrange(pixels, 'b t ... -> (b t) ...')
        features = self.encoder(
            flat, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]
        latent = self.adapter(self.projector(features))
        info['emb'] = rearrange(latent, '(b t) d -> b t d', b=batch_size)
        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])
        return info

    def predict(self, latent: torch.Tensor, action_embedding: torch.Tensor):
        prediction = self.predictor(latent, action_embedding)
        flat = rearrange(prediction, 'b t d -> (b t) d')
        prediction = self.pred_proj(flat)
        return rearrange(
            prediction, '(b t) d -> b t d', b=latent.size(0)
        )


__all__ = ['DistilledLeWM']
