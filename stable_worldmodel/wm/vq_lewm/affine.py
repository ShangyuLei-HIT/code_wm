"""Deterministic rigid affine transforms for latent-coordinate experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class RigidLatentTransform(nn.Module):
    """A frozen proper rotation plus translation in latent coordinates."""

    def __init__(
        self,
        rotation: torch.Tensor,
        center: torch.Tensor,
        translation: torch.Tensor,
    ) -> None:
        super().__init__()
        rotation = rotation.detach().float().contiguous()
        center = center.detach().float().contiguous()
        translation = translation.detach().float().contiguous()
        if rotation.ndim != 2 or rotation.size(0) != rotation.size(1):
            raise ValueError('rotation must be a square matrix')
        dim = rotation.size(0)
        if tuple(center.shape) != (dim,) or tuple(translation.shape) != (dim,):
            raise ValueError('center and translation must match rotation')
        identity = torch.eye(dim, dtype=rotation.dtype)
        if not torch.allclose(
            rotation.t() @ rotation, identity, rtol=1e-4, atol=1e-4
        ):
            raise ValueError('rotation is not orthogonal within float32 tolerance')
        if float(torch.linalg.det(rotation.double())) <= 0.0:
            raise ValueError('rotation must be proper (determinant > 0)')
        self.register_buffer('rotation', rotation, persistent=False)
        self.register_buffer('center', center, persistent=False)
        self.register_buffer('translation', translation, persistent=False)

    @property
    def affine_bias(self) -> torch.Tensor:
        return (
            self.center
            + self.translation
            - self.center @ self.rotation.t()
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        rotation = self.rotation.to(device=latent.device, dtype=latent.dtype)
        center = self.center.to(device=latent.device, dtype=latent.dtype)
        translation = self.translation.to(
            device=latent.device, dtype=latent.dtype
        )
        with torch.autocast(device_type=latent.device.type, enabled=False):
            return (latent - center) @ rotation.t() + center + translation

    def inverse(self, latent: torch.Tensor) -> torch.Tensor:
        rotation = self.rotation.to(device=latent.device, dtype=latent.dtype)
        center = self.center.to(device=latent.device, dtype=latent.dtype)
        translation = self.translation.to(
            device=latent.device, dtype=latent.dtype
        )
        with torch.autocast(device_type=latent.device.type, enabled=False):
            return (latent - center - translation) @ rotation + center


def load_rigid_latent_transform(
    checkpoint: str | Path,
    *,
    expected_dim: int | None = None,
) -> RigidLatentTransform:
    checkpoint = Path(checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    required = {'rotation', 'center', 'translation'}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f'latent transform is missing keys: {sorted(missing)}')
    transform = RigidLatentTransform(
        payload['rotation'], payload['center'], payload['translation']
    )
    if expected_dim is not None and transform.rotation.size(0) != expected_dim:
        raise ValueError(
            f'expected latent transform dimension {expected_dim}, '
            f'got {transform.rotation.size(0)}'
        )
    return transform


@torch.no_grad()
def initialize_adapter_from_transform(
    adapter: nn.Module,
    transform: RigidLatentTransform,
) -> None:
    """Initialize a Linear adapter so it exactly applies the transform."""
    if not isinstance(adapter, nn.Linear):
        raise TypeError('rigid latent initialization requires nn.Linear adapter')
    dim = transform.rotation.size(0)
    if adapter.in_features != dim or adapter.out_features != dim:
        raise ValueError('adapter and latent transform dimensions differ')
    if adapter.bias is None:
        raise ValueError('rigid latent initialization requires adapter bias')
    adapter.weight.copy_(
        transform.rotation.to(
            device=adapter.weight.device, dtype=adapter.weight.dtype
        )
    )
    adapter.bias.copy_(
        transform.affine_bias.to(
            device=adapter.bias.device, dtype=adapter.bias.dtype
        )
    )


__all__ = [
    'RigidLatentTransform',
    'initialize_adapter_from_transform',
    'load_rigid_latent_transform',
]
