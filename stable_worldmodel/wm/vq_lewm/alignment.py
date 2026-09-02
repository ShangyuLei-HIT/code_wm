"""Similarity-Procrustes alignment between independently trained latents."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def fit_similarity_procrustes(
    source: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit ``scale * source @ rotation + bias`` to ``reference``."""
    if source.shape != reference.shape:
        raise ValueError('source and reference anchors must have equal shapes')
    if source.ndim < 2 or source.size(-1) < 1:
        raise ValueError('anchors must have shape (..., latent_dim)')
    source = source.reshape(-1, source.size(-1)).double()
    reference = reference.reshape(-1, reference.size(-1)).double()
    if source.size(0) < 2:
        raise ValueError('at least two anchor vectors are required')
    if not torch.isfinite(source).all() or not torch.isfinite(reference).all():
        raise ValueError('anchors must be finite')

    source_mean = source.mean(dim=0, keepdim=True)
    reference_mean = reference.mean(dim=0, keepdim=True)
    source_centered = source - source_mean
    reference_centered = reference - reference_mean
    denominator = source_centered.square().sum()
    if denominator <= torch.finfo(source.dtype).eps:
        raise ValueError('source anchors have zero variance')
    u, singular_values, vh = torch.linalg.svd(
        source_centered.t() @ reference_centered,
        full_matrices=False,
    )
    rotation = u @ vh
    scale = singular_values.sum() / denominator
    bias = reference_mean - scale * source_mean @ rotation
    return rotation.float(), scale.float(), bias.float()


def apply_alignment(
    latent: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor | float,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply a row-vector similarity transform without changing shape."""
    rotation = rotation.to(device=latent.device, dtype=latent.dtype)
    scale = torch.as_tensor(scale, device=latent.device, dtype=latent.dtype)
    bias = bias.to(device=latent.device, dtype=latent.dtype)
    return scale * latent @ rotation + bias


class SimilarityAlignment(nn.Module):
    """Frozen, invertible similarity transform used only by offline tools."""

    def __init__(
        self,
        rotation: torch.Tensor,
        scale: torch.Tensor | float,
        bias: torch.Tensor,
    ) -> None:
        super().__init__()
        rotation = rotation.detach().float().contiguous()
        scale = torch.as_tensor(scale).detach().float().reshape(())
        bias = bias.detach().float().reshape(-1).contiguous()
        if rotation.ndim != 2 or rotation.size(0) != rotation.size(1):
            raise ValueError('rotation must be square')
        dim = rotation.size(0)
        if tuple(bias.shape) != (dim,):
            raise ValueError('bias dimension does not match rotation')
        identity = torch.eye(dim, dtype=rotation.dtype)
        if not torch.allclose(
            rotation.t() @ rotation, identity, rtol=1e-4, atol=1e-4
        ):
            raise ValueError('rotation is not orthogonal')
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError('scale must be finite and positive')
        self.register_buffer('rotation', rotation)
        self.register_buffer('scale', scale)
        self.register_buffer('bias', bias)

    @property
    def dimension(self) -> int:
        return self.rotation.size(0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return apply_alignment(latent, self.rotation, self.scale, self.bias)

    def inverse(self, aligned: torch.Tensor) -> torch.Tensor:
        rotation = self.rotation.to(device=aligned.device, dtype=aligned.dtype)
        scale = self.scale.to(device=aligned.device, dtype=aligned.dtype)
        bias = self.bias.to(device=aligned.device, dtype=aligned.dtype)
        return ((aligned - bias) / scale) @ rotation.t()

    def as_linear(self) -> nn.Linear:
        layer = nn.Linear(self.dimension, self.dimension)
        with torch.no_grad():
            layer.weight.copy_(self.scale * self.rotation.t())
            layer.bias.copy_(self.bias)
        layer.requires_grad_(False)
        return layer


def load_similarity_alignment(
    checkpoint: str | Path,
    *,
    expected_dim: int | None = None,
    source_task: str | None = None,
) -> SimilarityAlignment:
    payload = torch.load(
        Path(checkpoint).expanduser().resolve(),
        map_location='cpu',
        weights_only=True,
    )
    if 'alignments' in payload:
        alignments = payload['alignments']
        if source_task is None:
            if len(alignments) != 1:
                raise ValueError(
                    'source_task is required for a multi-source alignment '
                    f'checkpoint; available tasks: {sorted(alignments)}'
                )
            source_task = next(iter(alignments))
        if source_task not in alignments:
            raise KeyError(
                f'alignment checkpoint has no source task {source_task!r}; '
                f'available tasks: {sorted(alignments)}'
            )
        payload = alignments[source_task]
    required = {'rotation', 'scale', 'bias'}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f'alignment checkpoint is missing: {sorted(missing)}')
    alignment = SimilarityAlignment(
        payload['rotation'], payload['scale'], payload['bias']
    )
    if expected_dim is not None and alignment.dimension != expected_dim:
        raise ValueError(
            f'expected latent dimension {expected_dim}, '
            f'got {alignment.dimension}'
        )
    return alignment


def load_alignment_bundle(
    checkpoint: str | Path,
    *,
    expected_dim: int | None = None,
) -> dict[str, SimilarityAlignment]:
    """Load either a legacy single-source or a multi-source checkpoint."""
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if 'alignments' in payload:
        names = list(payload['alignments'])
    else:
        source = payload.get('source_task')
        if not source:
            raise KeyError(
                'legacy alignment checkpoint does not identify source_task'
            )
        names = [str(source)]
    return {
        name: load_similarity_alignment(
            path,
            expected_dim=expected_dim,
            source_task=name,
        )
        for name in names
    }


def _effective_rank(values: torch.Tensor) -> float:
    centered = values.double() - values.double().mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    probabilities = variance / variance.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def _canonical_correlations(
    first: torch.Tensor,
    second: torch.Tensor,
    regularization: float = 1e-6,
) -> torch.Tensor:
    first = first.double() - first.double().mean(0, keepdim=True)
    second = second.double() - second.double().mean(0, keepdim=True)
    denominator = max(1, first.size(0) - 1)
    cxx = first.t() @ first / denominator
    cyy = second.t() @ second / denominator
    cxy = first.t() @ second / denominator

    def inverse_sqrt(matrix: torch.Tensor) -> torch.Tensor:
        values, vectors = torch.linalg.eigh(matrix)
        floor = regularization * values.max().clamp_min(1.0)
        values = values.clamp_min(floor)
        return (vectors * values.rsqrt()) @ vectors.t()

    whitened = inverse_sqrt(cxx) @ cxy @ inverse_sqrt(cyy)
    return torch.linalg.svdvals(whitened).clamp(0.0, 1.0).float()


@torch.no_grad()
def alignment_metrics(
    source: torch.Tensor,
    reference: torch.Tensor,
    alignment: SimilarityAlignment | None = None,
) -> dict[str, float | list[float]]:
    """Return held-out residual and representation diagnostics."""
    if source.shape != reference.shape:
        raise ValueError('source and reference must have equal shapes')
    source = source.reshape(-1, source.size(-1)).float()
    reference = reference.reshape(-1, reference.size(-1)).float()
    mapped = alignment(source) if alignment is not None else source
    residual = mapped - reference
    mse = residual.square().mean()
    reference_variance = (
        reference - reference.mean(0, keepdim=True)
    ).square().mean()
    identity_mse = (source - reference).square().mean()
    correlations = _canonical_correlations(mapped, reference)
    return {
        'mse': float(mse),
        'rmse': float(mse.sqrt()),
        'normalized_rmse': float(
            (mse / reference_variance.clamp_min(1e-12)).sqrt()
        ),
        'r2': float(1.0 - mse / reference_variance.clamp_min(1e-12)),
        'cosine_similarity': float(
            F.cosine_similarity(mapped, reference, dim=-1).mean()
        ),
        'identity_mse': float(identity_mse),
        'mse_improvement_ratio': float(
            identity_mse / mse.clamp_min(1e-12)
        ),
        'svcca_mean': float(correlations.mean()),
        'canonical_correlations': correlations.cpu().tolist(),
        'source_effective_rank': _effective_rank(source),
        'mapped_effective_rank': _effective_rank(mapped),
        'reference_effective_rank': _effective_rank(reference),
    }


__all__ = [
    'SimilarityAlignment',
    'alignment_metrics',
    'apply_alignment',
    'fit_similarity_procrustes',
    'load_alignment_bundle',
    'load_similarity_alignment',
]
