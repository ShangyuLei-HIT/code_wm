"""Statistics, unbalanced optimal transport, and conservative code merging."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .distillation import nearest_code_indices


@dataclass
class CodeStatistics:
    counts: torch.Tensor
    mass: torch.Tensor
    mean_squared_error: torch.Tensor
    distortion: torch.Tensor
    radius: torch.Tensor
    heldout_counts: torch.Tensor | None = None

    def to_dict(self) -> dict[str, torch.Tensor | None]:
        return {
            'counts': self.counts,
            'mass': self.mass,
            'mean_squared_error': self.mean_squared_error,
            'distortion': self.distortion,
            'radius': self.radius,
            'heldout_counts': self.heldout_counts,
        }


@torch.no_grad()
def code_statistics(
    latents: torch.Tensor,
    codebook: torch.Tensor,
    *,
    heldout_latents: torch.Tensor | None = None,
    batch_size: int = 8192,
    codebook_chunk_size: int = 2048,
) -> CodeStatistics:
    """Accumulate real occupancy, distortion, and RMS cluster radius."""
    latents = latents.reshape(-1, latents.size(-1))
    codebook = codebook.float()
    k = codebook.size(0)
    device = codebook.device
    counts = torch.zeros(k, dtype=torch.long, device=device)
    squared_error = torch.zeros(k, dtype=torch.float64, device=device)
    for start in range(0, len(latents), batch_size):
        batch = latents[start : start + batch_size].to(device).float()
        assignments = nearest_code_indices(
            batch,
            codebook,
            codebook_chunk_size=codebook_chunk_size,
        ).squeeze(-1)
        error = (batch - codebook[assignments]).square().sum(-1).double()
        counts += torch.bincount(assignments, minlength=k)
        squared_error.scatter_add_(0, assignments, error)
    mass = counts.double() / counts.sum().clamp_min(1)
    mean_error = squared_error / counts.clamp_min(1).double()
    mean_error = torch.where(
        counts > 0, mean_error, torch.zeros_like(mean_error)
    )
    heldout_counts = None
    if heldout_latents is not None:
        heldout_counts = torch.zeros(k, dtype=torch.long, device=device)
        heldout_latents = heldout_latents.reshape(-1, heldout_latents.size(-1))
        for start in range(0, len(heldout_latents), batch_size):
            assignments = nearest_code_indices(
                heldout_latents[start : start + batch_size].to(device),
                codebook,
                codebook_chunk_size=codebook_chunk_size,
            ).squeeze(-1)
            heldout_counts += torch.bincount(assignments, minlength=k)
    return CodeStatistics(
        counts=counts.cpu(),
        mass=mass.float().cpu(),
        mean_squared_error=mean_error.float().cpu(),
        distortion=(mass * mean_error).float().cpu(),
        radius=mean_error.sqrt().float().cpu(),
        heldout_counts=(
            heldout_counts.cpu() if heldout_counts is not None else None
        ),
    )


@torch.no_grad()
def quantization_mse(
    latents: torch.Tensor,
    codebook: torch.Tensor,
    *,
    batch_size: int = 8192,
    codebook_chunk_size: int = 2048,
) -> float:
    latents = latents.reshape(-1, latents.size(-1))
    codebook = codebook.float()
    total = torch.zeros((), device=codebook.device, dtype=torch.float64)
    for start in range(0, len(latents), batch_size):
        batch = latents[start : start + batch_size].to(codebook.device).float()
        indices = nearest_code_indices(
            batch,
            codebook,
            codebook_chunk_size=codebook_chunk_size,
        ).squeeze(-1)
        total += (batch - codebook[indices]).square().sum().double()
    return float(total / max(1, latents.numel()))


@torch.no_grad()
def unbalanced_sinkhorn(
    cost: torch.Tensor,
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    *,
    epsilon: float,
    rho_source: float,
    rho_target: float,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Solve entropy/KL-regularized UOT with log-domain Sinkhorn updates."""
    if epsilon <= 0 or rho_source <= 0 or rho_target <= 0:
        raise ValueError('epsilon and both rho values must be positive')
    if cost.shape != (source_mass.numel(), target_mass.numel()):
        raise ValueError('cost shape and marginal sizes differ')
    if (source_mass <= 0).any() or (target_mass <= 0).any():
        raise ValueError('zero-mass codes must be removed before UOT')
    cost = cost.float()
    log_kernel = -cost / float(epsilon)
    if candidate_mask is not None:
        if candidate_mask.shape != cost.shape:
            raise ValueError('candidate mask shape differs from cost')
        if (~candidate_mask).all(dim=1).any() or (~candidate_mask).all(dim=0).any():
            raise ValueError('candidate mask isolates an active code')
        log_kernel = log_kernel.masked_fill(~candidate_mask, -torch.inf)
    log_a = source_mass.to(cost).log()
    log_b = target_mass.to(cost).log()
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    tau_source = rho_source / (rho_source + epsilon)
    tau_target = rho_target / (rho_target + epsilon)
    converged = False
    residual = float('inf')
    for iteration in range(max_iterations):
        previous_u = log_u
        previous_v = log_v
        log_u = tau_source * (
            log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        )
        log_v = tau_target * (
            log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        )
        residual = float(
            torch.maximum(
                (log_u - previous_u).abs().max(),
                (log_v - previous_v).abs().max(),
            )
        )
        if residual <= tolerance:
            converged = True
            break
    plan = (log_u[:, None] + log_kernel + log_v[None, :]).exp()
    return plan, {
        'iterations': iteration + 1,
        'residual': residual,
        'converged': converged,
        'transported_mass': float(plan.sum()),
    }


@torch.no_grad()
def extract_mutual_matches(
    plan: torch.Tensor,
    distances: torch.Tensor,
    source_stats: CodeStatistics,
    target_stats: CodeStatistics,
    *,
    source_active: torch.Tensor | None = None,
    target_active: torch.Tensor | None = None,
    keep_threshold: float = 0.5,
    mass_threshold: float = 0.5,
    radius_threshold: float = 1.0,
    ward_threshold: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Harden a soft UOT plan with conservative one-to-one gates."""
    source_active = (
        torch.arange(plan.size(0)) if source_active is None else source_active
    ).long().cpu()
    target_active = (
        torch.arange(plan.size(1)) if target_active is None else target_active
    ).long().cpu()
    plan = plan.float().cpu()
    distances = distances.float().cpu()
    source_mass = source_stats.mass[source_active]
    target_mass = target_stats.mass[target_active]
    row_sum = plan.sum(1)
    col_sum = plan.sum(0)
    row_best = plan.argmax(1)
    col_best = plan.argmax(0)
    source_local = torch.arange(plan.size(0))
    mutual = col_best[row_best] == source_local
    target_local = row_best
    pair_mass = plan[source_local, target_local]
    source_keep = row_sum / source_mass.clamp_min(1e-12)
    target_keep = col_sum / target_mass.clamp_min(1e-12)
    conditional_source = pair_mass / row_sum.clamp_min(1e-12)
    conditional_target = pair_mass / col_sum[target_local].clamp_min(1e-12)
    source_indices = source_active[source_local]
    target_indices = target_active[target_local]
    pair_distance = distances[source_local, target_local]
    radius_sum = (
        source_stats.radius[source_indices]
        + target_stats.radius[target_indices]
    )
    local_ratio = pair_distance / radius_sum.clamp_min(1e-12)
    a = source_stats.mass[source_indices]
    b = target_stats.mass[target_indices]
    ward = a * b / (a + b).clamp_min(1e-12) * pair_distance.square()
    denominator = (
        source_stats.distortion[source_indices]
        + target_stats.distortion[target_indices]
    )
    normalized_ward = ward / denominator.clamp_min(1e-12)
    accepted = (
        mutual
        & (
            torch.minimum(source_keep, target_keep[target_local])
            >= keep_threshold
        )
        & (
            torch.minimum(conditional_source, conditional_target)
            >= mass_threshold
        )
        & (local_ratio <= radius_threshold)
        & (normalized_ward <= ward_threshold)
    )
    return {
        'source_indices': source_indices[accepted],
        'target_indices': target_indices[accepted],
        'distance': pair_distance[accepted],
        'local_radius_ratio': local_ratio[accepted],
        'normalized_ward': normalized_ward[accepted],
        'source_keep': source_keep[accepted],
        'target_keep': target_keep[target_local][accepted],
        'source_conditional_mass': conditional_source[accepted],
        'target_conditional_mass': conditional_target[accepted],
        'mutual_candidate_count': mutual.sum(),
    }


@torch.no_grad()
def merge_codebooks(
    source_codebook: torch.Tensor,
    target_codebook: torch.Tensor,
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge accepted pairs while preserving source ids where possible."""
    source_codebook = source_codebook.float().cpu()
    target_codebook = target_codebook.float().cpu()
    source_indices = source_indices.long().cpu()
    target_indices = target_indices.long().cpu()
    if source_indices.unique().numel() != source_indices.numel():
        raise ValueError('source matches are not one-to-one')
    if target_indices.unique().numel() != target_indices.numel():
        raise ValueError('target matches are not one-to-one')
    fused = source_codebook.clone()
    target_to_source = torch.full(
        (len(target_codebook),), -1, dtype=torch.long
    )
    if len(source_indices):
        a = source_mass[source_indices].float()[:, None]
        b = target_mass[target_indices].float()[:, None]
        fused[source_indices] = (
            a * source_codebook[source_indices]
            + b * target_codebook[target_indices]
        ) / (a + b).clamp_min(1e-12)
        target_to_source[target_indices] = source_indices
    unmatched = torch.nonzero(
        target_to_source < 0, as_tuple=False
    ).squeeze(1)
    target_map = target_to_source
    target_map[unmatched] = torch.arange(
        len(source_codebook), len(source_codebook) + len(unmatched)
    )
    fused = torch.cat((fused, target_codebook[unmatched]), dim=0)
    source_map = torch.arange(len(source_codebook), dtype=torch.long)
    return fused, source_map, target_map


__all__ = [
    'CodeStatistics',
    'code_statistics',
    'extract_mutual_matches',
    'merge_codebooks',
    'quantization_mse',
    'unbalanced_sinkhorn',
]
