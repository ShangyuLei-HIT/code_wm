"""Align two codebooks and build concatenated or adaptive-UOT fusion."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from stable_worldmodel.wm.vq_lewm.alignment import (
    load_similarity_alignment,
)
from stable_worldmodel.wm.vq_lewm.distillation import (
    load_codebook_weights,
    resolve_weights_path,
    sha256_file,
)
from stable_worldmodel.wm.vq_lewm.fused_codebook import (
    code_statistics,
    extract_mutual_matches,
    merge_codebooks,
    quantization_mse,
    unbalanced_sinkhorn,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='scripts/train/config/multitask_vq_lewm.yaml',
    )
    parser.add_argument('--method', choices=('concat', 'uot'), default=None)
    return parser.parse_args()


def load_latents(path: str | Path) -> torch.Tensor:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location='cpu',
        weights_only=False,
    )
    if torch.is_tensor(payload):
        latents = payload
    elif isinstance(payload, dict) and 'latents' in payload:
        latents = payload['latents']
    else:
        raise ValueError(f'{path} does not contain a latents tensor')
    return latents.reshape(-1, latents.size(-1)).float()


def equal_sample(
    first: torch.Tensor,
    second: torch.Tensor,
    limit: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(len(first), len(second), limit)
    if count < 1:
        raise ValueError('no latent vectors are available for fusion')
    generator = torch.Generator().manual_seed(seed)
    first_indices = torch.randperm(len(first), generator=generator)[:count]
    second_indices = torch.randperm(len(second), generator=generator)[:count]
    return first[first_indices], second[second_indices]


def checkpoint_output(path: str | Path) -> tuple[Path, Path]:
    path = Path(path).expanduser().resolve()
    if path.suffix:
        return path, path.with_suffix('.metadata.json')
    return path / 'weights.pt', path / 'metadata.json'


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def normalize_cost(cost: torch.Tensor, mode: str) -> tuple[torch.Tensor, float]:
    if mode == 'none':
        return cost, 1.0
    if mode == 'mean':
        scale = cost.mean()
    elif mode == 'median':
        scale = cost.median()
    else:
        raise ValueError(f'unknown cost normalization: {mode}')
    return cost / scale.clamp_min(1e-12), float(scale)


def list_value(cfg, name):
    value = cfg.get(name)
    if isinstance(value, (float, int)):
        return [value]
    return list(value)


def tensor_summary(value: torch.Tensor) -> dict[str, float | int]:
    value = value.float()
    if not value.numel():
        return {'count': 0}
    return {
        'count': value.numel(),
        'min': float(value.min()),
        'mean': float(value.mean()),
        'median': float(value.median()),
        'max': float(value.max()),
    }


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    method = args.method or str(cfg.fusion.method)
    device = torch.device(str(cfg.fusion.device))
    reference_id = int(cfg.alignment.reference_task_id)
    source_id = int(cfg.alignment.source_task_id)
    reference_task = cfg.tasks[reference_id]
    source_task = cfg.tasks[source_id]
    reference_codebook = load_codebook_weights(
        reference_task.codebook_checkpoint
    ).to(device)
    source_codebook_raw = load_codebook_weights(
        source_task.codebook_checkpoint
    ).to(device)
    alignment = load_similarity_alignment(
        cfg.paths.alignment_checkpoint,
        expected_dim=reference_codebook.size(1),
    ).to(device)
    alignment_enabled = bool(cfg.alignment.get('enabled', True))
    source_codebook = (
        alignment(source_codebook_raw)
        if alignment_enabled
        else source_codebook_raw
    )
    if method == 'concat':
        fused = torch.cat((reference_codebook, source_codebook), dim=0).cpu()
        reference_map = torch.arange(len(reference_codebook))
        source_map = torch.arange(len(source_codebook)) + len(reference_codebook)
        selected = None
        report = {
            'method': 'concat',
            'num_reference_codes': len(reference_codebook),
            'num_source_codes': len(source_codebook),
            'num_merges': 0,
            'num_embeddings': len(fused),
        }
        statistics_payload = {}
    elif method == 'uot':
        reference_train = load_latents(reference_task.train_latents)
        source_train = load_latents(source_task.train_latents).to(device)
        source_train = (
            alignment(source_train) if alignment_enabled else source_train
        ).cpu()
        reference_val = load_latents(reference_task.validation_latents)
        source_val = load_latents(source_task.validation_latents).to(device)
        source_val = (
            alignment(source_val) if alignment_enabled else source_val
        ).cpu()
        reference_train, source_train = equal_sample(
            reference_train,
            source_train,
            int(cfg.fusion.statistics_samples_per_task),
            int(cfg.seed),
        )
        reference_stats = code_statistics(
            reference_train,
            reference_codebook,
            heldout_latents=reference_val,
            batch_size=int(cfg.fusion.assignment_batch_size),
            codebook_chunk_size=int(cfg.fusion.codebook_chunk_size),
        )
        source_stats = code_statistics(
            source_train,
            source_codebook,
            heldout_latents=source_val,
            batch_size=int(cfg.fusion.assignment_batch_size),
            codebook_chunk_size=int(cfg.fusion.codebook_chunk_size),
        )
        reference_active = torch.nonzero(
            reference_stats.counts > 0, as_tuple=False
        ).squeeze(1)
        source_active = torch.nonzero(
            source_stats.counts > 0, as_tuple=False
        ).squeeze(1)
        reference_active_codes = reference_codebook[reference_active.to(device)]
        source_active_codes = source_codebook[source_active.to(device)]
        distances = torch.cdist(reference_active_codes, source_active_codes)
        raw_cost = distances.square()
        if str(cfg.fusion.cost) == 'local':
            radii = (
                reference_stats.radius[reference_active, None].to(device)
                + source_stats.radius[source_active][None, :].to(device)
            )
            raw_cost = raw_cost / radii.square().clamp_min(1e-12)
        cost, cost_scale = normalize_cost(
            raw_cost, str(cfg.fusion.cost_normalization)
        )
        candidate_distance = cfg.fusion.get('candidate_distance')
        candidate_mask = None
        if candidate_distance is not None:
            candidate_mask = distances <= float(candidate_distance)
            candidate_mask.scatter_(1, distances.argmin(1)[:, None], True)
            candidate_mask.scatter_(0, distances.argmin(0)[None, :], True)

        baseline_reference = quantization_mse(
            reference_val, reference_codebook,
            batch_size=int(cfg.fusion.assignment_batch_size),
            codebook_chunk_size=int(cfg.fusion.codebook_chunk_size),
        )
        baseline_source = quantization_mse(
            source_val, source_codebook,
            batch_size=int(cfg.fusion.assignment_batch_size),
            codebook_chunk_size=int(cfg.fusion.codebook_chunk_size),
        )
        selected = None
        trials = []
        transport_diagnostics = None
        for rho, epsilon in itertools.product(
            list_value(cfg.fusion.search, 'rho'),
            list_value(cfg.fusion.search, 'epsilon'),
        ):
            plan, diagnostics = unbalanced_sinkhorn(
                cost,
                reference_stats.mass[reference_active].to(device),
                source_stats.mass[source_active].to(device),
                epsilon=float(epsilon),
                rho_source=float(rho),
                rho_target=float(rho),
                max_iterations=int(cfg.fusion.max_iterations),
                tolerance=float(cfg.fusion.tolerance),
                candidate_mask=candidate_mask,
            )
            plan_cpu = plan.cpu()
            for keep, mass, radius, ward in itertools.product(
                list_value(cfg.fusion.search, 'keep_threshold'),
                list_value(cfg.fusion.search, 'mass_threshold'),
                list_value(cfg.fusion.search, 'radius_threshold'),
                list_value(cfg.fusion.search, 'ward_threshold'),
            ):
                matches = extract_mutual_matches(
                    plan_cpu,
                    distances.cpu(),
                    reference_stats,
                    source_stats,
                    source_active=reference_active,
                    target_active=source_active,
                    keep_threshold=float(keep),
                    mass_threshold=float(mass),
                    radius_threshold=float(radius),
                    ward_threshold=float(ward),
                )
                candidate_fused, candidate_reference_map, candidate_source_map = (
                    merge_codebooks(
                        reference_codebook,
                        source_codebook,
                        reference_stats.mass,
                        source_stats.mass,
                        matches['source_indices'],
                        matches['target_indices'],
                    )
                )
                reference_qe = quantization_mse(
                    reference_val,
                    candidate_fused.to(device),
                    batch_size=int(cfg.fusion.assignment_batch_size),
                    codebook_chunk_size=int(cfg.fusion.codebook_chunk_size),
                )
                source_qe = quantization_mse(
                    source_val,
                    candidate_fused.to(device),
                    batch_size=int(cfg.fusion.assignment_batch_size),
                    codebook_chunk_size=int(cfg.fusion.codebook_chunk_size),
                )
                reference_ratio = reference_qe / max(baseline_reference, 1e-12)
                source_ratio = source_qe / max(baseline_source, 1e-12)
                trial = {
                    'rho': float(rho),
                    'epsilon': float(epsilon),
                    'keep_threshold': float(keep),
                    'mass_threshold': float(mass),
                    'radius_threshold': float(radius),
                    'ward_threshold': float(ward),
                    'num_merges': len(matches['source_indices']),
                    'num_embeddings': len(candidate_fused),
                    'reference_qe': reference_qe,
                    'source_qe': source_qe,
                    'reference_qe_ratio': reference_ratio,
                    'source_qe_ratio': source_ratio,
                }
                trials.append(trial)
                acceptable = (
                    reference_ratio
                    <= 1.0 + float(cfg.fusion.reference_qe_budget)
                    and source_ratio
                    <= 1.0 + float(cfg.fusion.source_qe_budget)
                )
                score = (
                    len(matches['source_indices']),
                    -max(reference_ratio, source_ratio),
                )
                if acceptable and (
                    selected is None or score > selected['score']
                ):
                    selected = {
                        'score': score,
                        'trial': trial,
                        'matches': matches,
                        'fused': candidate_fused,
                        'reference_map': candidate_reference_map,
                        'source_map': candidate_source_map,
                    }
                    transport_diagnostics = diagnostics
        if selected is None:
            raise RuntimeError(
                'no UOT candidate satisfies both quantization-error budgets'
            )
        fused = selected['fused']
        reference_map = selected['reference_map']
        source_map = selected['source_map']
        matches = selected['matches']
        report = {
            'method': 'uot',
            'num_reference_codes': len(reference_codebook),
            'num_source_codes': len(source_codebook),
            'active_reference_codes': len(reference_active),
            'active_source_codes': len(source_active),
            'mutual_candidate_count': int(matches['mutual_candidate_count']),
            'num_merges': len(matches['source_indices']),
            'merge_fraction': len(matches['source_indices'])
            / max(1, min(len(reference_codebook), len(source_codebook))),
            'num_embeddings': len(fused),
            'cost': str(cfg.fusion.cost),
            'cost_normalization': str(cfg.fusion.cost_normalization),
            'cost_scale': cost_scale,
            'candidate_distance': (
                float(candidate_distance)
                if candidate_distance is not None
                else None
            ),
            'baseline_reference_qe': baseline_reference,
            'baseline_source_qe': baseline_source,
            'selected': selected['trial'],
            'transport': transport_diagnostics,
            'match_distance': tensor_summary(matches['distance']),
            'match_local_radius_ratio': tensor_summary(
                matches['local_radius_ratio']
            ),
            'match_normalized_ward': tensor_summary(
                matches['normalized_ward']
            ),
            'trials': trials,
        }
        statistics_payload = {
            'reference': reference_stats.to_dict(),
            'source': source_stats.to_dict(),
            'matches': matches,
        }
    else:
        raise ValueError(f'unsupported fusion method: {method}')

    weights_path, metadata_path = checkpoint_output(
        cfg.paths.fused_codebook_checkpoint
    )
    atomic_torch_save(
        {
            'teacher.weight': fused.contiguous(),
            'reference_token_map': reference_map,
            'source_token_map': source_map,
        },
        weights_path,
    )
    metadata = {
        **report,
        'format_version': 1,
        'reference_task': str(reference_task.name),
        'source_task': str(source_task.name),
        'reference_codebook': str(
            Path(reference_task.codebook_checkpoint).expanduser().resolve()
        ),
        'source_codebook': str(
            Path(source_task.codebook_checkpoint).expanduser().resolve()
        ),
        'reference_codebook_sha256': sha256_file(
            resolve_weights_path(reference_task.codebook_checkpoint)
        ),
        'source_codebook_sha256': sha256_file(
            resolve_weights_path(source_task.codebook_checkpoint)
        ),
        'alignment_checkpoint': str(
            Path(cfg.paths.alignment_checkpoint).expanduser().resolve()
        ),
        'alignment_sha256': sha256_file(cfg.paths.alignment_checkpoint),
        'alignment_applied': alignment_enabled,
        'weights': str(weights_path),
        'statistics_file': str(weights_path.with_name('statistics.pt')),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + '\n'
    )
    atomic_torch_save(
        statistics_payload,
        weights_path.with_name('statistics.pt'),
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f'Saved fused K={len(fused)} codebook to {weights_path}', flush=True)


if __name__ == '__main__':
    main()
