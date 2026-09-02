"""Build an aligned concat or sequential multi-task UOT codebook."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from stable_worldmodel.wm.vq_lewm.alignment import load_alignment_bundle
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
        default='scripts/train/config/multitask_vq_lewm_three_tasks.yaml',
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


def sample_latents(latents: torch.Tensor, limit: int, seed: int):
    count = min(len(latents), limit)
    if count < 1:
        raise ValueError('no latent vectors are available for fusion')
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(latents), generator=generator)[:count]
    return latents[indices]


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


def normalize_cost(cost: torch.Tensor, mode: str):
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


def tensor_summary(value: torch.Tensor):
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


def task_order(cfg) -> list[int]:
    reference_id = int(cfg.alignment.reference_task_id)
    source_ids = cfg.alignment.get('source_task_ids')
    if source_ids is None:
        source_ids = [int(cfg.alignment.source_task_id)]
    order = [reference_id, *[int(value) for value in source_ids]]
    expected = set(range(len(cfg.tasks)))
    if len(order) != len(set(order)) or set(order) != expected:
        raise ValueError(
            'reference_task_id + source_task_ids must cover every task once'
        )
    return order


def task_budget(task, cfg) -> float:
    return float(task.get('qe_budget', cfg.fusion.get('qe_budget', 0.02)))


@torch.no_grad()
def choose_uot_merge(
    cfg,
    fused: torch.Tensor,
    source_codebook: torch.Tensor,
    fused_train: torch.Tensor,
    source_train: torch.Tensor,
    validation: dict[str, torch.Tensor],
    baseline_qe: dict[str, float],
    evaluation_names: list[str],
    source_name: str,
    device: torch.device,
):
    fused_stats = code_statistics(
        fused_train,
        fused.to(device),
        batch_size=int(cfg.fusion.assignment_batch_size),
        codebook_chunk_size=int(cfg.fusion.codebook_chunk_size),
    )
    source_stats = code_statistics(
        source_train,
        source_codebook.to(device),
        batch_size=int(cfg.fusion.assignment_batch_size),
        codebook_chunk_size=int(cfg.fusion.codebook_chunk_size),
    )
    fused_active = torch.nonzero(
        fused_stats.counts > 0, as_tuple=False
    ).squeeze(1)
    source_active = torch.nonzero(
        source_stats.counts > 0, as_tuple=False
    ).squeeze(1)
    fused_active_codes = fused[fused_active].to(device)
    source_active_codes = source_codebook[source_active].to(device)
    distances = torch.cdist(fused_active_codes, source_active_codes)
    raw_cost = distances.square()
    if str(cfg.fusion.cost) == 'local':
        radii = (
            fused_stats.radius[fused_active, None].to(device)
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

    selected = None
    trials = []
    selected_transport = None
    for rho, epsilon in itertools.product(
        list_value(cfg.fusion.search, 'rho'),
        list_value(cfg.fusion.search, 'epsilon'),
    ):
        plan, diagnostics = unbalanced_sinkhorn(
            cost,
            fused_stats.mass[fused_active].to(device),
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
                fused_stats,
                source_stats,
                source_active=fused_active,
                target_active=source_active,
                keep_threshold=float(keep),
                mass_threshold=float(mass),
                radius_threshold=float(radius),
                ward_threshold=float(ward),
            )
            candidate, fused_map, source_map = merge_codebooks(
                fused,
                source_codebook,
                fused_stats.mass,
                source_stats.mass,
                matches['source_indices'],
                matches['target_indices'],
            )
            qe = {
                name: quantization_mse(
                    validation[name],
                    candidate.to(device),
                    batch_size=int(cfg.fusion.assignment_batch_size),
                    codebook_chunk_size=int(cfg.fusion.codebook_chunk_size),
                )
                for name in evaluation_names
            }
            ratios = {
                name: qe[name] / max(baseline_qe[name], 1e-12)
                for name in evaluation_names
            }
            trial = {
                'rho': float(rho),
                'epsilon': float(epsilon),
                'keep_threshold': float(keep),
                'mass_threshold': float(mass),
                'radius_threshold': float(radius),
                'ward_threshold': float(ward),
                'num_merges': len(matches['source_indices']),
                'num_embeddings': len(candidate),
                'quantization_mse': qe,
                'quantization_mse_ratio': ratios,
            }
            trials.append(trial)
            acceptable = all(
                ratios[name]
                <= 1.0 + task_budget(
                    next(task for task in cfg.tasks if str(task.name) == name),
                    cfg,
                )
                for name in evaluation_names
            )
            score = (
                len(matches['source_indices']),
                -max(ratios.values()),
            )
            if acceptable and (selected is None or score > selected['score']):
                selected = {
                    'score': score,
                    'trial': trial,
                    'matches': matches,
                    'fused': candidate,
                    'fused_map': fused_map,
                    'source_map': source_map,
                }
                selected_transport = diagnostics
    if selected is None:
        raise RuntimeError(
            f'no UOT candidate for {source_name} satisfies all QE budgets'
        )
    matches = selected['matches']
    report = {
        'source_task': source_name,
        'num_codes_before': len(fused),
        'num_source_codes': len(source_codebook),
        'active_codes_before': len(fused_active),
        'active_source_codes': len(source_active),
        'mutual_candidate_count': int(matches['mutual_candidate_count']),
        'num_merges': len(matches['source_indices']),
        'num_codes_after': len(selected['fused']),
        'cost_scale': cost_scale,
        'selected': selected['trial'],
        'transport': selected_transport,
        'match_distance': tensor_summary(matches['distance']),
        'match_local_radius_ratio': tensor_summary(
            matches['local_radius_ratio']
        ),
        'match_normalized_ward': tensor_summary(
            matches['normalized_ward']
        ),
        'trials': trials,
    }
    statistics = {
        'fused': fused_stats.to_dict(),
        'source': source_stats.to_dict(),
        'matches': matches,
    }
    return selected, report, statistics


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    method = args.method or str(cfg.fusion.method)
    device = torch.device(str(cfg.fusion.device))
    order = task_order(cfg)
    reference_name = str(cfg.tasks[order[0]].name)
    alignment_enabled = bool(cfg.alignment.get('enabled', True))
    alignments = (
        load_alignment_bundle(
            cfg.paths.alignment_checkpoint,
            expected_dim=int(cfg.codebook.embedding_dim),
        )
        if alignment_enabled
        else {}
    )
    alignments = {name: value.to(device) for name, value in alignments.items()}

    codebooks = {}
    for task_id in order:
        task = cfg.tasks[task_id]
        name = str(task.name)
        codebook = load_codebook_weights(task.codebook_checkpoint).to(device)
        if name in alignments:
            codebook = alignments[name](codebook)
        codebooks[name] = codebook.cpu()

    task_maps = {}
    stage_reports = []
    statistics_payload = {}
    if method == 'concat':
        chunks = []
        offset = 0
        for task_id in order:
            name = str(cfg.tasks[task_id].name)
            chunks.append(codebooks[name])
            task_maps[name] = torch.arange(len(codebooks[name])) + offset
            offset += len(codebooks[name])
        fused = torch.cat(chunks, dim=0)
        report = {
            'method': 'concat',
            'num_tasks': len(order),
            'task_code_counts': {
                name: len(codebook) for name, codebook in codebooks.items()
            },
            'num_merges': 0,
            'num_embeddings': len(fused),
        }
    elif method == 'uot':
        train = {}
        validation = {}
        baseline_qe = {}
        for position, task_id in enumerate(order):
            task = cfg.tasks[task_id]
            name = str(task.name)
            task_train = load_latents(task.train_latents)
            task_validation = load_latents(task.validation_latents)
            if name in alignments:
                task_train = alignments[name](task_train.to(device)).cpu()
                task_validation = alignments[name](
                    task_validation.to(device)
                ).cpu()
            train[name] = sample_latents(
                task_train,
                int(cfg.fusion.statistics_samples_per_task),
                int(cfg.seed) + position,
            )
            validation[name] = task_validation
            baseline_qe[name] = quantization_mse(
                task_validation,
                codebooks[name].to(device),
                batch_size=int(cfg.fusion.assignment_batch_size),
                codebook_chunk_size=int(cfg.fusion.codebook_chunk_size),
            )

        fused = codebooks[reference_name]
        task_maps[reference_name] = torch.arange(len(fused))
        included = [reference_name]
        for task_id in order[1:]:
            source_name = str(cfg.tasks[task_id].name)
            selected, stage, stage_statistics = choose_uot_merge(
                cfg,
                fused,
                codebooks[source_name],
                torch.cat([train[name] for name in included], dim=0),
                train[source_name],
                validation,
                baseline_qe,
                [*included, source_name],
                source_name,
                device,
            )
            fused = selected['fused']
            if not torch.equal(
                selected['fused_map'], torch.arange(len(selected['fused_map']))
            ):
                raise RuntimeError('sequential fusion changed existing token ids')
            task_maps[source_name] = selected['source_map']
            included.append(source_name)
            stage_reports.append(stage)
            statistics_payload[source_name] = stage_statistics
        report = {
            'method': 'sequential_uot',
            'num_tasks': len(order),
            'reference_task': reference_name,
            'task_code_counts': {
                name: len(codebook) for name, codebook in codebooks.items()
            },
            'baseline_quantization_mse': baseline_qe,
            'num_merges': sum(stage['num_merges'] for stage in stage_reports),
            'num_embeddings': len(fused),
            'stages': stage_reports,
        }
    else:
        raise ValueError(f'unsupported fusion method: {method}')

    weights_path, metadata_path = checkpoint_output(
        cfg.paths.fused_codebook_checkpoint
    )
    weights_payload = {
        'teacher.weight': fused.contiguous(),
        'task_token_maps': task_maps,
    }
    weights_payload.update(
        {f'{name}_token_map': value for name, value in task_maps.items()}
    )
    atomic_torch_save(weights_payload, weights_path)

    task_metadata = []
    for task_id in order:
        task = cfg.tasks[task_id]
        name = str(task.name)
        checkpoint = resolve_weights_path(task.codebook_checkpoint)
        task_metadata.append(
            {
                'task_id': task_id,
                'name': name,
                'codebook': str(checkpoint),
                'codebook_sha256': sha256_file(checkpoint),
                'alignment_applied': name in alignments,
            }
        )
    alignment_path = (
        Path(cfg.paths.alignment_checkpoint).expanduser().resolve()
        if alignment_enabled
        else None
    )
    metadata = {
        **report,
        'format_version': 2,
        'task_order': [str(cfg.tasks[index].name) for index in order],
        'tasks': task_metadata,
        'alignment_checkpoint': (
            str(alignment_path) if alignment_path is not None else None
        ),
        'alignment_sha256': (
            sha256_file(alignment_path) if alignment_path is not None else None
        ),
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
