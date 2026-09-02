"""Create and audit deterministic K8192 rigid-transform conditions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from stable_worldmodel.wm.vq_lewm.affine import RigidLatentTransform
from stable_worldmodel.wm.vq_lewm.distillation import (
    sha256_file,
    squared_distances,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-codebook', required=True)
    parser.add_argument('--teacher-cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20260826)
    parser.add_argument('--translation-scale', type=float, default=1.0)
    parser.add_argument(
        '--mode',
        choices=('rigid', 'rotation_only', 'translation_only'),
        default='rigid',
    )
    parser.add_argument('--audit-samples', type=int, default=512)
    return parser.parse_args()


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_dump(value: dict, path: Path) -> None:
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def latent_moments(path: Path, chunk_size: int = 8192):
    latent = np.load(path, mmap_mode='r')
    dim = int(latent.shape[-1])
    vector_sum = np.zeros(dim, dtype=np.float64)
    squared_sum = 0.0
    count = 0
    for start in range(0, len(latent), chunk_size):
        chunk = np.asarray(
            latent[start : start + chunk_size], dtype=np.float64
        ).reshape(-1, dim)
        vector_sum += chunk.sum(axis=0)
        squared_sum += float(np.square(chunk).sum())
        count += len(chunk)
    center = vector_sum / count
    centered_squared_sum = squared_sum - count * float(center @ center)
    rms_radius = float(np.sqrt(max(0.0, centered_squared_sum / count)))
    return torch.from_numpy(center), rms_radius, tuple(latent.shape)


def proper_rotation(dim: int, seed: int):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    gaussian = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    rotation, upper = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diag(upper))
    signs[signs == 0] = 1
    rotation = rotation * signs
    if float(torch.linalg.det(rotation)) < 0:
        rotation[:, -1].neg_()
    direction = torch.randn(dim, generator=generator, dtype=torch.float64)
    direction /= direction.norm()
    return rotation, direction


def covariance_spectrum_and_rank(values: torch.Tensor):
    values = values.double()
    centered = values - values.mean(0)
    covariance = centered.t() @ centered / max(1, len(values))
    spectrum = torch.linalg.eigvalsh(covariance)
    clipped = spectrum.clamp_min(0.0)
    probabilities = clipped / clipped.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    return spectrum, effective_rank


def audit_transform(
    latent_path: Path,
    original_codebook: torch.Tensor,
    transformed_codebook: torch.Tensor,
    transform: RigidLatentTransform,
    sample_count: int,
    seed: int,
) -> dict:
    latent = np.load(latent_path, mmap_mode='r')
    generator = np.random.default_rng(seed)
    flat_count = int(np.prod(latent.shape[:-1]))
    flat_indices = np.sort(
        generator.choice(
            flat_count, size=min(sample_count, flat_count), replace=False
        )
    )
    row = flat_indices // latent.shape[1]
    step = flat_indices % latent.shape[1]
    sample = torch.from_numpy(
        np.asarray(latent[row, step], dtype=np.float32)
    )
    transformed = transform(sample)
    restored = transform.inverse(transformed)

    original_distances = squared_distances(sample, original_codebook)
    transformed_distances = squared_distances(
        transformed, transformed_codebook
    )
    original_values, original_indices = original_distances.topk(
        32, largest=False, dim=-1
    )
    transformed_values, transformed_indices = transformed_distances.topk(
        32, largest=False, dim=-1
    )
    original_probabilities = torch.softmax(-original_values, dim=-1)
    transformed_probabilities = torch.softmax(-transformed_values, dim=-1)

    permutation = torch.randperm(len(sample), generator=torch.Generator().manual_seed(seed))
    pair_count = len(sample) // 2
    left = sample[:pair_count]
    right = sample[permutation[:pair_count]]
    left_transformed = transform(left)
    right_transformed = transform(right)
    before = torch.linalg.vector_norm(left - right, dim=-1)
    after = torch.linalg.vector_norm(
        left_transformed - right_transformed, dim=-1
    )
    relative_pair_error = (
        (after - before).abs() / before.clamp_min(1e-12)
    )

    latent_spectrum, latent_rank = covariance_spectrum_and_rank(sample)
    transformed_latent_spectrum, transformed_latent_rank = (
        covariance_spectrum_and_rank(transformed)
    )
    latent_spectrum_relative_error = (
        (transformed_latent_spectrum - latent_spectrum).abs()
        / latent_spectrum.abs().clamp_min(1e-12)
    )
    codebook_spectrum, codebook_rank = covariance_spectrum_and_rank(
        original_codebook
    )
    transformed_codebook_spectrum, transformed_codebook_rank = (
        covariance_spectrum_and_rank(transformed_codebook)
    )
    codebook_spectrum_relative_error = (
        (transformed_codebook_spectrum - codebook_spectrum).abs()
        / codebook_spectrum.abs().clamp_min(1e-12)
    )

    return {
        'sample_count': len(sample),
        'inverse_max_abs_error': float((restored - sample).abs().max()),
        'pairwise_distance_max_relative_error': float(
            relative_pair_error.max()
        ),
        'squared_distance_max_abs_error': float(
            (transformed_distances - original_distances).abs().max()
        ),
        'top1_agreement': float(
            (transformed_indices[:, 0] == original_indices[:, 0])
            .float()
            .mean()
        ),
        'top32_ordered_agreement': float(
            (transformed_indices == original_indices).all(dim=-1)
            .float()
            .mean()
        ),
        'top32_probability_max_abs_error': float(
            (transformed_probabilities - original_probabilities).abs().max()
        ),
        'covariance_spectrum_max_relative_error': float(
            latent_spectrum_relative_error.max()
        ),
        'latent_effective_rank_original': float(latent_rank),
        'latent_effective_rank_transformed': float(transformed_latent_rank),
        'latent_effective_rank_relative_error': float(
            (transformed_latent_rank - latent_rank).abs()
            / latent_rank.abs().clamp_min(1e-12)
        ),
        'codebook_covariance_spectrum_max_relative_error': float(
            codebook_spectrum_relative_error.max()
        ),
        'codebook_effective_rank_original': float(codebook_rank),
        'codebook_effective_rank_transformed': float(
            transformed_codebook_rank
        ),
        'codebook_effective_rank_relative_error': float(
            (transformed_codebook_rank - codebook_rank).abs()
            / codebook_rank.abs().clamp_min(1e-12)
        ),
    }


def main():
    args = parse_args()
    source_dir = Path(args.source_codebook).expanduser().resolve()
    teacher_cache = Path(args.teacher_cache).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    source_weights = source_dir / 'weights.pt'
    latent_path = teacher_cache / 'train_teacher_latents.npy'
    if not source_weights.is_file():
        raise FileNotFoundError(source_weights)
    if not latent_path.is_file():
        raise FileNotFoundError(latent_path)
    if args.translation_scale < 0:
        raise ValueError('translation scale must be non-negative')

    source_hash = sha256_file(source_weights)
    existing_manifest = output_dir / 'rigid_transform_manifest.json'
    upgrade_existing = False
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text())
        expected = (
            existing.get('source_codebook_sha256') == source_hash
            and existing.get('transform_seed') == args.seed
            and existing.get('translation_scale') == args.translation_scale
            and existing.get('mode') == args.mode
        )
        required = [
            output_dir / 'weights.pt',
            output_dir / 'config.json',
            output_dir / 'transform.pt',
        ]
        if expected and all(path.is_file() for path in required):
            if existing.get('acceptance', {}).get('passed'):
                print(
                    f'Validated existing rigid codebook: {output_dir}',
                    flush=True,
                )
                return
            upgrade_existing = True
        else:
            raise RuntimeError(
                f'conflicting rigid codebook already exists: {output_dir}'
            )
    state = torch.load(source_weights, map_location='cpu', weights_only=True)
    if 'teacher.weight' not in state or 'student.weight' not in state:
        raise KeyError('source codebook must contain teacher and student weights')
    dim = int(state['teacher.weight'].shape[1])
    center, rms_radius, latent_shape = latent_moments(latent_path)
    rotation, direction = proper_rotation(dim, args.seed)
    if args.mode == 'translation_only':
        rotation = torch.eye(dim, dtype=torch.float64)
    translation = direction * (args.translation_scale * rms_radius)
    if args.mode == 'rotation_only':
        translation.zero_()
    transform = RigidLatentTransform(rotation, center, translation)

    transformed_state = dict(state)
    for key in ('teacher.weight', 'student.weight'):
        transformed_state[key] = transform(state[key].float()).contiguous()

    source_config = source_dir / 'config.json'
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not upgrade_existing:
        atomic_torch_save(transformed_state, output_dir / 'weights.pt')
        temporary_config = output_dir / '.config.json.tmp'
        shutil.copyfile(source_config, temporary_config)
        os.replace(temporary_config, output_dir / 'config.json')
    transform_payload = {
        'format_version': 1,
        'rotation': transform.rotation.cpu(),
        'center': transform.center.cpu(),
        'translation': transform.translation.cpu(),
        'transform_seed': torch.tensor(args.seed, dtype=torch.int64),
        'translation_scale': torch.tensor(
            args.translation_scale, dtype=torch.float64
        ),
        'rms_radius': torch.tensor(rms_radius, dtype=torch.float64),
    }
    if upgrade_existing:
        existing_state = torch.load(
            output_dir / 'weights.pt', map_location='cpu', weights_only=True
        )
        for key in ('teacher.weight', 'student.weight'):
            torch.testing.assert_close(
                existing_state[key], transformed_state[key],
                rtol=0.0, atol=0.0,
            )
        existing_transform = torch.load(
            output_dir / 'transform.pt', map_location='cpu', weights_only=True
        )
        for key in ('rotation', 'center', 'translation'):
            torch.testing.assert_close(
                existing_transform[key], transform_payload[key],
                rtol=0.0, atol=0.0,
            )
    else:
        atomic_torch_save(transform_payload, output_dir / 'transform.pt')

    audit = audit_transform(
        latent_path,
        state['teacher.weight'].float(),
        transformed_state['teacher.weight'].float(),
        transform,
        args.audit_samples,
        args.seed,
    )
    rotation64 = transform.rotation.double()
    orthogonality_error = float(
        (
            rotation64.t() @ rotation64
            - torch.eye(dim, dtype=torch.float64)
        )
        .abs()
        .max()
    )
    determinant = float(torch.linalg.det(rotation64))
    tolerances = {
        'determinant_abs_error': 1.0e-5,
        'orthogonality_max_abs_error': 1.0e-5,
        'inverse_max_abs_error': 1.0e-4,
        'pairwise_distance_max_relative_error': 1.0e-5,
        'top32_ordered_agreement_minimum': 0.99,
        'top32_probability_max_abs_error': 1.0e-3,
        'covariance_spectrum_max_relative_error': 1.0e-4,
        'effective_rank_relative_error': 1.0e-5,
    }
    checks = {
        'proper_determinant': abs(determinant - 1.0) <= 1.0e-5,
        'orthogonal': orthogonality_error <= 1.0e-5,
        'inverse': audit['inverse_max_abs_error'] <= 1.0e-4,
        'pairwise_distances': (
            audit['pairwise_distance_max_relative_error'] <= 1.0e-5
        ),
        'hard_tokens': audit['top1_agreement'] == 1.0,
        'top32_order': audit['top32_ordered_agreement'] >= 0.99,
        'top32_probabilities': (
            audit['top32_probability_max_abs_error'] <= 1.0e-3
        ),
        'latent_covariance_spectrum': (
            audit['covariance_spectrum_max_relative_error'] <= 1.0e-4
        ),
        'codebook_covariance_spectrum': (
            audit['codebook_covariance_spectrum_max_relative_error'] <= 1.0e-4
        ),
        'effective_rank': max(
            audit['latent_effective_rank_relative_error'],
            audit['codebook_effective_rank_relative_error'],
        ) <= 1.0e-5,
    }
    acceptance = {
        'passed': all(checks.values()),
        'tolerances': tolerances,
        'checks': checks,
    }
    if not acceptance['passed']:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f'rigid transform invariant checks failed: {failed}')

    manifest = {
        'format_version': 2,
        'source_codebook': str(source_dir),
        'source_codebook_sha256': source_hash,
        'transformed_codebook_sha256': sha256_file(output_dir / 'weights.pt'),
        'transform_sha256': sha256_file(output_dir / 'transform.pt'),
        'teacher_cache': str(teacher_cache),
        'teacher_latent_shape': list(latent_shape),
        'embedding_dim': dim,
        'transform_seed': args.seed,
        'mode': args.mode,
        'translation_scale': args.translation_scale,
        'rms_radius': rms_radius,
        'translation_norm': float(transform.translation.norm()),
        'rotation_determinant': determinant,
        'orthogonality_max_abs_error': orthogonality_error,
        'formula': 'T(z)=(z-center)@rotation.T+center+translation',
        'audit': audit,
        'acceptance': acceptance,
    }
    atomic_json_dump(manifest, existing_manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
