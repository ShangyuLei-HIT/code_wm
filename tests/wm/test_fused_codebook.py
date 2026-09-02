import torch

from stable_worldmodel.wm.vq_lewm.fused_codebook import (
    CodeStatistics,
    code_statistics,
    extract_mutual_matches,
    merge_codebooks,
    unbalanced_sinkhorn,
)


def statistics(mass):
    mass = torch.tensor(mass, dtype=torch.float32)
    count = (mass * 100).long()
    error = torch.ones_like(mass)
    return CodeStatistics(
        counts=count,
        mass=mass,
        mean_squared_error=error,
        distortion=mass * error,
        radius=torch.ones_like(mass),
    )


def test_code_statistics_does_not_assign_mass_to_dead_codes():
    codebook = torch.tensor([[0.0], [10.0], [100.0]])
    latents = torch.tensor([[0.1], [0.2], [9.9], [10.1]])
    result = code_statistics(latents, codebook, batch_size=2, codebook_chunk_size=2)
    assert result.counts.tolist() == [2, 2, 0]
    assert result.mass.tolist() == [0.5, 0.5, 0.0]
    assert result.radius[-1] == 0


def test_uot_hardening_and_merge_are_one_to_one():
    cost = torch.tensor([[0.0, 20.0], [20.0, 0.0]])
    mass = torch.tensor([0.5, 0.5])
    plan, diagnostics = unbalanced_sinkhorn(
        cost,
        mass,
        mass,
        epsilon=0.1,
        rho_source=1.0,
        rho_target=1.0,
        max_iterations=200,
    )
    assert diagnostics['transported_mass'] > 0
    matches = extract_mutual_matches(
        plan,
        cost.sqrt(),
        statistics([0.5, 0.5]),
        statistics([0.5, 0.5]),
        keep_threshold=0.0,
        mass_threshold=0.9,
        radius_threshold=1.0,
        ward_threshold=1.0,
    )
    assert matches['source_indices'].tolist() == [0, 1]
    assert matches['target_indices'].tolist() == [0, 1]
    fused, first_map, second_map = merge_codebooks(
        torch.tensor([[0.0], [10.0]]),
        torch.tensor([[0.2], [9.8]]),
        mass,
        mass,
        matches['source_indices'],
        matches['target_indices'],
    )
    assert fused.shape == (2, 1)
    assert first_map.tolist() == [0, 1]
    assert second_map.tolist() == [0, 1]
    torch.testing.assert_close(fused[:, 0], torch.tensor([0.1, 9.9]))


def test_unmatched_codes_are_retained_without_movement():
    first = torch.tensor([[0.0], [10.0]])
    second = torch.tensor([[0.2], [9.8], [50.0]])
    fused, _, second_map = merge_codebooks(
        first,
        second,
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.4, 0.4, 0.2]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
    )
    assert fused.shape == (3, 1)
    assert second_map.tolist() == [0, 1, 2]
    assert fused[-1] == 50.0
