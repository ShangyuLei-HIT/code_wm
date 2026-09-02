import torch

from stable_worldmodel.wm.vq_lewm.alignment import (
    SimilarityAlignment,
    alignment_metrics,
    fit_similarity_procrustes,
)
from stable_worldmodel.wm.vq_lewm.distillation import nearest_code_indices


def test_similarity_procrustes_recovers_exact_transform():
    torch.manual_seed(12)
    source = torch.randn(4096, 8)
    rotation, _ = torch.linalg.qr(torch.randn(8, 8))
    scale = torch.tensor(1.7)
    bias = torch.randn(8)
    reference = scale * source @ rotation + bias
    fitted_rotation, fitted_scale, fitted_bias = fit_similarity_procrustes(
        source[:3000], reference[:3000]
    )
    alignment = SimilarityAlignment(
        fitted_rotation, fitted_scale, fitted_bias
    )
    torch.testing.assert_close(
        alignment(source[3000:]),
        reference[3000:],
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        alignment.inverse(reference[3000:]),
        source[3000:],
        rtol=1e-4,
        atol=1e-4,
    )
    assert alignment_metrics(
        source[3000:], reference[3000:], alignment
    )['r2'] > 0.9999


def test_similarity_transform_preserves_nearest_code_assignments():
    torch.manual_seed(23)
    latent = torch.randn(128, 7)
    codebook = torch.randn(31, 7)
    rotation, _ = torch.linalg.qr(torch.randn(7, 7))
    alignment = SimilarityAlignment(rotation, 0.37, torch.randn(7))
    old = nearest_code_indices(latent, codebook)
    new = nearest_code_indices(alignment(latent), alignment(codebook))
    assert torch.equal(old, new)


def test_alignment_checkpoint_roundtrip(tmp_path):
    rotation = torch.eye(3)
    checkpoint = tmp_path / 'alignment.pt'
    torch.save(
        {'rotation': rotation, 'scale': torch.tensor(2.0), 'bias': torch.ones(3)},
        checkpoint,
    )
    from stable_worldmodel.wm.vq_lewm.alignment import load_similarity_alignment

    alignment = load_similarity_alignment(checkpoint, expected_dim=3)
    torch.testing.assert_close(alignment(torch.zeros(2, 3)), torch.ones(2, 3))
