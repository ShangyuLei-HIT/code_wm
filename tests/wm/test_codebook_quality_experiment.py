import numpy as np
from omegaconf import OmegaConf

from scripts.train.summarize_codebook_quality_experiment import (
    holm_adjust,
    paired_hierarchical_bootstrap,
)


def test_experiment_matrix_uses_two_seeds_and_four_gpus():
    cfg = OmegaConf.load(
        'scripts/train/config/codebook_quality_rigid_experiment.yaml'
    )
    OmegaConf.resolve(cfg)
    assert list(cfg.seeds) == [3072, 4096]
    assert list(cfg.compute.devices) == [0, 1, 2, 3]
    assert cfg.compute.nproc_per_node == 4
    assert cfg.compute.batch_size_per_gpu == 192
    assert (
        cfg.compute.nproc_per_node * cfg.compute.batch_size_per_gpu
        == cfg.compute.global_batch_size
        == 768
    )


def test_only_requested_seed3072_diagnostics_are_registered():
    cfg = OmegaConf.load(
        'scripts/train/config/codebook_quality_rigid_experiment.yaml'
    )
    assert cfg.diagnostics.seed == 3072
    assert cfg.diagnostics.trigger_gap_percentage_points == 5.0
    assert set(cfg.diagnostics.conditions) == {
        'k8192_rotation_only',
        'k8192_translation_only',
    }


def test_paired_hierarchical_bootstrap_detects_identical_outcomes():
    first = np.asarray([True, False, True, True])
    second = np.asarray([False, True, True, False])
    result = paired_hierarchical_bootstrap(
        [first, second],
        [first.copy(), second.copy()],
        samples=200,
        seed=7,
    )
    assert result['difference_percentage_points'] == 0.0
    assert result['ci90_percentage_points'] == [0.0, 0.0]
    assert result['ci95_percentage_points'] == [0.0, 0.0]


def test_holm_adjustment_is_monotone_across_comparison_family():
    adjusted = holm_adjust({'first': 0.01, 'second': 0.04, 'third': 0.2})
    assert np.isclose(adjusted['first'], 0.03)
    assert np.isclose(adjusted['second'], 0.08)
    assert np.isclose(adjusted['third'], 0.2)
