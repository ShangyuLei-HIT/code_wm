"""Fully-discrete codebook-series matrix, config generation, and cache reuse."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.train.run_fully_discrete_codebook_series import (
    CONDITION_ORDER,
    condition_cache_dir,
    condition_output_dir,
    make_heldout_config,
    make_training_config,
    verify_reusable_assets,
)
from scripts.train.vq_lewm_joint_distillation import load_cache_metadata

MATRIX = 'scripts/train/config/fully_discrete_codebook_series.yaml'


def load_matrix():
    matrix = OmegaConf.load(MATRIX)
    OmegaConf.resolve(matrix)
    return matrix


def test_matrix_covers_both_tasks_and_all_conditions():
    matrix = load_matrix()
    for task in ('pusht', 'cube'):
        assert list(matrix[task]['conditions'].keys()) == list(CONDITION_ORDER)
        for condition in CONDITION_ORDER:
            spec = matrix[task]['conditions'][condition]
            assert int(spec.num_embeddings) in (512, 2048, 8192)
            assert str(spec.transform_mode) == (
                'rigid' if 'rigid' in condition else 'none'
            )
    assert int(matrix.compute.batch_size_per_gpu) * int(
        matrix.compute.nproc_per_node
    ) == 768


def test_reused_pusht_codebooks_and_caches_exist():
    matrix = load_matrix()
    for condition in CONDITION_ORDER:
        spec = matrix.pusht['conditions'][condition]
        assert (Path(spec.codebook_checkpoint) / 'weights.pt').is_file()
        assert (Path(spec.cache_dir) / 'metadata.json').is_file()
        assert Path(spec.base_teacher_cache).is_dir()


def test_verify_reusable_assets_passes_against_existing_caches():
    matrix = load_matrix()

    class Logger:
        def __init__(self):
            self.stages = []

        def stage(self, name):
            self.stages.append(name)

    logger = Logger()
    verify_reusable_assets(matrix, logger)
    assert 'reusable_assets_verified' in logger.stages


def test_generated_pusht_configs_validate_against_existing_caches():
    matrix = load_matrix()
    for condition in CONDITION_ORDER:
        config_path = Path(make_training_config(matrix, 'pusht', condition))
        cfg = OmegaConf.load(config_path)
        metadata, train_indices, val_indices = load_cache_metadata(cfg)
        assert int(metadata['codebook_size']) == int(
            cfg.codebook.num_embeddings
        )
        assert len(train_indices) > 0 and len(val_indices) > 0


def test_generated_configs_keep_fully_discrete_protocol():
    matrix = load_matrix()
    seen_outputs = set()
    for task in ('pusht', 'cube'):
        for condition in CONDITION_ORDER:
            cfg = OmegaConf.load(
                make_training_config(matrix, task, condition)
            )
            OmegaConf.resolve(cfg)
            assert cfg.loss.latent_target == 'codebook'
            assert cfg.loss.prediction_source == 'codebook'
            assert cfg.loss.soft_kl_weight == 0.1
            assert list(cfg.phases.epochs) == [4, 10, 2]
            assert int(cfg.seed) == 3072
            assert bool(cfg.gates.enforce) is False
            assert int(cfg.data.train_batch_size_per_gpu) * 4 == 768
            assert int(cfg.data.cpu_workers_total) == 108
            assert str(cfg.paths.base_teacher_cache).startswith('/')
            expected_transform = 'rigid' in condition
            assert bool(cfg.latent_transform.enabled) is expected_transform
            output = Path(cfg.paths.output_dir)
            assert output not in seen_outputs
            seen_outputs.add(output)
            assert 'fully_discrete' in str(output)
            assert str(cfg.paths.output_dir) != str(cfg.paths.cache_dir)


def test_cube_conditions_build_new_caches_from_base_teacher_cache():
    matrix = load_matrix()
    base = Path(str(matrix.cube.base_teacher_cache)).resolve()
    assert (base / 'metadata.json').is_file()
    for condition in CONDITION_ORDER:
        assert (
            condition_cache_dir(matrix, 'cube', condition)
            == Path(matrix.experiment.root) / 'caches' / f'cube_{condition}'
        )
    rigid = Path(str(matrix.cube.rigid.output)).resolve()
    assert 'rigid' in str(rigid) and str(rigid).startswith('/')


def test_heldout_configs_pair_with_codebook_quality_protocol():
    matrix = load_matrix()
    for condition in CONDITION_ORDER:
        config_path = Path(make_heldout_config(matrix, 'pusht', condition))
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        assert Path(str(cfg.evaluation.selection_manifest)).is_file()
        shards = [
            Path(str(path)) for path in cfg.evaluation.test_manifests
        ]
        assert len(shards) == 4 and all(path.is_file() for path in shards)
        assert str(cfg.evaluation.output_root).endswith(
            'task_evaluation_heldout'
        )
        baselines = OmegaConf.to_container(cfg.evaluation.baselines)
        assert baselines['k512_original'] == 75.5
        assert baselines['k8192_rigid'] == 80.5
        assert baselines['prediction_only_no_sigreg'] == 3.5
        # eval-only variant must never drift from the training paths
        assert cfg.paths.output_dir == str(
            condition_output_dir(matrix, 'pusht', condition)
        )


def test_series_outputs_are_isolated_from_running_k8192_runs():
    matrix = load_matrix()
    running = {
        Path(
            'scripts/train/config/'
            'vq_lewm_joint_distillation_pusht_fully_discrete.yaml'
        ),
        Path(
            'scripts/train/config/'
            'vq_lewm_joint_distillation_cube_fully_discrete.yaml'
        ),
    }
    existing = set()
    for path in running:
        cfg = OmegaConf.load(path)
        OmegaConf.resolve(cfg)
        existing.add(Path(str(cfg.paths.output_dir)))
    for task in ('pusht', 'cube'):
        for condition in CONDITION_ORDER:
            output = condition_output_dir(matrix, task, condition)
            assert output not in existing
            assert output.name == condition
