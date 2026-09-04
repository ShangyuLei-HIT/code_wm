"""Fully-discrete (M5-style) single-task loss switches and experiment configs."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from scripts.train.vq_lewm_joint_distillation import JointObjective
from stable_worldmodel.wm.vq_lewm.distillation import (
    JointDistillationLeWM,
    sparse_topk_kl,
)

DIM = 4
CODES = 16
TOPK = 8
HISTORY = 3
STEPS = 4


class DummyEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(3, dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.float().mean(dim=(-2, -1))
        return SimpleNamespace(
            last_hidden_state=self.projection(pooled)[:, None]
        )


class DummyPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, latent, action):
        return self.linear(latent + action)


def make_model():
    torch.manual_seed(0)
    return JointDistillationLeWM(
        student_encoder=DummyEncoder(DIM),
        projector=nn.Linear(DIM, DIM),
        adapter=nn.Linear(DIM, DIM),
        action_encoder=nn.Linear(2, DIM),
        predictor=DummyPredictor(DIM),
        pred_proj=nn.Linear(DIM, DIM),
        embedding_dim=DIM,
        codebook_weights=torch.randn(CODES, DIM),
    )


def make_cfg(**loss_overrides):
    loss = {
        'prediction_weight': 1.0,
        'latent_weight': 1.0,
        'soft_kl_weight': 0.1,
        **loss_overrides,
    }
    return OmegaConf.create(
        {
            'wm': {'history_size': HISTORY},
            'codebook': {
                'temperature': 1.0,
                'distance_chunk_size': 2048,
            },
            'loss': loss,
        }
    )


def make_batch():
    torch.manual_seed(1)
    size = 5
    hard_tokens = torch.randint(0, CODES, (size, STEPS))
    selection = torch.randint(0, CODES, (size, STEPS, TOPK))
    probs = torch.rand(size, STEPS, TOPK)
    probs = probs / probs.sum(-1, keepdim=True)
    return {
        'pixels': torch.randn(size, STEPS, 3, 6, 6),
        'action': torch.randn(size, STEPS, 2),
        'teacher_latent': torch.randn(size, STEPS, DIM),
        'hard_tokens': hard_tokens,
        'topk_indices': selection,
        'topk_probs': probs,
    }


def make_objective(**loss_overrides):
    return JointObjective(make_model(), make_cfg(**loss_overrides))


def test_default_objective_reproduces_original_single_task_recipe():
    objective = make_objective()
    assert objective.latent_target == 'continuous'
    assert objective.prediction_source == 'codebook'
    assert objective.soft_kl_weight == pytest.approx(0.1)
    batch = make_batch()
    model = objective.model
    with torch.no_grad():
        student = model.encode_student(batch['pixels'])
        teacher_code = model.lookup_teacher_codes(batch['hard_tokens'])
        for alpha, expected_teacher in ((0.0, teacher_code), (1.0, student)):
            output = objective(batch, alpha)
            assert F.mse_loss(
                student.float(), batch['teacher_latent'].float()
            ) == pytest.approx(float(output['latent_loss']))
            prediction = model.predict(
                expected_teacher[:, :HISTORY], batch['action'][:, :HISTORY]
            )
            expected = F.mse_loss(
                prediction.float(), expected_teacher[:, 1 : HISTORY + 1].float()
            )
            assert expected == pytest.approx(float(output['prediction_loss']))
        expected_kl = sparse_topk_kl(
            student,
            model.codebook,
            batch['topk_indices'],
            batch['topk_probs'],
            temperature=1.0,
            codebook_chunk_size=2048,
        )
        assert float(expected_kl) == pytest.approx(
            float(objective(batch, 0.5)['soft_kl'])
        )


def test_fully_discrete_objective_targets_codebook_vectors():
    objective = make_objective(
        latent_target='codebook', prediction_source='codebook'
    )
    batch = make_batch()
    model = objective.model
    with torch.no_grad():
        student = model.encode_student(batch['pixels'])
        teacher_code = model.lookup_teacher_codes(batch['hard_tokens'])
        output = objective(batch, 0.0)
        expected = F.mse_loss(student.float(), teacher_code.float())
        assert expected == pytest.approx(float(output['latent_loss']))
        # prediction keeps the (single-task default) codebook source
        prediction = model.predict(
            teacher_code[:, :HISTORY], batch['action'][:, :HISTORY]
        )
        expected_prediction = F.mse_loss(
            prediction.float(), teacher_code[:, 1 : HISTORY + 1].float()
        )
        assert expected_prediction == pytest.approx(
            float(output['prediction_loss'])
        )


def test_continuous_prediction_source_uses_raw_teacher_latent():
    objective = make_objective(
        latent_target='continuous', prediction_source='continuous'
    )
    batch = make_batch()
    model = objective.model
    with torch.no_grad():
        output = objective(batch, 0.0)
        teacher = batch['teacher_latent'].float()
        prediction = model.predict(
            teacher[:, :HISTORY], batch['action'][:, :HISTORY]
        )
        expected = F.mse_loss(
            prediction.float(), teacher[:, 1 : HISTORY + 1].float()
        )
        assert expected == pytest.approx(float(output['prediction_loss']))


def test_zero_soft_kl_weight_skips_the_token_term():
    objective = make_objective(soft_kl_weight=0.0)
    output = objective(make_batch(), 0.5)
    assert float(output['soft_kl']) == 0.0


def test_invalid_switch_values_are_rejected():
    with pytest.raises(ValueError):
        make_objective(latent_target='banana')
    with pytest.raises(ValueError):
        make_objective(prediction_source='banana')


PUSHT_BASE = 'scripts/train/config/vq_lewm_joint_distillation.yaml'
CUBE_BASE = 'scripts/train/config/vq_lewm_joint_distillation_cube.yaml'
PUSHT_NEW = (
    'scripts/train/config/vq_lewm_joint_distillation_pusht_fully_discrete.yaml'
)
CUBE_NEW = (
    'scripts/train/config/vq_lewm_joint_distillation_cube_fully_discrete.yaml'
)
PUSHT_HELDOUT = (
    'scripts/train/config/'
    'vq_lewm_joint_distillation_pusht_fully_discrete_heldout.yaml'
)


def resolved(path):
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg


def test_fully_discrete_configs_keep_single_task_protocol():
    for base_path, new_path, nproc in ((PUSHT_BASE, PUSHT_NEW, 4), (CUBE_BASE, CUBE_NEW, 4)):
        base, new = resolved(base_path), resolved(new_path)
        # identical model, optimizer, phase schedule, precision and seed
        assert OmegaConf.to_container(new.model) == OmegaConf.to_container(
            base.model
        )
        assert OmegaConf.to_container(new.optimizer) == OmegaConf.to_container(
            base.optimizer
        )
        assert OmegaConf.to_container(new.phases) == OmegaConf.to_container(
            base.phases
        )
        assert new.seed == base.seed == 3072
        assert new.trainer.precision == base.trainer.precision
        # identical frozen assets and cache (reuse, never rebuild)
        assert new.paths.teacher_checkpoint == base.paths.teacher_checkpoint
        assert (
            new.paths.codebook_checkpoint == base.paths.codebook_checkpoint
        )
        assert new.paths.cache_dir == base.paths.cache_dir
        assert (
            new.paths.student_init_checkpoint
            == base.paths.student_init_checkpoint
        )
        # same loss weights; only the teacher representation is switched
        assert new.loss.prediction_weight == base.loss.prediction_weight == 1.0
        assert new.loss.latent_weight == base.loss.latent_weight == 1.0
        assert new.loss.soft_kl_weight == base.loss.soft_kl_weight == 0.1
        assert new.loss.latent_target == 'codebook'
        assert new.loss.prediction_source == 'codebook'
        # global batch stays 768 (P1 ran 3x256; codebook-quality/C1 ran 4x192)
        assert int(new.data.train_batch_size_per_gpu) * nproc == 768
        assert int(new.data.cpu_workers_total) == 108
        assert bool(new.gates.enforce) is False
        # data protocol and evaluation protocol unchanged
        assert new.data.dataset == base.data.dataset
        assert new.data.num_steps == base.data.num_steps == 4
        assert new.data.frameskip == base.data.frameskip == 5
        assert new.data.split_seed == base.data.split_seed
        assert list(new.evaluation.stages) == ['phase1', 'phase2', 'final']
        assert int(new.evaluation.num_eval) == 50
        assert int(new.evaluation.seed) == 42


def test_heldout_config_pairs_with_codebook_quality_protocol():
    training, heldout = resolved(PUSHT_NEW), resolved(PUSHT_HELDOUT)
    for key in (
        'teacher_checkpoint',
        'codebook_checkpoint',
        'cache_dir',
        'output_dir',
        'student_init_checkpoint',
    ):
        assert heldout.paths[key] == training.paths[key]
    manifests = [
        str(heldout.evaluation.selection_manifest),
        *[str(path) for path in heldout.evaluation.test_manifests],
    ]
    assert len(manifests) == 5
    for manifest in manifests:
        assert Path(manifest).is_file(), manifest
    assert str(heldout.evaluation.output_root).endswith(
        'task_evaluation_heldout'
    )
    baselines = OmegaConf.to_container(heldout.evaluation.baselines)
    assert baselines['k8192_original'] == 76.0
    assert baselines['prediction_only_no_sigreg'] == 3.5


def test_new_outputs_do_not_touch_existing_runs():
    outputs = [
        resolved(path).paths.output_dir
        for path in (PUSHT_BASE, CUBE_BASE, PUSHT_NEW, CUBE_NEW)
    ]
    assert len(set(outputs)) == 4
    for path in outputs[2:]:
        assert 'fully_discrete' in str(path)
