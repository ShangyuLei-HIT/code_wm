from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from scripts.train.build_multitask_fused_codebook import task_order
from scripts.train.multitask_vq_lewm_distillation import BalancedLoader
from stable_worldmodel.wm.vq_lewm.alignment import load_alignment_bundle
from stable_worldmodel.wm.vq_lewm.multitask import PaddedActionEncoder

SIX_TASKS = ['pusht', 'tworoom', 'cube', 'scene', 'reacher', 'humanoidmaze']
SIX_WIDTHS = (10, 10, 25, 25, 10, 105)
SHARED_WIDTH = 105


class RecordingEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, 3, bias=False)
        self.last_input = None

    def forward(self, value):
        self.last_input = value
        return self.projection(value)


def test_padded_action_encoder_accepts_six_task_widths():
    base = RecordingEncoder(SHARED_WIDTH)
    encoder = PaddedActionEncoder(base, input_dim=SHARED_WIDTH)
    for width in SIX_WIDTHS:
        sample = torch.ones(2, 4, width)
        assert encoder(sample).shape == (2, 4, 3)
        # Each block is zero-padded up to the shared 105-wide input.
        assert base.last_input.shape == (2, 4, SHARED_WIDTH)
        assert float(base.last_input[..., width:].abs().sum()) == 0.0
        assert float(base.last_input[..., :width].abs().sum()) > 0.0


def test_padded_action_encoder_rejects_over_six_task_width():
    encoder = PaddedActionEncoder(
        RecordingEncoder(SHARED_WIDTH), input_dim=SHARED_WIDTH
    )
    try:
        encoder(torch.randn(1, 2, SHARED_WIDTH + 1))
    except ValueError as error:
        assert 'exceeds shared width' in str(error)
    else:
        raise AssertionError('expected an over-wide action to fail')


def test_balanced_loader_pads_six_task_actions_before_concat():
    loaders = []
    for task_id, width in enumerate(SIX_WIDTHS):
        loaders.append(
            [
                {
                    'action': torch.ones(2, 4, width),
                    'task_id': torch.full((2,), task_id),
                }
            ]
        )
    batch = next(iter(BalancedLoader(loaders)))
    assert batch['action'].shape == (12, 4, SHARED_WIDTH)
    assert batch['task_id'].tolist() == [task for task in range(6) for _ in (0, 1)]
    # Every task's block is zero-padded from its own width up to 105.
    for task_id, width in enumerate(SIX_WIDTHS):
        rows = batch['action'][2 * task_id : 2 * task_id + 2]
        assert float(rows[..., :width].abs().sum()) > 0.0
        assert float(rows[..., width:].abs().sum()) == 0.0


def test_multi_source_alignment_bundle_covers_five_sources(tmp_path):
    checkpoint = tmp_path / 'alignment.pt'
    offsets = {
        'tworoom': torch.ones(3),
        'cube': -torch.ones(3),
        'scene': 2 * torch.ones(3),
        'reacher': -2 * torch.ones(3),
        'humanoidmaze': 0.5 * torch.ones(3),
    }
    torch.save(
        {
            'format_version': 2,
            'reference_task': 'pusht',
            'alignments': {
                name: {
                    'rotation': torch.eye(3),
                    'scale': torch.tensor(1.0),
                    'bias': bias,
                }
                for name, bias in offsets.items()
            },
        },
        checkpoint,
    )
    alignments = load_alignment_bundle(checkpoint, expected_dim=3)
    assert set(alignments) == set(offsets)
    for name, bias in offsets.items():
        torch.testing.assert_close(
            alignments[name](torch.zeros(2, 3)), bias.expand(2, 3)
        )


def test_six_task_configs_cover_all_tasks_and_use_new_paths():
    root = Path('scripts/train/config')
    configs = {
        'm2': 'multitask_vq_lewm_six_tasks.yaml',
        'm0': 'multitask_vq_lewm_six_tasks_m0_unaligned.yaml',
        'm4': 'multitask_vq_lewm_six_tasks_m4_continuous.yaml',
        'm5': 'multitask_vq_lewm_six_tasks_m5_codebook.yaml',
    }
    m2 = OmegaConf.load(root / configs['m2'])
    m0 = OmegaConf.load(root / configs['m0'])
    m3 = OmegaConf.load(root / 'multitask_lewm_six_tasks_baseline.yaml')
    assert [task.name for task in m2.tasks] == SIX_TASKS
    assert task_order(m2) == [0, 1, 2, 3, 4, 5]
    assert list(m2.alignment.source_task_ids) == [1, 2, 3, 4, 5]
    assert m2.model.num_tasks == m0.model.num_tasks == m3.model.num_tasks == 6
    assert m2.codebook.embedding_dim == 192
    assert m2.model.action_encoder.input_dim == SHARED_WIDTH
    assert m2.model.action_encoder.encoder.input_dim == SHARED_WIDTH
    assert len(m2.evaluation.devices) >= 6
    token = 'pusht_tworoom_cube_scene_reacher_humanoidmaze'
    for cfg in (m2, m0, m3):
        assert token in str(cfg.paths.output_dir)
        # Never write into the two- or three-task experiment families.
        assert 'pusht_tworoom_cube_uot_seed3072' not in str(cfg.paths.output_dir)
        assert 'pusht_tworoom_uot_seed3072' not in str(cfg.paths.output_dir)
    # M0 keeps the M2 task contract but only flips fusion/alignment.
    assert [task.name for task in m0.tasks] == SIX_TASKS
    assert m0.fusion.method == 'concat'
    assert m0.alignment.enabled is False
    assert m2.fusion.method == 'uot'
    # M4/M5 only vary the loss teacher-representation switches.
    m4 = OmegaConf.load(root / configs['m4'])
    m5 = OmegaConf.load(root / configs['m5'])
    assert m4.loss.token_weight == 0.0
    assert m4.loss.latent_target == 'continuous'
    assert m4.loss.prediction_source == 'continuous'
    assert m5.loss.token_weight == m2.loss.token_weight
    assert m5.loss.latent_target == 'codebook'
    assert m5.loss.prediction_source == 'codebook'


def test_new_task_c1_configs_point_at_isolated_outputs():
    root = Path('scripts/train/config')
    expected = {
        'scene': 'lewm_scene_scratch_seed3072_compat',
        'reacher': 'official_lewm_reacher_compat',
        'humanoidmaze': 'lewm_humanoidmaze_scratch_seed3072_compat',
    }
    for task, teacher in expected.items():
        cfg = OmegaConf.load(
            root / f'vq_lewm_joint_distillation_{task}.yaml'
        )
        assert cfg.paths.teacher_checkpoint.endswith(teacher)
        assert cfg.paths.student_init_checkpoint.endswith(teacher)
        assert cfg.paths.codebook_checkpoint.endswith(
            f'{teacher}_codebook_k8192'
        )
        assert cfg.paths.output_dir.endswith(
            f'joint_distillation/lewm_{task}_k8192_seed3072'
        )
        assert cfg.evaluation.plan_config == task
        assert cfg.gates.enforce is False
