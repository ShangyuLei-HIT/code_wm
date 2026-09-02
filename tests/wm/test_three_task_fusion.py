from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from scripts.train.build_multitask_fused_codebook import task_order
from scripts.train.multitask_vq_lewm_distillation import BalancedLoader
from stable_worldmodel.wm.vq_lewm.alignment import load_alignment_bundle
from stable_worldmodel.wm.vq_lewm.multitask import PaddedActionEncoder


class RecordingEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, 3, bias=False)
        self.last_input = None

    def forward(self, value):
        self.last_input = value
        return self.projection(value)


def test_padded_action_encoder_accepts_pusht_and_cube_widths():
    base = RecordingEncoder(25)
    encoder = PaddedActionEncoder(base, input_dim=25)
    short = torch.randn(2, 4, 10)
    long = torch.randn(2, 4, 25)
    assert encoder(short).shape == (2, 4, 3)
    torch.testing.assert_close(
        base.last_input[..., 10:], torch.zeros(2, 4, 15)
    )
    assert encoder(long).shape == (2, 4, 3)


def test_padded_action_encoder_rejects_too_wide_actions():
    encoder = PaddedActionEncoder(RecordingEncoder(25), input_dim=25)
    try:
        encoder(torch.randn(1, 2, 26))
    except ValueError as error:
        assert 'exceeds shared width' in str(error)
    else:
        raise AssertionError('expected an over-wide action to fail')


def test_balanced_loader_pads_three_task_actions_before_concat():
    loaders = []
    for task_id, width in enumerate((10, 10, 25)):
        loaders.append(
            [
                {
                    'action': torch.ones(2, 4, width),
                    'task_id': torch.full((2,), task_id),
                }
            ]
        )
    batch = next(iter(BalancedLoader(loaders)))
    assert batch['action'].shape == (6, 4, 25)
    assert batch['task_id'].tolist() == [0, 0, 1, 1, 2, 2]
    torch.testing.assert_close(
        batch['action'][:4, :, 10:], torch.zeros(4, 4, 15)
    )


def test_multi_source_alignment_checkpoint_roundtrip(tmp_path):
    checkpoint = tmp_path / 'alignment.pt'
    torch.save(
        {
            'format_version': 2,
            'reference_task': 'pusht',
            'alignments': {
                'tworoom': {
                    'rotation': torch.eye(3),
                    'scale': torch.tensor(2.0),
                    'bias': torch.ones(3),
                },
                'cube': {
                    'rotation': torch.eye(3),
                    'scale': torch.tensor(0.5),
                    'bias': -torch.ones(3),
                },
            },
        },
        checkpoint,
    )
    alignments = load_alignment_bundle(checkpoint, expected_dim=3)
    assert set(alignments) == {'tworoom', 'cube'}
    torch.testing.assert_close(
        alignments['tworoom'](torch.zeros(2, 3)), torch.ones(2, 3)
    )
    torch.testing.assert_close(
        alignments['cube'](torch.zeros(2, 3)), -torch.ones(2, 3)
    )


def test_three_task_configs_use_new_paths_and_cover_all_tasks():
    root = Path('scripts/train/config')
    m2 = OmegaConf.load(root / 'multitask_vq_lewm_three_tasks.yaml')
    m0 = OmegaConf.load(
        root / 'multitask_vq_lewm_three_tasks_m0_unaligned.yaml'
    )
    m3 = OmegaConf.load(root / 'multitask_lewm_three_tasks_baseline.yaml')
    assert [task.name for task in m2.tasks] == ['pusht', 'tworoom', 'cube']
    assert task_order(m2) == [0, 1, 2]
    assert m2.model.num_tasks == m0.model.num_tasks == m3.model.num_tasks == 3
    assert m2.codebook.embedding_dim == 192
    assert m2.model.action_encoder.input_dim == 25
    for cfg in (m2, m0, m3):
        assert 'pusht_tworoom_cube' in str(cfg.paths.output_dir)
        assert 'pusht_tworoom_uot_seed3072' not in str(cfg.paths.output_dir)
