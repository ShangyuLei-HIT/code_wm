from types import SimpleNamespace

import torch
from torch import nn

from stable_worldmodel.wm.vq_lewm.multitask import (
    MultiTaskContinuousLeWM,
    MultiTaskJointDistillationLeWM,
)


class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(3, dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.float().mean((-2, -1))
        return SimpleNamespace(last_hidden_state=self.linear(pooled)[:, None])


class Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, latent, condition):
        return self.linear(latent + condition)


def components(dim):
    return {
        'encoder': Encoder(dim),
        'projector': nn.Linear(dim, dim),
        'adapter': nn.Linear(dim, dim),
        'action_encoder': nn.Linear(2, dim),
        'predictor': Predictor(dim),
        'pred_proj': nn.Linear(dim, dim),
    }


def test_m3_has_no_teacher_alignment_or_codebook():
    model = MultiTaskContinuousLeWM(
        **components(4), embedding_dim=4, num_tasks=2
    )
    assert not hasattr(model, 'codebook')
    assert not any('teacher' in name or 'align' in name for name, _ in model.named_modules())


def test_m3_and_discrete_student_have_equal_trainable_parameter_count():
    torch.manual_seed(4)
    m3_parts = components(4)
    m3 = MultiTaskContinuousLeWM(
        **m3_parts, embedding_dim=4, num_tasks=2
    )
    torch.manual_seed(4)
    discrete_parts = components(4)
    discrete = MultiTaskJointDistillationLeWM(
        student_encoder=discrete_parts.pop('encoder'),
        codebook_weights=torch.randn(11, 4),
        embedding_dim=4,
        num_tasks=2,
        **discrete_parts,
    )
    m3_count = sum(parameter.numel() for parameter in m3.parameters())
    discrete_count = sum(parameter.numel() for parameter in discrete.parameters())
    assert m3_count == discrete_count
