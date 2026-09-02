from types import SimpleNamespace

import torch
from torch import nn

from stable_worldmodel.wm.vq_lewm.multitask import (
    MultiTaskJointDistillationLeWM,
)


class DummyEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(3, dim)

    def forward(self, pixels, interpolate_pos_encoding=True):
        pooled = pixels.float().mean((-2, -1))
        return SimpleNamespace(
            last_hidden_state=self.projection(pooled)[:, None]
        )


class DummyPredictor(nn.Module):
    def forward(self, latent, condition):
        return latent + condition


def make_model():
    dim = 4
    return MultiTaskJointDistillationLeWM(
        student_encoder=DummyEncoder(dim),
        projector=nn.Linear(dim, dim),
        adapter=nn.Linear(dim, dim),
        action_encoder=nn.Linear(2, dim),
        predictor=DummyPredictor(),
        pred_proj=nn.Identity(),
        embedding_dim=dim,
        num_tasks=2,
        codebook_weights=torch.randn(13, dim),
    )


def test_multitask_model_infers_fused_codebook_size():
    model = make_model()
    assert model.codebook.shape == (13, 4)
    assert all(
        parameter is not model.codebook for parameter in model.parameters()
    )


def test_task_id_changes_shared_predictor_conditioning():
    model = make_model()
    with torch.no_grad():
        model.task_embedding.weight[0].zero_()
        model.task_embedding.weight[1].fill_(1.0)
    latent = torch.zeros(3, 2, 4)
    action = torch.zeros(3, 2, 2)
    prediction0 = model.predict(latent, action, torch.zeros(3, dtype=torch.long))
    prediction1 = model.predict(latent, action, torch.ones(3, dtype=torch.long))
    torch.testing.assert_close(prediction1 - prediction0, torch.ones_like(prediction0))


def test_deployment_export_contains_task_embedding_but_no_teacher():
    export = make_model().deployment_state_dict()
    assert 'task_embedding' in export
    assert not any('teacher' in name for name in export)
