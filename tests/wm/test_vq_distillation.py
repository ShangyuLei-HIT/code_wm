from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from stable_worldmodel.planning.objective import GoalMSE
from stable_worldmodel.wm.vq_lewm.affine import (
    RigidLatentTransform,
    initialize_adapter_from_transform,
)
from stable_worldmodel.wm.vq_lewm.distillation import (
    JointDistillationLeWM,
    cosine_phase_lr,
    nearest_code_indices,
    phase_for_epoch,
    sequence_teacher_forcing,
    sha256_file,
    sparse_topk_kl,
    squared_distances,
    teacher_forcing_alpha,
)


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


def make_model(dim=4, codes=8):
    return JointDistillationLeWM(
        student_encoder=DummyEncoder(dim),
        projector=nn.Linear(dim, dim),
        adapter=nn.Linear(dim, dim),
        action_encoder=nn.Linear(2, dim),
        predictor=DummyPredictor(dim),
        pred_proj=nn.Linear(dim, dim),
        embedding_dim=dim,
        num_embeddings=codes,
        codebook_weights=torch.randn(codes, dim),
    )


def prediction_loss_from_mixed(student, teacher, mask):
    mixed, _ = sequence_teacher_forcing(student, teacher, 0.5, mask=mask)
    prediction = mixed[:, :-1].square()
    target = mixed[:, 1:]
    return F.mse_loss(prediction, target)


def test_training_model_has_no_teacher_network():
    model = make_model()
    module_names = [name for name, _ in model.named_modules()]
    assert not any('teacher' in name for name in module_names)
    assert not any(parameter is model.codebook for parameter in model.parameters())


def test_alpha_zero_prediction_gradient_is_exactly_zero():
    student = torch.randn(5, 4, 3, requires_grad=True)
    teacher = torch.randn_like(student)
    mask = torch.zeros(5, 1, 1, dtype=torch.bool)
    prediction_loss_from_mixed(student, teacher, mask).backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) == 0


def test_alpha_one_prediction_gradient_is_nonzero():
    student = torch.randn(5, 4, 3, requires_grad=True)
    teacher = torch.randn_like(student)
    mask = torch.ones(5, 1, 1, dtype=torch.bool)
    prediction_loss_from_mixed(student, teacher, mask).backward()
    assert torch.count_nonzero(student.grad) > 0


def test_sequence_mask_is_shared_by_history_and_target():
    student = torch.ones(16, 4, 3)
    teacher = torch.zeros_like(student)
    mixed, mask = sequence_teacher_forcing(student, teacher, 0.5)
    assert mask.shape == (16, 1, 1)
    assert torch.equal(mixed == 1, mask.expand_as(mixed))


def test_actual_student_fraction_tracks_alpha():
    student = torch.ones(20000, 4, 2)
    teacher = torch.zeros_like(student)
    _, mask = sequence_teacher_forcing(student, teacher, 0.37)
    assert abs(float(mask.float().mean()) - 0.37) < 0.02


def test_student_prediction_target_is_not_detached():
    student = torch.randn(4, 4, 3, requires_grad=True)
    teacher = torch.randn_like(student)
    mixed, _ = sequence_teacher_forcing(
        student,
        teacher,
        1.0,
        mask=torch.ones(4, 1, 1, dtype=torch.bool),
    )
    prediction = mixed[:, :-1] * 0.0
    target = mixed[:, 1:]
    F.mse_loss(prediction, target).backward()
    assert torch.count_nonzero(student.grad[:, 1:]) > 0


def test_sparse_top32_kl_matches_dense_student_normalizer():
    torch.manual_seed(7)
    student = torch.randn(2, 3, 5)
    codebook = torch.randn(41, 5)
    teacher_logits = torch.randn(2, 3, 41)
    values, indices = teacher_logits.topk(32, dim=-1)
    probs = values.softmax(dim=-1)
    actual = sparse_topk_kl(
        student,
        codebook,
        indices,
        probs,
        temperature=1.3,
        codebook_chunk_size=7,
    )
    dense_logits = -squared_distances(
        student.reshape(-1, 5), codebook
    ) / 1.3
    selected_log_probs = dense_logits.log_softmax(-1).gather(
        -1, indices.reshape(-1, 32)
    )
    flat_probs = probs.reshape(-1, 32)
    expected = 1.3**2 * (
        flat_probs * (flat_probs.log() - selected_log_probs)
    ).sum(-1).mean()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_nearest_chunking_is_exact():
    latent = torch.randn(3, 2, 4)
    codebook = torch.randn(29, 4)
    actual = nearest_code_indices(
        latent, codebook, k=5, codebook_chunk_size=6
    )
    expected = squared_distances(
        latent.reshape(-1, 4), codebook
    ).topk(5, largest=False).indices.view(3, 2, 5)
    assert torch.equal(actual, expected)


def test_phase_switch_lrs_drop_to_new_bases():
    assert cosine_phase_lr(0, 100, 1e-4, 1e-5, 0.0) == pytest.approx(1e-4)
    assert cosine_phase_lr(99, 100, 1e-4, 1e-5, 0.0) == pytest.approx(1e-5)
    assert cosine_phase_lr(0, 20, 1e-5, 1e-6, 0.0) == pytest.approx(1e-5)
    assert phase_for_epoch(0, (4, 10, 2)) == (1, 0)
    assert phase_for_epoch(4, (4, 10, 2)) == (2, 0)
    assert phase_for_epoch(14, (4, 10, 2)) == (3, 0)


def test_resume_step_recovers_phase_and_alpha():
    steps = 11
    global_step = 4 * steps + 37
    alpha = teacher_forcing_alpha(global_step, steps, (4, 10, 2))
    assert 0 < alpha < 1
    assert phase_for_epoch(7, (4, 10, 2))[0] == 2
    assert teacher_forcing_alpha(global_step, steps, (4, 10, 2)) == alpha


def test_codebook_checkpoint_is_immutable(tmp_path):
    path = tmp_path / 'weights.pt'
    torch.save({'teacher.weight': torch.randn(8, 4)}, path)
    before = sha256_file(path)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model.encode_student(torch.randn(2, 4, 3, 8, 8)).square().mean()
    loss.backward()
    optimizer.step()
    assert sha256_file(path) == before
    assert model.codebook.grad is None


def test_final_export_has_no_teacher_codebook_or_cache_dependency():
    export = make_model().deployment_state_dict()
    assert set(export) == {
        'student_encoder',
        'projector',
        'adapter',
        'action_encoder',
        'predictor',
        'pred_proj',
    }
    assert not any(
        word in key
        for key in export
        for word in ('teacher', 'codebook', 'cache')
    )


def test_smoke_schedule_visits_all_alpha_regimes():
    steps = 2
    assert teacher_forcing_alpha(0, steps, (4, 10, 2)) == 0
    assert 0 < teacher_forcing_alpha(13, steps, (4, 10, 2)) < 1
    assert teacher_forcing_alpha(28, steps, (4, 10, 2)) == 1


def test_rigid_transform_preserves_distances_and_has_inverse():
    torch.manual_seed(19)
    raw = torch.randn(8, 5)
    rotation, _ = torch.linalg.qr(torch.randn(5, 5))
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1].neg_()
    transform = RigidLatentTransform(
        rotation,
        center=torch.randn(5),
        translation=torch.randn(5),
    )
    transformed = transform(raw)
    torch.testing.assert_close(
        torch.cdist(transformed, transformed),
        torch.cdist(raw, raw),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        transform.inverse(transformed), raw, rtol=1e-5, atol=1e-5
    )


def test_goal_mse_is_invariant_to_rigid_transform():
    torch.manual_seed(29)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4))
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1].neg_()
    transform = RigidLatentTransform(
        rotation,
        center=torch.randn(4),
        translation=torch.randn(4),
    )
    predicted = torch.randn(2, 3, 5, 4)
    goal = torch.randn(2, 2, 4)
    objective = GoalMSE()
    original = objective(
        {'predicted_emb': predicted, 'goal_emb': goal}
    )
    transformed = objective(
        {
            'predicted_emb': transform(predicted),
            'goal_emb': transform(goal),
        }
    )
    torch.testing.assert_close(
        transformed, original, rtol=1e-5, atol=1e-5
    )


def test_transform_initialized_adapter_matches_transform():
    torch.manual_seed(23)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4))
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1].neg_()
    transform = RigidLatentTransform(
        rotation,
        center=torch.randn(4),
        translation=torch.randn(4),
    )
    adapter = nn.Linear(4, 4)
    initialize_adapter_from_transform(adapter, transform)
    latent = torch.randn(7, 4)
    torch.testing.assert_close(
        adapter(latent), transform(latent), rtol=1e-5, atol=1e-5
    )
