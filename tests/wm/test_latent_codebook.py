import torch

from stable_worldmodel.wm.latent_codebook import TeacherStudentCodebook


def test_codebook_loss_only_updates_student():
    codebook = TeacherStudentCodebook(
        num_embeddings=2,
        embedding_dim=2,
        teacher_momentum=0.9,
    )
    codebook.initialize(torch.tensor([[0.0, 0.0], [2.0, 2.0]]))
    inputs = torch.tensor([[0.5, 0.0], [1.5, 2.0]], requires_grad=True)

    output = codebook(inputs)
    output['codebook_loss'].backward()

    assert inputs.grad is None
    assert codebook.student.weight.grad is not None
    assert codebook.teacher.weight.grad is None


def test_teacher_provides_assignments_for_student_loss():
    codebook = TeacherStudentCodebook(2, 1)
    codebook.initialize(torch.tensor([[0.0], [10.0]]))
    with torch.no_grad():
        codebook.student.weight.copy_(torch.tensor([[10.0], [0.0]]))

    output = codebook(torch.tensor([[0.1]]))

    assert output['indices'].item() == 0
    torch.testing.assert_close(
        output['codebook_loss'], torch.tensor(9.9**2)
    )


def test_teacher_momentum_tracks_student_after_optimizer_step():
    codebook = TeacherStudentCodebook(1, 2, teacher_momentum=0.75)
    codebook.initialize(torch.tensor([[0.0, 4.0]]))
    with torch.no_grad():
        codebook.student.weight.copy_(torch.tensor([[4.0, 0.0]]))

    codebook.update_teacher()

    torch.testing.assert_close(
        codebook.teacher.weight, torch.tensor([[1.0, 3.0]])
    )
    assert codebook.teacher_updates.item() == 1


def test_lookup_uses_teacher_by_default():
    codebook = TeacherStudentCodebook(2, 1)
    codebook.initialize(torch.tensor([[1.0], [2.0]]))
    with torch.no_grad():
        codebook.student.weight.add_(10.0)

    indices = torch.tensor([0, 1])

    torch.testing.assert_close(
        codebook.lookup(indices), torch.tensor([[1.0], [2.0]])
    )
    torch.testing.assert_close(
        codebook.lookup(indices, use_teacher=False),
        torch.tensor([[11.0], [12.0]]),
    )
