import math

import torch

from stable_worldmodel.wm.codebook_evaluation import (
    collect_quantization_errors,
    distribution_statistics,
    save_quantization_violin_plot,
    save_training_loss_curve,
)
from stable_worldmodel.wm.latent_codebook import TeacherStudentCodebook


def test_quantization_errors_use_teacher_vector_and_frozen_latent_norm():
    codebook = TeacherStudentCodebook(2, 2)
    codebook.initialize(torch.tensor([[0.0, 0.0], [2.0, 0.0]]))
    latents = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

    errors = collect_quantization_errors(
        codebook, latents, batch_size=1
    )

    expected_absolute = torch.tensor([math.sqrt(17.0), 0.0])
    expected_relative = torch.tensor([math.sqrt(17.0) / 5.0, 0.0])
    torch.testing.assert_close(errors['absolute_l2'], expected_absolute)
    torch.testing.assert_close(errors['relative_l2'], expected_relative)


def test_distribution_statistics_reports_expected_center():
    stats = distribution_statistics(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    assert stats['mean'] == 2.5
    assert stats['median'] == 2.5
    torch.testing.assert_close(
        torch.tensor(stats['std']), torch.tensor(math.sqrt(1.25))
    )


def test_violin_plot_is_written(tmp_path):
    errors = {
        split: {
            'absolute_l2': torch.tensor([1.0, 2.0, 3.0]),
            'relative_l2': torch.tensor([0.1, 0.2, 0.3]),
        }
        for split in ('train', 'validation', 'test')
    }
    output_path = tmp_path / 'errors.png'

    save_quantization_violin_plot(
        errors, output_path, max_points_per_split=3, dpi=72
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
