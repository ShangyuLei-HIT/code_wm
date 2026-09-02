"""Teacher/student codebook for discretizing frozen continuous latents."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TeacherStudentCodebook(nn.Module):
    """Quantize frozen latents with a gradient student and EMA teacher.

    Nearest-neighbour assignments are computed against the teacher codebook.
    The selected student entries minimize squared L2 to stop-gradient z_e.
    After the optimizer updates the student, update_teacher must be called
    once so that the teacher momentum-tracks the student.

    Args:
        num_embeddings: Number of discrete codes.
        embedding_dim: Continuous latent dimension.
        teacher_momentum: EMA coefficient applied to the old teacher.
        source_checkpoint: Optional provenance for the frozen encoder.
        latent_key: Optional provenance for the encoded tensor.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        teacher_momentum: float = 0.99,
        source_checkpoint: str | None = None,
        latent_key: str = 'emb',
    ) -> None:
        super().__init__()
        if num_embeddings < 1:
            raise ValueError('num_embeddings must be positive')
        if embedding_dim < 1:
            raise ValueError('embedding_dim must be positive')
        if not 0.0 <= teacher_momentum < 1.0:
            raise ValueError('teacher_momentum must be in [0, 1)')

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.teacher_momentum = teacher_momentum
        self.source_checkpoint = source_checkpoint
        self.latent_key = latent_key

        self.student = nn.Embedding(num_embeddings, embedding_dim)
        self.teacher = nn.Embedding(num_embeddings, embedding_dim)
        self.teacher.weight.requires_grad_(False)
        self.register_buffer(
            'teacher_updates', torch.zeros((), dtype=torch.long)
        )

        bound = embedding_dim**-0.5
        nn.init.uniform_(self.student.weight, -bound, bound)
        self.teacher.weight.data.copy_(self.student.weight.data)

    @staticmethod
    def _squared_distances(
        inputs: torch.Tensor, codebook: torch.Tensor
    ) -> torch.Tensor:
        inputs = inputs.float()
        codebook = codebook.float()
        return (
            inputs.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1)
            - 2.0 * inputs @ codebook.t()
        )

    @torch.no_grad()
    def initialize(self, centers: torch.Tensor) -> None:
        """Initialize both codebooks from explicit continuous centers."""
        expected = (self.num_embeddings, self.embedding_dim)
        if tuple(centers.shape) != expected:
            raise ValueError(
                f'expected centers with shape {expected}, '
                f'got {tuple(centers.shape)}'
            )
        centers = centers.to(
            device=self.student.weight.device,
            dtype=self.student.weight.dtype,
        )
        self.student.weight.copy_(centers)
        self.teacher.weight.copy_(centers)
        self.teacher_updates.zero_()

    @torch.no_grad()
    def assign(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return teacher nearest-neighbour indices for frozen latents."""
        if inputs.size(-1) != self.embedding_dim:
            raise ValueError(
                f'expected latent dim {self.embedding_dim}, '
                f'got {inputs.size(-1)}'
            )
        flat = inputs.detach().reshape(-1, self.embedding_dim)
        distances = self._squared_distances(flat, self.teacher.weight)
        return distances.argmin(dim=1).view(inputs.shape[:-1])

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Assign with the teacher and compute the student codebook loss."""
        if inputs.size(-1) != self.embedding_dim:
            raise ValueError(
                f'expected latent dim {self.embedding_dim}, '
                f'got {inputs.size(-1)}'
            )

        frozen = inputs.detach().reshape(-1, self.embedding_dim).float()
        with torch.no_grad():
            distances = self._squared_distances(
                frozen, self.teacher.weight
            )
            flat_indices = distances.argmin(dim=1)

        student_quantized = F.embedding(
            flat_indices, self.student.weight
        ).float()
        teacher_quantized = F.embedding(
            flat_indices, self.teacher.weight
        ).float()

        per_vector_student_l2 = (student_quantized - frozen).square().sum(-1)
        per_vector_teacher_l2 = (teacher_quantized - frozen).square().sum(-1)
        codebook_loss = per_vector_student_l2.mean()
        counts = torch.bincount(
            flat_indices, minlength=self.num_embeddings
        )
        probabilities = counts.float() / counts.sum().clamp_min(1)
        perplexity = torch.exp(
            -(probabilities * (probabilities + 1e-12).log()).sum()
        )

        output_shape = (*inputs.shape[:-1], self.embedding_dim)
        return {
            'indices': flat_indices.view(inputs.shape[:-1]),
            'quantized': teacher_quantized.view(output_shape),
            'student_quantized': student_quantized.view(output_shape),
            'codebook_loss': codebook_loss,
            'codebook_mse': codebook_loss / self.embedding_dim,
            'teacher_l2': per_vector_teacher_l2.mean(),
            'counts': counts,
            'perplexity': perplexity,
        }

    @torch.no_grad()
    def update_teacher(self) -> None:
        """EMA-update the teacher after one student optimizer step."""
        self.teacher.weight.mul_(self.teacher_momentum).add_(
            self.student.weight, alpha=1.0 - self.teacher_momentum
        )
        self.teacher_updates.add_(1)

    def lookup(
        self, indices: torch.Tensor, *, use_teacher: bool = True
    ) -> torch.Tensor:
        """Map discrete indices to teacher (default) or student vectors."""
        table = self.teacher if use_teacher else self.student
        return table(indices)


__all__ = ['TeacherStudentCodebook']
