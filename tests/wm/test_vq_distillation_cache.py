from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.train.vq_lewm_joint_distillation import load_cache_metadata


def test_cache_metadata_mismatch_refuses_training(tmp_path):
    (tmp_path / 'metadata.json').write_text(
        '{"format_version": 1, "metadata_sha256": "wrong"}\n'
    )
    cfg = OmegaConf.create({'paths': {'cache_dir': str(tmp_path)}})
    with pytest.raises(RuntimeError, match='self-hash mismatch'):
        load_cache_metadata(cfg)


def test_training_entrypoint_never_loads_or_runs_teacher():
    source = Path(
        'scripts/train/vq_lewm_joint_distillation.py'
    ).read_text()
    assert 'load_pretrained' not in source
    assert 'teacher.encode' not in source


def test_assignment_cache_reads_existing_teacher_latents():
    source = Path(
        'scripts/train/cache_codebook_assignments.py'
    ).read_text()
    assert 'load_pretrained' not in source
    assert 'teacher.encode' not in source
    assert 'base_teacher_cache' in source
