from pathlib import Path

from omegaconf import OmegaConf


def test_multitask_training_is_teacher_free_and_infers_codebook_size():
    source = Path(
        'scripts/train/multitask_vq_lewm_distillation.py'
    ).read_text()
    assert 'load_pretrained' not in source
    assert 'teacher.encode' not in source
    cfg = OmegaConf.load('scripts/train/config/multitask_vq_lewm.yaml')
    assert 'num_embeddings' not in cfg.model
    assert str(cfg.model.codebook_checkpoint).endswith('pusht_tworoom_fused_uot')


def test_native_m3_has_no_teacher_alignment_or_codebook_dependency():
    source = Path('scripts/train/multitask_lewm_baseline.py').read_text()
    assert 'load_pretrained' not in source
    assert 'alignment_checkpoint' not in source
    assert 'fused_codebook_checkpoint' not in source


def test_balanced_training_zips_one_loader_per_task():
    source = Path(
        'scripts/train/multitask_vq_lewm_distillation.py'
    ).read_text()
    assert 'zip(*(iter(loader) for loader in self.loaders))' in source
    assert 'batch_size_per_task_per_gpu' in source
