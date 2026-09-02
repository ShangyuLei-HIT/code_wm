from pathlib import Path

from omegaconf import OmegaConf

from scripts.train.evaluate_multitask_distillation import evaluation_command


def test_tworoom_evaluation_uses_add_or_override_for_optional_keys():
    cfg = OmegaConf.create(
        {
            'evaluation': {'num_eval': 2, 'seed': 7, 'video': False},
        }
    )
    task = OmegaConf.create(
        {
            'name': 'tworoom',
            'evaluation_dataset': 'tworoom.h5',
        }
    )
    export = {'checkpoint': '/tmp/model.pt', 'directory': '/tmp/evaluation'}
    command = evaluation_command(Path('/project'), cfg, task, export)
    assert '--config-name' in command
    assert 'tworoom' in command
    assert '++eval.video=false' in command
    assert '++output.directory=/tmp/evaluation' in command
    assert '++output.append=false' in command
