from types import SimpleNamespace

import torch
from torch import nn

from scripts.train.evaluate_joint_distillation import (
    add_best_stage,
    flatten_deployment_modules,
    modules_from_phase_checkpoint,
    parse_task_result,
    parse_task_result_details,
)
from stable_worldmodel.wm.vq_lewm.quantized import CodebookQuantizedLeWM


class OffsetPredictor(nn.Module):
    def forward(self, latent, action):
        return latent + 0.6


def test_official_wrapper_quantizes_rollout_predictions_when_enabled():
    codebook = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    model = CodebookQuantizedLeWM(
        encoder=nn.Linear(2, 2),
        predictor=OffsetPredictor(),
        action_encoder=nn.Identity(),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
        codebook_weights=codebook,
        embedding_dim=2,
        quantize_rollout_predictions=True,
    )
    latent = torch.zeros(1, 2, 2)
    actual = model.predict(latent, torch.zeros_like(latent))
    torch.testing.assert_close(actual, torch.ones_like(actual))
    assert 'codebook' not in model.state_dict()


def test_official_wrapper_can_quantize_encoder_latents_only():
    codebook = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    model = CodebookQuantizedLeWM(
        encoder=nn.Linear(2, 2),
        predictor=OffsetPredictor(),
        action_encoder=nn.Identity(),
        codebook_weights=codebook,
        quantize_rollout_predictions=False,
    )
    latent = torch.zeros(1, 2, 2)
    actual = model.predict(latent, torch.zeros_like(latent))
    torch.testing.assert_close(actual, torch.full_like(actual, 0.6))
    quantized = model.quantize_latent(torch.tensor([[[0.8, 0.9]]]))
    torch.testing.assert_close(quantized, torch.ones_like(quantized))


def test_phase_checkpoint_is_flattened_for_deployment():
    names = (
        'student_encoder',
        'projector',
        'adapter',
        'action_encoder',
        'predictor',
        'pred_proj',
    )
    payload = {
        'model': {
            f'model.{name}.weight': torch.tensor([index])
            for index, name in enumerate(names)
        }
    }
    modules = modules_from_phase_checkpoint(payload)
    flat = flatten_deployment_modules(modules)
    assert set(flat) == {
        'encoder.weight',
        'projector.weight',
        'adapter.weight',
        'action_encoder.weight',
        'predictor.weight',
        'pred_proj.weight',
    }
    assert flat['encoder.weight'].item() == 0


def test_task_result_parser_uses_latest_appended_run(tmp_path):
    result = tmp_path / 'results.txt'
    result.write_text(
        "metrics: {'success_rate': 50.0}\nevaluation_time: 12.0 seconds\n"
        "metrics: {'success_rate': 78.0}\nevaluation_time: 10.5 seconds\n"
    )
    rate, seconds = parse_task_result(result)
    assert rate == 78.0
    assert seconds == 10.5


def test_task_result_details_prefers_structured_sidecar(tmp_path):
    result = tmp_path / 'results.txt'
    result.write_text("metrics: {'success_rate': 0.0}\n")
    result.with_suffix('.json').write_text(
        '{"metrics":{"success_rate":75.0,'
        '"episode_successes":[true,false,true,true]},'
        '"evaluation_time_seconds":3.5,'
        '"row_indices":[1,2,3,4],'
        '"manifest_path":"/manifest.json"}\n'
    )
    details = parse_task_result_details(result)
    assert details['success_rate'] == 75.0
    assert details['episode_successes'] == [True, False, True, True]
    assert details['row_indices'] == [1, 2, 3, 4]
    assert details['evaluation_time_seconds'] == 3.5


def test_best_stage_selection_records_deployable_checkpoint():
    summary = {
        'stages': [
            {'stage': 'phase1', 'success_rate': 38.0, 'successes': 19,
             'checkpoint': '/phase1/weights.pt'},
            {'stage': 'phase2', 'success_rate': 78.0, 'successes': 39,
             'checkpoint': '/phase2/weights.pt'},
            {'stage': 'final', 'success_rate': 76.0, 'successes': 38,
             'checkpoint': '/final/weights.pt'},
        ]
    }
    add_best_stage(summary)
    assert summary['best_stage'] == {
        'stage': 'phase2',
        'success_rate': 78.0,
        'successes': 39,
        'checkpoint': '/phase2/weights.pt',
    }
