import torch

from scripts.train.lewm import compute_lewm_losses


class FailIfCalled(torch.nn.Module):
    def forward(self, _):
        raise AssertionError('SIGReg must not run when its weight is zero')


def test_prediction_only_loss_skips_sigreg():
    pred = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    target = torch.zeros_like(pred)
    emb = torch.zeros(1, 2, 2)

    output = compute_lewm_losses(
        pred_emb=pred,
        tgt_emb=target,
        emb=emb,
        sigreg=FailIfCalled(),
        sigreg_weight=0.0,
    )

    assert set(output) == {'pred_loss', 'loss'}
    assert output['loss'] is output['pred_loss']
    output['loss'].backward()
    assert pred.grad is not None
