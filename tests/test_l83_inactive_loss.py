import torch

from locatemot.rmot.l83_target_bag_loss import l83_target_bag_loss


def test_inactive_has_absolute_negative_anchor():
    scores = torch.tensor([[0.5, -0.5]], requires_grad=True)
    loss, parts = l83_target_bag_loss(
        scores, torch.tensor([False]), ["inactive"], [[]], [["A", None]]
    )
    assert float(loss) > 0.0
    assert parts["inactive_count"] == 1
    loss.backward()
    assert float(scores.grad.abs().sum()) > 0
