import torch

from locatemot.rmot.l83_target_bag_loss import l83_target_bag_loss


def test_present_uncovered_is_masked_not_inactive():
    scores = torch.tensor([[1.0, -1.0]], requires_grad=True)
    loss, parts = l83_target_bag_loss(
        scores, torch.tensor([False]), ["present_uncovered"], [["missing"]], [["A", None]]
    )
    assert float(loss) == 0.0
    assert parts["inactive_count"] == 0
    assert parts["masked_missing_count"] == 2
