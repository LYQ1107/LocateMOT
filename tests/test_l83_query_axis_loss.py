import torch

from locatemot.rmot.l83_target_bag_loss import l83_target_bag_loss


def test_query_axis_uses_exact_same_frame_flip():
    scores = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
    loss, parts = l83_target_bag_loss(
        scores, torch.tensor([True, True]), ["positive", "positive"], [["A"], ["B"]], [["A", "B"], ["A", "B"]]
    )
    assert parts["query_flip_count"] == 2
    assert torch.isfinite(loss)
    loss.backward()
    assert float(scores.grad.abs().sum()) > 0
