import torch

from locatemot.rmot.l83_target_bag_loss import l83_target_bag_loss


def test_multi_target_weakest_positive_is_used():
    scores = torch.tensor([[3.0, -0.2, 1.0]], requires_grad=True)
    loss, parts = l83_target_bag_loss(
        scores, torch.tensor([True]), ["multi_positive"], [["A", "C"]], [["A", "C", "B"]]
    )
    assert float(loss) > 0.0
    assert parts["positive_bag_count"] == 2
    loss.backward()
    assert scores.grad is not None and float(scores.grad.abs().sum()) > 0


def test_bad_duplicate_is_not_the_weakest_positive_observation():
    duplicate = torch.tensor([[2.5, -3.0, 1.0]], requires_grad=True)
    single = torch.tensor([[2.5, 1.0]], requires_grad=True)
    kwargs = dict(membership_mask=torch.tensor([True]), categories=["positive"], target_ids=[["A"]])
    loss_dup, _ = l83_target_bag_loss(duplicate, candidate_gt=[["A", "A", "B"]], **kwargs)
    loss_single, _ = l83_target_bag_loss(single, candidate_gt=[["A", "B"]], **kwargs)
    assert torch.allclose(loss_dup, loss_single, atol=1e-6)
    loss_dup.backward()
    assert float(duplicate.grad[0, 1]) == 0.0
