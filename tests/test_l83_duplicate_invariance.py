import torch

from locatemot.rmot.l83_target_bags import bag_values, build_target_bag_layout


def test_duplicate_invariance():
    one = build_target_bag_layout(["A", None])
    two = build_target_bag_layout(["A", "A", None])
    assert float(bag_values(torch.tensor([2.0, 0.0]), one, ["A"])[1][0]) == 2.0
    assert float(bag_values(torch.tensor([2.0, 2.0, 0.0]), two, ["A"])[1][0]) == 2.0


def test_bad_duplicate_does_not_set_positive_bag_score():
    layout = build_target_bag_layout(["A", "A", "B"])
    _, scores, positive = bag_values(torch.tensor([2.5, -3.0, 1.0]), layout, ["A"])
    assert float(scores[0]) == 2.5
    assert float(scores[1]) == 1.0
    assert positive.tolist() == [True, False]
