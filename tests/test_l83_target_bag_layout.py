import torch

from locatemot.rmot.l83_target_bags import bag_values, build_target_bag_layout


def test_target_bag_layout_keeps_duplicate_rows_and_background_singletons():
    layout = build_target_bag_layout(["A", "A", None, "B", None])
    assert layout.target_to_rows["A"].tolist() == [0, 1]
    assert layout.background_rows.tolist() == [2, 4]
    keys, scores, positives = bag_values(torch.tensor([2.0, 1.0, 0.5, 3.0, -1.0]), layout, ["A"])
    assert keys == [("target", "A"), ("target", "B"), ("background", 2), ("background", 4)]
    assert torch.allclose(scores, torch.tensor([2.0, 3.0, 0.5, -1.0]))
    assert positives.tolist() == [True, False, False, False]
