from types import SimpleNamespace

import torch

from locatemot.evaluation.l83_target_bag_metrics import group_metrics


def test_unique_bag_ranking_does_not_spend_two_ranks_on_duplicate():
    data = SimpleNamespace(
        group_key="g", dataset="refer_kitti_v2", video="0016", frame_id=1,
        query_unit_keys=["u"], query_ids=[0], candidate_count=4,
        labels=torch.tensor([[True, True, False, False]]), membership_mask=torch.tensor([True]),
        categories=["multi_positive"], target_ids=[["A"]], candidate_gt=[["A", "A", "B", None]],
        row_keys_digest=["k"], row_offsets=[0, 1, 2, 3], candidate_indices=[0, 0, 1, 2], pool_ids=[0, 1, 1, 1],
    )
    record, _ = group_metrics(data, torch.tensor([[3.0, 2.0, 1.0, 0.0]]))
    assert record["target_bag_hit_at1"] == 1
    assert record["target_bag_recall_at5_numerator"] == 1
