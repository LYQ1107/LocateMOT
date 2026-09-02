from locatemot.evaluation.l83_target_bag_metrics import roc_auc


def test_true_roc_auc_differs_from_pair_accuracy():
    positive = [0.9, 0.2]
    negative = [0.8, 0.1]
    pair_accuracy = sum(p > n for p, n in zip(positive, negative)) / 2.0
    auc = roc_auc([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1])
    assert pair_accuracy == 1.0
    assert auc == 0.75
    assert auc != pair_accuracy
