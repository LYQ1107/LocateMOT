"""Stage L0-C tests: splits, labels, cache atomicity, model fwd/bwd, restore."""
import json
import os
import tempfile

import torch

from locatemot.data.token_cache import exists, read_frame_cache, write_frame_cache
from locatemot.models.track_decoder.model import PairwiseModel, TrackDecoderModel


def _fake_pair():
    return {
        "split": "train", "dataset": "youtube_vos_train", "video_id": "v",
        "reference_frame": 0, "current_frame": 5, "temporal_gap": 5,
        "protocol": "generic", "reference_token_id": "r", "current_token_id": "c",
        "reference_targets": [{"track_id": 1, "gt_box": [0, 0, 10, 10]}],
        "assignment_targets": [{"track_id": 1, "candidate_index": 0}],
        "no_match_targets": [2], "candidate_missing_targets": [3],
        "reference_target_count": 1, "current_candidate_count": 2,
    }


def test_split_no_overlap():
    ids = {}
    for split in ["train", "calibration", "heldout"]:
        d = json.load(open(f"configs/data/l0_c_{split}_videos.json"))
        ids[split] = {(e["dataset"], e["video_id"]) for e in d["videos"]}
    assert ids["train"] & ids["calibration"] == set()
    assert ids["train"] & ids["heldout"] == set()
    assert ids["calibration"] & ids["heldout"] == set()


def test_pair_label_semantics():
    rec = _fake_pair()
    # candidate_missing must NOT be supervised as no_match
    assert 3 in rec["candidate_missing_targets"]
    assert 3 not in rec["no_match_targets"]
    assert 3 not in [t["track_id"] for t in rec["assignment_targets"]]
    assert rec["no_match_targets"] == [2]


def test_cache_atomic_write_and_read():
    with tempfile.TemporaryDirectory() as tmp:
        key = "youtube_vos_train/v/00000/generic"
        feats = {"pbd": torch.zeros(3, 8)}
        write_frame_cache(tmp, key, {"pbd": feats["pbd"].numpy()}, {"candidate_count": 3})
        assert exists(tmp, key)
        loaded = read_frame_cache(tmp, key)
        assert loaded["meta"]["candidate_count"] == 3
        assert loaded["features"]["pbd"].shape == (3, 8)


def test_b3_forward_backward():
    model = PairwiseModel()
    b = _model_batch(2, 8, 8)
    pred = model(b)
    loss = pred["match_logits"].sigmoid().mean() + pred["no_match_logits"].sigmoid().mean()
    loss.backward()
    assert loss.item() > 0


def test_b4_forward_backward_with_empty_masks():
    model = TrackDecoderModel()
    b = _model_batch(2, 8, 8)
    b["ref_mask"][1] = False
    b["cur_mask"][1] = False
    b["ref_mask"][1, 0] = True  # dummy key (trainer behaviour)
    b["cur_mask"][1, 0] = True
    pred = model(b)
    assert torch.isfinite(pred["match_logits"]).all()
    loss = pred["match_logits"].sigmoid().mean()
    loss.backward()
    assert loss.item() > 0


def test_checkpoint_restore():
    model = TrackDecoderModel()
    sd = model.state_dict()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save({"model": sd, "step": 10}, f.name)
        path = f.name
    model2 = TrackDecoderModel()
    model2.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    for k in sd:
        assert torch.equal(sd[k], model2.state_dict()[k])
    os.unlink(path)


def _model_batch(b, m, n):
    return {
        "ref_pbd": torch.randn(b, m, 2048), "ref_region": torch.randn(b, m, 4608),
        "ref_geom": torch.rand(b, m, 5), "ref_gen": torch.rand(b, m),
        "ref_cat": torch.randn(b, m, 32), "ref_mask": torch.ones(b, m, dtype=torch.bool),
        "cur_pbd": torch.randn(b, n, 2048), "cur_region": torch.randn(b, n, 4608),
        "cur_geom": torch.rand(b, n, 5), "cur_gen": torch.rand(b, n),
        "cur_cat": torch.randn(b, n, 32), "cur_mask": torch.ones(b, n, dtype=torch.bool),
        "gap": torch.full((b, 1), 5.0),
    }
