#!/usr/bin/env python3
"""Held-out identity audit for L40, with calibration-only threshold fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l39_identity_prototype import L39IdentityPrototype
from locatemot.models.l40_raw_image_identity import L40RawImageIdentity
from tools.l40_raw_data import (HELDOUT_VIDEOS, WEIGHTS, StreamingClipEncoder, load_fragments,
                                make_pairs, sha256)

L39_CHECKPOINT = ROOT / "outputs/l39/train/identity_prototype_smoke100_retry/checkpoint_l39_identity_prototype_step100.pt"
L30_CHECKPOINT = ROOT / "outputs/l30/train/fragment_probe_step500/checkpoint_fragment_probe_step500.pt"
L30_OUT_DIM = 1


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    p, n = s[y], s[~y]
    if not len(p) or not len(n): return None
    order = np.argsort(s, kind="stable"); rank = np.empty(len(order), float); rank[order] = np.arange(1, len(order) + 1)
    return float((rank[y].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def average_precision(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    if not y.any(): return None
    order = np.argsort(-s, kind="stable"); ordered = y[order]; hits = np.flatnonzero(ordered)
    return float(np.mean([(ordered[:i + 1]).mean() for i in hits])) if len(hits) else 0.0


def threshold_f1(rows):
    values = np.asarray([x["score"] for x in rows], float); labels = np.asarray([x["label"] for x in rows], bool)
    if not len(values): return None
    candidates = np.unique(np.quantile(values, np.linspace(.01, .99, 128)))
    def f1(t):
        pred = values >= t; tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum()); fn = int((~pred & labels).sum())
        return 2 * tp / max(1, 2 * tp + fp + fn)
    return float(max(candidates, key=f1))


def l30_feature(a, b):
    parts = []
    for sl in (slice(0, 512), slice(512, 1024), slice(1024, 1408)):
        x, y = a[sl], b[sl]
        parts.append((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-6))
    parts += [(a[sl] - b[sl]).abs() for sl in (slice(1408, 1415), slice(1415, 1423), slice(1423, 1431))]
    return torch.cat([x.reshape(-1) for x in parts]).float()


def pad_raw(fragments, ids, embeds, device):
    n = len(ids); image_dim = int(embeds[ids[0]].shape[-1]); image = torch.zeros(n, 8, image_dim); numeric = torch.zeros(n, 8, 24); mask = torch.zeros(n, 8, dtype=torch.bool); times = torch.zeros(n, 8)
    for j, i in enumerate(ids):
        f = fragments[i]; k = min(8, len(f["obs"])); st = 8 - k; image[j, st:] = embeds[i][-k:]; numeric[j, st:] = torch.stack([x["numeric"] for x in f["obs"][-k:]])
        denom = max(1.0, float(max(f["frames"]) + 1)); times[j, st:] = torch.tensor([x["frame"] / denom for x in f["obs"][-k:]]); mask[j, st:] = True
    return image.to(device), numeric.to(device), mask.to(device), times.to(device)


def pad_l39(fragments, ids, device):
    values = torch.zeros(len(ids), 8, 1432); times = torch.zeros(len(ids), 8); mask = torch.zeros(len(ids), 8, dtype=torch.bool)
    for j, i in enumerate(ids):
        f = fragments[i]; k = min(8, len(f["obs"])); st = 8-k; values[j, st:] = torch.stack([x["frozen_features"] for x in f["obs"][-k:]])
        denom = max(1.0, float(max(f["frames"]) + 1)); times[j, st:] = torch.tensor([x["frame"] / denom for x in f["obs"][-k:]]); mask[j, st:] = True
    return values.to(device), mask.to(device), times.to(device)


def score_rows(rows, fragments, values, threshold):
    chosen = values >= threshold; labels = np.asarray([x["label"] for x in rows], bool)
    hard = np.asarray([x["kind"] == "same_frame_different_gt_hard" for x in rows], bool)
    inactive = np.asarray([x["kind"] == "inactive" for x in rows], bool)
    groups = defaultdict(lambda: {"p": [], "n": []})
    for x in rows:
        groups[x["a"]]["p" if x["label"] else "n"].append(x["score"])
    violations = [min(g["p"]) <= max(g["n"]) for g in groups.values() if g["p"] and g["n"]]
    pos = values[labels]; hard_scores = values[hard]; inactive_scores = values[inactive]
    source = {}
    for name, pred in (("main_to_main", lambda x: fragments[x["a"]]["source"] == 0 and fragments[x["b"]]["source"] == 0), ("main_to_reserve", lambda x: fragments[x["a"]]["source"] == 0 and fragments[x["b"]]["source"] == 1), ("reserve_to_main", lambda x: fragments[x["a"]]["source"] == 1 and fragments[x["b"]]["source"] == 0), ("reserve_to_reserve", lambda x: fragments[x["a"]]["source"] == 1 and fragments[x["b"]]["source"] == 1)):
        q = [i for i, x in enumerate(rows) if x["label"] and pred(x)]; source[name] = {"pairs": len(q), "recall": float(chosen[q].mean()) if q else None}
    cross = [i for i, x in enumerate(rows) if x["label"] and fragments[x["a"]]["source"] == 0 and fragments[x["b"]]["source"] == 1]
    return {"pairs": len(rows), "positive_pairs": int(labels.sum()), "hard_negative_pairs": int(hard.sum()), "inactive_pairs": int(inactive.sum()), "roc_auc": auc(values, labels), "pr_auc": average_precision(values, labels), "threshold": float(threshold) if threshold is not None else None, "same_gt_precision": float((chosen & labels).sum() / max(1, chosen.sum())), "same_gt_recall": float((chosen & labels).sum() / max(1, labels.sum())), "false_positive_per_pair": float((chosen & ~labels).sum() / max(1, len(rows))), "empty_rate": float((~chosen).mean()), "inactive_false_continuity": float((chosen[inactive]).mean()) if inactive.any() else None, "hard_negative_violation": float(np.mean(violations)) if violations else None, "same_gt_score_mean": float(pos.mean()) if len(pos) else None, "hard_negative_score_mean": float(hard_scores.mean()) if len(hard_scores) else None, "different_gt_separation_mean": float(pos.mean() - hard_scores.mean()) if len(pos) and len(hard_scores) else None, "main_to_reserve_recall": float(chosen[cross].mean()) if cross else None, "source_fragment": source, "violation_group_count": len(violations)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--crop-batch", type=int, default=32); args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT
    out = Path(args.out); out = out if out.is_absolute() else ROOT / out; out.mkdir(parents=True, exist_ok=False)
    start = time.time(); fragments, alignment = load_fragments(HELDOUT_VIDEOS, include_frozen_features=True); pairs = make_pairs(fragments)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    streamer = StreamingClipEncoder(device=device, weights=WEIGHTS, batch_size=args.crop_batch); embeds_list = streamer.encode(fragments, range(len(fragments))); embeds = {i: x for i, x in enumerate(embeds_list)}; del embeds_list, streamer
    l40 = L40RawImageIdentity(hidden=96, history=8).to(device); state = torch.load(args.checkpoint, map_location=device, weights_only=False); l40.load_state_dict(state["model"]); l40.eval()
    l39 = L39IdentityPrototype(hidden=96, history=8, prototype_dim=96).to(device); l39.load_state_dict(torch.load(L39_CHECKPOINT, map_location=device, weights_only=False)["model"]); l39.eval()
    l30_state = torch.load(L30_CHECKPOINT, map_location="cpu", weights_only=False); from tools.train_l30_fragment_association_probe import PairProbe; l30 = PairProbe(int(l30_state["model"]["linear.weight"].shape[1])); l30.load_state_dict(l30_state["model"]); l30.eval()
    raw_proto = {}; l39_proto = {}
    with torch.inference_mode():
        for start_id in range(0, len(fragments), 128):
            ids = list(range(start_id, min(len(fragments), start_id + 128))); raw_proto.update({i: z.cpu() for i, z in zip(ids, l40(*pad_raw(fragments, ids, embeds, device))["prototype"])}); a, m, t = pad_l39(fragments, ids, device); l39_proto.update({i: z.cpu() for i, z in zip(ids, l39(a, m, t)["prototype"])})
    for p in pairs:
        a, b = fragments[p["a"]], fragments[p["b"]]
        p["history"] = float(torch.dot(torch.stack([x["history"] for x in a["obs"]]).mean(0), torch.stack([x["history"] for x in b["obs"]]).mean(0)) / torch.stack([x["history"] for x in a["obs"]]).mean(0).norm().clamp_min(1e-6) / torch.stack([x["history"] for x in b["obs"]]).mean(0).norm().clamp_min(1e-6))
        p["l39"] = float(torch.dot(l39_proto[p["a"]], l39_proto[p["b"]])); p["l40"] = float(torch.dot(raw_proto[p["a"]], raw_proto[p["b"]])); p["l30"] = float(l30(l30_feature(a["obs"][-1]["frozen_features"], b["obs"][-1]["frozen_features"]).unsqueeze(0)).item())
    cal_idx = [i for i, p in enumerate(pairs) if fragments[p["a"]]["video"] == "0015"]
    test_idx = [i for i, p in enumerate(pairs) if fragments[p["a"]]["video"] in ("0016", "0017")]
    results = {}
    for name in ("history", "l30", "l39", "l40"):
        all_values = np.asarray([p[name] for p in pairs]);
        cal_rows = [{"score": float(all_values[i]), "label": pairs[i]["label"], "kind": pairs[i]["kind"], "a": pairs[i]["a"]} for i in cal_idx]
        test_rows = [{"score": float(all_values[i]), "label": pairs[i]["label"], "kind": pairs[i]["kind"], "a": pairs[i]["a"], "b": pairs[i]["b"]} for i in test_idx]
        threshold = threshold_f1(cal_rows); results[name] = score_rows(test_rows, fragments, all_values[test_idx], threshold); results[name]["calibration_pairs"] = len(cal_rows); results[name]["test_pairs"] = len(test_rows)
    reps = []
    for p in sorted([pairs[i] for i in test_idx if pairs[i]["kind"] == "same_frame_different_gt_hard"], key=lambda x: -x["l40"])[:20]:
        reps.append({"video": fragments[p["a"]]["video"], "frame": p["frame"], "track_a": fragments[p["a"]]["track_id"], "track_b": fragments[p["b"]]["track_id"], "source_a": fragments[p["a"]]["source"], "source_b": fragments[p["b"]]["source"], "gt_a": sorted(fragments[p["a"]]["gids"]), "gt_b": sorted(fragments[p["b"]]["gids"]), "scores": {k: p[k] for k in ("history", "l30", "l39", "l40")}})
    payload = {"format": "locatemot-l40-streaming-raw-image-identity-heldout-v1", "stage": "L40", "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(Path(args.checkpoint)), "l39_checkpoint": str(L39_CHECKPOINT.resolve()), "l39_checkpoint_sha256": sha256(L39_CHECKPOINT), "l30_checkpoint": str(L30_CHECKPOINT.resolve()), "l30_checkpoint_sha256": sha256(L30_CHECKPOINT), "heldout_videos": list(HELDOUT_VIDEOS), "calibration_video": "0015", "test_videos": ["0016", "0017"], "pair_count": len(pairs), "alignment_count": len(alignment), "screening_gt_used": False, "structure_selection_used_screening_gt": False, "raw_embeddings_persisted": False, "results": results, "representative_hard_negative_pairs": reps, "semantic_inputs_excluded": ["expression", "source_id", "pool_id", "group_id", "state_key"], "labels": "GT_PRIVILEGED_ORACLE from held-out train videos; no fixed screening labels", "token_level_alignment_verified": False, "motion_language_decomposition": "not claimed", "elapsed_sec": time.time() - start}
    out_file = out / "identity_audit.json"; out_file.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
