#!/usr/bin/env python3
"""Held-out train-video identity audit for the L39 prototype."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l39_identity_prototype import L39IdentityPrototype
from tools.train_l30_fragment_association_probe import PairProbe

CACHE_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
L30_CHECKPOINT = ROOT / "outputs/l30/train/fragment_probe_step500/checkpoint_fragment_probe_step500.pt"
L39_CHECKPOINT = ROOT / "outputs/l39/train/identity_prototype_smoke100_retry/checkpoint_l39_identity_prototype_step100.pt"
AUDIT = ROOT / "outputs/l39/audit/identity_probe_contract.json"
OUT = ROOT / "outputs/l39/eval/identity_audit.json"
CAL_VIDEO = "0015"
TEST_VIDEOS = ("0016", "0017")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def gt_set(value):
    if value is None: return set()
    if isinstance(value, (list, tuple, set)): return {str(x) for x in value}
    return {str(value)}


def load_fragments(videos):
    out = []
    for video in videos:
        c = torch.load(CACHE_ROOT / f"{video}.pt", map_location="cpu", weights_only=False)
        ptr = c["track_ptr"].tolist(); frames = c["obs_frame"].numpy(); source = c["obs_source"].numpy(); labels = c["obs_gt_ids"]
        for ti, track in enumerate(c["track_ids"].tolist()):
            begin, end = int(ptr[ti]), int(ptr[ti + 1]); ids = list(range(begin, end))[-8:]
            gids = set().union(*(gt_set(x) for x in labels[begin:end]))
            feat = c["obs_features"][torch.as_tensor(ids)].float(); fs = frames[ids].astype(np.float32)
            out.append({"video": video, "track_id": int(track), "features": feat,
                        "times": torch.as_tensor(fs / max(1.0, float(frames.max() + 1))),
                        "frames": set(int(x) for x in fs), "gids": gids,
                        "source": int(np.round(source[begin:end].mean())),
                        "labelled": bool(gids)})
    return out


def make_pairs(fragments, max_negative=8, max_inactive=4):
    by_video_frame = defaultdict(list)
    by_video_gt = defaultdict(list)
    for i, f in enumerate(fragments):
        for frame in f["frames"]: by_video_frame[(f["video"], frame)].append(i)
        for gid in f["gids"]: by_video_gt[(f["video"], gid)].append(i)
    pairs = []; seen = set()
    for a, fa in enumerate(fragments):
        if not fa["labelled"]: continue
        # All available same-GT fragment pairs are positives; cross-source ones
        # are retained for the main->reserve slice.
        pos_ids = sorted({b for g in fa["gids"] for b in by_video_gt[(fa["video"], g)]
                          if b != a and fragments[b]["track_id"] != fa["track_id"]})
        for b in pos_ids:
            frame = sorted(fa["frames"] & fragments[b]["frames"])
            key = (a, b, 1, "same_gt_fragment")
            if key not in seen:
                pairs.append({"a": a, "b": b, "label": 1, "kind": "same_gt_fragment",
                              "frame": frame[0] if frame else (max(fa["frames"]) if fa["frames"] else -1)})
                seen.add(key)
        hard = []; inactive = []
        for frame in sorted(fa["frames"]):
            for b in by_video_frame[(fa["video"], frame)]:
                if b == a: continue
                fb = fragments[b]
                if fa["gids"] & fb["gids"]: continue
                if fb["labelled"]: hard.append((b, frame))
                else: inactive.append((b, frame))
        for b, frame in sorted(set(hard))[:max_negative]:
            key = (a, b, 0, "same_frame_different_gt_hard")
            if key not in seen:
                pairs.append({"a": a, "b": b, "label": 0, "kind": "same_frame_different_gt_hard", "frame": frame}); seen.add(key)
        for b, frame in sorted(set(inactive))[:max_inactive]:
            key = (a, b, 0, "inactive")
            if key not in seen:
                pairs.append({"a": a, "b": b, "label": 0, "kind": "inactive", "frame": frame}); seen.add(key)
    if not pairs or not any(x["label"] for x in pairs) or not any(not x["label"] for x in pairs):
        raise RuntimeError("held-out identity pair audit has no positive/negative pairs")
    return pairs


def padded(fragments, ids, device):
    n = len(ids); values = torch.zeros((n, 8, 1432), device=device); times = torch.zeros((n, 8), device=device); mask = torch.zeros((n, 8), dtype=torch.bool, device=device)
    for j, i in enumerate(ids):
        k = min(8, len(fragments[i]["features"])); values[j, -k:] = fragments[i]["features"][-k:].to(device); times[j, -k:] = fragments[i]["times"][-k:].to(device); mask[j, -k:] = True
    return values, mask, times


def l30_features(a, b):
    out = []
    for sl in (slice(0, 512), slice(512, 1024), slice(1024, 1408)):
        x, y = a[sl], b[sl]; out.append((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-6))
    out.extend((a[sl] - b[sl]).abs() for sl in (slice(1408, 1415), slice(1415, 1423), slice(1423, 1431)))
    return torch.cat([x.reshape(-1) for x in out]).float()


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool); p = s[y]; n = s[~y]
    if not len(p) or not len(n): return None
    order = np.argsort(s, kind="stable"); rank = np.empty(len(order), float); rank[order] = np.arange(1, len(order) + 1)
    return float((rank[y].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def average_precision(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    if not y.any(): return None
    ordered = y[np.argsort(-s, kind="stable")]
    positions = np.flatnonzero(ordered)
    return float(np.mean([(ordered[:i + 1]).mean() for i in positions])) if len(positions) else 0.0


def choose_threshold(rows):
    values = np.asarray([x["score"] for x in rows], float); labels = np.asarray([x["label"] for x in rows], bool)
    candidates = np.unique(np.quantile(values, np.linspace(.01, .99, 128)))
    best = max(candidates, key=lambda t: 2 * int(((values >= t) & labels).sum()) / max(1, 2 * int(((values >= t) & labels).sum()) + int(((values >= t) & ~labels).sum()) + int(((values < t) & labels).sum())))
    return float(best)


def score_metrics(rows, threshold):
    s = np.asarray([x["score"] for x in rows], float); y = np.asarray([x["label"] for x in rows], bool); chosen = s >= threshold
    pos = y; hard = np.asarray([x["kind"] == "same_frame_different_gt_hard" for x in rows], bool); inactive = np.asarray([x["kind"] == "inactive" for x in rows], bool)
    same_pos = pos; precision = float((chosen & y).sum() / max(1, chosen.sum())); recall = float((chosen & y).sum() / max(1, y.sum()))
    pair_groups = defaultdict(lambda: {"p": [], "n": []})
    for x in rows:
        if x["kind"] == "same_gt_fragment": pair_groups[x["a"]]["p"].append(x["score"])
        elif x["kind"] == "same_frame_different_gt_hard": pair_groups[x["a"]]["n"].append(x["score"])
    violations = [min(g["p"]) <= max(g["n"]) for g in pair_groups.values() if g["p"] and g["n"]]
    cross = [x for x in rows if x["kind"] == "same_gt_fragment" and x["source_a"] != x["source_b"]]
    source = {}
    for name, filt in (("main_to_main", lambda x: x["source_a"] == 0 and x["source_b"] == 0), ("main_to_reserve", lambda x: x["source_a"] == 0 and x["source_b"] == 1), ("reserve_to_main", lambda x: x["source_a"] == 1 and x["source_b"] == 0), ("reserve_to_reserve", lambda x: x["source_a"] == 1 and x["source_b"] == 1)):
        q = [x for x in rows if x["kind"] == "same_gt_fragment" and filt(x)]; source[name] = {"pairs": len(q), "recall_at_threshold": float(np.mean([x["score"] >= threshold for x in q])) if q else None}
    same_scores = s[np.asarray([x["kind"] == "same_gt_fragment" for x in rows], bool)]
    hard_scores = s[hard]
    inactive_scores = s[inactive]
    return {"pairs": len(rows), "positive_pairs": int(y.sum()), "hard_negative_pairs": int(hard.sum()), "roc_auc": auc(s, y), "pr_auc": average_precision(s, y), "threshold_precision": precision, "same_gt_recall": recall, "same_gt_precision": precision, "same_gt_score_mean": float(same_scores.mean()) if len(same_scores) else None, "hard_negative_score_mean": float(hard_scores.mean()) if len(hard_scores) else None, "different_gt_separation_mean": float(same_scores.mean() - hard_scores.mean()) if len(same_scores) and len(hard_scores) else None, "inactive_score_mean": float(inactive_scores.mean()) if len(inactive_scores) else None, "hard_negative_violation": float(np.mean(violations)) if violations else None, "main_to_reserve_recall": float(np.mean([x["score"] >= threshold for x in cross])) if cross else None, "inactive_false_continuity": float(np.mean(s[inactive] >= threshold)) if inactive.any() else None, "source_fragment": source, "threshold": threshold}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default=str(L39_CHECKPOINT)); ap.add_argument("--out", default=str(OUT)); ap.add_argument("--device", default="cuda:0"); args = ap.parse_args(); assert Path.cwd().resolve() == ROOT
    fragments = load_fragments((CAL_VIDEO,) + TEST_VIDEOS); pairs = make_pairs(fragments)
    device = torch.device(args.device); model = L39IdentityPrototype(hidden=96, history=8, prototype_dim=96).to(device); model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"]); model.eval()
    state = torch.load(L30_CHECKPOINT, map_location="cpu", weights_only=False); l30 = PairProbe(int(state["model"]["linear.weight"].shape[1])); l30.load_state_dict(state["model"]); l30.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), 256):
            chunk = pairs[start:start + 256]; ai = [x["a"] for x in chunk]; bi = [x["b"] for x in chunk]; a, am, at = padded(fragments, ai, device); b, bm, bt = padded(fragments, bi, device)
            za = model(a, am, at)["prototype"].float().cpu(); zb = model(b, bm, bt)["prototype"].float().cpu()
            for row, x in enumerate(chunk):
                x["l39"] = float((za[row] * zb[row]).sum()); x["history_cosine"] = float((fragments[x["a"]]["features"][:,512:1024].mean(0) * fragments[x["b"]]["features"][:,512:1024].mean(0)).sum() / (fragments[x["a"]]["features"][:,512:1024].mean(0).norm() * fragments[x["b"]]["features"][:,512:1024].mean(0).norm()).clamp_min(1e-6)); x["l30_feature"] = l30_features(fragments[x["a"]]["features"].mean(0), fragments[x["b"]]["features"].mean(0))
    with torch.no_grad():
        for x in pairs: x["l30"] = float(l30(x["l30_feature"].unsqueeze(0)).item())
    def prep(name):
        for x in pairs: x["score"] = x[name]; x["source_a"] = fragments[x["a"]]["source"]; x["source_b"] = fragments[x["b"]]["source"]
        cal = [x for x in pairs if fragments[x["a"]]["video"] == CAL_VIDEO and x["kind"] != "inactive"]; test = [x for x in pairs if fragments[x["a"]]["video"] in TEST_VIDEOS]
        threshold = choose_threshold(cal); result = score_metrics(test, threshold); result["calibration_pair_count"] = len(cal); return result, test
    results = {}
    for name in ("history_cosine", "l30", "l39"):
        results[name], test_rows = prep(name)
    examples = sorted([x for x in test_rows if x["kind"] == "same_frame_different_gt_hard"], key=lambda x: -x["l39"])[:20]
    reps = [{"video": fragments[x["a"]]["video"], "frame": x["frame"],
             "track_a": fragments[x["a"]]["track_id"], "track_b": fragments[x["b"]]["track_id"],
             "kind": x["kind"], "l39": x["l39"], "l30": x["l30"],
             "history_cosine": x["history_cosine"], "source_a": x["source_a"],
             "source_b": x["source_b"], "gt_a": sorted(fragments[x["a"]]["gids"]),
             "gt_b": sorted(fragments[x["b"]]["gids"])} for x in examples]
    payload = {"format": "locatemot-l39-identity-heldout-audit-v1", "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha(Path(args.checkpoint)), "l30_checkpoint": str(L30_CHECKPOINT.resolve()), "l30_checkpoint_sha256": sha(L30_CHECKPOINT), "cache_manifest_sha256": sha(CACHE_ROOT / "manifest.json"), "train_only_calibration_video": CAL_VIDEO, "heldout_train_videos": list(TEST_VIDEOS), "screening_gt_used": False, "structure_selection_used_screening_gt": False, "labels": "GT_PRIVILEGED_ORACLE from held-out train-video labels, not screening", "results": results, "representative_hard_negative_pairs": reps, "semantic_inputs_excluded": ["expression", "source_id", "pool_id", "group_id", "state_key"], "token_level_alignment_verified": False, "motion_language_decomposition": "not claimed"}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
