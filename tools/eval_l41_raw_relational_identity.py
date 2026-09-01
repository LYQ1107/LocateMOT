#!/usr/bin/env python3
"""L41 held-out relational identity audit; no RMOT emission or screening."""
from __future__ import annotations

import argparse
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
from locatemot.models.l41_raw_relational_identity import L41RawRelationalIdentity
from tools.l41_raw_data import (HELDOUT_VIDEOS, WEIGHTS, StreamingClipPatchEncoder, load_fragments,
                                make_pairs, pad_patches, relation_features, sha256)

L39 = ROOT / "outputs/l39/train/identity_prototype_smoke100_retry/checkpoint_l39_identity_prototype_step100.pt"
L30 = ROOT / "outputs/l30/train/fragment_probe_step500/checkpoint_fragment_probe_step500.pt"
L40_AUDIT = ROOT / "outputs/l40/eval/identity_audit_v1/identity_audit.json"


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool); p, n = s[y], s[~y]
    if not len(p) or not len(n): return None
    order = np.argsort(s, kind="stable"); rank = np.empty(len(order), float); rank[order] = np.arange(1, len(order) + 1)
    return float((rank[y].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def ap(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    if not y.any(): return None
    order = np.argsort(-s, kind="stable"); z = y[order]; hit = np.flatnonzero(z)
    return float(np.mean([(z[:i + 1]).mean() for i in hit])) if len(hit) else 0.0


def threshold(rows):
    v = np.asarray([x["score"] for x in rows], float); y = np.asarray([x["label"] for x in rows], bool)
    if not len(v): return None
    cand = np.unique(np.quantile(v, np.linspace(.01, .99, 128)))
    def f1(t):
        p = v >= t; tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
        return 2 * tp / max(1, 2 * tp + fp + fn)
    return float(max(cand, key=f1))


def l30_feature(a, b):
    parts = []
    for sl in (slice(0, 512), slice(512, 1024), slice(1024, 1408)):
        x, y = a[sl], b[sl]; parts.append((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-6))
    parts += [(a[sl] - b[sl]).abs() for sl in (slice(1408, 1415), slice(1415, 1423), slice(1423, 1431))]
    return torch.cat([x.reshape(-1) for x in parts]).float()


def pad_l39(fragments, ids, device):
    x = torch.zeros(len(ids), 8, 1432); t = torch.zeros(len(ids), 8); m = torch.zeros(len(ids), 8, dtype=torch.bool)
    for j, i in enumerate(ids):
        f = fragments[i]; k = min(8, len(f["obs"])); st = 8-k; x[j, st:] = torch.stack([o["frozen_features"] for o in f["obs"][-k:]])
        den = max(1.0, float(max(f["frames"]) + 1)); t[j, st:] = torch.tensor([o["frame"] / den for o in f["obs"][-k:]]); m[j, st:] = True
    return x.to(device), m.to(device), t.to(device)


def metrics(rows, fragments, values, t):
    y = np.asarray([x["label"] for x in rows], bool); chosen = values >= t if t is not None else np.zeros(len(rows), bool)
    hard = np.asarray([x["kind"] == "same_frame_different_gt_hard" for x in rows], bool); inactive = np.asarray([x["kind"] == "inactive" for x in rows], bool)
    groups = defaultdict(lambda: {"p": [], "n": []})
    for x in rows: groups[x["a"]]["p" if x["label"] else "n"].append(x["score"])
    vio = [min(g["p"]) <= max(g["n"]) for g in groups.values() if g["p"] and g["n"]]
    pos, neg = values[y], values[hard]
    source = {}
    for name, fn in (("main_to_main", lambda x: fragments[x["a"]]["source"] == 0 and fragments[x["b"]]["source"] == 0), ("main_to_reserve", lambda x: fragments[x["a"]]["source"] == 0 and fragments[x["b"]]["source"] == 1), ("reserve_to_main", lambda x: fragments[x["a"]]["source"] == 1 and fragments[x["b"]]["source"] == 0), ("reserve_to_reserve", lambda x: fragments[x["a"]]["source"] == 1 and fragments[x["b"]]["source"] == 1)):
        idx = [i for i, x in enumerate(rows) if x["label"] and fn(x)]; source[name] = {"pairs": len(idx), "recall": float(chosen[idx].mean()) if idx else None}
    cross = [i for i, x in enumerate(rows) if x["label"] and fragments[x["a"]]["source"] == 0 and fragments[x["b"]]["source"] == 1]
    time_bins = {}
    for name, fn in (("dt_le_2", lambda d: d <= 2), ("dt_3_8", lambda d: 3 <= d <= 8), ("dt_gt_8", lambda d: d > 8)):
        idx = [i for i, x in enumerate(rows) if fn(abs(fragments[x["a"]]["obs"][-1]["frame"] - fragments[x["b"]]["obs"][-1]["frame"]))]
        time_bins[name] = {"pairs": len(idx), "positive_pairs": int(y[idx].sum()) if idx else 0, "mean_score": float(values[idx].mean()) if idx else None}
    return {"pairs": len(rows), "positive_pairs": int(y.sum()), "hard_negative_pairs": int(hard.sum()), "inactive_pairs": int(inactive.sum()), "roc_auc": auc(values, y), "pr_auc": ap(values, y), "threshold_f1_from_0015": float(t) if t is not None else None, "same_gt_precision": float((chosen & y).sum() / max(1, chosen.sum())), "same_gt_recall": float((chosen & y).sum() / max(1, y.sum())), "false_positive_rate_over_pairs": float((chosen & ~y).sum() / max(1, len(rows))), "inactive_false_continuity": float(chosen[inactive].mean()) if inactive.any() else None, "hard_negative_violation": float(np.mean(vio)) if vio else None, "violation_groups": len(vio), "same_gt_score_mean": float(pos.mean()) if len(pos) else None, "hard_negative_score_mean": float(neg.mean()) if len(neg) else None, "positive_minus_hard_mean": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else None, "main_to_reserve_recall": float(chosen[cross].mean()) if cross else None, "source_fragment": source, "time_gap_strata": time_bins}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--crop-batch", type=int, default=32); args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT
    out = Path(args.out); out = out if out.is_absolute() else ROOT / out; out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time(); fragments, alignment = load_fragments(HELDOUT_VIDEOS, include_frozen_features=True); pairs = make_pairs(fragments); device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    encoder = StreamingClipPatchEncoder(device=device, weights=WEIGHTS, batch_size=args.crop_batch); patch_list = encoder.encode(fragments, range(len(fragments))); patch_map = {i: x for i, x in enumerate(patch_list)}; del patch_list, encoder
    model = L41RawRelationalIdentity(hidden=96, history=8).to(device); model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"]); model.eval(); l39 = L39IdentityPrototype(hidden=96, history=8, prototype_dim=96).to(device); l39.load_state_dict(torch.load(L39, map_location=device, weights_only=False)["model"]); l39.eval()
    l39p = {}; l41s = []
    with torch.inference_mode():
        for start_id in range(0, len(fragments), 128):
            ids = list(range(start_id, min(len(fragments), start_id + 128))); left, lm = pad_patches(fragments, ids, patch_map, device); l41p = []
            for i in ids: l41p.append(None)
            # Prototypes are not needed; each pair is scored in batches below.
            a, am, at = pad_l39(fragments, ids, device); z = l39(a, am, at)["prototype"].cpu(); l39p.update({i: v for i, v in zip(ids, z)})
        for start_id in range(0, len(pairs), 128):
            chunk = pairs[start_id:start_id + 128]; ai = [x["a"] for x in chunk]; bi = [x["b"] for x in chunk]; la, lm = pad_patches(fragments, ai, patch_map, device); rb, rm = pad_patches(fragments, bi, patch_map, device); rel = torch.stack([relation_features(fragments[x["a"]], fragments[x["b"]]) for x in chunk]).to(device); s = model(la, rb, rel, lm, rm)["logit"].cpu().tolist(); l41s.extend(float(x) for x in s)
    l30_state = torch.load(L30, map_location="cpu", weights_only=False); from tools.train_l30_fragment_association_probe import PairProbe; l30 = PairProbe(int(l30_state["model"]["linear.weight"].shape[1])); l30.load_state_dict(l30_state["model"]); l30.eval()
    scores = {"l41": np.asarray(l41s, float), "history": np.asarray([float(torch.dot(torch.stack([o["history"] for o in fragments[p["a"]]["obs"]]).mean(0), torch.stack([o["history"] for o in fragments[p["b"]]["obs"]]).mean(0)) / torch.stack([o["history"] for o in fragments[p["a"]]["obs"]]).mean(0).norm().clamp_min(1e-6) / torch.stack([o["history"] for o in fragments[p["b"]]["obs"]]).mean(0).norm().clamp_min(1e-6)) for p in pairs]), "l39": np.asarray([float(torch.dot(l39p[p["a"]], l39p[p["b"]])) for p in pairs]), "l30": np.asarray([float(l30(l30_feature(fragments[p["a"]]["obs"][-1]["frozen_features"], fragments[p["b"]]["obs"][-1]["frozen_features"]).unsqueeze(0)).item()) for p in pairs])}
    cal = [i for i, p in enumerate(pairs) if fragments[p["a"]]["video"] == "0015"]; test = [i for i, p in enumerate(pairs) if fragments[p["a"]]["video"] in ("0016", "0017")]; results = {}
    for name, values in scores.items():
        cal_rows = [{"score": values[i], "label": pairs[i]["label"]} for i in cal]; test_rows = [{"score": values[i], "label": pairs[i]["label"], "kind": pairs[i]["kind"], "a": pairs[i]["a"], "b": pairs[i]["b"]} for i in test]; results[name] = metrics(test_rows, fragments, values[test], threshold(cal_rows)); results[name]["calibration_pairs"] = len(cal); results[name]["test_pairs"] = len(test)
    if L40_AUDIT.exists(): results["l40_prior_audit"] = json.loads(L40_AUDIT.read_text())["results"]["l40"]
    reps = []
    for i in sorted(test, key=lambda x: -scores["l41"][x])[:20]:
        p = pairs[i]; fa, fb = fragments[p["a"]], fragments[p["b"]]; reps.append({"video": fa["video"], "frame": p["frame"], "kind": p["kind"], "track_a": fa["track_id"], "track_b": fb["track_id"], "source_a": fa["source"], "source_b": fb["source"], "gt_a": sorted(fa["gids"]), "gt_b": sorted(fb["gids"]), "image_a": fa["obs"][-1]["image"], "image_b": fb["obs"][-1]["image"], "box_a": fa["obs"][-1]["box"], "box_b": fb["obs"][-1]["box"], "scores": {k: float(v[i]) for k, v in scores.items()}})
    payload = {"format": "locatemot-l41-streaming-raw-relational-identity-heldout-v1", "stage": "L41", "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(Path(args.checkpoint)), "l39_checkpoint": str(L39.resolve()), "l39_checkpoint_sha256": sha256(L39), "l30_checkpoint": str(L30.resolve()), "l30_checkpoint_sha256": sha256(L30), "heldout_videos": list(HELDOUT_VIDEOS), "calibration_video": "0015", "test_videos": ["0016", "0017"], "pair_count": len(pairs), "alignment_count": len(alignment), "screening_gt_used": False, "structure_selection_used_screening_gt": False, "raw_embeddings_persisted": False, "results": results, "representative_pairs": reps, "semantic_inputs_excluded": ["expression", "source_id", "pool_id", "group_id", "state_key"], "labels": "GT_PRIVILEGED_ORACLE from held-out train videos", "token_level_alignment_verified": False, "motion_language_decomposition": "not claimed", "elapsed_sec": time.time() - started}
    out.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
