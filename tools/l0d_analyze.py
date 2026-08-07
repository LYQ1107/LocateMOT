#!/usr/bin/env python
"""Stage L0-D: dump pair features, baseline predictions, and stratified diagnosis.

Subcommands:
  dump       build per-pair feature dump for calibration/heldout (cached .pt)
  predict    compute B0/B1/B2-box-end/B3/B4 assignments from the dump
  diagnose   full stratified diagnosis + temporal-gap confounding + hard flag
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.data.pair_dataset import (  # noqa: E402
    PairDataset,
    EMBED_DIM,
    _token_feature,
    _crop_feature,
    _count,
)
from locatemot.data.token_cache import read_frame_cache  # noqa: E402
from locatemot.evaluation.assignment import assign_tracks_to_candidates  # noqa: E402
from locatemot.evaluation.pair_metrics import evaluate_assignments  # noqa: E402
from locatemot.models.track_decoder.features import category_hash_embedding  # noqa: E402
from locatemot.models.track_decoder.model import PairwiseModel, TrackDecoderModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L0C/cache"
MANIFEST = os.path.join(ROOT, "outputs/l0_c/pair_manifest.jsonl")
OUT = os.path.join(ROOT, "outputs/l0_d")


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_pair(rec):
    ref = read_frame_cache(CACHE_ROOT, rec["reference_token_id"])
    cur = read_frame_cache(CACHE_ROOT, rec["current_token_id"])
    if ref is None or cur is None:
        return None
    ref_f, ref_m = ref["features"], ref["meta"]
    cur_f, cur_m = cur["features"], cur["meta"]
    n_full = min(cur_m.get("candidate_count", 0), 128)
    n_model = min(n_full, 32)
    item = None
    if n_full > 0:
        ds = PairDataset([rec], CACHE_ROOT)
        item = ds[0]
        M = int(item["ref_mask"].sum())
        if M == 0:
            return None
        ref_boxes = item["ref_boxes"][:M].numpy()
        ref_geom = item["ref_geom"][:M].numpy()
        ref_gen = item["ref_gen"][:M].numpy()
        ref_pbd = item["ref_pbd"][:M].numpy()
        ref_pbd_be = item["ref_pbd_be"][:M].numpy()
        ref_region = item["ref_region"][:M].numpy()
        ref_cat = item["ref_cat"][:M].numpy()
        match_targets = item["match_targets"][:M].numpy()
        no_match_targets = item["no_match_targets"][:M].numpy()
        candidate_missing = item["candidate_missing"][:M].numpy()
        visible = item["visible"][:M].numpy()
        gt_iou = item["gt_iou"][:M].numpy()
    else:
        # zero-candidate pairs: PairDataset returns an all-masked empty sample,
        # so build reference features directly with the same helpers.
        entries = []
        for t in rec["reference_targets"][:8]:
            tid = t["track_id"]
            c_idx = t.get("reference_candidate_index")
            if c_idx is not None and c_idx < _count(ref_f):
                entries.append(_token_feature(ref_f, int(c_idx)))
            else:
                entries.append(_crop_feature(ref_f, ref_m, tid))
        M = len(entries)
        if M == 0:
            return None
        ref_boxes = np.asarray([e["box"] for e in entries], dtype=np.float32)
        ref_geom = torch.stack([e["geom"] for e in entries]).numpy()
        ref_gen = np.asarray([e["gen"] for e in entries], dtype=np.float32)
        ref_pbd = torch.stack([e["pbd"] for e in entries]).numpy()
        ref_pbd_be = torch.stack([e["pbd_be"] for e in entries]).numpy()
        ref_region = torch.stack([e["region"] for e in entries]).numpy()
        cat = category_hash_embedding(rec["protocol"], EMBED_DIM).numpy()
        ref_cat = np.repeat(cat[None, :], M, axis=0)
        match_targets = np.full(M, -1, dtype=np.int64)
        no_match_targets = np.asarray(
            [1.0 if t["track_id"] in rec["no_match_targets"] else 0.0 for t in rec["reference_targets"][:M]],
            dtype=np.float32)
        candidate_missing = np.asarray(
            [t["track_id"] in rec["candidate_missing_targets"] for t in rec["reference_targets"][:M]],
            dtype=bool)
        visible = ~candidate_missing
        gt_iou = np.zeros(M, dtype=np.float32)
    # full candidate arrays (baselines use all candidates; models cap at 32)
    def full_arr(key, dim, fallback_name=None):
        src = cur_f.get(fallback_name or key)
        rows = []
        for j in range(n_full):
            if src is not None and j < src.shape[0]:
                rows.append(torch.from_numpy(src[j].float().numpy()))
            else:
                rows.append(torch.zeros(dim))
        return torch.stack(rows) if rows else torch.zeros((0, dim))
    full_pbd = full_arr("pbd_coord_mean_last", 2048)
    full_region = full_arr("region", 4608)
    full_geom = full_arr("geometry", 5)
    full_gen = torch.zeros(n_full)
    if "gen_score" in cur_f:
        g = cur_f["gen_score"].float()
        full_gen[: min(n_full, g.shape[0])] = g[: min(n_full, g.shape[0])]
    full_boxes = torch.zeros((n_full, 4))
    if "boxes" in cur_f:
        nb = min(n_full, cur_f["boxes"].shape[0])
        full_boxes[:nb] = cur_f["boxes"][:nb].float()
    iou = np.zeros((M, n_full), dtype=np.float32)
    cos_pbd = np.zeros((M, n_full), dtype=np.float32)
    cos_boxend = np.zeros((M, n_full), dtype=np.float32)
    cos_region = np.zeros((M, n_full), dtype=np.float32)
    for i in range(M):
        for j in range(n_full):
            iou[i, j] = _iou(ref_boxes[i].tolist(), full_boxes[j].tolist())
            cos_pbd[i, j] = _cos(ref_pbd[i], full_pbd[j].numpy())
            cos_region[i, j] = _cos(ref_region[i], full_region[j].numpy())
    # box-end cosine needs pbd_box_end_last; recompute from cache
    ref_be = ref_f.get("pbd_box_end_last")
    cur_be = cur_f.get("pbd_box_end_last")
    for i, t in enumerate(rec["reference_targets"][:M]):
        c_idx = t.get("reference_candidate_index")
        rv = None
        if c_idx is not None and ref_be is not None and c_idx < ref_be.shape[0]:
            rv = ref_be[c_idx].float().numpy()
        if rv is None:
            continue
        for j in range(min(cur_be.shape[0], n_full) if cur_be is not None else 0):
            cos_boxend[i, j] = _cos(rv, cur_be[j].float().numpy())
    cur_gt = {}
    cur_cache = read_frame_cache(CACHE_ROOT, rec["current_token_id"])
    if cur_cache:
        cur_gt = cur_cache["meta"].get("gt_boxes", {})
    return {
        "rec": {k: rec[k] for k in (
            "split", "dataset", "video_id", "protocol", "temporal_gap",
            "reference_target_count", "current_candidate_count", "visible_positives",
            "true_no_match_count", "candidate_missing_count", "current_token_id",
        )},
        "reference_targets": rec["reference_targets"][:M],
        "assignment_targets": rec["assignment_targets"],
        "no_match_targets": rec["no_match_targets"],
        "candidate_missing_targets": rec["candidate_missing_targets"],
        "ref_boxes": ref_boxes, "cur_boxes": full_boxes.numpy(),
        "ref_geom": ref_geom, "cur_geom": full_geom.numpy(),
        "ref_gen": ref_gen, "cur_gen": full_gen.numpy(),
        "ref_pbd": ref_pbd, "cur_pbd": full_pbd.numpy(),
        "ref_pbd_be": ref_pbd_be, "cur_pbd_be": (item["cur_pbd_be"][:n_model].numpy()
                                                 if item is not None and n_model
                                                 else full_pbd[:0].numpy()),
        "ref_region": ref_region, "cur_region": full_region.numpy(),
        "ref_cat": ref_cat,
        "cur_cat": item["cur_cat"][:n_model].numpy() if item is not None else np.zeros((n_model, EMBED_DIM), dtype=np.float32),
        "match_targets": match_targets,
        "no_match_targets": rec["no_match_targets"],
        "candidate_missing": candidate_missing,
        "visible": visible,
        "labels": item["labels"][:M, :n_model].numpy() if n_model else np.zeros((M, 0), dtype=np.float32),
        "n_model": n_model,
        "gt_iou": gt_iou,
        "gap": float(rec["temporal_gap"]),
        "iou": iou, "cos_pbd": cos_pbd, "cos_boxend": cos_boxend, "cos_region": cos_region,
        "cur_gt": cur_gt,
    }


def dump(args):
    records = [json.loads(l) for l in open(MANIFEST)]
    os.makedirs(OUT, exist_ok=True)
    for split in ("calibration", "heldout"):
        out_path = os.path.join(OUT, f"pairs_{split}.pt")
        if args.force or not os.path.exists(out_path):
            items = []
            recs = [r for r in records if r["split"] == split]
            for idx, rec in enumerate(recs):
                p = build_pair(rec)
                if p is not None:
                    items.append(p)
                if (idx + 1) % 200 == 0:
                    print(f"[dump] {split} {idx+1}/{len(recs)}", flush=True)
            torch.save(items, out_path)
            print(f"[dump] saved {len(items)} pairs -> {out_path}")


def _load_pairs(split):
    return torch.load(os.path.join(OUT, f"pairs_{split}.pt"), map_location="cpu", weights_only=False)


def _records_from_pairs(pairs):
    out = []
    for p in pairs:
        rec = dict(p["rec"])
        M = len(p["ref_boxes"])
        rec["reference_targets"] = p["reference_targets"]
        rec["assignment_targets"] = p["assignment_targets"]
        rec["no_match_targets"] = p["no_match_targets"]
        rec["candidate_missing_targets"] = p["candidate_missing_targets"]
        rec["candidate_boxes"] = [b.tolist() for b in p["cur_boxes"]]
        out.append(rec)
    return out


def _cur_gt(pairs):
    return {p["rec"]["current_token_id"]: p["cur_gt"] for p in pairs}


def _batch_tensors(pairs, start, end):
    chunk = pairs[start:end]
    maxM = max(len(p["ref_boxes"]) for p in chunk)
    maxN = max(max(p["n_model"] for p in chunk), 1)

    def pad(t, n, dim0=True):
        cur = t.shape[0]
        if cur >= n:
            return t[:n]
        if t.dim() == 1:
            return torch.cat([t, torch.zeros(n - cur, dtype=t.dtype)])
        return torch.cat([t, torch.zeros((n - cur,) + tuple(t.shape[1:]), dtype=t.dtype)])

    batch = {}
    for p in pairs[start:end]:
        empty = p["n_model"] == 0
        M = 1 if empty else p["ref_boxes"].shape[0]
        N = 1 if empty else p["n_model"]
        ref_pbd = np.zeros((M, 2048), dtype=np.float32) if empty else p["ref_pbd"]
        ref_pbd_be = np.zeros((M, 2048), dtype=np.float32) if empty else p["ref_pbd_be"]
        ref_region = np.zeros((M, 4608), dtype=np.float32) if empty else p["ref_region"]
        ref_geom = np.zeros((M, 5), dtype=np.float32) if empty else p["ref_geom"]
        ref_gen = np.zeros(M, dtype=np.float32) if empty else p["ref_gen"]
        ref_cat = np.zeros((M, EMBED_DIM), dtype=np.float32) if empty else p["ref_cat"]
        ref_boxes = np.zeros((1, 4), dtype=np.float32) if empty else p["ref_boxes"]
        cur_pbd = np.zeros((N, 2048), dtype=np.float32) if empty else p["cur_pbd"][:N]
        cur_pbd_be = np.zeros((N, 2048), dtype=np.float32) if empty else p["cur_pbd_be"][:N]
        cur_region = np.zeros((N, 4608), dtype=np.float32) if empty else p["cur_region"][:N]
        cur_geom = np.zeros((N, 5), dtype=np.float32) if empty else p["cur_geom"][:N]
        cur_gen = np.zeros(N, dtype=np.float32) if empty else p["cur_gen"][:N]
        cur_cat = np.zeros((N, EMBED_DIM), dtype=np.float32) if empty else p["cur_cat"][:N]
        cur_boxes = np.zeros((N, 4), dtype=np.float32) if empty else p["cur_boxes"][:N]
        ref_mask = torch.zeros(M, dtype=torch.bool) if empty else torch.ones(M, dtype=torch.bool)
        cur_mask = torch.zeros(N, dtype=torch.bool) if empty else torch.ones(N, dtype=torch.bool)
        for key, val in (
            ("ref_pbd", ref_pbd), ("ref_pbd_be", ref_pbd_be), ("ref_region", ref_region),
            ("ref_geom", ref_geom), ("ref_gen", ref_gen),
            ("ref_cat", ref_cat), ("ref_boxes", ref_boxes),
        ):
            batch.setdefault(key, []).append(pad(torch.from_numpy(val), maxM))
        batch.setdefault("ref_mask", []).append(
            torch.cat([ref_mask, torch.zeros(maxM - M, dtype=torch.bool)]))
        for key, val in (
            ("cur_pbd", cur_pbd), ("cur_pbd_be", cur_pbd_be), ("cur_region", cur_region),
            ("cur_geom", cur_geom), ("cur_gen", cur_gen),
            ("cur_cat", cur_cat), ("cur_boxes", cur_boxes),
        ):
            batch.setdefault(key, []).append(pad(torch.from_numpy(val), maxN))
        batch.setdefault("cur_mask", []).append(
            torch.cat([cur_mask, torch.zeros(maxN - N, dtype=torch.bool)]))
        lab = torch.zeros((maxM, maxN))
        if not empty:
            lab[:M, :N] = torch.from_numpy(p["labels"])
        batch.setdefault("labels", []).append(lab)
        batch.setdefault("gap", []).append(torch.tensor([p["gap"]], dtype=torch.float32))
    return {k: torch.stack(v) for k, v in batch.items()}


def _model_assignments(model, pairs, device="cpu", batch_size=32, pairwise=False, official_style=True,
                       nm_sigmoid=False):
    model.eval()
    model.to(device)
    all_preds = []
    with torch.no_grad():
        for s in range(0, len(pairs), batch_size):
            batch = _batch_tensors(pairs, s, s + batch_size)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = model(batch)
            if pairwise:
                B = len(batch["ref_boxes"])
                maxM = batch["ref_mask"].shape[1]
                maxN = batch["cur_mask"].shape[1]
                flat_match = pred["match_logits"].detach().cpu().reshape(B, maxM, maxN).numpy()
                flat_nm = pred["no_match_logits"].detach().cpu().reshape(B, maxM, maxN).numpy()
                for b in range(B):
                    if pairs[s + b]["n_model"] == 0:
                        all_preds.append([])
                        continue
                    M = len(pairs[s + b]["ref_boxes"])
                    N = pairs[s + b]["n_model"]
                    if official_style:
                        # reproduce official L0-C behavior: full batch-padded width
                        # is passed to the assignment (padded columns are assignable)
                        mm = flat_match[b, :M, :maxN]
                        nn = flat_nm[b, :M, :maxN].mean(axis=1)
                    else:
                        mm = flat_match[b, :M, :N]
                        nn = flat_nm[b, :M, :N].mean(axis=1)
                    if nm_sigmoid:
                        nn = 1.0 / (1.0 + np.exp(-nn))
                    all_preds.append(assign_tracks_to_candidates(mm, nn))
            else:
                match = pred["match_logits"]
                nm = pred["no_match_logits"]
                maxN = batch["cur_mask"].shape[1]
                for b in range(match.shape[0]):
                    if pairs[s + b]["n_model"] == 0:
                        all_preds.append([])
                        continue
                    M = len(pairs[s + b]["ref_boxes"])
                    N = pairs[s + b]["n_model"]
                    nn = nm[b, :M].cpu().numpy()
                    if nm_sigmoid:
                        nn = 1.0 / (1.0 + np.exp(-nn))
                    all_preds.append(assign_tracks_to_candidates(
                        match[b, :M, :(maxN if official_style else N)].cpu().numpy(),
                        nn,
                    ))
    return all_preds


def predict(args):
    pairs = _load_pairs("heldout")
    recs = _records_from_pairs(pairs)
    cur_gt = _cur_gt(pairs)
    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    results = {}
    results_clean = {}
    # B0 IoU, threshold 0.05 (frozen from L0-C calibration)
    b0 = [assign_tracks_to_candidates(p["iou"], np.full(p["iou"].shape[0], 0.05)) for p in pairs]
    results["B0_IoU"] = b0
    results_clean["B0_IoU"] = b0
    # B1 region cosine threshold 0.10
    b1 = [assign_tracks_to_candidates(p["cos_region"], np.full(p["cos_region"].shape[0], 0.10)) for p in pairs]
    results["B1_RegionCos"] = b1
    results_clean["B1_RegionCos"] = b1
    # B2 PBD coordinate threshold 0.05
    b2 = [assign_tracks_to_candidates(p["cos_pbd"], np.full(p["cos_pbd"].shape[0], 0.05)) for p in pairs]
    results["B2_PBDCos"] = b2
    results_clean["B2_PBDCos"] = b2
    # B2 box-end threshold 0.15 (official L0-C used PairDataset cap at 32)
    b2be = [assign_tracks_to_candidates(p["cos_boxend"][:, :p["n_model"]],
                                        np.full(p["cos_boxend"].shape[0], 0.15)) for p in pairs]
    results["B2_PBDBoxEnd"] = b2be
    results_clean["B2_PBDBoxEnd"] = b2be
    if os.path.exists(os.path.join(ROOT, "outputs/l0_c/checkpoints/b3/best.pt")):
        model = PairwiseModel()
        ck = torch.load(os.path.join(ROOT, "outputs/l0_c/checkpoints/b3/best.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        results["B3_PairwiseMLP"] = _model_assignments(model, pairs, device, batch_size=16, pairwise=True)
        results_clean["B3_PairwiseMLP"] = _model_assignments(
            model, pairs, device, batch_size=16, pairwise=True, official_style=False)
    if os.path.exists(os.path.join(ROOT, "outputs/l0_c/checkpoints/b4/best.pt")):
        model = TrackDecoderModel()
        ck = torch.load(os.path.join(ROOT, "outputs/l0_c/checkpoints/b4/best.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        results["B4_TrackDecoder"] = _model_assignments(model, pairs, device, batch_size=16)
        results_clean["B4_TrackDecoder"] = _model_assignments(
            model, pairs, device, batch_size=16, official_style=False)
    os.makedirs(os.path.join(OUT, "diagnosis"), exist_ok=True)
    torch.save(results, os.path.join(OUT, "baseline_assignments.pt"))
    torch.save(results_clean, os.path.join(OUT, "baseline_assignments_clean.pt"))
    # summary table
    rows = []
    for name, preds in results.items():
        m = evaluate_assignments(preds, recs, cur_gt)
        rows.append({"model": name, **{k: round(v, 4) for k, v in m.items() if isinstance(v, float)}})
    with open(os.path.join(OUT, "diagnosis/baseline_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2))


def _bucket_gap(gap):
    if gap <= 4:
        return "1-4"
    if gap <= 16:
        return "5-16"
    if gap <= 64:
        return "17-64"
    return ">64"


def _bucket_targets(m):
    if m == 1:
        return "1"
    if m <= 4:
        return "2-4"
    return "5-8"


def _bucket_density(n):
    if n <= 5:
        return "0-5"
    if n <= 15:
        return "6-15"
    return ">15"


def hard_flag(p):
    """Prediction-side hard competition flag, frozen definition.

    Uses only per-candidate IoU / PBD cosine margins, candidate density,
    reference count and shared plausible candidates (all available at inference).
    """
    iou = p["iou"]
    cos = p["cos_boxend"]
    M, N = iou.shape
    if M >= 5 or N >= 6:
        return True
    for i in range(M):
        ious = iou[i]
        coses = cos[i]
        if N >= 2:
            order = np.argsort(ious)[::-1]
            if ious[order[0]] - ious[order[1]] < 0.10:
                return True
            order_c = np.argsort(coses)[::-1]
            if coses[order_c[0]] - coses[order_c[1]] < 0.05:
                return True
        if (ious >= 0.30).sum() >= 3:
            return True
    for j in range(N):
        if (iou[:, j] >= 0.30).sum() >= 2:
            return True
    return False


def apply_iou_floor(pred_assignments, pairs, floor: float = 0.05):
    """Inference-time strong-prior floor: if the assigned candidate's IoU is
    below B0's calibration threshold, force NO_MATCH (keeps B0-level easy
    no-match recall; learned threshold may still raise the bar further)."""
    out = []
    for p, preds in zip(pairs, pred_assignments):
        new = []
        for ti, tag in preds:
            if tag.startswith("candidate:"):
                j = int(tag.split(":")[1])
                if j < p["iou"].shape[1] and p["iou"][ti, j] < floor:
                    new.append((ti, "NO_MATCH"))
                    continue
            new.append((ti, tag))
        out.append(new)
    return out


def diagnose(args):
    pairs = _load_pairs("heldout")
    recs = _records_from_pairs(pairs)
    cur_gt = _cur_gt(pairs)
    assigns = torch.load(os.path.join(OUT, "baseline_assignments.pt"), map_location="cpu", weights_only=False)
    os.makedirs(os.path.join(OUT, "diagnosis"), exist_ok=True)

    # 1. Stratified metrics for each baseline
    groups = defaultdict(list)
    for pi, p in enumerate(pairs):
        rec = recs[pi]
        keys = {
            "dataset": rec["dataset"].replace("_train", ""),
            "protocol": rec["protocol"],
            "gap": _bucket_gap(rec["temporal_gap"]),
            "target_count": _bucket_targets(rec["reference_target_count"]),
            "candidate_density": _bucket_density(rec["current_candidate_count"]),
            "hard_competition": "hard" if hard_flag(p) else "easy",
        }
        for k, v in keys.items():
            groups[(k, v)].append(pi)
    rows = []
    for model, preds in assigns.items():
        m = evaluate_assignments(preds, recs, cur_gt)
        rows.append({"model": model, "group": "all", "value": "all", "samples": len(pairs),
                     "e2e": round(m["e2e_accuracy"], 4), "conditional": round(m["conditional_accuracy"], 4),
                     "no_match_f1": round(m["no_match_f1"], 4), "id_f1": round(m["id_f1"], 4)})
        for (dim, val), idxs in groups.items():
            sub_preds = [preds[i] for i in idxs]
            sub_recs = [recs[i] for i in idxs]
            mm = evaluate_assignments(sub_preds, sub_recs, cur_gt)
            rows.append({"model": model, "group": dim, "value": val, "samples": len(idxs),
                         "e2e": round(mm["e2e_accuracy"], 4), "conditional": round(mm["conditional_accuracy"], 4),
                         "no_match_f1": round(mm["no_match_f1"], 4), "id_f1": round(mm["id_f1"], 4)})
    with open(os.path.join(OUT, "diagnosis/stratified_baselines.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 2. Temporal gap composition
    gap_rows = []
    for g in ("1-4", "5-16", "17-64", ">64"):
        idxs = [i for i, p in enumerate(pairs) if _bucket_gap(p["rec"]["temporal_gap"]) == g]
        if not idxs:
            continue
        st = {"gap": g, "samples": len(idxs)}
        ds = defaultdict(int)
        proto = defaultdict(int)
        tc = defaultdict(int)
        cand = []
        missing = nm_true = 0
        scales = []
        for i in idxs:
            p = pairs[i]
            rec = p["rec"]
            ds[rec["dataset"].replace("_train", "")] += 1
            proto[rec["protocol"]] += 1
            tc[_bucket_targets(rec["reference_target_count"])] += 1
            cand.append(rec["current_candidate_count"])
            missing += int(rec["candidate_missing_count"])
            nm_true += int(rec["true_no_match_count"])
            scales.extend([(b[2]-b[0])*(b[3]-b[1]) for b in p["ref_boxes"]])
        st["dataset"] = dict(ds)
        st["protocol"] = dict(proto)
        st["target_count"] = dict(tc)
        st["candidate_mean"] = round(float(np.mean(cand)), 2)
        st["candidate_median"] = float(np.median(cand))
        st["candidate_missing_refs"] = missing
        st["true_no_match_refs"] = nm_true
        st["ref_box_area_mean"] = round(float(np.mean(scales)), 1) if scales else None
        for model in ("B0_IoU", "B2_PBDCos", "B2_PBDBoxEnd", "B4_TrackDecoder"):
            if model not in assigns:
                continue
            sub_preds = [assigns[model][i] for i in idxs]
            mm = evaluate_assignments(sub_preds, [recs[i] for i in idxs], cur_gt)
            st[f"{model}_cond"] = round(mm["conditional_accuracy"], 4)
            st[f"{model}_e2e"] = round(mm["e2e_accuracy"], 4)
        gap_rows.append(st)
    with open(os.path.join(OUT, "diagnosis/gap_composition.json"), "w") as f:
        json.dump(gap_rows, f, ensure_ascii=False, indent=2)

    # 3. hard subset distribution + frozen config
    hard_def = {
        "iou_margin_threshold": 0.10,
        "pbd_cos_margin_threshold": 0.05,
        "plausible_iou_threshold": 0.30,
        "density_hard_threshold": 6,
        "reference_count_hard_threshold": 5,
        "shared_plausible_iou_threshold": 0.30,
        "note": "Prediction-side only; frozen before B5/B6 evaluation.",
    }
    hard_def["sha256"] = hashlib.sha256(json.dumps(hard_def, sort_keys=True).encode()).hexdigest()[:12]
    os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)
    with open(os.path.join(ROOT, "configs/l0_d_hard_subset.json"), "w") as f:
        json.dump(hard_def, f, ensure_ascii=False, indent=2)
    hard_counts = {"hard": 0, "easy": 0}
    for p in pairs:
        hard_counts["hard" if hard_flag(p) else "easy"] += 1
    print(json.dumps({"hard_counts": hard_counts, "rows": len(rows)}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("dump")
    d.add_argument("--force", action="store_true")
    pr = sub.add_parser("predict")
    pr.add_argument("--gpu", type=int, default=-1)
    di = sub.add_parser("diagnose")
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args)
    elif args.cmd == "predict":
        predict(args)
    elif args.cmd == "diagnose":
        diagnose(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
