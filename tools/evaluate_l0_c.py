#!/usr/bin/env python
"""Stage L0-C: evaluate B0-B4 on calibration and held-out splits."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.data.collate import collate_track_batch  # noqa: E402
from locatemot.data.pair_dataset import PairDataset  # noqa: E402
from locatemot.evaluation.assignment import assign_tracks_to_candidates  # noqa: E402
from locatemot.evaluation.pair_metrics import evaluate_assignments  # noqa: E402
from locatemot.evaluation.stratified_metrics import stratify  # noqa: E402
from locatemot.models.track_decoder.inference import assign_batch  # noqa: E402
from locatemot.models.track_decoder.model import PairwiseModel, TrackDecoderModel  # noqa: E402


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _cos(a, b):
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.dot(a, b))


def score_baseline(records, cache_root, mode):
    scores = []
    ds = PairDataset(records, cache_root)
    for rec in records:
        item = ds[records.index(rec)] if False else None
    # build features directly from cache to avoid dataset randomness
    from locatemot.data.token_cache import read_frame_cache
    for rec in records:
        ref = read_frame_cache(cache_root, rec["reference_token_id"])
        cur = read_frame_cache(cache_root, rec["current_token_id"])
        if ref is None or cur is None:
            scores.append(None)
            continue
        ref_f, ref_m = ref["features"], ref["meta"]
        cur_f, cur_m = cur["features"], cur["meta"]
        n = cur_m.get("candidate_count", 0)
        ref_boxes, cur_boxes = [], []
        ref_vecs, cur_vecs = [], []
        for t in rec["reference_targets"]:
            c_idx = t.get("reference_candidate_index")
            ref_boxes_arr = ref_f.get("boxes")
            if c_idx is not None and ref_boxes_arr is not None and c_idx < len(ref_boxes_arr):
                rb = ref_boxes_arr[c_idx].tolist()
                if mode == "b1":
                    rv = ref_f["region"][c_idx].float().numpy()
                elif mode == "b2":
                    rv = ref_f["pbd_coord_mean_last"][c_idx].float().numpy()
                else:
                    rv = None
            else:
                rb = ref_m.get("gt_boxes", {}).get(str(t["track_id"]), [0, 0, 0, 0])
                idx = [str(x) for x in ref_m.get("crop_object_ids", [])]
                if str(t["track_id"]) in idx and "crop_region" in ref_f:
                    if mode == "b1":
                        rv = ref_f["crop_region"][idx.index(str(t["track_id"]))].float().numpy()
                    else:
                        rv = np.zeros(2048, dtype=np.float32)
                else:
                    rv = None
            ref_boxes.append(rb)
            ref_vecs.append(rv)
        cur_boxes_arr = cur_f.get("boxes")
        cur_region_arr = cur_f.get("region")
        cur_pbd_arr = cur_f.get("pbd_coord_mean_last")
        for j in range(n):
            cur_boxes.append(cur_boxes_arr[j].tolist())
            if mode == "b1":
                cur_vecs.append(cur_region_arr[j].float().numpy())
            elif mode == "b2":
                cur_vecs.append(cur_pbd_arr[j].float().numpy())
            else:
                cur_vecs.append(None)
        M, N = len(ref_boxes), len(cur_boxes)
        sc = np.zeros((M, N), dtype=np.float32)
        for i in range(M):
            for j in range(N):
                if mode == "b0":
                    sc[i, j] = _iou(ref_boxes[i], cur_boxes[j])
                elif ref_vecs[i] is not None and cur_vecs[j] is not None:
                    sc[i, j] = _cos(ref_vecs[i], cur_vecs[j])
        scores.append(sc)
    return scores


def assign_with_threshold(scores, thr):
    out = []
    for sc in scores:
        if sc is None:
            out.append([(0, "NO_MATCH")])
            continue
        out.append(assign_tracks_to_candidates(sc, np.full(sc.shape[0], thr)))
    return out


def model_predictions(model, records, cache_root, device, pairwise=False):
    ds = PairDataset(records, cache_root)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_track_batch)
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = model(batch)
            B, M, N = batch["labels"].shape
            match = pred["match_logits"].reshape(B, M, N)
            nm = pred["no_match_logits"].reshape(B, M, N).mean(dim=2) if pairwise else pred["no_match_logits"]
            for b in range(B):
                valid_rows = batch["ref_mask"][b].cpu().numpy()
                orig_idx = list(range(M))
                valid_idx = [i for i in orig_idx if valid_rows[i]]
                if not valid_idx:
                    all_preds.append([])
                    continue
                sub = assign_tracks_to_candidates(
                    match[b][valid_idx].cpu().numpy(),
                    nm[b][valid_idx].cpu().numpy(),
                )
                all_preds.append([(valid_idx[ti], tag) for ti, tag in sub])
    return all_preds


def candidate_recall(cache_root, roots):
    from locatemot.data.token_cache import cache_key
    import glob
    metas = sorted(glob.glob(os.path.join(cache_root, "*", "*", "*", "*.meta.json")))
    per_proto = {}
    for mp in metas:
        meta = json.load(open(mp))
        if meta.get("candidate_count", 0) == 0:
            continue
        proto = meta["protocol"]
        per_proto.setdefault(proto, {"gt": 0, "recall": {0.3: 0, 0.5: 0, 0.7: 0}})
        st = per_proto[proto]
        feat_path = mp.replace(".meta.json", ".safetensors")
        from safetensors.torch import load_file
        feats = load_file(feat_path)
        boxes = feats["boxes"].cpu().numpy()
        for oid, gtb in meta.get("gt_boxes", {}).items():
            st["gt"] += 1
            best = max((_iou(box.tolist(), gtb) for box in boxes), default=0.0)
            for t in (0.3, 0.5, 0.7):
                if best >= t:
                    st["recall"][t] += 1
    out = {}
    for proto, st in per_proto.items():
        out[proto] = {f"recall@{t}": round(st["recall"][t] / max(1, st["gt"]), 4) for t in (0.3, 0.5, 0.7)}
        out[proto]["gt_objects"] = st["gt"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/l0_c/pair_manifest.jsonl")
    ap.add_argument("--cache-root", default="/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L0C/cache")
    ap.add_argument("--out", default="outputs/l0_c")
    ap.add_argument("--b3-ckpt", default="outputs/l0_c/checkpoints/b3/best.pt")
    ap.add_argument("--b4-ckpt", default="outputs/l0_c/checkpoints/b4/best.pt")
    ap.add_argument("--b4-current-ckpt", default="")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    records = [json.loads(l) for l in open(args.manifest)]
    calib = [r for r in records if r["split"] == "calibration"]
    heldout = [r for r in records if r["split"] == "heldout"]

    # gt boxes per current token for localization
    from locatemot.data.token_cache import read_frame_cache
    cur_gt = {}
    for rec in calib + heldout:
        cur = read_frame_cache(args.cache_root, rec["current_token_id"])
        if cur:
            cur_gt[rec["current_token_id"]] = cur["meta"].get("gt_boxes", {})
            boxes = cur["features"].get("boxes")
            rec["candidate_boxes"] = [b.tolist() for b in boxes] if boxes is not None else []

    results = []
    # B0-B2 with threshold selection on calibration
    for mode, name in [("b0", "B0_IoU"), ("b1", "B1_RegionCos"), ("b2", "B2_PBDCos")]:
        cal_scores = score_baseline(calib, args.cache_root, mode)
        held_scores = score_baseline(heldout, args.cache_root, mode)
        best_thr, best_acc = 0.0, -1.0
        for thr in np.arange(0.05, 0.65, 0.05):
            preds = assign_with_threshold(cal_scores, float(thr))
            m = evaluate_assignments(preds, calib, cur_gt)
            if m["conditional_accuracy"] > best_acc:
                best_acc = m["conditional_accuracy"]
                best_thr = float(thr)
        preds = assign_with_threshold(held_scores, best_thr)
        m = evaluate_assignments(preds, heldout, cur_gt)
        results.append({"model": name, "split": "heldout", "threshold": best_thr, **m})
        # box-end ablation for B2
    results.append(_ablation_b2(heldout, calib, args.cache_root, cur_gt))

    device = "cuda"
    if os.path.exists(args.b3_ckpt):
        model = PairwiseModel()
        ckpt = torch.load(args.b3_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        preds = model_predictions(model, heldout, args.cache_root, device, pairwise=True)
        m = evaluate_assignments(preds, heldout, cur_gt)
        results.append({"model": "B3_PairwiseMLP", "split": "heldout", "threshold": "-", **m})
    if os.path.exists(args.b4_ckpt):
        model = TrackDecoderModel()
        ckpt = torch.load(args.b4_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        preds = model_predictions(model, heldout, args.cache_root, device)
        m = evaluate_assignments(preds, heldout, cur_gt)
        results.append({"model": "B4_TrackDecoder", "split": "heldout", "threshold": "-", **m})
    if args.b4_current_ckpt and os.path.exists(args.b4_current_ckpt):
        model = TrackDecoderModel(query_direction="current_query")
        ckpt = torch.load(args.b4_current_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        preds = model_predictions(model, heldout, args.cache_root, device)
        m = evaluate_assignments(preds, heldout, cur_gt)
        results.append({"model": "B4_current_query", "split": "heldout", "threshold": "-", **m})

    os.makedirs(args.out, exist_ok=True)
    keys = list(results[0].keys())
    with open(os.path.join(args.out, "baseline_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    # stratified for best model (B4 if exists else B2)
    strat_model = "B4_TrackDecoder" if os.path.exists(args.b4_ckpt) else "B2_PBDCos"
    rows = stratify(preds if strat_model == "B4_TrackDecoder" else assign_with_threshold(held_scores, best_thr),
                    heldout, cur_gt, args.out)
    recall = candidate_recall(args.cache_root, None)
    with open(os.path.join(args.out, "candidate_recall.json"), "w") as f:
        json.dump(recall, f, ensure_ascii=False, indent=2)
    print(json.dumps({"results": results, "candidate_recall": recall}, ensure_ascii=False, indent=2))


def _ablation_b2(heldout, calib, cache_root, cur_gt):
    # box-end feature ablation
    ds = PairDataset(heldout, cache_root)
    preds = []
    for rec in heldout:
        item = ds[heldout.index(rec)]
        sc = np.zeros((item["ref_pbd"].shape[0], item["cur_pbd"].shape[0]), dtype=np.float32)
        # fallback: use box-end from cache
        from locatemot.data.token_cache import read_frame_cache
        ref = read_frame_cache(cache_root, rec["reference_token_id"])
        cur = read_frame_cache(cache_root, rec["current_token_id"])
        if ref and cur:
            rf, cf = ref["features"], cur["features"]
            for i, t in enumerate(rec["reference_targets"]):
                c_idx = t.get("reference_candidate_index")
                rv = None
                if c_idx is not None and "pbd_box_end_last" in rf and c_idx < len(rf["pbd_box_end_last"]):
                    rv = rf["pbd_box_end_last"][c_idx].float().numpy()
                box_end_arr = cf.get("pbd_box_end_last")
                if box_end_arr is None:
                    continue
                for j in range(min(box_end_arr.shape[0], sc.shape[1])):
                    cv = box_end_arr[j].float().numpy()
                    sc[i, j] = _cos(rv, cv) if rv is not None else 0.0
        preds.append(assign_tracks_to_candidates(sc, np.full(sc.shape[0], 0.15)))
    m = evaluate_assignments(preds, heldout, cur_gt)
    return {"model": "B2_ablation_PBDBoxEnd", "split": "heldout", "threshold": "0.15", **m}


if __name__ == "__main__":
    main()
