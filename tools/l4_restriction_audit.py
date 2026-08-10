"""Stage L4: U0 Track-All-Then-Filter (P0) vs Pre-Filter (P1) audit.

For a domain + specification (ALL / category / instance subset), runs the
frozen U0 tracker on (a) the full candidate stream (P0, then filtered to the
spec) and (b) the spec-restricted candidate stream (P1), then measures on
common candidates:
  - optimal-ID-aligned association disagreement (permutation invariant);
  - per-GT identity drift;
  - full-sequence windowed AssA/IDF1/IDSW per view (TrackEval-consistent).

NOTE: category/instance masks use GT category/id -> PRIVILEGED_SPEC_ORACLE
diagnostic, not a realistic user prompt. Reported separately.

Usage:
  python tools/l4_restriction_audit.py --domain bdd100k_train \
      --manifest outputs/l1_c/fixed_candidate_manifest/bdd100k_train.jsonl \
      --specs ALL,cat:car,cat:pedestrian,cat:truck,cat:bus \
      --gpu 9 --out outputs/l4/audit_bdd.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.data.token_cache import read_frame_cache  # noqa: E402
from locatemot.models.l1d_association import L1DAssociator  # noqa: E402
from locatemot.models.l4_spec_eq import L4SpecEqAssociator  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402
from tools.run_l2_oracle import windowed_metrics  # noqa: E402


def build_candidates(entry):
    root = entry["cache_root"]
    key = entry.get("cache_key") or (
        f"{entry['dataset']}/{entry['video_id']}/{int(entry['frame']):05d}/{entry['protocol']}")
    fr = read_frame_cache(root, key)
    if fr is None:
        return [], entry.get("image_size", [1280, 720])
    feats = fr["features"]
    boxes = np.asarray(feats.get("boxes", np.zeros((0, 4))), dtype=np.float64)
    n = len(boxes)
    cands = []
    for i in range(n):
        f = {
            "pbd": np.asarray(feats["pbd_coord_mean_last"][i], dtype=np.float32)
            if "pbd_coord_mean_last" in feats and len(feats["pbd_coord_mean_last"]) > i
            else np.zeros(2048, np.float32),
            "pbd_be": np.asarray(feats["pbd_box_end_last"][i], dtype=np.float32)
            if "pbd_box_end_last" in feats and len(feats["pbd_box_end_last"]) > i
            else np.zeros(2048, np.float32),
            "region": np.asarray(feats["region"][i], dtype=np.float32)
            if "region" in feats and len(feats["region"]) > i else np.zeros(4608, np.float32),
            "geom": np.asarray(feats["geometry"][i], dtype=np.float32)
            if "geometry" in feats and len(feats["geometry"]) > i
            else np.zeros(5, np.float32),
            "gen": float(feats["gen_score"][i]) if "gen_score" in feats
            and len(feats["gen_score"]) > i else 0.0,
        }
        cands.append({"box": boxes[i], "features": f, "index": i})
    return cands, entry.get("image_size", [1280, 720])


def _category(entry, gid):
    cats = entry.get("gt_categories") or {}
    if gid in cats:
        return cats[gid]
    # single-class domains (DanceTrack/MOT) have no category map.
    return "person"


def spec_mask(entry, spec):
    """Return (keep_idx_list, gt_boxes_filtered, candidate_gt_id)."""
    n = int(entry["candidate_count"])
    cand_gt = [None] * n
    for gid, m in entry.get("matched", {}).items():
        ci = int(m["candidate"])
        if 0 <= ci < n:
            cand_gt[ci] = gid
    if spec == "ALL":
        keep = list(range(n))
    elif spec.startswith("cat:"):
        c = spec.split(":", 1)[1]
        keep = [i for i in range(n) if cand_gt[i] is not None
                and _category(entry, cand_gt[i]) == c]
    elif spec.startswith("inst:"):
        gids = set(spec.split(":", 1)[1].split(","))
        keep = [i for i in range(n) if cand_gt[i] in gids]
    else:
        raise ValueError(spec)
    gt_boxes = {}
    for gid, box in entry.get("gt_boxes", {}).items():
        if spec == "ALL":
            gt_boxes[gid] = box
        elif spec.startswith("cat:"):
            if _category(entry, gid) == spec.split(":", 1)[1]:
                gt_boxes[gid] = box
        elif spec.startswith("inst:"):
            if gid in set(spec.split(":", 1)[1].split(",")):
                gt_boxes[gid] = box
    return keep, gt_boxes, cand_gt


def run_tracker(model, device, entries, spec, max_frames=None):
    tracker = OnlineTracker(variant="L1D", l1d=model, device=str(device),
                            output_all_candidates=True,
                            spec_idx=_spec_idx(spec))
    tracker.l1d_weights = (0.4, 0.2, 0.4)
    tracker.l1d_threshold = 0.25
    tracker.l1d_delta_scale = 0.3
    tracker.l1d_rel_threshold = 0.0
    rows = []  # (frame, orig_cand_idx, tid, box, gt_id, gt_cat)
    for ei, entry in enumerate(entries):
        if max_frames and ei >= max_frames:
            break
        all_cands, image_size = build_candidates(entry)
        keep, gt_boxes, cand_gt = spec_mask(entry, spec)
        tracker.image_size = image_size
        restricted = [all_cands[i] for i in keep]
        outputs = tracker.process_frame(int(entry["frame"]), restricted)
        frame = int(entry["frame"])
        for o, ci in zip(outputs, keep):
            gid = cand_gt[ci]
            rows.append((frame, ci, o["track_id"], o["box"].copy(),
                         gid, _category(entry, gid)))
    return rows


def _spec_idx(spec):
    if spec == "ALL":
        return 0
    if spec.startswith("cat:"):
        return 1
    if spec.startswith("inst:"):
        return 2
    return 0


def align_ids(pairs):
    """pairs: list of (frame, tidA, tidB, gid, cat).
    Returns disagreement rate after optimal ID mapping (Hungarian on counts)."""
    from scipy.optimize import linear_sum_assignment
    ids_a = sorted({p[1] for p in pairs})
    ids_b = sorted({p[2] for p in pairs})
    ia = {v: i for i, v in enumerate(ids_a)}
    ib = {v: i for i, v in enumerate(ids_b)}
    count = np.zeros((len(ids_a), len(ids_b)))
    for p in pairs:
        count[ia[p[1]], ib[p[2]]] += 1
    rows, cols = linear_sum_assignment(-count)
    mapping = {ids_a[r]: ids_b[c] for r, c in zip(rows, cols)}
    agree = sum(1 for p in pairs if mapping.get(p[1]) == p[2])
    return agree / max(1, len(pairs)), mapping, count


def per_gt_drift(pairs, mapping):
    """Per-GT agreement after the optimal P0->P1 ID mapping."""
    by_gid = defaultdict(lambda: [0, 0])
    for p in pairs:
        if p[3] is None:
            continue
        by_gid[p[3]][1] += 1
        if mapping.get(p[1]) == p[2]:
            by_gid[p[3]][0] += 1
    return {gid: ag / max(1, tot) for gid, (ag, tot) in by_gid.items()}


def view_metrics(rows, entries, spec):
    """windowed_metrics over full sequence; dets from common candidates."""
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r[0]].append(r)
    dets, gts = [], []
    for entry in entries:
        keep, gt_boxes, _ = spec_mask(entry, spec)
        keep_set = set(keep)
        fr = int(entry["frame"])
        det = [(r[2], r[3]) for r in by_frame.get(fr, []) if r[1] in keep_set]
        dets.append(det)
        gts.append([(gid, np.asarray(box, np.float64))
                    for gid, box in gt_boxes.items()])
    return windowed_metrics(dets, gts)


def video_audit(model, device, entries, spec, p0_rows=None):
    if p0_rows is None:
        p0_rows = run_tracker(model, device, entries, "ALL")
    if spec == "ALL":
        p1_rows = p0_rows
    else:
        p1_rows = run_tracker(model, device, entries, spec)
    keep_by_frame = {int(e["frame"]): set(spec_mask(e, spec)[0]) for e in entries}
    p0_by = defaultdict(dict)
    p1_by = defaultdict(dict)
    for r in p0_rows:
        if r[1] in keep_by_frame.get(r[0], set()):
            p0_by[r[0]][r[1]] = r
    for r in p1_rows:
        p1_by[r[0]][r[1]] = r
    pairs = []
    for frame, keep in keep_by_frame.items():
        for ci in sorted(keep):
            a = p0_by.get(frame, {}).get(ci)
            b = p1_by.get(frame, {}).get(ci)
            if a is None or b is None:
                continue
            pairs.append((frame, a[2], b[2], a[4], a[5]))
    agree, mapping, count = align_ids(pairs)
    drift = per_gt_drift(pairs, mapping)
    m0 = view_metrics(p0_rows, entries, spec)
    m1 = view_metrics(p1_rows, entries, spec)
    return {
        "n_pairs": len(pairs),
        "agree_rate": agree,
        "drift_rate": 1.0 - agree,
        "per_gt_agree": drift,
        "p0": m0,
        "p1": m1,
    }


def agg_videos(results_by_vid, domain, spec_family):
    total_pairs = sum(v["n_pairs"] for v in results_by_vid.values())
    total_agree = sum(round(v["agree_rate"] * v["n_pairs"]) for v in results_by_vid.values())
    agree_rate = total_agree / max(1, total_pairs)
    drift_counts = defaultdict(lambda: [0, 0])
    for v in results_by_vid.values():
        for gid, ag in v["per_gt_agree"].items():
            drift_counts[gid][0] += ag
            drift_counts[gid][1] += 1
    drift = {gid: a / max(1, n) for gid, (a, n) in drift_counts.items()}
    frac_gt_drift = 0.0
    if drift:
        frac_gt_drift = sum(1 for gid, a in drift.items() if a < 0.9) / len(drift)
    agg = {"domain": domain, "spec": spec_family,
           "n_videos": len(results_by_vid),
           "n_pairs": total_pairs, "agree_rate": agree_rate,
           "drift_rate": 1.0 - agree_rate,
           "n_gt_ids": len(drift),
           "frac_gt_agree_lt_0.9": frac_gt_drift,
           "p0_mean": {"assa": float(np.mean([v["p0"]["assa"] for v in results_by_vid.values()])),
                       "idf1": float(np.mean([v["p0"]["idf1"] for v in results_by_vid.values()])),
                       "idsw_sum": int(sum(v["p0"]["idsw"] for v in results_by_vid.values()))},
           "p1_mean": {"assa": float(np.mean([v["p1"]["assa"] for v in results_by_vid.values()])),
                       "idf1": float(np.mean([v["p1"]["idf1"] for v in results_by_vid.values()])),
                       "idsw_sum": int(sum(v["p1"]["idsw"] for v in results_by_vid.values()))},
           "per_video": results_by_vid}
    return agg


def resolve_specs(specs, entries, instance_k):
    """Return list of (family, concrete_spec) for one video."""
    out = []
    for spec in specs:
        if spec == "inst:auto":
            gt_len = defaultdict(int)
            for e in entries:
                for gid in e.get("gt_boxes", {}):
                    gt_len[gid] += 1
            top = [g for g, _ in sorted(gt_len.items(), key=lambda x: -x[1])[:instance_k]]
            out.append(("inst:auto", f"inst:{','.join(top)}"))
        else:
            out.append((spec, spec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--ckpt", default="outputs/l3/checkpoints/u0/final.pt")
    ap.add_argument("--gpu", type=int, default=9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--instance-k", type=int, default=2,
                    help="for inst:auto, keep top-k longest GT trajectories")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    if "spec_embed.weight" in state:
        model = L4SpecEqAssociator(n_spec=3, d_spec=16)
    else:
        model = L1DAssociator()
    model.load_state_dict(state)
    model.to(device).eval()

    by_video = defaultdict(list)
    with open(args.manifest) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    vids = sorted(by_video)
    if args.max_videos:
        vids = vids[: args.max_videos]
    for v in vids:
        by_video[v].sort(key=lambda e: e["frame"])

    specs = [s for s in args.specs.split(",") if s]
    t0 = time.time()
    results = {}
    for vid in vids:
        entries = by_video[vid]
        if args.max_frames:
            entries = entries[: args.max_frames]
        p0_rows = run_tracker(model, device, entries, "ALL")
        for family, concrete in resolve_specs(specs, entries, args.instance_k):
            va = video_audit(model, device, entries, concrete, p0_rows=p0_rows)
            results.setdefault(family, {})[vid] = va
        print(f"[{args.domain}] {vid} frames={len(entries)} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    summary = {family: agg_videos(results[family], args.domain, family)
               for family in results}
    out = {
        "domain": args.domain,
        "manifest": args.manifest,
        "ckpt": args.ckpt,
        "gpu": args.gpu,
        "specs": specs,
        "instance_k": args.instance_k,
        "n_videos": len(vids),
        "protocol": "PRIVILEGED_SPEC_ORACLE",
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    for fam, s in summary.items():
        print(f"[{args.domain}] {fam}: pairs={s['n_pairs']} "
              f"agree={s['agree_rate']:.4f} drift={s['drift_rate']:.4f} "
              f"P0 assa={s['p0_mean']['assa']:.4f}/{s['p0_mean']['idsw_sum']} "
              f"P1 assa={s['p1_mean']['assa']:.4f}/{s['p1_mean']['idsw_sum']}",
              flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
