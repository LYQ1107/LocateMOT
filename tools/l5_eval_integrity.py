"""L5 Phase 1: audit why L4 official TrackEval matches U0 while audit differs.

For a few videos, replays U0 / A2 / A5 / A5p in ALL mode and compares:
  - output txt rows (bytes / hashes);
  - per-frame candidate->track maps;
  - assignments restricted to GT-matched candidates (manifest `matched`);
  - final affinity / base / delta / reliability tensors.

Usage:
  python tools/l5_eval_integrity.py --gpu 6 \
      --out outputs/l5/eval_integrity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.data.token_cache import read_frame_cache  # noqa: E402
from locatemot.models.l1d_association import (  # noqa: E402
    L1DAssociator,
    compute_affinity_features,
)
from locatemot.models.l4_spec_eq import L4SpecEqAssociator  # noqa: E402
from locatemot.tracking.association import hungarian_max  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402

VIDEOS = [
    ("dancetrack_val", "outputs/l1_c/fixed_candidate_manifest/dancetrack_val.jsonl",
     "dancetrack0004", "outputs/l3/trackers/u0/dancetrack_val",
     "outputs/l4/trackeval/a5/trackers/dance_l3"),
    ("bdd100k_train", "outputs/l1_c/fixed_candidate_manifest/bdd100k_train.jsonl",
     "0000f77c-6257be58", "outputs/l3/trackers/u0/bdd100k_train",
     "outputs/l4/trackeval/a5/trackers/bdd_l3"),
    ("mot17_train", "outputs/l1_c/fixed_candidate_manifest/mot17_train.jsonl",
     "MOT17-02-SDP", "outputs/l3/trackers/u0/mot17_train",
     "outputs/l4/trackeval/a5/trackers/mot17_l3"),
]

CKPT = {
    "u0": "outputs/l3/checkpoints/u0/final.pt",
    "a2": "outputs/l4/checkpoints/a2/final.pt",
    "a5": "outputs/l4/checkpoints/a5/final.pt",
    "a5p": "outputs/l4/checkpoints/a5p/final.pt",
}


def load_model(tag, device):
    ck = torch.load(os.path.join(ROOT, CKPT[tag]), map_location="cpu",
                    weights_only=False)
    state = ck["model"] if "model" in ck else ck
    if "spec_embed.weight" in state:
        model = L4SpecEqAssociator(n_spec=3, d_spec=16)
    else:
        model = L1DAssociator()
    model.load_state_dict(state)
    return model.to(device).eval()


def build_candidates(entry):
    root = entry["cache_root"]
    key = entry.get("cache_key") or (
        f"{entry['dataset']}/{entry['video_id']}/{int(entry['frame']):05d}/{entry['protocol']}")
    fr = read_frame_cache(root, key)
    if fr is None:
        return [], entry.get("image_size", [1280, 720])
    feats = fr["features"]
    boxes = np.asarray(feats.get("boxes", np.zeros((0, 4))), dtype=np.float64)
    cands = []
    for i in range(len(boxes)):
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


class CaptureTracker(OnlineTracker):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.affinity_records = []

    def _associate_l1d(self, tracks, cur_feats, cur_boxes, frame_id):
        if not tracks or len(cur_boxes) == 0:
            return []
        T = len(tracks)
        N = len(cur_feats)
        tb = np.zeros((T, 4), np.float64)
        pb = np.zeros((T, 4), np.float64)
        pred_boxes = np.zeros((T, 4), np.float64)
        gaps = np.zeros(T, np.float32)
        ages = np.zeros(T, np.float32)
        hits = np.zeros(T, np.float32)
        ref = np.zeros((T, 2048), np.float32)
        anchor = np.zeros((T, 2048), np.float32)
        for i, trk in enumerate(tracks):
            tb[i] = trk.last_box
            pb[i] = trk.prev_box if trk.prev_box is not None else trk.last_box
            pred_boxes[i] = trk.kalman.predict() if trk.kalman is not None else trk.last_box
            last_frame = trk.history[-1].frame if trk.history else frame_id - 1
            gaps[i] = max(1, frame_id - last_frame)
            ages[i] = trk.age
            hits[i] = trk.hits
            f = trk.history[-1].features if trk.history else trk.last_features
            if f and f.get("pbd_be") is not None:
                ref[i] = np.asarray(f["pbd_be"], dtype=np.float32)
            if trk.anchor_features and trk.anchor_features.get("pbd_be") is not None:
                anchor[i] = np.asarray(trk.anchor_features["pbd_be"], dtype=np.float32)
            else:
                anchor[i] = ref[i]
        cb = np.asarray(cur_boxes, dtype=np.float64).reshape(N, 4)
        cp = np.zeros((N, 2048), np.float32)
        cg = np.zeros(N, np.float32)
        for i, f in enumerate(cur_feats):
            if f.get("pbd_be") is not None:
                cp[i] = np.asarray(f["pbd_be"], dtype=np.float32)
            cg[i] = float(f.get("gen", 0.0))
        feats = compute_affinity_features(
            tb, cb, ref, anchor, cp, cg, gaps, ages, hits, pb,
            self.l1d_weights, self.image_size, motion_pred_boxes=pred_boxes)
        batch = {
            "pair_feats": torch.as_tensor(feats["pair_feats"][None], device=self.device),
            "track_feats": torch.as_tensor(feats["track_feats"][None], device=self.device),
            "cand_feats": torch.as_tensor(feats["cand_feats"][None], device=self.device),
            "base": torch.as_tensor(feats["base"][None], device=self.device),
            "trk_mask": torch.ones(1, T, dtype=torch.bool, device=self.device),
            "cand_mask": torch.ones(1, N, dtype=torch.bool, device=self.device),
        }
        if getattr(self.l1d, "use_spec", False):
            batch["spec"] = torch.full((1,), int(self.spec_idx), dtype=torch.long,
                                       device=self.device)
        with torch.no_grad():
            pred = self.l1d(batch)
            final = pred["final"][0].cpu().numpy()
        if self.l1d_rel_threshold > 0.0 or abs(self.l1d_delta_scale - 0.6) > 1e-6:
            base = feats["base"]
            rel = pred["reliability"][0].cpu().numpy()
            delta = pred["delta"][0].cpu().numpy()
            gate = rel * (rel >= self.l1d_rel_threshold)
            final = base + (self.l1d_delta_scale / 0.6) * gate[:, None] * delta
        self.affinity_records.append({
            "frame": int(frame_id),
            "base": feats["base"],
            "final": final,
            "delta": pred["delta"][0].cpu().numpy(),
            "rel": pred["reliability"][0].cpu().numpy(),
        })
        return [(r, c, float(final[r, c]))
                for r, c in hungarian_max(final, self.l1d_threshold)]


def replay(entries, model, device, tag):
    tr = CaptureTracker(variant="L1D", l1d=model, device=str(device),
                        output_all_candidates=True, spec_idx=0)
    tr.l1d_weights = (0.4, 0.2, 0.4)
    tr.l1d_threshold = 0.25
    tr.l1d_delta_scale = 0.3
    tr.l1d_rel_threshold = 0.0
    rows = []
    for e in entries:
        cands, image_size = build_candidates(e)
        tr.image_size = image_size
        outs = tr.process_frame(int(e["frame"]), cands)
        for ci, o in enumerate(outs):
            rows.append((int(e["frame"]), ci, int(o["track_id"]),
                         tuple(float(x) for x in o["box"]),
                         float(o.get("score", 1.0))))
    return rows, tr.affinity_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--out", default="outputs/l5/eval_integrity.json")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"device": str(device), "videos": {}}
    replays = defaultdict(dict)
    for tag in ["u0", "a2", "a5", "a5p"]:
        model = load_model(tag, device)
        for domain, manifest, vid, u0_dir, l4_dir in VIDEOS:
            entries = []
            with open(os.path.join(ROOT, manifest)) as f:
                for line in f:
                    e = json.loads(line)
                    if e["video_id"] == vid:
                        entries.append(e)
            entries.sort(key=lambda e: e["frame"])
            rows, affs = replay(entries, model, device, tag)
            replays[vid][tag] = {"rows": rows, "affs": affs}
            rows_key = hash_rows(rows)
            u0_txt = os.path.join(ROOT, u0_dir, f"{vid}.txt")
            l4_txt = os.path.join(ROOT, l4_dir, f"{vid}.txt")
            result["videos"].setdefault(vid, {})[tag] = {
                "n_rows": len(rows),
                "rows_hash": rows_key,
                "u0_txt_sha": file_sha(u0_txt),
                "l4_txt_sha": file_sha(l4_txt),
                "n_aff": len(affs),
            }
            print(f"[integrity] {tag} {vid} rows={len(rows)} "
                  f"hash={rows_key[:12]} affs={len(affs)}", flush=True)
        del model
        torch.cuda.empty_cache()
    # cross-tag comparisons
    for domain, manifest, vid, _u0_dir, _l4_dir in VIDEOS:
        entries = []
        with open(os.path.join(ROOT, manifest)) as f:
            for line in f:
                e = json.loads(line)
                if e["video_id"] == vid:
                    entries.append(e)
        gt_cand = {}
        for e in entries:
            gt_cand[int(e["frame"])] = {
                int(m["candidate"]): gid for gid, m in e.get("matched", {}).items()}
        base = replays[vid]["u0"]
        base_map = frame_map(base["rows"])
        base_aff = {a["frame"]: a for a in base["affs"]}
        for tag in ["a2", "a5", "a5p"]:
            other = replays[vid][tag]
            other_map = frame_map(other["rows"])
            other_aff = {a["frame"]: a for a in other["affs"]}
            cmp = compare_maps(base_map, other_map, gt_cand)
            cmp.update(compare_affinity(base_aff, other_aff))
            result["videos"][vid][f"u0_vs_{tag}"] = cmp
            print(f"[integrity] {vid} u0_vs_{tag}: "
                  f"{json.dumps({k: v for k, v in cmp.items() if not isinstance(v, float) or v > 1e-6})}",
                  flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("saved", args.out, flush=True)


def hash_rows(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(repr(r).encode())
    return h.hexdigest()


def file_sha(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def frame_map(rows):
    out = defaultdict(dict)
    for frame, ci, tid, _box, _score in rows:
        out[frame][ci] = tid
    return out


def compare_maps(base, other, gt_cand):
    frames = sorted(set(base) & set(other))
    n_cand = 0
    n_diff = 0
    n_gt = 0
    n_gt_diff = 0
    for fr in frames:
        b = base[fr]
        o = other[fr]
        for ci in set(b) | set(o):
            n_cand += 1
            if b.get(ci) != o.get(ci):
                n_diff += 1
                if ci in gt_cand.get(fr, {}):
                    n_gt_diff += 1
            if ci in gt_cand.get(fr, {}):
                n_gt += 1
    return {
        "n_frames_cmp": len(frames),
        "n_cand": n_cand,
        "n_cand_diff": n_diff,
        "cand_disagree_rate": n_diff / max(1, n_cand),
        "n_gt_matched": n_gt,
        "n_gt_diff": n_gt_diff,
        "gt_disagree_rate": n_gt_diff / max(1, n_gt),
    }


def compare_affinity(base_aff, other_aff):
    frames = sorted(set(base_aff) & set(other_aff))
    max_final = 0.0
    max_base = 0.0
    max_delta = 0.0
    n_aff = 0
    n_argmax_diff = 0
    n_argmax = 0
    for fr in frames:
        a = base_aff[fr]
        b = other_aff[fr]
        if a["final"].shape != b["final"].shape:
            continue
        n_aff += 1
        max_final = max(max_final, float(np.abs(a["final"] - b["final"]).max()))
        max_base = max(max_base, float(np.abs(a["base"] - b["base"]).max()))
        max_delta = max(max_delta, float(np.abs(a["delta"] - b["delta"]).max()))
        if a["final"].size:
            n_argmax += a["final"].shape[0]
            n_argmax_diff += int((a["final"].argmax(1) != b["final"].argmax(1)).sum())
    return {
        "n_aff_frames": n_aff,
        "max_abs_final_diff": max_final,
        "max_abs_base_diff": max_base,
        "max_abs_delta_diff": max_delta,
        "row_argmax_diff_rate": n_argmax_diff / max(1, n_argmax),
    }


if __name__ == "__main__":
    main()
