"""Stage L11: high-precision temporal pseudo-tracklets for OVMOT repair.

Input  : outputs/l10/data/tao_train/*.pkl  (DLA candidates + C-TAO GT +
         CLIP/PBD features; matched candidates carry cand_gt).
Output : outputs/l11/data/pseudo_tracks/<video>.pkl (small sidecar):
         per frame: gt_id [N], pseudo_id [N], link_id [N],
         pseudo_conf/birth_conf/cont_conf [N], ignore_flag [N],
         rel_target [N].

Class A: candidates matched one-to-one to C-TAO base_and_novel GT at
IoU >= 0.30 (greedy; same rule as L10's matcher but denser GT and a
lower, evidence-based threshold).  Their gt_id is used for full
supervision.

Class B: candidates with no GT overlap >= 0.30 that form high-confidence
temporal tracklets (forward + backward cycle-consistent, appearance and
category consistent).  Their pseudo_id is used as confidence-weighted
pseudo identity supervision.

Class C/D: GT-overlapping duplicates, uncertain candidates and
background-like detections -> ignore_flag=1, never NEW.

The raw pre-exclusion linker output is also stored as link_id so the
quality audit can measure pseudo same-ID precision on the GT-covered
subset (base_and_novel latent GT, eval only).

Design evidence: reports/l11_temporal_pseudotrack_literature_audit.md

Clean reimplementation of mechanisms from U2MOT (uncertainty-gated
association), Walker (forward-backward/cycle consistency), COVTrack++
(multi-cue instead of IoU-only pseudo labels), PS-MOT (confidence-gated
pseudo labels).  No external code copied.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = ROOT / "outputs" / "l10" / "data" / "tao_train"
DEFAULT_OUT_DIR = ROOT / "outputs" / "l11" / "data" / "pseudo_tracks"
DEFAULT_GT_JSON = ("/data1/LWR/vranlee/SERVER_ONLY/avis/"
                   "LocateMOT_reference_repos/covtrack/saved_models/"
                   "ctao_dataset/ctao_base_and_novel.json")

# linkage hyper-parameters (calibrated in reports/l11_pseudotrack_quality.md)
GT_MATCH_IOU = 0.30        # class-A match threshold (dense C-TAO)
GT_AMBIG_IOU = 0.30        # unmatched det overlapping GT >= this -> IGNORE
IOU_PRED_GATE = 0.15       # motion gate: IoU with constant-velocity pred
APP_GATE = 0.70            # min (clip+pbd)/2 cosine in [0,1]
MATCH_GATE = 0.62          # combined linkage score threshold
MAX_GAP = 2                # allow short detector flicker (pkl-frame steps)
MIN_TRACKLET_LEN = 3       # observations
MIN_MEAN_APP = 0.80        # tracklet-level appearance self-consistency
MIN_CYCLE_RATE = 0.80      # forward links confirmed backward
MIN_MEAN_GEN = 0.25        # mean detector confidence
SUPPRESS_IOU = 0.50        # near-duplicate unmatched det -> keep top score
W = dict(iou=0.25, app=0.45, cat=0.20, gap=0.10)

# populated in main() before Pool fork
_LATENT = None
_SCENES = None
_DATA_DIR = DEFAULT_DATA_DIR
_OUT_DIR = DEFAULT_OUT_DIR


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    den = ar + br - inter
    return inter / den if den > 1e-9 else 0.0


def cos_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def app_sim(o1, o2):
    s = (cos_sim(o1["clip"], o2["clip"])
         + cos_sim(o1["pbd"], o2["pbd"])) / 2.0
    return (s + 1.0) / 2.0


def match_dets(dets, gts, thr=GT_MATCH_IOU):
    """Greedy one-to-one IoU>=thr match; per-det gt id or None."""
    scores = []
    for j, d in enumerate(dets):
        for gid, gb in gts:
            v = iou(d, gb)
            if v >= thr:
                scores.append((v, j, gid))
    scores.sort(reverse=True)
    used_d, used_g, out = set(), set(), [None] * len(dets)
    for v, j, gid in scores:
        if j in used_d or gid in used_g:
            continue
        used_d.add(j)
        used_g.add(gid)
        out[j] = gid
    return out


def predict_box(tracklet, obs, f):
    """Constant-velocity prediction from the tracklet's last two obs."""
    last = tracklet[-1]
    gap = f - last["fi"]
    if len(tracklet) >= 2:
        prev = tracklet[-2]
        g1 = max(1, last["fi"] - prev["fi"])
        vx = (last["box"][0] - prev["box"][0]) / g1
        vy = (last["box"][1] - prev["box"][1]) / g1
        vx2 = (last["box"][2] - prev["box"][2]) / g1
        vy2 = (last["box"][3] - prev["box"][3]) / g1
        return [last["box"][0] + vx * gap, last["box"][1] + vy * gap,
                last["box"][2] + vx2 * gap, last["box"][3] + vy2 * gap]
    return last["box"]


def link_score(tracklet, obs, f, image_size):
    last = tracklet[-1]
    gap = f - last["fi"]
    if gap <= 0 or gap > MAX_GAP:
        return -1e9, 0.0, 0.0
    if obs["label"] != last["label"]:
        return -1e9, 0.0, 0.0
    pb = predict_box(tracklet, obs, f)
    iou_p = iou(pb, obs["box"])
    a = app_sim(last, obs)
    if iou_p < IOU_PRED_GATE or a < APP_GATE:
        return -1e9, iou_p, a
    w = max(0.0, min(1.0, a))
    gap_pen = np.exp(-(gap - 1) / 2.0)
    s = (W["iou"] * min(1.0, iou_p / 0.5)
         + W["app"] * w
         + W["cat"] * 1.0
         + W["gap"] * gap_pen)
    return s, iou_p, a


def _link_tracklets(obs, by_frame, n_frames, image_size):
    """Forward greedy linking; returns list of tracklets (obs lists)."""
    tracklets = []
    last_by_track = []
    for fi in range(n_frames):
        cur = [k for k in by_frame[fi] if obs[k]["gt"] is None
               and not obs[k]["dup"]]
        if not cur:
            continue
        pairs = []
        for ti in range(len(tracklets)):
            last = last_by_track[ti]
            if fi - last["fi"] > MAX_GAP:
                continue
            for k in cur:
                s, _, _ = link_score(tracklets[ti], obs[k], obs[k]["fi"],
                                     image_size)
                if s > -1e8:
                    pairs.append((s, ti, k))
        pairs.sort(reverse=True)
        used_t, used_k, assigned = set(), set(), {}
        for s, ti, k in pairs:
            if ti in used_t or k in used_k or s < MATCH_GATE:
                continue
            used_t.add(ti)
            used_k.add(k)
            assigned[k] = ti
            tracklets[ti].append(obs[k])
            last_by_track[ti] = obs[k]
        for k in cur:
            if k not in assigned:
                tracklets.append([obs[k]])
                last_by_track.append(obs[k])
    return tracklets


def _cycle_check(tracklets, obs, by_frame, image_size):
    for tl in tracklets:
        for a, b in zip(tl, tl[1:]):
            best, best_s = None, -1e9
            for k in by_frame[a["fi"]]:
                o = obs[k]
                if o["gt"] is not None or o["dup"]:
                    continue
                s, _, _ = link_score([o], b, b["fi"], image_size)
                if s > best_s:
                    best_s, best = s, k
            a["cycle_ok"] = best is not None and obs[best] is a


def build_video(rec, video_idx):
    frames = rec["frames"]
    n_frames = len(frames)
    scene_key = rec["video_id"][len("train-"):]
    scene_dir = _SCENES.get(scene_key)
    obs = []
    for fi, fr in enumerate(frames):
        n = len(fr["boxes"])
        fn = (f"train/{scene_dir}/frame{int(fr['frame']):04d}.jpg"
              if scene_dir else None)
        gts = _LATENT.get(fn, []) if fn else []
        gt_match = match_dets(fr["boxes"], gts, GT_MATCH_IOU)
        for j in range(n):
            max_iou = max((iou(fr["boxes"][j], gb) for _, gb in gts),
                          default=0.0)
            obs.append({
                "fi": fi, "frame": int(fr["frame"]), "j": j,
                "box": np.asarray(fr["boxes"][j], np.float64),
                "gen": float(fr["gen"][j]), "label": int(fr["label"][j]),
                "clip": np.asarray(fr["clip"][j], np.float32),
                "pbd": np.asarray(fr["pbd"][j], np.float32),
                "gt": gt_match[j], "gt_max_iou": float(max_iou),
                "dup": False,
            })
    by_frame = defaultdict(list)
    for k, o in enumerate(obs):
        by_frame[o["fi"]].append(k)
    # near-duplicate suppression among unmatched candidates
    for fi, ks in by_frame.items():
        unmatched = [k for k in ks if obs[k]["gt"] is None]
        unmatched.sort(key=lambda k: -obs[k]["gen"])
        for i in range(len(unmatched)):
            for j in range(i + 1, len(unmatched)):
                a, b = unmatched[i], unmatched[j]
                if obs[a]["dup"] or obs[b]["dup"]:
                    continue
                if (obs[a]["label"] == obs[b]["label"]
                        and iou(obs[a]["box"], obs[b]["box"]) >= SUPPRESS_IOU):
                    obs[b]["dup"] = True
    tracklets = _link_tracklets(obs, by_frame, n_frames, rec["image_size"])
    _cycle_check(tracklets, obs, by_frame, rec["image_size"])
    # keep high-confidence tracklets
    kept = []
    for ti, tl in enumerate(tracklets):
        if len(tl) < MIN_TRACKLET_LEN:
            continue
        apps = [app_sim(a, b) for a, b in zip(tl, tl[1:])]
        cycles = [1.0 if a.get("cycle_ok") else 0.0 for a in tl[:-1]]
        mean_app = float(np.mean(apps)) if apps else 0.0
        mean_gen = float(np.mean([o["gen"] for o in tl]))
        cycle_rate = float(np.mean(cycles)) if cycles else 0.0
        if mean_app < MIN_MEAN_APP or cycle_rate < MIN_CYCLE_RATE \
                or mean_gen < MIN_MEAN_GEN:
            continue
        conf = min(0.97, mean_gen * mean_app * np.sqrt(cycle_rate))
        kept.append((ti, tl, conf))
    # sidecars
    sidecar_frames = []
    for fi, fr in enumerate(frames):
        n = len(fr["boxes"])
        sidecar_frames.append({
            "gt_id": [None] * n,
            "pseudo_id": [None] * n,
            "link_id": [None] * n,
            "pseudo_conf": np.zeros(n, np.float32),
            "birth_conf": np.zeros(n, np.float32),
            "cont_conf": np.zeros(n, np.float32),
            "ignore_flag": np.zeros(n, np.int8),
            "rel_target": np.zeros(n, np.float32),
        })
    for k, o in enumerate(obs):
        if o["gt"] is not None:
            sidecar_frames[o["fi"]]["gt_id"][o["j"]] = o["gt"]
            sidecar_frames[o["fi"]]["rel_target"][o["j"]] = 1.0
    stats = {"tracklets_before_filter": len(tracklets),
             "tracklets_kept": 0, "link_cands": 0, "pseudo_cands": 0,
             "mean_len": 0.0, "mean_cycle": 0.0, "mean_app": 0.0}
    tracklet_stats = []
    for ti, tl, conf in kept:
        pid = f"P{video_idx:05d}_{ti}"
        apps = [app_sim(a, b) for a, b in zip(tl, tl[1:])]
        cycles = [1.0 if a.get("cycle_ok") else 0.0 for a in tl[:-1]]
        tracklet_stats.append({
            "id": pid, "len": len(tl),
            "cycle_rate": round(float(np.mean(cycles)), 4),
            "mean_app": round(float(np.mean(apps)), 4),
            "mean_gen": round(float(np.mean([o["gen"] for o in tl])), 4),
            "frames": [o["frame"] for o in tl],
        })
        for oi, o in enumerate(tl):
            sc = sidecar_frames[o["fi"]]
            sc["link_id"][o["j"]] = pid
            sc["pseudo_conf"][o["j"]] = conf
            sc["birth_conf"][o["j"]] = conf if oi == 0 else 0.0
            sc["cont_conf"][o["j"]] = conf if oi > 0 else 0.0
            if o["gt_max_iou"] < GT_AMBIG_IOU:
                sc["pseudo_id"][o["j"]] = pid
                sc["rel_target"][o["j"]] = conf
            else:
                # GT-overlapping duplicate/ambiguous: not NEW, no pseudo
                sc["ignore_flag"][o["j"]] = 1
        stats["tracklets_kept"] += 1
        stats["link_cands"] += len(tl)
        stats["mean_len"] += len(tl)
        stats["mean_cycle"] += float(np.mean(cycles))
        stats["mean_app"] += float(np.mean(apps))
    # ignore all unmatched candidates with no pseudo id
    for k, o in enumerate(obs):
        if o["gt"] is None and o["dup"]:
            sidecar_frames[o["fi"]]["ignore_flag"][o["j"]] = 1
    for fi, sc in enumerate(sidecar_frames):
        for j in range(len(sc["pseudo_id"])):
            if sc["pseudo_id"][j] is not None:
                stats["pseudo_cands"] += 1
    if stats["tracklets_kept"]:
        stats["mean_len"] /= stats["tracklets_kept"]
        stats["mean_cycle"] /= stats["tracklets_kept"]
        stats["mean_app"] /= stats["tracklets_kept"]
    return {"video_id": rec["video_id"], "frames": sidecar_frames,
            "stats": stats, "tracklet_stats": tracklet_stats,
            "n_frames": n_frames}


def load_gt():
    nv = json.load(open(DEFAULT_GT_JSON))
    anns = defaultdict(list)
    for a in nv["annotations"]:
        anns[a["image_id"]].append(a)
    img_file = {}
    scenes = {}
    for im in nv["images"]:
        img_file[im["id"]] = im["file_name"]
        parts = im["file_name"].split("/")
        if len(parts) >= 4:
            sd = "/".join(parts[1:-1])
            scenes.setdefault(sd.replace("/", "-"), sd)
    latent = defaultdict(list)
    for iid, al in anns.items():
        fn = img_file.get(iid)
        if fn is None:
            continue
        for a in al:
            x, y, w, h = a["bbox"]
            latent[fn].append((str(a["track_id"]), [x, y, x + w, y + h]))
    return latent, scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--index", default=str(DEFAULT_DATA_DIR / "index.json"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--videos", nargs="*", default=None)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = json.loads(Path(args.index).read_text())
    names = sorted(index["videos"].keys())
    if args.videos:
        names = [n for n in names if n in set(args.videos)]
    if args.max_videos:
        names = names[:args.max_videos]
    global _DATA_DIR, _OUT_DIR, _LATENT, _SCENES
    _DATA_DIR = data_dir
    _OUT_DIR = out_dir
    print("[l11pseudo] loading base_and_novel GT ...", flush=True)
    _LATENT, _SCENES = load_gt()
    print(f"[l11pseudo] latent files={len(_LATENT)} scenes={len(_SCENES)}",
          flush=True)
    import multiprocessing as mp
    results = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, names)):
            if r is None:
                continue
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"[l11pseudo] {i+1}/{len(names)} videos", flush=True)
    tot = {"tracklets_kept": 0, "link_cands": 0, "pseudo_cands": 0,
           "mean_len": 0.0, "mean_cycle": 0.0, "mean_app": 0.0}
    for name, st in results:
        for k in ("tracklets_kept", "link_cands", "pseudo_cands"):
            tot[k] += st[k]
        for k in ("mean_len", "mean_cycle", "mean_app"):
            tot[k] += st[k] * st["tracklets_kept"]
    if tot["tracklets_kept"]:
        for k in ("mean_len", "mean_cycle", "mean_app"):
            tot[k] /= tot["tracklets_kept"]
    print(f"[l11pseudo] videos={len(results)} {tot}", flush=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(tot, f, indent=2)


def _worker(name):
    out_path = _OUT_DIR / f"{name}.pkl"
    if out_path.exists():
        return None
    rec = pickle.load(open(_DATA_DIR / f"{name}.pkl", "rb"))
    side = build_video(rec, hash(name) % 100000)
    with open(out_path, "wb") as f:
        pickle.dump(side, f, protocol=4)
    return name, side["stats"]


if __name__ == "__main__":
    main()
