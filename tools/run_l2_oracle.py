"""Stage L2: online decision states + counterfactual rollout oracle.

Replays the BEST_STRONG_BASE (L1DK base: 0.4*IoU + 0.2*PBD + 0.4*Kalman
motion IoU, threshold 0.25) with the exact AC shell used for the baseline
matrix, records conflict decision states, and evaluates counterfactual
assignments by freezing the base policy and rolling forward H frames.

Windowed utilities are computed from TrackEval formulas (see
docs/l2_trackeval_objective_audit.md):
  windowed AssA: co-occurrence Jaccard weighted by match counts;
  windowed IDF1: optimal global ID mapping over co-occurrence counts;
  window IDSW: CLEAR-style switch counting within the window.

Usage:
  python tools/run_l2_oracle.py --raw outputs/l1_d/raw/dancetrack_val.pkl \
      --domain dancetrack_val --out outputs/l2/oracle --horizons 4,8,16,32
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.models.l1d_association import compute_affinity_features  # noqa: E402
from locatemot.tracking.association import hungarian_max, iou_matrix  # noqa: E402
from locatemot.tracking.motion import KalmanBoxTracker7  # noqa: E402

WEIGHTS = (0.4, 0.2, 0.4)
THRESHOLD = 0.25
MAX_AGE = 30


def kalman_state(kf):
    return {
        "x": kf.x.copy(),
        "P": kf.P.copy(),
        "age": kf.age,
        "time_since_update": kf.time_since_update,
        "last_observation": kf.last_observation.copy(),
        "observations": {k: v.copy() for k, v in kf.observations.items()},
        "velocity": None if kf.velocity is None else kf.velocity.copy(),
        "hit_streak": kf.hit_streak,
        "hits": kf.hits,
    }


def kalman_from_state(st):
    kf = KalmanBoxTracker7(np.array([0, 0, 1, 1]))
    kf.x = st["x"].copy()
    kf.P = st["P"].copy()
    kf.age = st["age"]
    kf.time_since_update = st["time_since_update"]
    kf.last_observation = st["last_observation"].copy()
    kf.observations = {k: v.copy() for k, v in st["observations"].items()}
    kf.velocity = None if st["velocity"] is None else st["velocity"].copy()
    kf.hit_streak = st["hit_streak"]
    kf.hits = st["hits"]
    return kf


class L2Track:
    __slots__ = ("tid", "true_gt", "last_box", "prev_box", "age", "hits",
                 "lost_age", "last_frame", "status", "ref_pbd", "anchor_pbd",
                 "kalman", "gt_seq", "gt_frames", "birth_frame")

    def __init__(self, tid, true_gt, box, pbd, frame):
        self.tid = tid
        self.true_gt = true_gt
        self.last_box = box.copy()
        self.prev_box = box.copy()
        self.age = 1
        self.hits = 1
        self.lost_age = 0
        self.last_frame = frame
        self.status = "ACTIVE"  # AC shell births are active immediately
        self.ref_pbd = pbd
        self.anchor_pbd = pbd
        self.kalman = KalmanBoxTracker7(box)
        self.gt_seq = [true_gt]  # matched gt id per matched frame
        self.gt_frames = [frame]
        self.birth_frame = frame

    def snapshot(self):
        return {
            "tid": self.tid,
            "true_gt": self.true_gt,
            "last_box": self.last_box.copy(),
            "prev_box": self.prev_box.copy(),
            "age": self.age,
            "hits": self.hits,
            "lost_age": self.lost_age,
            "last_frame": self.last_frame,
            "status": self.status,
            "ref_pbd": self.ref_pbd.copy(),
            "anchor_pbd": self.anchor_pbd.copy(),
            "kalman": kalman_state(self.kalman),
            "gt_seq": list(self.gt_seq),
            "gt_frames": list(self.gt_frames),
            "birth_frame": self.birth_frame,
        }

    @staticmethod
    def from_snapshot(s):
        t = L2Track(s["tid"], s["true_gt"], np.asarray(s["last_box"]),
                    np.asarray(s["ref_pbd"]), s["last_frame"])
        t.prev_box = np.asarray(s["prev_box"])
        t.age = s["age"]
        t.hits = s["hits"]
        t.lost_age = s["lost_age"]
        t.status = s["status"]
        t.ref_pbd = np.asarray(s["ref_pbd"])
        t.anchor_pbd = np.asarray(s["anchor_pbd"])
        t.kalman = kalman_from_state(s["kalman"])
        t.gt_seq = list(s["gt_seq"])
        t.gt_frames = list(s["gt_frames"])
        t.birth_frame = s["birth_frame"]
        return t


def track_contamination(t: L2Track, cur_gt=None):
    """Prediction-side identity history statistics (uses GT only for audit)."""
    seq = [g for g in t.gt_seq if g is not None]
    n = len(seq)
    if n == 0:
        return {
            "n_gt_hits": 0, "dominant_gt": None, "purity": 0.0,
            "past_idsw": 0, "fragments": 0, "age": t.age, "hits": t.hits,
            "gap": 0, "current_gt": cur_gt,
        }
    cnt = Counter(seq)
    dom, dom_n = cnt.most_common(1)[0]
    # past ID switches: consecutive matched gt ids differ (non-None)
    idsw = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
    frags = 1 + sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
    return {
        "n_gt_hits": n,
        "dominant_gt": dom,
        "purity": dom_n / n,
        "past_idsw": idsw,
        "fragments": frags,
        "age": t.age,
        "hits": t.hits,
        "gap": 0,
        "current_gt": cur_gt,
    }


def _pbd_arrays(cands):
    n = len(cands)
    pb = np.zeros((n, 2048), np.float32)
    gen = np.zeros(n, np.float32)
    for i, c in enumerate(cands):
        pb[i] = np.asarray(c["pbd"], dtype=np.float32).reshape(-1)[:2048]
        gen[i] = float(c.get("gen", 0.0))
    return pb, gen


def run_base_frame(tracks, cands, image_size, cur_frame=None):
    """Return (assigns, base, feats) using the exact L1DK base policy."""
    T = len(tracks)
    N = len(cands)
    if T == 0 or N == 0:
        return [], np.zeros((T, N), np.float32), None, None
    tb = np.stack([t.last_box for t in tracks])
    pb = np.stack([t.prev_box for t in tracks])
    rb = np.stack([t.ref_pbd for t in tracks])
    ab = np.stack([t.anchor_pbd for t in tracks])
    cb = np.stack([c["box"] for c in cands])
    cp, cg = _pbd_arrays(cands)
    gaps = np.asarray([1] * T, np.float32)
    ages = np.asarray([t.age for t in tracks], np.float32)
    hits = np.asarray([t.hits for t in tracks], np.float32)
    pred_boxes = np.stack([t.kalman.predict() for t in tracks])
    for i, t in enumerate(tracks):
        gaps[i] = max(1, (cur_frame if cur_frame is not None else t.last_frame + 1)
                      - t.last_frame)
    feats = compute_affinity_features(
        tb, cb, rb, ab, cp, cg, gaps, ages, hits, pb, WEIGHTS,
        image_size, motion_pred_boxes=pred_boxes)
    base = feats["base"]
    assigns = hungarian_max(base, THRESHOLD)
    return assigns, base, feats, pred_boxes


def run_egra_frame(tracks, cands, image_size, model, device, pred_boxes=None,
                   gaps=None):
    """EGRA (L1DK_d03) assignment: base + 0.5*rel*delta."""
    import torch
    T = len(tracks)
    N = len(cands)
    if T == 0 or N == 0:
        return []
    tb = np.stack([t.last_box for t in tracks])
    pb = np.stack([t.prev_box for t in tracks])
    rb = np.stack([t.ref_pbd for t in tracks])
    ab = np.stack([t.anchor_pbd for t in tracks])
    cb = np.stack([c["box"] for c in cands])
    cp, cg = _pbd_arrays(cands)
    if gaps is None:
        gaps = np.asarray([1] * T, np.float32)
    ages = np.asarray([t.age for t in tracks], np.float32)
    hits = np.asarray([t.hits for t in tracks], np.float32)
    if pred_boxes is None:
        pred_boxes = np.stack([t.kalman.predict() for t in tracks])
    feats = compute_affinity_features(
        tb, cb, rb, ab, cp, cg, gaps, ages, hits, pb, WEIGHTS,
        image_size, motion_pred_boxes=pred_boxes)
    batch = {
        "pair_feats": torch.as_tensor(feats["pair_feats"][None], device=device),
        "track_feats": torch.as_tensor(feats["track_feats"][None], device=device),
        "cand_feats": torch.as_tensor(feats["cand_feats"][None], device=device),
        "base": torch.as_tensor(feats["base"][None], device=device),
        "trk_mask": torch.ones(1, T, dtype=torch.bool, device=device),
        "cand_mask": torch.ones(1, N, dtype=torch.bool, device=device),
    }
    with torch.no_grad():
        pred = model(batch)
    base = feats["base"]
    rel = pred["reliability"][0].cpu().numpy()
    delta = pred["delta"][0].cpu().numpy()
    final = base + 0.5 * rel[:, None] * delta  # delta scale 0.3 / 0.6
    return hungarian_max(final, THRESHOLD)


def apply_assignment(tracks, cands, assigns, frame, next_tid, max_age=MAX_AGE):
    """Apply assignment to mutable track/cand lists; returns
    (next_tid, born_tid_by_ci)."""
    matched_t = set()
    matched_c = set()
    for ti, ci in assigns:
        t = tracks[ti]
        c = cands[ci]
        t.kalman.update(c["box"])
        t.prev_box = t.last_box.copy()
        t.last_box = c["box"].copy()
        t.age += 1
        t.hits += 1
        t.lost_age = 0
        t.last_frame = frame
        t.ref_pbd = c["pbd"].copy()
        t.gt_seq.append(c.get("gt"))
        t.gt_frames.append(frame)
        matched_t.add(ti)
        matched_c.add(ci)
    for i, t in enumerate(tracks):
        if i in matched_t:
            continue
        t.lost_age += 1
        t.age += 1
        t.kalman.update(None)
        if t.lost_age > max_age:
            t.status = "TERMINATED"
        elif t.lost_age >= 2 and t.status == "ACTIVE":
            t.status = "LOST"
    # births
    born_tid_by_ci = {}
    for ci, c in enumerate(cands):
        if ci in matched_c:
            continue
        tr = L2Track(next_tid, c.get("gt"), c["box"], c["pbd"], frame)
        tracks.append(tr)
        born_tid_by_ci[ci] = next_tid
        next_tid += 1
    return next_tid, born_tid_by_ci


def conflict_components(base, threshold=THRESHOLD, topk=2):
    """Connected components of the bipartite affinity graph (edges>=threshold
    or row/col top-k). Returns list of (track_idxs, cand_idxs)."""
    T, N = base.shape
    if T == 0 or N == 0:
        return []
    edge = base >= threshold
    if T > 0:
        for r in range(T):
            if N > 0:
                top_cols = np.argsort(-base[r])[:topk]
                edge[r, top_cols] = True
    if N > 0:
        for c in range(N):
            if T > 0:
                top_rows = np.argsort(-base[:, c])[:topk]
                edge[top_rows, c] = True
    parent = list(range(T + N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in range(T):
        for c in range(N):
            if edge[r, c]:
                union(r, T + c)
    comps = defaultdict(list)
    for r in range(T):
        comps[find(r)].append(("t", r))
    for c in range(N):
        comps[find(T + c)].append(("c", c))
    out = []
    for nodes in comps.values():
        ts = sorted(i for k, i in nodes if k == "t")
        cs = sorted(i for k, i in nodes if k == "c")
        if len(ts) + len(cs) >= 2 and (len(ts) >= 2 or len(cs) >= 2):
            out.append((ts, cs))
    return out


def partial_matchings(ts, cs, cap=200):
    """All injective partial maps from ts to cs (list of dict t_idx->c_idx)."""
    res = []

    def rec(i, used, cur):
        if len(res) >= cap:
            return
        res.append(dict(cur))
        if i >= len(ts):
            return
        t = ts[i]
        for c in cs:
            if c in used:
                continue
            cur[t] = c
            used.add(c)
            rec(i + 1, used, cur)
            used.remove(c)
            del cur[t]
            if len(res) >= cap:
                return
        rec(i + 1, used, cur)

    rec(0, set(), {})
    return res


def generate_actions(base, assigns, comp, cand_gt, track_true_gts, rng=None):
    """Return list of global assignments (list of (ti,ci)) for this component.
    Includes base, GT-local, top-score alternatives, all-new, worst."""
    ts, cs = comp
    base_map = {ti: ci for ti, ci in assigns if ti in ts}
    actions = []
    base_action = [(ti, ci) for ti, ci in assigns if ti in ts]
    actions.append(base_action)

    # GT-local action
    gt_map = {}
    for ti in ts:
        for ci in cs:
            if cand_gt[ci] is not None and track_true_gts[ti] == cand_gt[ci]:
                gt_map[ti] = ci
                break
    if gt_map:
        actions.append(sorted(gt_map.items()))
    # better: build from track true_gt (available via state) — handled by caller

    # score all partial matchings, take top alternatives different from base
    def score(pm):
        s = 0.0
        for ti, ci in pm.items():
            s += float(base[ti, ci])
        return s

    cands = []
    if len(ts) <= 4 and len(cs) <= 4:
        pms = partial_matchings(ts, cs, cap=300)
        scored = sorted(pms, key=score, reverse=True)
        seen = set()
        for pm in scored:
            act = sorted(pm.items())
            key = tuple(act)
            if key in seen:
                continue
            seen.add(key)
            if key != tuple(sorted(base_action)):
                cands.append(act)
            if len(cands) >= 6:
                break
        # include worst matching for sanity
        if scored:
            worst = sorted(scored, key=score)[0]
            key = tuple(sorted(worst.items()))
            if key not in seen:
                cands.append(sorted(worst.items()))
    else:
        # large component: random full matchings (diverse global alternatives)
        if rng is None:
            rng = np.random.default_rng(0)
        for _ in range(6):
            perm = rng.permutation(len(cs))
            act = [(ts[i], cs[perm[i]]) for i in range(min(len(ts), len(cs)))]
            cands.append(act)
    # all-new
    cands.append([])
    for act in cands[:8]:
        actions.append(act)
    return actions


def complete_assignment(base, fixed, assigns):
    """Complete a partial global assignment with base Hungarian on the rest."""
    fixed = list(fixed)
    used_t = {ti for ti, _ in fixed}
    used_c = {ci for _, ci in fixed}
    T, N = base.shape
    free_t = [i for i in range(T) if i not in used_t]
    free_c = [j for j in range(N) if j not in used_c]
    if free_t and free_c:
        sub = base[np.ix_(free_t, free_c)]
        rem = hungarian_max(sub, THRESHOLD)
        for ri, ci in rem:
            fixed.append((free_t[ri], free_c[ci]))
    return fixed


def windowed_metrics(dets, gts, alpha=0.05, id_threshold=0.5):
    """TrackEval-style metrics restricted to a window.
    dets[t]: list of (pred_id, box); gts[t]: list of (gt_id, box)."""
    T = len(dets)
    # collect ids
    gt_ids = sorted({g for fr in gts for g, _ in fr})
    pr_ids = sorted({p for fr in dets for p, _ in fr})
    gid2i = {g: i for i, g in enumerate(gt_ids)}
    pid2j = {p: j for j, p in enumerate(pr_ids)}
    ng, npd = len(gt_ids), len(pr_ids)
    potential = np.zeros((ng, npd))
    gt_count = np.zeros((ng, 1))
    pr_count = np.zeros((1, npd))
    sims = []
    gt_ids_t = []
    pr_ids_t = []
    for t in range(T):
        if not gts[t] or not dets[t]:
            sims.append(np.zeros((len(gts[t]), len(dets[t]))))
            gt_ids_t.append([gid2i[g] for g, _ in gts[t]])
            pr_ids_t.append([pid2j[p] for p, _ in dets[t]])
            continue
        gb = np.stack([b for _, b in gts[t]])
        db = np.stack([b for _, b in dets[t]])
        sim = iou_matrix(gb, db)
        sims.append(sim)
        gt_ids_t.append([gid2i[g] for g, _ in gts[t]])
        pr_ids_t.append([pid2j[p] for p, _ in dets[t]])
        denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
        sim_iou = np.zeros_like(sim)
        mask = denom > 1e-6
        sim_iou[mask] = sim[mask] / denom[mask]
        g_idx = np.asarray(gt_ids_t[-1])[:, None]
        p_idx = np.asarray(pr_ids_t[-1])[None, :]
        potential[g_idx, p_idx] += sim_iou
        gt_count[g_idx] += 1
        pr_count[0, p_idx] += 1

    global_align = potential / np.maximum(
        1, gt_count + pr_count - potential)
    matches_count = np.zeros_like(potential)
    tp = 0
    for t in range(T):
        if not gts[t] or not dets[t]:
            continue
        g_idx = np.asarray(gt_ids_t[t])
        p_idx = np.asarray(pr_ids_t[t])
        sm = sims[t]
        score_mat = global_align[g_idx[:, None], p_idx[None, :]] * sm
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-score_mat)
        matched = sm[rows, cols] >= alpha - 1e-9
        rows, cols = rows[matched], cols[matched]
        tp += len(rows)
        matches_count[g_idx[rows], p_idx[cols]] += 1
    ass_a = matches_count / np.maximum(
        1, gt_count + pr_count - matches_count)
    assa = float(np.sum(matches_count * ass_a) / max(1, tp))

    # IDF1-style optimal mapping (threshold 0.5)
    pot = np.zeros((ng, npd))
    for t in range(T):
        if not gts[t] or not dets[t]:
            continue
        m = sims[t] >= id_threshold - 1e-9
        rows, cols = np.nonzero(m)
        g_idx = np.asarray(gt_ids_t[t])
        p_idx = np.asarray(pr_ids_t[t])
        pot[g_idx[rows], p_idx[cols]] += 1
    from scipy.optimize import linear_sum_assignment
    n = ng + npd
    fp = np.zeros((n, n))
    fn = np.zeros((n, n))
    fp[ng:, :npd] = 1e10
    fn[:ng, npd:] = 1e10
    gc = gt_count[:, 0]
    pc = pr_count[0]
    for g in range(ng):
        fn[g, :npd] = gc[g]
        fn[g, npd + g] = gc[g]
    for p in range(npd):
        fp[:ng, p] = pc[p]
        fp[ng + p, p] = pc[p]
    fn[:ng, :npd] -= pot
    fp[:ng, :npd] -= pot
    rows, cols = linear_sum_assignment(fn + fp)
    idfn = int(fn[rows, cols].sum())
    idfp = int(fp[rows, cols].sum())
    idtp = int(gt_count.sum()) - idfn
    idf1 = idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn)

    # CLEAR-style IDSW within window
    idsw = 0
    prev_trk = {}
    for t in range(T):
        if not gts[t] or not dets[t]:
            continue
        g_idx = np.asarray(gt_ids_t[t])
        p_idx = np.asarray(pr_ids_t[t])
        sm = sims[t]
        # greedy: match each gt to max-IoU pred (then resolve conflicts)
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-sm)
        for gi, pi in zip(r, c):
            gid = gt_ids[g_idx[gi]]
            pid = pr_ids[p_idx[pi]]
            if prev_trk.get(gid) is not None and prev_trk[gid] != pid:
                idsw += 1
            prev_trk[gid] = pid
    return {
        "assa": assa,
        "idf1": idf1,
        "idsw": idsw,
        "tp": int(tp),
        "n_gt_dets": int(gt_count.sum()),
        "n_pr_dets": int(pr_count.sum()),
    }


def build_window_io(state, frame_idx, raw_frames, H, image_size):
    """Return dets/gts arrays for window [frame_idx, frame_idx+H)."""
    dets, gts = [], []
    for k in range(H):
        if frame_idx + k >= len(raw_frames):
            break
        fr = raw_frames[frame_idx + k]
        # predicted ids come from state tracks matched to candidates at that frame
        # (caller passes a mapping from candidate -> pred id)
        dets.append([])
        gts.append([])
        for gid, box in fr.get("gt_boxes", {}).items():
            gts[-1].append((gid, np.asarray(box, np.float64)))
    return dets, gts


def replay_video(frames, egra_model=None, device=None, verify_tid=False):
    """Replay one video; returns list of per-frame records (compact) and
    final trackers dict (tid -> rows) if verify_tid."""
    tracks = []
    next_tid = 1
    records = []
    out_rows = []  # per-frame list of (frame, tid, box, gen) for ALL candidates
    for idx, fr in enumerate(frames):
        tracks[:] = [t for t in tracks if t.status != "TERMINATED"]
        cands = [{
            "box": np.asarray(fr["boxes"][i], np.float64),
            "pbd": np.asarray(fr["pbd_be"][i], np.float32),
            "gen": float(fr["gen"][i]),
            "gt": fr["cand_gt"][i],
        } for i in range(len(fr["boxes"]))]
        assigns, base, feats, pred_boxes = run_base_frame(
            tracks, cands, fr["image_size"], cur_frame=fr["frame"])
        rec = {
            "idx": idx,
            "frame": int(fr["frame"]),
            "image_size": list(fr["image_size"]),
            "n_track": len(tracks),
            "n_cand": len(cands),
            "assigns": list(assigns),
            "cand_gt": list(fr["cand_gt"]),
            "gt_boxes": fr.get("gt_boxes", {}),
            "track_snaps": [t.snapshot() for t in tracks],
            "cands": [{
                "box": c["box"].copy(), "pbd": c["pbd"].copy(),
                "gen": c["gen"], "gt": c["gt"],
            } for c in cands],
            "contam": [track_contamination(t, cur_gt=None) for t in tracks],
        }
        if T := len(tracks):
            rec["base"] = base
        else:
            rec["base"] = None
        if egra_model is not None:
            gaps = np.asarray(
                [max(1, fr["frame"] - t.last_frame) for t in tracks], np.float32)
            rec["egra_assigns"] = run_egra_frame(
                tracks, cands, fr["image_size"], egra_model, device,
                pred_boxes=pred_boxes, gaps=gaps)
        records.append(rec)
        next_tid, born = apply_assignment(
            tracks, cands, assigns, fr["frame"], next_tid)
        tid_by_ci = {ci: tracks[ti].tid for ti, ci in assigns}
        tid_by_ci.update(born)
        for ci, c in enumerate(cands):
            out_rows.append(
                (fr["frame"], tid_by_ci[ci], c["box"].copy(), c["gen"]))
    return records, out_rows


def verify_against_tracker(video_id, rows, ref_dir):
    ref = os.path.join(ref_dir, f"{video_id}.txt")
    if not os.path.exists(ref):
        return None
    ref_rows = defaultdict(list)
    for line in open(ref):
        p = line.strip().split(",")
        fr = int(float(p[0]))  # raw tracker files use manifest frame ids
        tid = int(float(p[1]))
        x1, y1, x2, y2 = map(float, p[2:6])
        ref_rows.setdefault(fr, []).append((tid, [x1, y1, x1 + x2, y1 + y2]))
    agree = total = 0
    for _fr, tid, box, _gen in rows:
        fr = _fr
        if ref_rows.get(fr) is not None:
            total += 1
            rl = ref_rows[fr]
            for rtid, rbox in rl:
                if np.allclose(box, rbox, atol=1e-3):
                    agree += int(tid == rtid)
                    break
    return agree, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", default="outputs/l2/oracle")
    ap.add_argument("--horizons", default="4,8,16,32")
    ap.add_argument("--max-conflicts", type=int, default=40)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--actions", type=int, default=8)
    ap.add_argument("--egra", action="store_true")
    ap.add_argument("--verify", default="")
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    horizons = [int(x) for x in args.horizons.split(",")]
    rng = np.random.default_rng(args.seed)

    egra_model = None
    device = None
    if args.egra:
        import torch
        from locatemot.models.l1d_association import L1DAssociator
        device = torch.device("cpu")
        egra_model = L1DAssociator()
        ck = torch.load(
            os.path.join(ROOT, "outputs/l1_d/checkpoints/l1d_k/final.pt"),
            map_location="cpu", weights_only=False)
        egra_model.load_state_dict(ck["model"] if "model" in ck else ck)
        egra_model.eval()

    with open(args.raw, "rb") as f:
        frames = pickle.load(f)
    by_video = defaultdict(list)
    for fr in frames:
        by_video[fr["video_id"]].append(fr)
    vids = sorted(by_video)
    if args.max_videos:
        rng.shuffle(vids)
        vids = vids[: args.max_videos]

    t0 = time.time()
    oracle_events = []
    corr_stats = []
    verify_stats = []
    base_ids_total = 0
    base_ids_agree = 0
    n_conflict = 0
    n_videos = 0
    for vi, vid in enumerate(vids):
        vframes = by_video[vid]
        records, rows = replay_video(vframes, egra_model, device)
        n_videos += 1
        if args.verify:
            v = verify_against_tracker(vid, rows, args.verify)
            if v:
                verify_stats.append((vid, v[0], v[1]))
                base_ids_agree += v[0]
                base_ids_total += v[1]
        # pick up to max_conflicts evenly spread
        conflict_idxs = [i for i, r in enumerate(records)
                         if r["base"] is not None and conflict_components(
                             r["base"], THRESHOLD)]
        if args.max_conflicts and len(conflict_idxs) > args.max_conflicts:
            step = len(conflict_idxs) / args.max_conflicts
            conflict_idxs = [conflict_idxs[int(k * step)] for k in range(args.max_conflicts)]
        for ci in conflict_idxs:
            rec = records[ci]
            comps = conflict_components(rec["base"], THRESHOLD)
            if not comps:
                continue
            tracks = [L2Track.from_snapshot(s) for s in rec["track_snaps"]]
            cands = [dict(c) for c in rec["cands"]]
            assigns = list(rec["assigns"])
            n_conflict += 1
            # store causal state features for later TUM training
            event = {
                "domain": args.domain,
                "video": vid,
                "frame_idx": rec["idx"],
                "frame": rec["frame"],
                "image_size": rec["image_size"],
                "track_snaps": rec["track_snaps"],
                "cands": rec["cands"],
                "base": rec["base"],
                "assigns": assigns,
                "contam": rec["contam"],
                "components": comps,
                "horizons": horizons,
                "actions": [],
                "action_utils": {},
            }
            # generate actions for first component only (keep runtime bounded)
            comp = comps[0]
            actions = generate_actions(
                rec["base"], assigns, comp,
                [c["gt"] for c in rec["cands"]],
                [s["true_gt"] for s in rec["track_snaps"]],
                rng=rng)
            # dedupe actions
            seen = set()
            uniq = []
            for act in actions:
                key = tuple(sorted(act))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(act)
            uniq = uniq[: args.actions]
            # evaluate each action
            for ai, act in enumerate(uniq):
                tr = [L2Track.from_snapshot(s) for s in rec["track_snaps"]]
                ca = [dict(c) for c in rec["cands"]]
                full = complete_assignment(rec["base"], act, assigns)
                nid, _ = apply_assignment(tr, ca, full, rec["frame"], 10_000_000)
                # roll forward
                utils = {}
                for H in horizons:
                    # frame t already applied; simulate t+1..t+H
                    t2 = [L2Track.from_snapshot(t.snapshot()) for t in tr]
                    dets = [None] * H
                    gts = [None] * H
                    for k in range(1, H + 1):
                        fidx = rec["idx"] + k
                        if fidx >= len(vframes):
                            break
                        fr = vframes[fidx]
                        t2[:] = [t for t in t2 if t.status != "TERMINATED"]
                        cands2 = [{
                            "box": np.asarray(fr["boxes"][i], np.float64),
                            "pbd": np.asarray(fr["pbd_be"][i], np.float32),
                            "gen": float(fr["gen"][i]),
                            "gt": fr["cand_gt"][i],
                        } for i in range(len(fr["boxes"]))]
                        a2, b2, _, _ = run_base_frame(
                            t2, cands2, fr["image_size"], cur_frame=fr["frame"])
                        pred_map = {}
                        for ti, ci in a2:
                            pred_map[ci] = t2[ti].tid
                        for ci in range(len(cands2)):
                            if ci not in pred_map:
                                pred_map[ci] = None  # born next
                        # apply to get births/ids
                        nid, born = apply_assignment(
                            t2, cands2, a2, fr["frame"], nid)
                        for ci, c in enumerate(cands2):
                            if pred_map[ci] is None:
                                pred_map[ci] = born[ci]
                        dets[k - 1] = [(pred_map[ci], c["box"].copy())
                                       for ci, c in enumerate(cands2)]
                        gts[k - 1] = [(gid, np.asarray(box, np.float64))
                                      for gid, box in fr.get("gt_boxes", {}).items()]
                    dets = [d if d is not None else [] for d in dets]
                    gts = [g if g is not None else [] for g in gts]
                    w = windowed_metrics(dets, gts)
                    utils[f"H{H}"] = w
                event["actions"].append(act)
                event["action_utils"][str(ai)] = {
                    "act": act,
                    "utils": utils,
                }
            oracle_events.append(event)
        # correction audit (EGRA vs base)
        if egra_model is not None:
            for r in records:
                if r["base"] is None or "egra_assigns" not in r:
                    continue
                bm = {ti: ci for ti, ci in r["assigns"]}
                em = {ti: ci for ti, ci in r["egra_assigns"]}
                all_t = sorted(set(bm) | set(em))
                for ti in all_t:
                    if bm.get(ti) == em.get(ti):
                        continue
                    c = r["contam"][ti]
                    old_gt = bm.get(ti)
                    new_gt = em.get(ti)
                    tgt = None
                    if old_gt is not None:
                        tgt = r["cand_gt"][old_gt]
                    elif new_gt is not None:
                        tgt = r["cand_gt"][new_gt]
                    # local correctness of EGRA edge
                    helpful = False
                    if new_gt is not None and tgt is not None:
                        helpful = bool(r["cand_gt"][new_gt] == tgt)
                    corr_stats.append({
                        "video": vid,
                        "frame": r["frame"],
                        "track_gt": tgt,
                        "old_cand_gt": None if old_gt is None else r["cand_gt"][old_gt],
                        "new_cand_gt": None if new_gt is None else r["cand_gt"][new_gt],
                        "local_helpful": helpful,
                        "contam": c,
                    })
        print(f"[{args.domain}] {vid} conflicts={len(conflict_idxs)} "
              f"elapsed={time.time() - t0:.1f}s", flush=True)

    # aggregate
    agg = {
        "domain": args.domain,
        "n_videos": n_videos,
        "n_conflicts": n_conflict,
        "horizons": horizons,
        "n_actions_per_event": None,
        "headroom": {},
        "mismatch": {},
        "verify": {
            "agree": base_ids_agree,
            "total": base_ids_total,
            "accuracy": (base_ids_agree / base_ids_total) if base_ids_total else None,
            "per_video": verify_stats[:5],
        },
    }
    # headroom per H
    for H in horizons:
        base_u = []
        best_u = []
        base_idsw = []
        best_idsw = []
        n_diff = 0
        local_wrong_best = 0
        local_correct_best = 0
        for ev in oracle_events:
            acts = ev["actions"]
            utils = [ev["action_utils"][str(i)] for i in range(len(acts))]
            cand_gt = [c["gt"] for c in ev["cands"]]
            key = f"H{H}"
            vals = []
            for u in utils:
                w = u["utils"].get(key)
                vals.append(w["assa"] if w else 0.0)
            if not vals:
                continue
            b = utils[0]["utils"].get(key)
            if b is None:
                continue
            bu = b["assa"]
            bi = b["idsw"]
            bi_assa = max(vals)
            best_idx = int(np.argmax(vals))
            bi_idsw = utils[best_idx]["utils"][key]["idsw"]
            base_u.append(bu)
            best_u.append(bi_assa)
            base_idsw.append(bi)
            best_idsw.append(bi_idsw)
            if bi_assa > bu + 1e-6:
                n_diff += 1
            # local correctness of best action vs base
            # base action local correct? compare base assign to GT
            base_act = acts[0]
            base_local_correct = 0
            for ti, ci in base_act:
                if (cand_gt[ci] is not None
                        and ev["track_snaps"][ti]["true_gt"] == cand_gt[ci]):
                    base_local_correct += 1
            best_act = acts[best_idx]
            best_local_correct = 0
            for ti, ci in best_act:
                if (cand_gt[ci] is not None
                        and ev["track_snaps"][ti]["true_gt"] == cand_gt[ci]):
                    best_local_correct += 1
            if best_idx != 0:
                if best_local_correct > base_local_correct:
                    local_wrong_best += 1
                if best_local_correct < base_local_correct:
                    local_correct_best += 1
        agg["headroom"][f"H{H}"] = {
            "n": len(base_u),
            "mean_base_assa": float(np.mean(base_u)) if base_u else None,
            "mean_best_assa": float(np.mean(best_u)) if best_u else None,
            "mean_gain_assa": float(np.mean(np.asarray(best_u) - np.asarray(base_u))) if base_u else None,
            "mean_base_idsw": float(np.mean(base_idsw)) if base_idsw else None,
            "mean_best_idsw": float(np.mean(best_idsw)) if best_idsw else None,
            "frac_better": n_diff / max(1, len(base_u)),
            "best_local_wrong_gt": local_wrong_best,
            "best_local_correct_gt": local_correct_best,
        }
    agg["corrections"] = {
        "n": len(corr_stats),
        "helpful": sum(1 for c in corr_stats if c["local_helpful"]),
        "harmful": sum(1 for c in corr_stats if not c["local_helpful"]),
        "on_contaminated": sum(
            1 for c in corr_stats if c["contam"]["purity"] < 0.8 and c["contam"]["n_gt_hits"] >= 2),
        "contam_helpful": sum(
            1 for c in corr_stats
            if c["local_helpful"] and c["contam"]["purity"] < 0.8 and c["contam"]["n_gt_hits"] >= 2),
    }
    with open(os.path.join(args.out, f"oracle_{args.domain}.json"), "w") as f:
        json.dump(agg, f, indent=2, default=str)
    with open(os.path.join(args.out, f"events_{args.domain}.pkl"), "wb") as f:
        pickle.dump(oracle_events, f, protocol=4)
    print(json.dumps(agg, indent=2, default=str))


if __name__ == "__main__":
    main()
