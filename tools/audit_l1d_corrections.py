"""Stage L1-D: helpful/harmful correction audit.

Compares the base tracker output (per-candidate track id) with the L1-D
output on the same association-controlled candidate set.  For each GT-valid
event (GT object matched to a candidate), an event is correct when the
assigned track id equals the first track id ever assigned to that GT
(immutable birth identity, i.e. the identity the residual model is trained
to preserve/recover).

Usage:
  python tools/audit_l1d_corrections.py \
      --manifest outputs/l1_c/fixed_candidate_manifest/dancetrack_calibration.jsonl \
      --base outputs/l1_c/trackeval/L1DB_w0.70.30.0_t0.30 \
      --l1d outputs/l1_c/trackeval/L1D \
      --out outputs/l1_d/correction_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def load_tracker(path):
    rows = defaultdict(list)
    if not os.path.exists(path):
        return rows
    for line in open(path):
        p = line.strip().split(",")
        if len(p) < 7:
            continue
        rows[int(float(p[0]))].append((int(float(p[1])), list(map(float, p[2:6]))))
    return rows


def load_manifest(path):
    by_video = defaultdict(list)
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    for v in by_video:
        by_video[v].sort(key=lambda e: e["frame"])
    return by_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--l1d", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_video = load_manifest(args.manifest)
    stats = {"helpful": 0, "harmful": 0, "preserved": 0, "base_wrong": 0,
             "base_correct": 0, "l1d_wrong": 0, "events": 0}
    per_video = defaultdict(lambda: dict(stats))
    for vid, entries in by_video.items():
        base_tr = load_tracker(os.path.join(args.base, f"{vid}.txt"))
        l1d_tr = load_tracker(os.path.join(args.l1d, f"{vid}.txt"))
        first_tid = {}
        prev_base = {}
        prev_l1d = {}
        for e in entries:
            fr = int(e["frame"])
            bp = {i: tid for i, (tid, _) in enumerate(base_tr.get(fr, []))}
            lp = {i: tid for i, (tid, _) in enumerate(l1d_tr.get(fr, []))}
            for gid, m in e.get("matched", {}).items():
                ci = int(m["candidate"])
                bt = bp.get(ci)
                lt = lp.get(ci)
                if bt is None or lt is None:
                    continue
                if gid not in first_tid:
                    first_tid[gid] = bt
                # continuity: same track id as this GT got in the previous frame
                base_ok = (gid not in prev_base) or (bt == prev_base[gid])
                l1d_ok = (gid not in prev_l1d) or (lt == prev_l1d[gid])
                prev_base[gid] = bt
                prev_l1d[gid] = lt
                stats["events"] += 1
                per_video[vid]["events"] += 1
                if base_ok:
                    stats["base_correct"] += 1
                    per_video[vid]["base_correct"] += 1
                    if l1d_ok:
                        stats["preserved"] += 1
                        per_video[vid]["preserved"] += 1
                    else:
                        stats["harmful"] += 1
                        per_video[vid]["harmful"] += 1
                else:
                    stats["base_wrong"] += 1
                    per_video[vid]["base_wrong"] += 1
                    if l1d_ok:
                        stats["helpful"] += 1
                        per_video[vid]["helpful"] += 1
                    else:
                        stats["l1d_wrong"] += 1
                        per_video[vid]["l1d_wrong"] += 1

    summary = {
        **stats,
        "correction_precision": round(
            stats["helpful"] / max(1, stats["helpful"] + stats["harmful"]), 4),
        "correction_coverage": round(
            stats["helpful"] / max(1, stats["base_wrong"]), 4),
        "preservation_rate": round(
            stats["preserved"] / max(1, stats["base_correct"]), 4),
        "base_acc": round(stats["base_correct"] / max(1, stats["events"]), 4),
        "l1d_acc": round(
            (stats["preserved"] + stats["helpful"]) / max(1, stats["events"]), 4),
        "birth_identity_note": "base_correct/harmful use per-method frame-to-frame "
                               "continuity, not birth identity",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_video": dict(per_video)},
                  f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
