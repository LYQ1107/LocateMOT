#!/usr/bin/env python
"""Stage L0-C: write video split files (phase=splits) and pair manifest
(phase=pairs, requires cache metadata)."""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.data.pair_manifest import (  # noqa: E402
    choose_frames,
    load_frozen_split,
    overlap,
    select_subset,
    split_hash,
    video_frame_names,
    write_split_json,
)


SPLITS = {
    "train": ("l0_c_train_videos.json", "unified_train_split.json"),
    "calibration": ("l0_c_calibration_videos.json", "unified_calibration_split.json"),
    "heldout": ("l0_c_heldout_videos.json", "identity_heldout_split.json"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["splits", "pairs"], required=True)
    ap.add_argument("--config", default="configs/stage_l0_c.yaml")
    ap.add_argument("--out", default="outputs/l0_c")
    ap.add_argument("--cache-root", default="/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L0C/cache")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    split_dir = os.path.join(os.getcwd(), "configs", "data")
    os.makedirs(split_dir, exist_ok=True)

    if args.phase == "splits":
        selected = {}
        full = {
            name: load_frozen_split(cfg["data"]["frozen_splits"][name])
            for name in SPLITS
        }
        # select heldout and calibration first, then exclude them from train
        sizes = cfg["data"]["split_sizes"]
        selected["heldout"] = select_subset(full["heldout"], sizes["heldout"]["youtube"], sizes["heldout"]["mose"], cfg["seed"])
        selected["calibration"] = select_subset(full["calibration"], sizes["calibration"]["youtube"], sizes["calibration"]["mose"], cfg["seed"])
        exclude = {
            (e["dataset"], e["video_id"])
            for e in selected["calibration"] + selected["heldout"]
        }
        train_pool = [e for e in full["train"] if (e["dataset"], e["video_id"]) not in exclude]
        selected["train"] = select_subset(train_pool, sizes["train"]["youtube"], sizes["train"]["mose"], cfg["seed"])

        for name, (out_name, src_name) in SPLITS.items():
            entries = load_frozen_split(cfg["data"]["frozen_splits"][name])
            write_split_json(
                os.path.join(split_dir, out_name),
                name, selected[name], cfg["seed"], cfg["data"]["frozen_splits"][name],
            )
        report = []
        for name, sel in selected.items():
            report.append({
                "split": name,
                "videos": len(sel),
                "hash": split_hash(sel),
            })
        for a, b in [("train", "calibration"), ("train", "heldout"), ("calibration", "heldout")]:
            ov = overlap(selected[a], selected[b])
            report.append({"overlap": f"{a}-{b}", "count": len(ov), "items": ov[:5]})
        os.makedirs("reports", exist_ok=True)
        with open("reports/l0_c_split_report.json", "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    # pairs phase: read split json + cache metadata
    roots = {"youtube": cfg["data"]["youtube_vos_root"], "mose": cfg["data"]["mose_root"]}
    pair_records = []
    stats = []
    for split_name, (out_name, _) in SPLITS.items():
        split_path = os.path.join(split_dir, out_name)
        split = json.load(open(split_path))
        for entry in split["videos"]:
            dataset = entry["dataset"]
            vid = entry["video_id"]
            frames = video_frame_names(dataset, vid, roots)
            chosen = [int(frames[i]) for i in choose_frames(len(frames), cfg["data"]["frames_per_video"])]
            for ri, r_frame in enumerate(chosen):
                for ci, c_frame in enumerate(chosen):
                    if ci == ri:
                        continue
                    gap = abs(c_frame - r_frame)
                    protocols = ["category_guided"] if "youtube" in dataset else []
                    protocols.append("generic")
                    for proto in protocols:
                        r_key = _key(dataset, vid, r_frame, proto)
                        c_key = _key(dataset, vid, c_frame, proto)
                        r_meta = _load_meta(args.cache_root, r_key)
                        c_meta = _load_meta(args.cache_root, c_key)
                        if r_meta is None or c_meta is None:
                            continue
                        rec = _build_pair(
                            split_name, dataset, vid, r_frame, c_frame, gap, proto, r_meta, c_meta,
                            r_key, c_key,
                        )
                        if rec is not None:
                            pair_records.append(rec)
        print(f"[pairs] {split_name}: {len(pair_records)} cumulative")
    if args.max_pairs and len(pair_records) > args.max_pairs:
        pair_records = pair_records[: args.max_pairs]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pair_manifest.jsonl"), "w") as f:
        for r in pair_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "pair_manifest_statistics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "pairs", "unique_videos", "positive", "no_match", "candidate_missing", "multi_reference"])
        by_split = {}
        for r in pair_records:
            by_split.setdefault(r["split"], []).append(r)
        for split_name, recs in by_split.items():
            w.writerow([
                split_name, len(recs), len({(r["dataset"], r["video_id"]) for r in recs}),
                sum(1 for r in recs if r["visible_positives"]),
                sum(1 for r in recs if r["true_no_match_count"]),
                sum(1 for r in recs if r["candidate_missing_count"]),
                sum(1 for r in recs if r["reference_target_count"] > 1),
            ])
    print("pair manifest written:", len(pair_records))


def _key(dataset, vid, frame, proto):
    return f"{dataset}/{vid}/{frame:05d}/{proto}"


def _load_meta(cache_root, key):
    p = os.path.join(cache_root, key + ".meta.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def _build_pair(split_name, dataset, vid, r_frame, c_frame, gap, proto, r_meta, c_meta, r_key, c_key):
    r_targets = []
    for tid in r_meta.get("gt_object_ids", []):
        r_targets.append({
            "track_id": tid,
            "gt_box": r_meta["gt_boxes"].get(str(tid)),
            "reference_candidate_index": r_meta.get("matched_candidates", {}).get(str(tid)),
            "reference_token_id": r_key,
        })
    r_targets = [t for t in r_targets if t["gt_box"] is not None][:8]
    if not r_targets:
        return None
    c_gt_boxes = c_meta.get("gt_boxes", {})
    c_matched = c_meta.get("matched_candidates", {})
    targets = []
    no_match = []
    candidate_missing = []
    for t in r_targets:
        tid = t["track_id"]
        if str(tid) in c_gt_boxes:
            c_idx = c_matched.get(str(tid))
            if c_idx is not None:
                targets.append({"track_id": tid, "candidate_index": c_idx})
            else:
                candidate_missing.append(tid)
        else:
            no_match.append(tid)
    if not targets and not no_match and not candidate_missing:
        return None
    return {
        "split": split_name,
        "dataset": dataset,
        "video_id": vid,
        "reference_frame": r_frame,
        "current_frame": c_frame,
        "temporal_gap": int(gap),
        "protocol": proto,
        "reference_token_id": r_key,
        "current_token_id": c_key,
        "reference_targets": r_targets,
        "reference_track_ids": [t["track_id"] for t in r_targets],
        "reference_boxes": [t["gt_box"] for t in r_targets],
        "current_candidate_count": c_meta.get("candidate_count", 0),
        "assignment_targets": targets,
        "no_match_targets": no_match,
        "candidate_missing_targets": candidate_missing,
        "visible_positives": len(targets),
        "true_no_match_count": len(no_match),
        "candidate_missing_count": len(candidate_missing),
        "reference_target_count": len(r_targets),
    }


if __name__ == "__main__":
    main()
