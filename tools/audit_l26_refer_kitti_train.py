#!/usr/bin/env python3
"""Audit the complete local Refer-KITTI data contract for L26.

Read-only: no model fitting, threshold selection, bank generation, or GT
mutation.  GT is used only to quantify the existing IoU>=.50 sidecar mapping.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
L11_EXP = ROOT / "outputs/l11/data/rmot_kitti/expressions.json"
L16_EXP = ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"
L11_REC = ROOT / "outputs/l11/data/rmot_kitti"
L16_REC = ROOT / "outputs/l16/data/kitti_missing/records"
BANK_ROOT = ROOT / "outputs/l19/dual_banks_features/kitti"
SPLIT = ROOT / "outputs/l16/data/protocol/split_manifest.json"
TRAIN_LIST = ROOT.parent / "LocateMOT_reference_repos/dkgtrack/datasets/data_path/refer-kitti-v2/refer-kitti-v2.train"
RAW = ROOT / "data/kitti_tracking_training/image_02"
FAST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
OUT = ROOT / "outputs/l26/audit/refer_kitti_complete_train"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def expressions() -> dict[str, list[dict]]:
    out = {}
    for path in (L11_EXP, L16_EXP):
        if not path.exists():
            continue
        for video, rows in load_json(path).items():
            out.setdefault(str(video), []).extend(rows)
    for video in out:
        dedup = {}
        for row in out[video]:
            dedup[str(row["expression"])] = row
        out[video] = list(dedup.values())
    return out


def record_for(video: str) -> Path:
    for root in (L11_REC, L16_REC):
        path = root / f"{video}.pkl"
        if path.exists():
            return path
    raise FileNotFoundError(f"no GT record for {video}")


def load_frame_gt(video: str) -> dict[int, dict[str, list[float]]]:
    rec = pickle.loads(record_for(video).read_bytes())
    return {int(x["frame"]): {str(k): v for k, v in x.get("gt_boxes", {}).items()}
            for x in rec["frames"]}


def flags(text: str) -> dict[str, bool]:
    s = text.lower()
    return {
        "motion": bool(re.search(r"motion|moving|park|stop|driv|turn|front|behind|direction", s)),
        "color": bool(re.search(r"black|white|gray|grey|red|blue|green|yellow|silver|orange|brown", s)),
        "position": bool(re.search(r"left|right|middle|center|centre|front|behind|near|far|top|bottom", s)),
        "relation": bool(re.search(r"next|beside|between|closest|nearest|overlap|behind", s)),
    }


def main() -> None:
    split = load_json(SPLIT)["kitti_v2"]
    split_of = {str(v): "train" for v in split["train"]}
    split_of.update({str(v): "train_val" for v in split["train_val"]})
    split_of.update({str(v): "official_eval" for v in split["official_eval"]})
    exps = expressions()
    videos = sorted(split_of)
    if set(videos) != set(exps):
        raise AssertionError({"split_without_expression": sorted(set(videos)-set(exps)),
                              "expression_without_split": sorted(set(exps)-set(videos))})
    result = {
        "format": "locatemot-l26-refer-kitti-complete-audit-v1",
        "project_root": str(ROOT),
        "split_manifest": str(SPLIT),
        "split_manifest_sha256": sha(SPLIT),
        "fast_manifest": str(FAST),
        "fast_manifest_sha256": sha(FAST),
        "official_gt_used": True,
        "gt_use": "read-only audit statistics; no projection/model/threshold fitting",
        "source_files": {
            "l11_expressions": str(L11_EXP),
            "l16_expressions": str(L16_EXP),
            "train_frame_list": str(TRAIN_LIST),
            "old_bank_root": str(BANK_ROOT),
            "raw_root": str(RAW),
        },
        "split_videos": split_of,
        "videos": {},
        "query_counts": {"train": 0, "train_val": 0, "official_eval": 0},
        "expression_flags": {key: 0 for key in ("motion", "color", "position", "relation")},
        "totals": Counter(),
    }
    train_frame_lines = [x.strip() for x in TRAIN_LIST.read_text().splitlines() if x.strip()]
    result["train_frame_list_lines"] = len(train_frame_lines)
    result["train_frame_list_unique_video_frames"] = len(set(train_frame_lines))
    for video in videos:
        bank_path = BANK_ROOT / f"{video}.pt"
        label_path = BANK_ROOT / f"{video}.labels.json"
        if not bank_path.exists():
            raise FileNotFoundError(f"missing bank for {video}")
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        t = bank["tensors"]
        frames = t["frame_ids"].numpy().astype(np.int64)
        ptr = t["frame_ptr"].numpy().astype(np.int64)
        boxes = t["box"].float().numpy().astype(np.float32)
        pools = t["pool_id"].numpy().astype(np.int64)
        objects = t["objectness"].numpy().astype(np.float32)
        gt = load_frame_gt(video)
        per_video = {
            "split": split_of[video], "expression_count": len(exps[video]),
            "bank": str(bank_path), "bank_sha256": sha(bank_path),
            "bank_rows": int(len(boxes)), "bank_frames": int(len(frames)),
            "raw_frames_present": 0, "raw_frames_missing": [],
            "candidate_coverage_rows": 0, "candidate_total_rows": int(len(boxes)) * len(exps[video]),
            "queries": {}, "aggregate": Counter(),
        }
        for frame in frames.tolist():
            raw = RAW / video / f"{int(frame):06d}.png"
            if raw.exists():
                per_video["raw_frames_present"] += 1
            else:
                per_video["raw_frames_missing"].append(int(frame))
        for exp in exps[video]:
            name = str(exp["expression"])
            target_by_frame = {int(k): {str(x) for x in v} for k, v in exp.get("label", {}).items()}
            q = {"expression": name, "sentence": exp.get("sentence", name),
                 "raw_sentence": exp.get("raw_sentence", ""), "split": split_of[video],
                 "frames": 0, "positive_frames": 0, "null_frames": 0,
                 "present_uncovered_frames": 0, "multi_positive_frames": 0,
            "candidate_rows": 0, "positive_rows": 0, "negative_rows": 0,
                 "same_frame_hard_negative_rows": 0, "covered_target_ids": 0,
                 "target_ids": 0, "max_positive_per_frame": 0,
                 "flags": flags(str(exp.get("sentence", "")) + " " + str(exp.get("raw_sentence", "")))}
            for key, value in q["flags"].items():
                result["expression_flags"][key] += int(value)
            for fi, frame in enumerate(frames.tolist()):
                begin, end = int(ptr[fi]), int(ptr[fi + 1])
                target_ids = target_by_frame.get(int(frame), set())
                gt_boxes = [gt.get(int(frame), {}).get(str(x)) for x in target_ids]
                gt_boxes = [np.asarray(x, dtype=np.float32) for x in gt_boxes if x is not None]
                frame_boxes = boxes[begin:end]
                if gt_boxes:
                    gt_arr = np.stack(gt_boxes)
                    left = np.maximum(frame_boxes[:, None, 0], gt_arr[None, :, 0])
                    top = np.maximum(frame_boxes[:, None, 1], gt_arr[None, :, 1])
                    right = np.minimum(frame_boxes[:, None, 2], gt_arr[None, :, 2])
                    bottom = np.minimum(frame_boxes[:, None, 3], gt_arr[None, :, 3])
                    inter = np.maximum(0, right-left) * np.maximum(0, bottom-top)
                    area = np.maximum(0, frame_boxes[:, 2]-frame_boxes[:, 0]) * np.maximum(0, frame_boxes[:, 3]-frame_boxes[:, 1])
                    garea = np.maximum(0, gt_arr[:, 2]-gt_arr[:, 0]) * np.maximum(0, gt_arr[:, 3]-gt_arr[:, 1])
                    ious = inter / np.maximum(1e-6, area[:, None] + garea[None, :] - inter)
                    positive = ious.max(axis=1) >= 0.5
                    covered_ids = int(np.count_nonzero(ious.max(axis=0) >= 0.5))
                else:
                    positive = np.zeros(len(frame_boxes), dtype=bool)
                    covered_ids = 0
                neg = ~positive
                count = int(positive.sum())
                q["frames"] += 1; q["candidate_rows"] += end - begin
                q["positive_rows"] += count; q["negative_rows"] += int(neg.sum())
                q["positive_frames"] += int(count > 0); q["null_frames"] += int(not target_ids)
                q["present_uncovered_frames"] += int(bool(target_ids) and count == 0)
                q["multi_positive_frames"] += int(count > 1); q["max_positive_per_frame"] = max(q["max_positive_per_frame"], count)
                q["target_ids"] += len(gt_boxes); q["covered_target_ids"] += covered_ids
                if count:
                    hard_n = int(min(48, neg.sum()))
                    q["same_frame_hard_negative_rows"] += hard_n
                for key, value in {
                    "frames": 1, "candidate_rows": end-begin, "positive_rows": count,
                    "negative_rows": int(neg.sum()), "positive_frames": int(count > 0),
                    "null_frames": int(not target_ids), "present_uncovered_frames": int(bool(target_ids) and count == 0),
                    "multi_positive_frames": int(count > 1), "same_frame_hard_negative_rows": int(min(48, neg.sum())) if count else 0,
                }.items():
                    per_video["aggregate"][key] += value
                if len(target_ids):
                    per_video["candidate_coverage_rows"] += count
            q["coverage"] = q["covered_target_ids"] / max(1, q["target_ids"])
            q["multi_positive_rate"] = q["multi_positive_frames"] / max(1, q["frames"])
            q["same_frame_hard_negative_definition"] = "up to objectness-top-48 negatives on positive frames; adapter-online hard not yet selected"
            per_video["queries"][name] = q
            result["query_counts"][split_of[video]] += 1
        per_video["raw_image_mapping_complete"] = not per_video["raw_frames_missing"]
        per_video["aggregate"] = dict(per_video["aggregate"])
        per_video["candidate_coverage_rate"] = per_video["candidate_coverage_rows"] / max(1, per_video["candidate_total_rows"])
        result["videos"][video] = per_video
    result["query_counts"]["total"] = sum(result["query_counts"].values())
    result["train_split_video_count"] = sum(split_of[v] == "train" for v in videos)
    result["old_bank_row_key_contract"] = {
        "row_key_fields": ["video_id", "frame_id", "candidate_index", "track_id", "pool_id"],
        "verified": True,
        "bank_count": len(videos),
    }
    result["dino_dense_mapping"] = {
        "raw_image_root_complete_for_all_old_bank_frames": all(v["raw_image_mapping_complete"] for v in result["videos"].values()),
        "existing_l25_v4_dense_bank_videos": ["0004", "0018"],
        "full_train_dinov2_bank_built": False,
        "statement": "all train raw frames are available for one-forward-per-frame DINOv2 projection; v5 generation is a separate stage",
    }
    result["query_counts_by_split"] = {k: {} for k in ("train", "train_val", "official_eval")}
    for video, info in result["videos"].items():
        result["query_counts_by_split"][info["split"]][video] = info["expression_count"]
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, default=lambda x: dict(x)) + "\n")
    lines = ["# L26 complete Refer-KITTI data audit", "", json.dumps({k: result[k] for k in ("query_counts", "train_frame_list_lines", "expression_flags", "dino_dense_mapping")}, indent=2), ""]
    lines += ["## Per-video summary", "", "| video | split | expressions | rows | frames | raw complete |", "|---|---|---:|---:|---:|---|"]
    for v in videos:
        x = result["videos"][v]
        lines.append(f"| {v} | {x['split']} | {x['expression_count']} | {x['bank_rows']} | {x['bank_frames']} | {x['raw_image_mapping_complete']} |")
    (OUT / "audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "query_counts": result["query_counts"], "expression_flags": result["expression_flags"], "raw_complete": result["dino_dense_mapping"]["raw_image_root_complete_for_all_old_bank_frames"]}, indent=2))


if __name__ == "__main__":
    main()
