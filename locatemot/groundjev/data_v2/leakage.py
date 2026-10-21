"""Cross-dataset leakage and duplicate-source audit.

The ten datasets deliberately overlap in *source videos*: BURST re-annotates a
subset of TAO videos, RMOT26 draws query clips from MOT17/MOT20/DanceTrack, and
AerialMind shares VisDrone/UAVDT families.  Treating those as extra independent
videos would silently leak evaluation video content into training.

This module only *reports*; it never edits a dataset.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DANCE_VAL_ROOT = Path("/data2/user/LocateMOT/data/downloads/DanceTrack/extracted/val")
RMOT26_ANNOTATION = Path("/data3/user/LocateMOT_FullDataV2/downloads/RMOT26/train/annotations.json")
REFER_V2_EXPRESSION = Path("/data2/user/LocateMOT/data/refer_kitti_v2/expression")

# Registered fixed development scopes.  `train_inclusive=True` marks scopes whose
# video also appears in a training split: results on them are diagnostics, not
# held-out numbers.
EVAL_SCOPES: dict[str, dict[str, Any]] = {
    "dt1203": {
        "dataset": "dancetrack",
        "video_ids": {"dancetrack0004"},
        "frames": "1-1203",
        "train_inclusive": False,
        "note": "DanceTrack validation video; held out by the legal split",
    },
    "rmot_single800": {
        "dataset": "refer_kitti_v2",
        "video_ids": {"0007"},
        "frames": "1-800",
        "train_inclusive": False,
        "note": "Refer-KITTI-V2 validation video 0007",
    },
    "rmot_multi800": {
        "dataset": "refer_kitti_v2",
        "video_ids": {"0007"},
        "frames": "1-800",
        "train_inclusive": False,
        "note": "same video as rmot_single800, multi-target expressions",
    },
    "mot20_challenge": {
        "dataset": "mot20",
        "video_ids": {"MOT20-05"},
        "frames": "1659-3315",
        "train_inclusive": True,
        "note": "MOT20 TRAIN video MOT20-05; only a diagnostic range",
    },
    "tao_raw_full40": {
        "dataset": "tao_amodal",
        "video_ids": set(),       # resolved from the TAO annotations at runtime
        "frames": "40 annotation frames of one TAO TRAIN video",
        "train_inclusive": True,
        "note": "TAO TRAIN video; a diagnostic scope, never a held-out score",
    },
}


@dataclass
class LeakageFinding:
    check: str
    status: str            # PASS | FAIL | WARN | INFO
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def dance_val_sequences() -> set[str]:
    if not DANCE_VAL_ROOT.is_dir():
        return set()
    return {p.name for p in DANCE_VAL_ROOT.iterdir() if p.is_dir()}


def dance_train_sequences() -> set[str]:
    names: set[str] = set()
    for split in ("train1", "train2"):
        directory = DANCE_VAL_ROOT.parent / split
        if directory.is_dir():
            names |= {p.name for p in directory.iterdir() if p.is_dir()}
    return names


def refer_kitti_legal_sets() -> dict[str, tuple[str, ...]]:
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS

    key = "refer_kitti_v2" if "refer_kitti_v2" in LEGAL_SPLITS else "refer_kitti_v1"
    return {part: tuple(LEGAL_SPLITS[key][part])
            for part in ("fit", "calibration", "validation", "official_eval_reserved")}


def rmot26_source_counts() -> Counter[str]:
    if not RMOT26_ANNOTATION.is_file():
        return Counter()
    payload = json.loads(RMOT26_ANNOTATION.read_text())
    return Counter(sample["id"].split("_")[0] for sample in payload)


def check_dancetrack_boundary() -> LeakageFinding:
    train, val = dance_train_sequences(), dance_val_sequences()
    overlap = sorted(train & val)
    return LeakageFinding(
        "dancetrack_train_vs_val_videos",
        "PASS" if not overlap else "FAIL",
        {"train_videos": len(train), "val_videos": len(val), "overlap": overlap,
         "dt1203_video_in_val": "dancetrack0004" in val},
    )


def check_refer_kitti_boundary() -> LeakageFinding:
    parts = refer_kitti_legal_sets()
    dev = set(parts["fit"]) | set(parts["calibration"])
    held = set(parts["validation"]) | set(parts["official_eval_reserved"])
    overlap = sorted(dev & held)
    return LeakageFinding(
        "refer_kitti_dev_vs_heldout_videos",
        "PASS" if not overlap else "FAIL",
        {"dev_videos": len(dev), "heldout_videos": len(held), "overlap": overlap,
         "rmot800_video_held_out": "0007" in held},
    )


def check_rmot26_against_scopes() -> LeakageFinding:
    counts = rmot26_source_counts()
    val = dance_val_sequences()
    leaked = {video: counts[video] for video in sorted(set(counts) & val)}
    eval_hits = {}
    for name, scope in EVAL_SCOPES.items():
        hits = {video: counts[video] for video in scope["video_ids"] if counts.get(video)}
        if hits:
            eval_hits[name] = hits
    total_samples = sum(counts.values())
    leaked_samples = sum(leaked.values())
    return LeakageFinding(
        "rmot26_sources_inside_held_out_videos",
        "FAIL" if leaked else "PASS",
        {
            "rmot26_samples_total": total_samples,
            "rmot26_samples_from_held_out_dance_val": leaked_samples,
            "held_out_source_videos": leaked,
            "eval_scope_hits": eval_hits,
            "policy": "exclude every RMOT26 sample whose source video is a held-out video "
                      "before training; keep the record of the exclusion",
        },
    )


def check_burst_tao_identity() -> LeakageFinding:
    tao_path = Path("/data2/user/TAO-Amodal/annotations/train_with_freeform.json")
    burst_path = Path("/data2/user/TAO-Amodal/BURST_annotations/train/train_visibility.json")
    if not tao_path.is_file() or not burst_path.is_file():
        return LeakageFinding("burst_videos_subset_of_tao", "INFO", {"reason": "annotation file missing"})
    tao_videos = {v["name"].split("/")[-1] for v in json.loads(tao_path.read_text())["videos"]}
    burst_videos = {s["seq_name"] for s in json.loads(burst_path.read_text())["sequences"]}
    shared = burst_videos & tao_videos
    return LeakageFinding(
        "burst_videos_subset_of_tao",
        "WARN" if shared else "PASS",
        {"tao_train_videos": len(tao_videos), "burst_train_videos": len(burst_videos),
         "shared_videos": len(shared),
         "policy": "shared videos are counted once; BURST adds masks, not new videos"},
    )


def check_scope_train_inclusivity() -> LeakageFinding:
    inclusive = sorted(name for name, scope in EVAL_SCOPES.items() if scope["train_inclusive"])
    return LeakageFinding(
        "eval_scopes_present_in_training_data",
        "WARN" if inclusive else "PASS",
        {"train_inclusive_scopes": inclusive,
         "policy": "scopes listed here may be reported only as in-training diagnostics; "
                   "they are never held-out results",
         "held_out_scopes": sorted(name for name, scope in EVAL_SCOPES.items()
                                   if not scope["train_inclusive"])},
    )


def run_audit() -> dict[str, Any]:
    findings = [
        check_dancetrack_boundary(),
        check_refer_kitti_boundary(),
        check_rmot26_against_scopes(),
        check_burst_tao_identity(),
        check_scope_train_inclusivity(),
    ]
    return {
        "scopes": {name: {k: (sorted(v) if isinstance(v, set) else v)
                          for k, v in scope.items()} for name, scope in EVAL_SCOPES.items()},
        "findings": [finding.to_dict() for finding in findings],
        "status": "FAIL" if any(f.status == "FAIL" for f in findings)
                  else ("WARN" if any(f.status == "WARN" for f in findings) else "PASS"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FullDataV2 leakage audit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leakage_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({"status": report["status"],
                      "findings": {f["check"]: f["status"] for f in report["findings"]},
                      "out": str(args.out / "leakage_report.json")}, indent=1, ensure_ascii=False))
