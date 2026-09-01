#!/usr/bin/env python3
"""Audit train versus fixed held-out expression/domain strata without fitting."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
EXP = (ROOT / "outputs/l11/data/rmot_kitti/expressions.json", ROOT / "outputs/l16/data/kitti_missing/records/expressions.json")
SPLIT = ROOT / "outputs/l16/data/protocol/split_manifest.json"
FAST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
AUDIT = ROOT / "outputs/l26/audit/refer_kitti_complete_train/audit.json"
OUT = ROOT / "outputs/l26/fallback/F6_domain_length_audit"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expressions():
    rows = {}
    for path in EXP:
        for video, items in json.loads(path.read_text()).items():
            for item in items:
                rows[(str(video), str(item["expression"]))] = {"video": str(video), **item}
    return list(rows.values())


def flags(text: str) -> dict[str, bool]:
    s = text.lower()
    return {"motion": bool(re.search(r"motion|moving|direction|park|turn|drive|travel", s)),
            "color": bool(re.search(r"black|white|red|silver|blue|dark|light|color|hue", s)),
            "position": bool(re.search(r"left|right|front|back|behind|side|near|far", s)),
            "relation": bool(re.search(r"same|opposite|next|between|near", s))}


def summarize(rows):
    lengths = [len(str(x.get("sentence", x["expression"])).split()) for x in rows]
    chars = [len(str(x.get("sentence", x["expression"]))) for x in rows]
    f = {k: sum(flags(str(x.get("sentence", x["expression"])))[k] for x in rows) for k in ("motion", "color", "position", "relation")}
    by_video = Counter(x["video"] for x in rows)
    return {"queries": len(rows), "word_length": {"mean": sum(lengths) / max(1, len(lengths)), "min": min(lengths) if lengths else 0, "max": max(lengths) if lengths else 0}, "character_length": {"mean": sum(chars) / max(1, len(chars)), "min": min(chars) if chars else 0, "max": max(chars) if chars else 0}, "flag_counts": f, "video_counts": dict(sorted(by_video.items()))}


def main():
    split = json.loads(SPLIT.read_text())["kitti_v2"]
    split_of = {str(v): "train" for v in split["train"]}
    split_of.update({str(v): "train_val" for v in split["train_val"]})
    split_of.update({str(v): "official_eval" for v in split["official_eval"]})
    rows = expressions()
    fast = json.loads(FAST.read_text())
    screen_keys = {(str(x["video"]), str(x["expression"])) for x in fast["queries"] if x.get("split") == "screening"}
    train = [x for x in rows if split_of.get(x["video"]) == "train"]
    screen = [x for x in rows if (x["video"], str(x["expression"])) in screen_keys]
    audit = json.loads(AUDIT.read_text())
    video_rows = audit["videos"]
    strata = {}
    for name, subset in (("train", train), ("screening", screen)):
        s = summarize(subset)
        positive_frames = []; multi = []; candidate_rows = []; hard_rows = []; coverage = []
        for x in subset:
            item = video_rows.get(x["video"], {}).get("queries", {}).get(x["expression"])
            if item:
                positive_frames.append(item.get("positive_frames", 0)); multi.append(item.get("multi_positive_frames", 0)); candidate_rows.append(item.get("candidate_rows", 0)); hard_rows.append(item.get("same_frame_hard_negative_rows", 0)); coverage.append(item.get("coverage", 0.0))
        s["audit_aggregates"] = {"mean_positive_frames": sum(positive_frames) / max(1, len(positive_frames)), "mean_multi_positive_frames": sum(multi) / max(1, len(multi)), "mean_candidate_rows": sum(candidate_rows) / max(1, len(candidate_rows)), "mean_same_frame_hard_negative_rows": sum(hard_rows) / max(1, len(hard_rows)), "mean_coverage": sum(coverage) / max(1, len(coverage))}
        strata[name] = s
    report = {"format": "locatemot-l26-domain-shift-audit-v1", "selection_or_fitting": False, "split_manifest": str(SPLIT), "split_manifest_sha256": sha(SPLIT), "fast_manifest": str(FAST), "fast_manifest_sha256": sha(FAST), "official_gt_used": True, "gt_use": "read-only source audit aggregation; no model or threshold fitting", "train": strata["train"], "screening": strata["screening"], "screening_query_keys": sorted([list(x) for x in screen_keys]), "interpretation": "descriptive train/held-out strata only; not a candidate-selection decision"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "audit.md").write_text("# L26 F6 train/held-out domain and expression-length audit\n\nThis is descriptive only; no screening labels were used for fitting or threshold selection.\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
