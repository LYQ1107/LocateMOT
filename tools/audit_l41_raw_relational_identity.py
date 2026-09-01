#!/usr/bin/env python3
"""Stage L41 reversible pair and provenance contract audit."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l41_raw_data import (FIT_VIDEOS, HELDOUT_VIDEOS, WEIGHTS, crop_box, load_fragments,
                                make_pairs, sha256)


def main():
    assert Path.cwd().resolve() == ROOT
    out = ROOT / "outputs/l41/audit"; out.mkdir(parents=True, exist_ok=True); started = time.time()
    videos = FIT_VIDEOS + HELDOUT_VIDEOS; fragments, alignment = load_fragments(videos); pairs = make_pairs(fragments)
    missing = []; invalid = []; pair_keys = set(); duplicate = 0; dt = []; kinds = Counter(); cross_source = Counter()
    for p in pairs:
        a, b = fragments[p["a"]], fragments[p["b"]]; kinds[p["kind"]] += 1
        key = (a["video"], p["frame"], a["track_id"], b["track_id"], p["kind"]); duplicate += int(key in pair_keys); pair_keys.add(key)
        if a["source"] == 0 and b["source"] == 1: cross_source["main_to_reserve"] += int(p["label"])
        for f in (a, b):
            ob = min(f["obs"], key=lambda x: abs(x["frame"] - p["frame"]))
            path = Path(ob["image"])
            if not path.exists(): missing.append(str(path)); continue
            with Image.open(path) as im:
                box = crop_box(ob["box"], im.width, im.height)
                if box[2] <= box[0] or box[3] <= box[1]: invalid.append({"path": str(path), "box": ob["box"]})
        dt.append(abs(a["obs"][-1]["frame"] - b["obs"][-1]["frame"]))
    labels = {"same_gt_cross_frame_or_fragment": "GT_PRIVILEGED_ORACLE", "same_frame_different_gt_hard": "GT_PRIVILEGED_ORACLE", "inactive": "GT_PRIVILEGED_ORACLE"}
    payload = {"schema_version": "locatemot-l41-relational-identity-contract-v1", "stage": "L41", "project_root": str(ROOT), "started_at": started, "completed_at": time.time(), "fit_videos": list(FIT_VIDEOS), "heldout_videos": list(HELDOUT_VIDEOS), "fragment_count": len(fragments), "pair_count": len(pairs), "positive_pair_count": sum(x["label"] for x in pairs), "hard_negative_pair_count": sum(x["kind"] == "same_frame_different_gt_hard" for x in pairs), "inactive_pair_count": sum(x["kind"] == "inactive" for x in pairs), "pair_kind_counts": dict(kinds), "pair_key": "(video, frame, track/fragment, observation pair)", "duplicate_pair_keys": duplicate, "missing_raw_paths": len(missing), "invalid_crops": len(invalid), "raw_image_reversible": not missing and not invalid, "time_gap_stats": {"min": int(min(dt)), "median": float(torch.tensor(dt).float().median()), "max": int(max(dt))} if dt else {}, "main_to_reserve_positive_pairs": int(cross_source["main_to_reserve"]), "bank_cache_alignment_fragments": len(alignment), "weights": str(WEIGHTS), "weights_exists": WEIGHTS.exists(), "weights_sha256": sha256(WEIGHTS) if WEIGHTS.exists() else None, "visual_encoder": "frozen OpenAI CLIP ViT-B/16 spatial patch tokens reduced to 2x2 cells; no persistent cache", "model_inputs": ["paired crop patch tokens", "relative geometry", "time gap", "motion", "lifecycle", "context/objectness"], "semantic_inputs_excluded": ["expression", "source_id", "pool_id", "group_id", "state_key"], "source_usage": "diagnostic stratification only", "labels": labels, "fixed_screening_gt_used": False, "decision": {"enter_l41_smoke": bool(not missing and not invalid and duplicate == 0 and WEIGHTS.exists())}, "elapsed_sec": time.time() - started}
    (out / "relational_identity_contract.json").write_text(json.dumps(payload, indent=2) + "\n"); (out / "README.md").write_text("# L41 relational identity contract\n\nPairs are reversible to two raw observation crops. Image pixels and patch tokens are streaming/in-memory only.\n")
    if not payload["decision"]["enter_l41_smoke"]: (out / "NO_RELATIONAL_IDENTITY_INPUT.md").write_text("# NO_RELATIONAL_IDENTITY_INPUT\n\nThe L41 pair contract failed.\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
