"""Real-weight smoke test for TrackJEV native decisions (M3).

Runs the pretrained OneJev-0.8B on one legal MOT frame with GT-derived candidate
boxes.  Candidate boxes are GT here on purpose: this validates the *plumbing*
(prompt construction, option caps, answer parsing, assignment, memory questions)
and must never be reported as a tracking result.

Usage:
  python tools/groundjev/trackjev_native_smoke.py --device cuda:5 --out outputs/...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "OneJev"))

from locatemot.groundjev.data_v2 import mot_challenge
from locatemot.groundjev.trackjev import candidate_competition as cc
from locatemot.groundjev.trackjev import decision_schema as ds
from locatemot.groundjev.trackjev.model import TrackJEVConfig, TrackJEVModel
from locatemot.groundjev.trackjev.native_onejev_adapter import NativeOneJevAdapter

DEFAULT_CHECKPOINT = "/data2/user/LocateMOT_GroundJEV/models/OneJev-0.8B"
SEQ = Path("/data2/user/LocateMOT/data/MOT17/MOT17/train/MOT17-02-FRCNN")
FRAME = 2   # frame 1 exists so the causal history (previous-frame GT) is non-empty


def xywh_to_xyxy(box):
    x, y, w, h = (float(v) for v in box[:4])
    return [x, y, x + w, y + h]


def build_case():
    images = mot_challenge.list_images(mot_challenge.image_dir(SEQ))
    image_path = images[FRAME - 1]
    boxes = mot_challenge.parse_mot_gt(SEQ / "gt" / "gt.txt")
    frame_boxes = [b for b in boxes if b.frame == FRAME]
    # keep the case small: the first 6 GT objects of the frame
    frame_boxes = sorted(frame_boxes, key=lambda b: b.identity)[:6]
    gt_boxes = [b.box for b in frame_boxes]
    gt_ids = [b.identity for b in frame_boxes]
    # candidates = GT boxes plus one deliberate duplicate of the first box
    # Runtime GroundAnything/OnlineTracker candidates are pixel xyxy.  Keep
    # GT in MOT xywh separately for the supervision-only audit below.
    candidates = [{"box": xywh_to_xyxy(b.box)} for b in frame_boxes]
    duplicate = list(candidates[0]["box"])
    duplicate[0] += 3.0
    duplicate[1] += 2.0
    candidates.append({"box": duplicate})
    return str(image_path), candidates, gt_boxes, gt_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ask-risk", action="store_true")
    parser.add_argument("--batch-candidates", type=int, default=4)
    args = parser.parse_args()

    from PIL import Image

    image_path, candidates, gt_boxes, gt_ids = build_case()
    image = Image.open(image_path).convert("RGB")
    candidate_boxes = [c["box"] for c in candidates]
    edges = cc.sparse_edges(candidate_boxes, box_format="xyxy")
    competition = cc.competition_rows(
        candidate_boxes, cc.entity_clusters(candidate_boxes, edges), edges, box_format="xyxy")

    # causal memory built from the GT boxes of the *previous* frame only
    previous = [b for b in mot_challenge.parse_mot_gt(SEQ / "gt" / "gt.txt")
                if b.frame == FRAME - 1]
    tracks = [{"track_id": b.identity, "box": xywh_to_xyxy(b.box),
               "predicted_box": xywh_to_xyxy(b.box),
               "lost_age": 0, "age": 1, "state": "ACTIVE", "last_reliable_timestamp": 0.0}
              for b in previous[:8]]

    started = time.time()
    adapter = NativeOneJevAdapter(args.checkpoint, device=args.device,
                                  batch_candidates=args.batch_candidates,
                                  ask_risk=args.ask_risk)
    model = TrackJEVModel(adapter, TrackJEVConfig(use_competition=True,
                                                  assignment_conditioned_memory=True,
                                                  use_consequence_questions=args.ask_risk))
    step = model.step(image=image, candidates=candidates, tracks=tracks, timestamp=1.0,
                      specification={"task": "mot", "vocabulary": ["person"],
                                     "expression": None})
    elapsed = time.time() - started

    candidate_boxes_xywh = [[c["box"][0], c["box"][1],
                             c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]]
                            for c in candidates]
    supervision = cc.admission_targets(candidate_boxes_xywh, gt_boxes, gt_ids,
                                       clusters=cc.entity_clusters(
                                           candidate_boxes_xywh,
                                           cc.sparse_edges(candidate_boxes_xywh)))
    payload = {
        "status": "TRACKJEV_NATIVE_SMOKE_COMPLETE",
        "scope": ("plumbing validation on GT-derived candidates of one real MOT17 frame; "
                  "NOT a tracking result"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": args.device,
        "frame": {"sequence": SEQ.name, "frame": FRAME, "image": image_path,
                  "image_size": list(image.size)},
        "candidates": len(candidates),
        "history_tracks": len(tracks),
        "competition_groups": competition["competition_groups"],
        "admission_targets": [t.to_dict() for t in supervision],
        "elapsed_seconds": round(elapsed, 2),
        "step": step.to_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    decisions = step.decisions
    valid = [d for d in decisions if d.decision != "UNKNOWN"]
    print(json.dumps({
        "status": payload["status"],
        "candidates": len(candidates),
        "decisions": [{"i": d.candidate_index, "decision": d.decision, "id": d.track_id,
                       "p": round(d.probability, 3), "admitted": d.admitted} for d in decisions],
        "assignment": {str(k): v for k, v in step.assignment.items()},
        "memory": {str(k): v for k, v in step.memory_updates.items()},
        "valid_decisions": len(valid),
        "assignment_diagnostics": step.diagnostics["assignment_diagnostics"],
        "option_count": step.diagnostics["adapter_metadata"].get("option_count"),
        "shortlist_size": step.diagnostics["shortlist_size"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "out": str(args.out),
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
