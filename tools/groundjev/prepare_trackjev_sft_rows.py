"""Export TrackJEV SFT rows from native GroundAnything candidates plus real GT.

Row schema follows the upstream OneJev SFT format exactly:

    {id, split, source, task, state, question, target, media, supervision}

with one row per question: ``admit_<i>`` (noul), ``identity_<i>`` (choice),
``memory_<i>`` (noul, assignment-conditioned) and optionally ``risk_<i>`` (score).

Hard rules (they are what make this export different from the historical one):

* identity shortlist is **causal**: it only contains identities seen before this
  frame, with their real lost-age; the true id is never injected;
* a candidate whose true id is *not* in the shortlist is exported as an
  ``admit`` row and an ``UNKNOWN``-free ``REJECT``/no-identity row is *not*
  fabricated: the identity row is skipped and the miss is counted;
* memory rows exist only for candidates that receive a known identity, so the
  loss can never train ``UPDATE`` without an assignment;
* duplicate candidates are supervised ``REJECT`` because GT says another box
  already owns that entity — this is the supervision the old pipeline lacked.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locatemot.groundjev.data_v2 import mot_challenge
from locatemot.groundjev.trackjev import candidate_competition as cc
from locatemot.groundjev.trackjev import decision_schema as ds

SHORTLIST_WINDOW = 30
# evidence sheets live OUTSIDE the regenerable row shards so a re-export cannot
# delete media that a running training job is streaming
MEDIA_ROOT = ROOT / "outputs" / "groundjev" / "full_data_v2" / "trackjev_media"
SHORTLIST_SIZE = 31   # matches MAX_SHORTLIST used at inference


def build_crop_sheet(image, boxes, out_path: Path, cell=(160, 192), columns=4):
    from PIL import Image, ImageDraw

    rows = max(1, (len(boxes) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "white")
    draw = ImageDraw.Draw(sheet)
    for index, box in enumerate(boxes):
        x, y = (index % columns) * cell[0], (index // columns) * cell[1]
        draw.text((x + 2, y + 2), f"Candidate {index}", fill="black")
        # GroundAnything candidates are pixel xyxy.  Keep the crop and state
        # geometry in that convention; GT remains MOT xywh below.
        crop = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        crop = crop.convert("RGB")
        crop.thumbnail((cell[0] - 6, cell[1] - 26))
        sheet.paste(crop, (x + 3, y + 22))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path



def _iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    """Candidate boxes are xyxy (perception/PIL); GT boxes are xywh (MOT format)."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return [x1, y1, x2 - x1, y2 - y1]


def _identity_frames(gt: list[Any]) -> dict[int, list[int]]:
    frames: dict[int, list[int]] = defaultdict(list)
    for box in gt:
        frames[box.identity].append(box.frame)
    return {identity: sorted(values) for identity, values in frames.items()}


def load_candidates(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def export_video(*, candidates_path: Path, image_root: Path, gt_path: Path,
                 video: str, out_dir: Path, fps: float = 30.0,
                 annotation_exhaustive: bool = True,
                 max_frames: Optional[int] = None,
                 seed_history_from_gt: bool = False) -> dict[str, Any]:
    gt = mot_challenge.parse_mot_gt(gt_path)
    if candidates_path is None:
        # PLUMBING-ONLY mode: candidates are GT boxes plus one deliberate duplicate
        # per object.  Never use these rows as training data.
        frame_rows = []
        for frame in sorted({b.frame for b in gt})[: (max_frames or 5)]:
            boxes = []
            for b in [x for x in gt if x.frame == frame][:6]:
                boxes.append({"box": list(b.box), "label": "person"})
                dup = list(b.box)
                dup[0] += 3.0
                dup[1] += 2.0
                boxes.append({"box": dup, "label": "person"})
            frame_rows.append({"frame": frame, "candidates": boxes,
                               "image_path": str(image_root / f"{frame:06d}.jpg"),
                               "candidate_source": "GT_DERIVED_PLUMBING_ONLY"})
    else:
        frame_rows = load_candidates(candidates_path)
    gt_by_frame: dict[int, list[Any]] = defaultdict(list)
    for box in gt:
        gt_by_frame[box.frame].append(box)

    last_seen: dict[int, int] = {}
    last_box: dict[int, list[float]] = {}
    seen_ids: set[int] = set()
    if seed_history_from_gt and frame_rows:
        first_frame = int(frame_rows[0].get("frame"))
        # identities genuinely observed EARLIER in this TRAIN video are legitimate
        # memory content; nothing from the current or future frames is injected.
        by_frame: dict[int, dict[int, list[float]]] = {}
        for box in gt:
            if box.frame < first_frame:
                by_frame.setdefault(box.frame, {})[box.identity] = list(box.box)
        for identity, frames in _identity_frames(gt).items():
            prior = [f for f in frames if f < first_frame]
            if prior:
                last_frame = max(prior)
                last_seen[identity] = last_frame
                last_box[identity] = by_frame[last_frame][identity]
                if first_frame - last_frame <= SHORTLIST_WINDOW:
                    seen_ids.add(identity)
    rows_written = 0
    counters: Counter[str] = Counter()
    out_rows = out_dir / "rows.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    with out_rows.open("w") as output:
        for entry in frame_rows[: max_frames] if max_frames else frame_rows:
            frame = int(entry.get("frame"))
            candidates = entry.get("candidates") or []
            if not candidates:
                counters["frames_without_candidates"] += 1
                continue
            candidate_boxes = [list(c["box"]) for c in candidates]  # native pixel xyxy
            boxes = [xyxy_to_xywh(box) for box in candidate_boxes]  # GT matching only: MOT xywh
            image_path = Path(entry.get("image_path") or (image_root / f"{frame:06d}.jpg"))
            if not image_path.is_file():
                counters["missing_images"] += 1
                continue
            frame_gt = gt_by_frame.get(frame, [])
            gt_boxes = [b.box for b in frame_gt]
            gt_ids = [b.identity for b in frame_gt]
            visibility_by_id = {b.identity: b.visibility for b in frame_gt}

            edges = cc.sparse_edges(candidate_boxes, None, box_format="xyxy")
            clusters = cc.entity_clusters(boxes, edges)
            competition = cc.competition_rows(candidate_boxes, clusters, edges, box_format="xyxy")
            admission = cc.admission_targets(boxes, gt_boxes, gt_ids, clusters=clusters,
                                             annotation_exhaustive=annotation_exhaustive)

            # causal shortlist: identities already seen, ordered by recency of loss
            def retrieval_score(identity: int) -> float:
                box = last_box.get(identity)
                if box is None:
                    return 0.0
                return max((_iou_xywh(box, cbox) for cbox in boxes), default=0.0)

            candidates_in_window = [i for i in seen_ids
                                    if frame - last_seen.get(i, -10 ** 9) <= SHORTLIST_WINDOW]
            ranked = sorted(candidates_in_window,
                            key=lambda i: (-retrieval_score(i), frame - last_seen.get(i, 0), i))
            shortlist = ranked[:SHORTLIST_SIZE]
            options = ds.build_identity_options(
                [{"track_id": i, "lost_age": max(0, frame - last_seen.get(i, frame) - 1)}
                 for i in shortlist])

            image = Image.open(image_path).convert("RGB")
            sheet_path = MEDIA_ROOT / video / f"{frame:06d}_candidates.png"
            build_crop_sheet(image, candidate_boxes, sheet_path)
            media = [
                {"type": "image", "path": str(image_path), "w": image.width, "h": image.height},
                {"type": "image", "path": str(sheet_path),
                 "w": 4 * 160, "h": max(1, (len(boxes) + 3) // 4) * 192},
            ]
            state = {
                "current_frame": "<image:1>",
                "timestamp": (frame - 1) / fps,
                "task": {"task": "mot", "vocabulary": ["person"], "expression": None},
                "candidates": [{"index": i, "box": list(c["box"]),
                                **({"label": c["label"]} if c.get("label") else {})}
                               for i, c in enumerate(candidates)],
                "historical_tracks": [
                    {"id": t, "lost_age": max(0, frame - last_seen.get(t, frame) - 1),
                     "state": "ACTIVE" if frame - last_seen.get(t, frame) == 1 else "LOST"}
                    for t in shortlist],
                "current_candidate_crop_sheet": "<image:2>",
                "identity_option_limit": SHORTLIST_SIZE,
                **competition,
            }
            supervision_meta = {
                "proposal_source": "native_G0",
                "frame": frame,
                "candidate_source": str(candidates_path),
                "candidate_source_kind": entry.get("candidate_source", "native_export"),
                "gt_labels_outside_model_state": True,
                "actor_state_corrected_by_GT": False,
                "shortlist_policy": "causal ids seen within window, ordered by recency",
                "shortlist_ids": shortlist,
                "annotation_exhaustive": annotation_exhaustive,
            }

            for target in admission:
                index = target.candidate
                prefix = f"{video}_{frame:06d}_c{index}"
                # ---- admission row
                if target.label in ("valid_entity", "duplicate"):
                    output.write(json.dumps({
                        "id": f"{prefix}_admit", "split": "train", "source": video, "task": "identity",
                        "state": state, "media": media,
                        "question": {"type": "noul",
                                     "instructions": ds.ADMIT_TEXT.format(index=index)},
                        "target": {"yes" if target.label == "valid_entity" else "no": 1.0},
                        "supervision": {**supervision_meta, "admission_label": target.label,
                                        "matched_gt": target.matched_gt,
                                        "duplicate_of": target.duplicate_of},
                    }) + "\n")
                    rows_written += 1
                    counters["admit_rows"] += 1

                # ---- identity row
                identity_target: Optional[str] = None
                if target.label == "duplicate":
                    identity_target = ds.REJECT_TOKEN
                    counters["reject_targets"] += 1
                elif target.label == "valid_entity" and target.matched_gt is not None:
                    if target.matched_gt in set(shortlist):
                        identity_target = str(int(target.matched_gt))
                        counters["match_targets"] += 1
                    else:
                        # the true id exists but was not retrieved: the honest target is
                        # NO_MATCH (hold), never NEW and never a silently dropped row
                        identity_target = ds.NO_MATCH_TOKEN
                        counters["no_match_targets"] += 1
                if identity_target is None:
                    continue
                output.write(json.dumps({
                    "id": f"{prefix}_identity", "split": "train", "source": video, "task": "identity",
                    "state": state, "media": media,
                    "question": {"type": "choice",
                                 "instructions": (f"Which identity owns current candidate {index}? "
                                                  "Compare visual content, history, motion and "
                                                  "competing identities. Never use a class label "
                                                  "alone as identity."),
                                 "criteria": {o.token: o.text for o in options}},
                    "target": {identity_target: 1.0},
                    "supervision": {**supervision_meta, "identity_target": identity_target},
                }) + "\n")
                rows_written += 1
                counters["identity_rows"] += 1

                # ---- assignment-conditioned memory row (BOTH classes)
                # yes: a real identity was assigned AND the observation is reliable
                # no : duplicate/NO_MATCH, or an assigned id observed under heavy occlusion
                if identity_target is not None:
                    visibility = visibility_by_id.get(target.matched_gt) if target.matched_gt is not None else None
                    reliable = (identity_target not in (ds.REJECT_TOKEN, ds.NO_MATCH_TOKEN)
                                and (visibility is None or visibility >= 0.6))
                    output.write(json.dumps({
                        "id": f"{prefix}_memory", "split": "train", "source": video, "task": "identity",
                        "state": {**state, "assigned_identity": identity_target},
                        "media": media,
                        "question": {"type": "noul",
                                     "instructions": ds.MEMORY_TEXT.format(index=index,
                                                                           track_id=identity_target)},
                        "target": {"yes" if reliable else "no": 1.0},
                        "supervision": {**supervision_meta, "assigned_identity": identity_target,
                                        "memory_condition": "final assignment",
                                        "memory_target_source": ("reliable_assigned_observation"
                                                                 if reliable else
                                                                 "duplicate_no_match_or_occluded"),
                                        "visibility": visibility},
                    }) + "\n")
                    rows_written += 1
                    counters["memory_rows"] += 1
                    counters["memory_yes" if reliable else "memory_no"] += 1

            last_seen.update({b.identity: frame for b in frame_gt})
            last_box.update({b.identity: list(b.box) for b in frame_gt})
            seen_ids.update(seen_ids | {b.identity for b in frame_gt})
            counters["frames"] += 1

    summary = {
        "video": video,
        "rows_file": str(out_rows),
        "rows": rows_written,
        "counters": dict(counters),
        "shortlist_policy": {"window": SHORTLIST_WINDOW, "size": SHORTLIST_SIZE},
        "candidates_source": str(candidates_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=None,
                        help="native candidate jsonl; omit with --gt-candidates for a plumbing test")
    parser.add_argument("--gt-candidates", action="store_true")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed-history-from-gt", action="store_true")
    args = parser.parse_args()
    summary = export_video(candidates_path=args.candidates, image_root=args.image_root,
                           gt_path=args.gt, video=args.video, out_dir=args.out,
                           fps=args.fps, max_frames=args.max_frames,
                           seed_history_from_gt=args.seed_history_from_gt)
    print(json.dumps(summary, indent=1, ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
