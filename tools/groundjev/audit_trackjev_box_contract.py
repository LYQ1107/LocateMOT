"""Audit the box convention used by exported TrackJEV rows.

The native GroundAnything candidate contract is pixel ``xyxy`` while MOT GT
annotations are top-left ``xywh``.  This read-only audit compares the stored
competition matrices with both interpretations without changing the source
rows or any active training job.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _open_rows(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def _candidate_boxes(state: dict[str, Any]) -> list[list[float]]:
    return [list(candidate["box"]) for candidate in state.get("candidates", [])]


def _xyxy_to_xywh(box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1, y1, x2 - x1, y2 - y1]


def _iou_rows(payload: dict[str, Any]) -> list[str]:
    return list(payload.get("competition_iou_rows_milli") or [])


def audit(paths: list[Path], *, sample_limit: int = 5, max_rows: int | None = None) -> dict[str, Any]:
    from locatemot.groundjev.trackjev import candidate_competition as cc

    counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    scanned = 0
    for path in paths:
        with _open_rows(path) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                if max_rows is not None and scanned >= max_rows:
                    return {"status": "TRACKJEV_BOX_CONTRACT_AUDIT_COMPLETE",
                            "paths": [str(path_) for path_ in paths],
                            "counts": dict(counts), "samples": samples,
                            "row_limit": max_rows,
                            "interpretation": {
                                "native_runtime": "xyxy",
                                "mot_gt_matching": "xywh",
                                "active_manifest_unchanged": True,
                            }}
                row = json.loads(line)
                scanned += 1
                counts["rows"] += 1
                state = row.get("state") or {}
                boxes_xyxy = _candidate_boxes(state)
                stored = _iou_rows(state)
                if not boxes_xyxy or not stored:
                    counts["rows_without_competition"] += 1
                    continue
                counts["rows_with_competition"] += 1

                edges_xyxy = cc.sparse_edges(boxes_xyxy, box_format="xyxy")
                clusters_xyxy = cc.entity_clusters(boxes_xyxy, edges_xyxy)
                expected_xyxy = cc.competition_rows(
                    boxes_xyxy, clusters_xyxy, edges_xyxy, box_format="xyxy")
                boxes_xywh = [_xyxy_to_xywh(box) for box in boxes_xyxy]
                edges_xywh = cc.sparse_edges(boxes_xywh)
                clusters_xywh = cc.entity_clusters(boxes_xywh, edges_xywh)
                expected_xywh = cc.competition_rows(boxes_xywh, clusters_xywh, edges_xywh)
                # This is the historical runtime failure mode: raw xyxy values
                # passed to a function that assumes xywh, without conversion.
                edges_raw_xywh = cc.sparse_edges(boxes_xyxy)
                clusters_raw_xywh = cc.entity_clusters(boxes_xyxy, edges_raw_xywh)
                expected_raw_xywh = cc.competition_rows(
                    boxes_xyxy, clusters_raw_xywh, edges_raw_xywh)

                matches_xyxy = stored == _iou_rows(expected_xyxy)
                matches_gt_xywh = stored == _iou_rows(expected_xywh)
                matches_raw_xywh = stored == _iou_rows(expected_raw_xywh)
                if matches_xyxy:
                    counts["stored_matches_xyxy"] += 1
                if matches_gt_xywh:
                    counts["stored_matches_gt_converted_xywh"] += 1
                if matches_raw_xywh:
                    counts["stored_matches_raw_as_xywh"] += 1
                if not matches_xyxy and not matches_gt_xywh and not matches_raw_xywh:
                    counts["stored_matches_neither"] += 1
                if state.get("competition_box_format") == "xyxy":
                    counts["explicit_xyxy_tag"] += 1
                elif state.get("competition_box_format") == "xywh":
                    counts["explicit_xywh_tag"] += 1
                else:
                    counts["missing_box_format_tag"] += 1
                if len(samples) < sample_limit and (not matches_raw_xywh or not state.get("competition_box_format")):
                    samples.append({
                        "path": str(path), "line": line_number, "id": row.get("id"),
                        "candidates": len(boxes_xyxy),
                        "stored_matches_xyxy": matches_xyxy,
                        "stored_matches_gt_converted_xywh": matches_gt_xywh,
                        "stored_matches_raw_as_xywh": matches_raw_xywh,
                        "stored_first_row": stored[0] if stored else None,
                        "expected_xyxy_first_row": _iou_rows(expected_xyxy)[0],
                        "expected_gt_converted_xywh_first_row": _iou_rows(expected_xywh)[0],
                        "expected_raw_as_xywh_first_row": _iou_rows(expected_raw_xywh)[0],
                        "box_format_tag": state.get("competition_box_format"),
                    })
    return {"status": "TRACKJEV_BOX_CONTRACT_AUDIT_COMPLETE",
            "paths": [str(path) for path in paths],
            "counts": dict(counts), "samples": samples, "row_limit": max_rows,
            "interpretation": {
                "native_runtime": "xyxy",
                "mot_gt_matching": "xywh",
                "active_manifest_unchanged": True,
            }}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-rows", type=int,
                        help="bounded deterministic prefix audit; omit only for a deliberate full scan")
    args = parser.parse_args()
    result = audit(args.rows, max_rows=args.max_rows)
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
