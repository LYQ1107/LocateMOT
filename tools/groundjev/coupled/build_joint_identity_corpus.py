#!/usr/bin/env python3
"""Build the internal GroundJEV joint identity corpus from legal TRAIN rows.

This adapter does not rewrite GroundAnything's native JSONL schema.  It keeps
the causal state/media and places identity-choice, Yes/No, birth, lifecycle and
memory labels in an explicit internal record with validity masks.  Formal TEST
labels are rejected by the source/scope audit before any row is emitted.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object row in {path}")
                yield value


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _first_media(row: dict[str, Any]) -> str | None:
    media = row.get("media") or row.get("images") or []
    if not media:
        return None
    first = media[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        return first.get("path") or first.get("image_path") or first.get("url")
    return None


def _state(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("state") or {}
    if not isinstance(state, dict):
        raise ValueError("causal state must be an object")
    # Never copy a future/GT field into the model input.  The native exporters
    # already use historical_tracks/current candidates; this filter documents
    # the contract and catches accidental teacher injection.
    forbidden = {"future", "future_frames", "gt_boxes", "ground_truth", "target_ids"}
    leaked = sorted(key for key in state if key.lower() in forbidden)
    if leaked:
        raise ValueError(f"future/GT fields found in model state: {leaked}")
    return state


def _joint_from_yesno(row: dict[str, Any], index: int) -> dict[str, Any]:
    state = _state(row)
    target = row.get("target") or {}
    yes = "yes" in target
    proposal = state.get("yestrack_proposal") or {}
    histories = state.get("historical_tracks") or []
    options = []
    for track in histories:
        if isinstance(track, dict) and track.get("id") is not None:
            options.append({"track_id": int(track["id"]),
                            "lost_age": int(track.get("lost_age", 0) or 0),
                            "source": "causal_state"})
    return {
        "id": f"joint_yesno_{index}_{row.get('id', index)}",
        "split": str(row.get("split", "train")),
        "video_id": str(row.get("source") or state.get("video_id") or "unknown"),
        "frame_id": state.get("frame_id", state.get("frame_index", index)),
        "current_image": _first_media(row),
        "task": str(row.get("task", "identity_pair")),
        "task_query": (row.get("question") or {}).get("instructions", ""),
        "native_candidates": state.get("candidates") or state.get("current_candidates") or [],
        "candidate_visual_references": state.get("candidate_visual_references") or [],
        "past_track_memory": histories,
        "dynamic_identity_options": options + ["NEW", "NO_MATCH", "REJECT"],
        "identity_choice_target": proposal.get("proposed_identity") if proposal.get("kind") == "identity" else None,
        "identity_yesno_targets": {"yes": 1.0 if yes else 0.0,
                                    "valid": True,
                                    "proposal": proposal},
        "birth_target": {"value": None, "valid": False},
        "lifecycle_targets": {"value": ("REACTIVATE" if yes and proposal.get("kind") == "occlusion_recovery" else None),
                               "valid": proposal.get("kind") == "occlusion_recovery"},
        "memory_update_targets": {"value": None, "valid": False},
        "grounding_target": row.get("grounding_target"),
        "label_validity_masks": {"identity_choice": proposal.get("kind") == "identity" and yes,
                                  "yesno": True, "birth": False, "lifecycle": proposal.get("kind") == "occlusion_recovery",
                                  "memory": False},
        "proposal_source": row.get("supervision", {}).get("proposal_source", "native_yesno"),
        "source_provenance": {"row_id": row.get("id"), "family": row.get("supervision", {}).get("family"),
                              "future_evidence": bool(row.get("supervision", {}).get("future_evidence", False)),
                              "official_test_labels_read": bool(row.get("supervision", {}).get("official_test_labels_read", False))},
    }


def _joint_from_question(row: dict[str, Any], index: int) -> dict[str, Any]:
    target = row.get("source_target") or {}
    label = str(row.get("label") or (next(iter(target)) if target else "UNKNOWN"))
    state = _state(row)
    histories = state.get("historical_tracks") or []
    options = [int(track["id"]) for track in histories
               if isinstance(track, dict) and track.get("id") is not None]
    return {
        "id": f"joint_choice_{index}_{row.get('id', index)}",
        "split": str(row.get("split", "train")),
        "video_id": str(row.get("unit") or "unknown"),
        "frame_id": state.get("frame_id", state.get("frame_index", index)),
        "current_image": _first_media(row),
        "task": str(row.get("task", "identity")),
        "task_query": str(row.get("question", "")),
        "native_candidates": state.get("candidates") or state.get("current_candidates") or [],
        "candidate_visual_references": state.get("candidate_visual_references") or [],
        "past_track_memory": histories,
        "dynamic_identity_options": options + ["NEW", "NO_MATCH", "REJECT"],
        "identity_choice_target": label if label in set(map(str, options)) | {"NEW", "NO_MATCH", "REJECT"} else None,
        "identity_yesno_targets": {"yes": None, "valid": False},
        "birth_target": {"value": label == "NEW", "valid": label in {"NEW", "MATCH", "REACTIVATE"}},
        "lifecycle_targets": {"value": "REACTIVATE" if label == "REACTIVATE" else None,
                               "valid": label == "REACTIVATE"},
        "memory_update_targets": {"value": None, "valid": False},
        "grounding_target": row.get("grounding_target"),
        "label_validity_masks": {"identity_choice": label != "UNKNOWN", "yesno": False,
                                  "birth": label in {"NEW", "MATCH", "REACTIVATE"},
                                  "lifecycle": label == "REACTIVATE", "memory": False},
        "proposal_source": "trackjev_choice_export",
        "source_provenance": {"row_id": row.get("id"), "family": "identity_choice",
                              "future_evidence": False, "official_test_labels_read": False},
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for index, row in enumerate(read_rows(args.yesno)):
        if args.max_rows and len(rows) >= args.max_rows:
            break
        record = _joint_from_yesno(row, index)
        rows.append(record)
        counts["yesno"] = counts.get("yesno", 0) + 1
    if args.questions and args.questions.is_file():
        for index, row in enumerate(read_rows(args.questions), start=len(rows)):
            if args.max_rows and len(rows) >= args.max_rows:
                break
            # MOT20-05 is an independent held-out scope in this project and
            # must not enter a training corpus through old question exports.
            if "MOT20-05" in str(row.get("unit", "")):
                continue
            record = _joint_from_question(row, index)
            rows.append(record)
            counts["choice"] = counts.get("choice", 0) + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "format": "groundjev-coupled-corpus-v1",
        "status": "COUPLED_INTERNAL_CORPUS_BUILT",
        "rows": str(args.out.resolve()),
        "rows_sha256": digest(args.out),
        "row_count": len(rows),
        "counts": counts,
        "yesno_source": str(args.yesno.resolve()),
        "yesno_source_sha256": digest(args.yesno),
        "choice_source": str(args.questions.resolve()) if args.questions else None,
        "choice_source_sha256": digest(args.questions) if args.questions else None,
        "official_test_labels_read": False,
        "future_evidence_in_model_input": False,
        "groundtruth_teacher_injection": False,
        "stable_seed_source_version": "native-yesno-v2-stable-sha-seed",
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yesno", type=Path, required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
