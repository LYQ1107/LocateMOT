#!/usr/bin/env python3
"""Build an auditable GroundJEV joint identity corpus.

The input rows are the already registered MOT/RMOT/OVMOT supervision rows.
This tool only normalizes their schema and records provenance; it never
creates a candidate box, identity target, or lifecycle label.  In particular,
OVMOT teacher crops remain explicitly marked as teacher rows until a complete
native TAO TRAIN candidate export is joined later.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SCHEMA_VERSION = "groundjev-joint-identity-v1"
_MOT_TASKS = {"mot", "identity", "identity_pair", "occlusion_recovery", "lifecycle"}
_RMOT_TASKS = {"rmot", "query_membership", "membership"}
_OVMOT_TASKS = {"ovmot", "category_group", "category_leaf"}


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield line_number, row


def _task_name(row: Mapping[str, Any]) -> str:
    task = str(row.get("task", "")).lower()
    if task in _MOT_TASKS:
        nested = row.get("state", {}).get("task", {}) if isinstance(row.get("state"), Mapping) else {}
        nested_task = str(nested.get("task", "")).lower()
        return "mot" if nested_task in {"", "mot"} else nested_task
    if task in _RMOT_TASKS:
        return "rmot"
    if task in _OVMOT_TASKS:
        return "ovmot"
    nested = row.get("state", {}).get("task", {}) if isinstance(row.get("state"), Mapping) else {}
    nested_task = str(nested.get("task", "")).lower()
    if nested_task in {"mot", "rmot", "ovmot"}:
        return nested_task
    raise ValueError(f"cannot infer MOT/RMOT/OVMOT task from row {row.get('id')!r}")


def _first_image(row: Mapping[str, Any]) -> str | None:
    for item in row.get("media", []) or []:
        if isinstance(item, Mapping) and item.get("type", "image") == "image" and item.get("path"):
            return str(item["path"])
    value = row.get("current_image")
    return None if value is None else str(value)


def _frame_id(row: Mapping[str, Any]) -> int | float | None:
    for key in ("frame_id", "frame"):
        if row.get(key) is not None:
            value = row[key]
            return value if isinstance(value, (int, float)) else str(value)
    supervision = row.get("supervision")
    if isinstance(supervision, Mapping) and supervision.get("frame") is not None:
        return supervision["frame"]
    state = row.get("state")
    if isinstance(state, Mapping) and state.get("timestamp") is not None:
        return state["timestamp"]
    return None


def _candidate_source(row: Mapping[str, Any]) -> str:
    if row.get("proposal_source"):
        return str(row["proposal_source"])
    supervision = row.get("supervision")
    if isinstance(supervision, Mapping) and supervision.get("candidate_source"):
        return str(supervision["candidate_source"])
    raise ValueError(f"row {row.get('id')!r} has no explicit candidate_source/proposal_source")


def _state_value(row: Mapping[str, Any], key: str, default: Any) -> Any:
    if key in row:
        return copy.deepcopy(row[key])
    state = row.get("state")
    if isinstance(state, Mapping):
        return copy.deepcopy(state.get(key, default))
    return copy.deepcopy(default)


def _targets(row: Mapping[str, Any], task: str) -> dict[str, Any]:
    """Copy registered targets without inventing missing supervision."""
    target = row.get("target")
    yesno = row.get("identity_yesno_targets")
    if yesno is None and task == "rmot" and isinstance(target, Mapping) and ({"yes", "no"} & set(target)):
        yesno = {"yes": target.get("yes"), "no": target.get("no"), "valid": True}
    return {
        "identity_choice": copy.deepcopy(row.get("identity_choice_target", target if task == "mot" else None)),
        "identity_yesno": copy.deepcopy(yesno),
        "birth": copy.deepcopy(row.get("birth_target")),
        "lifecycle": copy.deepcopy(row.get("lifecycle_targets")),
        "memory_update": copy.deepcopy(row.get("memory_update_targets")),
        "grounding": copy.deepcopy(row.get("grounding_target")),
    }


def normalize_row(row: Mapping[str, Any], *, source_path: Path, source_row: int,
                  reject_forbidden: bool = True) -> dict[str, Any]:
    """Normalize one registered row while preserving its supervision boundary."""
    task = _task_name(row)
    supervision = row.get("supervision") if isinstance(row.get("supervision"), Mapping) else {}
    provenance = row.get("source_provenance") if isinstance(row.get("source_provenance"), Mapping) else {}
    future = bool(provenance.get("future_evidence", supervision.get("future_images_read", False)))
    official_test = bool(provenance.get("official_test_labels_read", False))
    if reject_forbidden and (future or official_test):
        raise ValueError(
            f"{source_path}:{source_row}: forbidden future/test evidence in row {row.get('id')!r}"
        )
    proposal_source = _candidate_source(row)
    state = row.get("state") if isinstance(row.get("state"), Mapping) else {}
    candidates = row.get("native_candidates", state.get("candidates", []))
    candidate_refs = row.get("candidate_visual_references", [])
    memory = row.get("past_track_memory", state.get("historical_tracks", []))
    options = row.get("dynamic_identity_options")
    if options is None:
        options = []
        for track in memory or []:
            if isinstance(track, Mapping) and track.get("id") is not None:
                options.append({"track_id": int(track["id"]), "source": "causal_state"})
        if task == "mot":
            options.extend(["NEW", "NO_MATCH", "REJECT"])
    targets = _targets(row, task)
    return {
        "id": str(row.get("id", f"{source_path.name}:{source_row}")),
        "split": str(row.get("split", "train")),
        "video_id": str(row.get("video_id", row.get("source", "unknown"))),
        "frame_id": _frame_id(row),
        "current_image": _first_image(row),
        "task": task,
        "task_query": row.get("task_query", row.get("question", {}).get("instructions")
                            if isinstance(row.get("question"), Mapping) else None),
        "native_candidates": copy.deepcopy(candidates),
        "candidate_visual_references": copy.deepcopy(candidate_refs),
        "past_track_memory": copy.deepcopy(memory),
        "dynamic_identity_options": copy.deepcopy(options),
        "identity_choice_target": targets["identity_choice"],
        "identity_yesno_targets": targets["identity_yesno"],
        "birth_target": targets["birth"],
        "lifecycle_targets": targets["lifecycle"],
        "memory_update_targets": targets["memory_update"],
        "grounding_target": targets["grounding"],
        "label_validity_masks": copy.deepcopy(row.get("label_validity_masks", {})),
        "proposal_source": proposal_source,
        "source_provenance": {
            **copy.deepcopy(dict(provenance)),
            "source_path": str(source_path.resolve()),
            "source_row": int(source_row),
            "future_evidence": future,
            "official_test_labels_read": official_test,
            "gt_in_model_input": bool(
                provenance.get("gt_in_model_input", False)
                or supervision.get("GT_IDs_in_model_input", False)
                or supervision.get("GT_class_or_identity_in_state", False)
            ),
        },
    }


def _is_teacher_source(source: str) -> bool:
    lower = source.lower()
    return any(token in lower for token in ("teacher", "gt_", "ground_truth", "crop"))


def build(input_paths: Iterable[Path], *, output: Path, limit_per_source: int | None = None,
          require_native_candidates: bool = False) -> dict[str, Any]:
    input_paths = [Path(path).resolve() for path in input_paths]
    all_rows: list[dict[str, Any]] = []
    input_receipts = []
    for path in input_paths:
        count = 0
        for line_number, raw in iter_rows(path):
            if limit_per_source is not None and count >= int(limit_per_source):
                break
            normalized = normalize_row(raw, source_path=path, source_row=line_number)
            source = str(normalized["proposal_source"])
            if require_native_candidates and (not normalized["native_candidates"] or _is_teacher_source(source)):
                raise ValueError(
                    f"{path}:{line_number}: row is not a native candidate row ({source})"
                )
            all_rows.append(normalized)
            count += 1
        input_receipts.append({"path": str(path), "rows_read": count, "sha256": sha256_file(path)})
    if not all_rows:
        raise ValueError("no rows were built")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tasks = Counter(row["task"] for row in all_rows)
    sources = Counter(row["proposal_source"] for row in all_rows)
    native_rows = sum(bool(row["native_candidates"]) and not _is_teacher_source(row["proposal_source"])
                      for row in all_rows)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "rows": len(all_rows),
        "task_counts": dict(sorted(tasks.items())),
        "proposal_source_counts": dict(sorted(sources.items())),
        "native_candidate_rows": native_rows,
        "teacher_or_empty_candidate_rows": len(all_rows) - native_rows,
        "native_ovmot_training_rows": bool(
            tasks.get("ovmot", 0)
            and all(row["task"] != "ovmot" or (
                bool(row["native_candidates"]) and not _is_teacher_source(row["proposal_source"])
            ) for row in all_rows)
        ),
        "input_files": input_receipts,
        "forbidden_rows_rejected": True,
        "future_evidence_in_model_input": False,
        "official_test_labels_read": False,
    }
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    manifest.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", dest="inputs", required=True,
                        help="registered JSONL source; repeat for MOT/RMOT/OVMOT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-source", type=int)
    parser.add_argument("--require-native-candidates", action="store_true",
                        help="reject GT/teacher rows and empty candidate rows")
    args = parser.parse_args()
    receipt = build([Path(path) for path in args.inputs], output=Path(args.output),
                    limit_per_source=args.limit_per_source,
                    require_native_candidates=args.require_native_candidates)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
