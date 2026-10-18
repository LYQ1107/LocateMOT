"""Small CPU-only helpers shared by YesTrack verification data tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from locatemot.groundjev.yes_verification import assert_prompt_has_no_ground_truth


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"record at line {line_number} is not an object")
            records.append(value)
        return records
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return [record for record in value if isinstance(record, dict)]
    if isinstance(value, dict):
        for key in ("rows", "records", "items", "candidates"):
            if isinstance(value.get(key), list):
                return [record for record in value[key] if isinstance(record, dict)]
        return [value]
    raise ValueError(f"unsupported input record container: {path}")


def first(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def as_media(record: dict[str, Any]) -> list[Any]:
    value = first(record, "media", "images", "image", "image_path", "path", default=[])
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, list):
        return value
    return []


def infer_video(record: dict[str, Any]) -> str:
    return str(first(record, "video", "sequence", "video_id", "source_video", default="unknown"))


def infer_frame(record: dict[str, Any]) -> int:
    return int(first(record, "frame", "frame_idx", "timestamp", default=0))


def infer_split(record: dict[str, Any], default: str = "train") -> str:
    return str(first(record, "split", "partition", default=default))


def label_from_record(record: dict[str, Any]) -> int | None:
    value = first(record, "target", "target_yes", "is_target", "label", "y_true", "same_identity")
    if value is None:
        return None
    if isinstance(value, str):
        return int(value.strip().lower() in {"yes", "true", "1", "positive"})
    return int(bool(value))


def question_contains_future(record: dict[str, Any], current_frame: int) -> bool:
    if bool(first(record, "future", "future_frame", "uses_future", default=False)):
        return True
    for key in ("history_frames", "evidence_frames"):
        values = record.get(key)
        if isinstance(values, list) and any(int(value) >= current_frame for value in values):
            return True
    return False


def row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def make_row(
    *,
    record: dict[str, Any],
    task: str,
    question: str,
    target: int | None,
    question_type: str,
    proposal_source: str,
    media: Iterable[Any] | None = None,
    current_frame: int | None = None,
    expression: str | None = None,
) -> dict[str, Any]:
    assert_prompt_has_no_ground_truth(question)
    frame = infer_frame(record) if current_frame is None else int(current_frame)
    media_values = list(as_media(record) if media is None else media)
    if question_contains_future(record, frame):
        raise ValueError(f"future evidence is not legal for frame {frame}")
    return {
        "id": str(first(record, "id", "row_id", default=row_hash(record))),
        "split": infer_split(record),
        "source": str(first(record, "source", "dataset", default="unknown")),
        "task": task,
        "state": "verification",
        "question": question,
        "target": "Yes" if target == 1 else "No" if target == 0 else None,
        "media": media_values,
        "frame": frame,
        "video": infer_video(record),
        "supervision": {
            "target_yes": target,
            "proposal_source": proposal_source,
            "question_type": question_type,
            "gt_used_for_prompt": False,
            "future_evidence": False,
        },
    }


def write_jsonl_and_manifest(rows: list[dict[str, Any]], out: Path, *, tool: str, source: Path) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "status": "YESNO_CORPUS_PREPARED_CPU_ONLY",
        "tool": tool,
        "rows": str(out.resolve()),
        "rows_sha256": digest,
        "row_count": len(rows),
        "source": str(source.resolve()),
        "model_training_started": False,
        "gt_in_prompt": False,
        "future_evidence": False,
        "native_schema": "locatemot-yesno-v1",
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
