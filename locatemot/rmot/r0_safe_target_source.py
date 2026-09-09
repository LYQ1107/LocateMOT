"""Source-isolated, frame-specific target records for the R0A/R0 sidecar.

Only this module may touch the raw L49/L11/L13/L16 annotation sources.  The
V2 L11 source is a monolithic JSON object, so this module scans its bytes to
locate top-level values and calls ``json.loads`` only for explicitly
allowlisted video values.  Skipped values are never deserialized or exposed.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
V1_EXPR_ROOT = ASSET_ROOT / "outputs/l13/data/refer_kitti_v1/expression"
V2_OLD_PATH = ASSET_ROOT / "outputs/l11/data/rmot_kitti/expressions.json"
V2_NEW_PATH = ASSET_ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"
L49_SOURCE = ASSET_ROOT / "locatemot/rmot/l49_data.py"
FORBIDDEN_SCOPE_VIDEOS = {"0005", "0011", "0013", "0019"}

# These sets are copied from the audited local l49_data.py L49_SPLITS source
# text.  They exclude official_eval and are used only to reproduce its query
# ordering; they do not authorize reading any target payload.
V1_SOURCE_VIDEOS = (
    "0001", "0002", "0003", "0004", "0006", "0007", "0008", "0009",
    "0010", "0012", "0014", "0015", "0016", "0018", "0020",
)
V2_SOURCE_VIDEOS = (
    "0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009",
    "0010", "0012", "0014", "0015", "0016", "0017", "0020",
)


class R0SafeSourceError(RuntimeError):
    """Raised for malformed source data or an unsafe source access."""


@dataclass(frozen=True)
class R0SafeQueryRecord:
    dataset: str
    video: str
    query_id: int
    sentence: str
    target: dict[int, tuple[str, ...]]
    label_source: str


@dataclass(frozen=True)
class R0SafeLoadResult:
    records: tuple[R0SafeQueryRecord, ...]
    manifest: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _skip_ws(data: bytes, index: int) -> int:
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    return index


def _scan_string_end(data: bytes, index: int) -> int:
    if index >= len(data) or data[index] != ord('"'):
        raise R0SafeSourceError(f"expected JSON string at byte {index}")
    cursor = index + 1
    escaped = False
    while cursor < len(data):
        value = data[cursor]
        if escaped:
            # Lexical validation only.  JSON decoding of the key itself later
            # validates unicode escapes and the complete string grammar.
            escaped = False
            cursor += 1
            continue
        if value == ord('\\'):
            escaped = True
            cursor += 1
            continue
        if value == ord('"'):
            return cursor + 1
        cursor += 1
    raise R0SafeSourceError("unterminated JSON string")


def _scan_value_end(data: bytes, index: int) -> int:
    index = _skip_ws(data, index)
    if index >= len(data):
        raise R0SafeSourceError("missing JSON value")
    first = data[index]
    if first == ord('"'):
        return _scan_string_end(data, index)
    if first in (ord('{'), ord('[')):
        stack = [first]
        cursor = index + 1
        in_string = False
        escaped = False
        while cursor < len(data):
            value = data[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif value == ord('\\'):
                    escaped = True
                elif value == ord('"'):
                    in_string = False
                cursor += 1
                continue
            if value == ord('"'):
                in_string = True
            elif value in (ord('{'), ord('[')):
                stack.append(value)
            elif value in (ord('}'), ord(']')):
                if not stack:
                    raise R0SafeSourceError("unexpected JSON closing delimiter")
                opening = stack.pop()
                expected = ord('}') if opening == ord('{') else ord(']')
                if value != expected:
                    raise R0SafeSourceError("mismatched JSON value delimiter")
                if not stack:
                    return cursor + 1
            cursor += 1
        raise R0SafeSourceError("unterminated JSON object/array")
    cursor = index
    while cursor < len(data) and data[cursor] not in (ord(','), ord('}')):
        cursor += 1
    end = _skip_ws(data, cursor)
    if end == index:
        raise R0SafeSourceError(f"empty scalar at byte {index}")
    return end


def _scan_top_level_member_ranges(path: Path) -> tuple[bytes, list[tuple[str, int, int]]]:
    data = path.read_bytes()
    index = _skip_ws(data, 0)
    if index >= len(data) or data[index] != ord('{'):
        raise R0SafeSourceError(f"top-level source is not an object: {path}")
    index = _skip_ws(data, index + 1)
    members: list[tuple[str, int, int]] = []
    if index < len(data) and data[index] == ord('}'):
        return data, members
    while True:
        index = _skip_ws(data, index)
        key_start = index
        key_end = _scan_string_end(data, key_start)
        try:
            key = json.loads(data[key_start:key_end].decode("utf-8"))
        except Exception as exc:  # pragma: no cover - JSON parser detail
            raise R0SafeSourceError(f"invalid top-level key in {path}") from exc
        if not isinstance(key, str):
            raise R0SafeSourceError(f"non-string top-level key in {path}")
        index = _skip_ws(data, key_end)
        if index >= len(data) or data[index] != ord(':'):
            raise R0SafeSourceError(f"missing colon after top-level key {key!r}")
        value_start = _skip_ws(data, index + 1)
        value_end = _scan_value_end(data, value_start)
        members.append((key, value_start, value_end))
        index = _skip_ws(data, value_end)
        if index >= len(data):
            raise R0SafeSourceError("truncated top-level JSON object")
        if data[index] == ord(','):
            index += 1
            continue
        if data[index] == ord('}'):
            index += 1
            if _skip_ws(data, index) != len(data):
                raise R0SafeSourceError("trailing bytes after top-level JSON object")
            return data, members
        raise R0SafeSourceError("expected comma or closing brace in top-level JSON object")


def load_allowed_top_level_members(path: Path, allowed_keys: set[str]) -> dict[str, Any]:
    """Deserialize only explicitly allowed top-level values.

    The complete file is byte-scanned, including skipped values, to locate
    boundaries.  A skipped value is never passed to ``json.loads`` and is not
    converted into a Python object.
    """
    data, members = _scan_top_level_member_ranges(path)
    result: dict[str, Any] = {}
    decoded_keys: list[str] = []
    for key, value_start, value_end in members:
        if key in allowed_keys:
            result[key] = json.loads(data[value_start:value_end].decode("utf-8"))
            decoded_keys.append(key)
        # Deliberately no operation on the skipped value bytes.
    if set(decoded_keys) != set(allowed_keys):
        missing = sorted(set(allowed_keys) - set(decoded_keys))
        raise R0SafeSourceError(f"allowlisted top-level keys missing from {path}: {missing}")
    return result


def _count_array_items(data: bytes, start: int, end: int) -> int:
    """Count array elements lexically without deserializing their payload."""
    index = _skip_ws(data, start)
    if index >= end or data[index] != ord('['):
        raise R0SafeSourceError("expected array value while counting source metadata")
    index += 1
    depth = 0
    in_string = False
    escaped = False
    comma_count = 0
    nonempty = False
    stack: list[int] = []
    while index < end - 1:
        value = data[index]
        if in_string:
            nonempty = True
            if escaped:
                escaped = False
            elif value == ord('\\'):
                escaped = True
            elif value == ord('"'):
                in_string = False
            index += 1
            continue
        if value == ord('"'):
            in_string = True
            nonempty = True
        elif value in (ord('{'), ord('[')):
            stack.append(value)
            depth += 1
            nonempty = True
        elif value in (ord('}'), ord(']')):
            if value == ord(']') and not stack:
                break
            if not stack:
                raise R0SafeSourceError("unexpected closing delimiter while counting array")
            opening = stack.pop()
            expected = ord('}') if opening == ord('{') else ord(']')
            if value != expected:
                raise R0SafeSourceError("mismatched delimiter while counting array")
            depth -= 1
        elif value == ord(',') and depth == 0:
            comma_count += 1
        elif value not in b" \t\r\n":
            nonempty = True
        index += 1
    if index >= end or data[index] != ord(']') or depth != 0 or stack or in_string:
        raise R0SafeSourceError("malformed array while counting source metadata")
    return comma_count + 1 if nonempty else 0


def _normalize_target_map(raw: Any) -> dict[int, tuple[str, ...]]:
    if not isinstance(raw, dict):
        raise R0SafeSourceError(f"target label is not a mapping: {type(raw).__name__}")
    result: dict[int, tuple[str, ...]] = {}
    for raw_frame, raw_ids in raw.items():
        try:
            frame = int(raw_frame)
        except (TypeError, ValueError) as exc:
            raise R0SafeSourceError(f"invalid target frame key {raw_frame!r}") from exc
        if raw_ids is None:
            values: tuple[str, ...] = ()
        elif isinstance(raw_ids, (list, tuple, set, frozenset)):
            values = tuple(sorted({str(value) for value in raw_ids}))
        else:
            values = (str(raw_ids),)
        previous = result.get(frame)
        if previous is not None and previous != values:
            raise R0SafeSourceError(f"conflicting normalized target frame key {frame}")
        result[frame] = values
    return result


def _record_from_item(dataset: str, video: str, item: Any, source: Path) -> tuple[str, str, dict[int, tuple[str, ...]], str]:
    if not isinstance(item, dict):
        raise R0SafeSourceError(f"query record is not an object: {source}")
    expression = str(item.get("expression", ""))
    sentence = str(item.get("sentence", expression))
    if not sentence:
        raise R0SafeSourceError(f"empty sentence in allowed source: {source}")
    target = _normalize_target_map(item.get("label", {}))
    return expression, sentence, target, str(source.resolve())


def _base_manifest(pairs: set[tuple[str, str]], purpose: str) -> dict[str, Any]:
    if purpose not in {"train", "dev", "internal", "audit"}:
        raise ValueError(purpose)
    if not pairs:
        raise R0SafeSourceError(f"empty source scope for {purpose}")
    if any(dataset not in {"refer_kitti_v1", "refer_kitti_v2"} for dataset, _ in pairs):
        raise R0SafeSourceError(f"invalid dataset in source scope: {pairs}")
    if {video for _, video in pairs} & FORBIDDEN_SCOPE_VIDEOS:
        raise R0SafeSourceError("forbidden official-eval video in source scope")
    return {
        "format": "locatemot-r0a-safe-source-read-v1",
        "status": "complete",
        "purpose": purpose,
        "allowed_video_pairs": [f"{dataset}|{video}" for dataset, video in sorted(pairs)],
        "source_paths": [],
        "source_sha256": {},
        "allowed_top_level_keys": [],
        "seen_top_level_key_count": 0,
        "decoded_top_level_keys": [],
        "skipped_top_level_keys_count": 0,
        "forbidden_top_level_keys_seen": [],
        "forbidden_payload_deserialized": False,
        "official_test_labels_read": False,
        "target_union_used": False,
        "pseudo_dense_labels_used": False,
        "nearest_frame_fallback": False,
        "forward_fill": False,
        "backward_fill": False,
    }


def _finish_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["source_paths"] = sorted(set(manifest["source_paths"]))
    manifest["decoded_top_level_keys"] = sorted(set(manifest["decoded_top_level_keys"]))
    manifest["allowed_top_level_keys"] = sorted(set(manifest["allowed_top_level_keys"]))
    manifest["forbidden_top_level_keys_seen"] = sorted(set(manifest["forbidden_top_level_keys_seen"]))
    manifest["skipped_top_level_keys_count"] = max(
        0, int(manifest["seen_top_level_key_count"]) - len(manifest["decoded_top_level_keys"])
    )
    manifest["source_sha256"] = {
        str(path): sha256_file(Path(path)) for path in manifest["source_paths"]
    }
    return manifest


def load_safe_v1_records(allowed_video_pairs: set[tuple[str, str]]) -> list[R0SafeQueryRecord]:
    return list(load_safe_v1_records_with_manifest(allowed_video_pairs, purpose="audit").records)


def load_safe_v1_records_with_manifest(
    allowed_video_pairs: set[tuple[str, str]], *, purpose: str = "audit"
) -> R0SafeLoadResult:
    pairs = {(str(dataset), str(video)) for dataset, video in allowed_video_pairs}
    if any(dataset != "refer_kitti_v1" for dataset, _ in pairs):
        raise R0SafeSourceError("V1 safe loader received a non-V1 pair")
    manifest = _base_manifest(pairs, purpose)
    manifest["source_code_contract"] = {
        "source": str(L49_SOURCE),
        "v1_source": str(V1_EXPR_ROOT),
        "query_order": "source rows sorted by (video, expression, sentence), query_id enumerated globally",
        "label_field": "item.label",
    }
    counts: dict[str, int] = {}
    for video in V1_SOURCE_VIDEOS:
        directory = V1_EXPR_ROOT / video
        files = sorted(directory.glob("*.json"))
        counts[video] = len(files)
    offsets: dict[str, int] = {}
    running = 0
    for video in V1_SOURCE_VIDEOS:
        offsets[video] = running
        running += counts[video]

    records: list[R0SafeQueryRecord] = []
    for _dataset, video in sorted(pairs):
        directory = V1_EXPR_ROOT / video
        files = sorted(directory.glob("*.json"))
        if not files:
            raise R0SafeSourceError(f"no V1 expression files for allowed video {video}")
        local: list[tuple[str, str, dict[int, tuple[str, ...]], str]] = []
        for path in files:
            item = json.loads(path.read_text(encoding="utf-8"))
            local.append(_record_from_item("refer_kitti_v1", video, item, path))
        local.sort(key=lambda value: (value[0], value[1], value[3]))
        for index, (expression, sentence, target, source) in enumerate(local):
            records.append(R0SafeQueryRecord("refer_kitti_v1", video, offsets[video] + index, sentence, target, source))
        manifest["source_paths"].extend(str(path.resolve()) for path in files)
    return R0SafeLoadResult(tuple(sorted(records, key=lambda r: (r.video, r.query_id))), _finish_manifest(manifest))


def _load_v2_source(path: Path, allowed_videos: set[str]) -> tuple[dict[str, list[Any]], dict[str, int], list[str], list[str], int]:
    data, ranges = _scan_top_level_member_ranges(path)
    all_keys = [key for key, _, _ in ranges]
    available = set(all_keys)
    selected = allowed_videos & available
    # This is the only operation that deserializes values.  It receives only
    # allowlisted top-level video keys.
    values = load_allowed_top_level_members(path, selected) if selected else {}
    counts: dict[str, int] = {}
    for key, start, end in ranges:
        if key in V2_SOURCE_VIDEOS:
            counts[key] = _count_array_items(data, start, end)
    forbidden_seen = sorted(available & FORBIDDEN_SCOPE_VIDEOS)
    skipped = [key for key in all_keys if key not in selected]
    return values, counts, all_keys, forbidden_seen, len(skipped)


def load_safe_v2_records(allowed_video_pairs: set[tuple[str, str]]) -> list[R0SafeQueryRecord]:
    return list(load_safe_v2_records_with_manifest(allowed_video_pairs, purpose="audit").records)


def load_safe_v2_records_with_manifest(
    allowed_video_pairs: set[tuple[str, str]], *, purpose: str = "audit"
) -> R0SafeLoadResult:
    pairs = {(str(dataset), str(video)) for dataset, video in allowed_video_pairs}
    if any(dataset != "refer_kitti_v2" for dataset, _ in pairs):
        raise R0SafeSourceError("V2 safe loader received a non-V2 pair")
    allowed_videos = {video for _, video in pairs}
    if not allowed_videos <= set(V2_SOURCE_VIDEOS):
        raise R0SafeSourceError(f"V2 pair outside audited L49 source scope: {sorted(allowed_videos)}")
    manifest = _base_manifest(pairs, purpose)
    manifest["source_code_contract"] = {
        "source": str(L49_SOURCE),
        "v2_old_source": str(V2_OLD_PATH),
        "v2_new_source": str(V2_NEW_PATH),
        "query_order": "source rows sorted by (video, expression, sentence), query_id enumerated globally",
        "merge_order": "old L11 then newer L16; merged key is (video, expression), later record overwrites",
        "label_field": "item.label",
    }
    old_values, old_counts, old_keys, old_forbidden, old_skipped = _load_v2_source(V2_OLD_PATH, allowed_videos)
    new_values, new_counts, new_keys, new_forbidden, new_skipped = _load_v2_source(V2_NEW_PATH, allowed_videos)
    if set(old_keys) & set(new_keys):
        raise R0SafeSourceError("unexpected overlapping V2 old/new top-level video keys; precedence cannot be proven")
    manifest["source_paths"] = [str(V2_OLD_PATH.resolve()), str(V2_NEW_PATH.resolve())]
    manifest["seen_top_level_key_count"] = len(old_keys) + len(new_keys)
    manifest["allowed_top_level_keys"] = sorted(allowed_videos)
    manifest["decoded_top_level_keys"] = sorted(set(old_values) | set(new_values))
    manifest["forbidden_top_level_keys_seen"] = sorted(set(old_forbidden) | set(new_forbidden))
    manifest["skipped_top_level_keys_count"] = old_skipped + new_skipped
    manifest["forbidden_payload_deserialized"] = False
    if set(manifest["forbidden_top_level_keys_seen"]) & set(manifest["decoded_top_level_keys"]):
        raise R0SafeSourceError("forbidden top-level key was selected for decoding")

    merged: dict[tuple[str, str], tuple[str, str, dict[int, tuple[str, ...]], str]] = {}
    for values, source in ((old_values, V2_OLD_PATH), (new_values, V2_NEW_PATH)):
        for video in sorted(values):
            items = values[video]
            if not isinstance(items, list):
                raise R0SafeSourceError(f"V2 allowed video value is not a list: {video}")
            for item in items:
                expression, sentence, target, label_source = _record_from_item("refer_kitti_v2", video, item, source)
                merged[(video, expression)] = (expression, sentence, target, label_source)

    counts_by_video: dict[str, int] = {}
    for video in V2_SOURCE_VIDEOS:
        if video in allowed_videos:
            counts_by_video[video] = sum(1 for key in merged if key[0] == video)
        elif video in old_counts:
            counts_by_video[video] = old_counts[video]
        elif video in new_counts:
            counts_by_video[video] = new_counts[video]
        else:
            counts_by_video[video] = 0
    offsets: dict[str, int] = {}
    running = 0
    for video in V2_SOURCE_VIDEOS:
        offsets[video] = running
        running += counts_by_video[video]

    records: list[R0SafeQueryRecord] = []
    for video in sorted(allowed_videos):
        local = [value for (candidate_video, _), value in merged.items() if candidate_video == video]
        local.sort(key=lambda value: (value[0], value[1], value[3]))
        for index, (_expression, sentence, target, source) in enumerate(local):
            records.append(R0SafeQueryRecord("refer_kitti_v2", video, offsets[video] + index, sentence, target, source))
    return R0SafeLoadResult(tuple(sorted(records, key=lambda r: (r.video, r.query_id))), _finish_manifest(manifest))


def load_safe_query_records_with_manifest(
    allowed_video_pairs: set[tuple[str, str]], *, purpose: str = "audit"
) -> R0SafeLoadResult:
    pairs = {(str(dataset), str(video)) for dataset, video in allowed_video_pairs}
    if not pairs:
        raise R0SafeSourceError("empty allowed query-record scope")
    records: list[R0SafeQueryRecord] = []
    manifests: list[dict[str, Any]] = []
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        subset = {pair for pair in pairs if pair[0] == dataset}
        if not subset:
            continue
        result = (
            load_safe_v1_records_with_manifest(subset, purpose=purpose)
            if dataset == "refer_kitti_v1"
            else load_safe_v2_records_with_manifest(subset, purpose=purpose)
        )
        records.extend(result.records)
        manifests.append(result.manifest)
    if len({(r.dataset, r.video, r.query_id) for r in records}) != len(records):
        raise R0SafeSourceError("duplicate safe query key")
    manifest = _base_manifest(pairs, purpose)
    manifest["source_paths"] = sorted({path for item in manifests for path in item["source_paths"]})
    manifest["source_sha256"] = {
        path: sha256_file(Path(path)) for path in manifest["source_paths"]
        if Path(path).is_file()
    }
    manifest["allowed_top_level_keys"] = sorted({key for item in manifests for key in item["allowed_top_level_keys"]})
    manifest["seen_top_level_key_count"] = sum(int(item["seen_top_level_key_count"]) for item in manifests)
    manifest["decoded_top_level_keys"] = sorted({key for item in manifests for key in item["decoded_top_level_keys"]})
    manifest["skipped_top_level_keys_count"] = sum(int(item["skipped_top_level_keys_count"]) for item in manifests)
    manifest["forbidden_top_level_keys_seen"] = sorted({key for item in manifests for key in item["forbidden_top_level_keys_seen"]})
    manifest["forbidden_payload_deserialized"] = any(item["forbidden_payload_deserialized"] for item in manifests)
    manifest["source_code_contract"] = {
        "l49_source": str(L49_SOURCE),
        "loader": "R0SafeQueryRecord only; raw sources are touched by this module",
        "component_manifests": manifests,
    }
    manifest["status"] = "complete" if not manifest["forbidden_payload_deserialized"] else "blocked"
    return R0SafeLoadResult(tuple(sorted(records, key=lambda r: (r.dataset, r.video, r.query_id))), _finish_manifest(manifest))


def load_safe_query_records(root: Path) -> list[R0SafeQueryRecord]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise R0SafeSourceError(f"safe target artifact is not complete: {root}")
    if manifest.get("forbidden_payload_deserialized") is not False:
        raise R0SafeSourceError("safe target artifact does not prove forbidden payload isolation")
    if manifest.get("official_test_labels_read") is not False:
        raise R0SafeSourceError("safe target artifact has official-test label flag")
    rows: list[R0SafeQueryRecord] = []
    for line in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        target_raw = item.get("target", {})
        target = _normalize_target_map(target_raw)
        rows.append(R0SafeQueryRecord(
            str(item["dataset"]), str(item["video"]), int(item["query_id"]),
            str(item["sentence"]), target, str(item["label_source"]),
        ))
    if len({(r.dataset, r.video, r.query_id) for r in rows}) != len(rows):
        raise R0SafeSourceError(f"duplicate query key in safe artifact: {root}")
    return rows


def write_safe_target_artifact(
    records: Iterable[R0SafeQueryRecord],
    out: Path,
    *,
    scope_name: str,
    allowed_video_pairs: set[tuple[str, str]],
    source_manifest: dict[str, Any],
) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty safe target artifact: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda r: (r.dataset, r.video, r.query_id))
    if any((r.dataset, r.video) not in allowed_video_pairs for r in rows):
        raise R0SafeSourceError("safe artifact contains a query outside its allow-list")
    manifest = {
        "format": "locatemot-r0a-safe-target-artifact-v1",
        "status": "complete",
        "scope_name": scope_name,
        "allowed_video_pairs": [f"{dataset}|{video}" for dataset, video in sorted(allowed_video_pairs)],
        "query_count": len(rows),
        "source_manifest": source_manifest,
        "source_paths": source_manifest.get("source_paths", []),
        "source_sha256": source_manifest.get("source_sha256", {}),
        "allowed_top_level_keys": source_manifest.get("allowed_top_level_keys", []),
        "seen_top_level_key_count": source_manifest.get("seen_top_level_key_count", 0),
        "decoded_top_level_keys": source_manifest.get("decoded_top_level_keys", []),
        "skipped_top_level_keys_count": source_manifest.get("skipped_top_level_keys_count", 0),
        "forbidden_top_level_keys_seen": source_manifest.get("forbidden_top_level_keys_seen", []),
        "forbidden_payload_deserialized": False,
        "official_test_labels_read": False,
        "target_union_used": False,
        "pseudo_dense_labels_used": False,
        "nearest_frame_fallback": False,
        "forward_fill": False,
        "backward_fill": False,
        "labels_in_visual_cache": False,
    }
    with (out / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps({
                "dataset": record.dataset,
                "video": record.video,
                "query_id": record.query_id,
                "sentence": record.sentence,
                "target": {str(frame): list(ids) for frame, ids in sorted(record.target.items())},
                "label_source": record.label_source,
            }, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "ASSET_ROOT", "FORBIDDEN_SCOPE_VIDEOS", "L49_SOURCE", "R0SafeLoadResult",
    "R0SafeQueryRecord", "R0SafeSourceError", "V1_EXPR_ROOT", "V1_SOURCE_VIDEOS",
    "V2_NEW_PATH", "V2_OLD_PATH", "V2_SOURCE_VIDEOS", "_normalize_target_map",
    "load_allowed_top_level_members", "load_safe_query_records",
    "load_safe_query_records_with_manifest", "load_safe_v1_records",
    "load_safe_v1_records_with_manifest", "load_safe_v2_records",
    "load_safe_v2_records_with_manifest", "sha256_file", "write_safe_target_artifact",
]
