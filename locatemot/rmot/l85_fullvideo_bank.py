"""L85 query-independent full-video bank contract helpers.

The L69 bank is already materialized for every legal train-pool and internal
validation video.  This module never rebuilds it and never loads expression
or GT data while constructing a row index.  It is intentionally independent
of the L49 legacy ``begin``/``end`` ranges.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
L69_FEATURE_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L69_DUAL_ROOT = ROOT / "outputs/l69/attempt9/budget40_dual_bank/kitti"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
ALL_VIDEOS = (
    "0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008",
    "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018", "0020",
)
INTERNAL_V1 = ("0004", "0018")
INTERNAL_V2 = ("0016", "0017", "0020")
OBS_FIELDS = ("clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle", "objectness")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(path), "resolved_path": str(resolved), "exists": resolved.is_file(),
        "bytes": resolved.stat().st_size if resolved.is_file() else None,
        "mtime_ns": resolved.stat().st_mtime_ns if resolved.exists() else None,
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
        "is_symlink": path.is_symlink(), "symlink_target": str(path.readlink()) if path.is_symlink() else None,
    }


def bank_path(video: str, kind: str = "features") -> Path:
    root = L69_FEATURE_ROOT if kind == "features" else L69_DUAL_ROOT
    return root / f"{str(video)}.pt"


def label_path(video: str, kind: str = "features") -> Path:
    return bank_path(video, kind).with_suffix(".labels.json")


def audit_bank(video: str) -> dict[str, Any]:
    """Load one bank, validate its native frame contract, then release it."""
    path = bank_path(video)
    package = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(package, dict) or not isinstance(package.get("tensors"), dict):
        raise AssertionError(f"invalid L69 package: {path}")
    tensors = package["tensors"]
    required = set(OBS_FIELDS) | {
        "frame", "frame_ids", "frame_ptr", "candidate_index", "track_id", "pool_id",
        "box", "raw_rank", "source_score", "uidm_h", "pbd", "uidm_ref_pbd", "uidm_anchor_pbd",
    }
    missing = sorted(required - set(tensors))
    if missing:
        raise AssertionError(f"{video} missing fields: {missing}")
    frame_ids = tensors["frame_ids"].long()
    frame_ptr = tensors["frame_ptr"].long()
    total = int(tensors["track_id"].numel())
    if frame_ptr.numel() != frame_ids.numel() + 1 or int(frame_ptr[-1]) != total:
        raise AssertionError(f"{video} frame_ptr total mismatch")
    if not bool(torch.all(frame_ptr[1:] >= frame_ptr[:-1])):
        raise AssertionError(f"{video} frame_ptr is not monotone")
    frame_rows = tensors["frame"].long()
    if total and not bool(torch.isfinite(tensors["box"].float()).all()):
        raise AssertionError(f"{video} nonfinite boxes")
    for name in OBS_FIELDS + ("box", "raw_rank", "source_score"):
        if not bool(torch.isfinite(tensors[name].float()).all()):
            raise AssertionError(f"{video} nonfinite {name}")
    frame_stats = []
    unique_tracks: set[int] = set()
    duplicate_candidate_rows = 0
    for index, frame in enumerate(frame_ids.tolist()):
        start, end = int(frame_ptr[index]), int(frame_ptr[index + 1])
        values = frame_rows[start:end]
        if len(values) and not bool(torch.all(values == int(frame))):
            raise AssertionError(f"{video} frame row mismatch at {frame}")
        candidates = [int(x) for x in tensors["candidate_index"][start:end].tolist()]
        duplicate_candidate_rows += len(candidates) - len(set(candidates))
        unique_tracks.update(int(x) for x in tensors["track_id"][start:end].tolist())
        frame_stats.append({"frame_id": int(frame), "begin": start, "end": end, "rows": end - start,
                            "pools": sorted(set(int(x) for x in tensors["pool_id"][start:end].tolist())),
                            "duplicate_candidate_index_rows": len(candidates) - len(set(candidates))})
    metadata = package.get("metadata", {})
    result = {
        "video": str(video), "feature_file": file_meta(path),
        "dual_file": file_meta(bank_path(video, "dual")), "feature_label_file": file_meta(label_path(video)),
        "frame_count": int(frame_ids.numel()), "row_count": total,
        "track_count": len(unique_tracks), "duplicate_candidate_index_rows": duplicate_candidate_rows,
        "frame_stats": frame_stats, "image_size": metadata.get("image_size"),
        "metadata_flags": {key: metadata.get(key) for key in (
            "causal", "query_independent", "rmot_only_reserve_namespace", "reserve_budget",
            "preserve_source_ids", "reserve_id_offset", "reserve_tracker_max_gap")},
        "all_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
    }
    del package
    return result


def internal_videos() -> tuple[str, ...]:
    return INTERNAL_V1 + INTERNAL_V2


def bank_source_manifest(videos: Iterable[str] = ALL_VIDEOS) -> list[dict[str, Any]]:
    return [{"video": str(video), "features": file_meta(bank_path(str(video))),
             "dual": file_meta(bank_path(str(video), "dual")),
             "labels": file_meta(label_path(str(video)))} for video in videos]


__all__ = [
    "ALL_VIDEOS", "EXPECTED_MANIFEST_SHA", "INTERNAL_V1", "INTERNAL_V2", "L69_DUAL_ROOT",
    "L69_FEATURE_ROOT", "MANIFEST", "OBS_FIELDS", "audit_bank", "bank_path", "bank_source_manifest",
    "file_meta", "internal_videos", "label_path", "sha256_file",
]
