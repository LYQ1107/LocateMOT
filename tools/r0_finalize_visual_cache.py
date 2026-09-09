#!/usr/bin/env python3
"""Finalize disjoint R0 visual-cache rank manifests without copying tensors."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r0_dense_data import load_l69_bank, native_frame_ids, sha256_file  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/outputs/l19/protocol/kitti_fast_eval_manifest.json")
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> int:
    cache_root = args.cache_root.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty finalized cache output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {"format": "locatemot-r0-visual-cache-finalizer-v1", "status": "incomplete", "command": " ".join([sys.executable, *sys.argv]), "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD, "cache_root": str(cache_root), "outputs": {"root": str(out)}, "manifest_sha256": sha256_file(MANIFEST), "expected_manifest_sha256": MANIFEST_SHA, "labels_in_cache": False, "query_independent": True, "finite": True, "candidate_deletion": False, "candidate_truncation": False, "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "failure_root_cause": None, "next_action": "audit finalized cache before R0 training"}
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 finalizer cwd: {Path.cwd()}")
        if base["manifest_sha256"] != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        rank_dirs = sorted(path for path in cache_root.glob("rank*") if path.is_dir())
        if not rank_dirs:
            raise FileNotFoundError(f"no rank directories under {cache_root}")
        rank_manifests: list[dict[str, Any]] = []
        all_items: list[dict[str, Any]] = []
        expected_pairs: set[tuple[str, str]] = set()
        declared_rank_ids: set[int] = set()
        for rank_dir in rank_dirs:
            manifest_path = rank_dir / "manifest.jsonl"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            rank_items = read_lines(manifest_path)
            rank_summary = json.loads((rank_dir / "summary.json").read_text(encoding="utf-8")) if (rank_dir / "summary.json").is_file() else {}
            rank_id = int(rank_dir.name.replace("rank", ""))
            if rank_id in declared_rank_ids:
                raise AssertionError(f"duplicate rank directory: {rank_id}")
            declared_rank_ids.add(rank_id)
            declared_pairs = rank_summary.get("pairs", [])
            if not isinstance(declared_pairs, list):
                raise AssertionError(f"rank summary lacks declared pairs: {rank_dir}")
            for pair in declared_pairs:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise AssertionError(f"invalid declared cache pair: {pair!r}")
                expected_pairs.add((str(pair[0]), str(pair[1])))
            rank_manifests.append({"rank": rank_id, "summary": rank_summary, "manifest_path": str(manifest_path), "item_count": len(rank_items)})
            for item in rank_items:
                cache_path = Path(item["path"])
                if not cache_path.is_file():
                    raise FileNotFoundError(cache_path)
                loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
                if not isinstance(loaded, dict) or loaded.get("dataset") is None or loaded.get("group_key") is None:
                    raise AssertionError(f"missing cache dataset/group_key: {cache_path}")
                relative = cache_path.relative_to(rank_dir)
                if len(relative.parts) < 3:
                    raise AssertionError(f"cache path lacks dataset/video/frame contract: {cache_path}")
                path_dataset, path_video = str(relative.parts[0]), str(relative.parts[1])
                path_frame = int(Path(relative.parts[2]).stem)
                expected_dataset = str(item.get("dataset", path_dataset))
                expected_video = str(item.get("video", path_video))
                expected_frame = int(item.get("frame_id", path_frame))
                if (loaded.get("dataset"), loaded.get("video"), int(loaded.get("frame_id", -1))) != (expected_dataset, expected_video, expected_frame):
                    raise AssertionError(f"cache manifest/item identity drift: {cache_path}")
                if (str(loaded.get("dataset")), str(loaded.get("video")), int(loaded.get("frame_id", -1))) != (path_dataset, path_video, path_frame):
                    raise AssertionError(f"cache path/item identity drift: {cache_path}")
                if loaded.get("group_key") != f"{loaded['dataset']}|{loaded['video']}|{loaded['frame_id']}":
                    raise AssertionError(f"invalid cache group_key: {cache_path}")
                for field in ("inner_tokens", "context_tokens", "boxes_normalized"):
                    tensor = loaded.get(field)
                    if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor.float()).all()):
                        raise FloatingPointError(f"invalid cached tensor {field}: {cache_path}")
                if loaded.get("labels_in_cache") is not False or loaded.get("query_independent") is not True:
                    raise AssertionError(f"cache flags drift: {cache_path}")
                all_items.append({"dataset": str(loaded["dataset"]), "video": str(loaded["video"]), "frame_id": int(loaded["frame_id"]), "path": str(cache_path), "candidate_count": int(item["candidate_count"]), "rank": int(rank_dir.name.replace("rank", ""))})
                del loaded
        keys = [(item["dataset"], item["video"], item["frame_id"]) for item in all_items]
        if len(keys) != len(set(keys)):
            raise AssertionError("duplicate finalized cache frame key")
        observed_pairs = {(item["dataset"], item["video"]) for item in all_items}
        if observed_pairs != expected_pairs:
            raise AssertionError(f"cache pair coverage drift: declared={sorted(expected_pairs)} observed={sorted(observed_pairs)}")
        pairs = sorted(expected_pairs)
        expected: list[tuple[str, str, int]] = []
        for dataset, video in pairs:
            expected.extend((dataset, video, frame) for frame in native_frame_ids(video))
        if sorted(keys) != sorted(expected):
            raise AssertionError(f"cache frame coverage drift: expected={len(expected)} actual={len(keys)}")
        if any(int(item["candidate_count"]) < 0 for item in all_items):
            raise AssertionError("negative candidate count")
        out_items = sorted(all_items, key=lambda item: (item["dataset"], item["video"], item["frame_id"]))
        (out / "manifest.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in out_items), encoding="utf-8")
        payload = {**base, "status": "complete", "rank_count": len(rank_dirs), "rank_manifests": rank_manifests, "pair_count": len(pairs), "pairs": [list(pair) for pair in pairs], "frame_count": len(all_items), "candidate_rows": sum(item["candidate_count"] for item in all_items), "all_expected_frames_exactly_once": True, "wall_seconds": time.perf_counter() - started}
        write_json(out / "summary.json", payload); write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 visual cache finalizer — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback_path": str((out / "INCOMPLETE.md").resolve()), "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
