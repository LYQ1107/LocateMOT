#!/usr/bin/env python3
"""Build the compact, pure GroundingDINO/BERT language cache for L89.

Only key/text metadata is read from the L49 manifests.  The saved tensors are
the frozen language-model ``embedded`` sequence and its true token mask; no
labels, candidate rows, scores, or image state are persisted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L49_DATA = ROOT / "outputs/l49/data"
FIT_COUNT = 5314

if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def key_only_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read only the fields needed to identify an expression.

    The source JSON objects contain supervision, but those fields are never
    copied into this list or used by cache construction.
    """
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    for name in ("train_units.jsonl", "calibration_units.jsonl", "validation_units.jsonl"):
        source = L49_DATA / name
        count = 0
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                dataset = str(raw.get("dataset"))
                if dataset not in {"refer_kitti_v1", "refer_kitti_v2"}:
                    continue
                sentence = str(raw.get("sentence") or raw.get("expression") or "")
                if not sentence:
                    raise AssertionError(f"empty sentence in {name}")
                key = (dataset, str(raw["video"]), int(raw["query_id"]))
                value = {"unit_key": str(raw["unit_key"]), "dataset": dataset,
                         "video": str(raw["video"]), "query_id": int(raw["query_id"]),
                         "sentence": sentence}
                previous = rows.get(key)
                if previous is not None and previous["sentence"] != sentence:
                    raise AssertionError(f"sentence changed for {key}")
                rows[key] = value
                count += 1
        source_counts[name] = count
    result = sorted(rows.values(), key=lambda x: (x["dataset"], x["video"], x["query_id"]))
    if source_counts.get("train_units.jsonl") != FIT_COUNT:
        raise AssertionError(f"fit source count drift: {source_counts}")
    if not result:
        raise AssertionError("no L89 expressions")
    return result, source_counts


def _extract_text(text_dict: dict[str, Any], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    embedded = text_dict.get("embedded")
    mask = text_dict.get("text_token_mask")
    if not torch.is_tensor(embedded) or not torch.is_tensor(mask):
        raise AssertionError(f"GroundingDINO language output lacks embedded/text_token_mask: {sorted(text_dict)}")
    if embedded.ndim != 3 or mask.ndim != 2 or embedded.shape[0] != batch_size or embedded.shape[:2] != mask.shape:
        raise AssertionError(f"language output shape drift: {tuple(embedded.shape)} / {tuple(mask.shape)}")
    if embedded.shape[-1] != 256:
        raise AssertionError(f"L89 language embedding dimension drift: {tuple(embedded.shape)}")
    mask = mask.bool()
    if not bool(mask.any(dim=1).all()) or not bool(torch.isfinite(embedded.float()).all()):
        raise FloatingPointError("nonfinite/empty language output")
    return embedded.detach().cpu(), mask.detach().cpu()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_ROOT / "outputs/l89/cache/language_tokens")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89 language cache: {out}")
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong L89 worktree cwd: {Path.cwd()}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    rows, source_counts = key_only_rows()
    manifest_sha = sha256_file(MANIFEST)
    if manifest_sha != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for registered language cache")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    import locatemot.rmot as _rmot_package
    runtime_path = str(WORK_ROOT / "locatemot" / "rmot")
    if runtime_path not in [str(value) for value in _rmot_package.__path__]:
        _rmot_package.__path__.append(runtime_path)
    from locatemot.rmot.l88_grounding_runtime import build_groundingdino, make_text_batch

    model = None
    entries: list[dict[str, Any]] = []
    try:
        model, model_info = build_groundingdino(device)
        model.eval()
        if not all(not p.requires_grad for p in model.parameters()):
            raise AssertionError("GroundingDINO language cache model is not frozen")
        batch_size = max(1, int(args.batch_size))
        for begin in range(0, len(rows), batch_size):
            batch_rows = rows[begin:begin + batch_size]
            sentences = [str(row["sentence"]) for row in batch_rows]
            with torch.inference_mode():
                text_dict, _captions, _token_maps = make_text_batch(model, sentences, device)
            embedded, mask = _extract_text(text_dict, len(batch_rows))
            for offset, row in enumerate(batch_rows):
                key = __import__("locatemot.rmot.l89_language_cache", fromlist=["cache_key"]).cache_key(
                    row["dataset"], row["video"], row["query_id"], row["sentence"])
                filename = f"{key}.pt"
                payload = {
                    "format": "locatemot-l89-language-item-v1",
                    "tokens": embedded[offset].to(dtype=torch.float16).contiguous(),
                    "text_token_mask": mask[offset].bool().contiguous(),
                    "labels_in_cache": False, "candidate_gt_in_cache": False,
                    "candidate_deletion": False, "candidate_truncation": False,
                }
                torch.save(payload, out / filename)
                entries.append({
                    "format": "locatemot-l89-language-index-v1", "cache_key": key,
                    "dataset": row["dataset"], "video": row["video"], "query_id": row["query_id"],
                    "sentence_sha256": __import__("locatemot.rmot.l89_language_cache", fromlist=["sentence_sha256"]).sentence_sha256(row["sentence"]),
                    "file": filename, "shape": list(embedded[offset].shape),
                    "valid_tokens": int(mask[offset].sum()),
                })
            del text_dict, embedded, mask
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if len(entries) != len(rows):
            raise AssertionError("language cache entry count drift")
        (out / "manifest.jsonl").write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8"
        )
        summary = {
            "format": "locatemot-l89-language-cache-v1", "status": "complete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "entry_count": len(entries), "source_counts": source_counts,
            "source_files": [{"path": str(L49_DATA / name), "sha256": sha256_file(L49_DATA / name),
                              "bytes": (L49_DATA / name).stat().st_size} for name in ("train_units.jsonl", "calibration_units.jsonl", "validation_units.jsonl")],
            "manifest_sha256": manifest_sha, "model": model_info,
            "dtype_on_disk": "float16", "token_dim": 256,
            "labels_in_cache": False, "candidate_gt_in_cache": False,
            "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "failure_root_cause": None, "next_action": "run L89 contract smoke",
        }
        write_json(out / "summary.json", summary)
        write_json(out / "provenance.json", summary | {"format": "locatemot-l89-language-provenance-v1"})
        write_json(out / "status.json", {
            "format": "locatemot-l89-language-cache-v1", "status": "complete",
            "entry_count": len(entries), "summary": str((out / "summary.json").resolve()),
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "failure_root_cause": None, "next_action": "run L89 contract smoke",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        return 0
    except Exception:
        trace = __import__("traceback").format_exc()
        (out / "INCOMPLETE.md").write_text("# L89 language cache — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {
            "format": "locatemot-l89-language-cache-v1", "status": "incomplete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "failure_root_cause": "first traceback in INCOMPLETE.md",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
