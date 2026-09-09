#!/usr/bin/env python3
"""Build only missing pure language items for the isolated R0 scope.

The existing L89 cache is immutable.  This tool reads key/text metadata from
the already-isolated R0 safe artifacts, computes the missing frozen
GroundingDINO/BERT language sequences, and writes no candidate, label, image,
or detector-feature state.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.l89_language_cache import cache_key, sentence_sha256  # noqa: E402
from locatemot.rmot.r0_dense_data import EXPECTED_MANIFEST_SHA, MANIFEST, sha256_file  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
DEFAULT_SAFE_ROOT = WORK_ROOT / "outputs/r0/data/safe_targets_retry3"
DEFAULT_EXISTING = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/cache/language_tokens_retry1"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def safe_query_rows(root: Path) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(root.glob("v*/queries.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            key = (str(raw["dataset"]), str(raw["video"]), int(raw["query_id"]))
            value = {"dataset": key[0], "video": key[1], "query_id": key[2], "sentence": str(raw["sentence"])}
            if key in seen and seen[key]["sentence"] != value["sentence"]:
                raise AssertionError(f"safe query sentence drift: {key}")
            seen[key] = value
    return [seen[key] for key in sorted(seen)]


def existing_keys(root: Path) -> set[str]:
    path = root.resolve() / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.add(str(json.loads(line)["cache_key"]))
    return result


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 missing-language cache: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {
        "format": "locatemot-r0-missing-language-cache-v1",
        "status": "incomplete",
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd().resolve()),
        "luna_thread": THREAD,
        "safe_target_root": str(args.safe_root.resolve()),
        "existing_language_cache": str(args.existing.resolve()),
        "output": str(out),
        "manifest_sha256": sha256_file(MANIFEST),
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "labels_in_cache": False,
        "candidate_state_in_cache": False,
        "image_state_in_cache": False,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
        "failure_root_cause": None,
        "next_action": "audit merged R0 language coverage",
    }
    model = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 missing-language cwd: {Path.cwd()}")
        if base["manifest_sha256"] != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        rows = safe_query_rows(args.safe_root.resolve())
        present = existing_keys(args.existing.resolve())
        missing = [row for row in rows if cache_key(row["dataset"], row["video"], row["query_id"], row["sentence"]) not in present]
        if not missing:
            raise AssertionError("no missing language entries; refusing redundant rebuild")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable for missing-language cache")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        from locatemot.rmot.l88_grounding_runtime import build_groundingdino, make_text_batch
        model, model_info = build_groundingdino(device)
        model.eval()
        if not all(not parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError("language cache model is not frozen")
        entries: list[dict[str, Any]] = []
        batch_size = max(1, int(args.batch_size))
        for begin in range(0, len(missing), batch_size):
            batch_rows = missing[begin:begin + batch_size]
            with torch.inference_mode():
                text_dict, _captions, _token_maps = make_text_batch(
                    model, [row["sentence"] for row in batch_rows], device
                )
            embedded = text_dict.get("embedded")
            mask = text_dict.get("text_token_mask")
            if not torch.is_tensor(embedded) or not torch.is_tensor(mask):
                raise AssertionError("missing-language runtime returned no embedded/mask")
            if embedded.ndim != 3 or mask.ndim != 2 or embedded.shape[0] != len(batch_rows) or embedded.shape[:2] != mask.shape or embedded.shape[-1] != 256:
                raise AssertionError(f"missing-language shape drift: {tuple(embedded.shape)} {tuple(mask.shape)}")
            if not bool(torch.isfinite(embedded.float()).all()) or not bool(mask.bool().any(dim=1).all()):
                raise FloatingPointError("missing-language output nonfinite or empty")
            for offset, row in enumerate(batch_rows):
                key = cache_key(row["dataset"], row["video"], row["query_id"], row["sentence"])
                filename = f"{key}.pt"
                payload = {
                    "format": "locatemot-l89-language-item-v1",
                    "tokens": embedded[offset].detach().cpu().to(dtype=torch.float16).contiguous(),
                    "text_token_mask": mask[offset].detach().cpu().bool().contiguous(),
                    "labels_in_cache": False,
                    "candidate_gt_in_cache": False,
                    "candidate_deletion": False,
                    "candidate_truncation": False,
                }
                torch.save(payload, out / filename)
                entries.append({
                    "format": "locatemot-l89-language-index-v1",
                    "cache_key": key,
                    "dataset": row["dataset"],
                    "video": row["video"],
                    "query_id": row["query_id"],
                    "sentence_sha256": sentence_sha256(row["sentence"]),
                    "file": filename,
                    "shape": list(embedded[offset].shape),
                    "valid_tokens": int(mask[offset].sum()),
                })
            del text_dict, embedded, mask
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        (out / "manifest.jsonl").write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
        payload = {
            **base,
            "status": "complete",
            "existing_entry_count": len(present),
            "requested_query_count": len(rows),
            "missing_query_count": len(missing),
            "entry_count": len(entries),
            "model": model_info,
            "dtype_on_disk": "float16",
            "token_dim": 256,
            "labels_in_cache": False,
            "candidate_gt_in_cache": False,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "next_action": "audit merged R0 language coverage",
        }
        write_json(out / "summary.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 missing language cache — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback_path": str((out / "INCOMPLETE.md").resolve()), "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 2
    finally:
        if model is not None:
            del model
        gc.collect()
        if "device" in locals() and device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
