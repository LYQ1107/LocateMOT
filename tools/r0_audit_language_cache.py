#!/usr/bin/env python3
"""Audit coverage of the frozen, pure L89 language-token cache for R0."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r0_dense_data import EXPECTED_MANIFEST_SHA, MANIFEST, sha256_file  # noqa: E402
from locatemot.rmot.l89_language_cache import L89LanguageTokenCache  # noqa: E402


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SAFE_ROOT = WORK_ROOT / "outputs/r0/data/safe_targets_retry3"
DEFAULT_CACHE = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/cache/language_tokens_retry1"
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def query_rows(root: Path) -> list[tuple[str, str, int, str]]:
    seen: dict[tuple[str, str, int], tuple[str, str, int, str]] = {}
    for path in sorted(root.glob("v*/queries.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["dataset"]), str(row["video"]), int(row["query_id"]))
            value = (*key, str(row["sentence"]))
            prior = seen.get(key)
            if prior is not None and prior[3] != value[3]:
                raise AssertionError(f"sentence drift across safe scopes: {key}")
            seen[key] = value
    return [seen[key] for key in sorted(seen)]


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty language audit: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {
        "format": "locatemot-r0-language-cache-audit-v1",
        "status": "incomplete",
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd().resolve()),
        "luna_thread": THREAD,
        "language_cache": str(args.cache.resolve()),
        "safe_target_root": str(args.safe_root.resolve()),
        "manifest_sha256": sha256_file(MANIFEST),
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "labels_used_for_training": False,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
        "failure_root_cause": None,
        "next_action": "build query-independent R0 visual cache",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 language audit cwd: {Path.cwd()}")
        if base["manifest_sha256"] != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        rows = query_rows(args.safe_root.resolve())
        cache = L89LanguageTokenCache(args.cache.resolve())
        supplement = L89LanguageTokenCache(args.supplement.resolve()) if args.supplement else None
        missing: list[list[Any]] = []
        finite_bad: list[list[Any]] = []
        shape_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for dataset, video, query_id, sentence in rows:
            try:
                key = __import__("locatemot.rmot.l89_language_cache", fromlist=["cache_key"]).cache_key(
                    dataset, video, query_id, sentence
                )
                if key in cache._index:
                    item = cache.get(dataset, video, query_id, sentence)
                    source = "existing"
                elif supplement is not None and key in supplement._index:
                    item = supplement.get(dataset, video, query_id, sentence)
                    source = "supplement"
                else:
                    raise KeyError(f"merged language cache miss: {dataset}|{video}|{query_id}")
                shape = f"{tuple(item.tokens.shape)}|{tuple(item.mask.shape)}"
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
                domain_counts[dataset] = domain_counts.get(dataset, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
                if not bool(item.mask.any()) or not bool(item.tokens.isfinite().all()):
                    finite_bad.append([dataset, video, query_id])
            except Exception as exc:
                missing.append([dataset, video, query_id, f"{type(exc).__name__}: {exc}"])
        if missing or finite_bad:
            raise AssertionError(f"language cache contract failed: missing={missing[:3]} finite_bad={finite_bad[:3]}")
        result = {
            **base,
            "status": "complete",
            "cache_entry_count": cache.entry_count,
            "supplement_entry_count": supplement.entry_count if supplement is not None else 0,
            "unique_query_count": len(rows),
            "covered_query_count": len(rows),
            "missing_query_count": len(missing),
            "finite_bad_count": len(finite_bad),
            "shape_counts": shape_counts,
            "domain_counts": domain_counts,
            "source_counts": source_counts,
            "tokens_only": True,
            "forbidden_label_fields_read_from_cache": False,
            "cache_has_no_candidate_or_target_state": True,
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(out / "audit.json", result)
        write_json(out / "provenance.json", result)
        write_json(out / "status.json", result)
        return 0
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text("# R0 language cache audit — INCOMPLETE\n\n" + traceback.format_exc(), encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback_path": str((out / "INCOMPLETE.md").resolve()), "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-root", type=Path, default=SAFE_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--supplement", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
