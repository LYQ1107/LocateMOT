#!/usr/bin/env python3
"""Audit the R0A source contract and run one selective-reader CPU fixture.

This tool intentionally does not load any real annotation payload.  It reads
the audited L49 source as code text and exercises the allowlist reader only on
an ephemeral synthetic JSON object whose skipped value is never decoded.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r0_safe_target_source import (  # noqa: E402
    L49_SOURCE,
    V1_EXPR_ROOT,
    V2_NEW_PATH,
    V2_OLD_PATH,
    load_allowed_top_level_members,
    sha256_file,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
EXPECTED_SOURCE_SHA = "a1aae91cfbd8aadfba2e432302b06ff7b3e950fe486b48e1f5a4e80481819b3b"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty audit directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_text = L49_SOURCE.read_text(encoding="utf-8")
    source_sha = sha256_file(L49_SOURCE)
    source_contract = {
        "path": str(L49_SOURCE),
        "sha256": source_sha,
        "expected_sha256": EXPECTED_SOURCE_SHA,
        "signature": "load_l49_queries(dataset: str) -> list[dict[str, Any]]",
        "v1_source": str(V1_EXPR_ROOT),
        "v2_old_source": str(V2_OLD_PATH),
        "v2_new_source": str(V2_NEW_PATH),
        "precedence": "iterate V2_META_OLD then V2_META_NEW; merged key=(video, expression); later source overwrites",
        "returned_schema": ["dataset", "video", "expression", "sentence", "target", "label_source", "query_id", "split"],
        "query_order": "sort by (video, expression, sentence), then enumerate query_id globally",
        "source_line_snippets": {
            "paths": "\n".join(source_text.splitlines()[20:25]),
            "v2_merge": "\n".join(source_text.splitlines()[97:107]),
            "sort_and_id": "\n".join(source_text.splitlines()[107:115]),
        },
    }
    fixture = {
        "trainA": {"secret": [1, 2, 3]},
        "forbidden": {"must_not_decode": [4, 5, 6]},
        "trainB": [{"x": "brace } in string"}, {"x": "quote \\\""}],
    }
    with tempfile.TemporaryDirectory(prefix="locatemot_r0a_safe_reader_") as temp:
        fixture_path = Path(temp) / "fixture.json"
        fixture_path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
        decoded = load_allowed_top_level_members(fixture_path, {"trainA", "trainB"})
    decoded_keys = sorted(decoded)
    skipped = ["forbidden"]
    fixture_passed = decoded_keys == ["trainA", "trainB"] and "forbidden" not in decoded
    payload = {
        "format": "locatemot-r0a-safe-source-code-audit-v1",
        "status": "complete" if fixture_passed and source_sha == EXPECTED_SOURCE_SHA else "invalid",
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd().resolve()),
        "luna_thread": THREAD,
        "source_contract": source_contract,
        "synthetic_selective_reader": {
            "allowed_keys": ["trainA", "trainB"],
            "decoded_keys": decoded_keys,
            "skipped_keys": skipped,
            "forbidden_key_in_result": "forbidden" in decoded,
            "forbidden_payload_deserialized": False,
            "passed": fixture_passed,
        },
        "raw_annotation_payloads_deserialized": False,
        "forbidden_payload_deserialized": False,
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "ordinary_mot_ovmot_touched": False,
        "failure_root_cause": None if fixture_passed and source_sha == EXPECTED_SOURCE_SHA else "source hash or synthetic reader contract mismatch",
        "next_action": "build allowlisted safe target artifacts" if fixture_passed and source_sha == EXPECTED_SOURCE_SHA else "stop and repair safe source contract",
        "elapsed_seconds": time.perf_counter() - started,
    }
    for name in ("source_code_audit.json", "provenance.json", "status.json"):
        write_json(out / name, payload)
    if payload["status"] != "complete":
        (out / "INCOMPLETE.md").write_text("# R0A safe source audit — INCOMPLETE\n\n" + json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if payload["status"] == "complete" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
