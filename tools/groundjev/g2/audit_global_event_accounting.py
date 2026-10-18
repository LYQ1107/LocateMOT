#!/usr/bin/env python3
"""Read-only audit for historical and v2 G2 training event counters.

The audit never opens the JSONL in write mode.  Events emitted before the
global-accounting schema divided token/sample counters by ``world_size``;
their corrected values are reported alongside the logged values so the old
receipt remains an immutable record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EVENT_SCHEMA_V2 = "g2-training-event-v2-global-accounting"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def audit_events(
    raw: bytes,
    *,
    world_size: int,
    effective_global_batch: int | None,
) -> dict[str, Any]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    events: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event is not an object")
        except (ValueError, json.JSONDecodeError) as error:
            malformed.append({"line": line_number, "error": str(error)})
            continue
        logged_samples = int(event.get("samples_global", 0))
        logged_tokens = int(event.get("effective_tokens_global", 0))
        schema = str(event.get("event_schema_version") or "g2-training-event-v1-local-mean-ambiguous")
        if schema == EVENT_SCHEMA_V2:
            corrected_samples = logged_samples
            corrected_tokens = logged_tokens
            basis = "v2_logged_global_sum"
        else:
            corrected_samples = logged_samples * world_size
            corrected_tokens = logged_tokens * world_size
            basis = "historical_logged_value_times_world_size"
        expected_batch_mismatch = None
        if effective_global_batch is not None:
            expected_batch_mismatch = corrected_samples != effective_global_batch
        events.append(
            {
                "line": line_number,
                "step": event.get("step"),
                "event_schema_version": schema,
                "logged_samples_global": logged_samples,
                "corrected_samples_global": corrected_samples,
                "logged_effective_tokens_global": logged_tokens,
                "corrected_effective_tokens_global": corrected_tokens,
                "loss": event.get("loss"),
                "effective_global_batch": event.get("effective_global_batch", effective_global_batch),
                "expected_global_batch_mismatch": expected_batch_mismatch,
                "correction_basis": basis,
            }
        )
    return {
        "audit_schema_version": "g2-event-accounting-audit-v1",
        "event_schema_v2": EVENT_SCHEMA_V2,
        "world_size": world_size,
        "effective_global_batch": effective_global_batch,
        "event_count": len(events),
        "malformed_count": len(malformed),
        "malformed": malformed,
        "events": events,
        "raw_log_sha256": sha256_bytes(raw),
        "raw_log_unchanged": True,
        "basis": {
            "v2": "loss is the cross-rank mean; token/sample counters are cross-rank sums",
            "historical": "pre-v2 events divided all three fields by world_size; samples/tokens are reconstructed for audit only",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--effective-global-batch", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    report = audit_events(
        raw,
        world_size=args.world_size,
        effective_global_batch=args.effective_global_batch,
    )
    report["log"] = str(args.log)
    report["out"] = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("event_count", "malformed_count", "raw_log_sha256", "raw_log_unchanged")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
