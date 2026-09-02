#!/usr/bin/env python3
"""Create required compact L82 artifact aliases without changing evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = (args.source if args.source.is_absolute() else ROOT / args.source).resolve()
    destination = (args.destination if args.destination.is_absolute() else ROOT / args.destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    payload = json.loads(source.read_text())
    payload.setdefault("format", "locatemot-l82-loss-pathology-v1")
    payload["artifact_alias"] = {
        "source": str(source),
        "source_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        "copy_is_read_only_evidence": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "complete", "source": str(source), "destination": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
