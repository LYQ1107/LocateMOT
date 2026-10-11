#!/usr/bin/env python3
"""Queue the full BDD validation audit after a known evaluation worker frees its GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-result", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    while True:
        try:
            os.kill(args.wait_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(10)
    result = json.loads(args.wait_result.read_text())
    if result["state"] != "COMPLETE":
        raise RuntimeError("Previous GPU evaluation did not complete; inspect its failure before continuing")
    reference = json.loads((ROOT / "outputs/unified/u2/protocol/bdd_gt_reference_comparison.json").read_text())
    if reference["status"] != "PASS" or reference["validation_frames"] != 39973:
        raise RuntimeError("BDD raw GT has not passed the complete official-reference comparison")
    python = str(ROOT / ".venv/bin/python")
    evaluator = str(ROOT / "tools/unified/u2_mot_portability.py")
    protocol = str(ROOT / "outputs/unified/u2/protocol/bdd100k.json")
    common = [python, "-u", evaluator, "--protocol", protocol]
    print(json.dumps({"state": "BDD_SMOKE_RUNNING", "gpu": args.gpu}), flush=True)
    subprocess.run(common + ["--worker-index", "0", "--worker-count", "1", "--gpu", str(args.gpu),
                             "--smoke", "--out", str(ROOT / "outputs/unified/u2/perception/bdd_protocol_smoke")], check=True)
    print(json.dumps({"state": "BDD_FULL_VALIDATION_RUNNING", "gpu": args.gpu, "images": 39973}), flush=True)
    subprocess.run(common + ["--gpus", str(args.gpu),
                             "--out", str(ROOT / "outputs/unified/u2/perception/bdd_frozen_foundation")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
