#!/usr/bin/env python3
"""Emit the fixed upstream reference and local contract audit."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REFERENCES = {
    "GroundAnything": {"commit": "e054ab329cfe3f1b88ca8cf279bcab3e2f12934e", "license": "Apache-2.0"},
    "OneJev": {"commit": "6ce62b3a57a75deb60fde3bcc3dbaa4068c73650", "license": "Apache-2.0"},
    "YesTrack": {"commit": "ed42b8315b28c94dc595d83b0d49bf57002bb281", "license": "MIT"},
    "MOTIP": {"commit": "ffc0e905ac196a603027eca8d18fb0dff48c8bcc", "license": "Apache-2.0"},
    "MeMOTR": {"repository": "MCG-NJU/MeMOTR", "license": "MIT"},
    "SAM2": {"commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4", "license": "Apache-2.0"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    local = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "files": [
            "locatemot/groundjev/coupled/shared_ground_state.py",
            "locatemot/groundjev/coupled/g2_feature_bridge.py",
            "locatemot/groundjev/coupled/g2_jev_readout.py",
            "locatemot/groundjev/coupled/lifecycle_controller.py",
            "locatemot/groundjev/coupled/memory_writer.py",
            "locatemot/groundjev/coupled/identity_memory.py",
        ],
    }
    payload = {"format": "groundjev-coupled-source-audit-v1", "references": REFERENCES, "local": local}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
