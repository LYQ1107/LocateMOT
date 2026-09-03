#!/usr/bin/env python3
"""Run the unchanged local TrackEval wrapper on the isolated L87-B output."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get(
    "LOCATEMOT_ASSET_ROOT", "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
)).resolve()
if str(WORK_ROOT) not in sys.path: sys.path.insert(0, str(WORK_ROOT))

from tools import l86_run_trackeval as base  # noqa: E402


def main() -> int:
    base.ROOT = WORK_ROOT
    base.MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
    result = base.main()
    if result == 0 and "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
        for name in ("trackeval_summary.json", "provenance.json", "status.json"):
            path = out / name
            if not path.is_file(): continue
            value = json.loads(path.read_text())
            value.update({"format": "locatemot-l87b-internal-fullvideo-trackeval-v1",
                          "stage": "L87-B zero-train corrected reselection and deployment",
                          "work_root": str(WORK_ROOT), "asset_root": str(ASSET_ROOT),
                          "base_l86_commit": "97bff208929474d4c4b0d659c80e7eba2f3f5d0a",
                          "screening_gt_used": False, "official_test_labels_read": False,
                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
                          "no_hota_or_trackeval": False})
            path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    return result

if __name__ == "__main__": raise SystemExit(main())
