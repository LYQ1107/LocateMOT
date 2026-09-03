#!/usr/bin/env python3
"""L87-A: L86-compatible fresh training with corrected temporal negatives.

The L86 loop is reused so architecture, curriculum, optimizer, seed and
checkpoint cadence remain identical.  Only the imported temporal loss is
replaced by ``l87a_loss``.  Old assets are read through LOCATEMOT_ASSET_ROOT;
all new outputs are rooted in this isolated worktree.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get(
    "LOCATEMOT_ASSET_ROOT", "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
)).resolve()
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))
sys.path.insert(0, str(WORK_ROOT / "locatemot" / "rmot"))

from l87a_losses import l87a_loss  # noqa: E402
from tools import l86_train_full_rmot as base  # noqa: E402


def _replace(value: object) -> object:
    if isinstance(value, str):
        return value.replace("l86", "l87a").replace("L86", "L87-A")
    if isinstance(value, dict):
        return {key: _replace(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace(item) for item in value]
    return value


def _annotate(out: Path) -> None:
    for name in ("config.json", "provenance.json", "status.json"):
        path = out / name
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        value = _replace(value)
        value.update({
            "stage": "L87-A corrected temporal retraining",
            "work_root": str(WORK_ROOT),
            "asset_root": str(ASSET_ROOT),
            "temporal_negative_contract": "(previous_available | current_available) - referred_targets",
            "temporal_negative_source": "real candidate_gt target bags; no synthetic objectness negatives",
            "new_science_change_only": "temporal negative construction",
            "gpu_mapping": [0, 2, 8],
            "base_l86_commit": "97bff208929474d4c4b0d659c80e7eba2f3f5d0a",
            "common_eval_policy_sha": os.environ.get("L87_COMMON_EVAL_POLICY_SHA", "recorded at launch"),
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
        })
        path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    # Redirect the L86 implementation through the isolated source/output root
    # and replace only its temporal loss symbol before entering the loop.
    base.ROOT = WORK_ROOT
    base.MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
    base.DEFAULT_CACHE = ASSET_ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
    base.l86_loss = l87a_loss
    result = base.main()
    if result == 0 and int(os.environ.get("RANK", "0")) == 0:
        out = Path("outputs/l87a/train/joint40").resolve()
        # The command supplies --out; recover it without changing the base
        # parser/loop contract.
        if "--out" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
        _annotate(out)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
