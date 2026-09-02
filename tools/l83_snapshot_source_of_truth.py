#!/usr/bin/env python3
"""Record and validate the immutable L83 source-of-truth snapshot."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
EXPECTED_HEAD = "75e9f9cd0482645c07a9f71ad4419b0c5f57132b"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
UIDM = ROOT / "outputs/l11/checkpoints/uidm_l11_main/step11000.pt"
EXPECTED_UIDM = "f6529743630e335b947118bf860cc2215a95315a4c551351e6866ea9e1076343"
L69_MANIFEST = ROOT / "outputs/l69/attempt9/budget40_features/kitti/manifest.json"
L69_ROOT = L69_MANIFEST.parent
L48_TEXT = ROOT / "outputs/l48/data/text_cache.pt"
L81 = ROOT / "outputs/l81/train/probe500_retry1/checkpoint_l81_step100.pt"
L82 = ROOT / "outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l82_candidate_reference_checkpoint_epoch10.pt"
L59 = ROOT / "outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l59_fused_roi_checkpoint_epoch10.pt"
L81_CONTROL = ROOT / "outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l81_candidate_evidence_checkpoint_epoch10.pt"
MMDET_ROOT = Path("/data1/LWR/vranlee/LLM/mmdetection-3.3.0").resolve()
MMDET_CONFIG = MMDET_ROOT / "configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
MMDET_WEIGHT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth").resolve()
BERT_ROOT = Path("/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594").resolve()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def meta(path: Path, directory_manifest: bool = False) -> dict[str, Any]:
    path = path.resolve()
    result: dict[str, Any] = {
        "path": str(path), "exists": path.exists(), "is_file": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
        "sha256": sha256_file(path),
    }
    if directory_manifest and path.is_dir():
        result["files"] = [
            {"relative": str(item.relative_to(path)), "bytes": item.stat().st_size,
             "sha256": sha256_file(item)}
            for item in sorted(path.rglob("*")) if item.is_file()
        ]
    return result


def git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    head = git(["rev-parse", "HEAD"])
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"])
    assets = {
        "fixed_manifest": meta(MANIFEST),
        "uidm_step11000": meta(UIDM),
        "l69_feature_manifest": meta(L69_MANIFEST),
        "l69_feature_files": meta(L69_ROOT, directory_manifest=True),
        "l48_text_cache": meta(L48_TEXT),
        "l81_step100": meta(L81),
        "l82_candidate_reference": meta(L82),
        "l59_control": meta(L59),
        "l81_control": meta(L81_CONTROL),
        "groundingdino_config": meta(MMDET_CONFIG),
        "groundingdino_weight": meta(MMDET_WEIGHT),
        "bert_snapshot": meta(BERT_ROOT, directory_manifest=True),
        "mmdetection_checkout": {
            "path": str(MMDET_ROOT), "exists": MMDET_ROOT.is_dir(),
            "git_head": None,
            "status": "local checkout has no verifiable git HEAD",
        },
    }
    mismatches: list[str] = []
    if head != EXPECTED_HEAD:
        mismatches.append(f"git HEAD {head!r} != {EXPECTED_HEAD!r}")
    if assets["fixed_manifest"]["sha256"] != EXPECTED_MANIFEST:
        mismatches.append("fixed manifest SHA drift")
    if assets["uidm_step11000"]["sha256"] != EXPECTED_UIDM:
        mismatches.append("UIDM step11000 SHA drift")
    missing = [name for name, value in assets.items() if name != "mmdetection_checkout" and not value.get("exists")]
    if missing:
        mismatches.append(f"missing assets: {missing}")
    status = "complete" if not mismatches else "source_of_truth_mismatch"
    payload = {
        "format": "locatemot-l83-source-of-truth-v1", "status": status,
        "project_root": str(ROOT), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "branch": branch, "git_head": head, "expected_git_head": EXPECTED_HEAD,
        "git_worktree_dirty_allowed": True, "inputs": assets, "mismatches": mismatches,
        "command": " ".join([str(Path(__file__).resolve())]),
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "l81_modified": False, "l82_modified": False,
        "uidm_shared_checkpoint_modified": False,
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_alignment": "UNALIGNED",
    }
    out = ROOT / "outputs/l83/preregister/source_of_truth.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if mismatches:
        raise SystemExit("source_of_truth_mismatch: " + "; ".join(mismatches))
    print(json.dumps({"status": status, "branch": branch, "head": head, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
