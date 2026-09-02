#!/usr/bin/env python3
"""Create the immutable L84 source-of-truth snapshot.

The snapshot is deliberately L84-only.  It records hashes and metadata but
does not alter any frozen asset or materialize feature tensors.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
EXPECTED_BASE_HEAD = "5fd92b87927d60c831e4ac75774929f07f371d7e"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
UIDM = ROOT / "outputs/l11/checkpoints/uidm_l11_main/step11000.pt"
EXPECTED_UIDM = "f6529743630e335b947118bf860cc2215a95315a4c551351e6866ea9e1076343"
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L48 = ROOT / "outputs/l48/data/text_cache.pt"
L81 = ROOT / "outputs/l81/train/probe500_retry1/checkpoint_l81_step100.pt"
L82 = ROOT / "outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l82_candidate_reference_checkpoint_epoch10.pt"
L59 = ROOT / "outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l59_fused_roi_checkpoint_epoch10.pt"
L81_CONTROL = ROOT / "outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l81_candidate_evidence_checkpoint_epoch10.pt"
MMDET_ROOT = Path("/data1/LWR/vranlee/LLM/mmdetection-3.3.0").resolve()
MMDET_CONFIG = MMDET_ROOT / "configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
MMDET_WEIGHT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main/download/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth").resolve()
BERT_ROOT = Path("/home/lwr/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594").resolve()
SOURCE_FILES = (
    ROOT / "locatemot/evaluation/l83_target_bag_metrics.py",
    ROOT / "locatemot/rmot/l83_target_bag_loss.py",
    ROOT / "locatemot/rmot/l83_target_bags.py",
    ROOT / "locatemot/models/l83_grounding_state_audit.py",
    ROOT / "tools/l83_audit_decoder_sharpness.py",
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def meta(path: Path, directory_manifest: bool = False) -> dict[str, Any]:
    path = path.resolve()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
        "sha256": sha256_file(path),
    }
    if directory_manifest and path.is_dir():
        result["files"] = [
            {
                "relative": str(item.relative_to(path)),
                "bytes": item.stat().st_size,
                "mtime_ns": item.stat().st_mtime_ns,
                "sha256": sha256_file(item),
            }
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ]
    return result


def git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
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
        "l69_feature_manifest": meta(L69_ROOT / "manifest.json"),
        "l69_feature_files": meta(L69_ROOT, directory_manifest=True),
        "l48_text_cache": meta(L48),
        "l81_step100": meta(L81),
        "l82_candidate_reference": meta(L82),
        "l59_control": meta(L59),
        "l81_control": meta(L81_CONTROL),
        "groundingdino_config": meta(MMDET_CONFIG),
        "groundingdino_weight": meta(MMDET_WEIGHT),
        "bert_snapshot": meta(BERT_ROOT, directory_manifest=True),
        "l83_source_files": {str(path.relative_to(ROOT)): meta(path) for path in SOURCE_FILES},
    }
    mismatches: list[str] = []
    if head != EXPECTED_BASE_HEAD:
        mismatches.append(f"starting HEAD {head!r} != L83 authoritative commit {EXPECTED_BASE_HEAD!r}")
    if assets["fixed_manifest"]["sha256"] != EXPECTED_MANIFEST:
        mismatches.append("fixed manifest SHA drift")
    if assets["uidm_step11000"]["sha256"] != EXPECTED_UIDM:
        mismatches.append("UIDM step11000 SHA drift")
    missing = [
        name for name, value in assets.items()
        if name != "l83_source_files"
        and name != "mmdetection_checkout"
        and not value.get("exists", False)
    ]
    missing.extend(
        f"l83_source_files:{name}"
        for name, value in assets["l83_source_files"].items()
        if not value.get("exists", False)
    )
    if missing:
        mismatches.append(f"missing assets: {missing}")
    payload = {
        "format": "locatemot-l84-source-of-truth-v1",
        "status": "complete" if not mismatches else "source_of_truth_mismatch",
        "project_root": str(ROOT),
        "luna_thread": THREAD,
        "branch": branch,
        "git_head": head,
        "expected_base_head": EXPECTED_BASE_HEAD,
        "inputs": assets,
        "mismatches": mismatches,
        "local_mmdetection": {
            "path": str(MMDET_ROOT),
            "git_head": None,
            "status": "local checkout has no verifiable git HEAD",
            "official_reference": {
                "repository": "https://github.com/open-mmlab/mmdetection",
                "tag": "v3.3.0",
                "commit": "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a",
            },
        },
        "candidate_deletion": False,
        "candidate_truncation": False,
        "features_persistent": False,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_alignment": "UNALIGNED",
        "uidm_shared_checkpoint_modified": False,
        "command": " ".join([str(Path(__file__).resolve())]),
    }
    out = ROOT / "outputs/l84/preregister/source_of_truth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    frozen = {
        "format": "locatemot-l84-frozen-assets-v1",
        "status": payload["status"],
        "luna_thread": THREAD,
        "source_of_truth": str(out),
        "assets": assets,
        "expected_manifest_sha256": EXPECTED_MANIFEST,
        "candidate_deletion": False,
        "candidate_truncation": False,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
    }
    frozen_out = ROOT / "outputs/l84/preregister/frozen_assets.json"
    frozen_out.write_text(json.dumps(frozen, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    if mismatches:
        raise SystemExit("source_of_truth_mismatch: " + "; ".join(mismatches))
    print(json.dumps({"status": payload["status"], "branch": branch, "head": head, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
