#!/usr/bin/env python3
"""Strictly reload the completed L83 faithful-probe packages.

This is a post-run implementation audit. It uses only checkpoint metadata and
deterministic synthetic finite inputs; no calibration, validation, screening,
or official-test labels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l83_faithful_rank_probe import L83FaithfulRankProbe, L83RankProbeConfig

THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
SOURCE = ROOT / "outputs/l83/train/faithful_bag_attempt1/checkpoints"
REPRESENTATIONS = ("l59_fused_roi", "l81_candidate_evidence", "l82_candidate_reference")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    manifest_sha = sha256(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST:
        raise AssertionError(f"manifest SHA drift: {manifest_sha}")

    torch.manual_seed(20260829)
    representation_results = []
    all_ok = True
    for name in REPRESENTATIONS:
        checkpoint = SOURCE / f"{name}_checkpoint_epoch10.pt"
        package = torch.load(checkpoint, map_location="cpu")
        config_dict = package.get("model_config", {})
        config = L83RankProbeConfig(**config_dict)
        first = L83FaithfulRankProbe(config)
        second = L83FaithfulRankProbe(config)
        first.load_state_dict(package["model_state_dict"], strict=True)
        second.load_state_dict(package["model_state_dict"], strict=True)
        first.eval()
        second.eval()
        sample = torch.linspace(
            -1.0, 1.0, steps=2 * 5 * config.input_dim, dtype=torch.float32
        ).reshape(2, 5, config.input_dim)
        with torch.inference_mode():
            first_output = first(sample)["interaction"]
            second_output = second(sample)["interaction"]
        max_difference = float((first_output - second_output).abs().max())
        finite = bool(torch.isfinite(first_output).all() and torch.isfinite(second_output).all())
        shape = list(first_output.shape)
        strict_ok = shape == [2, 5] and finite and max_difference == 0.0
        all_ok = all_ok and strict_ok
        representation_results.append({
            "representation": name,
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
            },
            "package_format": package.get("format"),
            "package_step": package.get("step"),
            "config": config_dict,
            "package_parameter_count": package.get("model_parameter_count"),
            "strict_load": True,
            "synthetic_input_shape": list(sample.shape),
            "output_shape": shape,
            "finite": finite,
            "max_reload_output_difference": max_difference,
            "strict_reload_pass": strict_ok,
        })
        del package, first, second, sample, first_output, second_output

    status = "complete" if all_ok else "reload_contract_fail"
    common = {
        "format": "locatemot-l83-faithful-reload-audit-v1",
        "status": status,
        "command": " ".join([sys.executable] + sys.argv),
        "inputs": {
            "checkpoint_dir": str(SOURCE),
            "manifest": {"path": str(MANIFEST), "sha256": manifest_sha},
        },
        "outputs": {"representations": list(REPRESENTATIONS)},
        "failure_root_cause": None if all_ok else "strict reload or finite output check failed",
        "next_action": "retain faithful probe evidence; do not continue conditional L83 phases" if all_ok else "preserve INCOMPLETE.md and inspect first failed representation",
        "luna_thread": THREAD,
        "seed": 20260829,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
        "training_run": False,
        "validation_labels_read": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
        "representation_results": representation_results,
    }
    write_json(out / "reload_audit.json", common)
    write_json(out / "status.json", common)
    if not all_ok:
        (out / "INCOMPLETE.md").write_text(
            "L83 strict reload audit failed; see reload_audit.json for the first failed representation.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
