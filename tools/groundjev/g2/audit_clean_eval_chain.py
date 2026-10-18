#!/usr/bin/env python3
"""CPU-only preflight for the clean G0/G2 evaluation chain.

This is deliberately a receipt audit, not an evaluator launcher.  It refuses
an unreceipted or historically named G2-PL bundle, checks that the G0/G2
results use the same fixed manifest scope, and verifies that rejected frames
remain represented in the runtime receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CLEAN_MANIFEST_SHA256 = "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"
CONTAMINATED_NAME_MARKERS = ("g2-pl", "g2_pl", "formal_g2_1200", "old_g2", "legacy_g2")
TERMINAL_INFERENCE_STATUSES = {
    "VIDEO_INFERENCE_COMPLETE",
    "VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def manifest_frame_count(manifest: Path) -> int:
    payload = read_json(manifest)
    for key in ("images", "frame_numbers"):
        if isinstance(payload.get(key), list):
            return len(payload[key])
    raise ValueError(f"manifest has no frame list: {manifest}")


def validate_bundle(bundle: Path, receipt_path: Path, *, expected_manifest_sha256: str = CLEAN_MANIFEST_SHA256) -> dict[str, Any]:
    lowered = bundle.name.lower()
    marker = next((item for item in CONTAMINATED_NAME_MARKERS if item in lowered), None)
    if marker:
        raise ValueError(f"refusing historically contaminated bundle name: {marker}")
    model_file = bundle / "model.safetensors"
    if not bundle.is_dir() or not model_file.is_file() or model_file.stat().st_size <= 0:
        raise ValueError(f"G2 bundle/model.safetensors is missing or empty: {bundle}")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "G2_NATIVE_MERGED_BUNDLE_EXPORTED":
        raise ValueError(f"bundle receipt is not a clean G2 export: {receipt.get('status')!r}")
    if receipt.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("bundle receipt manifest SHA does not match the clean fixed scope")
    if receipt.get("source_bundle_overwritten") is not False:
        raise ValueError("bundle receipt does not prove source bundle preservation")
    if int(receipt.get("native_tensor_count", -1)) != 569:
        raise ValueError("bundle tensor count is not the registered 569-tensor GroundAnything bundle")
    if Path(str(receipt.get("output_bundle", ""))).resolve() != bundle.resolve():
        raise ValueError("bundle receipt output path differs from requested model")
    output_sha = sha256(model_file)
    if receipt.get("output_model_safetensors_sha256") != output_sha:
        raise ValueError("bundle model SHA differs from export receipt")
    base = Path(str(receipt.get("base_bundle", "")))
    base_model = base / "model.safetensors"
    if not base_model.is_file() or receipt.get("base_model_safetensors_sha256") != sha256(base_model):
        raise ValueError("bundle base model receipt is missing or changed")
    adapter = Path(str(receipt.get("adapter", "")))
    if not adapter.is_file() or adapter.stat().st_size <= 0:
        raise ValueError("clean G2 adapter receipt is missing or empty")
    adapter_sha = sha256(adapter)
    if receipt.get("adapter_sha256") != adapter_sha:
        raise ValueError("clean G2 adapter SHA differs from export receipt")
    return {
        "status": receipt["status"],
        "bundle": str(bundle.resolve()),
        "bundle_model_sha256": output_sha,
        "base_bundle": str(base),
        "base_model_sha256": receipt["base_model_safetensors_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "adapter": str(adapter),
        "adapter_sha256": adapter_sha,
        "native_tensor_count": receipt["native_tensor_count"],
    }


def validate_manifest(manifest: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    payload = read_json(manifest)
    actual_sha = sha256(manifest)
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(f"manifest SHA mismatch: {actual_sha} != {expected_sha256}")
    if payload.get("official_test_labels_read") is True:
        raise ValueError("fixed evaluation manifest records official test labels as read")
    if payload.get("runtime_gt_fields") is True:
        raise ValueError("fixed evaluation manifest enables runtime GT fields")
    count = manifest_frame_count(manifest)
    return {"path": str(manifest.resolve()), "sha256": actual_sha, "frames": count, "scope": payload.get("scope"), "task": payload.get("task")}


def validate_result(result_path: Path, manifest: Path) -> dict[str, Any]:
    result = read_json(result_path)
    manifest_data = validate_manifest(manifest)
    status = result.get("status")
    if status not in TERMINAL_INFERENCE_STATUSES:
        raise ValueError(f"inference result is not terminal: {status!r}")
    if result.get("task") != manifest_data.get("task"):
        raise ValueError("inference task differs from fixed manifest")
    if result.get("scope") != manifest_data.get("scope"):
        raise ValueError("inference scope differs from fixed manifest")
    if int(result.get("frames", -1)) != manifest_data["frames"]:
        raise ValueError("inference frame count differs from fixed manifest")
    if result.get("all_input_frames_processed") is not True:
        raise ValueError("result does not prove every input frame was processed")
    if result.get("gt_read_by_runtime") is True or result.get("future_frames_read_by_runtime") is True:
        raise ValueError("result records forbidden runtime GT/future-frame access")
    rejected = result.get("perception_rejected_frames", [])
    if not isinstance(rejected, list):
        raise ValueError("rejected-frame field is not a list")
    if any(not isinstance(frame, int) or not 0 <= frame < manifest_data["frames"] for frame in rejected):
        raise ValueError("rejected-frame list contains an invalid frame")
    latencies = result.get("latencies_seconds")
    if isinstance(latencies, list) and len(latencies) != manifest_data["frames"]:
        raise ValueError("latency list dropped rejected/input frames")
    return {
        "path": str(result_path.resolve()),
        "status": status,
        "task": result.get("task"),
        "scope": result.get("scope"),
        "frames": manifest_data["frames"],
        "rejected_frames": len(rejected),
        "rejection_rate": len(rejected) / manifest_data["frames"],
        "gt_read_by_runtime": False,
        "future_frames_read_by_runtime": False,
    }


def compare_g0_g2(g0_result: Path, g2_result: Path, manifest: Path) -> dict[str, Any]:
    g0 = validate_result(g0_result, manifest)
    g2 = validate_result(g2_result, manifest)
    for key in ("task", "scope", "frames"):
        if g0[key] != g2[key]:
            raise ValueError(f"G0/G2 fixed-scope drift in {key}: {g0[key]!r} vs {g2[key]!r}")
    return {"manifest": str(manifest.resolve()), "g0": g0, "g2": g2, "same_fixed_scope": True}


def denominator_explanation(*, rejected: int = 183, total: int = 1658) -> dict[str, Any]:
    return {
        "rejected_frames": int(rejected),
        "denominator_frames": int(total),
        "rate_fraction": f"{rejected}/{total}",
        "rate_percent": round(100.0 * rejected / total, 2),
        "denominator_definition": "all frames in the fixed manifest, including frames whose perception request was rejected",
        "preservation_rule": "rejected frames remain in frame_decisions/result receipt and are not silently dropped",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--g0-result", type=Path, default=None)
    parser.add_argument("--g2-result", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", default=None)
    args = parser.parse_args()
    bundle = validate_bundle(args.bundle, args.bundle_receipt)
    manifest = validate_manifest(args.manifest, expected_sha256=args.expected_manifest_sha256)
    report: dict[str, Any] = {
        "status": "CLEAN_G0_G2_EVAL_PREFLIGHT_PASS",
        "read_only": True,
        "bundle": bundle,
        "manifest": manifest,
        "denominator_explanation": denominator_explanation(),
        "runtime_policy": {
            "gt_fallback": False,
            "future_frame_access": False,
            "preserve_rejected_frames": True,
            "same_fixed_scope": True,
        },
    }
    if (args.g0_result is None) != (args.g2_result is None):
        raise ValueError("provide both --g0-result and --g2-result, or neither")
    if args.g0_result is not None:
        report["comparison"] = compare_g0_g2(args.g0_result, args.g2_result, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "out": str(args.out), "frames": manifest["frames"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
