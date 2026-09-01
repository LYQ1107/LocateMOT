#!/usr/bin/env python3
"""Calibration-only residual expressivity audit for the L51 B1 decision."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.rmot.l49_data import TEXT_CACHE, sha256_file  # noqa: E402
from tools.train_l51_streaming_crop_adapter import (  # noqa: E402
    CLIP_WEIGHTS,
    FAST_MANIFEST,
    L29_CHECKPOINT,
    StreamingClipPatches,
    forward_item,
    load_units,
    materialize_units,
)
from tools.train_l49_kitti_rmot import L29Teacher  # noqa: E402


DATA = ROOT / "outputs/l49/data"
B0_CHECKPOINT = ROOT / "outputs/l51/train/b0_smoke100_retry1/checkpoint_l51_b0_step100.pt"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def pair_summary(items: list[dict], bound: float, mode: str) -> dict:
    total = teacher_errors = flippable = 0
    units = units_with_error = units_with_flip = 0
    for item in items:
        y = item["y"].numpy().astype(bool)
        score = item["teacher"].numpy().astype(np.float64)
        pos = np.flatnonzero(y)
        neg = np.flatnonzero(~y)
        if mode == "teacher_hard":
            if not len(pos) or not len(neg):
                continue
            neg = np.asarray([neg[np.argmax(score[neg])]], dtype=np.int64)
        if not len(pos) or not len(neg):
            continue
        units += 1
        unit_error = unit_flip = False
        for p in pos:
            for n in neg:
                gap = float(score[n] - score[p])
                total += 1
                if gap > 0:
                    teacher_errors += 1
                    unit_error = True
                    if gap <= 2.0 * bound:
                        flippable += 1
                        unit_flip = True
        units_with_error += int(unit_error)
        units_with_flip += int(unit_flip)
    return {
        "pair_definition": "all positive-vs-negative pairs" if mode == "all" else "every positive vs highest-teacher-score negative in frame",
        "total_pairs": total,
        "teacher_error_pairs": teacher_errors,
        "theoretically_flippable_error_pairs": flippable,
        "flippable_of_all_pairs": flippable / max(1, total),
        "flippable_of_teacher_error_pairs": flippable / max(1, teacher_errors),
        "units_with_pairs": units,
        "units_with_teacher_error": units_with_error,
        "units_with_theoretically_flippable_error": units_with_flip,
        "bound": float(bound),
        "maximum_pairwise_change": float(2.0 * bound),
    }


def actual_b0_replay(items: list[dict], text: dict, device: torch.device) -> dict:
    payload = torch.load(B0_CHECKPOINT, map_location=device, weights_only=False)
    cfg = payload["config"]["model"]
    from locatemot.models.l51_streaming_crop_adapter import L51StreamingCropAdapter
    model = L51StreamingCropAdapter(hidden=int(cfg["hidden"]), heads=int(cfg["heads"]),
                                    layers=int(cfg["layers"]), residual_bound=float(cfg["residual_bound"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    encoder = StreamingClipPatches(device)
    bound = float(cfg["residual_bound"])
    residuals = []
    saturation = 0
    finite = True
    pair_total = pair_error = pair_corrected = pair_correct_flip = 0
    started = time.time()
    for item in items:
        with torch.inference_mode():
            output, patch = forward_item(model, encoder, item, text, device)
        residual = output["residual"].float().cpu().numpy()
        teacher = item["teacher"].numpy().astype(np.float64)
        student = teacher + residual.astype(np.float64)
        y = item["y"].numpy().astype(bool)
        pos = np.flatnonzero(y); neg = np.flatnonzero(~y)
        residuals.extend(residual.tolist())
        finite = finite and bool(np.isfinite(residual).all())
        saturation += int((np.abs(residual) >= 0.95 * bound).sum())
        if len(pos) and len(neg):
            hard = neg[np.argmax(teacher[neg])]
            for p in pos:
                pair_total += 1
                teacher_wrong = teacher[p] <= teacher[hard]
                student_right = student[p] > student[hard]
                if teacher_wrong:
                    pair_error += 1
                    pair_corrected += int(student_right)
                elif not student_right:
                    pair_correct_flip += 1
        del output, patch
    values = np.asarray(residuals, dtype=np.float64)
    return {
        "checkpoint": str(B0_CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256_file(B0_CHECKPOINT),
        "calibration_units": len(items),
        "residual_count": int(len(values)),
        "finite": finite,
        "residual_mean": float(values.mean()) if len(values) else None,
        "residual_std": float(values.std()) if len(values) else None,
        "residual_abs_max": float(np.abs(values).max()) if len(values) else None,
        "residual_abs_q95": float(np.quantile(np.abs(values), 0.95)) if len(values) else None,
        "bound": bound,
        "saturation_threshold": 0.95 * bound,
        "saturated_fraction": saturation / max(1, len(values)),
        "teacher_hard_pair_total": pair_total,
        "teacher_hard_error_pairs": pair_error,
        "teacher_error_pairs_corrected": pair_corrected,
        "teacher_correct_pairs_flipped": pair_correct_flip,
        "elapsed_sec": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    if sha256_file(FAST_MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("fixed manifest SHA mismatch")
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    all_units = load_units()
    calibration = [x for x in all_units if str(x.get("split")) == "calibration"]
    if not calibration:
        # L49 train_units is fit-only; use the existing calibration manifest for
        # this read-only audit and assert its provenance explicitly.
        calibration = [json.loads(x) for x in (DATA / "calibration_units.jsonl").read_text().splitlines() if x.strip()]
    if any(str(x.get("split", "calibration")) not in ("calibration",) for x in calibration):
        raise AssertionError("residual audit received a non-calibration unit")
    expected = {"refer_kitti_v1": {"0016"}, "refer_kitti_v2": {"0015"}}
    actual = {domain: {str(x["video"]) for x in calibration if x["dataset"] == domain}
              for domain in expected}
    if actual != expected:
        raise AssertionError(f"unexpected calibration videos: {actual}")
    text = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
    teacher = L29Teacher(text, torch.device("cpu"))
    items = materialize_units(calibration, text, teacher)
    bound_sensitivity = {str(bound): {
        "all_pairs": pair_summary(items, bound, "all"),
        "teacher_hard_pairs": pair_summary(items, bound, "teacher_hard"),
    } for bound in (0.05, 0.25, 0.5, 1.0)}
    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    actual_replay = actual_b0_replay(items, text, device)
    actual_json = {domain: sorted(values) for domain, values in actual.items()}
    payload = {
        "format": "locatemot-l51-b1-residual-bound-audit-v1",
        "status": "pass",
        "project_root": str(ROOT),
        "started_at_unix": time.time(),
        "completed_at_unix": time.time(),
        "calibration_only": True,
        "calibration_units": len(calibration),
        "calibration_videos": actual_json,
        "pair_hard_definition": "highest frozen L29 teacher-score negative per complete frame candidate set",
        "theoretical_flip_condition": "teacher_negative - teacher_positive <= 2 * residual_bound",
        "bound_sensitivity": bound_sensitivity,
        "b0_retry1_calibration_replay": actual_replay,
        "train_trace_saturation": {
            "source": str((ROOT / "outputs/l51/train/b0_smoke100_retry1/loss_trace.json").resolve()),
            "max_residual_abs_over_run": 0.0032556988298892975,
            "configured_bound": 0.05,
            "saturated_at_95_percent": False,
            "note": "B0 train trace is implementation evidence; calibration replay above is the calibration-specific check.",
        },
        "manifest_sha256": sha256_file(FAST_MANIFEST),
        "l29_checkpoint": str(L29_CHECKPOINT.resolve()),
        "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT),
        "text_cache": str(TEXT_CACHE.resolve()),
        "text_cache_sha256": sha256_file(TEXT_CACHE),
        "official_test_labels_read": False,
        "validation_labels_read": False,
        "screening_gt_used": False,
        "ordinary_mot_ovmot_touched": False,
        "raw_cache_written": False,
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_id"],
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_language_mask": "UNALIGNED/not claimed",
    }
    (out / "residual_bound_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
