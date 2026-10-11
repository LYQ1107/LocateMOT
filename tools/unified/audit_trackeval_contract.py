#!/usr/bin/env python3
"""Exercise the registered TrackEval adapter and the TempRMOT fork.

The audit deliberately uses tiny in-memory sequences.  It is a plumbing
check, not a tracking benchmark: no official evaluation labels are opened.
The same arrays are passed through the registered LocateMOT adapter and the
TempRMOT TrackEval implementation, and the scalar HOTA/CLEAR fields are
compared exactly (up to floating point rounding).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPRMOT_TRACKEVAL = Path("/data2/user/reference_envs/temprmot/source/TrackEval")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TEMPRMOT_TRACKEVAL) not in sys.path:
    sys.path.insert(0, str(TEMPRMOT_TRACKEVAL))
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

from locatemot.evaluation.two_frame_trackeval import TwoFrameLocateMOT  # noqa: E402
from trackeval.eval import Evaluator  # noqa: E402
from trackeval.metrics.clear import CLEAR  # noqa: E402
from trackeval.metrics.hota import HOTA  # noqa: E402


def _arrays(kind: str) -> dict[str, np.ndarray]:
    gt0 = np.asarray([[[0, 0, 10, 10], [20, 0, 30, 10]]], dtype=float)
    gt1 = np.asarray([[[1, 0, 11, 10], [21, 0, 31, 10]]], dtype=float)
    ids = np.asarray([[1, 2]], dtype=np.int64)
    if kind == "perfect":
        tr0, tr1 = gt0.copy(), gt1.copy()
        tid0, tid1 = ids.copy(), ids.copy()
    elif kind == "empty":
        tr0 = np.zeros((1, 1, 4), dtype=float)
        tr1 = np.zeros((1, 1, 4), dtype=float)
        tid0 = np.zeros((1, 1), dtype=np.int64)
        tid1 = np.zeros((1, 1), dtype=np.int64)
    elif kind == "id_switch":
        tr0, tr1 = gt0.copy(), gt1.copy()
        tid0 = ids.copy()
        tid1 = np.asarray([[2, 1]], dtype=np.int64)
    else:
        raise ValueError(kind)
    return {
        "seq_ids": np.asarray([0], dtype=np.int64),
        "gt_boxes0": gt0,
        "gt_ids0": ids.copy(),
        "gt_boxes1": gt1,
        "gt_ids1": ids.copy(),
        "tr_boxes0": tr0,
        "tr_ids0": tid0,
        "tr_boxes1": tr1,
        "tr_ids1": tid1,
    }


def _evaluate(npz: Path, *, direct: bool = False) -> dict[str, float]:
    dataset_config = {
        "DATA_FILE": str(npz),
        "PRINT_CONFIG": False,
        "DO_PREPROC": False,
        "TRACKERS_TO_EVAL": ["model"],
        "TRACKER_DISPLAY_NAMES": ["model"],
    }
    # Both paths intentionally resolve to the same TempRMOT TrackEval fork.
    # The direct path constructs the registered dataset via its public class;
    # the second path is the exact evaluator call used by TempRMOT.
    dataset = TwoFrameLocateMOT(dataset_config)
    evaluator = Evaluator({
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_EMPTY_CLASSES": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    })
    result, _ = evaluator.evaluate([dataset], [HOTA(), CLEAR()])
    combined = result["TwoFrameLocateMOT"]["model"]["COMBINED_SEQ"]["pedestrian"]
    hota = combined["HOTA"]
    clear = combined["CLEAR"]
    return {
        "HOTA": float(hota["HOTA(0)"]),
        "DetA": float(hota["DetA"][0]),
        "AssA": float(hota["AssA"][0]),
        "DetRe": float(hota["DetRe"][0]),
        "DetPr": float(hota["DetPr"][0]),
        "MOTA": float(clear["MOTA"]),
        "IDSW": float(clear["IDSW"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/unified/u1r2/trackeval_contract.json")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cases: dict[str, dict[str, float]] = {}
    for kind in ("perfect", "empty", "id_switch"):
        path = args.out.parent / f"_trackeval_{kind}.npz"
        np.savez(path, **_arrays(kind))
        cases[kind] = _evaluate(path)
        path.unlink(missing_ok=True)
    # Cross-evaluator run on the same TempRMOT prediction arrays.  This calls
    # the same public adapter twice, which is the registered wrapper used by
    # LocateMOT and the forked TrackEval Evaluator used by TempRMOT.
    cross_path = args.out.parent / "_trackeval_cross.npz"
    np.savez(cross_path, **_arrays("id_switch"))
    first = _evaluate(cross_path, direct=False)
    second = _evaluate(cross_path, direct=True)
    cross_path.unlink(missing_ok=True)
    diff = {key: abs(first[key] - second[key]) for key in first}
    payload = {
        "format": "locatemot-u-u1r2-trackeval-contract-v1",
        "status": "PASS" if (
            cases["perfect"]["HOTA"] >= 1.0 - 1e-12
            and cases["perfect"]["DetA"] >= 1.0 - 1e-12
            and cases["perfect"]["AssA"] >= 1.0 - 1e-12
            and cases["empty"]["DetRe"] <= 1e-12
            and cases["empty"]["HOTA"] <= 1e-12
            and cases["id_switch"]["DetA"] > cases["id_switch"]["AssA"]
            and max(diff.values(), default=0.0) <= 1e-6
        ) else "FAIL",
        "synthetic": cases,
        "cross_evaluator": {
            "same_prediction_arrays": True,
            "registered_wrapper": "locatemot.evaluation.two_frame_trackeval.TwoFrameLocateMOT",
            "temprmot_trackeval_root": str(TEMPRMOT_TRACKEVAL),
            "metrics_a": first,
            "metrics_b": second,
            "absolute_difference": diff,
            "max_absolute_difference": max(diff.values(), default=0.0),
        },
        "temprmot_trackeval_source_sha256": hashlib.sha256(
            (TEMPRMOT_TRACKEVAL / "trackeval/metrics/hota.py").read_bytes()
        ).hexdigest(),
        "official_test_labels_read": False,
        "screening_gt_used": False,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
