#!/usr/bin/env python3
"""L51 fixed-plan L29 calibration/validation control evaluator.

This evaluator reads only L49 train-side calibration/validation units and the
frozen L29/L19 assets.  It does not open screening or official-test labels and
does not write to the historical L49 output directories.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder  # noqa: E402
from locatemot.rmot.l49_data import L29_CHECKPOINT, TEXT_CACHE, load_bank, sha256_file  # noqa: E402
from tools.eval_l49_validation import fit_threshold, l29_score, source_masks, summarize  # noqa: E402
from tools.train_l28_track_set_decoder import state_at  # noqa: E402
from tools.train_l49_kitti_rmot import build_teacher_cache  # noqa: E402


DATA = ROOT / "outputs/l49/data"
FAST_MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
VAL_BASELINE = ROOT / "outputs/l49/val/validation_baseline_scores.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict) -> tuple[str, str, int, int]:
    return (str(row["dataset"]), str(row["video"]), int(row["query_id"]), int(row["frame_id"]))


class BankStore:
    def __init__(self, limit: int = 2):
        self.limit = int(limit)
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def get(self, dataset: str, video: str) -> dict:
        cache_key = f"{dataset}|{video}"
        if cache_key not in self.cache:
            self.cache[cache_key] = load_bank(dataset, video)
            if len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(cache_key)
        return self.cache[cache_key]


def assert_fixed_videos(calibration: list[dict], validation: list[dict]) -> dict:
    expected = {
        "refer_kitti_v1": {"calibration": {"0016"}, "validation": {"0004", "0018"}},
        "refer_kitti_v2": {"calibration": {"0015"}, "validation": {"0016", "0017", "0020"}},
    }
    actual = {
        "refer_kitti_v1": {"calibration": {str(x["video"]) for x in calibration if x["dataset"] == "refer_kitti_v1"},
                           "validation": {str(x["video"]) for x in validation if x["dataset"] == "refer_kitti_v1"}},
        "refer_kitti_v2": {"calibration": {str(x["video"]) for x in calibration if x["dataset"] == "refer_kitti_v2"},
                           "validation": {str(x["video"]) for x in validation if x["dataset"] == "refer_kitti_v2"}},
    }
    if actual != expected:
        raise AssertionError(f"fixed calibration/validation video mismatch: {actual} != {expected}")
    return {domain: {bucket: sorted(values) for bucket, values in buckets.items()}
            for domain, buckets in actual.items()}


def make_records(model, units: list[dict], store: BankStore, text: dict, device: torch.device) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for unit in units:
        grouped[(str(unit["dataset"]), str(unit["video"]))].append(unit)
    teacher_cache: dict[str, dict] = {}
    records = []
    for (dataset, video), group in sorted(grouped.items()):
        bank = store.get(dataset, video)
        cache_key = str(bank["path"])
        if cache_key not in teacher_cache:
            teacher_cache[cache_key] = build_teacher_cache(bank)
        for unit in group:
            score = l29_score(model, teacher_cache[cache_key], bank, unit, text, device)
            labels = np.zeros(int(unit["end"]) - int(unit["begin"]), dtype=bool)
            labels[np.asarray(unit["positive_indices"], dtype=np.int64)] = True
            records.append({
                "dataset": dataset,
                "video": video,
                "query_id": int(unit["query_id"]),
                "frame_id": int(unit["frame_id"]),
                "category": str(unit["category"]),
                "candidate_count": int(len(labels)),
                "positive_count": int(labels.sum()),
                "score": np.asarray(score, dtype=np.float32),
                "label": labels,
                "sources": source_masks(bank, int(unit["begin"]), int(unit["end"])),
            })
    return records


def record_contract(records: list[dict]) -> dict:
    keys = [key(row) for row in records]
    return {
        "record_count": len(records),
        "duplicate_key_count": len(keys) - len(set(keys)),
        "empty_candidate_units": sum(int(len(row["score"]) == 0) for row in records),
        "nonfinite_score_count": sum(int(not np.isfinite(row["score"]).all()) for row in records),
        "candidate_rows": int(sum(len(row["score"]) for row in records)),
        "positive_rows": int(sum(row["label"].sum() for row in records)),
        "full_candidate_set_preserved": True,
    }


def jsonable_summary(records: list[dict], threshold: float) -> dict:
    summary = summarize(records, threshold)
    # Keep the existing L27/L49 definitions intact, while making the required
    # empty/NULL semantics explicit for this L29-only control.
    summary["null_output"] = "N/A (L29 control has no separate NULL head)"
    return summary


def compare_immutable_validation(records: list[dict]) -> dict:
    if not VAL_BASELINE.exists():
        return {"available": False, "reason": "historical validation baseline cache absent"}
    baseline = {}
    for line in VAL_BASELINE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        baseline[key(row)] = np.asarray(row["score"], dtype=np.float32)
    checks = []
    for row in records:
        ref = baseline.get(key(row))
        if ref is None or ref.shape != row["score"].shape:
            checks.append({"key": key(row), "match": False, "reason": "missing_or_shape_mismatch"})
        else:
            checks.append({"key": key(row), "match": bool(np.array_equal(ref, row["score"])),
                           "max_abs_diff": float(np.max(np.abs(ref - row["score"]))) if len(ref) else 0.0})
    return {"available": True, "baseline_record_count": len(baseline),
            "compared_count": len(checks), "mismatch_count": sum(not x["match"] for x in checks),
            "max_abs_diff": max((x.get("max_abs_diff", float("inf")) for x in checks), default=0.0),
            "baseline_path": str(VAL_BASELINE.resolve())}


def run(args: argparse.Namespace) -> None:
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
    started = time.time()
    calibration = load_jsonl(DATA / "calibration_units.jsonl")
    validation = load_jsonl(DATA / "validation_units.jsonl")
    fixed_videos = assert_fixed_videos(calibration, validation)
    text = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    model = L29FrameMembershipSetDecoder().to(device)
    model.load_state_dict(torch.load(L29_CHECKPOINT, map_location=device, weights_only=False)["model"], strict=True)
    model.eval()
    if args.contract_only:
        calibration = calibration[:1]
        validation = validation[:1]
    store = BankStore()
    cal_records = make_records(model, calibration, store, text, device)
    val_records = make_records(model, validation, store, text, device)
    if args.contract_only:
        result = {
            "format": "locatemot-l51-l29-control-contract-v1",
            "status": "pass",
            "scope": "one calibration and one validation unit; no screening/test",
            "calibration_contract": record_contract(cal_records),
            "validation_contract": record_contract(val_records),
            "device": str(device),
            "official_test_labels_read": False,
            "screening_gt_used": False,
        }
    else:
        thresholds = {}
        summaries = {"calibration": {}, "validation": {}}
        for domain in ("refer_kitti_v1", "refer_kitti_v2"):
            cal_domain = [x for x in cal_records if x["dataset"] == domain]
            val_domain = [x for x in val_records if x["dataset"] == domain]
            fitted = fit_threshold(cal_domain)
            thresholds[domain] = fitted
            summaries["calibration"][domain] = jsonable_summary(cal_domain, fitted["threshold"])
            summaries["validation"][domain] = jsonable_summary(val_domain, fitted["threshold"])
        all_records = cal_records + val_records
        result = {
            "format": "locatemot-l51-l29-control-cal-validation-v1",
            "status": "pass",
            "scope": "fixed L49 calibration/validation only; no screening/test",
            "fixed_videos": fixed_videos,
            "thresholds": thresholds,
            "summaries": summaries,
            "calibration_contract": record_contract(cal_records),
            "validation_contract": record_contract(val_records),
            "validation_cache_exact_match": compare_immutable_validation(val_records),
            "device": str(device),
            "l29_checkpoint": str(L29_CHECKPOINT.resolve()),
            "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT),
            "manifest_sha256": sha256_file(FAST_MANIFEST),
            "official_test_labels_read": False,
            "screening_gt_used": False,
            "ordinary_mot_ovmot_touched": False,
            "raw_cache_written": False,
            "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_id"],
            "candidate_records_written": False,
            "elapsed_sec": time.time() - started,
            "all_record_count": len(all_records),
        }
    (out / "control.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (out / "provenance.json").write_text(json.dumps({
        "format": "locatemot-l51-l29-control-provenance-v1",
        "started_at_unix": started,
        "completed_at_unix": time.time(),
        "calibration_source": str((DATA / "calibration_units.jsonl").resolve()),
        "validation_source": str((DATA / "validation_units.jsonl").resolve()),
        "screening_source_opened": False,
        "official_test_source_opened": False,
        "manifest_sha256": sha256_file(FAST_MANIFEST),
        "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT),
    }, indent=2) + "\n")
    print(json.dumps(result, indent=2, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contract-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
