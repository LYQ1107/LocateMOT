#!/usr/bin/env python3
"""Finalize the frozen L49 fit/validation/test error matrix.

This command only aggregates already written prediction/label records and
TrackEval logs.  It cannot alter the selected checkpoint or calibration.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.eval_l49_fit_error_matrix import (  # noqa: E402
    compact_summary,
    load_score_records,
    matrix_buckets,
    rank_flip_decomposition,
    record_key,
    sha256_file,
)

PRETEST = ROOT / "outputs/l49/val/error_matrix_pretest.json"
TEST_ROOT = ROOT / "outputs/l49/test/official_frozen_step250"


def load_test_with_teacher(path: Path):
    records = load_score_records(path, "test", checkpoint_step=250)
    baselines = []
    for record in records:
        if "teacher_score" not in record:
            raise KeyError(f"test record lacks teacher_score: {path}")
        base = dict(record)
        teacher_score = np.asarray(record["teacher_score"], dtype=np.float32)
        base["score"] = teacher_score
        base["semantic_score"] = teacher_score.copy()
        base["identity_score"] = np.zeros_like(teacher_score)
        base["continuation_score"] = np.zeros_like(teacher_score)
        baselines.append(base)
    return records, baselines


def parse_trackeval(path: Path):
    lines = path.read_text().splitlines()
    specs = {
        "HOTA": ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr",
                 "LocA", "RHOTA", "HOTA(0)", "LocA(0)", "HOTALocA(0)"],
        "CLEAR": ["MOTA", "MOTP", "MODA", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR",
                  "sMOTA", "CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag"],
        "Identity": ["IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP"],
    }
    output = {}
    for section, names in specs.items():
        marker = next((i for i, line in enumerate(lines)
                       if line.startswith(section + ":")), None)
        if marker is None:
            output[section] = {"status": "missing"}
            continue
        combined = next((line for line in lines[marker + 1:]
                         if line.strip().startswith("COMBINED")), None)
        if combined is None:
            output[section] = {"status": "missing_combined"}
            continue
        values = [float(value) for value in re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", combined)]
        output[section] = {"status": "parsed", **{
            name: values[index] if index < len(values) else None
            for index, name in enumerate(names)
        }}
    output["log"] = str(path.resolve())
    output["log_sha256"] = sha256_file(path)
    return output


def test_detail(records, baselines, thresholds):
    by_key = {record_key(record): record for record in baselines}
    return {
        "checkpoint_step": 250,
        "per_domain": {
            domain: compact_summary([x for x in records if x["dataset"] == domain],
                                     thresholds[domain])
            for domain in ("refer_kitti_v1", "refer_kitti_v2")
        },
        "buckets": matrix_buckets(records, by_key, thresholds),
        "rank_flip_decomposition": rank_flip_decomposition(records, by_key),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretest", default=str(PRETEST))
    parser.add_argument("--out", default=str(ROOT / "outputs/l49/val/error_matrix.json"))
    args = parser.parse_args()
    started = time.time()
    pretest_path = Path(args.pretest)
    pretest = json.loads(pretest_path.read_text())
    thresholds = {key: float(value) for key, value in
                  pretest["thresholds_inherited_from_calibration"].items()}
    all_test = []
    baseline_all = []
    test_provenance = {}
    for domain, short in (("refer_kitti_v1", "v1"), ("refer_kitti_v2", "v2")):
        score_path = TEST_ROOT / short / "test_scores.jsonl"
        baseline_path = TEST_ROOT / short / "test_baseline_scores.jsonl"
        records, baselines = load_test_with_teacher(score_path)
        all_test.extend(records)
        baseline_all.extend(baselines)
        summary_path = TEST_ROOT / short / "test_run_summary.json"
        test_provenance[domain] = {
            "score_records": str(score_path.resolve()),
            "score_records_sha256": sha256_file(score_path),
            "baseline_records_written_by_test_runner": str(baseline_path.resolve()),
            "baseline_records_sha256": sha256_file(baseline_path),
            "record_count": len(records),
            "test_run_summary": json.loads(summary_path.read_text()),
            "trackeval": parse_trackeval(TEST_ROOT / short / "trackeval_official.log"),
            "labels_read_after_selection": True,
            "labels_used_for_selection": False,
        }
        domain_records = [x for x in records if x["dataset"] == domain]
        domain_baselines = [x for x in baselines if x["dataset"] == domain]
        detailed = test_detail(domain_records, domain_baselines, thresholds)
        test_provenance[domain]["detailed_domain_check"] = detailed["per_domain"][domain]
        test_provenance[domain]["rank_flip_domain_check"] = {
            key: {name: value for name, value in row.items()}
            for key, row in detailed["rank_flip_decomposition"].items()
        }
        del records, baselines, domain_records, domain_baselines

    # The combined test records are retained in memory only for the final
    # split-level matrix.  They are already the frozen score/label cache.
    baseline_by_key = {record_key(record): record for record in baseline_all}
    test_detail_all = {
        "checkpoint_step": 250,
        "per_domain": {
            domain: compact_summary([x for x in all_test if x["dataset"] == domain], thresholds[domain])
            for domain in ("refer_kitti_v1", "refer_kitti_v2")
        },
        "buckets": matrix_buckets(all_test, baseline_by_key, thresholds),
        "rank_flip_decomposition": rank_flip_decomposition(all_test, baseline_by_key),
    }

    output = copy.deepcopy(pretest)
    output["format"] = "locatemot-l49-error-matrix-v2"
    output["matrix_status"] = "fit_validation_test_complete"
    output["completed_at_unix"] = time.time()
    output["official_test_labels_read"] = True
    output["screening_or_test_labels_used_for_selection"] = False
    output["selected_detailed_matrix"]["test"] = test_detail_all
    output["test"] = {
        "status": "complete",
        "official_test_labels_read_after_checkpoint_and_calibration_freeze": True,
        "labels_used_for_model_checkpoint_or_threshold_selection": False,
        "per_domain": test_provenance,
        "trackeval_protocol": "unchanged references/l8/TrackEval_rmot/scripts/run_mot_challenge.py",
    }
    output["provenance"]["test_run_provenance"] = str(
        (TEST_ROOT / "test_run_provenance.json").resolve())
    output["provenance"]["test_run_provenance_sha256"] = sha256_file(
        TEST_ROOT / "test_run_provenance.json")
    output["provenance"]["test_score_files"] = {
        domain: str((TEST_ROOT / short / "test_scores.jsonl").resolve())
        for domain, short in (("refer_kitti_v1", "v1"), ("refer_kitti_v2", "v2"))
    }
    output["provenance"]["official_seqmaps"] = {
        domain: str((TEST_ROOT / short / "seqmap_official.txt").resolve())
        for domain, short in (("refer_kitti_v1", "v1"), ("refer_kitti_v2", "v2"))
    }
    output["provenance"]["official_test_labels_read_after_selection"] = True
    output["provenance"]["test_records"] = len(all_test)
    output["provenance"]["elapsed_sec_finalize"] = time.time() - started
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    md = ROOT / "outputs/l49/val/error_matrix.md"
    md.write_text(
        "# L49 fit/validation/test error matrix\n\n"
        "The matrix uses fit and video-disjoint validation records plus the "
        "single frozen official V1/V2 test pass. Calibration thresholds and "
        "the step-250 checkpoint were frozen before test labels were read.\n\n"
        f"Machine-readable artifact: `{out.resolve()}`\n"
    )
    print(json.dumps({"matrix": str(out.resolve()), "status": output["matrix_status"],
                      "test_records": len(all_test), "elapsed_sec": time.time() - started,
                      "official_test_labels_read_after_freeze": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
