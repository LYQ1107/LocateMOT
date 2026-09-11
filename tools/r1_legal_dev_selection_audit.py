#!/usr/bin/env python3
"""Audit the frozen L89E legal-dev choice before R1 training.

This is read-only.  It does not rerun TrackEval and deliberately does not
invent an EXACT_INDEX_DEDUP comparison: the frozen full-video artifacts do
not contain the candidate score arrays needed to reproduce that variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from tools.r1_common import MANIFEST_SHA, SEED, THREAD, check_manifest, file_meta, write_json  # noqa: E402

ANCHOR = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch004.pt")
SELECTION = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E/outputs/l89e/dev/selection_attempt1/checkpoint_selection.json")
MATRIX = Path("/data2/usr_for_deadline/locatemot_l89e/dev_trackeval_matrix_attempt1/trackeval_matrix.json")
FULL_SUMMARY = Path("/data2/usr_for_deadline/locatemot_l89e/dev_true_fullvideo_attempt1/candidate_epoch004_shortlist04/summary.json")
DEV_SCORES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/eval/dev_scores_joint40/score_records.jsonl")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _epoch(record: dict[str, Any]) -> int:
    return int((record.get("checkpoint") or {}).get("epoch", -1))


def _metric(result: dict[str, Any], name: str) -> Any:
    raw = result.get("metrics_raw") or {}
    percent = result.get("metrics_percent") or {}
    aliases = {
        "detpr": ("DetPr___AUC",), "detre": ("DetRe___AUC",),
        "assa": ("AssA___AUC",), "hota": ("HOTA___AUC",),
    }
    for key in aliases.get(name, ()):
        if key in raw:
            return raw[key]
        if key in percent:
            return float(percent[key]) / 100.0
    return None


def run(args: argparse.Namespace) -> int:
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 audit output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([str(sys.executable), *sys.argv])
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R1 audit cwd: {Path.cwd()}")
        manifest_sha = check_manifest()
        selection = load_json(SELECTION)
        matrix = load_json(MATRIX)
        full_summary = load_json(FULL_SUMMARY)
        if selection.get("status") != "complete" or matrix.get("status") != "complete":
            raise AssertionError("frozen legal-dev selection or TrackEval matrix is incomplete")
        selected = selection.get("final_selection") or {}
        if str(selected.get("rule")) != "B":
            raise AssertionError(f"registered R1 anchor rule drift: {selected.get('rule')}")
        selected_info = selected.get("checkpoint_info") or {}
        if Path(str(selected_info.get("path"))).resolve() != ANCHOR.resolve():
            raise AssertionError("registered R1 anchor path drift")
        if str(selected_info.get("sha256")) != sha256_file(ANCHOR):
            raise AssertionError("registered R1 anchor SHA drift")
        rule_object = selected.get("rule_object") or {}
        if float(rule_object.get("candidate_threshold")) != 1.0 or \
           float(rule_object.get("presence_threshold")) != 0.5 or \
           float(rule_object.get("null_margin")) != 0.0:
            raise AssertionError("R1 requires the frozen Rule-B thresholds")

        matrix_results = []
        for result in matrix.get("results", []):
            info = result.get("checkpoint_info") or {}
            if int(info.get("epoch", -1)) == 4 and str(result.get("rule")) == "B":
                matrix_results.append(result)
        if len(matrix_results) != 1:
            raise AssertionError(f"expected one epoch-004 Rule-B matrix item, got {len(matrix_results)}")
        matrix_result = matrix_results[0]
        per_dataset = {}
        for item in matrix_result.get("per_dataset", []):
            per_dataset[str(item["dataset"])] = {
                "hota": _metric(item, "hota"), "detpr": _metric(item, "detpr"),
                "detre": _metric(item, "detre"), "assa": _metric(item, "assa"),
                "id_switches": (item.get("metrics_counts") or {}).get("IDSW"),
                "sequence_count": item.get("sequence_count"),
            }

        # The original legal-dev score file contains labels, but is strictly
        # scoped to fit/dev query-frame rows.  Only the frozen epoch-004 rows
        # are read here, and no candidate score is changed or selected.
        epoch_rows = 0
        target_present = 0
        presence_misses = 0
        by_dataset: dict[str, dict[str, int]] = {}
        unique_units: set[str] = set()
        candidate_counts: list[int] = []
        with DEV_SCORES.resolve().open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if _epoch(record) != 4:
                    continue
                epoch_rows += 1
                key = str(record["unit_key"])
                if key in unique_units:
                    raise AssertionError(f"duplicate frozen legal-dev unit: {key}")
                unique_units.add(key)
                dataset = str(record["dataset"])
                bucket = by_dataset.setdefault(dataset, {"rows": 0, "target_present": 0, "presence_miss": 0})
                bucket["rows"] += 1
                candidate_counts.append(int(record["candidate_count"]))
                present = bool(record.get("target_present"))
                target_present += int(present)
                bucket["target_present"] += int(present)
                miss = present and float(record["presence_logit"]) < float(rule_object["presence_threshold"])
                presence_misses += int(miss)
                bucket["presence_miss"] += int(miss)
        if epoch_rows != 498 or target_present != 400 or presence_misses != 3:
            raise AssertionError(
                f"frozen presence audit mismatch rows={epoch_rows}, target_present={target_present}, misses={presence_misses}"
            )

        # No valid full-video score-array artifact for an exact candidate-index
        # dedup replay exists in the frozen inputs.  The existing "dedup"
        # references concern checkpoint shortlist paths, not candidate rows.
        exact_variant = {
            "status": "unavailable",
            "candidate_score_arrays": False,
            "full_video_trackeval_matrix": False,
            "candidate_index_dedup_replay_found": False,
            "reason": "frozen full-video prediction audits retain selected masks/keys but not score arrays; existing dedup references are checkpoint-shortlist dedup, not candidate suppression",
            "selection_consequence": "freeze_RAW_conservative_fallback",
        }
        payload = {
            "format": "locatemot-r1-legal-dev-selection-audit-v1",
            "status": "complete", "stage": "pre-formal-R1 legal-dev and presence audit",
            "command": command, "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED,
            "manifest_sha256": manifest_sha, "anchor": {"path": str(ANCHOR), "sha256": sha256_file(ANCHOR)},
            "registered_selection": {
                "path": str(SELECTION), "sha256": sha256_file(SELECTION), "checkpoint_epoch": 4,
                "rule": "B", "rule_object": rule_object,
                "phase_policy": selected.get("phase_policy"),
            },
            "raw_legal_dev_trackeval": {
                "matrix_path": str(MATRIX), "matrix_sha256": sha256_file(MATRIX),
                "overall": {key: matrix_result.get(key) for key in ("hota", "deta", "assa", "distinct_target_recall", "inactive_false_acceptance")},
                "per_dataset": per_dataset,
                "trackeval_git_head": None,
                "trackeval_git_head_note": "local checkout has no verifiable HEAD",
                "true_fullvideo": bool(matrix.get("native_timeline") and matrix.get("timeline_contract_passed")),
            },
            "full_video_summary": {
                "path": str(FULL_SUMMARY), "sha256": sha256_file(FULL_SUMMARY),
                "query_frame_pairs": full_summary.get("actual_query_frame_pairs"),
                "candidate_rows": full_summary.get("candidate_rows_scored"),
                "all_candidate_rows_scored": full_summary.get("all_candidate_rows_scored"),
                "candidate_deletion": full_summary.get("candidate_deletion"),
                "candidate_truncation": full_summary.get("candidate_truncation"),
            },
            "presence_only_audit": {
                "source": str(DEV_SCORES), "source_sha256": sha256_file(DEV_SCORES),
                "epoch": 4, "rows": epoch_rows, "unique_units": len(unique_units),
                "target_present_units": target_present, "presence_threshold": float(rule_object["presence_threshold"]),
                "presence_only_misses": presence_misses,
                "miss_fraction_over_target_present": presence_misses / target_present,
                "by_dataset": by_dataset, "candidate_count_min": min(candidate_counts),
                "candidate_count_max": max(candidate_counts),
            },
            "exact_index_dedup_comparison": exact_variant,
            "r1_presence_residual_decision": {
                "registered_activation_threshold": 0.05,
                "observed_miss_fraction": presence_misses / target_present,
                "enabled": False,
                "reason": "3/400=.0075 < .05; frozen anchor presence is retained without a trainable residual",
            },
            "scope": {
                "legal_dev_labels_read": True, "screening_gt_used": False,
                "official_test_labels_read": False, "fixed_calibration_read": False,
                "fixed_validation_read": False, "training_run": False,
                "new_checkpoint_created": False, "candidate_rows_modified": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": "reused_frozen_artifact_only",
            },
            "failure_root_cause": None,
            "next_action": "run bounded R1 wiring smoke with RAW policy and presence_residual=false, then formal six-epoch fit if it passes",
            "inputs": {key: file_meta(value) for key, value in {
                "selection": SELECTION, "matrix": MATRIX, "full_video_summary": FULL_SUMMARY,
                "dev_scores": DEV_SCORES, "anchor": ANCHOR,
            }.items()},
        }
        write_json(out / "contract.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-r1-legal-dev-provenance-v1"})
        write_json(out / "status.json", {
            "format": payload["format"], "status": "complete", "command": command,
            "output": str(out), "failure_root_cause": None, "next_action": payload["next_action"],
            "selection_policy": "RAW_conservative_fallback_exact_variant_unavailable",
            "presence_residual_enabled": False, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        })
        print(json.dumps({"status": "complete", "epoch4_rows": epoch_rows,
                          "target_present": target_present, "presence_misses": presence_misses,
                          "presence_miss_fraction": presence_misses / target_present,
                          "selection_policy": "RAW"}, sort_keys=True))
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text(
            "# R1 legal-dev selection audit — INCOMPLETE\n\n```text\n" +
            __import__("traceback").format_exc() +
            "```\n", encoding="utf-8")
        write_json(out / "status.json", {
            "format": "locatemot-r1-legal-dev-selection-audit-v1", "status": "incomplete",
            "command": command, "cwd": str(Path.cwd().resolve()), "thread": THREAD,
            "failure_root_cause": "see INCOMPLETE.md", "next_action": "repair only the first audit contract error",
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
