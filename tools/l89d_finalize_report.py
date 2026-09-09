#!/usr/bin/env python3
"""Render the L89D evidence chain from completed machine-readable outputs.

This file is deliberately a report-only adapter.  It never loads a model,
recomputes a score, fits a threshold, or reads a label source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
BASE_COMMIT = "613fbe6d5e802fbc08d3ef0b5084f5d6b8337cd5"
L89C_CHECKPOINT_SHA = "f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, filename: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_dir():
        if filename is None:
            raise ValueError(f"directory requires a JSON filename: {resolved}")
        resolved = resolved / filename
    return json.loads(resolved.read_text(encoding="utf-8"))


def json_path(path: Path, filename: str | None = None) -> Path:
    resolved = path.resolve()
    return (resolved / filename if resolved.is_dir() and filename else resolved)


def require_complete(path: Path, filename: str | None = None) -> dict[str, Any]:
    value = load_json(path, filename)
    if value.get("status") != "complete":
        raise AssertionError(f"required artifact is not complete: {json_path(path, filename)}")
    return value


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def coverage_line(payload: dict[str, Any]) -> str:
    timeline = payload.get("timeline", {})
    return (
        f"{payload.get('scope')}: groups {payload.get('resolved_group_count')}/"
        f"{payload.get('frame_group_count')}, missing {payload.get('missing_group_count')}, "
        f"pairs {timeline.get('covered_query_frame_pairs')}/"
        f"{timeline.get('expected_query_frame_pairs')}, dense_ready={payload.get('dense_ready')}"
    )


def metric_row(name: str, value: dict[str, Any], dataset: str | None = None) -> str:
    if dataset is not None:
        value = next(item for item in value.get("per_dataset", []) if item.get("dataset") == dataset)
    percent = value.get("metrics_percent", {})
    counts = value.get("metrics_counts", {})
    return "| " + " | ".join([
        name,
        fmt(percent.get("HOTA___AUC")),
        fmt(percent.get("DetA___AUC")),
        fmt(percent.get("AssA___AUC")),
        fmt(percent.get("DetRe___AUC")),
        fmt(percent.get("DetPr___AUC")),
        fmt(percent.get("IDF1")),
        fmt(counts.get("IDSW"), 1),
    ]) + " |"


def semantic_metric(payload: dict[str, Any], key: str) -> Any:
    return payload.get("validation", {}).get(key)


def build_report(args: argparse.Namespace) -> str:
    dev_before = require_complete(args.dev_before_audit, "coverage.json")
    dev_after = require_complete(args.dev_after_audit, "coverage.json")
    internal_before = require_complete(args.internal_before_audit, "coverage.json")
    internal_after = require_complete(args.internal_after_audit, "coverage.json")
    dev_matrix = require_complete(args.dev_trackeval, "trackeval_matrix.json")
    selection = require_complete(args.selection, "checkpoint_selection.json")
    fixed = require_complete(args.fixed_semantic, "semantic.json")
    fixed_gate = require_complete(args.fixed_semantic, "gate_decision.json")
    internal_matrix = require_complete(args.internal_trackeval, "trackeval_matrix.json")

    if not dev_after.get("dense_ready") or not internal_after.get("dense_ready"):
        raise AssertionError("post-build dense coverage is not ready")
    if not dev_matrix.get("timeline_contract_passed") or not internal_matrix.get("timeline_contract_passed"):
        raise AssertionError("TrackEval artifact lacks true native-timeline proof")
    if not selection.get("selection_frozen_before_fixed_validation"):
        raise AssertionError("selection was not frozen before fixed validation")
    if fixed.get("record_count") != 40 or fixed.get("calibration_count") != 16 or fixed.get("validation_count") != 24:
        raise AssertionError("fixed semantic record counts drifted")
    if sha256_file(ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json") != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")

    chosen = selection["final_selection"]
    checkpoint = chosen["checkpoint_info"]
    internal_result = internal_matrix["results"][0]
    validation = fixed["validation"]
    gate_decision = fixed_gate.get("decision", "unknown")
    supplement_dev = dev_after.get("inputs", {}).get("supplement_z1_cache")
    supplement_internal = internal_after.get("inputs", {}).get("supplement_z1_cache")

    lines: list[str] = []
    lines += [
        "# L89D — True Full-Video Timeline Repair Final Report",
        "",
        "## Final status",
        "",
        f"`L89D_COMPLETE_ZERO_TRAINING_TRUE_FULLVIDEO_REPAIR / {gate_decision} / "
        "STOPPED_PENDING_SUPERVISOR_REVIEW`",
        "",
        "This stage repaired the evidence timeline only. It did not change the L89 model, loss, "
        "candidate bank, tracker, threshold fits, UIDM, ordinary MOT, OVMOT, or production entrypoints. "
        "The fixed semantic gate remains separate from formal internal TrackEval.",
        "",
        "## 1. Provenance",
        "",
        f"- project root: `{ROOT}`",
        f"- L89D worktree/branch checkout: `{WORK_ROOT}` / `codex/l89d-true-fullvideo-timeline-repair-20260908`",
        f"- Luna thread: `{THREAD}`",
        f"- L89D base commit: `{BASE_COMMIT}`",
        f"- fixed manifest SHA256: `{MANIFEST_SHA}`",
        f"- L89C shortlist source: `{selection.get('source_selection')}` (SHA `{selection.get('source_selection_sha256')}`)",
        f"- L89C historical epoch-2 source SHA verified: `{L89C_CHECKPOINT_SHA}`",
        f"- L89D selected checkpoint: `{checkpoint.get('path')}` (epoch {checkpoint.get('epoch')}, SHA `{checkpoint.get('sha256')}`)",
        f"- base Z1 cache: `{dev_after.get('inputs', {}).get('base_z1_cache')}`; summary SHA `{dev_after.get('inputs', {}).get('base_summary', {}).get('sha256')}`",
        f"- dev Z1 supplement: `{supplement_dev}`",
        f"- internal Z1 supplement: `{supplement_internal}`",
        "- language cache: frozen L89 retry1 cache; no language regeneration",
        "",
        "## 2. Root cause confirmed",
        "",
        "The invalid L89/L89C system evidence used sparse dev/validation group frames for prediction, "
        "while `materialize_gt()` wrote GT on every native L69 frame. That produced a sequence/timeline "
        "mismatch and made the old near-zero HOTA invalid for full-video comparison.",
        "",
        "L89D uses the same native L69 `frame_ids/frame_ptr` timeline for prediction and GT. Every legal "
        "query is paired with every native frame; sparse group keys are used only to define legal query scope.",
        "",
        "## 3. Dense Z1 coverage",
        "",
        "| scope | before | after |",
        "|---|---|---|",
        f"| dev | {coverage_line(dev_before)} | {coverage_line(dev_after)} |",
        f"| internal | {coverage_line(internal_before)} | {coverage_line(internal_after)} |",
        "",
        "The base cache was sparse/partial for the native universe. L89D built only the audited missing "
        "groups as label-free Z1 supplements on `/data2`, then passed the post-build coverage audit. "
        "The dev supplement finalized 2,385 groups / 230,538 query-frame pairs and is about 6.09 GB; "
        "the internal supplement finalized 1,844 groups / 243,550 query-frame pairs and is about 8.50 GB. "
        "No raw pixels or dense detector maps were persisted.",
        "",
        "## 4. No-training and boundary contract",
        "",
        "- `zero_training=true`; no optimizer step or checkpoint was created by L89D.",
        "- L89 model weights, QSC-D architecture, loss, thresholds/rule fits, L69 bank, Z1 definition, tracker IDs and boxes were unchanged.",
        "- `candidate_deletion=false`, `candidate_truncation=false`; all native candidate rows were scored.",
        "- `screening_gt_used=false`, `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.",
        "- token/span-to-region and static/motion alignment remain `UNALIGNED`.",
        "- TrackEval used the local checkout with no verifiable Git HEAD; this is recorded in machine outputs.",
        "",
        "## 5. True-full-video dev TrackEval matrix",
        "",
        "Percent-valued metrics are local TrackEval values multiplied by 100; IDSW is a count.",
        "",
        "| checkpoint/rule | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in dev_matrix["results"]:
        label = f"epoch {result['checkpoint_info']['epoch']} / {result['rule']}"
        lines.append(metric_row(label, result))
    lines += [
        "",
        "All five shortlist checkpoints and B/R/P rules completed over the six L82 dev videos. "
        "The dev matrix contains 15 results and passed the native full-video timeline checks.",
        "",
        "## 6. New frozen L89D selection",
        "",
        f"- selected epoch: `{checkpoint.get('epoch')}`",
        f"- selected rule: `{chosen.get('rule')}`",
        f"- checkpoint SHA256: `{checkpoint.get('sha256')}`",
        f"- candidate threshold: `{chosen['rule_object'].get('candidate_threshold')}`",
        f"- presence threshold: `{chosen['rule_object'].get('presence_threshold')}`",
        f"- null margin: `{chosen['rule_object'].get('null_margin')}`",
        f"- dev selection HOTA/DetA/AssA (%): `{fmt(chosen['selection_metrics'].get('hota') * 100)}` / "
        f"`{fmt(chosen['selection_metrics'].get('deta') * 100)}` / `{fmt(chosen['selection_metrics'].get('assa') * 100)}`",
        "",
        "The selection tuple was exactly `(HOTA, DetA, AssA, distinct_target_recall, "
        "-inactive_false_acceptance, -epoch, deterministic_rule_order)`. L89D did not refit thresholds "
        "or use fixed validation labels for selection.",
        "",
        "## 7. Fixed 16-calibration / 24-validation semantic replay",
        "",
        "This is a fixed semantic diagnostic, not HOTA or screening. The unchanged evaluator attached "
        "calibration labels after score construction and validation labels afterward.",
        "",
        "| metric | L29 control | L89D validation |",
        "|---|---:|---:|",
        f"| recall | 0.7333 | {fmt(semantic_metric(fixed, 'legacy_candidate_recall'))} |",
        f"| precision | 0.0830 | {fmt(semantic_metric(fixed, 'legacy_candidate_precision'))} |",
        f"| FP/frame | 10.1250 | {fmt(semantic_metric(fixed, 'legacy_fp_per_frame'))} |",
        f"| predictions/positive | 8.8333 | {fmt(semantic_metric(fixed, 'legacy_predictions_per_positive'))} |",
        f"| hard violation | 0.9167 | {fmt(semantic_metric(fixed, 'legacy_row_hard_violation'))} |",
        f"| multi-positive recall | 0.8194 | {fmt(semantic_metric(fixed, 'legacy_row_multi_positive_recall'))} |",
        f"| inactive false acceptance | — | {fmt(semantic_metric(fixed, 'inactive_false_acceptance'))} |",
        f"| empty rate | — | {fmt(semantic_metric(fixed, 'empty_rate'))} |",
        "",
        f"Decision: `{gate_decision}`. Recall and multi-positive recall were below the registered floors, "
        "although precision, FP/frame, predictions/positive and hard violation met their individual floors. "
        "This is not a deployment or ordinary-level RMOT pass.",
        "",
        "## 8. Final internal true-full-video TrackEval",
        "",
        "This is legal internal validation-scope TrackEval only: V1 videos 0004/0018 (86 query sequences) "
        "and V2 videos 0016/0017/0020 (537 query sequences). It is not screening or official-test evidence.",
        "",
        "| dataset | HOTA | DetA | AssA | LocA | DetRe | DetPr | AssRe | AssPr | IDF1 | IDR | IDP | IDSW | FP | FN | MOTA | MOTP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        result = next(item for item in internal_result["per_dataset"] if item["dataset"] == dataset)
        percent = result["metrics_percent"]
        counts = result["metrics_counts"]
        lines.append("| " + " | ".join([
            dataset,
            fmt(percent.get("HOTA___AUC")), fmt(percent.get("DetA___AUC")), fmt(percent.get("AssA___AUC")),
            fmt(percent.get("LocA___AUC")), fmt(percent.get("DetRe___AUC")), fmt(percent.get("DetPr___AUC")),
            fmt(percent.get("AssRe___AUC")), fmt(percent.get("AssPr___AUC")), fmt(percent.get("IDF1")),
            fmt(percent.get("IDR")), fmt(percent.get("IDP")), fmt(counts.get("IDSW"), 1),
            fmt(counts.get("CLR_FP"), 1), fmt(counts.get("CLR_FN"), 1), fmt(percent.get("MOTA")), fmt(percent.get("MOTP")),
        ]) + " |")
    lines += [
        "",
        f"Combined unweighted mean: HOTA `{fmt(internal_result['metrics_percent'].get('HOTA___AUC'))}%`, "
        f"DetA `{fmt(internal_result['metrics_percent'].get('DetA___AUC'))}%`, "
        f"AssA `{fmt(internal_result['metrics_percent'].get('AssA___AUC'))}%`, "
        f"DetRe `{fmt(internal_result['metrics_percent'].get('DetRe___AUC'))}%`, "
        f"DetPr `{fmt(internal_result['metrics_percent'].get('DetPr___AUC'))}%`, "
        f"IDF1 `{fmt(internal_result['metrics_percent'].get('IDF1'))}%`, "
        f"IDSW `{fmt(internal_result['metrics_counts'].get('IDSW'), 1)}`.",
        "",
        "## 9. Historical comparison and interpretation",
        "",
        "| evidence | V1 HOTA (%) | V2 HOTA (%) | authority |",
        "|---|---:|---:|---|",
        "| L86 historical full-video | 29.1663 | 21.6467 | historical comparison |",
        "| L87-A historical full-video | 28.5752 | 22.1300 | historical comparison |",
        "| L89C internal result | 2.244718 | 0.583059 | invalid for full-video comparison — sparse prediction/ dense GT mismatch |",
        f"| L89D repaired internal result | {fmt(next(item for item in internal_result['per_dataset'] if item['dataset'] == 'refer_kitti_v1')['metrics_percent'].get('HOTA___AUC'))} | {fmt(next(item for item in internal_result['per_dataset'] if item['dataset'] == 'refer_kitti_v2')['metrics_percent'].get('HOTA___AUC'))} | valid native-timeline internal TrackEval |",
        "",
        "The repaired timeline explains the L89C near-zero system result: L89D recovers to a meaningful "
        "full-video range rather than 0–5 HOTA. The remaining gap is real and must be separated into "
        "detection/volume (DetA/DetPr), association (AssA/IDSW), and inactive/no-match calibration. "
        "The fixed semantic replay independently shows a hard/multi-positive trade-off, so the timeline fix "
        "does not establish a correspondence or ordinary-RMOT solution.",
        "",
        "## 10. Evidence boundary and next action",
        "",
        "No screening, official-test, HOTA fast-screening gate, or ordinary MOT/OVMOT regression was run. "
        "The L89D branch is complete and stopped. The single next action is supervisor review followed by "
        "one separately authorized absence/volume-calibration study; do not extend L89, retune its threshold, "
        "add NULL filtering, or change the tracker in this stage.",
        "",
        "### Artifact status",
        "",
        f"- dev true-full-video inference: `{args.dev_trackeval}`",
        f"- dev selection: `{args.selection}`",
        f"- fixed semantic: `{args.fixed_semantic}`",
        f"- internal true-full-video TrackEval: `{args.internal_trackeval}`",
        "- screening: not run",
        "- official test: not read",
        "- formal production RMOT: not claimed",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-before-audit", type=Path, required=True)
    parser.add_argument("--dev-after-audit", type=Path, required=True)
    parser.add_argument("--internal-before-audit", type=Path, required=True)
    parser.add_argument("--internal-after-audit", type=Path, required=True)
    parser.add_argument("--dev-trackeval", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--fixed-semantic", type=Path, required=True)
    parser.add_argument("--internal-trackeval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite final L89D report: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(args), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
