"""Fair fixed-manifest comparison for the Stage L20 structural ablations.

The prediction caches are produced by the L20 fast evaluator for A1--A6.
A0 intentionally reuses the already-complete L19 step-1000 prediction cache
because it has the same frozen manifest.  This report applies one common
calibration-only threshold sweep, one common screening TrackEval invocation,
and one common set of source/NULL summaries to all seven variants.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.eval_l18_carr import (  # noqa: E402
    run_trackeval, trainval_queries, write_trainval_gt,
)


MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EVALUATOR = ROOT / "tools/eval_l20_fast.py"
TRACKING_PROTOCOL = ROOT / "tools/eval_l18_carr.py"
TRACKEVAL_RUN = ROOT / "tools/eval_l13_rmot.py"
DEFAULT_A0_ROOT = ROOT / "outputs/l19/eval/fast_round1_step1000"
DEFAULT_OUTPUT = ROOT / "outputs/l20/eval/fast_structural_comparison"

L20_FIELDS = {
    "frame", "track_id", "box", "score", "raw_logit", "null_logit",
    "source", "group_id", "group_size", "cross_pool", "current_match",
    "membership", "observation", "gt_iou", "null_target", "state",
}
BASELINE_FIELDS = {
    "frame", "track_id", "box", "score", "source", "current_match",
    "gt_iou",
}
SWEEP_QUANTILES = np.linspace(0.01, 0.995, 96)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def load_manifest(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text())
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists() and sidecar.read_text().split()[0] != digest:
        raise ValueError(f"manifest SHA sidecar mismatch: {path}")
    if payload.get("selection_uses_model_scores", True):
        raise ValueError("fast manifest selection may not depend on model scores")
    rows = payload.get("queries", [])
    if len(rows) != 160 or sum(row.get("split") == "calibration" for row in rows) != 64 \
            or sum(row.get("split") == "screening" for row in rows) != 96:
        raise ValueError("unexpected fixed fast manifest cardinality")
    return payload, digest


def json_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_checkpoint_info(path: Path) -> dict:
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg", {})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "model_name": checkpoint.get("model_name"),
        "step": int(checkpoint.get("step", -1)),
        "cfg_sha256": json_sha(cfg),
        "cfg": {
            key: cfg[key] for key in (
                "hidden", "heads", "temporal_points", "hook_points",
                "use_source_adapters", "use_grouping", "use_null",
            ) if key in cfg
        },
    }


def _load_records(label: str, root: Path, expected: dict[int, dict],
                  manifest_sha: str, baseline: bool) -> dict:
    run_path = root / "run_manifest.json"
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    run = json.loads(run_path.read_text())
    if run.get("complete") is not True:
        raise ValueError(f"incomplete fast run: {run_path}")
    if run.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"manifest SHA mismatch: {run_path}")
    completed = run.get("query_indices") if baseline else \
        run.get("completed_query_indices")
    if completed is None:
        completed = []
    completed = sorted(int(value) for value in completed)
    if completed != sorted(expected):
        raise ValueError(
            f"{label} query coverage mismatch: {len(completed)} vs {len(expected)}")
    records = []
    for index in completed:
        result_path = root / "queries" / f"q{index:05d}.json"
        score_path = root / "scores" / f"q{index:05d}.npz"
        marker = root / "complete" / f"q{index:05d}.complete"
        if not result_path.exists() or not score_path.exists() or not marker.exists():
            raise ValueError(f"incomplete query artifacts: {result_path}")
        meta = json.loads(result_path.read_text())
        if (meta.get("complete") is not True or
                int(meta.get("query_index", -1)) != index or
                meta.get("manifest_sha256") != manifest_sha):
            raise ValueError(f"query provenance mismatch: {result_path}")
        if index not in expected or (meta.get("video"), meta.get("expression")) != \
                (expected[index]["video"], expected[index]["expression"]):
            raise ValueError(f"query identity mismatch: {result_path}")
        with np.load(score_path, allow_pickle=False) as loaded:
            data = {key: np.asarray(loaded[key]) for key in loaded.files}
        required = BASELINE_FIELDS if baseline else L20_FIELDS
        if set(data) != required:
            raise ValueError(f"field mismatch in {score_path}: {sorted(data)}")
        if len(data["score"]) != int(meta.get("rows", -1)):
            raise ValueError(f"row count mismatch: {result_path}")
        if not np.isfinite(data["score"]).all() or not np.isfinite(data["box"]).all():
            raise ValueError(f"non-finite prediction data: {score_path}")
        records.append((meta, data))
    checkpoint = Path(run["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    if not checkpoint.exists() or sha256_file(checkpoint) != run.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint provenance mismatch: {run_path}")
    info = load_checkpoint_info(checkpoint)
    if run.get("cfg_sha256") and run.get("cfg_sha256") != info["cfg_sha256"]:
        raise ValueError(f"checkpoint cfg provenance mismatch: {run_path}")
    return {
        "label": label, "root": str(root), "run_manifest": str(run_path),
        "run_manifest_sha256": sha256_file(run_path), "run": run,
        "checkpoint": info, "records": records, "baseline": baseline,
    }


def concatenate(records: list[tuple[dict, dict]], split: str) -> dict[str, np.ndarray]:
    chosen = [data for meta, data in records
              if split == "all" or meta["split"] == split]
    if not chosen:
        return {}
    return {key: np.concatenate([data[key] for data in chosen])
            for key in chosen[0]}


def gt_counts(records: list[tuple[dict, dict]], gt_root: Path) -> dict[int, int]:
    result = {}
    for meta, _data in records:
        index = int(meta["query_index"])
        path = gt_root / str(meta["video"]) / str(meta["expression"]) / "gt.txt"
        result[index] = sum(1 for line in path.read_text().splitlines() if line.strip())
    return result


def frame_summaries(records: list[tuple[dict, dict]], threshold: float,
                    split: str) -> tuple[list[dict], bool]:
    output = []
    has_null = True
    for meta, data in records:
        if split != "all" and meta["split"] != split:
            continue
        frames = np.asarray(data["frame"], np.int64)
        scores = np.asarray(data["score"], np.float64)
        selected = scores >= float(threshold)
        null_values = data.get("null_target")
        if null_values is None:
            has_null = False
        for frame in np.unique(frames):
            indices = np.flatnonzero(frames == frame)
            row = {
                "query_index": int(meta["query_index"]),
                "frame": int(frame),
                "selected": int(selected[indices].sum()),
                "empty": bool(not selected[indices].any()),
            }
            if null_values is not None:
                row["target_null"] = bool(np.asarray(null_values)[indices[0]])
            output.append(row)
    return output, has_null


def threshold_metrics(records: list[tuple[dict, dict]], threshold: float,
                      split: str, gt_by_query: dict[int, int]) -> dict:
    data = concatenate(records, split)
    if not data:
        return {"split": split, "threshold": float(threshold), "candidate_rows": 0}
    score = np.asarray(data["score"], np.float64)
    labels = np.asarray(data["current_match"], bool)
    source = np.asarray(data["source"], np.int64)
    selected = score >= float(threshold)
    tp = int(np.count_nonzero(selected & labels))
    fp = int(np.count_nonzero(selected & ~labels))
    fn = int(np.count_nonzero(~selected & labels))
    result = {
        "split": split, "threshold": float(threshold),
        "candidate_rows": int(len(score)), "selected": int(selected.sum()),
        "candidate_positive_rows": int(labels.sum()), "tp": tp, "fp": fp,
        "fn": fn, "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
    }
    for source_id, name in ((0, "main"), (1, "reserve"), (2, "cross_pool")):
        pool = source == source_id
        pool_selected = pool & selected
        pool_positive = pool & labels
        pool_tp = int(np.count_nonzero(pool_selected & labels))
        result[f"{name}_candidates"] = int(pool.sum())
        result[f"{name}_selected"] = int(pool_selected.sum())
        result[f"{name}_positive"] = int(pool_positive.sum())
        result[f"{name}_acceptance_rate"] = float(
            pool_selected.sum() / max(1, pool.sum()))
        result[f"{name}_selected_recall"] = float(
            pool_tp / max(1, pool_positive.sum()))
        result[f"{name}_selected_precision"] = float(
            pool_tp / max(1, pool_selected.sum()))

    frames, has_null = frame_summaries(records, threshold, split)
    empty = [row["empty"] for row in frames]
    result["frame_count"] = len(frames)
    result["empty_output_rate"] = float(np.mean(empty)) if empty else None
    result["prediction_count_per_frame"] = {
        "mean": float(np.mean([row["selected"] for row in frames])) if frames else None,
        "median": float(np.median([row["selected"] for row in frames])) if frames else None,
    }
    result["gt_detection_count"] = int(sum(
        gt_by_query[int(meta["query_index"])] for meta, _data in records
        if split == "all" or meta["split"] == split))
    result["predictions_per_gt"] = float(
        result["selected"] / max(1, result["gt_detection_count"]))
    if has_null:
        null_frames = [row for row in frames if row.get("target_null")]
        covered_frames = [row for row in frames if not row.get("target_null")]
        null_empty = [row["empty"] for row in null_frames]
        covered_empty = [row["empty"] for row in covered_frames]
        result["target_null_frames"] = len(null_frames)
        result["target_null_frame_rate"] = float(
            len(null_frames) / max(1, len(frames)))
        result["null_trigger_rate"] = float(np.mean(null_empty)) if null_empty else None
        result["covered_empty_output_rate"] = float(
            np.mean(covered_empty)) if covered_empty else None
    else:
        result["target_null_frames"] = None
        result["target_null_frame_rate"] = None
        result["null_trigger_rate"] = None
        result["covered_empty_output_rate"] = None
    return result


def sweep_thresholds(records: list[tuple[dict, dict]],
                     gt_by_query: dict[int, int]) -> tuple[float, dict]:
    calibration = concatenate(records, "calibration")
    scores = np.asarray(calibration["score"], np.float64)
    scores = scores[np.isfinite(scores)]
    candidates = np.unique(np.concatenate((
        np.quantile(scores, SWEEP_QUANTILES),
        np.asarray([scores.min(), scores.max()], np.float64),
    )))
    rows = []
    for threshold in candidates.tolist():
        metrics = threshold_metrics(records, float(threshold), "calibration",
                                    gt_by_query)
        precision, recall = metrics["precision"], metrics["recall"]
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        useful = recall >= 0.45
        rows.append({
            "threshold": float(threshold), "f1": float(f1),
            "useful_recall_ge_0.45": bool(useful),
            "precision": precision, "recall": recall,
            "selected": metrics["selected"], "tp": metrics["tp"],
            "fp": metrics["fp"], "fn": metrics["fn"],
        })
    best = max(rows, key=lambda row: (
        int(row["useful_recall_ge_0.45"]), row["f1"], row["precision"],
        row["recall"], -abs(row["threshold"])))
    top = sorted(rows, key=lambda row: (
        int(row["useful_recall_ge_0.45"]), row["f1"], row["precision"],
        row["recall"], -abs(row["threshold"])), reverse=True)[:12]
    return float(best["threshold"]), {
        "candidate_count": len(rows),
        "quantiles": [float(value) for value in SWEEP_QUANTILES.tolist()],
        "candidate_min": float(scores.min()), "candidate_max": float(scores.max()),
        "selection_rule": "calibration only: recall>=0.45, then F1, precision, recall",
        "selected": best,
        "top_candidates": top,
    }


def prepare_trackeval(out_root: Path, records, threshold: float,
                      gt_root: Path) -> tuple[set[tuple[str, str]], Path]:
    result_root = out_root / "uidm18"
    allowed = set()
    for meta, data in records:
        if meta["split"] != "screening":
            continue
        video, expression = str(meta["video"]), str(meta["expression"])
        allowed.add((video, expression))
        gt = gt_root / video / expression / "gt.txt"
        destination = result_root / video / expression / "gt.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(gt.resolve())
        prediction = result_root / video / expression / "predict.txt"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        keep = np.asarray(data["score"], np.float64) >= float(threshold)
        with prediction.open("w") as handle:
            for index in np.flatnonzero(keep):
                x1, y1, x2, y2 = [float(value) for value in data["box"][index]]
                score = 1.0 / (1.0 + math.exp(-float(np.clip(
                    data["score"][index], -40, 40))))
                handle.write(
                    f"{int(data['frame'][index]) + 1},{int(data['track_id'][index])},"
                    f"{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},"
                    f"{score:.6f},-1,-1,-1\n"
                )
    return allowed, gt_root / "seqmap.txt"


def metric_row(metrics: dict) -> dict:
    return {name: metrics.get(name) for name in (
        "HOTA___AUC", "DetA___AUC", "AssA___AUC", "DetRe___AUC",
        "DetPr___AUC", "IDF1",
    )}


def markdown_report(report: dict) -> str:
    lines = [
        "# LocateMOT Stage L20 fair fast RMOT structural comparison",
        "",
        f"- manifest: `{report['provenance']['manifest']}`",
        f"- manifest SHA256: `{report['provenance']['manifest_sha256']}`",
        "- calibration/screening split: `64/96`; TrackEval uses screening only",
        "- no official V1/V2/Dance, full KITTI, MOT, or OVMOT evaluation",
        "",
        "## Fixed-threshold screening results",
        "",
        "Thresholds are selected from calibration labels only, then frozen for screening.",
        "",
        "| variant | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | pred/GT | main accept | reserve accept | null trigger |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, value in report["variants"].items():
        m = value["trackeval"]["metrics"]
        fixed = value["fixed_threshold_screening"]
        fmt = lambda x: "NA" if x is None else f"{float(x):.3f}"
        lines.append(
            f"| {label} | {fmt(m.get('HOTA___AUC'))} | {fmt(m.get('DetA___AUC'))} | "
            f"{fmt(m.get('AssA___AUC'))} | {fmt(m.get('DetRe___AUC'))} | "
            f"{fmt(m.get('DetPr___AUC'))} | {fmt(m.get('IDF1'))} | "
            f"{fmt(fixed.get('predictions_per_gt'))} | "
            f"{fmt(fixed.get('main_acceptance_rate'))} | "
            f"{fmt(fixed.get('reserve_acceptance_rate'))} | "
            f"{fmt(fixed.get('null_trigger_rate'))} |"
        )
    lines.extend(["", "## Level 1 gate", ""])
    lines.append("| variant | HOTA delta vs A0 | HOTA +5 | DetPr >=20 | DetRe >=45 | main recall retained | gate |")
    lines.append("|---|---:|:---:|:---:|:---:|:---:|:---:|")
    for label, value in report["variants"].items():
        gate = value.get("level1_gate", {})
        lines.append(
            f"| {label} | {gate.get('hota_delta_vs_A0', 'NA') if label != 'A0' else '—'} | "
            f"{gate.get('hota_plus5', '—')} | {gate.get('detpr_near20', '—')} | "
            f"{gate.get('detre_not_collapsed', '—')} | "
            f"{gate.get('main_recall_not_materially_down', '—')} | "
            f"{gate.get('passed', '—')} |"
        )
    lines.extend(["", "The JSON report contains the complete calibration sweep summaries, provenance, source acceptance, NULL/empty-output diagnostics, and structural failure diagnosis.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--a0-root", default=str(DEFAULT_A0_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--a1-root", default="outputs/l20/eval/fast_A1_100")
    parser.add_argument("--a2-root", default="outputs/l20/eval/fast_A2_100")
    parser.add_argument("--a3-root", default="outputs/l20/eval/fast_A3_100")
    parser.add_argument("--a4-root", default="outputs/l20/eval/fast_A4_100")
    parser.add_argument("--a5-root", default="outputs/l20/eval/fast_A5_100")
    parser.add_argument("--a6-root", default="outputs/l20/eval/fast_A6_100")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest, manifest_sha = load_manifest(manifest_path)
    expected = {int(row["query_index"]): row for row in manifest["queries"]}
    output_root = Path(args.output).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite fair comparison output: {output_root}")

    variants = {
        "A0": _load_records("A0", Path(args.a0_root).resolve(), expected,
                            manifest_sha, baseline=True),
        "A1": _load_records("A1", Path(args.a1_root).resolve(), expected,
                            manifest_sha, baseline=False),
        "A2": _load_records("A2", Path(args.a2_root).resolve(), expected,
                            manifest_sha, baseline=False),
        "A3": _load_records("A3", Path(args.a3_root).resolve(), expected,
                            manifest_sha, baseline=False),
        "A4": _load_records("A4", Path(args.a4_root).resolve(), expected,
                            manifest_sha, baseline=False),
        "A5": _load_records("A5", Path(args.a5_root).resolve(), expected,
                            manifest_sha, baseline=False),
        "A6": _load_records("A6", Path(args.a6_root).resolve(), expected,
                            manifest_sha, baseline=False),
    }
    queries, gt_root, seqmap, sequences, _protocol = trainval_queries("trainval_kitti")
    write_trainval_gt("trainval_kitti", queries, gt_root)
    gt_by_query = gt_counts(variants["A0"]["records"], gt_root)

    report = {
        "format": "locatemot-l20-fair-fast-rmot-comparison-v1",
        "provenance": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "manifest_query_count": len(expected),
            "calibration_query_count": 64,
            "screening_query_count": 96,
            "selection_uses_model_scores": manifest["selection_uses_model_scores"],
            "bank_root": str(ROOT / "outputs/l19/dual_banks_features"),
            "prediction_evaluator_l20": str(EVALUATOR),
            "prediction_evaluator_l20_sha256": sha256_file(EVALUATOR),
            "prediction_evaluator_a0_reused": str(ROOT / "tools/eval_l19_fast.py"),
            "prediction_evaluator_a0_reused_sha256": sha256_file(
                ROOT / "tools/eval_l19_fast.py"),
            "trackeval_protocol_code": str(TRACKING_PROTOCOL),
            "trackeval_protocol_code_sha256": sha256_file(TRACKING_PROTOCOL),
            "trackeval_runner": str(TRACKEVAL_RUN),
            "official_eval_used": False,
            "full_kitti_used": False,
            "mot_ovmot_used": False,
            "calibration_gt_used_for_threshold_only": True,
            "screening_gt_used_for_threshold_selection": False,
        },
        "threshold_protocol": {
            "fixed_threshold_definition": "one calibration-selected scalar frozen before screening",
            "sweep_split": "calibration",
            "screening_split": "screening",
            "quantile_grid": [float(value) for value in SWEEP_QUANTILES.tolist()],
            "candidate_endpoints": ["calibration_score_min", "calibration_score_max"],
            "selection_rule": "recall>=0.45, then F1, precision, recall, -abs(threshold)",
            "test_or_screening_gt_used_for_selection": False,
        },
        "variants": {},
    }

    for label, value in variants.items():
        threshold, sweep = sweep_thresholds(value["records"], gt_by_query)
        fixed_screening = threshold_metrics(
            value["records"], threshold, "screening", gt_by_query)
        fixed_calibration = threshold_metrics(
            value["records"], threshold, "calibration", gt_by_query)
        variant_root = output_root / label
        allowed, eval_seqmap = prepare_trackeval(
            variant_root, value["records"], threshold, gt_root)
        metrics, log = run_trackeval(
            "trainval_kitti", variant_root, eval_seqmap, sequences, allowed)
        explicit_null = bool(value["checkpoint"]["cfg"].get("use_null", False)) \
            if not value["baseline"] else False
        fixed_screening["explicit_null_enabled"] = explicit_null
        fixed_screening["prediction_root"] = str(variant_root)
        fixed_screening["gt_root"] = str(gt_root)
        report["variants"][label] = {
            "checkpoint": value["checkpoint"],
            "prediction_root": value["root"],
            "prediction_evaluator": str(
                ROOT / "tools/eval_l19_fast.py" if value["baseline"] else EVALUATOR),
            "prediction_evaluator_sha256": sha256_file(
                ROOT / "tools/eval_l19_fast.py" if value["baseline"] else EVALUATOR),
            "prediction_run_manifest": value["run_manifest"],
            "prediction_run_manifest_sha256": value["run_manifest_sha256"],
            "prediction_query_count": len(value["records"]),
            "prediction_candidate_rows": int(sum(
                len(data["score"]) for _meta, data in value["records"])),
            "fixed_threshold": threshold,
            "calibration_sweep": sweep,
            "fixed_threshold_calibration": fixed_calibration,
            "fixed_threshold_screening": fixed_screening,
            "trackeval": {"metrics": metrics, "log": str(log), "split": "screening"},
            "trackeval_allowed_query_count": len(allowed),
        }

    baseline = report["variants"]["A0"]
    baseline_metrics = baseline["trackeval"]["metrics"]
    baseline_fixed = baseline["fixed_threshold_screening"]
    for label, value in report["variants"].items():
        metrics = value["trackeval"]["metrics"]
        fixed = value["fixed_threshold_screening"]
        if label == "A0":
            value["level1_gate"] = {
                "baseline": True, "passed": False,
            }
            continue
        delta = float(metrics.get("HOTA___AUC", 0.0) -
                      baseline_metrics.get("HOTA___AUC", 0.0))
        gate = {
            "hota_delta_vs_A0": round(delta, 6),
            "hota_plus5": bool(delta >= 5.0),
            "detpr_near20": bool(metrics.get("DetPr___AUC", 0.0) >= 20.0),
            "detre_not_collapsed": bool(metrics.get("DetRe___AUC", 0.0) >= 45.0),
            "main_recall_not_materially_down": bool(
                fixed.get("main_selected_recall", 0.0) >=
                baseline_fixed.get("main_selected_recall", 0.0) - 0.05),
            "predictions_per_gt_not_higher": bool(
                fixed.get("predictions_per_gt", float("inf")) <=
                baseline_fixed.get("predictions_per_gt", float("inf"))),
        }
        gate["passed"] = all(value for key, value in gate.items()
                              if key != "hota_delta_vs_A0")
        value["level1_gate"] = gate
        value["structural_diagnosis"] = {
            "recall_collapsed": bool(fixed.get("recall", 0.0) <
                                     baseline_fixed.get("recall", 0.0) - 0.10),
            "precision_failure": bool(fixed.get("precision", 0.0) < 0.20),
            "output_volume_reduction_vs_A0": float(
                1.0 - fixed.get("predictions_per_gt", float("inf")) /
                max(1e-9, baseline_fixed.get("predictions_per_gt", 1.0))),
            "interpretation": (
                "precision/DetA failure without recall collapse"
                if fixed.get("recall", 0.0) >= baseline_fixed.get("recall", 0.0) - 0.10
                and fixed.get("precision", 0.0) < 0.20 else
                "recall collapse or mixed failure"
            ),
        }
    report["level1_decision"] = {
        "passed": any(value["level1_gate"].get("passed", False)
                       for value in report["variants"].values()),
        "passing_variants": [label for label, value in report["variants"].items()
                             if value["level1_gate"].get("passed", False)],
        "action_if_none_pass": "stop long training and retain structural failure decomposition",
    }
    report["report_code_sha256"] = sha256_file(Path(__file__).resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "comparison.json", report)
    (output_root / "comparison.md").write_text(markdown_report(report) + "\n")
    print(json.dumps({
        "report": str(output_root / "comparison.json"),
        "markdown": str(output_root / "comparison.md"),
        "manifest_sha256": manifest_sha,
        "level1": report["level1_decision"],
        "trackeval": {
            label: value["trackeval"]["metrics"]
            for label, value in report["variants"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
