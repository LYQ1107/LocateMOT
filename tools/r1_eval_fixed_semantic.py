#!/usr/bin/env python3
"""Run the frozen R1 fixed 16-calibration/24-validation diagnostic.

Legal-dev checkpoint selection is complete before this script is allowed to
open the fixed-unit labels.  Feature construction is performed first for
each key-only record; the L49 target and the L69 candidate sidecar are then
attached explicitly for metrics.  This is an internal diagnostic, not
screening or official-test evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.models.r1_aligned_track_conditioning import FrozenL89EAnchor  # noqa: E402
from locatemot.rmot.r1_aligned_cache import (  # noqa: E402
    R0_VISUAL_MANIFEST,
    R1AlignedCacheIndex,
    R1FeatureAssembler,
    row_digest,
    sha256_file,
    write_json,
)
from tools.l88_eval_metrics import metric  # noqa: E402
from tools.r1_common import (  # noqa: E402
    MANIFEST,
    MANIFEST_SHA,
    SEED,
    THREAD,
    check_manifest,
    file_meta,
    load_fixed_l62_key_order,
    standard_flags,
)
from tools.r1_infer_legal_dev import (  # noqa: E402
    L89E_ANCHOR,
    L89E_SELECTION,
    _load_sidecar,
)


FIXED_L62_ROWS = WORK_ROOT / "../LocateMOT/outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
L49_CAL = WORK_ROOT / "../LocateMOT/outputs/l49/data/calibration_units.jsonl"
L49_VAL = WORK_ROOT / "../LocateMOT/outputs/l49/data/validation_units.jsonl"
ALIGNED_CACHE = WORK_ROOT / "outputs/r1/cache/eval_attempt2"
VISUAL_SUPPLEMENT = Path("/data2/usr_for_deadline/locatemot_r1_fixed_visual_supplement_attempt1")
SELECTION_SUMMARY = WORK_ROOT / "outputs/r1/legal_dev_aggregate_attempt11/summary.json"
L29_CONTROL = {
    "legacy_candidate_recall": 0.7333333333333333,
    "legacy_candidate_precision": 0.0830188679245283,
    "legacy_fp_per_frame": 10.125,
    "legacy_predictions_per_positive": 8.833333333333334,
    "legacy_row_hard_violation": 0.9166666666666666,
    "legacy_row_multi_positive_recall": 0.8194444444444443,
}
RULE = {"name": "B", "candidate_threshold": 1.0, "presence_threshold": 0.5, "null_margin": 0.0}


def _read_jsonl_label(path: Path, unit_key: str) -> dict[str, Any]:
    """Read one label row at the explicit post-feature boundary."""
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if str(value.get("unit_key")) == str(unit_key):
                return {
                    "unit_key": str(value["unit_key"]),
                    "target_ids": [str(item) for item in value.get("target_ids", [])],
                    "source": str(path.resolve()),
                    "category_from_l49": str(value.get("category", "unknown")),
                }
    raise KeyError(f"fixed label row missing: {unit_key} in {path}")


def _score_frame(anchor: FrozenL89EAnchor, sidecar: torch.nn.Module, frame: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    device = frame.z1.device
    n = int(frame.candidate_count)
    text_tokens = frame.text_tokens.unsqueeze(0).clone()
    text_mask = frame.text_mask.unsqueeze(0).clone()
    text_global = frame.text_global.unsqueeze(0).clone()
    frame_global = frame.frame_global.unsqueeze(0).clone()
    z0, z1, z4 = (value.unsqueeze(0).clone() for value in (frame.z0, frame.z1, frame.z4))
    history = frame.history_observations.clone()
    history_mask = frame.history_mask.clone()
    history_frames = frame.history_frame_ids.clone()
    current = history[:, -1].clone()
    geometry = frame.geometry.clone()
    geometry_mask = torch.ones(geometry.shape[:2], dtype=torch.bool, device=device)
    raw = frame.raw_visual_tokens.unsqueeze(0).clone()
    boxes = frame.boxes_norm.clone()
    if tuple(z1.shape) != (1, n, 256) or tuple(raw.shape) != (1, n, 72, 256):
        raise AssertionError(f"fixed semantic input shape drift: {frame.unit_key if hasattr(frame, 'unit_key') else frame.frame_id}")
    with torch.inference_mode():
        base = anchor(z1, text_tokens, text_mask, text_global, frame_global, current,
                      history, history_mask, history_frames, int(frame.frame_id))
        side = sidecar(z0, z1, z4, text_tokens, text_mask, text_global, raw,
                       geometry, geometry_mask, boxes, base["candidate_energy"],
                       base["presence_logit"], base["null_logit"])
    for name, value in side.items():
        if torch.is_tensor(value) and not bool(torch.isfinite(value.float()).all()):
            raise FloatingPointError(f"nonfinite R1 fixed semantic output: {name}")
    base_score = base["candidate_energy"][0].float().cpu().tolist()
    side_score = side["final_energy"][0].float().cpu().tolist()
    if len(base_score) != n or len(side_score) != n:
        raise AssertionError("fixed semantic candidate score length drift")
    anchor_record = {"score": [float(value) for value in base_score],
                     "presence_logit": float(base["presence_logit"].reshape(-1)[0].item()),
                     "null_logit": float(base["null_logit"].reshape(-1)[0].item())}
    side_record = {"score": [float(value) for value in side_score],
                   "presence_logit": float(side["final_presence"].reshape(-1)[0].item()),
                   "null_logit": float(side["final_null"].reshape(-1)[0].item())}
    del text_tokens, text_mask, text_global, frame_global, z0, z1, z4, history, history_mask
    del history_frames, current, geometry, geometry_mask, raw, boxes, base, side
    return anchor_record, side_record


def _metric_view(records: list[dict[str, Any]], score_field: str, *, candidate_only: bool) -> dict[str, Any]:
    view: list[dict[str, Any]] = []
    for row in records:
        score = row[score_field]
        view.append({
            "unit_key": row["unit_key"], "dataset": row["dataset"], "video": row["video"],
            "score": score["score"], "presence_logit": score["presence_logit"],
            "null_logit": score["null_logit"], "labels": row["labels"],
            "candidate_gt": row["candidate_gt"], "target_ids": row["target_ids"],
            "category": row["category"], "row_keys": row["row_keys"],
        })
    if candidate_only:
        return metric(view, float(RULE["candidate_threshold"]), -1.0e9, -1.0e9)
    return metric(view, float(RULE["candidate_threshold"]), float(RULE["presence_threshold"]), float(RULE["null_margin"]))


def run(args: argparse.Namespace) -> int:
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty fixed semantic output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([str(sys.executable), *sys.argv])
    flags = standard_flags(training_run=False, hota_trackeval_run=False)
    base = {
        "format": "locatemot-r1-fixed-semantic-v1", "status": "incomplete", "command": command,
        "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED, "scope": "fixed_l62_16cal_24val",
        "manifest_sha256": None, "rule": RULE, "selection_summary": str(SELECTION_SUMMARY.resolve()),
        "selection_labels_used": False, "prediction_before_label": True,
        "labels_attached_after_features": 0, "labels_attached_after_selection": True,
        "candidate_deletion": False, "candidate_truncation": False,
        "no_new_feature_cache": True, "failure_root_cause": None,
        **flags,
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"R1 fixed semantic evaluator must run from {WORK_ROOT}")
        manifest_sha = check_manifest()
        base["manifest_sha256"] = manifest_sha
        if manifest_sha != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        fixed = load_fixed_l62_key_order()
        if len(fixed) != 40 or [item["split"] for item in fixed[:16]] != ["calibration"] * 16 or \
                [item["split"] for item in fixed[16:]] != ["validation"] * 24:
            raise AssertionError("fixed 16/24 order contract failed")
        selection_payload = json.loads(SELECTION_SUMMARY.resolve().read_text(encoding="utf-8"))
        if selection_payload.get("status") != "complete":
            raise AssertionError("legal-dev selection summary is not complete")
        selected = selection_payload.get("selection")
        if not isinstance(selected, dict) or set(selected) != {"refer_kitti_v1", "refer_kitti_v2"}:
            raise AssertionError("selected V1/V2 legal-dev checkpoints missing")
        checkpoint_paths = {dataset: Path(str(selected[dataset]["selected_checkpoint"])).resolve()
                            for dataset in selected}
        checkpoint_shas = {dataset: sha256_file(path) for dataset, path in checkpoint_paths.items()}
        if any(checkpoint_shas[dataset] != str(selected[dataset]["selected_checkpoint_sha256"])
               for dataset in selected):
            raise AssertionError("selected checkpoint SHA drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("requested CUDA but it is unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        anchor = FrozenL89EAnchor(L89E_ANCHOR).to(device=device)
        anchor.eval()
        if any(parameter.requires_grad for parameter in anchor.parameters()):
            raise AssertionError("R1 anchor is not frozen")
        sidecars: dict[str, torch.nn.Module] = {}
        sidecar_info: dict[str, Any] = {}
        for dataset, path in checkpoint_paths.items():
            sidecars[dataset], sidecar_info[dataset] = _load_sidecar(path, device)
        aligned = R1AlignedCacheIndex(ALIGNED_CACHE)
        assembler = R1FeatureAssembler(aligned, device, visual_roots=[R0_VISUAL_MANIFEST, VISUAL_SUPPLEMENT])
        records: list[dict[str, Any]] = []
        label_attach_count = 0
        for index, key_record in enumerate(fixed):
            # The returned key record deliberately has no target/category fields.
            forbidden = {"target_ids", "positive_indices", "positive_count", "category", "labels", "target_present"}
            if forbidden.intersection(key_record):
                raise AssertionError(f"pre-feature key record exposes labels: {key_record['unit_key']}")
            frame = assembler.prepare(dict(key_record), attach_labels=False)
            anchor_score, side_score = _score_frame(anchor, sidecars[str(key_record["dataset"])], frame)
            # Only now open the L49 target row and the L69 candidate sidecar.
            label_path = L49_CAL if key_record["split"] == "calibration" else L49_VAL
            label = _read_jsonl_label(label_path, str(key_record["unit_key"]))
            supervision = assembler.store.attach_frame_labels(frame, label["target_ids"])
            label_attach_count += 1
            if supervision["labels"].numel() != frame.candidate_count:
                raise AssertionError(f"fixed semantic label length drift: {key_record['unit_key']}")
            row_keys = [list(value) for value in frame.row_keys]
            if len(row_keys) != frame.candidate_count or len({tuple(value) for value in row_keys}) != frame.candidate_count:
                raise AssertionError(f"fixed semantic row-key drift: {key_record['unit_key']}")
            if [int(value) for value in frame.history_frame_ids[frame.history_mask].tolist()
                    if int(value) > int(frame.frame_id)]:
                raise AssertionError(f"future history in fixed semantic: {key_record['unit_key']}")
            records.append({
                "dataset": str(key_record["dataset"]), "video": str(key_record["video"]),
                "query_id": int(key_record["query_id"]), "frame_id": int(key_record["frame_id"]),
                "unit_key": str(key_record["unit_key"]), "split": str(key_record["split"]),
                "sentence": str(key_record["sentence"]), "candidate_count": int(frame.candidate_count),
                "candidate_indices": [int(value) for value in frame.candidate_indices],
                "row_keys": row_keys, "row_key_digest": row_digest(frame.row_keys),
                "labels": [bool(value) for value in supervision["labels"].tolist()],
                "candidate_gt": list(supervision["candidate_gt"]), "target_ids": list(supervision["target_ids"]),
                "covered_target_ids": list(supervision["covered_target_ids"]),
                "category": str(supervision["category"]), "target_present": bool(supervision["target_present"]),
                "candidate_present": bool(supervision["candidate_present"]),
                "present_uncovered": bool(supervision["present_uncovered"]),
                "anchor": anchor_score, "r1": side_score,
                "labels_attached_after_feature_construction": True,
                "labels_attached_after_selection": True,
            })
            del frame, supervision, label
            if (index + 1) % 8 == 0:
                gc.collect()
        if len(records) != 40 or label_attach_count != 40:
            raise AssertionError("fixed semantic record count drift")
        if [row["unit_key"] for row in records] != [row["unit_key"] for row in fixed]:
            raise AssertionError("fixed semantic unit order drift")
        for row in records:
            if len(row["r1"]["score"]) != row["candidate_count"] or not all(math.isfinite(value) for value in row["r1"]["score"]):
                raise AssertionError(f"fixed semantic score finiteness/length drift: {row['unit_key']}")
        semantic = {
            "format": "locatemot-r1-fixed-semantic-metrics-v1", "status": "complete",
            "scope": "fixed_l62_16cal_24val", "rule": RULE,
            "unit_count": len(records), "calibration_units": 16, "validation_units": 24,
            "candidate_row_count": int(sum(row["candidate_count"] for row in records)),
            "anchor_candidate_only": _metric_view(records, "anchor", candidate_only=True),
            "anchor_rule_b": _metric_view(records, "anchor", candidate_only=False),
            "r1_candidate_only": _metric_view(records, "r1", candidate_only=True),
            "r1_rule_b": _metric_view(records, "r1", candidate_only=False),
            "l29_immutable_control": L29_CONTROL,
            "candidate_set_note": "R1 L69 budget-40 rows differ from immutable L29 rows; no row-paired equality claimed",
            "selection": selected,
            "selection_checkpoint_sha256": checkpoint_shas,
            "flags": flags,
        }
        write_json(out / "semantic.json", semantic)
        with (out / "score_records.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        provenance = {
            **base, "status": "complete", "outputs": {"semantic": str((out / "semantic.json").resolve()),
                                                        "score_records": str((out / "score_records.jsonl").resolve())},
            "inputs": {"manifest": file_meta(MANIFEST), "l62_order": file_meta(FIXED_L62_ROWS),
                       "l49_calibration": file_meta(L49_CAL), "l49_validation": file_meta(L49_VAL),
                       "l89e_anchor": file_meta(L89E_ANCHOR), "l89e_selection": file_meta(L89E_SELECTION),
                       "legal_dev_selection_summary": file_meta(SELECTION_SUMMARY),
                       "aligned_cache_status": file_meta(ALIGNED_CACHE / "status.json"),
                       "visual_manifest": file_meta(R0_VISUAL_MANIFEST / "manifest.jsonl"),
                       "visual_supplement": file_meta(VISUAL_SUPPLEMENT / "manifest.jsonl")},
            "fixed_key_order_count": len(fixed), "row_order_complete": True,
            "candidate_rows_retained": True, "labels_attached_after_features_count": label_attach_count,
            "label_files_used_only_after_feature_score": True, "checkpoint_paths": {k: str(v) for k, v in checkpoint_paths.items()},
            "checkpoint_shas": checkpoint_shas, "anchor_sha256": sha256_file(L89E_ANCHOR),
            "aligned_cache": str(ALIGNED_CACHE.resolve()), "visual_roots": [str(R0_VISUAL_MANIFEST), str(VISUAL_SUPPLEMENT)],
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started,
            "next_action": "review legal-dev TrackEval and fixed semantic diagnostic; no screening/official evaluation in R1",
        }
        write_json(out / "provenance.json", provenance)
        write_json(out / "gate_decision.json", {
            "format": "locatemot-r1-fixed-semantic-decision-v1", "status": "diagnostic_only",
            "decision": "STOPPED_PENDING_SUPERVISOR_REVIEW", "semantic_gate": "not_a_preregistered_final_gate",
            "reason": "R1 legal-dev selection is complete; this fixed 40-unit result is an internal diagnostic",
            "r1_rule_b": semantic["r1_rule_b"], "anchor_rule_b": semantic["anchor_rule_b"],
            "l29_immutable_control": L29_CONTROL, "candidate_set_comparable": False,
            **flags, "next_action": provenance["next_action"],
        })
        write_json(out / "status.json", {"format": base["format"], "status": "complete", "output": str(out),
                                          "unit_count": 40, "candidate_row_count": semantic["candidate_row_count"],
                                          "candidate_deletion": False, "candidate_truncation": False,
                                          "labels_attached_after_features_count": label_attach_count,
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                                          "failure_root_cause": None, "next_action": provenance["next_action"]})
        return 0
    except BaseException as exc:
        trace = __import__("traceback").format_exc()
        (out / "INCOMPLETE.md").write_text("# R1 fixed semantic — INCOMPLETE\n\n```text\n" + trace + "```\n", encoding="utf-8")
        failure = f"{type(exc).__name__}: {exc}"
        write_json(out / "provenance.json", {**base, "failure_root_cause": failure,
                                               "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                                               "wall_seconds": time.perf_counter() - started})
        write_json(out / "status.json", {"format": base["format"], "status": "incomplete",
                                          "output": str(out), "failure_root_cause": failure,
                                          "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                                          "next_action": "repair first fixed semantic contract error in a new attempt"})
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
