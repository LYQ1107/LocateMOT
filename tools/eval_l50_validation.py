#!/usr/bin/env python3
"""Train/validation-only evaluation for the L50-B targeted or long run.

Calibration thresholds are fitted from the single calibration video per
domain.  Fit and video-disjoint validation labels are used for diagnostics;
the script has no official-test path and never opens a test label.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l50_domain_balanced_semantic import (  # noqa: E402
    L50DomainBalancedSemanticMatcher,
)
from locatemot.rmot.l49_data import (  # noqa: E402
    L29_CHECKPOINT,
    TEXT_CACHE,
    load_bank,
    sha256_file,
    unit_features,
)
from tools.eval_l49_validation import source_masks, summarize  # noqa: E402

DATA = ROOT / "outputs/l49/data"
CONTRACT = ROOT / "outputs/l49/audit/kitti_data_contract.json"
BASELINE_AUDIT = ROOT / "outputs/l50/audit/baseline_replay.json"
FIT_BASELINE = ROOT / "outputs/l49/val/fit_baseline_scores_selected.jsonl"
VAL_BASELINE = ROOT / "outputs/l49/val/validation_baseline_scores.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record_key(row: dict) -> tuple[str, str, int, int]:
    return (str(row["dataset"]), str(row["video"]), int(row["query_id"]), int(row["frame_id"]))


def parse_cached_records(path: Path, expected_split: str) -> list[dict]:
    records = []
    if not path.exists():
        raise FileNotFoundError(path)
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("split") not in (None, expected_split):
            raise AssertionError(f"unexpected split in {path}: {row.get('split')}")
        result = dict(row)
        result.pop("split", None)
        result["score"] = np.asarray(result["score"], dtype=np.float32)
        result["label"] = np.asarray(result["label"], dtype=bool)
        result["sources"] = {key: np.asarray(value, dtype=bool)
                              for key, value in result["sources"].items()}
        records.append(result)
    if len({record_key(x) for x in records}) != len(records):
        raise AssertionError(f"duplicate keys in {path}")
    return records


class BankStore:
    def __init__(self, limit: int = 2):
        self.limit = int(limit)
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def get(self, dataset: str, video: str):
        key = f"{dataset}|{video}"
        if key not in self.cache:
            self.cache[key] = load_bank(dataset, video)
            if len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]


def fit_threshold(records: list[dict]) -> dict:
    values = np.concatenate([np.asarray(x["score"], dtype=np.float64) for x in records if len(x["score"])]) if records else np.zeros(0)
    labels = np.concatenate([np.asarray(x["label"], dtype=bool) for x in records if len(x["label"])]) if records else np.zeros(0, dtype=bool)
    if not len(values):
        return {"threshold": 0.0, "objective": "calibration_candidate_f1", "units": 0,
                "labels_source": "calibration_only", "validation_or_test_labels_used": False}
    candidates = np.unique(values)
    if len(candidates) > 256:
        candidates = np.quantile(values, np.linspace(0, 1, 256))
    best = None
    for threshold in candidates.tolist() + [float(values.max()) + 1e-6, float(values.min()) - 1e-6]:
        chosen = values >= threshold
        tp = int((chosen & labels).sum()); fp = int((chosen & ~labels).sum()); fn = int((~chosen & labels).sum())
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        key = (f1, -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), tp, fp, fn)
    return {"threshold": best[1], "objective": "calibration_candidate_f1", "units": len(records),
            "tp": best[2], "fp": best[3], "fn": best[4], "labels_source": "calibration_only",
            "validation_or_test_labels_used": False}


def compact(summary: dict) -> dict:
    keys = ("frame_units", "candidate_rows", "positive_rows", "positive_frame_units", "precision", "recall",
            "top1_frame_recall", "top5_frame_recall", "false_positive_candidates_per_frame", "empty_output_rate",
            "null_frame_false_acceptance", "predictions_per_positive", "multi_positive_recall",
            "hard_violation_rate", "strict_min_positive_margin", "best_positive_margin",
            "average_positive_margin", "source_precision", "null_max_score", "threshold")
    return {key: summary.get(key) for key in keys}


def make_records(model, units, store, text, device):
    model.eval()
    grouped = defaultdict(list)
    for unit in units:
        grouped[(str(unit["dataset"]), str(unit["video"]))].append(unit)
    records = []
    for (dataset, video), group in sorted(grouped.items()):
        bank = store.get(dataset, video)
        for unit in group:
            values = unit_features(unit, bank, text, history=8)
            moved = {key: value.to(device, non_blocking=True) for key, value in values.items()
                     if key != "target"}
            with torch.inference_mode():
                output = model(moved["clip"], moved["history_clip"], moved["geometry"], moved["motion"],
                               moved["context"], moved["lifecycle"], moved["objectness"], moved["text"],
                               moved["text_mask"], moved["relation"])
            begin, end = int(unit["begin"]), int(unit["end"])
            label = values["target"].numpy().astype(bool)
            records.append({
                "dataset": dataset, "video": video, "query_id": int(unit["query_id"]),
                "expression": unit.get("expression", ""), "sentence": unit["sentence"],
                "frame_id": int(unit["frame_id"]), "category": unit["category"],
                "candidate_count": int(len(label)), "positive_count": int(label.sum()),
                "score": output["semantic_logit"].float().cpu().numpy(),
                "label": label,
                "sources": source_masks(bank, begin, end),
            })
    return records


def rank_flips(records: list[dict], teacher_by_key: dict) -> dict:
    counts = defaultdict(int)
    for row in records:
        teacher_row = teacher_by_key.get(record_key(row))
        if teacher_row is None:
            continue
        label = np.asarray(row["label"], dtype=bool)
        pos = np.flatnonzero(label); neg = np.flatnonzero(~label)
        if not len(pos) or not len(neg):
            continue
        teacher = np.asarray(teacher_row["score"], dtype=np.float64)
        student = np.asarray(row["score"], dtype=np.float64)
        teacher_order = (teacher[pos, None] > teacher[None, neg]).reshape(-1)
        student_order = (student[pos, None] > student[None, neg]).reshape(-1)
        counts["frame_units"] += 1
        counts["pair_count"] += int(len(teacher_order))
        counts["teacher_correct_pairs"] += int(teacher_order.sum())
        counts["student_correct_pairs"] += int(student_order.sum())
        counts["teacher_correct_student_flip"] += int((teacher_order & ~student_order).sum())
        counts["teacher_error_student_correction"] += int((~teacher_order & student_order).sum())
        counts["total_rank_flip"] += int((teacher_order != student_order).sum())
    pairs = max(1, counts["pair_count"])
    return {**dict(counts),
            "teacher_correct_student_flip_rate": counts["teacher_correct_student_flip"] / pairs,
            "teacher_error_student_correction_rate": counts["teacher_error_student_correction"] / pairs,
            "total_rank_flip_rate": counts["total_rank_flip"] / pairs}


def scale_stats(records: list[dict], teacher_by_key: dict) -> dict:
    scores = np.concatenate([np.asarray(row["score"], dtype=np.float64) for row in records]) if records else np.zeros(0)
    teachers = [np.asarray(teacher_by_key[record_key(row)]["score"], dtype=np.float64)
                for row in records if record_key(row) in teacher_by_key]
    teacher_values = np.concatenate(teachers) if teachers else np.zeros(0)
    return {
        "candidate_rows": int(len(scores)),
        "score_mean": float(scores.mean()) if len(scores) else None,
        "score_std": float(scores.std()) if len(scores) else None,
        "score_q01": float(np.quantile(scores, .01)) if len(scores) else None,
        "score_q99": float(np.quantile(scores, .99)) if len(scores) else None,
        "teacher_mean": float(teacher_values.mean()) if len(teacher_values) else None,
        "teacher_std": float(teacher_values.std()) if len(teacher_values) else None,
    }


def jsonable(row: dict) -> dict:
    result = dict(row)
    result["score"] = np.asarray(result["score"], dtype=np.float32).tolist()
    result["label"] = np.asarray(result["label"], dtype=bool).tolist()
    result["sources"] = {key: np.asarray(value, dtype=bool).tolist() for key, value in result["sources"].items()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="outputs/l50/train/targeted500")
    parser.add_argument("--out-root", default="outputs/l50/eval/targeted500")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.time()
    try:
        contract = json.loads(CONTRACT.read_text())
        if contract.get("stage") != "L49-A":
            raise RuntimeError("unexpected L49 contract stage")
        baseline_audit = json.loads(BASELINE_AUDIT.read_text())
        if baseline_audit.get("official_test_labels_read") or baseline_audit.get("test_paths_opened"):
            raise RuntimeError("L50-A audit is not test-free")
        train_units = load_jsonl(DATA / "train_units.jsonl")
        cal_units = load_jsonl(DATA / "calibration_units.jsonl")
        val_units = load_jsonl(DATA / "validation_units.jsonl")
        text = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
        if not ({x["sentence"] for x in train_units + cal_units + val_units}
                <= set(text["sentence_to_index"])):
            raise AssertionError("a train/calibration/validation expression is missing from text cache")
        fit_baseline = parse_cached_records(FIT_BASELINE, "fit")
        val_baseline = parse_cached_records(VAL_BASELINE, "validation")
        baseline_by_key = {record_key(row): row for row in val_baseline}
        baseline_fit_by_key = {record_key(row): row for row in fit_baseline}
        if len(fit_baseline) != len(train_units) or len(val_baseline) != len(val_units):
            raise AssertionError("immutable baseline cache count does not match L49 units")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        checkpoint_root = Path(args.checkpoint_root)
        if not checkpoint_root.is_absolute():
            checkpoint_root = ROOT / checkpoint_root
        checkpoints = sorted(checkpoint_root.glob("checkpoint_l50_step*.pt"),
                             key=lambda p: int(p.stem.split("step")[-1]))
        if not checkpoints:
            raise FileNotFoundError(f"no L50 checkpoints under {checkpoint_root}")
        store = BankStore(limit=2)
        results = {}
        all_jsonl = []
        baseline_thresholds = {
            domain: float(value) for domain, value in
            baseline_audit["calibration"]["selected_thresholds"].items()
        }
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("official_test_labels_read") or payload.get("screening_gt_used"):
                raise RuntimeError(f"checkpoint provenance is not train-only: {checkpoint}")
            cfg = payload.get("model_config", {})
            model = L50DomainBalancedSemanticMatcher(hidden=int(cfg.get("hidden", 256)), heads=int(cfg.get("heads", 4)),
                                                     dropout=0.1).to(device)
            model.load_state_dict(payload["model"], strict=True)
            step = int(payload.get("steps", payload.get("checkpoint_step", checkpoint.stem.split("step")[-1])))
            cal_records = make_records(model, cal_units, store, text, device)
            fit_records = make_records(model, train_units, store, text, device)
            val_records = make_records(model, val_units, store, text, device)
            threshold_by_domain = {}
            per_domain = {}
            for domain in ("refer_kitti_v1", "refer_kitti_v2"):
                cal_domain = [x for x in cal_records if x["dataset"] == domain]
                fit_domain = [x for x in fit_records if x["dataset"] == domain]
                val_domain = [x for x in val_records if x["dataset"] == domain]
                threshold = fit_threshold(cal_domain)
                threshold_by_domain[domain] = threshold
                l50_fit = summarize(fit_domain, threshold["threshold"])
                l50_val = summarize(val_domain, threshold["threshold"])
                l29_fit = summarize([x for x in fit_baseline if x["dataset"] == domain], baseline_thresholds[domain])
                l29_val = summarize([x for x in val_baseline if x["dataset"] == domain], baseline_thresholds[domain])
                per_domain[domain] = {
                    "calibration": {"units": len(cal_domain), "threshold": threshold},
                    "fit": {"l50": compact(l50_fit), "l29": compact(l29_fit)},
                    "validation": {"l50": compact(l50_val), "l29": compact(l29_val)},
                    "delta_vs_l29": {
                        "top1": l50_val.get("top1_frame_recall", 0) - l29_val.get("top1_frame_recall", 0),
                        "recall": l50_val.get("recall", 0) - l29_val.get("recall", 0),
                        "precision": l50_val.get("precision", 0) - l29_val.get("precision", 0),
                        "fp_per_frame": l50_val.get("false_positive_candidates_per_frame", 0) - l29_val.get("false_positive_candidates_per_frame", 0),
                        "hard_violation": l50_val.get("hard_violation_rate") - l29_val.get("hard_violation_rate"),
                        "multi_positive_recall": l50_val.get("multi_positive_recall", 0) - l29_val.get("multi_positive_recall", 0),
                        "empty": l50_val.get("empty_output_rate", 0) - l29_val.get("empty_output_rate", 0),
                    },
                }
            fit_rank = rank_flips(fit_records, baseline_fit_by_key)
            val_rank = rank_flips(val_records, baseline_by_key)
            fit_scale = scale_stats(fit_records, baseline_fit_by_key)
            val_scale = scale_stats(val_records, baseline_by_key)
            domain_gates = {}
            for domain, item in per_domain.items():
                delta = item["delta_vs_l29"]
                base = item["validation"]["l29"]
                new = item["validation"]["l50"]
                domain_gates[domain] = {
                    "recall_not_below_l29_003": delta["recall"] >= -0.03,
                    "multi_positive_not_below_l29_003": delta["multi_positive_recall"] >= -0.03,
                    "hard_violation_improved_by_005": delta["hard_violation"] <= -0.05,
                    "precision_not_collapsed": delta["precision"] >= -0.03,
                    "fp_not_abnormally_increased": delta["fp_per_frame"] <= 0.50,
                    "not_empty_driven": new.get("empty_output_rate", 0) <= base.get("empty_output_rate", 0) + 0.05,
                }
            results[str(step)] = {
                "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
                "calibration_thresholds": threshold_by_domain, "per_domain": per_domain,
                "rank_flips_vs_l29": {"fit": fit_rank, "validation": val_rank},
                "score_scale": {"fit": fit_scale, "validation": val_scale,
                                "fit_to_validation_mean_delta": (val_scale["score_mean"] - fit_scale["score_mean"]),
                                "fit_to_validation_std_ratio": (val_scale["score_std"] / max(1e-9, fit_scale["score_std"]))},
                "gate": {"per_domain": domain_gates,
                         "both_domains_pass": all(all(flags.values()) for flags in domain_gates.values())},
            }
            for split, rows in (("calibration", cal_records), ("fit", fit_records), ("validation", val_records)):
                for row in rows:
                    all_jsonl.append({"split": split, "checkpoint_step": step, **jsonable(row)})
            del model, payload, cal_records, fit_records, val_records
            gc.collect()
        passing = [int(step) for step, value in results.items() if value["gate"]["both_domains_pass"]]
        selected_step = max(passing, key=lambda x: np.mean([
            results[str(x)]["per_domain"][domain]["validation"]["l50"]["precision"]
            for domain in ("refer_kitti_v1", "refer_kitti_v2")])) if passing else None
        output = {
            "format": "locatemot-l50-validation-error-matrix-v1", "stage": "L50-B",
            "project_root": str(ROOT), "started_at_unix": started, "completed_at_unix": time.time(),
            "checkpoint_root": str(checkpoint_root.resolve()), "checkpoint_results": results,
            "passing_steps": passing, "selected_step": selected_step,
            "selection_rule": "no checkpoint selected unless both-domain validation gate passes",
            "calibration_labels_only": True, "validation_used_for_gate": True,
            "official_test_labels_read": False, "test_paths_opened": False,
            "screening_gt_used": False, "ordinary_mot_ovmot_touched": False,
            "manifest_sha256": "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa",
            "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT),
            "data_contract_sha256": sha256_file(CONTRACT), "text_cache_sha256": sha256_file(TEXT_CACHE),
            "elapsed_sec": time.time() - started,
        }
        (out / "error_matrix.json").write_text(json.dumps(output, indent=2) + "\n")
        with (out / "scores.jsonl").open("w") as handle:
            for row in all_jsonl:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        (out / "provenance.json").write_text(json.dumps({
            "format": "locatemot-l50-validation-provenance-v1", "checkpoint_count": len(results),
            "train_units": len(train_units), "calibration_units": len(cal_units), "validation_units": len(val_units),
            "official_test_labels_read": False, "test_paths_opened": False,
            "screening_gt_used": False, "ordinary_mot_ovmot_touched": False,
            "elapsed_sec": time.time() - started,
        }, indent=2) + "\n")
        print(json.dumps({"status": "pass", "checkpoints": sorted(results, key=int),
                          "passing_steps": passing, "selected_step": selected_step,
                          "elapsed_sec": time.time() - started}, indent=2), flush=True)
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text(
            "# L50 validation incomplete\n\n"
            f"First actionable error: `{type(exc).__name__}: {exc}`\n"
            "No official test labels were opened. Historical outputs were not modified.\n"
        )
        raise


if __name__ == "__main__":
    main()
