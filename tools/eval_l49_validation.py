#!/usr/bin/env python3
"""L49 calibration-only thresholding and video-disjoint validation report."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder  # noqa: E402
from locatemot.models.l49_kitti_rmot import L49KittiRMOT  # noqa: E402
from locatemot.rmot.l49_data import (  # noqa: E402
    L29_CHECKPOINT, TEXT_CACHE, load_bank, sha256_file, unit_features,
)
from tools.train_l49_kitti_rmot import build_teacher_cache, valid_track_indices  # noqa: E402
from tools.train_l28_track_set_decoder import state_at  # noqa: E402

DATA = ROOT / "outputs/l49/data"
CONTRACT = ROOT / "outputs/l49/audit/kitti_data_contract.json"
TRAIN_ROOT = ROOT / "outputs/l49/train/joint_long5000"
OUT = ROOT / "outputs/l49/val"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class BankStore:
    def __init__(self, limit=2):
        self.limit = int(limit)
        self.cache = OrderedDict()

    def get(self, dataset, video):
        key = str(video)
        if key not in self.cache:
            self.cache[key] = load_bank(dataset, key)
            if len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]


def source_masks(bank: dict, begin: int, end: int):
    pool = bank["tensors"].get("pool_id")
    count = int(end - begin)
    if pool is None:
        return {"unknown": np.ones(count, dtype=bool)}
    values = pool[int(begin):int(end)].long().numpy()
    return {"main": values == 0, "reserve": values != 0}


def l29_score(model, cache, bank, unit, text, device):
    frame = int(unit["frame_id"])
    obs, mask, obs_time, _, _ = state_at(cache, frame, history=8)
    if not len(obs):
        return np.full(int(unit["end"] - unit["begin"]), -20.0, dtype=np.float32)
    index = text["sentence_to_index"][unit["sentence"]]
    with torch.inference_mode():
        encoded = model.encode_observations(obs.to(device), mask.to(device), obs_time.to(device))
        out = model.forward_encoded(encoded, encoded[1], text["token_hidden"][index].to(device),
                                    text["attention_mask"][index].bool().to(device))
    tracks = cache["track_ids"][torch.as_tensor(valid_track_indices(cache, frame), dtype=torch.long)].tolist()
    values = {int(track): float(value) for track, value in
              zip(tracks, out["current_membership_logits"].float().cpu().tolist())}
    tensors = bank["tensors"]
    return np.asarray([values.get(int(tensors["track_id"][row]), -20.0)
                       for row in range(int(unit["begin"]), int(unit["end"]))], dtype=np.float32)


def auc_score(values, labels):
    values = np.asarray(values, dtype=np.float64); labels = np.asarray(labels, dtype=bool)
    pos = values[labels]; neg = values[~labels]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64); ranks[order] = np.arange(1, len(values) + 1)
    return float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(values, labels):
    values = np.asarray(values, dtype=np.float64); labels = np.asarray(labels, dtype=bool)
    total = int(labels.sum())
    if not total:
        return None
    order = np.argsort(-values, kind="stable"); hits = labels[order]
    cumulative = np.cumsum(hits)
    return float((cumulative[hits] / np.arange(1, len(hits) + 1)[hits]).sum() / total)


def fit_threshold(records):
    values = np.concatenate([x["score"] for x in records if len(x["score"])]) if records else np.zeros(0)
    labels = np.concatenate([x["label"] for x in records if len(x["label"])]) if records else np.zeros(0, dtype=bool)
    if not len(values):
        return {"threshold": 0.0, "objective": "calibration_frame_candidate_f1", "units": 0}
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
    return {"threshold": best[1], "objective": "calibration_frame_candidate_f1",
            "units": len(records), "tp": best[2], "fp": best[3], "fn": best[4],
            "screening_or_test_labels_used": False}


def dist(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "q10": float(np.quantile(x, .1)), "q90": float(np.quantile(x, .9))}


def summarize(records, threshold):
    if not records:
        return {"frame_units": 0, "threshold": float(threshold)}
    flat_score = np.concatenate([x["score"] for x in records])
    flat_label = np.concatenate([x["label"] for x in records])
    tp = fp = fn = 0; empty = 0; null_false = 0; top1 = top5 = 0
    positive_units = multi_units = multi_recall = 0
    strict = []; best = []; average = []; violations = []
    source_counts = defaultdict(lambda: [0, 0])
    null_scores = []
    for rec in records:
        score = np.asarray(rec["score"], dtype=np.float32); label = np.asarray(rec["label"], dtype=bool)
        chosen = score >= float(threshold)
        tp += int((chosen & label).sum()); fp += int((chosen & ~label).sum()); fn += int((~chosen & label).sum())
        empty += int(not chosen.any())
        null_false += int(not label.any() and chosen.any())
        null_scores.append(float(score.max()) if len(score) else -20.0)
        for name, mask in rec["sources"].items():
            source_counts[name][0] += int((chosen & mask).sum())
            source_counts[name][1] += int((chosen & mask & label).sum())
        pos = np.flatnonzero(label); neg = np.flatnonzero(~label)
        if not len(pos):
            continue
        positive_units += 1
        order = np.argsort(-score, kind="stable")
        top1 += int(label[order[:1]].any()); top5 += int(label[order[:5]].any())
        if len(pos) > 1:
            multi_units += 1
            multi_recall += float((chosen & label).sum() / len(pos))
        if len(neg):
            strict_value = float(score[pos].min() - score[neg].max())
            best_value = float(score[pos].max() - score[neg].max())
            average_value = float(score[pos].mean() - score[neg].max())
            strict.append(strict_value); best.append(best_value); average.append(average_value)
            violations.append(strict_value < 0)
    selected = tp + fp
    precision = tp / max(1, selected)
    recall = tp / max(1, tp + fn)
    return {
        "frame_units": len(records), "candidate_rows": int(len(flat_label)),
        "positive_rows": int(flat_label.sum()), "positive_frame_units": positive_units,
        "roc_auc": auc_score(flat_score, flat_label), "pr_auc": average_precision(flat_score, flat_label),
        "precision": precision, "recall": recall,
        "top1_frame_recall": top1 / max(1, positive_units), "top5_frame_recall": top5 / max(1, positive_units),
        "false_positive_candidates_per_frame": fp / max(1, len(records)),
        "empty_output_rate": empty / max(1, len(records)),
        "null_frame_false_acceptance": null_false / max(1, len(records)),
        "predictions_per_positive": selected / max(1, int(flat_label.sum())),
        "multi_positive_frame_count": multi_units,
        "multi_positive_recall": multi_recall / max(1, multi_units),
        "strict_min_positive_margin": dist(strict), "best_positive_margin": dist(best),
        "average_positive_margin": dist(average),
        "hard_violation_rate": float(np.mean(violations)) if violations else None,
        "source_precision": {name: {"accepted": value[0], "true_positive": value[1],
                                     "precision": value[1] / max(1, value[0])}
                             for name, value in sorted(source_counts.items())},
        "null_max_score": dist(null_scores), "threshold": float(threshold),
    }


def checkpoint_paths():
    paths = sorted(TRAIN_ROOT.glob("checkpoint_l49_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
    if not paths:
        raise FileNotFoundError(f"no L49 checkpoints under {TRAIN_ROOT}")
    return paths


def make_records(model, checkpoint_step, units, store, text, device, stage):
    model.eval()
    by_video = defaultdict(list)
    for unit in units:
        by_video[(unit["dataset"], unit["video"])].append(unit)
    records = []
    for (dataset, video), values in sorted(by_video.items()):
        bank = store.get(dataset, video)
        for unit in values:
            values_cpu = unit_features(unit, bank, text, history=8)
            moved = {key: value.to(device, non_blocking=True) for key, value in values_cpu.items()
                     if key != "target"}
            with torch.inference_mode():
                output = model(moved["clip"], moved["history_clip"], moved["geometry"], moved["motion"],
                               moved["context"], moved["lifecycle"], moved["objectness"], moved["text"],
                               moved["text_mask"], moved["relation"], moved["history_sequence"],
                               moved["history_mask"], stage=stage)
            begin, end = int(unit["begin"]), int(unit["end"])
            label = values_cpu["target"].numpy().astype(bool)
            sources = source_masks(bank, begin, end)
            records.append({
                "dataset": dataset, "video": video, "query_id": int(unit["query_id"]),
                "expression": unit["expression"], "sentence": unit["sentence"],
                "frame_id": int(unit["frame_id"]), "category": unit["category"],
                "candidate_count": int(len(label)), "positive_count": int(label.sum()),
                "score": output["final_logit"].float().cpu().numpy(),
                "semantic_score": output["semantic_logit"].float().cpu().numpy(),
                "identity_score": output["identity_logit"].float().cpu().numpy(),
                "continuation_score": output["continuation_logit"].float().cpu().numpy(),
                "null_logit": float(output["null_logit"].float().cpu().item()),
                "label": label, "sources": sources,
            })
    return records


def jsonable_record(record):
    result = dict(record)
    for key in ("score", "semantic_score", "identity_score", "continuation_score"):
        result[key] = np.asarray(result[key], dtype=np.float32).tolist()
    result["label"] = np.asarray(result["label"], dtype=bool).tolist()
    result["sources"] = {key: np.asarray(value, dtype=bool).tolist() for key, value in result["sources"].items()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", default=str(OUT))
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    contract = json.loads(CONTRACT.read_text())
    if contract.get("decision") != "enter_B0":
        raise RuntimeError("L49 contract is not enter_B0")
    calibration = load_jsonl(DATA / "calibration_units.jsonl")
    validation = load_jsonl(DATA / "validation_units.jsonl")
    text = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
    device = torch.device(args.device if args.device != "cpu" or not torch.cuda.is_available() else "cpu")
    teacher = L29FrameMembershipSetDecoder().to(device)
    teacher.load_state_dict(torch.load(L29_CHECKPOINT, map_location=device, weights_only=False)["model"], strict=True)
    teacher.eval()
    store = BankStore(limit=2)
    teacher_cache = OrderedDict()
    def baseline_records(units):
        grouped = defaultdict(list)
        for unit in units:
            grouped[(unit["dataset"], unit["video"])].append(unit)
        result = []
        for (dataset, video), values in sorted(grouped.items()):
            bank = store.get(dataset, video)
            key = str(bank["path"])
            if key not in teacher_cache:
                teacher_cache[key] = build_teacher_cache(bank)
            cache = teacher_cache[key]
            for unit in values:
                score = l29_score(teacher, cache, bank, unit, text, device)
                label = np.zeros(int(unit["end"] - unit["begin"]), dtype=bool)
                label[np.asarray(unit["positive_indices"], dtype=np.int64)] = True
                result.append({"dataset": dataset, "video": video, "query_id": int(unit["query_id"]),
                               "expression": unit["expression"], "sentence": unit["sentence"],
                               "frame_id": int(unit["frame_id"]), "category": unit["category"],
                               "candidate_count": len(label), "positive_count": int(label.sum()),
                               "score": score, "semantic_score": score, "identity_score": np.zeros_like(score),
                               "continuation_score": np.zeros_like(score), "null_logit": 0.0,
                               "label": label, "sources": source_masks(bank, int(unit["begin"]), int(unit["end"]))})
        return result
    baseline_cal = baseline_records(calibration)
    baseline_val = baseline_records(validation)
    checkpoint_results = {}
    all_jsonl = []
    for checkpoint in checkpoint_paths():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = payload.get("model_config", {})
        model = L49KittiRMOT(hidden=int(cfg.get("hidden", 256)), heads=int(cfg.get("heads", 4)),
                             history_length=int(cfg.get("history_length", 8))).to(device)
        model.load_state_dict(payload["model"], strict=True)
        step = int(payload.get("checkpoint_step", checkpoint.stem.split("step")[-1]))
        stage = "semantic_warmup" if step <= int(payload.get("warmup_steps", 1000)) else "identity_continuation_null_sequence"
        cal_records = make_records(model, step, calibration, store, text, device, stage)
        val_records = make_records(model, step, validation, store, text, device, stage)
        threshold_by_domain = {}
        summary_by_domain = {}
        base_by_domain = {}
        for domain in ("refer_kitti_v1", "refer_kitti_v2"):
            cal_domain = [x for x in cal_records if x["dataset"] == domain]
            val_domain = [x for x in val_records if x["dataset"] == domain]
            base_domain = [x for x in baseline_val if x["dataset"] == domain]
            threshold = fit_threshold(cal_domain)
            threshold_by_domain[domain] = threshold
            summary_by_domain[domain] = summarize(val_domain, threshold["threshold"])
            base_threshold = fit_threshold([x for x in baseline_cal if x["dataset"] == domain])
            base_by_domain[domain] = summarize(base_domain, base_threshold["threshold"])
        del model, payload
        gc.collect()
        per_domain = {}
        for domain in summary_by_domain:
            new = summary_by_domain[domain]; base = base_by_domain[domain]
            per_domain[domain] = {
                "baseline": base, "l49_final": new,
                "delta": {"top1": new.get("top1_frame_recall", 0) - base.get("top1_frame_recall", 0),
                          "top5": new.get("top5_frame_recall", 0) - base.get("top5_frame_recall", 0),
                          "recall": new.get("recall", 0) - base.get("recall", 0),
                          "precision": new.get("precision", 0) - base.get("precision", 0),
                          "fp_per_frame": new.get("false_positive_candidates_per_frame", 0) - base.get("false_positive_candidates_per_frame", 0),
                          "hard_violation": new.get("hard_violation_rate") - base.get("hard_violation_rate") if new.get("hard_violation_rate") is not None and base.get("hard_violation_rate") is not None else None,
                          "multi_positive_recall": new.get("multi_positive_recall", 0) - base.get("multi_positive_recall", 0),
                          "empty": new.get("empty_output_rate", 0) - base.get("empty_output_rate", 0)},
            }
        macro_f1 = float(np.mean([2 * x["l49_final"]["precision"] * x["l49_final"]["recall"] /
                                  max(1e-9, x["l49_final"]["precision"] + x["l49_final"]["recall"])
                                  for x in per_domain.values()]))
        checkpoint_results[str(step)] = {
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "stage": stage, "thresholds": threshold_by_domain, "per_domain": per_domain,
            "validation_macro_f1": macro_f1,
        }
        for row in cal_records:
            all_jsonl.append({"split": "calibration", "checkpoint_step": step, **jsonable_record(row)})
        for row in val_records:
            all_jsonl.append({"split": "validation", "checkpoint_step": step, **jsonable_record(row)})
    gates = {}
    for step, result in checkpoint_results.items():
        flags = {}
        for domain, values in result["per_domain"].items():
            delta = values["delta"]
            flags[domain] = {
                "recall_not_substantially_lower": delta["recall"] >= -0.03,
                "multi_positive_drop_within_003": delta["multi_positive_recall"] >= -0.03,
                "hard_violation_clear_improvement": delta["hard_violation"] is not None and delta["hard_violation"] <= -0.05,
                "precision_not_collapsed": delta["precision"] >= -0.03,
                "not_empty_driven": values["l49_final"]["empty_output_rate"] < 0.90,
            }
        gates[step] = {"per_domain": flags,
                       "both_domains_pass": all(all(x.values()) for x in flags.values())}
    passing = [int(step) for step, gate in gates.items() if gate["both_domains_pass"]]
    selected_step = max(passing, key=lambda x: checkpoint_results[str(x)]["validation_macro_f1"]) if passing else max(
        (int(x) for x in checkpoint_results), key=lambda x: checkpoint_results[str(x)]["validation_macro_f1"])
    selected = checkpoint_results[str(selected_step)]
    selected_payload = {
        "format": "locatemot-l49-validation-selection-v1", "stage": "C1-validation",
        "selected_step": selected_step, "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selection_rule": "highest validation macro candidate F1 among checkpoints passing both-domain gate; otherwise preregistered highest validation macro candidate F1",
        "validation_gate": "pass" if passing else "failed",
        "passing_steps": passing, "selected_thresholds": selected["thresholds"],
        "calibration_labels_only": True, "validation_labels_for_selection": True,
        "official_test_labels_read": False, "screening_gt_read": False,
        "per_domain": selected["per_domain"], "all_gates": gates,
    }
    (out / "validation_metrics.json").write_text(json.dumps({
        "format": "locatemot-l49-validation-metrics-v1", "completed_at_unix": time.time(),
        "checkpoint_results": checkpoint_results, "gates": gates,
        "data_contract_sha256": sha256_file(CONTRACT), "text_cache_sha256": sha256_file(TEXT_CACHE),
        "official_test_labels_read": False, "screening_gt_read": False,
    }, indent=2) + "\n")
    (out / "calibration.json").write_text(json.dumps({
        "format": "locatemot-l49-calibration-v1", "per_checkpoint": {
            step: result["thresholds"] for step, result in checkpoint_results.items()},
        "selected_step": selected_step, "selected_thresholds": selected["thresholds"],
        "labels_source": "L49 calibration videos only", "validation_or_test_labels_used": False,
    }, indent=2) + "\n")
    (out / "selected_checkpoint.json").write_text(json.dumps(selected_payload, indent=2) + "\n")
    with (out / "validation_scores.jsonl").open("w") as handle:
        for row in all_jsonl:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "validation_run_provenance.json").write_text(json.dumps({
        "format": "locatemot-l49-validation-provenance-v1", "started_at_unix": started,
        "completed_at_unix": time.time(), "checkpoint_count": len(checkpoint_results),
        "calibration_units": len(calibration), "validation_units": len(validation),
        "data_contract_sha256": sha256_file(CONTRACT), "text_cache_sha256": sha256_file(TEXT_CACHE),
        "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT), "device": str(device),
        "official_test_labels_read": False, "screening_gt_read": False,
        "ordinary_mot_ovmot_touched": False,
    }, indent=2) + "\n")
    print(json.dumps({"selected_step": selected_step, "validation_gate": selected_payload["validation_gate"],
                      "passing_steps": passing, "checkpoint_count": len(checkpoint_results),
                      "elapsed_sec": time.time() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
