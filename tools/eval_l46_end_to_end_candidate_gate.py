#!/usr/bin/env python3
"""L46 B1 fixed-100-unit candidate gate.

Only the fixed screening slice is replayed here.  The single threshold is the
already-frozen L29 calibration threshold from the L44/L29 contract; no L46 or
screening labels are used to choose a threshold, unit, checkpoint, or branch.
The immutable L27 cache supplies the candidate rows/labels for final
screening statistics, while the L46 model is run on the corresponding full
L19 candidate set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.models.l46_end_to_end_query_region_track import (
    L46EndToEndQueryRegionTrackDecoder,
)
from tools.audit_l44_integrated_contract import L19, L28, L29, FAST, V5
from tools.audit_l29_emission_contract import build_cache as build_l19_sequence_cache
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import load_expressions
from tools.train_l28_track_set_decoder import state_at
from tools.train_l42_current_frame_grounding import numeric_for

SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"
L44_REFERENCE = ROOT / "outputs/l44/eval/candidate_gate_100_retry2.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_bank(video: str):
    blob = torch.load(L19 / f"{video}.pt", map_location="cpu", weights_only=False)
    t = blob["tensors"]
    return {
        "clip": t["clip"].float(), "history_clip": t["history_clip"].float(),
        "geometry": t["geometry"].float(), "motion": t["motion"].float(),
        "context": t["context"].float(), "lifecycle": t["lifecycle"].float(),
        "objectness": t["objectness"].float(), "box": t["box"].float(),
        "track": t["track_id"].long(), "pool": t["pool_id"].long(),
        "frame_ids": t["frame_ids"].long(), "ptr": t["frame_ptr"].long(),
    }


def load_text(entries):
    manifest = json.loads((V5 / "text_manifest.json").read_text())["expressions"]
    text_index = {(str(x["video"]), str(x["expression"])): int(x["query_index"])
                  for x in manifest}
    needed = sorted({text_index[(str(e["video"]), str(e["expression"]))]
                    for e in entries})
    blob = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    hidden = blob["token_hidden"][needed].float()
    mask = blob["attention_mask"][needed].bool()
    return text_index, hidden, mask, {old: i for i, old in enumerate(needed)}


def valid_track_indices(cache, cutoff):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def history_for(cache, bank, rows, frame, history_len=8):
    feature = cache["obs_features"]
    ptr = cache["track_ptr"].tolist()
    frames = cache["obs_frame"].tolist()
    track_to_index = {int(t): i for i, t in enumerate(cache["track_ids"].tolist())}
    values = torch.zeros((len(rows), history_len, int(feature.shape[1])), dtype=torch.float32)
    mask = torch.zeros((len(rows), history_len), dtype=torch.bool)
    times = torch.zeros((len(rows), history_len), dtype=torch.float32)
    for i, row in enumerate(rows):
        ti = track_to_index.get(int(bank["track"][row]))
        if ti is None:
            continue
        begin, end = int(ptr[ti]), int(ptr[ti + 1])
        eligible = [j for j in range(begin, end) if int(frames[j]) <= int(frame)]
        chosen = eligible[-history_len:]
        offset = history_len - len(chosen)
        if not chosen:
            continue
        values[i, offset:] = feature[torch.as_tensor(chosen)].float()
        mask[i, offset:] = True
        times[i, offset:] = torch.as_tensor(
            np.asarray([frames[j] for j in chosen], dtype=np.float32) /
            max(1.0, float(frame) + 1.0))
    return values, mask, times


def metric_distribution(values):
    if not values:
        return {"count": 0, "mean": None, "median": None}
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(len(x)), "mean": float(x.mean()),
            "median": float(np.median(x)), "q10": float(np.quantile(x, .1)),
            "q90": float(np.quantile(x, .9))}


def metric_summary(records, threshold):
    tp = fp = fn = selected = empty = null_accept = 0
    top1, top5, strict, best, average, fp_frame = [], [], [], [], [], []
    multi_recall = []
    source = {"main": [0, 0, 0], "reserve": [0, 0, 0]}
    query_sequences = defaultdict(list)
    for record in records:
        y = np.asarray(record["label"], dtype=bool)
        score = np.asarray(record["score"], dtype=np.float64)
        chosen = score >= float(threshold)
        tp += int((chosen & y).sum()); fp += int((chosen & ~y).sum())
        fn += int((~chosen & y).sum()); selected += int(chosen.sum())
        empty += int(not chosen.any()); null_accept += int(not y.any() and chosen.any())
        fp_frame.append(int((chosen & ~y).sum()))
        query_sequences[int(record["query_index"])].append(
            (int(record["frame"]), set(record["track_id"][chosen].tolist())))
        if y.any():
            order = np.argsort(-score, kind="stable")
            top1.append(float(y[order[:1]].any())); top5.append(float(y[order[:5]].any()))
            neg = score[~y]; pos = score[y]
            if len(neg):
                strict.append(float(pos.min() - neg.max()))
                best.append(float(pos.max() - neg.max()))
                average.append(float(pos.mean() - neg.max()))
            if y.sum() > 1:
                multi_recall.append(float((chosen & y).sum() / max(1, int(y.sum()))))
        for sid, name in ((0, "main"), (1, "reserve")):
            pool = np.asarray(record["source"]) == sid
            source[name][0] += int((chosen & pool).sum())
            source[name][1] += int((y & pool).sum())
            source[name][2] += int((chosen & pool & y).sum())
    switches = 0
    for sequence in query_sequences.values():
        sequence.sort(); previous = set()
        for _, current in sequence:
            if previous and current and current != previous:
                switches += 1
            previous = current
    source_metrics = {
        name: {"selected": vals[0], "positive": vals[1], "true_positive": vals[2],
               "precision": vals[2] / max(1, vals[0]),
               "recall": vals[2] / max(1, vals[1])}
        for name, vals in source.items()
    }
    positives = int(sum(np.asarray(r["label"], dtype=bool).sum() for r in records))
    return {
        "frame_units": int(len(records)),
        "candidate_rows": int(sum(len(r["label"]) for r in records)),
        "positive_rows": positives, "selected": selected, "tp": tp, "fp": fp,
        "fn": fn, "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "top1_frame_recall": float(np.mean(top1)) if top1 else None,
        "top5_frame_recall": float(np.mean(top5)) if top5 else None,
        "strict_min_positive_margin": metric_distribution(strict),
        "best_positive_margin": metric_distribution(best),
        "average_positive_margin": metric_distribution(average),
        "hard_violation_rate": float(np.mean(np.asarray(strict) < 0)) if strict else None,
        "multi_positive_frame_count": int(len(multi_recall)),
        "multi_positive_recall": float(np.mean(multi_recall)) if multi_recall else None,
        "false_positive_candidates_per_frame": float(np.mean(fp_frame)) if fp_frame else None,
        "empty_output_rate": empty / max(1, len(records)),
        "null_frame_false_acceptance": null_accept / max(1, len(records)),
        "predictions_per_gt_positive": selected / max(1, positives),
        "source_precision": source_metrics, "identity_switch_proxy": int(switches),
    }


def fixed_teacher_hard(y, objectness, teacher, limit=24, prelimit=96):
    neg = np.flatnonzero(~np.asarray(y, dtype=bool))
    if not len(neg):
        return np.empty(0, dtype=np.int64)
    pre = neg[np.argsort(-np.asarray(objectness)[neg], kind="stable")[:prelimit]]
    return pre[np.argsort(-np.asarray(teacher)[pre], kind="stable")[:limit]]


def rank_stats(teacher, student, y, objectness):
    pos = np.flatnonzero(y)
    hard = fixed_teacher_hard(y, objectness, teacher)
    if not len(pos) or not len(hard):
        return {"pairs": 0, "teacher_correct": 0, "teacher_error": 0,
                "teacher_correct_flips": 0, "teacher_error_corrections": 0}
    td = teacher[pos, None] - teacher[hard][None, :]
    sd = student[pos, None] - student[hard][None, :]
    correct = td > 0
    return {
        "pairs": int(td.size), "teacher_correct": int(correct.sum()),
        "teacher_error": int((~correct).sum()),
        "teacher_correct_flips": int((correct & (sd < 0)).sum()),
        "teacher_error_corrections": int((~correct & (sd > 0)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="outputs/l46/eval/candidate_gate_100.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cap", type=int, default=100)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()}")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    entries = make_entries()
    arrays = load_caches(SCORE_ROOT, entries, ("A_C1_S2000",))["A_C1_S2000"]
    available = []
    for entry in entries:
        if entry["split"] != "screening":
            continue
        for frame, _ in frame_groups(arrays[(entry["video"], entry["expression"])]):
            available.append((str(entry["video"]), str(entry["expression"]), int(frame)))
    available.sort()
    chosen_count = min(args.cap, len(available))
    selected = {available[i] for i in np.linspace(0, len(available) - 1,
                                                   chosen_count, dtype=int)}
    if len(selected) != chosen_count:
        raise AssertionError("fixed screening selection is not unique")
    needed = []
    for entry in entries:
        if entry["split"] != "screening":
            continue
        data = arrays[(entry["video"], entry["expression"])]
        for frame, _ in frame_groups(data):
            unit = (str(entry["video"]), str(entry["expression"]), int(frame))
            if unit in selected:
                needed.append((unit, entry))
    by_frame = defaultdict(list)
    for unit, entry in needed:
        by_frame[(unit[0], unit[2])].append((unit, entry))
    text_index, text_hidden, text_mask, text_remap = load_text(entries)
    videos = sorted({str(entry["video"]) for _, entry in needed})
    banks = {video: load_bank(video) for video in videos}
    caches = {}
    cache_sources = {}
    for video in videos:
        path = L28 / f"{video}.pt"
        if path.exists():
            caches[video] = torch.load(path, map_location="cpu", weights_only=False)
            cache_sources[video] = {"path": str(path.resolve()), "kind": "L28_persistent_cache"}
        else:
            caches[video] = build_l19_sequence_cache(video)
            cache_sources[video] = {"path": str((L19 / f"{video}.pt").resolve()),
                                    "kind": "L19_read_only_in_memory_sequence_replay"}

    device = torch.device(args.device)
    teacher_model = L29FrameMembershipSetDecoder().to(device)
    teacher_model.load_state_dict(torch.load(L29, map_location=device,
                                              weights_only=False)["model"], strict=True)
    teacher_model.eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model_config = state["config"]["model_config"]
    model = L46EndToEndQueryRegionTrackDecoder(**model_config).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    records = {"teacher": [], "l46": []}
    raw = {"physical_frame_units": 0, "expression_frame_units": 0,
           "candidate_rows": 0, "missing_bank_rows": 0,
           "missing_teacher_tracks": 0, "nonfinite_scores": 0,
           "text_region_entropy": [], "set_entropy": []}
    rank_total = Counter()
    for (video, frame), unit_entries in sorted(by_frame.items()):
        bank = banks[video]; cache = caches[video]
        frame_index = {int(x): i for i, x in enumerate(bank["frame_ids"].tolist())}
        if int(frame) not in frame_index:
            raw["missing_bank_rows"] += 1
            continue
        fi = frame_index[int(frame)]
        begin, end = int(bank["ptr"][fi]), int(bank["ptr"][fi + 1])
        rows = list(range(begin, end))
        bank_tracks = bank["track"][rows].numpy().astype(np.int64)
        if len(np.unique(bank_tracks)) != len(bank_tracks):
            raise RuntimeError(f"duplicate current-frame track ids {video}/{frame}")
        history, history_mask, history_time = history_for(cache, bank, rows, frame, 8)
        obs, obs_mask, obs_time, _, _ = state_at(cache, frame, history=8)
        valid = valid_track_indices(cache, frame)
        valid_ids = cache["track_ids"][torch.as_tensor(valid)].numpy().astype(np.int64)
        with torch.inference_mode():
            encoded = teacher_model.encode_observations(
                obs.to(device), obs_mask.to(device), obs_time.to(device))
        raw["physical_frame_units"] += 1; raw["candidate_rows"] += len(rows)
        for unit, entry in unit_entries:
            data = arrays[(entry["video"], entry["expression"])]
            idx = np.flatnonzero(data["frame"] == int(frame))
            cache_tracks = data["track_id"][idx].astype(np.int64)
            cache_pos = {int(track): int(i) for i, track in enumerate(cache_tracks)}
            if set(cache_pos) != set(bank_tracks):
                raw["missing_bank_rows"] += abs(len(set(cache_pos) ^ set(bank_tracks)))
                continue
            aligned = np.asarray([cache_pos[int(track)] for track in bank_tracks], dtype=np.int64)
            y = data["label"][idx][aligned].astype(bool)
            source = data["source"][idx][aligned].astype(np.int8)
            old_text = text_index[(str(entry["video"]), str(entry["expression"]))]
            ti = text_remap[old_text]
            qh, qm = text_hidden[ti].to(device), text_mask[ti].to(device)
            with torch.inference_mode():
                teacher_out = teacher_model.forward_encoded(encoded, encoded[1], qh, qm)
                teacher_map = {int(track): float(score) for track, score in
                               zip(valid_ids, teacher_out["current_membership_logits"].float().cpu().tolist())}
                teacher = np.asarray([teacher_map.get(int(track), -20.0)
                                      for track in bank_tracks], dtype=np.float32)
                raw["missing_teacher_tracks"] += sum(int(track not in teacher_map) for track in bank_tracks)
                model_out = model(
                    bank["clip"][rows].float().unsqueeze(1).to(device), qh,
                    numeric_for(bank, rows).float().to(device), history.to(device),
                    history_mask.to(device), history_time.to(device),
                    candidate_mask=torch.ones(len(rows), dtype=torch.bool, device=device),
                    text_mask=qm, teacher=torch.from_numpy(teacher).to(device),
                )
                score = model_out["membership_logits"].float().cpu().numpy()
            if not np.isfinite(teacher).all() or not np.isfinite(score).all():
                raw["nonfinite_scores"] += 1
                continue
            raw["text_region_entropy"].append(float(model_out["text_region_attention_entropy"]))
            raw["set_entropy"].append(float(model_out["set_attention_entropy"]))
            base = {"video": video, "expression": str(entry["expression"]),
                    "query_index": int(entry["query_index"]), "frame": int(frame),
                    "track_id": bank_tracks, "label": y, "source": source}
            records["teacher"].append({**base, "score": teacher})
            records["l46"].append({**base, "score": score})
            rank = rank_stats(teacher, score, y,
                              bank["objectness"][rows].numpy())
            for key, value in rank.items():
                rank_total[key] += int(value)
            raw["expression_frame_units"] += 1
    if raw["missing_bank_rows"] or raw["missing_teacher_tracks"]:
        raise RuntimeError(
            f"L46 B1 alignment incomplete: missing_bank_rows={raw['missing_bank_rows']} "
            f"missing_teacher_tracks={raw['missing_teacher_tracks']}")
    if raw["nonfinite_scores"]:
        raise FloatingPointError(f"nonfinite L46 scores in {raw['nonfinite_scores']} units")

    if not L44_REFERENCE.exists():
        raise FileNotFoundError(L44_REFERENCE)
    ref = json.loads(L44_REFERENCE.read_text())
    threshold = float(ref["calibration"]["threshold"])
    threshold_contract = {
        "threshold": threshold,
        "source": str(L44_REFERENCE.resolve()),
        "source_sha256": sha256(L44_REFERENCE),
        "source_semantics": "frozen L29 calibration-only balanced-F1 threshold",
        "screening_gt_used_for_threshold": False,
        "new_threshold_search": False,
    }
    metrics = {name: metric_summary(value, threshold)
               for name, value in records.items()}
    reference_integrated = ref["strategies"].get("integrated", {})
    reference_l44 = {
        "source": str(L44_REFERENCE.resolve()),
        "checkpoint": ref.get("checkpoint"),
        "screening": reference_integrated.get("screening"),
        "calibration": reference_integrated.get("calibration"),
    }
    teacher = metrics["teacher"]; student = metrics["l46"]
    gate = {
        "top1_delta": student["top1_frame_recall"] - teacher["top1_frame_recall"],
        "recall_delta": student["recall"] - teacher["recall"],
        "hard_violation_delta": student["hard_violation_rate"] - teacher["hard_violation_rate"],
        "precision_delta": student["precision"] - teacher["precision"],
        "fp_frame_delta": student["false_positive_candidates_per_frame"] - teacher["false_positive_candidates_per_frame"],
        "multi_positive_recall_delta": student["multi_positive_recall"] - teacher["multi_positive_recall"],
        "top1_preserved_within_0.02": student["top1_frame_recall"] >= teacher["top1_frame_recall"] - .02,
        "recall_preserved_within_0.03": student["recall"] >= teacher["recall"] - .03,
        "hard_violation_improved_by_0.05": student["hard_violation_rate"] <= teacher["hard_violation_rate"] - .05,
        "precision_preserved_within_0.01": student["precision"] >= teacher["precision"] - .01,
        "fp_frame_preserved_within_0.10": student["false_positive_candidates_per_frame"] <= teacher["false_positive_candidates_per_frame"] + .10,
        "multi_positive_preserved_within_0.03": student["multi_positive_recall"] >= teacher["multi_positive_recall"] - .03,
        "not_empty_output_driven": student["empty_output_rate"] <= teacher["empty_output_rate"] + .05,
        "not_null_collapse_driven": student["null_frame_false_acceptance"] <= teacher["null_frame_false_acceptance"] + .05,
    }
    gate_keys = tuple(key for key in gate if key.startswith(("top1_preserved", "recall_preserved",
                                                              "hard_violation_improved", "precision_preserved",
                                                              "fp_frame_preserved", "multi_positive_preserved",
                                                              "not_empty", "not_null")))
    payload = {
        "format": "locatemot-l46-end-to-end-query-region-track-candidate-gate-v1",
        "stage": "L46-B1-fixed-100-screening-frame-units",
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "teacher_checkpoint": str(L29.resolve()), "teacher_checkpoint_sha256": sha256(L29),
        "manifest": str(FAST.resolve()), "manifest_sha256": sha256(FAST),
        "score_cache": str(SCORE_ROOT.resolve()),
        "score_cache_model": "A_C1_S2000 immutable rows/labels for fixed-unit final statistics",
        "selected_screening_units": sorted(selected),
        "selected_screening_units_sha256": json_sha(sorted(selected)),
        "counts": {"manifest_queries": len(entries), "calibration_queries": 64,
                   "screening_queries": 96, "screening_units_available": len(available),
                   "screening_units_selected": len(selected),
                   "screening_expression_frame_units": raw["expression_frame_units"]},
        "threshold_contract": threshold_contract,
        "screening_gt_used_for_threshold": False,
        "screening_gt_used_for_model_or_checkpoint_selection": False,
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key"],
        "token_span_region_alignment": "UNALIGNED; no verified token/span boxes",
        "motion_language_decomposition": "not claimed; no verified motion-language mask",
        "cache_sources": cache_sources,
        "raw_replay": {**raw, "text_region_entropy_mean": float(np.mean(raw["text_region_entropy"])) if raw["text_region_entropy"] else None,
                       "set_entropy_mean": float(np.mean(raw["set_entropy"])) if raw["set_entropy"] else None},
        "strategies": {"l29_teacher": {"metrics": teacher, "threshold": threshold},
                       "l46_full": {"metrics": student, "threshold": threshold},
                       "l44_integrated_reference": reference_l44},
        "rank_flip_diagnostics": {
            **dict(rank_total),
            "teacher_correct_flip_ratio": rank_total["teacher_correct_flips"] / max(1, rank_total["teacher_correct"]),
            "teacher_error_correction_ratio": rank_total["teacher_error_corrections"] / max(1, rank_total["teacher_error"]),
        },
        "gates_relative_to_l29_teacher": gate,
        "decision": "pass" if all(gate[key] for key in gate_keys) else "fail",
    }
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (out.parent / "README.md").write_text(
        "# L46 B1 fixed 100-unit candidate gate\n\n"
        "The L29 calibration threshold is frozen; the 100 screening units are "
        "selected by the immutable L27 cache protocol. Screening labels are "
        "used only for final reporting.\n")
    print(json.dumps({"out": str(out), "decision": payload["decision"],
                      "teacher": teacher, "l46": student,
                      "gate": gate, "rank_flip": payload["rank_flip_diagnostics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
