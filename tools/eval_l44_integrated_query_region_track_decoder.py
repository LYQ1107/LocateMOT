#!/usr/bin/env python3
"""L44 B1 candidate gate on the fixed 100-unit held-out slice.

The evaluator keeps the L29 current-membership logit as an explicit control
and evaluates the L44 integrated decoder on the same complete per-frame
candidate sets.  One threshold is fitted from calibration labels of the
teacher only, then reused unchanged for teacher, zero-residual, and L44
outputs.  Screening labels are consumed only after the choices are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.models.l44_integrated_query_region_track_decoder import (
    L44IntegratedQueryRegionTrackDecoder,
)
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import V5
from tools.train_l28_track_set_decoder import state_at
from tools.train_l42_current_frame_grounding import (
    StreamingCropPatchEncoder,
    numeric_for,
)
from tools.train_l44_integrated_query_region_track_decoder import history_for
from tools.audit_l44_integrated_contract import L19, L28, L29, FAST
from tools.audit_l29_emission_contract import build_cache as build_l19_sequence_cache
from tools.l40_raw_data import WEIGHTS

SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_bank(video: str):
    """Load only frozen L19 observation fields; no label sidecar is read."""
    d = torch.load(L19 / f"{video}.pt", map_location="cpu", weights_only=False)
    t = d["tensors"]
    return {
        "box": t["box"].float(), "frame": t["frame"].long(),
        "track": t["track_id"].long(), "objectness": t["objectness"].float(),
        "geometry": t["geometry"].float(), "motion": t["motion"].float(),
        "context": t["context"].float(), "lifecycle": t["lifecycle"].float(),
        "history_clip": t["history_clip"].float(),
        "frame_ids": t["frame_ids"].long(), "ptr": t["frame_ptr"].long(),
    }


def load_text(entries):
    manifest = json.loads((V5 / "text_manifest.json").read_text())["expressions"]
    index = {(str(x["video"]), str(x["expression"])): int(x["query_index"])
             for x in manifest}
    needed = sorted({index[(str(e["video"]), str(e["expression"]))]
                     for e in entries})
    text = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    hidden = text["token_hidden"][needed].float()
    mask = text["attention_mask"][needed].bool()
    remap = {old: i for i, old in enumerate(needed)}
    del text
    return index, hidden, mask, remap


def valid_track_indices(cache, cutoff: int):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def choose_threshold(records):
    values = np.concatenate([x["score"] for x in records if len(x["score"])])
    labels = np.concatenate([x["label"].astype(bool)
                             for x in records if len(x["label"])])
    if not len(values) or not labels.any():
        raise RuntimeError("calibration teacher records have no usable labels")
    best = None
    candidates = np.unique(np.quantile(values, np.linspace(.01, .995, 160)))
    for threshold in candidates:
        chosen = values >= float(threshold)
        tp = int((chosen & labels).sum())
        fp = int((chosen & ~labels).sum())
        fn = int((~chosen & labels).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        item = (f1, precision, recall, -float(threshold), float(threshold),
                tp, fp, fn)
        if best is None or item > best:
            best = item
    return {
        "threshold": best[4], "source": "single_L29_teacher_calibration_only_balanced_F1",
        "precision": best[1], "recall": best[2], "f1": best[0],
        "tp": best[5], "fp": best[6], "fn": best[7],
        "calibration_rows": int(len(labels)),
        "calibration_positive_rows": int(labels.sum()),
    }


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(len(x)), "mean": float(x.mean()),
            "median": float(np.median(x)), "max": float(x.max())}


def metric_summary(records, threshold):
    tp = fp = fn = selected = empty = null_accept = 0
    top1, top5, strict, best, average = [], [], [], [], []
    multi_recall, fp_frame = [], []
    transitions = defaultdict(list)
    source = {"main": [0, 0, 0], "reserve": [0, 0, 0]}
    for record in records:
        y = np.asarray(record["label"], dtype=bool)
        score = np.asarray(record["score"], dtype=np.float64)
        chosen = score >= float(threshold)
        tp += int((chosen & y).sum()); fp += int((chosen & ~y).sum())
        fn += int((~chosen & y).sum()); selected += int(chosen.sum())
        empty += int(not chosen.any())
        null_accept += int(not y.any() and chosen.any())
        fp_frame.append(int((chosen & ~y).sum()))
        transitions[int(record["query_index"])].append(
            (int(record["frame"]), set(record["track_id"][chosen].tolist())))
        order = np.argsort(-score, kind="stable")
        if y.any():
            top1.append(float(y[order[:1]].any()))
            top5.append(float(y[order[:5]].any()))
            pos, neg = score[y], score[~y]
            if len(neg):
                strict.append(float(pos.min() - neg.max()))
                best.append(float(pos.max() - neg.max()))
                average.append(float(pos.mean() - neg.max()))
            if y.sum() > 1:
                multi_recall.append(float((chosen & y).sum() / max(1, int(y.sum()))))
        for sid, name in ((0, "main"), (1, "reserve")):
            mask = np.asarray(record["source"]) == sid
            source[name][0] += int((chosen & mask).sum())
            source[name][1] += int((y & mask).sum())
            source[name][2] += int((chosen & mask & y).sum())

    switches = 0
    for sequence in transitions.values():
        sequence.sort()
        previous = set()
        for _, current in sequence:
            if previous and current and current != previous:
                switches += 1
            previous = current
    source_metrics = {
        name: {"selected": vals[0], "positive": vals[1],
               "true_positive": vals[2],
               "precision": vals[2] / max(1, vals[0]),
               "recall": vals[2] / max(1, vals[1])}
        for name, vals in source.items()
    }
    pos_total = int(sum(np.asarray(x["label"], dtype=bool).sum()
                        for x in records))
    return {
        "frame_units": int(len(records)),
        "candidate_rows": int(sum(len(x["label"]) for x in records)),
        "positive_rows": pos_total, "selected": selected, "tp": tp,
        "fp": fp, "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "top1_frame_recall": float(np.mean(top1)) if top1 else None,
        "top5_frame_recall": float(np.mean(top5)) if top5 else None,
        "strict_min_positive_margin": distribution(strict),
        "best_positive_margin": distribution(best),
        "average_positive_margin": distribution(average),
        "hard_violation_rate": float(np.mean(np.asarray(strict) < 0)) if strict else None,
        "multi_positive_frame_count": int(len(multi_recall)),
        "multi_positive_recall": float(np.mean(multi_recall)) if multi_recall else None,
        "false_positive_candidates_per_frame": float(np.mean(fp_frame)) if fp_frame else None,
        "empty_output_rate": empty / max(1, len(records)),
        "null_frame_false_acceptance": null_accept / max(1, len(records)),
        "predictions_per_gt_positive": selected / max(1, pos_total),
        "source_precision": source_metrics,
        "identity_switch_proxy": int(switches),
    }


def fixed_teacher_hard(y, objectness, teacher, limit=24, prelimit=96):
    negative = np.flatnonzero(~np.asarray(y, dtype=bool))
    if not len(negative):
        return np.empty(0, dtype=np.int64)
    pre = negative[np.argsort(-np.asarray(objectness)[negative], kind="stable")[:prelimit]]
    return pre[np.argsort(-np.asarray(teacher)[pre], kind="stable")[:limit]]


def rank_stats(teacher, student, y, hard):
    pos = np.flatnonzero(y)
    hard = np.asarray(hard, dtype=np.int64)
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


def add_counts(total, item):
    for key, value in item.items():
        total[key] = total.get(key, 0) + int(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="outputs/l44/eval/candidate_gate_100.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cap", type=int, default=100)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()}")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    entries = make_entries()
    arrays = load_caches(SCORE_ROOT, entries, ("A_C1_S2000",))["A_C1_S2000"]
    text_index, hidden, text_mask, text_remap = load_text(entries)

    available = []
    for entry in entries:
        if entry["split"] != "screening":
            continue
        data = arrays[(entry["video"], entry["expression"])]
        available.extend((str(entry["video"]), str(entry["expression"]), int(frame))
                        for frame, _ in frame_groups(data))
    available.sort()
    if not available:
        raise RuntimeError("no screening frame units in immutable L27 cache")
    selected_screen = {
        available[i] for i in np.linspace(0, len(available) - 1,
                                          min(args.cap, len(available)), dtype=int)
    }
    if len(selected_screen) != min(args.cap, len(available)):
        raise AssertionError("selected screening units are not unique")

    needed = []
    for entry in entries:
        data = arrays[(entry["video"], entry["expression"])]
        for frame, _ in frame_groups(data):
            unit = (str(entry["video"]), str(entry["expression"]), int(frame))
            if entry["split"] == "calibration" or unit in selected_screen:
                needed.append((unit, entry))
    by_frame = defaultdict(list)
    for unit, entry in needed:
        by_frame[(unit[0], unit[2])].append((unit, entry))

    videos = sorted({str(entry["video"]) for _, entry in needed})
    banks = {video: load_bank(video) for video in videos}
    caches = {}
    cache_sources = {}
    for video in videos:
        cache_path = L28 / f"{video}.pt"
        if cache_path.exists():
            caches[video] = torch.load(cache_path, map_location="cpu", weights_only=False)
            cache_sources[video] = {"path": str(cache_path.resolve()), "kind": "L28_persistent_cache"}
        else:
            # L28's persistent cache is train-side only.  For screening videos
            # use the existing read-only L19 sequence builder in RAM; it reads
            # frozen bank tensors and never writes a replacement cache.
            caches[video] = build_l19_sequence_cache(video)
            cache_sources[video] = {
                "path": str((L19 / f"{video}.pt").resolve()),
                "kind": "L19_read_only_in_memory_sequence_replay",
            }

    device = torch.device(args.device)
    teacher_model = L29FrameMembershipSetDecoder().to(device)
    teacher_model.load_state_dict(
        torch.load(L29, map_location=device, weights_only=False)["model"], strict=True)
    teacher_model.eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model_config = state["config"]["model_config"]
    model = L44IntegratedQueryRegionTrackDecoder(**model_config).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    del state
    encoder = StreamingCropPatchEncoder(device, batch_size=32)

    records = {
        "calibration": {"teacher": [], "zero_teacher": [], "integrated": []},
        "screening": {"teacher": [], "zero_teacher": [], "integrated": []},
    }
    raw = {
        "physical_frame_units": 0, "expression_frame_units": 0,
        "candidate_rows": 0, "candidate_crops": 0,
        "missing_cache_rows": 0, "missing_teacher_tracks": 0,
        "nonfinite_scores": 0, "teacher_integrated_exact_zero_count": 0,
        "teacher_rank_pairs": 0, "teacher_hard_pair_rows": 0,
        "residual_max_abs": 0.0, "residual_sum": 0.0, "residual_count": 0,
    }
    rank_total = {"pairs": 0, "teacher_correct": 0, "teacher_error": 0,
                  "teacher_correct_flips": 0, "teacher_error_corrections": 0}

    for (video, frame), unit_entries in sorted(by_frame.items()):
        bank = banks[video]
        frame_to_index = {int(value): index
                          for index, value in enumerate(bank["frame_ids"].tolist())}
        if int(frame) not in frame_to_index:
            raw["missing_cache_rows"] += 1
            continue
        fi = frame_to_index[int(frame)]
        begin, end = int(bank["ptr"][fi]), int(bank["ptr"][fi + 1])
        rows = list(range(begin, end))
        bank_tracks = bank["track"][rows].numpy().astype(np.int64)
        if len(np.unique(bank_tracks)) != len(bank_tracks):
            raise RuntimeError(f"duplicate current-frame track IDs: {video}/{frame}")
        patches = encoder.encode(video, bank, rows)
        numeric = numeric_for(bank, rows)
        stub_query = unit_entries[0][1]
        history, history_mask, history_time, _, _ = history_for(
            caches[video], bank, rows, int(frame),
            {"target": {}}, history_len=int(model_config["history_len"]),
        )
        obs, obs_mask, obs_time, _, _ = state_at(
            caches[video], int(frame), history=int(model_config["history_len"]))
        valid = valid_track_indices(caches[video], int(frame))
        valid_ids = caches[video]["track_ids"][torch.as_tensor(valid)].numpy().astype(np.int64)
        with torch.inference_mode():
            encoded = teacher_model.encode_observations(
                obs.to(device), obs_mask.to(device), obs_time.to(device))
        raw["physical_frame_units"] += 1
        raw["candidate_rows"] += len(rows)
        raw["candidate_crops"] += len(rows)

        # Each expression sees exactly the same physical-frame candidate set;
        # only its token sequence and label view changes.
        for unit, entry in unit_entries:
            data = arrays[(entry["video"], entry["expression"])]
            idx = np.flatnonzero(data["frame"] == int(frame))
            cache_tracks = data["track_id"][idx].astype(np.int64)
            cache_pos = {int(track): int(i) for i, track in enumerate(cache_tracks)}
            if set(cache_pos) != set(bank_tracks):
                raw["missing_cache_rows"] += abs(len(set(cache_pos) ^ set(bank_tracks)))
                continue
            aligned = np.asarray([cache_pos[int(track)] for track in bank_tracks], dtype=np.int64)
            labels = data["label"][idx][aligned].astype(bool)
            sources = data["source"][idx][aligned].astype(np.int8)
            tracks = bank_tracks.copy()
            old_index = text_index[(str(entry["video"]), str(entry["expression"]))]
            text_row = text_remap[old_index]
            qh = hidden[text_row].to(device)
            qm = text_mask[text_row].to(device)

            with torch.inference_mode():
                teacher_out = teacher_model.forward_encoded(
                    encoded, encoded[1], qh, qm)
                teacher_map = {
                    int(track): float(score)
                    for track, score in zip(valid_ids,
                                            teacher_out["current_membership_logits"].float().cpu().tolist())
                }
                teacher = np.asarray(
                    [teacher_map.get(int(track), -20.0) for track in tracks],
                    dtype=np.float32,
                )
                raw["missing_teacher_tracks"] += sum(
                    int(int(track) not in teacher_map) for track in tracks
                )
                out = model(
                    patches.to(device).float(), qh,
                    numeric.to(device).float(), history.to(device).float(),
                    history_mask.to(device), history_time.to(device).float(),
                    torch.from_numpy(teacher).to(device),
                    torch.ones(len(rows), dtype=torch.bool, device=device), qm,
                )
                integrated = out["final_membership_logits"].float().cpu().numpy()
                residual = out["residual"].float().cpu().numpy()
            if not np.isfinite(integrated).all() or not np.isfinite(residual).all():
                raw["nonfinite_scores"] += 1
                continue
            raw["residual_max_abs"] = max(raw["residual_max_abs"],
                                           float(np.abs(residual).max(initial=0.0)))
            raw["residual_sum"] += float(residual.sum())
            raw["residual_count"] += int(residual.size)
            if np.array_equal(integrated, teacher):
                raw["teacher_integrated_exact_zero_count"] += 1
            hard = fixed_teacher_hard(
                labels, bank["objectness"][rows].numpy(), teacher)
            rank_item = rank_stats(teacher, integrated, labels, hard)
            add_counts(rank_total, rank_item)
            raw["teacher_rank_pairs"] += rank_item["pairs"]
            raw["teacher_hard_pair_rows"] += len(hard)
            kind = "calibration" if entry["split"] == "calibration" else "screening"
            base = {"video": video, "expression": str(entry["expression"]),
                    "query_index": int(entry["query_index"]), "frame": int(frame),
                    "track_id": tracks, "label": labels, "source": sources}
            records[kind]["teacher"].append({**base, "score": teacher})
            records[kind]["zero_teacher"].append({**base, "score": teacher.copy()})
            records[kind]["integrated"].append({**base, "score": integrated})
            raw["expression_frame_units"] += 1

        del patches, numeric, history, history_mask, history_time, encoded

    del encoder, model, teacher_model, banks, caches
    if raw["missing_cache_rows"] or raw["missing_teacher_tracks"]:
        raise RuntimeError(
            f"L44 B1 alignment incomplete: missing_cache_rows={raw['missing_cache_rows']} "
            f"missing_teacher_tracks={raw['missing_teacher_tracks']}"
        )
    if raw["nonfinite_scores"]:
        raise FloatingPointError(f"nonfinite L44 scores in {raw['nonfinite_scores']} units")

    calibration = choose_threshold(records["calibration"]["teacher"])
    threshold = float(calibration["threshold"])
    strategy = {}
    for name in records["calibration"]:
        strategy[name] = {
            "calibration_threshold": calibration,
            "calibration": metric_summary(records["calibration"][name], threshold),
            "screening": metric_summary(records["screening"][name], threshold),
        }

    teacher_screen = strategy["teacher"]["screening"]
    integrated_screen = strategy["integrated"]["screening"]
    def delta(key):
        return integrated_screen[key] - teacher_screen[key]

    hard_delta = delta("hard_violation_rate")
    pair_total = max(1, rank_total["teacher_correct"])
    gate = {
        "top1_delta": delta("top1_frame_recall"),
        "recall_delta": delta("recall"),
        "hard_violation_delta": hard_delta,
        "precision_delta": delta("precision"),
        "fp_frame_delta": delta("false_positive_candidates_per_frame"),
        "multi_positive_recall_delta": delta("multi_positive_recall"),
        "empty_rate_delta": delta("empty_output_rate"),
        "null_false_acceptance_delta": delta("null_frame_false_acceptance"),
        "teacher_correct_flip_ratio": rank_total["teacher_correct_flips"] / pair_total,
        "teacher_error_correction_ratio": rank_total["teacher_error_corrections"] /
        max(1, rank_total["teacher_error"]),
    }
    gate.update({
        "top1_preserved": gate["top1_delta"] >= -.02,
        "recall_preserved": gate["recall_delta"] >= -.03,
        "hard_violation_improved_by_0.05": hard_delta <= -.05,
        "teacher_correct_flip_under_1pct": gate["teacher_correct_flip_ratio"] <= .01,
        "precision_preserved": gate["precision_delta"] >= -.01,
        "fp_frame_preserved": gate["fp_frame_delta"] <= .10,
        "multi_positive_preserved": gate["multi_positive_recall_delta"] >= -.03,
        "empty_not_massively_increased": gate["empty_rate_delta"] <= .05,
        "null_not_massively_increased": gate["null_false_acceptance_delta"] <= .05,
    })
    gate_keys = (
        "top1_preserved", "recall_preserved", "hard_violation_improved_by_0.05",
        "teacher_correct_flip_under_1pct", "precision_preserved",
        "fp_frame_preserved", "multi_positive_preserved",
        "empty_not_massively_increased", "null_not_massively_increased",
    )
    selected_units = sorted(selected_screen)
    payload = {
        "format": "locatemot-l44-integrated-query-region-track-decoder-candidate-gate-v1",
        "stage": "L44-B1-fixed-100-frame-unit-candidate-gate",
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "teacher_checkpoint": str(L29.resolve()), "teacher_checkpoint_sha256": sha256(L29),
        "manifest": str(FAST.resolve()), "manifest_sha256": sha256(FAST),
        "score_cache": str(SCORE_ROOT.resolve()),
        "score_cache_model": "A_C1_S2000 immutable candidate/label cache; used for replay labels only",
        "cache_checkpoint_usage": "L28 persistent cache where present; L19 read-only in-memory sequence replay for screening-only videos; L29 current-membership teacher",
        "sequence_cache_sources": cache_sources,
        "text_cache": str(V5.resolve()), "text_cache_sha256": sha256(V5 / "text_tokens.pt"),
        "image_encoder": {"weights": str(WEIGHTS), "weights_sha256": sha256(WEIGHTS),
                          "pixel_storage": "transient RAM only"},
        "counts": {
            "manifest_queries": len(entries), "calibration_queries": sum(e["split"] == "calibration" for e in entries),
            "screening_queries": sum(e["split"] == "screening" for e in entries),
            "screening_units_available": len(available),
            "screening_units_selected": len(selected_screen),
            "calibration_expression_frame_units": len(records["calibration"]["teacher"]),
            "screening_expression_frame_units": len(records["screening"]["teacher"]),
        },
        "selected_screening_units_sha256": json_sha(selected_units),
        "raw_replay": raw,
        "threshold_contract": "one L29-teacher balanced-F1 threshold fitted on calibration and reused unchanged; no top-k, NULL gate, post-filter, or screening selection",
        "calibration": calibration,
        "strategies": strategy,
        "rank_flip_diagnostics": {
            "hard_set_contract": "objectness top-96 then frozen teacher score top-24; fixed before student comparison",
            "teacher_hard_pair_counts": rank_total,
            "teacher_correct_flip_ratio": rank_total["teacher_correct_flips"] / pair_total,
            "teacher_error_correction_ratio": rank_total["teacher_error_corrections"] /
            max(1, rank_total["teacher_error"]),
        },
        "residual_diagnostics": {
            "max_abs": raw["residual_max_abs"],
            "mean": raw["residual_sum"] / max(1, raw["residual_count"]),
            "count": raw["residual_count"],
            "configured_bound": float(model_config["residual_bound"]),
            "bound_satisfied": raw["residual_max_abs"] <= float(model_config["residual_bound"]) + 1e-5,
        },
        "gates_relative_to_l29_teacher": gate,
        "screening_gt_used_for_threshold": False,
        "screening_gt_used_for_model_selection": False,
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key"],
        "token_level_alignment_verified": False,
        "motion_language_decomposition": "not claimed; no verified motion-language mask",
        "decision": "pass" if all(gate[key] for key in gate_keys) else "fail",
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(out), "decision": payload["decision"],
                      "gates": gate, "teacher": teacher_screen,
                      "integrated": integrated_screen}, indent=2), flush=True)


if __name__ == "__main__":
    main()
