#!/usr/bin/env python3
"""Frozen L86 full-video inference for the legal internal validation videos.

All L69 candidates are scored in native order.  The frozen dev-selected rule
only decides which existing tracker observations are emitted; it never
creates IDs, changes boxes, or filters the candidate bank.  Internal GT files
are materialized only after predictions and the rule are frozen.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
INTERNAL = {"refer_kitti_v1": ("0004", "0018"), "refer_kitti_v2": ("0016", "0017", "0020")}
DEFAULT_CACHE = ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l86_full_rmot import L86Config, L86FullRMOT  # noqa: E402
from locatemot.rmot.l80_data import L80BankStore  # noqa: E402
from locatemot.rmot.l86_clip_data import _clip_history  # noqa: E402
from locatemot.rmot.l85_runtime import capture_group_z1_batched, load_validation_key_rows  # noqa: E402
from tools.l85_infer_fullvideo_rmot import materialize_gt, prepare_trackeval_dirs, sequence_id  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def load_model(path: Path, device: torch.device) -> tuple[L86FullRMOT, dict[str, Any]]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    model = L86FullRMOT(L86Config(**package["model_config"]))
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"strict L86 checkpoint reload failed: {result}")
    model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"path": str(path.resolve()), "sha256": sha256(path), "epoch": int(package.get("epoch", 0)),
                   "step": int(package.get("step", 0)), "model_config": package["model_config"], "strict_reload": True}


def query_rows_for_video(all_rows: list[dict[str, Any]], dataset: str, video: str) -> list[dict[str, Any]]:
    found: dict[int, str] = {}
    for row in all_rows:
        if str(row["dataset"]) != dataset or str(row["video"]) != video:
            continue
        qid = int(row["query_id"]); sentence = str(row["sentence"])
        if qid in found and found[qid] != sentence:
            raise AssertionError(f"query sentence drift {dataset}|{video}|{qid}")
        found[qid] = sentence
    if not found:
        raise AssertionError(f"no internal validation queries {dataset}|{video}")
    return [{"dataset": dataset, "video": video, "query_id": qid, "sentence": sentence,
             "expression": sentence} for qid, sentence in sorted(found.items())]


def frame_groups(dataset: str, video: str, queries: list[dict[str, Any]], store: L80BankStore) -> list[dict[str, Any]]:
    store._store.load_video(video)
    frame_ids = [int(value) for value in store._store.tensors["frame_ids"].tolist()]
    groups = []
    for frame in frame_ids:
        groups.append({"group_key": f"{dataset}|{video}|{frame}", "dataset": dataset, "video": video,
                       "frame_id": frame,
                       "queries": [dict(row, unit_key=f"{dataset}|{video}|{int(row['query_id'])}|{frame}", frame_id=frame)
                                   for row in queries]})
    return groups


def sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0.0:
        exp_value = math.exp(-value); return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value); return exp_value / (1.0 + exp_value)


def score_group(item: dict[str, Any], group: dict[str, Any], store: L80BankStore,
                model: L86FullRMOT, device: torch.device) -> list[dict[str, Any]]:
    rows = [dict(row) for row in group["queries"]]
    if item.get("labels_in_cache") or item.get("format") != "locatemot-l85-z1-semantic-group-v1":
        raise AssertionError(f"invalid label-free L85 item {group['group_key']}")
    query_ids = [int(value) for value in item["query_ids"]]
    if query_ids != [int(row["query_id"]) for row in rows]:
        raise AssertionError(f"full-video query order drift {group['group_key']}")
    first = store.build_unit(rows[0])
    if int(item["candidate_count"]) != first.candidate_count or [int(x) for x in item["row_offsets"]] != first.row_offsets:
        raise AssertionError(f"full-video candidate contract drift {group['group_key']}")
    history, history_mask, history_frames = _clip_history(first, True, length=4)
    if int((history_frames > int(first.frame_id)).sum()) != 0:
        raise AssertionError(f"future history in full-video group {group['group_key']}")
    z1 = item["z1"].float().clone().to(device)
    text = item["text_global"].float().clone().to(device)
    frame_global = item["frame_global"].float().clone().to(device)
    current = first.observations.float().clone().to(device)
    history = history.float().clone().to(device); history_mask = history_mask.to(device); history_frames = history_frames.to(device)
    with torch.inference_mode():
        output = model(z1, text, frame_global, current, history, history_mask, history_frames, first.frame_id, temporal_enabled=True)
    scores = output["candidate_energy"].float().cpu().numpy()
    presence = output["presence_logit"].float().cpu().numpy()
    null = output["null_logit"].float().cpu().numpy()
    if not all(np.isfinite(value).all() for value in (scores, presence, null)):
        raise FloatingPointError(f"nonfinite L86 full-video score {group['group_key']}")
    records = []
    for q, row in enumerate(rows):
        row_keys = [(str(first.dataset), str(first.video), int(row["query_id"]), int(first.frame_id),
                     str(first.bank_path), int(offset)) for offset in first.row_offsets]
        if len(scores[q]) != first.candidate_count or len(row_keys) != first.candidate_count:
            raise AssertionError(f"full-video score row count drift {row['unit_key']}")
        records.append({"format": "locatemot-l86-fullvideo-score-v1", "unit_key": str(row["unit_key"]),
                        "group_key": str(group["group_key"]), "dataset": str(first.dataset), "video": str(first.video),
                        "query_id": int(row["query_id"]), "frame_id": int(first.frame_id),
                        "candidate_count": int(first.candidate_count), "row_offsets": [int(x) for x in first.row_offsets],
                        "row_keys": [list(x) for x in row_keys], "candidate_indices": [int(x) for x in first.candidate_indices],
                        "track_ids": [int(x) for x in first.track_ids], "pool_ids": [int(x) for x in first.pool_ids],
                        "score": scores[q].astype(np.float64).tolist(), "presence_logit": float(presence[q]),
                        "null_logit": float(null[q]), "future_history_count": int((history_frames > first.frame_id).sum()),
                        "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
                        "labels_attached": False, "finite_scores": True})
    del output, first, z1, text, frame_global, current, history, history_mask, history_frames
    return records


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L86 full-video output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256(MANIFEST) != MANIFEST_SHA: raise AssertionError("fixed manifest SHA drift")
        selection_path = args.selection.resolve(); selection = json.loads(selection_path.read_text())
        selected = selection["selected"]; checkpoint_info = selected["checkpoint_info"]; rule = selected["rule_fit"]
        checkpoint = Path(checkpoint_info["path"]).resolve()
        if sha256(checkpoint) != str(checkpoint_info["sha256"]): raise AssertionError("selected checkpoint SHA drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        model, loaded = load_model(checkpoint, device)
        all_rows = load_validation_key_rows()
        requested = [args.dataset] if args.dataset != "all" else ["refer_kitti_v1", "refer_kitti_v2"]
        store = L80BankStore(max_history=8)
        from locatemot.rmot.l82_grounding_runtime import GroundingCandidateReferenceRuntime
        runtime = GroundingCandidateReferenceRuntime(device)
        prediction_audits_path = out / "prediction_audits.jsonl"
        prediction_audits = prediction_audits_path.open("w")
        video_queries: dict[str, dict[str, list[dict[str, Any]]]] = {}
        counts = {"frames": 0, "queries": 0, "candidate_rows": 0, "selected_rows": 0}
        try:
            for dataset in requested:
                if dataset not in INTERNAL: raise ValueError(dataset)
                video_queries[dataset] = {}
                for video in INTERNAL[dataset]:
                    queries = query_rows_for_video(all_rows, dataset, video)
                    paths = prepare_trackeval_dirs(out, dataset)
                    # The imported helper only uses the supplied path map; the
                    # tracker directory below is created explicitly as l86.
                    paths["tracker_data"] = out / dataset / "trackers" / "l86" / "data"
                    paths["tracker_data"].mkdir(parents=True, exist_ok=True)
                    groups = frame_groups(dataset, video, queries, store)
                    video_queries[dataset][video] = queries
                    for frame_index, group in enumerate(groups):
                        item = capture_group_z1_batched(group, device, runtime=runtime, bank_store=store,
                                                        query_batch_size=int(args.query_batch_size))
                        records = score_group(item, group, store, model, device)
                        boxes = np.asarray(item["boxes"], dtype=np.float32)
                        if boxes.shape != (int(item["candidate_count"]), 4): raise AssertionError("box shape drift")
                        for record in records:
                            score = np.asarray(record["score"], dtype=np.float64)
                            gate = (float(record["presence_logit"]) >= float(rule["presence_threshold"]) and
                                    float(record["presence_logit"]) - float(record["null_logit"]) >= float(rule["null_margin"]))
                            selected_rows = np.flatnonzero((score >= float(rule["candidate_threshold"])) & bool(gate))
                            tracks = [int(x) for x in record["track_ids"]]
                            if len(tracks) != len(set(tracks)):
                                raise AssertionError(f"duplicate query-independent track IDs in frame {record['unit_key']}")
                            seq = sequence_id(video, int(record["query_id"]))
                            path = paths["tracker_data"] / f"{seq}.txt"
                            with path.open("a") as handle:
                                for local in selected_rows.tolist():
                                    x1, y1, x2, y2 = [float(x) for x in boxes[local]]
                                    handle.write(f"{int(group['frame_id']) + 1},{tracks[local]},{x1:.6f},{y1:.6f},"
                                                 f"{x2-x1:.6f},{y2-y1:.6f},{sigmoid(score[local]):.8f},1,1,1\n")
                            audit = {"dataset": dataset, "video": video, "query_id": int(record["query_id"]),
                                     "frame_id": int(group["frame_id"]), "unit_key": str(record["unit_key"]),
                                     "candidate_rows_scored": int(record["candidate_count"]), "selected_rows": int(len(selected_rows)),
                                     "presence_gate": bool(gate), "candidate_rows_retained": True,
                                     "candidate_deletion": False, "candidate_truncation": False, "labels_attached": False}
                            prediction_audits.write(json.dumps(audit, ensure_ascii=False) + "\n")
                            counts["queries"] += 1; counts["candidate_rows"] += int(record["candidate_count"]); counts["selected_rows"] += int(len(selected_rows))
                        counts["frames"] += 1
                        del item, records
                        gc.collect()
                        if device.type == "cuda" and frame_index % 20 == 0: torch.cuda.empty_cache()
                    print(f"[l86-infer] {dataset} video={video} frames={len(groups)} queries={len(queries)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        finally:
            prediction_audits.close(); runtime.close(); store._store._bank = None; store._store._text_cache = None
            del runtime, store, model
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
        # GT is intentionally attached only after the complete strategy and
        # every prediction row have been frozen and written.
        eval_summary: dict[str, Any] = {}
        for dataset in requested:
            paths = prepare_trackeval_dirs(out, dataset); paths["tracker_data"] = out / dataset / "trackers" / "l86" / "data"
            paths["tracker_data"].mkdir(parents=True, exist_ok=True)
            seqs, query_audits, record_audits = materialize_gt(dataset, video_queries[dataset], paths)
            eval_summary[dataset] = {"sequences": seqs, "sequence_count": len(seqs), "query_gt_audits": query_audits,
                                     "record_audits": record_audits, "labels_attached_after_predictions": True,
                                     "tracker_data": str(paths["tracker_data"].resolve())}
        summary = {"format": "locatemot-l86-fullvideo-inference-v1", "status": "complete",
                   "scope": "internal full-video validation", "full_video": True, "command": command,
                   "cwd": str(ROOT), "luna_thread": THREAD, "seed": SEED, "datasets": requested,
                   "selected_checkpoint": loaded, "selection_source": str(selection_path), "selection_sha256": sha256(selection_path),
                   "emission_rule": rule, "prediction_strategy_frozen_before_gt": True, "prediction_audits": str(prediction_audits_path.resolve()),
                   "prediction_counts": counts, "eval_summary": eval_summary,
                   "candidate_bank": "immutable L69 budget-40; native frame pointers", "all_candidate_rows_scored": True,
                   "candidate_deletion": False, "candidate_truncation": False, "model_selection_used_validation": False,
                   "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                   "hota_trackeval_run": False, "trackeval_run": False, "no_hota_or_trackeval": True,
                   "z1_representation_changed": False, "groundingdino_lora_used": False,
                   "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
                   "manifest_sha256": MANIFEST_SHA, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                   "wall_seconds": time.perf_counter() - started, "failure_root_cause": None,
                   "next_action": "run independent L86 TrackEval wrapper"}
        write_json(out / "summary.json", summary); write_json(out / "provenance.json", summary); write_json(out / "status.json", summary)
        return 0
    except Exception:
        trace = traceback.format_exc(); (out / "INCOMPLETE.md").write_text("# L86 full-video inference — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l86-fullvideo-inference-v1", "status": "incomplete",
                                         "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False,
                                         "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                         "hota_trackeval_run": False})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--selection", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--dataset", choices=("all", "refer_kitti_v1", "refer_kitti_v2"), default="all")
    parser.add_argument("--query-batch-size", type=int, default=8)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
