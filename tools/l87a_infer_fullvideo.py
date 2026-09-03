#!/usr/bin/env python3
"""L87-A internal full-video inference under corrected candidate-vs-NULL rule."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get(
    "LOCATEMOT_ASSET_ROOT", "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
)).resolve()
if str(WORK_ROOT) not in sys.path: sys.path.insert(0, str(WORK_ROOT))
if str(ASSET_ROOT) not in sys.path: sys.path.append(str(ASSET_ROOT))

import locatemot.models as _locatemot_models  # noqa: E402
_asset_models = str(ASSET_ROOT / "locatemot" / "models")
if _asset_models not in _locatemot_models.__path__:
    _locatemot_models.__path__.append(_asset_models)
import locatemot.rmot as _locatemot_rmot  # noqa: E402
_asset_rmot = str(ASSET_ROOT / "locatemot" / "rmot")
if _asset_rmot not in _locatemot_rmot.__path__:
    _locatemot_rmot.__path__.append(_asset_rmot)

from locatemot.rmot.l80_data import L80BankStore  # noqa: E402
from tools.l86_infer_fullvideo import (  # noqa: E402
    frame_groups, load_model, query_rows_for_video, score_group,
)
from tools.l85_infer_fullvideo_rmot import (  # noqa: E402
    materialize_gt, prepare_trackeval_dirs, sequence_id,
)
from locatemot.rmot.l85_fullvideo_bank import INTERNAL_V1, INTERNAL_V2  # noqa: E402
from locatemot.rmot.l85_runtime import load_validation_key_rows  # noqa: E402

THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DEFAULT_CACHE = ASSET_ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
INTERNAL = {"refer_kitti_v1": tuple(INTERNAL_V1), "refer_kitti_v2": tuple(INTERNAL_V2)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", choices=("all", "refer_kitti_v1", "refer_kitti_v2"), default="all")
    parser.add_argument("--query-batch-size", type=int, default=8)
    args = parser.parse_args(); out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L87-A full-video output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    try:
        if Path.cwd().resolve() != WORK_ROOT: raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256(MANIFEST) != MANIFEST_SHA: raise AssertionError("fixed manifest SHA drift")
        selection_path = args.selection.resolve(); selection = json.loads(selection_path.read_text())
        selected = selection["selected"]; info = selected["checkpoint_info"]; rule = selected["rule_fit"]
        checkpoint = Path(info["path"]).resolve()
        if sha256(checkpoint) != str(info["sha256"]): raise AssertionError("selected checkpoint SHA drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        model, loaded = load_model(checkpoint, device)
        all_rows = load_validation_key_rows()
        datasets = [args.dataset] if args.dataset != "all" else ["refer_kitti_v1", "refer_kitti_v2"]
        store = L80BankStore(max_history=8)
        from locatemot.rmot.l82_grounding_runtime import GroundingCandidateReferenceRuntime
        runtime = GroundingCandidateReferenceRuntime(device)
        prediction_audits_path = out / "prediction_audits.jsonl"
        prediction_audits = prediction_audits_path.open("w")
        video_queries: dict[str, dict[str, list[dict[str, Any]]]] = {}
        counts = {"frames": 0, "queries": 0, "candidate_rows": 0, "selected_rows": 0}
        try:
            for dataset in datasets:
                if dataset not in INTERNAL: raise ValueError(dataset)
                video_queries[dataset] = {}
                for video in INTERNAL[dataset]:
                    queries = query_rows_for_video(all_rows, dataset, video)
                    paths = prepare_trackeval_dirs(out, dataset)
                    paths["tracker_data"] = out / dataset / "trackers" / "l87a" / "data"
                    paths["tracker_data"].mkdir(parents=True, exist_ok=True)
                    groups = frame_groups(dataset, video, queries, store)
                    video_queries[dataset][video] = queries
                    for frame_index, group in enumerate(groups):
                        # The actual capture call is kept explicit to avoid
                        # changing the frozen L85 representation contract.
                        from locatemot.rmot.l85_runtime import capture_group_z1_batched
                        captured = capture_group_z1_batched(group, device, runtime=runtime, bank_store=store,
                                                            query_batch_size=int(args.query_batch_size))
                        records = score_group(captured, group, store, model, device)
                        boxes = np.asarray(captured["boxes"], dtype=np.float32)
                        if boxes.shape != (int(captured["candidate_count"]), 4): raise AssertionError("box shape drift")
                        for record in records:
                            scores = np.asarray(record["score"], dtype=np.float64)
                            if not np.isfinite(scores).all(): raise FloatingPointError("nonfinite L87-A score")
                            gate = float(record["presence_logit"]) >= float(rule["presence_threshold"])
                            selected_rows = np.flatnonzero(
                                gate & (scores >= float(rule["candidate_threshold"])) &
                                ((scores - float(record["null_logit"])) >= float(rule["null_margin"]))
                            )
                            tracks = [int(value) for value in record["track_ids"]]
                            if len(tracks) != len(set(tracks)):
                                raise AssertionError(f"duplicate track IDs in {record['unit_key']}")
                            seq = sequence_id(video, int(record["query_id"]))
                            path = paths["tracker_data"] / f"{seq}.txt"
                            with path.open("a") as handle:
                                for local in selected_rows.tolist():
                                    x1, y1, x2, y2 = [float(value) for value in boxes[local]]
                                    handle.write(f"{int(group['frame_id']) + 1},{tracks[local]},{x1:.6f},{y1:.6f},"
                                                 f"{x2-x1:.6f},{y2-y1:.6f},{1.0:.8f},1,1,1\n")
                            audit = {"format": "locatemot-l87a-fullvideo-prediction-audit-v1",
                                     "dataset": dataset, "video": video, "query_id": int(record["query_id"]),
                                     "frame_id": int(group["frame_id"]), "unit_key": str(record["unit_key"]),
                                     "candidate_rows_scored": int(record["candidate_count"]),
                                     "selected_rows": int(len(selected_rows)), "presence_gate": bool(gate),
                                     "candidate_vs_null_used": True, "candidate_rows_retained": True,
                                     "candidate_deletion": False, "candidate_truncation": False,
                                     "labels_attached": False, "finite_scores": True}
                            prediction_audits.write(json.dumps(audit, ensure_ascii=False) + "\n")
                            counts["queries"] += 1; counts["candidate_rows"] += int(record["candidate_count"])
                            counts["selected_rows"] += int(len(selected_rows))
                        counts["frames"] += 1
                        del captured, records
                        gc.collect()
                        if device.type == "cuda" and frame_index % 20 == 0: torch.cuda.empty_cache()
                    print(f"[l87a-infer] {dataset} video={video} frames={len(groups)} queries={len(queries)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        finally:
            prediction_audits.close(); runtime.close(); store._store._bank = None; store._store._text_cache = None
            del runtime, store, model
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
        eval_summary: dict[str, Any] = {}
        for dataset in datasets:
            paths = prepare_trackeval_dirs(out, dataset)
            paths["tracker_data"] = out / dataset / "trackers" / "l87a" / "data"
            paths["tracker_data"].mkdir(parents=True, exist_ok=True)
            seqs, query_audits, record_audits = materialize_gt(dataset, video_queries[dataset], paths)
            eval_summary[dataset] = {"sequences": seqs, "sequence_count": len(seqs),
                                     "query_gt_audits": query_audits, "record_audits": record_audits,
                                     "labels_attached_after_predictions": True,
                                     "tracker_data": str(paths["tracker_data"].resolve())}
        summary = {"format": "locatemot-l87a-fullvideo-inference-v1", "status": "complete",
                   "evidence_type": "internal full-video validation inference; TrackEval is separate",
                   "scope": "internal full-video validation", "full_video": True, "command": command,
                   "work_root": str(WORK_ROOT), "asset_root": str(ASSET_ROOT), "cwd": str(WORK_ROOT),
                   "luna_thread": THREAD, "seed": 20260829, "datasets": datasets,
                   "selected_checkpoint": loaded, "selection_source": str(selection_path),
                   "selection_sha256": sha256(selection_path), "emission_rule": rule,
                   "prediction_strategy_frozen_before_gt": True, "prediction_audits": str(prediction_audits_path.resolve()),
                   "prediction_counts": counts, "eval_summary": eval_summary,
                   "candidate_bank": "immutable L69 budget-40 native frame pointers", "all_candidate_rows_scored": True,
                   "candidate_deletion": False, "candidate_truncation": False, "model_selection_used_validation": False,
                   "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                   "hota_trackeval_run": False, "no_hota_or_trackeval": True, "z1_representation_changed": False,
                   "groundingdino_lora_used": False, "token_span_region_alignment": "UNALIGNED",
                   "static_motion_alignment": "UNALIGNED", "manifest_sha256": MANIFEST_SHA,
                   "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                   "wall_seconds": time.perf_counter() - started, "failure_root_cause": None,
                   "next_action": "run independent corrected L87-A TrackEval wrapper"}
        write_json(out / "summary.json", summary); write_json(out / "provenance.json", summary); write_json(out / "status.json", summary)
        return 0
    except Exception:
        trace = traceback.format_exc(); (out / "INCOMPLETE.md").write_text("# L87-A full-video inference — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l87a-fullvideo-inference-v1", "status": "incomplete",
                                         "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False,
                                         "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                         "hota_trackeval_run": False})
        raise

if __name__ == "__main__": raise SystemExit(main())
