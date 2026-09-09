#!/usr/bin/env python3
"""Phase-consistent sparse rescoring for the Stage-S L89 checkpoints only."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_TOOLS = ROOT / "tools"
for value in (WORK_ROOT, WORK_ROOT / "tools", ROOT, ASSET_TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from l89d_fullvideo_common import (  # noqa: E402
    BASE_Z1_CACHE,
    L89LanguageTokenCache,
    MANIFEST_SHA,
    SEED,
    THREAD,
    WORK_ROOT as COMMON_WORK_ROOT,
    L80BankStore,
    command_line,
    manifest_assertion,
    sha256_file,
    standard_flags,
    write_json,
)
from l89_score_dev import language_for, make_record, package_info  # noqa: E402
from l88c_eval_metrics import fit_rule_set  # noqa: E402
from locatemot.models.l89_full_rmot import L89Config, L89FullRMOT  # noqa: E402
from locatemot.rmot.l86_clip_data import L86ClipStore  # noqa: E402
from l89e_phase_policy import history_for_batch, phase_policy_for_epoch  # noqa: E402


FORMAT = "locatemot-l89e-stage-s-rescore-v1"
S_EPOCHS = (2, 4, 6, 8)
EXPECTED_EPOCHS = list(range(2, 41, 2))


def _load_stage_s_model(path: Path, device: torch.device) -> tuple[L89FullRMOT, dict[str, Any], Any]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    info = package_info(path, package)
    policy = phase_policy_for_epoch(int(info["epoch"]), str(info["phase"]))
    if policy.phase != "S" or policy.temporal_enabled:
        raise AssertionError(f"stage-S checkpoint policy drift: {path}")
    model = L89FullRMOT(L89Config(**package["model_config"])).to(device=device, dtype=torch.float32)
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"strict L89 reload failed: {path}: {loaded}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, info, policy


def _validate_record(record: dict[str, Any], policy: Any) -> None:
    if record.get("format") != "locatemot-l89-score-record-v1":
        raise AssertionError("L89 score record format drift")
    n = int(record["candidate_count"])
    for name in ("score", "candidate_energy", "r_static", "r_total", "candidate_prior", "labels", "candidate_gt", "row_keys"):
        if len(record.get(name, [])) != n:
            raise AssertionError(f"L89 Stage-S row length drift: {name}/{record.get('unit_key')}")
    if not np.isfinite(np.asarray(record["score"], dtype=np.float64)).all():
        raise FloatingPointError(f"nonfinite Stage-S scores: {record.get('unit_key')}")
    if not bool(record.get("phase_consistent_temporal")):
        raise AssertionError("Stage-S phase consistency flag missing")
    if record.get("history_contract") != "zero_history" or int(record.get("history_valid_count", -1)) != 0:
        raise AssertionError(f"Stage-S history contract drift: {record.get('unit_key')}")
    if record.get("candidate_deletion") or record.get("candidate_truncation"):
        raise AssertionError("candidate deletion/truncation in Stage-S record")
    if not bool(record.get("candidate_rows_retained")):
        raise AssertionError("candidate retention flag missing")


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty Stage-S output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    started = time.perf_counter()
    store: L86ClipStore | None = None
    try:
        if Path.cwd().resolve() != COMMON_WORK_ROOT:
            raise RuntimeError(f"wrong L89E worktree cwd: {Path.cwd()}")
        manifest_assertion()
        checkpoint_dir = args.checkpoint_dir.resolve()
        paths = sorted(
            checkpoint_dir.glob("checkpoint_l89_epoch*.pt"),
            key=lambda item: int(item.stem.split("epoch")[-1]),
        )
        actual = [int(item.stem.split("epoch")[-1]) for item in paths]
        if actual != EXPECTED_EPOCHS:
            raise AssertionError(f"L89 even checkpoint drift: {actual}")
        stage_paths = [path for path in paths if int(path.stem.split("epoch")[-1]) in S_EPOCHS]
        if [int(path.stem.split("epoch")[-1]) for path in stage_paths] != list(S_EPOCHS):
            raise AssertionError("Stage-S checkpoint set drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        store = L86ClipStore(args.z1_cache.resolve(), load_cache_into_ram=False)
        dev_keys = list(store.dev_keys)
        if len(dev_keys) != 138:
            raise AssertionError(f"L89 dev group count drift: {len(dev_keys)}")
        expected_records = sum(len(store.groups[key]["queries"]) for key in dev_keys)
        if expected_records != 498:
            raise AssertionError(f"L89 Stage-S record denominator drift: {expected_records}")
        language = L89LanguageTokenCache(args.language_cache.resolve())
        record_path = out / "score_records.jsonl"
        summaries: list[dict[str, Any]] = []
        with record_path.open("w", encoding="utf-8") as handle:
            for checkpoint_path in stage_paths:
                model, info, policy = _load_stage_s_model(checkpoint_path, device)
                checkpoint_records: list[dict[str, Any]] = []
                for index, group_key in enumerate(dev_keys):
                    frame = store.build_frame(group_key, temporal_enabled=policy.temporal_enabled)
                    history, history_mask, history_frames = history_for_batch(frame, policy)
                    if int(history_mask.sum()) != 0:
                        raise AssertionError(f"nonzero Stage-S history: {frame.group_key}")
                    tokens, mask = language_for(language, store, frame, device)
                    with torch.inference_mode():
                        output = model(
                            frame.z1.float().to(device), tokens, mask,
                            frame.text_global.float().to(device), frame.frame_global.float().to(device),
                            frame.current_observation.float().to(device), history.float().to(device),
                            history_mask.bool().to(device), history_frames.long().to(device),
                            frame.frame_id, temporal_enabled=policy.temporal_enabled,
                        )
                    for query_index in range(len(frame.query_ids)):
                        record = make_record(frame, query_index, output, info)
                        record.update({
                            "phase_consistent_temporal": True,
                            "temporal_policy": policy.to_dict(),
                            "history_contract": policy.history_mode,
                            "history_valid_count": int(history_mask.sum().item()),
                            "history_valid_count_per_candidate": [int(x) for x in history_mask.sum(dim=1).tolist()],
                            "stage_s_rescored": True,
                            "fixed_calibration_read": False,
                            "fixed_validation_read": False,
                        })
                        _validate_record(record, policy)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        checkpoint_records.append(record)
                    store.cache_items.clear()
                    del frame, history, history_mask, history_frames, tokens, mask, output
                    gc.collect()
                    if (index + 1) % 25 == 0 or index + 1 == len(dev_keys):
                        print(
                            f"[l89e-stage-s] epoch={info['epoch']} group={index + 1}/{len(dev_keys)} "
                            f"elapsed={time.perf_counter() - started:.1f}s",
                            flush=True,
                        )
                if len(checkpoint_records) != expected_records:
                    raise AssertionError(f"Stage-S records drift: {len(checkpoint_records)} != {expected_records}")
                unit_keys = [str(row["unit_key"]) for row in checkpoint_records]
                if len(unit_keys) != len(set(unit_keys)):
                    raise AssertionError(f"duplicate Stage-S unit keys at epoch {info['epoch']}")
                rules = fit_rule_set(checkpoint_records)
                if set(("B", "R", "P")) - set(rules):
                    raise AssertionError("corrected B/R/P rules missing")
                summaries.append({
                    "checkpoint_info": info,
                    "record_count": len(checkpoint_records),
                    "rule_fits": rules,
                    "phase_policy": policy.to_dict(),
                    "phase_consistent_temporal": True,
                    "history_contract": "zero_history",
                    "fit_dev_only": True,
                })
                del model, info, policy
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        payload = {
            "format": FORMAT,
            "status": "complete",
            "stage": "L89E Stage-S phase-consistent sparse dev rescore",
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "seed": SEED,
            "checkpoint_epochs": list(S_EPOCHS),
            "checkpoint_count": len(summaries),
            "dev_group_count": len(dev_keys),
            "record_count_per_checkpoint": [item["record_count"] for item in summaries],
            "record_count": int(sum(item["record_count"] for item in summaries)),
            "score_records": str(record_path.resolve()),
            "score_records_sha256": sha256_file(record_path),
            "checkpoint_summaries": summaries,
            "stage_s_rescored": True,
            "temporal_enabled": False,
            "history_contract": "zero_history",
            "fit_dev_labels_only": True,
            "fixed_calibration_read": False,
            "fixed_validation_read": False,
            "all_candidate_rows_scored": True,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "phase_consistent_temporal": True,
            "manifest_sha256": MANIFEST_SHA,
            "z1_cache": str(args.z1_cache.resolve()),
            "language_cache": str(args.language_cache.resolve()),
            "persistent_dense_cache_written": False,
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started,
            **standard_flags(hota_trackeval_run=False),
            "failure_root_cause": None,
            "next_action": "merge phase-consistent Stage-S rows with immutable T/J sparse rows",
        }
        write_json(out / "summary.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-l89e-stage-s-provenance-v1"})
        write_json(out / "status.json", {
            "format": FORMAT,
            "status": "complete",
            "record_count": payload["record_count"],
            "checkpoint_epochs": list(S_EPOCHS),
            "phase_consistent_temporal": True,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "zero_training": True,
        })
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89E Stage-S rescore — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {
            "format": FORMAT,
            "status": "incomplete",
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "repair the first Stage-S contract error and use a new output",
            **standard_flags(hota_trackeval_run=False),
        })
        raise
    finally:
        if store is not None:
            store.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--z1-cache", type=Path, default=BASE_Z1_CACHE)
    parser.add_argument("--language-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
