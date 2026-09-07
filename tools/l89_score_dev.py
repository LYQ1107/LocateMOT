#!/usr/bin/env python3
"""Score every registered even L89 checkpoint on the legal fit/dev groups.

This pass is intentionally separate from checkpoint selection.  It reads only
the L82 video-disjoint fit/dev grouping (labels are attached after each full
L69 row set is built), writes every candidate row, and never reads the fixed
calibration/validation slice.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
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
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DEFAULT_Z1 = ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
DEFAULT_LANG = WORK_ROOT / "outputs/l89/cache/language_tokens_retry1"
EXPECTED_EPOCHS = list(range(2, 41, 2))

for path in (WORK_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import locatemot.models as _models_package  # noqa: E402
import locatemot.rmot as _rmot_package  # noqa: E402

for package, name in ((_models_package, "models"), (_rmot_package, "rmot")):
    package_path = str(WORK_ROOT / "locatemot" / name)
    if package_path not in [str(value) for value in package.__path__]:
        package.__path__.append(package_path)

from locatemot.models.l89_full_rmot import L89Config, L89FullRMOT  # noqa: E402
from locatemot.rmot.l86_clip_data import L86ClipStore  # noqa: E402
from l88_eval_metrics import fit_rule_set as fit_rules  # noqa: E402
from locatemot.rmot.l89_language_cache import L89LanguageTokenCache  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def package_info(path: Path, package: dict[str, Any]) -> dict[str, Any]:
    if package.get("format") != "locatemot-l89-checkpoint-v1":
        raise AssertionError(f"invalid L89 checkpoint format: {path}")
    if int(package.get("seed", -1)) != SEED or str(package.get("manifest_sha256")) != MANIFEST_SHA:
        raise AssertionError(f"L89 checkpoint seed/manifest drift: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "epoch": int(package["epoch"]),
            "optimizer_step": int(package["optimizer_step"]), "phase": str(package["phase"]),
            "model_config": package["model_config"], "strict_package": True}


def language_for(cache: L89LanguageTokenCache, store: L86ClipStore, frame: Any,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    by_qid = {int(row["query_id"]): str(row["sentence"])
              for row in store.groups[str(frame.group_key)]["queries"]}
    sentences = [by_qid[int(qid)] for qid in frame.query_ids]
    return cache.get_batch(frame.dataset, frame.video, frame.query_ids, sentences, device)


def make_record(frame: Any, q: int, output: dict[str, torch.Tensor], checkpoint: dict[str, Any]) -> dict[str, Any]:
    label = frame.labels[q]
    n = len(frame.row_offsets)
    fields = {name: output[name][q].detach().float().cpu().numpy().astype(np.float64).tolist()
              for name in ("candidate_energy", "r_static", "r_total")}
    fields["candidate_prior"] = output["candidate_prior"].detach().float().cpu().numpy().astype(np.float64).tolist()
    for name in fields:
        if len(fields[name]) != n or not np.isfinite(np.asarray(fields[name], dtype=np.float64)).all():
            raise AssertionError(f"L89 dev score length/finite drift: {frame.group_key}:{name}")
    record: dict[str, Any] = {
        "format": "locatemot-l89-score-record-v1", "checkpoint": checkpoint,
        "group_key": str(frame.group_key), "unit_key": str(label["unit_key"]),
        "dataset": str(frame.dataset), "video": str(frame.video), "query_id": int(frame.query_ids[q]),
        "frame_id": int(frame.frame_id), "candidate_count": n,
        "row_offsets": [int(x) for x in frame.row_offsets], "row_keys": [list(x) for x in frame.row_keys],
        "candidate_indices": [int(x) for x in frame.candidate_indices], "track_ids": [int(x) for x in frame.track_ids],
        "score": fields["candidate_energy"], "candidate_energy": fields["candidate_energy"],
        "r_static": fields["r_static"], "r_total": fields["r_total"], "candidate_prior": fields["candidate_prior"],
        "presence_logit": float(output["presence_logit"][q].detach().float().cpu()),
        "null_logit": float(output["null_logit"][q].detach().float().cpu()),
        "future_history_count": int((frame.history_frame_ids > int(frame.frame_id)).sum()),
        "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
        "finite_scores": True, "labels_attached_after_feature_construction": True,
        "labels": [bool(x) for x in label["labels"].tolist()],
        "target_ids": [str(x) for x in label["target_ids"]],
        "candidate_gt": [None if x is None else str(x) for x in label["candidate_gt"]],
        "positive_indices": [int(x) for x in label["positive_indices"]], "positive_count": int(label["positive_count"]),
        "target_present": bool(label["target_present"]), "candidate_present": bool(label["candidate_present"]),
        "coverage_mask": bool(label["coverage_mask"]), "category": str(label["category"]),
        "declared_category": str(label.get("declared_category", "unknown")),
        "label_source": str(label["label_source"]),
    }
    if record["future_history_count"] != 0:
        raise AssertionError(f"future L89 dev history: {record['unit_key']}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--z1-cache", type=Path, default=DEFAULT_Z1)
    parser.add_argument("--language-cache", type=Path, default=DEFAULT_LANG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=0)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89 dev score output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    store = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89 worktree cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if args.max_checkpoints < 0 or args.max_groups < 0:
            raise ValueError("limits must be nonnegative")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        checkpoint_paths = sorted(args.checkpoint_dir.resolve().glob("checkpoint_l89_epoch*.pt"),
                                  key=lambda p: int(p.stem.split("epoch")[-1]))
        actual = [int(p.stem.split("epoch")[-1]) for p in checkpoint_paths]
        if actual != EXPECTED_EPOCHS:
            raise AssertionError(f"L89 even checkpoint drift: {actual}")
        if args.max_checkpoints:
            checkpoint_paths = checkpoint_paths[:args.max_checkpoints]
        store = L86ClipStore(args.z1_cache.resolve(), load_cache_into_ram=False)
        dev_keys = [str(value) for value in store.dev_keys]
        if len(dev_keys) != 138:
            raise AssertionError(f"L89 dev group count drift: {len(dev_keys)}")
        if args.max_groups:
            dev_keys = dev_keys[:args.max_groups]
        expected_records = sum(len(store.groups[key]["queries"]) for key in dev_keys)
        language = L89LanguageTokenCache(args.language_cache.resolve())
        records_path = out / "score_records.jsonl"
        checkpoint_summaries: list[dict[str, Any]] = []
        with records_path.open("w", encoding="utf-8") as handle:
            for checkpoint_path in checkpoint_paths:
                package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                info = package_info(checkpoint_path, package)
                model = L89FullRMOT(L89Config(**package["model_config"])).to(device=device, dtype=torch.float32)
                reload_result = model.load_state_dict(package["model_state_dict"], strict=True)
                if reload_result.missing_keys or reload_result.unexpected_keys:
                    raise AssertionError(f"L89 strict checkpoint load failed: {reload_result}")
                model.eval()
                checkpoint_records: list[dict[str, Any]] = []
                with torch.inference_mode():
                    for index, group_key in enumerate(dev_keys):
                        frame = store.build_frame(group_key, temporal_enabled=True)
                        tokens, mask = language_for(language, store, frame, device)
                        output = model(
                            frame.z1.to(device), tokens, mask, frame.text_global.to(device), frame.frame_global.to(device),
                            frame.current_observation.to(device), frame.history_observations.to(device), frame.history_mask.to(device),
                            frame.history_frame_ids.to(device), frame.frame_id, temporal_enabled=True,
                        )
                        for q in range(len(frame.query_ids)):
                            record = make_record(frame, q, output, info)
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            checkpoint_records.append(record)
                        store.cache_items.clear()
                        del frame, tokens, mask, output
                        if (index + 1) % 25 == 0 or index + 1 == len(dev_keys):
                            print(f"[l89-dev] epoch={info['epoch']} group={index + 1}/{len(dev_keys)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
                if len(checkpoint_records) != expected_records:
                    raise AssertionError(f"L89 dev record count drift epoch={info['epoch']}: {len(checkpoint_records)} != {expected_records}")
                unique = {str(row["unit_key"]) for row in checkpoint_records}
                if len(unique) != len(checkpoint_records):
                    raise AssertionError(f"duplicate L89 dev unit keys epoch={info['epoch']}")
                rules = fit_rules(checkpoint_records)
                checkpoint_summaries.append({"checkpoint_info": info, "record_count": len(checkpoint_records),
                                             "rule_fits": rules, "fit_dev_only": True})
                del model, package
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        payload = {
            "format": "locatemot-l89-dev-score-v1", "status": "complete",
            "stage": "internal fit/dev checkpoint scoring; threshold/selection not yet frozen",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
            "checkpoint_count": len(checkpoint_summaries), "checkpoint_epochs": [x["checkpoint_info"]["epoch"] for x in checkpoint_summaries],
            "dev_group_count": len(dev_keys), "dev_record_count_per_checkpoint": [x["record_count"] for x in checkpoint_summaries],
            "score_records": str(records_path.resolve()), "score_records_sha256": sha256_file(records_path),
            "checkpoint_summaries": checkpoint_summaries, "fit_dev_labels_only": True,
            "fixed_calibration_read": False, "fixed_validation_read": False,
            "all_candidate_rows_scored": True, "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False, "no_hota_or_trackeval": True,
            "z1_representation_changed": False, "groundingdino_lora_used": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started,
            "failure_root_cause": None, "next_action": "run L89 checkpoint shortlist on fit/dev rules",
        }
        write_json(out / "dev_scores.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-l89-dev-score-provenance-v1",
                                                         "inputs": {"checkpoint_dir": str(args.checkpoint_dir.resolve()),
                                                                    "z1_cache": str(args.z1_cache.resolve()),
                                                                    "language_cache": str(args.language_cache.resolve()),
                                                                    "manifest_sha256": MANIFEST_SHA}})
        write_json(out / "status.json", {"format": "locatemot-l89-dev-score-v1", "status": "complete",
                                          "checkpoint_count": len(checkpoint_summaries), "dev_group_count": len(dev_keys),
                                          "record_count_per_checkpoint": expected_records,
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                                          "failure_root_cause": None, "next_action": "run L89 checkpoint shortlist on fit/dev rules"})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89 dev score — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {"format": "locatemot-l89-dev-score-v1", "status": "incomplete",
                                          "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise
    finally:
        if store is not None:
            store.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
