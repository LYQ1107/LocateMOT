#!/usr/bin/env python3
"""One-group L89 forward/backward/reload contract smoke.

This is deliberately a fit-group smoke.  It exercises the QSC-D set decoder,
the unchanged L87-A loss, and causal history without instantiating the
GroundingDINO detector or writing feature tensors.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DEFAULT_Z1 = ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
DEFAULT_LANG = WORK_ROOT / "outputs/l89/cache/language_tokens"

for path in (WORK_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def language_for(cache: Any, store: Any, frame: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    by_qid = {int(row["query_id"]): str(row["sentence"]) for row in store.groups[frame.group_key]["queries"]}
    sentences = [by_qid[int(qid)] for qid in frame.query_ids]
    return cache.get_batch(frame.dataset, frame.video, frame.query_ids, sentences, device)


def grad_report(model: torch.nn.Module) -> dict[str, Any]:
    total = 0.0
    finite = True
    nonzero = 0
    by_name: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        norm = float(value.norm())
        by_name[name] = norm
        total += norm
        nonzero += int(norm > 0.0)
        finite = finite and bool(torch.isfinite(value).all())
    return {"total_norm": total, "nonzero_parameter_grads": nonzero, "finite": finite, "by_name": by_name}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--z1-cache", type=Path, default=DEFAULT_Z1)
    parser.add_argument("--language-cache", type=Path, default=DEFAULT_LANG)
    parser.add_argument("--out", type=Path, default=WORK_ROOT / "outputs/l89/audit/contract_smoke")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89 contract output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong L89 worktree cwd: {Path.cwd()}")
    if sha256_file(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        import locatemot.rmot as _rmot_package
        runtime_path = str(WORK_ROOT / "locatemot" / "rmot")
        if runtime_path not in [str(value) for value in _rmot_package.__path__]:
            _rmot_package.__path__.append(runtime_path)
        import locatemot.models as _models_package
        models_path = str(WORK_ROOT / "locatemot" / "models")
        if models_path not in [str(value) for value in _models_package.__path__]:
            _models_package.__path__.append(models_path)
        from locatemot.models.l89_full_rmot import L89Config, L89FullRMOT
        from locatemot.rmot.l86_clip_data import L86ClipStore
        from locatemot.rmot.l87a_losses import l87a_loss
        from locatemot.rmot.l89_language_cache import L89LanguageTokenCache

        store = L86ClipStore(args.z1_cache.resolve(), load_cache_into_ram=True)
        lang = L89LanguageTokenCache(args.language_cache.resolve())
        if not store.train_keys:
            raise AssertionError("no L89 fit groups")
        # Deterministic real fit group with a positive and a negative candidate
        # so both sides of the correspondence loss are exercised.
        selected = None
        for group_key in store.train_keys:
            group = store.groups[str(group_key)]
            label_rows = [store.labels_by_key[str(row["unit_key"])] for row in group["queries"]]
            if any(int(row.get("positive_count", 0)) > 0 and int(row.get("candidate_count", 0)) > int(row.get("positive_count", 0)) for row in label_rows):
                selected = str(group_key)
                break
        if selected is None:
            raise AssertionError("no fit group with positive and negative rows")
        clip = store.build_clip(selected, temporal_enabled=True, clip_length=4)
        current = clip[-1]
        candidate_count = len(current.row_offsets)
        if candidate_count <= 0 or len(current.row_keys) != candidate_count:
            raise AssertionError("candidate row contract failed")
        if current.row_offsets != sorted(current.row_offsets):
            raise AssertionError("native row order changed")
        if current.history_frame_ids.numel() and bool((current.history_frame_ids[current.history_mask] > current.frame_id).any()):
            raise AssertionError("future history entered contract smoke")
        current_tokens, current_mask = language_for(lang, store, current, device)
        model = L89FullRMOT(L89Config()).to(device=device, dtype=torch.float32)
        model.train()
        previous_outputs: list[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]] = []
        with torch.autocast(device_type=device.type, enabled=False):
            current_output = model(
                current.z1.to(device), current_tokens, current_mask,
                current.text_global.to(device), current.frame_global.to(device),
                current.current_observation.to(device), current.history_observations.to(device),
                current.history_mask.to(device), current.history_frame_ids.to(device), current.frame_id,
                temporal_enabled=True,
            )
            current_output["r_total"].retain_grad()
            for previous in clip[:-1]:
                tokens, mask = language_for(lang, store, previous, device)
                previous_output = model(
                    previous.z1.to(device), tokens, mask,
                    previous.text_global.to(device), previous.frame_global.to(device),
                    previous.current_observation.to(device), previous.history_observations.to(device),
                    previous.history_mask.to(device), previous.history_frame_ids.to(device), previous.frame_id,
                    temporal_enabled=True,
                )
                previous_outputs.append((previous_output, previous.labels))
            loss, info = l87a_loss(current_output, current.labels, current.current_observation.to(device), previous_outputs, temporal_enabled=True)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("contract loss nonfinite")
            loss.backward()
        gradients = grad_report(model)
        if not gradients["finite"] or gradients["nonzero_parameter_grads"] <= 0:
            raise AssertionError("L89 adapter gradients are absent/nonfinite")
        grad_values = current_output["r_total"].grad
        if grad_values is None or not bool(torch.isfinite(grad_values).all()):
            raise AssertionError("L89 correspondence gradients unavailable")
        positive_positions = []
        negative_positions = []
        first_labels = current.labels[0]
        for index, value in enumerate(first_labels["labels"].tolist()):
            (positive_positions if value else negative_positions).append(index)
        if not positive_positions or not negative_positions:
            raise AssertionError("selected group lost positive/negative rows")
        positive_grad = float(grad_values[0, positive_positions].abs().max())
        negative_grad = float(grad_values[0, negative_positions].abs().max())
        if positive_grad <= 0.0 or negative_grad <= 0.0:
            raise AssertionError(f"positive/negative gradient contract failed: {positive_grad}/{negative_grad}")
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        reload_model = L89FullRMOT(L89Config()).to(device=device, dtype=torch.float32)
        reload_model.load_state_dict(state, strict=True)
        reload_model.eval(); model.eval()
        with torch.inference_mode():
            before = model(
                current.z1.to(device), current_tokens, current_mask, current.text_global.to(device), current.frame_global.to(device),
                current.current_observation.to(device), current.history_observations.to(device), current.history_mask.to(device),
                current.history_frame_ids.to(device), current.frame_id, temporal_enabled=True)["candidate_energy"]
            after = reload_model(
                current.z1.to(device), current_tokens, current_mask, current.text_global.to(device), current.frame_global.to(device),
                current.current_observation.to(device), current.history_observations.to(device), current.history_mask.to(device),
                current.history_frame_ids.to(device), current.frame_id, temporal_enabled=True)["candidate_energy"]
        reload_diff = float((before - after).abs().max())
        if reload_diff > 1e-6:
            raise AssertionError(f"strict reload output drift: {reload_diff}")
        report = {
            "format": "locatemot-l89-contract-smoke-v1", "status": "complete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "selected_fit_group": selected, "clip_group_count": len(clip),
            "candidate_count": candidate_count, "row_offsets": current.row_offsets,
            "row_keys": [list(key) for key in current.row_keys],
            "duplicate_candidate_index_rows": len(current.candidate_indices) - len(set(current.candidate_indices)),
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "future_history_count": int((current.history_frame_ids[current.history_mask] > current.frame_id).sum()),
            "z1_shape": list(current.z1.shape), "language_tokens_shape": list(current_tokens.shape),
            "language_mask_shape": list(current_mask.shape), "output_shapes": {key: list(value.shape) for key, value in current_output.items()},
            "loss": float(loss.detach()), "loss_info": info, "gradients": gradients,
            "positive_row_gradient_max": positive_grad, "negative_row_gradient_max": negative_grad,
            "reload_max_abs_diff": reload_diff, "strict_reload": True,
            "detector_instantiated": False, "frozen_input_cache_only": True,
            "manifest_sha256": MANIFEST_SHA, "z1_cache_summary": str((args.z1_cache.resolve() / "summary.json")),
            "language_cache_entries": lang.entry_count,
            "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "failure_root_cause": None, "next_action": "run L89 registered 40-epoch training",
        }
        write_json(out / "contract.json", report)
        write_json(out / "provenance.json", report | {"format": "locatemot-l89-contract-provenance-v1"})
        write_json(out / "status.json", {"format": "locatemot-l89-contract-smoke-v1", "status": "complete",
                                          "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": None, "next_action": "run L89 registered 40-epoch training",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        return 0
    except Exception:
        trace = __import__("traceback").format_exc()
        (out / "INCOMPLETE.md").write_text("# L89 contract smoke — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {"format": "locatemot-l89-contract-smoke-v1", "status": "incomplete",
                                          "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
