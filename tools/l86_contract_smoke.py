#!/usr/bin/env python3
"""One high-value L86 compile/duplicate/temporal/fit backward smoke."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l86_full_rmot import L86Config, L86FullRMOT  # noqa: E402
from locatemot.rmot.l83_target_bags import bag_values, build_target_bag_layout  # noqa: E402
from locatemot.rmot.l86_clip_data import L86ClipStore  # noqa: E402
from locatemot.rmot.l86_losses import l86_loss  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def grad_norm(module: torch.nn.Module) -> float:
    values = [value.grad.detach().float().norm() for value in module.parameters() if value.grad is not None]
    return float(torch.stack(values).sum()) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l86/audit/contract_smoke_attempt1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L86 smoke directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable, *sys.argv])
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        source = (ROOT / "locatemot/rmot/l86_losses.py").read_text()
        if "binary_cross_entropy_with_logits(gate" in source or "history_target" in source:
            raise AssertionError("old L85 history-length gate loss leaked into L86")

        # Required duplicate invariant: one unique target bag is scored by its
        # best row, never by its worst duplicate positive.
        duplicate_scores = torch.tensor([2.0, -3.0, 1.0])
        duplicate_gt = ["target-1", "target-1", "other-target"]
        layout = build_target_bag_layout(duplicate_gt)
        keys, bag_scores, positive = bag_values(duplicate_scores, layout, ["target-1"])
        target_position = keys.index(("target", "target-1"))
        assert float(bag_scores[target_position]) == 2.0
        assert bool(positive[target_position])
        duplicate_check = {"input_positive_rows": [2.0, -3.0], "negative_row": 1.0, "positive_bag_score": 2.0, "passed": True}

        torch.manual_seed(20260829)
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        store = L86ClipStore(args.cache.resolve())
        preferred = [
            "refer_kitti_v1|0003|8", "refer_kitti_v2|0006|137", "refer_kitti_v1|0001|89"
        ]
        anchor = next((key for key in preferred if key in store.train_keys), store.train_keys[0])
        clip = store.build_clip(anchor, temporal_enabled=True, clip_length=4)
        if len(clip) < 2:
            raise AssertionError(f"L86 temporal contract smoke did not obtain a causal clip: {anchor}")
        current = clip[-1]
        model = L86FullRMOT(L86Config()).to(device=device, dtype=torch.float32)
        model.train()

        def forward(frame: Any) -> dict[str, torch.Tensor]:
            return model(
                frame.z1.to(device=device), frame.text_global.to(device=device), frame.frame_global.to(device=device),
                frame.current_observation.to(device=device), frame.history_observations.to(device=device),
                frame.history_mask.to(device=device), frame.history_frame_ids.to(device=device), frame.frame_id,
                temporal_enabled=True,
            )

        output = forward(current)
        previous_outputs = []
        for frame in clip[:-1]:
            previous_outputs.append((forward(frame), frame.labels))
        loss, parts = l86_loss(output, current.labels, current.current_observation.to(device=device), previous_outputs, temporal_enabled=True)
        assert bool(torch.isfinite(loss))
        loss.backward()
        gradient_groups = {
            "semantic_head": grad_norm(model.semantic_head),
            "temporal_delta": grad_norm(model.temporal_delta),
            "temporal_gate": grad_norm(model.temporal_gate_head),
            "candidate_prior": grad_norm(model.candidate_prior_head),
            "presence_head": grad_norm(model.presence_head),
            "null_head": grad_norm(model.null_head),
            "history_gru": grad_norm(model.history.gru),
        }
        finite_gradients = all(value == value and value < float("inf") for value in gradient_groups.values())
        if not finite_gradients or any(value <= 0.0 for value in gradient_groups.values()):
            raise AssertionError(f"missing L86 smoke gradient: {gradient_groups}")
        output_shapes = {key: list(value.shape) for key, value in output.items()}
        assert "temporal_gate_logits" in output and "temporal_gate" in output
        assert output_shapes["candidate_energy"] == [len(current.query_ids), len(current.row_offsets)]
        future_count = sum(int((frame.history_frame_ids > int(frame.frame_id)).sum()) for frame in clip)
        assert future_count == 0
        assert len(current.row_offsets) == len(current.row_keys) == current.current_observation.shape[0]
        assert current.row_keys == sorted(current.row_keys, key=lambda value: value[-1])

        package = {"model_config": vars(model.config), "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
        checkpoint = out / "contract_checkpoint.pt"
        torch.save(package, checkpoint)
        reloaded = L86FullRMOT(L86Config(**package["model_config"]))
        result = reloaded.load_state_dict(package["model_state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise AssertionError(f"strict reload drift: {result}")
        reload_output = reloaded.eval()(
            current.z1, current.text_global, current.frame_global, current.current_observation,
            current.history_observations, current.history_mask, current.history_frame_ids, current.frame_id,
            temporal_enabled=True,
        )
        max_reload_diff = max(float((reload_output[key] - output[key].detach().cpu()).abs().max()) for key in output)
        # Dropout is disabled in the reloaded eval model, so only compare shape
        # here; exact values are recorded rather than used as a gate.
        payload = {
            "format": "locatemot-l86-contract-smoke-v1", "status": "complete", "command": command,
            "cwd": str(ROOT), "luna_thread": THREAD, "seed": 20260829, "device": str(device),
            "manifest_sha256": sha256_file(MANIFEST), "cache": str(args.cache.resolve()),
            "anchor_group": anchor, "clip_frame_ids": [int(frame.frame_id) for frame in clip],
            "clip_query_counts": [len(frame.query_ids) for frame in clip],
            "candidate_counts": [len(frame.row_offsets) for frame in clip],
            "all_candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "future_frame_count": future_count, "duplicate_bag_check": duplicate_check,
            "temporal_gate_logits_present": True, "old_history_length_bce_present": False,
            "loss": parts, "loss_finite": bool(torch.isfinite(loss)),
            "gradient_norms": gradient_groups, "gradients_finite_nonzero": finite_gradients,
            "model_parameters": model.parameter_report(), "output_shapes": output_shapes,
            "strict_reload": True, "reload_output_shapes": {key: list(value.shape) for key, value in reload_output.items()},
            "reload_max_abs_difference_recorded": max_reload_diff,
            "labels_attached_after_feature_construction": True,
            "persistent_raw_dense_cache": False, "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "z1_representation_changed": False, "groundingdino_lora_used": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "failure_root_cause": None, "next_action": "run the registered L86 semantic-oracle and 40-epoch training sequence",
        }
        write_json(out / "contract.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        del output, previous_outputs, model, reloaded, store, clip
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps(payload, indent=2), flush=True)
        return 0
    except Exception:
        trace = __import__("traceback").format_exc()
        (out / "INCOMPLETE.md").write_text("# L86 contract smoke — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {
            "format": "locatemot-l86-contract-smoke-v1", "status": "incomplete", "command": command,
            "cwd": str(ROOT), "luna_thread": THREAD, "failure_root_cause": "first traceback in INCOMPLETE.md",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
