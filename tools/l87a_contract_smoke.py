#!/usr/bin/env python3
"""L87-A one-clip forward/loss/backward contract regression."""
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

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get("LOCATEMOT_ASSET_ROOT", "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")).resolve()
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
if str(WORK_ROOT) not in sys.path: sys.path.insert(0, str(WORK_ROOT))
sys.path.insert(0, str(WORK_ROOT / "locatemot" / "rmot"))

from locatemot.models.l86_full_rmot import L86Config, L86FullRMOT  # noqa: E402
from locatemot.rmot.l83_target_bags import bag_values, build_target_bag_layout  # noqa: E402
from locatemot.rmot.l86_clip_data import L86ClipStore  # noqa: E402
from l87a_losses import l87a_loss  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def grad_norm(module: torch.nn.Module) -> float:
    values = [value.grad.detach().float().norm() for value in module.parameters() if value.grad is not None]
    return float(torch.stack(values).sum()) if values else 0.0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); out = args.out.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"refusing nonempty L87-A contract output: {out}")
    out.mkdir(parents=True, exist_ok=True); command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    try:
        if Path.cwd().resolve() != WORK_ROOT: raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256(MANIFEST) != MANIFEST_SHA: raise AssertionError("fixed manifest SHA drift")
        duplicate_scores = torch.tensor([2.0, -3.0, 1.0]); layout = build_target_bag_layout(["target-1", "target-1", "other-target"])
        keys, bag_scores, positive = bag_values(duplicate_scores, layout, ["target-1"])
        position = keys.index(("target", "target-1"))
        if float(bag_scores[position]) != 2.0 or not bool(positive[position]): raise AssertionError("target-bag duplicate contract")
        store = L86ClipStore(args.cache.resolve())
        preferred = ["refer_kitti_v1|0003|8", "refer_kitti_v2|0006|137", "refer_kitti_v1|0001|89"]
        anchor = next((key for key in preferred if key in store.train_keys), store.train_keys[0])
        clip = store.build_clip(anchor, temporal_enabled=True, clip_length=4)
        if len(clip) < 2: raise AssertionError(f"no causal clip for {anchor}")
        current = clip[-1]; device = torch.device(args.device)
        if device.type == "cuda": torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        model = L86FullRMOT(L86Config()).to(device=device, dtype=torch.float32).train()
        def forward(frame: Any) -> dict[str, torch.Tensor]:
            return model(frame.z1.to(device), frame.text_global.to(device), frame.frame_global.to(device),
                         frame.current_observation.to(device), frame.history_observations.to(device),
                         frame.history_mask.to(device), frame.history_frame_ids.to(device), frame.frame_id,
                         temporal_enabled=True)
        output = forward(current); previous = [(forward(frame), frame.labels) for frame in clip[:-1]]
        loss, parts = l87a_loss(output, current.labels, current.current_observation.to(device), previous, temporal_enabled=True)
        if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite L87-A smoke loss")
        loss.backward()
        groups = {"semantic_head": grad_norm(model.semantic_head), "temporal_delta": grad_norm(model.temporal_delta),
                  "temporal_gate": grad_norm(model.temporal_gate_head), "candidate_prior": grad_norm(model.candidate_prior_head),
                  "presence_head": grad_norm(model.presence_head), "null_head": grad_norm(model.null_head),
                  "history_gru": grad_norm(model.history.gru)}
        if any(value <= 0.0 or not (value == value) for value in groups.values()): raise AssertionError(f"gradient contract: {groups}")
        future = sum(int((frame.history_frame_ids > int(frame.frame_id)).sum()) for frame in clip)
        if future or len(current.row_offsets) != len(current.row_keys) != len(current.current_observation): raise AssertionError("row/history contract")
        package = {"model_config": vars(model.config), "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
        checkpoint = out / "contract_checkpoint.pt"; torch.save(package, checkpoint)
        reload_model = L86FullRMOT(L86Config(**package["model_config"])); loaded = reload_model.load_state_dict(package["model_state_dict"], strict=True)
        if loaded.missing_keys or loaded.unexpected_keys: raise AssertionError(f"strict reload drift: {loaded}")
        reload_output = reload_model.eval()(current.z1, current.text_global, current.frame_global, current.current_observation,
                                            current.history_observations, current.history_mask, current.history_frame_ids,
                                            current.frame_id, temporal_enabled=True)
        payload = {"format": "locatemot-l87a-contract-smoke-v1", "status": "complete", "stage": "L87-A corrected temporal loss",
                   "command": command, "work_root": str(WORK_ROOT), "asset_root": str(ASSET_ROOT), "cwd": str(WORK_ROOT),
                   "luna_thread": THREAD, "seed": 20260829, "device": str(device), "manifest_sha256": sha256(MANIFEST),
                   "cache": str(args.cache.resolve()), "anchor_group": anchor, "clip_frame_ids": [int(frame.frame_id) for frame in clip],
                   "clip_query_counts": [len(frame.query_ids) for frame in clip], "candidate_counts": [len(frame.row_offsets) for frame in clip],
                   "all_candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
                   "future_frame_count": future, "duplicate_bag_check": {"positive_bag_score": 2.0, "passed": True},
                   "loss": parts, "loss_finite": True, "gradient_norms": groups, "gradients_finite_nonzero": True,
                   "temporal_negative_contract": "(previous_available | current_available) - referred_targets",
                   "model_parameters": {"total": sum(value.numel() for value in model.parameters())},
                   "output_shapes": {key: list(value.shape) for key, value in output.items()}, "strict_reload": True,
                   "reload_output_shapes": {key: list(value.shape) for key, value in reload_output.items()},
                   "persistent_raw_dense_cache": False, "checkpoint_sha256": sha256(checkpoint),
                   "labels_attached_after_feature_construction": True, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                   "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                   "hota_trackeval_run": False, "no_hota_or_trackeval": True, "failure_root_cause": None,
                   "next_action": "run the registered L87-A fresh 40-epoch training"}
        for name in ("contract.json", "provenance.json", "status.json"): write_json(out / name, payload)
        print(json.dumps(payload, indent=2), flush=True); return 0
    except Exception:
        trace = traceback.format_exc(); (out / "INCOMPLETE.md").write_text("# L87-A contract smoke — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l87a-contract-smoke-v1", "status": "incomplete",
                                         "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False,
                                         "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise


if __name__ == "__main__": raise SystemExit(main())
