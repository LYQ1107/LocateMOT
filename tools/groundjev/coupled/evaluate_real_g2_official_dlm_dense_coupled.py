#!/usr/bin/env python3
"""Evaluate a dense-trained memory projector on the official G2 DLM path."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
SOURCE_ROOT = RUNTIME_ROOT / "external/GroundAnything"
sys.path[:0] = [str(PROJECT_ROOT), str(RUNTIME_ROOT), str(SOURCE_ROOT)]

from locatemot.groundjev.coupled.memory_tokens import IdentityBoxStateTokenProjector  # noqa: E402
from locatemot.groundjev.native_output_repair import repair_gam_bbox_output  # noqa: E402
from tools.groundjev.coupled.evaluate_real_g2_dense_multibox import (  # noqa: E402
    _greedy_recall,
    _read_all_boxes,
    _row,
)
from tools.groundjev.coupled.evaluate_real_g2_official_dlm_dense import (  # noqa: E402
    _prepare_prompt,
    _token_contract,
)
from tools.groundjev.coupled.evaluate_real_g2_official_dlm_memory import _run_official  # noqa: E402
from tools.groundjev.coupled.train_real_g2_coupled_memory_jev import _read_boxes  # noqa: E402
from tools.groundjev.coupled.train_real_g2_coupled_memory_jev import _mean_token  # noqa: E402
from tools.groundjev.coupled.real_g2_memory_prefill import _prepare  # noqa: E402


def _image_path(root: Path, frame: int) -> Path:
    for width in (6, 8):
        candidate = root / "img1" / f"{frame:0{width}d}.jpg"
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"missing image frame {frame}: {root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--g2-adapter", required=True)
    parser.add_argument("--coupled-checkpoint", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(SOURCE_ROOT)
    from tools.groundjev.g2.compat import patch_transformers_causal_mask
    from tools.groundjev.g2.train_g2_dlm import _strict_load_trainable_state, load_model
    from train.dlm.data import DLMDataCollator
    from infer.dlm import DLMInferenceConfig, DLMInferenceEngine

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("official coupled evaluation requires CUDA")
    patch_transformers_causal_mask()
    start = time.perf_counter()
    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    saved = torch.load(Path(args.g2_adapter).resolve(), map_location="cpu", weights_only=False)
    _strict_load_trainable_state(model, saved["model_trainable_state"])
    coupled = torch.load(Path(args.coupled_checkpoint).resolve(), map_location="cpu", weights_only=False)
    model.set_padding_free_packing(False)
    model.gradient_checkpointing_disable()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    collator = DLMDataCollator(
        processor, max_length=32768, max_joint_length=65536, model_family="groundinganything_qwen3"
    )
    tokenizer = processor.tokenizer
    contract = _token_contract(tokenizer)
    config = DLMInferenceConfig(
        mode="hierarchy_dynamic", block_size=32, sub_block_size=8, denoise_steps=8,
        confidence_threshold=0.9, acceptance_policy="confidence", entropy_threshold=0.8,
        enforce_structured_termination=True, structured_max_coordinate_groups=0,
    )
    engine = DLMInferenceEngine(model)
    marker_token_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    specs = [
        {
            "name": "DanceTrack-0001",
            "root": Path("/data2/user/LocateMOT/data/DanceTrack/train/dancetrack0001"),
            "gt": Path("/data2/user/LocateMOT/data/DanceTrack/train/dancetrack0001/gt/gt.txt"),
            "width": 1280, "height": 720, "target_id": 1,
        },
        {
            "name": "DanceTrack-0016",
            "root": Path("/data2/user/LocateMOT/data/DanceTrack/train/dancetrack0016"),
            "gt": Path("/data2/user/LocateMOT/data/DanceTrack/train/dancetrack0016/gt/gt.txt"),
            "width": 1920, "height": 1080, "target_id": 1,
        },
    ]
    results: dict[str, Any] = {}
    visual_calls = 0
    for spec in specs:
        sequence_rows: list[dict[str, Any]] = []
        # Evaluate the trained transition and a temporal development transition.
        for previous_frame, current_frame in ((1, 2), (2, 3)):
            previous_all = _read_all_boxes(spec["gt"], frames=[previous_frame], width=spec["width"], height=spec["height"])[previous_frame]
            current_all = _read_all_boxes(spec["gt"], frames=[current_frame], width=spec["width"], height=spec["height"])[current_frame]
            previous_target = _read_boxes(
                spec["gt"], target_id=spec["target_id"], frame_start=previous_frame,
                frame_count=1, width=spec["width"], height=spec["height"],
            )[previous_frame]
            previous_features = _prepare(
                model, collator,
                _row(_image_path(spec["root"], previous_frame), previous_all),
                device,
            )
            visual_calls += 1
            current_prompt = _prepare_prompt(
                model, collator,
                _row(_image_path(spec["root"], current_frame), current_all),
                device,
            )
            input_dim = int(previous_features["image_tokens"].shape[-1])
            hidden = int(model.lm_head.weight.shape[-1])
            projector = IdentityBoxStateTokenProjector(input_dim, hidden, hidden_dim=512).to(device)
            projector.load_state_dict(coupled["prompt_box_state_projector"])
            projector.eval()
            memory = projector(
                {spec["target_id"]: _mean_token(previous_features["image_tokens"])},
                {spec["target_id"]: torch.tensor([value / 999.0 for value in previous_target], device=device)},
                ordered_ids=[spec["target_id"]],
            )
            c0 = _run_official(
                engine, config, model, tokenizer, current_prompt, contract,
                max_new_tokens=args.max_new_tokens,
            )
            visual_calls += 1
            c2 = _run_official(
                engine, config, model, tokenizer, current_prompt, contract,
                max_new_tokens=args.max_new_tokens, memory_tokens=memory,
                marker_token_id=marker_token_id,
            )
            visual_calls += 1
            for condition in (c0, c2):
                predicted = [pair[1] for pair in condition["recovered_pairs"]]
                condition["recall_at_iou_0_5"] = _greedy_recall(predicted, current_all)
                target_box = current_all[0]
                condition["best_iou_vs_first_dense_box"] = max(
                    (_iou(target_box, box) for box in predicted), default=0.0
                )
            sequence_rows.append({
                "previous_frame": previous_frame,
                "current_frame": current_frame,
                "target_box_previous": list(previous_target),
                "target_boxes_current": [list(box) for box in current_all],
                "C0_official": c0,
                "C2_trained_box_state_memory": c2,
            })
        results[spec["name"]] = {"rows": sequence_rows}

    receipt = {
        "status": "PHASE_C_REAL_G2_OFFICIAL_DLM_DENSE_COUPLED_EVALUATION_COMPLETE",
        "project_root": str(PROJECT_ROOT),
        "g2_checkpoint": str(Path(args.checkpoint).resolve()),
        "g2_adapter": str(Path(args.g2_adapter).resolve()),
        "coupled_checkpoint": str(Path(args.coupled_checkpoint).resolve()),
        "adapter_step": int(saved["step"]),
        "coupled_step": int(coupled["step"]),
        "base_model_sha256": args.model_sha256,
        "decoder": "official infer.dlm.DLMInferenceEngine hierarchy_dynamic",
        "coupling": "trained IdentityBoxStateTokenProjector + official marker-slot injection",
        "dense_answer_contract": True,
        "visual_embed_calls": visual_calls,
        "independent_vlm_instances": 0,
        "g2_frozen": True,
        "formal_g2_run_untouched": True,
        "results": results,
        "g2_trainable_parameters": audit.get("trainable_parameters"),
        "elapsed_seconds": time.perf_counter() - start,
        "scientific_claim": "bounded official-decoder coupled evaluation on two DanceTrack development clips; no MOT/RMOT/OVMOT benchmark gain claim",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1)


if __name__ == "__main__":
    raise SystemExit(main())
