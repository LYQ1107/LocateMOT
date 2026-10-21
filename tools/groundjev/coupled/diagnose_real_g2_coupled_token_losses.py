#!/usr/bin/env python3
"""Trace C0/C2 teacher-forced token probabilities after coupled training."""
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.groundjev.coupled.causal_memory import causal_prefill_with_memory  # noqa: E402
from locatemot.groundjev.coupled.memory_tokens import IdentityMemoryTokenProjector  # noqa: E402
from tools.groundjev.coupled.train_real_g2_coupled_memory_jev import _read_boxes  # noqa: E402
from tools.groundjev.coupled.real_g2_memory_prefill import _prepare, _row  # noqa: E402


def _trace(model: Any, prepared: dict[str, Any], memory: torch.Tensor, *, enabled: bool,
           tokenizer: Any) -> dict[str, Any]:
    target = prepared["answer_tokens"].to(prepared["embeds"].device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prefill = causal_prefill_with_memory(
            model.language_model, model.lm_head,
            prepared["embeds"], prepared["mask"], prepared["positions"], memory,
            native_position_ids=model._native_position_ids,
            enabled=enabled, inference_mode=False,
        )
        cache = prefill["outputs"].past_key_values
        if cache is None:
            raise RuntimeError("token trace did not receive a KV cache")
        logits = prefill["logits"]
        axes = int(prefill["prepared"].position_ids.shape[0])
        batch_size = int(prefill["prepared"].inputs_embeds.shape[0])
        rope_deltas = getattr(model.multimodal_model, "rope_deltas", None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.detach().clone()
        rows = []
        for index in range(int(target.numel())):
            vector = logits[0].float()
            probabilities = torch.softmax(vector, dim=-1)
            token_id = int(target[index])
            greedy_id = int(vector.argmax())
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            greedy_text = tokenizer.decode([greedy_id], skip_special_tokens=False)
            is_coordinate = token_text.startswith("<") and token_text.endswith(">") and token_text[1:-1].isdigit()
            rows.append({
                "position": index,
                "target_token_id": token_id,
                "target_text": token_text,
                "greedy_token_id": greedy_id,
                "greedy_text": greedy_text,
                "target_probability": float(probabilities[token_id]),
                "target_loss": float((-torch.log(probabilities[token_id].clamp_min(1e-12))).cpu()),
                "is_coordinate_token": is_coordinate,
            })
            if index == int(target.numel()) - 1:
                break
            token = target[index].reshape(1, 1)
            position = int(cache.get_seq_length())
            positions = model._generation_position_ids(
                torch.tensor([position], device=token.device, dtype=torch.long),
                axes, batch_size, rope_deltas,
            )
            outputs = model.language_model(
                input_ids=token, attention_mask=None,
                position_ids=model._native_position_ids(positions),
                past_key_values=cache, use_cache=True,
                cache_position=torch.tensor([position], device=token.device, dtype=torch.long),
                return_dict=True,
            )
            logits = model.lm_head(outputs.last_hidden_state[:, -1])
            cache = outputs.past_key_values
    coordinate = [row for row in rows if row["is_coordinate_token"]]
    non_coordinate = [row for row in rows if not row["is_coordinate_token"]]
    return {
        "memory_enabled": enabled,
        "memory_token_count": int(memory.shape[0]),
        "target_text": tokenizer.decode(target.tolist(), skip_special_tokens=False),
        "rows": rows,
        "coordinate_token_count": len(coordinate),
        "coordinate_mean_loss": sum(row["target_loss"] for row in coordinate) / max(len(coordinate), 1),
        "non_coordinate_mean_loss": sum(row["target_loss"] for row in non_coordinate) / max(len(non_coordinate), 1),
        "greedy_token_accuracy": sum(row["target_token_id"] == row["greedy_token_id"] for row in rows) / max(len(rows), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--g2-adapter", required=True)
    parser.add_argument("--coupled-checkpoint", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    for prefix, default_root, default_gt, default_prev, default_current, default_id, default_width, default_height in (
        ("a", "/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03",
         "/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03/gt/gt.txt", 82, 83, 27, 1173, 880),
        ("b", "/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05",
         "/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05/gt/gt.txt", 13, 14, 17, 1654, 1080),
    ):
        parser.add_argument(f"--{prefix}-root", default=default_root)
        parser.add_argument(f"--{prefix}-gt", default=default_gt)
        parser.add_argument(f"--{prefix}-previous-frame", type=int, default=default_prev)
        parser.add_argument(f"--{prefix}-current-frame", type=int, default=default_current)
        parser.add_argument(f"--{prefix}-target-id", type=int, default=default_id)
        parser.add_argument(f"--{prefix}-width", type=int, default=default_width)
        parser.add_argument(f"--{prefix}-height", type=int, default=default_height)
    args = parser.parse_args()
    runtime_root = Path(args.runtime_root).resolve()
    source_root = Path(args.source_root).resolve()
    os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(source_root)
    sys.path[:0] = [str(runtime_root), str(source_root)]
    from tools.groundjev.g2.compat import patch_transformers_causal_mask
    from tools.groundjev.g2.train_g2_dlm import load_model, _strict_load_trainable_state
    from train.dlm.data import DLMDataCollator

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("token trace requires CUDA")
    patch_transformers_causal_mask()
    start = time.perf_counter()
    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    saved = torch.load(Path(args.g2_adapter).resolve(), map_location="cpu", weights_only=False)
    _strict_load_trainable_state(model, saved["model_trainable_state"])
    coupled = torch.load(Path(args.coupled_checkpoint).resolve(), map_location="cpu", weights_only=False)
    model.set_padding_free_packing(False); model.gradient_checkpointing_disable(); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    collator = DLMDataCollator(processor, max_length=32768, max_joint_length=65536,
                               model_family="groundinganything_qwen3")
    traces = []
    input_dim = None; hidden = None
    for prefix in ("a", "b"):
        root = Path(getattr(args, f"{prefix}_root")).resolve()
        gt = Path(getattr(args, f"{prefix}_gt")).resolve()
        previous_frame = int(getattr(args, f"{prefix}_previous_frame")); current_frame = int(getattr(args, f"{prefix}_current_frame"))
        target_id = int(getattr(args, f"{prefix}_target_id"))
        boxes = _read_boxes(gt, target_id=target_id, frame_start=previous_frame,
                            frame_count=current_frame - previous_frame + 1,
                            width=int(getattr(args, f"{prefix}_width")), height=int(getattr(args, f"{prefix}_height")))
        previous_image = root / "img1" / f"{previous_frame:06d}.jpg"
        current_image = root / "img1" / f"{current_frame:06d}.jpg"
        previous = _prepare(model, collator, _row(previous_image, "person", [boxes[previous_frame]]), device)
        current = _prepare(model, collator, _row(current_image, "person", [boxes[current_frame]]), device)
        input_dim = int(previous["image_tokens"].shape[-1]); hidden = int(current["embeds"].shape[-1])
        projector = IdentityMemoryTokenProjector(input_dim, hidden, hidden_dim=512).to(device)
        projector.load_state_dict(coupled["prompt_memory_projector"]); projector.eval()
        summary = previous["image_tokens"].float().mean(dim=0)
        memory = projector({target_id: summary}, ordered_ids=[target_id])
        empty = torch.empty((0, hidden), device=device)
        traces.append({
            "sequence": root.name, "previous_frame": previous_frame, "current_frame": current_frame,
            "target_id": target_id, "target_box_0_999": list(boxes[current_frame]),
            "visual_embed_calls": 2,
            "conditions": {
                "C0_disabled": _trace(model, current, empty, enabled=False, tokenizer=processor.tokenizer),
                "C2_trained_memory": _trace(model, current, memory, enabled=True, tokenizer=processor.tokenizer),
            },
        })
    result = {
        "status": "PHASE_C_REAL_G2_COUPLED_TOKEN_LOSS_DIAGNOSTIC_COMPLETE",
        "project_root": str(PROJECT_ROOT),
        "g2_adapter": str(Path(args.g2_adapter).resolve()),
        "coupled_checkpoint": str(Path(args.coupled_checkpoint).resolve()),
        "base_model_sha256": args.model_sha256,
        "traces": traces,
        "visual_embed_calls": 4,
        "independent_vlm_instances": 0,
        "parameter_updates": 0,
        "formal_g2_run_untouched": True,
        "elapsed_seconds": time.perf_counter() - start,
        "scientific_claim": "coordinate-token probability diagnosis only; no performance claim",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
