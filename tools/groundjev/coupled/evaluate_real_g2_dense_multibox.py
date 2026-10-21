#!/usr/bin/env python3
"""Evaluate the frozen G2 dense contract with all GT person boxes retained."""
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

from locatemot.groundjev.native_output_repair import repair_gam_bbox_output  # noqa: E402
from tools.groundjev.coupled.real_g2_memory_prefill import _answer, _generate_multi_box, _prepare  # noqa: E402
from tools.groundjev.coupled.train_real_g2_coupled_memory_jev import _iou  # noqa: E402


def _read_all_boxes(path: Path, *, frames: list[int], width: int, height: int) -> dict[int, list[tuple[int, int, int, int]]]:
    wanted = set(frames)
    result: dict[int, list[tuple[int, int, int, int]]] = {frame: [] for frame in frames}
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame = int(parts[0])
        if frame not in wanted:
            continue
        x, y, w, h = (float(parts[index]) for index in range(2, 6))
        result[frame].append((
            round(max(0.0, min(999.0, x / width * 999.0))),
            round(max(0.0, min(999.0, y / height * 999.0))),
            round(max(0.0, min(999.0, (x + w) / width * 999.0))),
            round(max(0.0, min(999.0, (y + h) / height * 999.0))),
        ))
    for frame in frames:
        result[frame].sort()
        if not result[frame]:
            raise RuntimeError(f"no boxes for frame {frame}")
    return result


def _row(image: Path, boxes: list[tuple[int, int, int, int]]) -> dict[str, Any]:
    return {
        "id": f"groundjev_dense_multibox_{image.stem}",
        "messages": [
            {"role": "user", "content": "<image>\nLocate all the instances that match the following categories: person.", "loss": None},
            {"role": "assistant", "content": _answer("person", boxes), "loss": None},
        ],
        "images": [{"path": str(image), "bytes": None}],
        "source": "GroundJEV:real_dense_multibox_contract",
    }


def _greedy_recall(predicted: list[list[int]], target: list[tuple[int, int, int, int]], threshold: float = 0.5) -> dict[str, float | int]:
    used: set[int] = set()
    matched = 0
    for box in target:
        best = max(((_iou(box, candidate), index) for index, candidate in enumerate(predicted) if index not in used), default=(0.0, -1))
        if best[0] >= threshold:
            matched += 1; used.add(best[1])
    return {"target_boxes": len(target), "predicted_boxes": len(predicted), "matched_boxes": matched,
            "recall_at_iou_0_5": matched / max(len(target), 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--g2-adapter", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    runtime_root = Path(args.runtime_root).resolve(); source_root = Path(args.source_root).resolve()
    os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(source_root)
    sys.path[:0] = [str(runtime_root), str(source_root)]
    from tools.groundjev.g2.compat import patch_transformers_causal_mask
    from tools.groundjev.g2.train_g2_dlm import load_model, _strict_load_trainable_state
    from train.dlm.data import DLMDataCollator
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("dense multibox evaluation requires CUDA")
    patch_transformers_causal_mask(); start = time.perf_counter()
    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    saved = torch.load(Path(args.g2_adapter).resolve(), map_location="cpu", weights_only=False)
    _strict_load_trainable_state(model, saved["model_trainable_state"])
    model.set_padding_free_packing(False); model.gradient_checkpointing_disable(); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    collator = DLMDataCollator(processor, max_length=32768, max_joint_length=65536, model_family="groundinganything_qwen3")
    specs = [
        ("MOT20-03", Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03"), Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03/gt/gt.txt"), 82, 1173, 880),
        ("MOT20-05", Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05"), Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05/gt/gt.txt"), 13, 1654, 1080),
    ]
    tokenizer = processor.tokenizer
    structural = {"box_start": int(tokenizer.convert_tokens_to_ids("<|box_start|>")), "box_end": int(tokenizer.convert_tokens_to_ids("<|box_end|>"))}
    coordinate_range = (int(tokenizer.convert_tokens_to_ids("<0>")), int(tokenizer.convert_tokens_to_ids("<999>")))
    stop_ids = {int(tokenizer.convert_tokens_to_ids("<|im_end|>"))}
    results = {}; visual_calls = 0
    for name, root, gt, start_frame, width, height in specs:
        frames = list(range(start_frame, start_frame + 3))
        boxes = _read_all_boxes(gt, frames=frames, width=width, height=height)
        rows = []
        for frame in frames:
            prepared = _prepare(model, collator, _row(root / "img1" / f"{frame:06d}.jpg", boxes[frame]), device)
            visual_calls += 1
            generated = _generate_multi_box(model, prepared, torch.empty((0, prepared["embeds"].shape[-1]), device=device), enabled=False,
                                            max_new_tokens=512, stop_ids=stop_ids, structural=structural, coordinate_range=coordinate_range)
            text = tokenizer.decode(generated["token_ids"], skip_special_tokens=False)
            recovered, report = repair_gam_bbox_output(text)
            predicted = [list(box) for _label, box in recovered]
            rows.append({"frame": frame, "target_boxes": [list(box) for box in boxes[frame]], "generated_text": text,
                         "recovered_pairs": [[label, list(box)] for label, box in recovered],
                         "recall": _greedy_recall(predicted, boxes[frame]), "repair_report": report,
                         "closed_boxes": generated["closed_boxes"], "grammar_forced_box_end": generated["grammar_forced_box_end"]})
        results[name] = {"rows": rows, "mean_recall_at_iou_0_5": sum(r["recall"]["recall_at_iou_0_5"] for r in rows) / len(rows)}
    receipt = {"status": "PHASE_C_REAL_G2_DENSE_MULTIBOX_CONTRACT_COMPLETE", "project_root": str(PROJECT_ROOT),
               "g2_adapter": str(Path(args.g2_adapter).resolve()), "adapter_step": int(saved["step"]),
               "base_model_sha256": args.model_sha256, "g2_frozen": True, "formal_g2_run_untouched": True,
               "visual_embed_calls": visual_calls, "independent_vlm_instances": 0, "results": results,
               "dense_multi_box_decoder": "official coordinate slots; stop only at im_end; no single-box truncation",
               "g2_trainable_parameters": audit.get("trainable_parameters"), "elapsed_seconds": time.perf_counter() - start,
               "scientific_claim": "dense G2 contract diagnostic only; no tracking or coupling gain claim"}
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(receipt,ensure_ascii=False),flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
