#!/usr/bin/env python3
"""G2-A real forward/backward/optimizer/checkpoint gate.

The gate loads the published GroundAnything-DLM checkpoint, injects the
project-local LoRA module, and runs one dense MOT row, one referring RMOT row,
and one TAO-Amodal OVMOT row through the unchanged native DLM loss.  It is a
small engineering gate, not a benchmark and not a substitute for the full
ten-dataset run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ONEJEV = Path("/data2/user/LocateMOT_OneJev_CVPR")
sys.path[:0] = [str(ROOT), str(ROOT / "external/GroundAnything"), str(ONEJEV)]
os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")


def tao_row() -> dict[str, Any]:
    from grounding_anything.protocol import (
        GroundingEntry,
        build_dense_bbox_prompt,
        canonicalize_phrase,
        serialize_answer,
        validate_prompt_answer_pair,
    )

    frames_root = Path("/data2/user/TAO-Amodal")
    annotation_root = Path("/data2/user/LocateMOT/data/TAO-Amodal")
    annotation = json.loads((annotation_root / "annotations/train.json").read_text())
    categories = {int(item["id"]): item["name"] for item in annotation["categories"]}
    by_image: dict[int, list[dict[str, Any]]] = {}
    for item in annotation["annotations"]:
        by_image.setdefault(int(item["image_id"]), []).append(item)
    selected = None
    for image in annotation["images"]:
        path = frames_root / "frames" / image["file_name"]
        if not path.is_file():
            continue
        anns = by_image.get(int(image["id"]), [])
        if anns:
            selected = (image, path, anns)
            break
    if selected is None:
        raise RuntimeError("TAO-Amodal train frames/annotations cannot form an OVMOT row")
    image, path, anns = selected
    phrase = None
    chosen: list[dict[str, Any]] = []
    for category_id, name in categories.items():
        candidate = [item for item in anns if int(item["category_id"]) == category_id]
        if not candidate:
            continue
        try:
            canonical = canonicalize_phrase(name)
        except Exception:
            continue
        phrase, chosen = canonical, candidate
        break
    if phrase is None:
        raise RuntimeError("TAO-Amodal first train frame has no canonical category")
    width, height = int(image["width"]), int(image["height"])
    coordinates = []
    for item in chosen:
        x, y, w, h = [float(value) for value in item["bbox"]]
        x2, y2 = x + w, y + h
        coordinates.append(
            tuple(
                round(value / (width - 1 if index % 2 == 0 else height - 1) * 999)
                for index, value in enumerate((x, y, x2, y2))
            )
        )
    answer = serialize_answer([GroundingEntry(phrase, tuple(coordinates))], "bbox")
    prompt = build_dense_bbox_prompt([phrase])
    validate_prompt_answer_pair(prompt, answer, "dense_bbox")
    return {
        "id": f"tao_amodal_train_{image['id']}_{phrase}",
        "source": "tao_amodal_train",
        "images": [{"path": str(path), "bytes": None}],
        "messages": [
            {"role": "user", "content": "<image>\n" + prompt},
            {"role": "assistant", "content": answer},
        ],
        "audit": {"image_id": image["id"], "category": phrase, "boxes": len(chosen)},
    }


def make_rows(plan_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The historical native row builder lives in the OneJev research checkout;
    # import it by path so the active GroundJEV checkout cannot accidentally
    # resolve a different ROOT constant.
    import importlib.util

    # Its annotation helper is also versioned between the two checkouts.  Bind
    # the OneJev implementation under the import name used by the builder
    # before loading the builder itself.
    annotation_name = "tools.groundjev.prepare_tracking_decisions"
    annotation_source = ONEJEV / "tools/groundjev/prepare_tracking_decisions.py"
    annotation_spec = importlib.util.spec_from_file_location(annotation_name, annotation_source)
    if annotation_spec is None or annotation_spec.loader is None:
        raise RuntimeError(f"cannot load annotation helper: {annotation_source}")
    annotation_module = importlib.util.module_from_spec(annotation_spec)
    sys.modules[annotation_name] = annotation_module
    annotation_spec.loader.exec_module(annotation_module)

    source = ONEJEV / "tools/groundjev/native_ground_training_data.py"
    spec = importlib.util.spec_from_file_location("locatemot_onejev_native_ground_training_data", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native row builder: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_pairs = module.build_pairs

    plan = json.loads(plan_path.read_text())
    pairs, corpus = build_pairs(plan)
    dense = next(pair for pair in pairs if pair["family"] == "dense")["rows"][0]["sft"]
    refer = next(pair for pair in pairs if pair["family"] == "referring")["rows"][0]["sft"]
    return [dense, refer, tao_row()], {"native_builder_counts": corpus}


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_model(checkpoint: Path):
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from models.dlm.vlm import GAMQwen3DLM
    from tools.groundjev.g2.groundanything_lora import (
        freeze_all_except_projector_and_lora,
        inject_qwen3_lora,
        trainable_audit,
    )

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    raw = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    Config = get_class_from_dynamic_module(
        "configuration_groundinganything.GroundAnythingVLMConfig", str(checkpoint), local_files_only=True
    )
    Model = get_class_from_dynamic_module(
        "modeling_groundinganything.GroundAnythingVLMForConditionalGeneration", str(checkpoint), local_files_only=True
    )
    config = Config.from_dict(raw.to_dict())
    config._attn_implementation = "eager"
    # The published checkpoint carries one additional spatial token relative
    # to the generic config JSON.  Match the model shape to the actual tensor,
    # exactly as the verified native G1 loader does.
    with safe_open(checkpoint / "model.safetensors", framework="pt") as tensors:
        config.text_config.vocab_size = tensors.get_slice(
            "base_model.model.language_model.embed_tokens.weight"
        ).get_shape()[0]
    if config.text_config.vocab_size != len(processor.tokenizer):
        raise RuntimeError("checkpoint vocabulary and tokenizer length disagree")
    mask = processor.tokenizer.encode("|<MASK>|", add_special_tokens=False)
    if len(mask) != 1:
        raise RuntimeError(f"GroundAnything mask token is not atomic: {mask}")
    with init_empty_weights(include_buffers=False):
        model = GAMQwen3DLM(
            Model(config),
            mask[0],
            processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            block_size=32,
        )
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True, assign=True)
    injection = inject_qwen3_lora(model, rank=16, alpha=32, dropout=0.05)
    model.to("cuda:0")
    model.base_model.model.visual.projector.float()
    freeze_all_except_projector_and_lora(model)
    audit = trainable_audit(model)
    if audit["vision_trainable_parameters"] != 0:
        raise RuntimeError("vision tower unexpectedly trainable")
    if audit["projector_trainable_parameters"] <= 0 or audit["lora_trainable_parameters"] <= 0:
        raise RuntimeError(f"G2 trainable audit failed: {audit}")
    if audit["other_unexpected_trainable_parameters"]:
        raise RuntimeError(f"unexpected trainable modules: {audit}")
    model.train()
    model.config.use_cache = False
    return model, processor, injection, audit


def main() -> int:
    import numpy as np
    import torch
    from torch._functorch import config as functorch_config
    from train.dlm.data import DLMDataCollator
    from tools.groundjev.g2.compat import patch_transformers_causal_mask

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=ONEJEV / "configs/groundjev/native_ground_tracking_g1_20261018.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/GroundAnything-DLM")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("G2 gradient gate requires CUDA")
    functorch_config.donated_buffer = False
    mask_compatibility = patch_transformers_causal_mask()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    rows, corpus = make_rows(args.plan)
    model, processor, injection, audit = load_model(args.checkpoint)
    collator = DLMDataCollator(processor, max_length=2048, max_joint_length=4096, model_family="groundinganything_qwen3")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.base_model.model.visual.projector.parameters()), "lr": 2e-6},
            {"params": [p for name, p in model.named_parameters() if ".lora_A." in name or ".lora_B." in name], "lr": 7e-6},
        ],
        weight_decay=0.1,
    )
    records = []
    for row in rows:
        batch = move_batch(collator([row]), "cuda:0")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch)
            loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite G2 native loss for {row.get('source')}")
        projector = list(model.base_model.model.visual.projector.parameters())
        adapter = [p for name, p in model.named_parameters() if ".lora_A." in name or ".lora_B." in name]
        pgrads = torch.autograd.grad(loss, projector, retain_graph=True, allow_unused=True)
        agrads = torch.autograd.grad(loss, adapter, retain_graph=True, allow_unused=True)
        projector_norm = float(sum(g.float().square().sum() for g in pgrads if g is not None).sqrt())
        adapter_norm = float(sum(g.float().square().sum() for g in agrads if g is not None).sqrt())
        if projector_norm <= 0 or adapter_norm <= 0:
            raise RuntimeError(f"zero native gradients for {row.get('source')}: {projector_norm}, {adapter_norm}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not np.isfinite(grad_norm):
            raise RuntimeError("non-finite clipped gradient norm")
        if any(parameter.grad is not None for name, parameter in model.named_parameters() if not parameter.requires_grad):
            raise RuntimeError("a frozen G2 parameter received a gradient")
        optimizer.step()
        records.append({
            "source": row.get("source"),
            "id": row.get("id"),
            "loss": float(loss.detach()),
            "projector_gradient_norm": projector_norm,
            "lora_gradient_norm": adapter_norm,
            "clipped_gradient_norm": grad_norm,
        })
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "G2_A_NATIVE_FORWARD_BACKWARD_CHECKPOINT_COMPLETE",
        "checkpoint": str(args.checkpoint),
        "transformers_mask_compatibility": mask_compatibility,
        "device": "cuda:0 (physical GPU selected by CUDA_VISIBLE_DEVICES)",
        "injection": injection.__dict__,
        "trainable_audit": audit,
        "rows": records,
        "corpus": corpus,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "optimizer": "AdamW(projector_lr=2e-6,lora_lr=7e-6,weight_decay=0.1)",
        "checkpoint_gate": "optimizer step completed; adapter/projector state remains in live model and is exported by the formal trainer",
    }
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(out)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
