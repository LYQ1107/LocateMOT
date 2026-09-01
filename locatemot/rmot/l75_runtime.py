"""Runtime and representation helpers for the L75 candidate-marked sidecar.

The helper keeps the large LocateAnything model in memory only for the active
unit.  It never writes visual features.  A frozen visual forward is made once
per image; candidate-specific visual sequences are then constructed in
memory by adding a small marker to the absolute merged image-token cells.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from locatemot.models.l73_postfusion import overlap_indices

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
MODEL_DIR = ROOT / "models/LocateAnything-3B"
PROMPT_TEMPLATE = "Judge whether the marked candidate matches: {expression}"
IMAGE_TOKEN_INDEX = 151665
SEED = 20260829
MERGE_KERNEL = (2, 2)
PATCH_SIZE = 14


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_file_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(MODEL_DIR.rglob("*")):
        if path.is_file():
            files.append({
                "path": str(path),
                "relative": str(path.relative_to(MODEL_DIR)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"root": str(MODEL_DIR), "files": files,
            "manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def _flat_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [int(item) for item in value[0]]
    return [int(item) for item in value]


def find_subsequence(values: list[int], query: list[int]) -> list[int]:
    if not query:
        return []
    for start in range(0, len(values) - len(query) + 1):
        if values[start:start + len(query)] == query:
            return list(range(start, start + len(query)))
    return []


def image_positions(input_ids: torch.Tensor, image_token_index: int) -> list[int]:
    values = _flat_ids(input_ids)
    return [index for index, value in enumerate(values) if value == int(image_token_index)]


def expression_positions(tokenizer: Any, expression: str, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor | None,
                         image_pos: list[int]) -> tuple[list[int], str]:
    full = _flat_ids(input_ids)
    valid = ([True] * len(full) if attention_mask is None else
             [bool(value) for value in attention_mask.detach().cpu().reshape(-1).tolist()])
    excluded = set(image_pos)
    for method, text in (("exact", expression), ("leading_space", " " + expression)):
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        encoded = _flat_ids(encoded)
        positions = find_subsequence(full, encoded)
        if positions and all(valid[pos] and pos not in excluded for pos in positions):
            return positions, method
    fallback = [index for index, flag in enumerate(valid)
                if flag and index not in excluded]
    return fallback, "whole_text_minus_image_unresolved"


def build_messages(processor: Any, image: Any, text: str) -> tuple[dict[str, Any], str]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }]
    prompt = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[prompt], images=images, videos=videos, return_tensors="pt")
    return inputs, prompt


def load_locateanything(device: str = "cuda:0") -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load the audited local model without any remote fallback."""
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    import transformers

    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, attn_implementation="sdpa",
    ).to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("LocateAnything base is not fully frozen")
    # The local model's config, rather than a caller guess, is authoritative.
    image_token_index = int(getattr(model.config, "image_token_index", IMAGE_TOKEN_INDEX))
    runtime = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "model_dtype": str(next(model.parameters()).dtype),
        "device": str(device),
        "image_token_index": image_token_index,
        "model_manifest": model_file_manifest(),
        "offline": True,
        "base_parameters_frozen": True,
    }
    return model, processor, tokenizer, runtime


def prepare_visual(model: Any, processor: Any, tokenizer: Any, image: Any,
                   expression: str, boxes: list[list[float]]) -> dict[str, Any]:
    """Prepare one image/expression and all candidate cell maps.

    ``extract_feature`` and ``mlp1`` are called exactly once here.  The
    returned ``base_visual`` is detached and cloned before any marker can be
    connected to an adapter graph.
    """
    inputs, prompt = build_messages(processor, image, PROMPT_TEMPLATE.format(expression=expression))
    device = next(model.parameters()).device
    image_token_index = int(getattr(model.config, "image_token_index", IMAGE_TOKEN_INDEX))
    ids_cpu = inputs["input_ids"].detach().cpu()
    mask_cpu = inputs.get("attention_mask")
    mask_cpu = mask_cpu.detach().cpu() if mask_cpu is not None else None
    image_pos = image_positions(ids_cpu, image_token_index)
    if not image_pos:
        raise AssertionError("prompt contains no image token positions")
    expr_pos, span_method = expression_positions(
        tokenizer, expression, ids_cpu, mask_cpu, image_pos
    )
    if not expr_pos:
        raise AssertionError("prompt contains no usable expression positions")
    grid_cpu = torch.as_tensor(inputs["image_grid_hws"], dtype=torch.int32).cpu()
    if grid_cpu.ndim != 2 or grid_cpu.shape[-1] != 2 or grid_cpu.shape[0] != 1:
        raise AssertionError(f"unexpected image_grid_hws shape {tuple(grid_cpu.shape)}")
    pixel = inputs["pixel_values"].to(device=device, dtype=next(model.vision_model.parameters()).dtype)
    grid = grid_cpu.to(device=device, dtype=torch.int32)
    with torch.no_grad():
        raw_list = model.extract_feature(pixel, grid)
        if not isinstance(raw_list, (list, tuple)) or len(raw_list) != 1:
            raise AssertionError(f"unexpected visual feature container {type(raw_list).__name__}")
        raw = torch.cat(raw_list, dim=0)
        projected = model.mlp1(raw)
        base_visual = projected.detach().clone()
    if base_visual.ndim != 2 or base_visual.shape[0] != len(image_pos):
        raise AssertionError(
            f"visual/image token mismatch {tuple(base_visual.shape)} vs {len(image_pos)}"
        )
    if not bool(torch.isfinite(base_visual.float()).all()):
        raise AssertionError("nonfinite projected visual values")
    processed = processor.image_processor.rescale(image, list(MERGE_KERNEL))
    original_size = (int(image.width), int(image.height))
    processed_size = (int(processed.width), int(processed.height))
    grid_hw = [int(value) for value in grid_cpu[0].tolist()]
    cells = []
    mapping = []
    for box in boxes:
        item = overlap_indices(
            box, original_size, processed_size, grid_hw,
            patch_size=PATCH_SIZE, merge_kernel=MERGE_KERNEL,
        )
        indices = [int(value) for value in item["indices"]]
        if any(value < 0 or value >= len(image_pos) for value in indices):
            raise AssertionError("candidate cell outside image-token lattice")
        cells.append(indices)
        mapping.append(item)
    del pixel, grid, raw_list, raw, projected, inputs
    return {
        "base_visual": base_visual,
        "input_ids": ids_cpu,
        "attention_mask": mask_cpu,
        "image_positions": image_pos,
        "expression_positions": expr_pos,
        "expression_span_method": span_method,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "image_grid_hw": grid_hw,
        "processed_size": [processed_size[0], processed_size[1]],
        "original_size": [original_size[0], original_size[1]],
        "candidate_cells": cells,
        "candidate_mappings": mapping,
        "projected_visual_shape": [int(value) for value in base_visual.shape],
        "projected_visual_finite": bool(torch.isfinite(base_visual.float()).all()),
    }


def marked_visual_batch(base_visual: torch.Tensor, cell_lists: list[list[int]],
                        marker: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build candidate-specific visual sequences and a complete cell mask."""
    if base_visual.ndim != 2:
        raise AssertionError(f"base visual must be [M,D], got {tuple(base_visual.shape)}")
    batch = len(cell_lists)
    if batch == 0:
        raise AssertionError("empty candidate batch")
    m, dim = base_visual.shape
    if marker.shape != (dim,):
        raise AssertionError(f"marker shape {tuple(marker.shape)} != {(dim,)}")
    mask = torch.zeros((batch, m, 1), device=base_visual.device, dtype=marker.dtype)
    for index, cells in enumerate(cell_lists):
        if any(int(cell) < 0 or int(cell) >= m for cell in cells):
            raise AssertionError("marker cell outside visual sequence")
        if cells:
            mask[index, torch.as_tensor(sorted(set(int(c) for c in cells)), device=mask.device), 0] = 1.0
    base = base_visual.unsqueeze(0).expand(batch, -1, -1).clone()
    marked = base + mask.to(dtype=base.dtype) * marker.to(dtype=base.dtype).view(1, 1, -1)
    if not bool(torch.isfinite(marked.float()).all()):
        raise AssertionError("nonfinite candidate-marked visual values")
    return marked, mask.squeeze(-1).bool()


def region_value_batch(marked_visual: torch.Tensor, cell_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad all candidate cell values, retaining an explicit fallback for empty boxes."""
    if marked_visual.ndim != 3 or marked_visual.shape[0] != len(cell_lists):
        raise AssertionError("marked visual/candidate list mismatch")
    batch, _, dim = marked_visual.shape
    max_cells = max(1, max((len(cells) for cells in cell_lists), default=0))
    values = marked_visual.new_zeros((batch, max_cells, dim))
    valid = torch.zeros((batch, max_cells), device=marked_visual.device, dtype=torch.bool)
    global_value = marked_visual.mean(dim=1)
    for index, cells in enumerate(cell_lists):
        unique = sorted(set(int(cell) for cell in cells))
        if unique:
            take = torch.as_tensor(unique, device=marked_visual.device, dtype=torch.long)
            values[index, :len(unique)] = marked_visual[index].index_select(0, take)
            valid[index, :len(unique)] = True
        else:
            # Keep an explicit row for an empty geometrical mapping.  This is
            # not candidate deletion and is reported as mapping_empty.
            values[index, 0] = global_value[index]
            valid[index, 0] = True
    return values, valid


def language_forward(model: Any, prepared: dict[str, Any], marked_visual: torch.Tensor,
                     inference: bool = False) -> torch.Tensor:
    """Run the local Qwen decoder with a candidate-marked visual sequence."""
    device = next(model.parameters()).device
    batch = int(marked_visual.shape[0])
    ids = prepared["input_ids"].to(device=device).expand(batch, -1).clone()
    mask_cpu = prepared.get("attention_mask")
    mask = mask_cpu.to(device=device).expand(batch, -1).clone() if mask_cpu is not None else None
    visual = marked_visual.to(device=device)
    kwargs = dict(
        input_ids=ids,
        visual_features=visual,
        image_token_index=int(getattr(model.config, "image_token_index", IMAGE_TOKEN_INDEX)),
        attention_mask=mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    context = torch.inference_mode() if inference else torch.enable_grad()
    with context:
        output = model.language_model.model(**kwargs)
        hidden = output.last_hidden_state
    if hidden.ndim != 3 or hidden.shape[0] != batch:
        raise AssertionError(f"unexpected language hidden shape {tuple(hidden.shape)}")
    if not bool(torch.isfinite(hidden.float()).all()):
        raise AssertionError("nonfinite candidate-marked language hidden")
    return hidden


class LoRALinear(nn.Module):
    """Auditable FP32 rank-r update around one frozen linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"L75 LoRA target must be nn.Linear, got {type(base).__name__}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = nn.Dropout(float(dropout))
        # The detector is loaded onto GPU before wrappers are attached.  Keep
        # the small trainable matrices on the wrapped linear's device; a CPU
        # default here would only fail once the first LoRA projection runs.
        target_device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(
            self.rank, base.in_features, dtype=torch.float32, device=target_device
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            base.out_features, self.rank, dtype=torch.float32, device=target_device
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))
        self.scaling = self.alpha / float(self.rank)
        self.enabled = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base_out = self.base(value)
        if not self.enabled:
            return base_out
        source = self.dropout(value).float()
        delta = F.linear(F.linear(source, self.lora_A), self.lora_B) * self.scaling
        return base_out + delta.to(dtype=base_out.dtype)


def attach_language_lora(model: Any, rank: int = 8, alpha: float = 16.0,
                         target_layers: int = 4) -> dict[str, Any]:
    """Wrap exactly q/k/v/o in the final four local Qwen layers."""
    decoder = model.language_model.model
    layers = getattr(decoder, "layers", None)
    if layers is None or len(layers) < int(target_layers):
        raise AssertionError("cannot locate enough Qwen decoder layers")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    start = len(layers) - int(target_layers)
    targets = []
    for layer_index in range(start, len(layers)):
        attention = layers[layer_index].self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            original = getattr(attention, name)
            if isinstance(original, LoRALinear):
                raise AssertionError(f"LoRA already attached at layer {layer_index} {name}")
            wrapped = LoRALinear(original, rank=rank, alpha=alpha, dropout=0.0)
            setattr(attention, name, wrapped)
            targets.append({
                "module": f"language_model.model.layers.{layer_index}.self_attn.{name}",
                "layer": layer_index, "name": name, "rank": int(rank),
                "alpha": float(alpha), "dropout": 0.0,
                "A_shape": list(wrapped.lora_A.shape),
                "B_shape": list(wrapped.lora_B.shape),
                "A_dtype": str(wrapped.lora_A.dtype),
                "B_dtype": str(wrapped.lora_B.dtype),
                "B_zero_initialized": True,
            })
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if len(trainable) != len(targets) * 2:
        raise AssertionError(f"unexpected L75 LoRA trainable parameter count: {len(trainable)}")
    return {
        "target_layers": list(range(start, len(layers))),
        "targets": targets,
        "rank": int(rank), "alpha": float(alpha), "dropout": 0.0,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def set_lora_enabled(model: Any, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = bool(enabled)


def lora_modules(model: Any) -> dict[str, LoRALinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def lora_state_dict(model: Any) -> dict[str, torch.Tensor]:
    result = {}
    for name, module in lora_modules(model).items():
        result[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
        result[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
    return result


def load_lora_state_dict(model: Any, state: dict[str, torch.Tensor], strict: bool = True) -> dict[str, Any]:
    modules = lora_modules(model)
    expected = {f"{name}.{suffix}" for name in modules for suffix in ("lora_A", "lora_B")}
    actual = set(state)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if strict and (missing or unexpected):
        raise AssertionError(f"LoRA state mismatch missing={missing} unexpected={unexpected}")
    for name, module in modules.items():
        for suffix, target in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
            key = f"{name}.{suffix}"
            if key in state:
                source = state[key].to(device=target.device, dtype=target.dtype)
                if tuple(source.shape) != tuple(target.shape):
                    raise AssertionError(f"LoRA shape mismatch at {key}")
                target.data.copy_(source)
    return {"missing": missing, "unexpected": unexpected, "strict": bool(strict)}


def frozen_target_digest(model: Any) -> str:
    """Content digest for all wrapped base linears (plus names/shapes)."""
    digest = hashlib.sha256()
    for name, module in sorted(lora_modules(model).items()):
        digest.update(name.encode())
        for suffix, parameter in (("weight", module.base.weight), ("bias", module.base.bias)):
            if parameter is None:
                continue
            value = parameter.detach().float().cpu().contiguous()
            digest.update(f"{suffix}:{tuple(value.shape)}".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def trainable_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
