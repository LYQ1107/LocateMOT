"""Private L79 CLIP runtime and differentiable multi-scale feature extraction."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
CLIP_WEIGHT = Path("/home/lwr/.cache/clip/ViT-B-16.pt").resolve()
CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VISUAL_TAP_BLOCKS = (3, 7, 11)
LORA_BLOCKS = (8, 9, 10, 11)


def preprocess_full_frame(path: str | Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Resize the complete wide KITTI frame to 224x224; no center crop."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(CLIP_STD, dtype=torch.float32)[:, None, None]
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device=device, dtype=dtype)


class LoRALinear(nn.Module):
    """A private FP32 LoRA wrapper around a frozen CLIP linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 32, alpha: float = 16.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"L79 LoRA target must be nn.Linear, got {type(base)}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = 0.0
        self.enabled = False
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=torch.float32, device=base.weight.device))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the wrapped weight for torch MultiheadAttention internals."""
        if not self.enabled:
            return self.base.weight
        delta = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        return self.base.weight + delta.to(dtype=self.base.weight.dtype)

    @property
    def bias(self) -> torch.Tensor | None:
        """Expose the wrapped bias for torch MultiheadAttention internals."""
        return self.base.bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base_value = self.base(value)
        if not self.enabled:
            return base_value
        adapted = F.linear(F.linear(value.float(), self.lora_A), self.lora_B)
        return base_value + (adapted * self.scaling).to(dtype=base_value.dtype)


def attach_lora(clip_model: nn.Module, enabled: bool = False, rank: int = 32, alpha: float = 16.0) -> list[str]:
    visual = clip_model.visual
    names = []
    for block_index in LORA_BLOCKS:
        block = visual.transformer.resblocks[block_index]
        if not isinstance(block.attn.out_proj, LoRALinear):
            block.attn.out_proj = LoRALinear(block.attn.out_proj, rank=rank, alpha=alpha)
        block.attn.out_proj.enabled = bool(enabled)
        names.extend([
            f"visual.transformer.resblocks.{block_index}.attn.out_proj.lora_A",
            f"visual.transformer.resblocks.{block_index}.attn.out_proj.lora_B",
        ])
    for parameter in clip_model.parameters():
        if "lora_A" not in str(parameter) and "lora_B" not in str(parameter):
            parameter.requires_grad_(False)
    set_lora_enabled(clip_model, enabled)
    return names


def set_lora_enabled(clip_model: nn.Module, enabled: bool) -> None:
    for block_index in LORA_BLOCKS:
        target = clip_model.visual.transformer.resblocks[block_index].attn.out_proj
        if not isinstance(target, LoRALinear):
            raise AssertionError("L79 LoRA attachment drifted")
        target.enabled = bool(enabled)
        target.lora_A.requires_grad_(bool(enabled))
        target.lora_B.requires_grad_(bool(enabled))


def lora_parameters(clip_model: nn.Module) -> Iterable[nn.Parameter]:
    for module in clip_model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def lora_state_dict(clip_model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in clip_model.named_parameters()
        if name.endswith("lora_A") or name.endswith("lora_B")
    }


def load_lora_state_dict(clip_model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = {name for name, _ in clip_model.named_parameters() if name.endswith("lora_A") or name.endswith("lora_B")}
    if set(state) != expected:
        raise AssertionError(f"L79 LoRA key mismatch: expected {sorted(expected)}, got {sorted(state)}")
    with torch.no_grad():
        for name, value in state.items():
            target = dict(clip_model.named_parameters())[name]
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def load_clip_visual(device: torch.device, enable_lora: bool = False) -> nn.Module:
    if not CLIP_WEIGHT.is_file():
        raise FileNotFoundError(CLIP_WEIGHT)
    import clip  # local OpenAI CLIP package; local path avoids any download
    model, _preprocess = clip.load(str(CLIP_WEIGHT), device=device, jit=False)
    model.eval()
    # The locally installed OpenAI CLIP implementation's LayerNorm explicitly
    # casts activations to FP32.  Keeping the frozen base in FP32 and using a
    # CUDA BF16 autocast scope in ``visual_pyramid`` is the smallest safe
    # compatibility contract; it avoids an invalid BF16 LayerNorm weight/input
    # pairing while retaining BF16 matmul kernels where supported.
    model.to(device=device)
    attach_lora(model, enabled=enable_lora)
    return model


def visual_pyramid(clip_model: nn.Module, image: torch.Tensor, with_grad: bool = False) -> torch.Tensor:
    """Return `[3,B,196,768]` taps after visual blocks 3/7/11."""
    visual = clip_model.visual
    context = torch.enable_grad() if with_grad else torch.no_grad()
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if image.device.type == "cuda" else nullcontext()
    with amp, context:
        x = image.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        taps = []
        for index, block in enumerate(visual.transformer.resblocks):
            x = block(x)
            if index in VISUAL_TAP_BLOCKS:
                taps.append(x.permute(1, 0, 2)[:, 1:, :])
        if len(taps) != 3:
            raise AssertionError(f"CLIP pyramid tap count drift: {len(taps)}")
        result = torch.stack(taps, dim=0)
        if result.shape[2:] != (196, 768):
            raise AssertionError(f"unexpected CLIP pyramid shape: {tuple(result.shape)}")
        if not bool(torch.isfinite(result.float()).all()):
            raise AssertionError("nonfinite CLIP visual pyramid")
        return result


class MemoryFrameCache:
    """Small process-local cache; no tensors are persisted to disk."""

    def __init__(self, max_items: int = 32) -> None:
        self.max_items = int(max_items)
        self.values: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()

    def get(self, key: tuple[str, int]) -> torch.Tensor | None:
        value = self.values.pop(key, None)
        if value is not None:
            self.values[key] = value
        return value

    def put(self, key: tuple[str, int], value: torch.Tensor) -> None:
        self.values.pop(key, None)
        self.values[key] = value.detach()
        while len(self.values) > self.max_items:
            _old_key, old_value = self.values.popitem(last=False)
            del old_value

    def clear(self) -> None:
        self.values.clear()


def l79_task_enabled(task: str) -> bool:
    """RMOT-only guard; ordinary/OVMOT tasks bypass without loading L79."""
    return str(task).lower() in {"rmot", "refer_kitti_v1", "refer_kitti_v2", "refer_dance"}


__all__ = [
    "CLIP_SHA256", "CLIP_WEIGHT", "LORA_BLOCKS", "LoRALinear", "MemoryFrameCache",
    "attach_lora", "load_clip_visual", "load_lora_state_dict", "lora_parameters",
    "lora_state_dict", "preprocess_full_frame", "set_lora_enabled", "visual_pyramid",
    "l79_task_enabled",
]
