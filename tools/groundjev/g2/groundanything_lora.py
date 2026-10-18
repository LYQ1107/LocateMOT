"""Small, DLM-preserving LoRA injection used by the GroundAnything G2 run.

The official GroundAnything DLM wrapper is a training-time model rather than a
``transformers.PreTrainedModel``.  PEFT's ``CAUSAL_LM`` wrapper therefore tries
to access generation-only methods that the DLM intentionally does not expose.
This module keeps the original DLM object and replaces only registered Qwen3
linear projections with a transparent LoRA module.  The base projection remains
available for the native forward path and is frozen; adapter state can be saved
and merged without changing the spatial-token protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen linear projection plus a low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base)!r}")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        # Start exactly at the frozen G0 function.  B=0 is the standard LoRA
        # initialization and lets the first optimizer update be measured as a
        # genuine adapter effect.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    @torch.no_grad()
    def merge_(self) -> nn.Linear:
        """Merge the adapter into the frozen projection and return that layer."""
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base.weight.add_(delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        return self.base


@dataclass(frozen=True)
class LoRAInjection:
    names: tuple[str, ...]
    rank: int
    alpha: float
    dropout: float


def _replace_child(parent: nn.Module, name: str, replacement: nn.Module) -> None:
    if name not in parent._modules:
        raise KeyError(f"module child disappeared while injecting LoRA: {name}")
    parent._modules[name] = replacement


def inject_qwen3_lora(
    model: nn.Module,
    *,
    target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> LoRAInjection:
    """Replace exact Qwen3 projection leaf names while preserving DLM forward."""
    targets = set(target_modules)
    if not targets:
        raise ValueError("at least one LoRA target module is required")
    replaced: list[str] = []
    for full_name, module in list(model.named_modules()):
        if not full_name or full_name.rsplit(".", 1)[-1] not in targets:
            continue
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"registered LoRA target is not linear: {full_name}: {type(module)!r}")
        parent_name, child_name = full_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        _replace_child(
            parent,
            child_name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        replaced.append(full_name)
    if not replaced:
        raise RuntimeError("no registered Qwen3 projection was found for LoRA injection")
    return LoRAInjection(tuple(sorted(replaced)), int(rank), float(alpha), float(dropout))


def freeze_all_except_projector_and_lora(model: nn.Module) -> None:
    """Apply the G2 parameter policy after adapter injection."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projector = model.base_model.model.visual.projector
    for parameter in projector.parameters():
        parameter.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)


def trainable_audit(model: nn.Module) -> dict[str, object]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    projector = [(n, p) for n, p in trainable if "base_model.model.visual.projector" in n]
    lora = [(n, p) for n, p in trainable if ".lora_A." in n or ".lora_B." in n]
    unexpected = [
        n for n, _ in trainable
        if n not in {name for name, _ in projector + lora}
    ]
    return {
        "vision_trainable_parameters": 0,
        "projector_trainable_parameters": int(sum(p.numel() for _, p in projector)),
        "lora_trainable_parameters": int(sum(p.numel() for _, p in lora)),
        "original_language_base_trainable_parameters": 0,
        "other_unexpected_trainable_parameters": unexpected,
        "trainable_names": [n for n, _ in trainable],
    }


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only Projector and LoRA tensors for a portable G2 artifact."""
    state = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            state[name] = parameter.detach().cpu().clone()
    return state


def merge_all_lora(model: nn.Module) -> list[str]:
    """Merge every injected adapter in place and replace it by its base layer."""
    merged: list[str] = []
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parent_name, child_name = full_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        _replace_child(parent, child_name, module.merge_())
        merged.append(full_name)
    return sorted(merged)
