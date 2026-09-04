"""Auditable low-rank parametrizations for the L88 GroundingDINO branch.

The parametrization is attached to the actual weight tensor consumed by the
live MMDetection forward.  This is important for PyTorch/ MMCV
``MultiheadAttention``: wrapping a child ``out_proj`` module is not sufficient
because the underlying ``in_proj_weight`` and ``out_proj.weight`` are read
directly by ``torch.nn.functional.multi_head_attention_forward``.

Only the A/B factors returned by :func:`adapter_state_dict` are checkpointed;
the pretrained detector is always rebuilt from its immutable checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn.utils import parametrize


RANK = 16
ALPHA = 32.0
SCALE = ALPHA / RANK


class LoRAWeight(nn.Module):
    """A differentiable ``W + alpha/rank * B @ A`` parametrization."""

    def __init__(self, out_features: int, in_features: int, *, rank: int = RANK,
                 alpha: float = ALPHA) -> None:
        super().__init__()
        if int(rank) <= 0 or float(alpha) <= 0:
            raise ValueError("rank and alpha must be positive")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.A = nn.Parameter(torch.empty(self.rank, int(in_features), dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(int(out_features), self.rank, dtype=torch.float32))
        # Standard low-rank initialization: the exact zero B makes the full
        # detector forward equal to the pretrained model at initialization.
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        # Keep the adapter master factors in FP32, while matching the base
        # weight's dtype/device at the point of use.
        delta = (self.B @ self.A).to(device=base.device, dtype=base.dtype) * self.scale
        return base + delta


@dataclass
class LoRATarget:
    module_path: str
    parameter_name: str
    shape: tuple[int, ...]
    rank: int
    alpha: float
    scale: float
    lora: LoRAWeight

    @property
    def key(self) -> str:
        return f"{self.module_path}.{self.parameter_name}"

    @property
    def parameter_count(self) -> int:
        return int(self.lora.A.numel() + self.lora.B.numel())


def _get_module(root: nn.Module, path: str) -> nn.Module:
    value: nn.Module = root
    for part in path.split(".") if path else ():
        if not hasattr(value, part):
            raise AttributeError(f"module path not found: {path}")
        value = getattr(value, part)
        if not isinstance(value, nn.Module):
            raise TypeError(f"not a module at {path}: {type(value).__name__}")
    return value


def _get_tensor(module: nn.Module, name: str) -> torch.Tensor:
    value = getattr(module, name, None)
    if not isinstance(value, torch.Tensor):
        raise AttributeError(f"parameter tensor not found: {name}")
    return value


class LoRAInjector:
    """Owns registered targets and exposes only the adapter state."""

    def __init__(self, model: nn.Module, targets: list[LoRATarget]) -> None:
        self.model = model
        self.targets = targets
        self._by_key = {target.key: target for target in targets}
        if len(self._by_key) != len(targets):
            raise AssertionError("duplicate L88 LoRA target")

    @property
    def parameter_count(self) -> int:
        return int(sum(target.parameter_count for target in self.targets))

    def parameters(self) -> Iterable[nn.Parameter]:
        for target in self.targets:
            yield target.lora.A
            yield target.lora.B

    def trainable_named_parameters(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for target in self.targets:
            result[f"{target.key}.A"] = int(target.lora.A.numel())
            result[f"{target.key}.B"] = int(target.lora.B.numel())
        return result

    def adapter_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            target.key: {
                "A": target.lora.A.detach().cpu().clone(),
                "B": target.lora.B.detach().cpu().clone(),
            }
            for target in self.targets
        }

    def load_adapter_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        expected = set(self._by_key)
        actual = {str(key) for key in state}
        if strict and actual != expected:
            raise AssertionError(f"L88 LoRA target state mismatch missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}")
        for key, target in self._by_key.items():
            if key not in state:
                continue
            item = state[key]
            if set(item) != {"A", "B"}:
                raise AssertionError(f"invalid L88 factors for {key}")
            a = torch.as_tensor(item["A"], dtype=torch.float32)
            b = torch.as_tensor(item["B"], dtype=torch.float32)
            if tuple(a.shape) != tuple(target.lora.A.shape) or tuple(b.shape) != tuple(target.lora.B.shape):
                raise AssertionError(f"L88 factor shape drift for {key}: {tuple(a.shape)} / {tuple(b.shape)}")
            target.lora.A.data.copy_(a.to(target.lora.A))
            target.lora.B.data.copy_(b.to(target.lora.B))

    def zero_update(self) -> dict[str, torch.Tensor]:
        saved = {target.key: target.lora.B.detach().clone() for target in self.targets}
        with torch.no_grad():
            for target in self.targets:
                target.lora.B.zero_()
        return saved

    def restore_update(self, saved: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for target in self.targets:
                if target.key not in saved:
                    raise KeyError(target.key)
                target.lora.B.copy_(saved[target.key].to(target.lora.B))

    def base_parameter_digest(self) -> str:
        """Digest only original (non-LoRA) parametrized tensors."""
        digest = hashlib.sha256()
        for target in self.targets:
            module = _get_module(self.model, target.module_path)
            original = getattr(module, "parametrizations")[target.parameter_name].original
            digest.update(target.key.encode())
            digest.update(original.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "format": "locatemot-l88-lora-target-manifest-v1",
            "rank": RANK,
            "alpha": ALPHA,
            "scale": SCALE,
            "dropout": 0.0,
            "bias": "frozen/no adapter bias",
            "zero_initialized_B": all(bool(torch.equal(target.lora.B.detach(), torch.zeros_like(target.lora.B))) for target in self.targets),
            "targets": [
                {
                    "key": target.key,
                    "module_path": target.module_path,
                    "parameter_name": target.parameter_name,
                    "shape": list(target.shape),
                    "A_shape": list(target.lora.A.shape),
                    "B_shape": list(target.lora.B.shape),
                    "rank": target.rank,
                    "alpha": target.alpha,
                    "scale": target.scale,
                    "parameter_count": target.parameter_count,
                    "base_requires_grad": bool(getattr(_get_module(self.model, target.module_path), "parametrizations")[target.parameter_name].original.requires_grad),
                }
                for target in self.targets
            ],
            "target_count": len(self.targets),
            "adapter_parameter_count": self.parameter_count,
            "trainable_named_parameters": self.trainable_named_parameters(),
        }


def _register(model: nn.Module, module_path: str, parameter_name: str, *, rank: int, alpha: float,
              targets: list[LoRATarget]) -> None:
    # Callers may specify ``out_proj.weight`` or ``v_proj.weight`` while the
    # parametrization API itself needs the immediate parent module and leaf
    # parameter name.
    if "." in parameter_name:
        relative_parent, leaf = parameter_name.rsplit(".", 1)
        module_path = f"{module_path}.{relative_parent}"
        parameter_name = leaf
    module = _get_module(model, module_path)
    tensor = _get_tensor(module, parameter_name)
    if tensor.ndim != 2:
        raise ValueError(f"L88 LoRA requires matrix {module_path}.{parameter_name}, got {tuple(tensor.shape)}")
    if parameter_name in getattr(module, "parametrizations", {}):
        raise AssertionError(f"target already parametrized: {module_path}.{parameter_name}")
    tensor.requires_grad_(False)
    lora = LoRAWeight(int(tensor.shape[0]), int(tensor.shape[1]), rank=rank, alpha=alpha)
    # Parametrization uses the original Parameter for the frozen base and
    # evaluates LoRAWeight.forward at every actual linear/MHA weight access.
    parametrize.register_parametrization(module, parameter_name, lora, unsafe=False)
    original = module.parametrizations[parameter_name].original
    original.requires_grad_(False)
    targets.append(LoRATarget(module_path, parameter_name, tuple(int(x) for x in tensor.shape), rank, alpha, alpha / rank, lora))


def _mha_targets(model: nn.Module, module_path: str, *, rank: int, alpha: float,
                 targets: list[LoRATarget]) -> None:
    module = _get_module(model, module_path)
    # MMCV MultiheadAttention owns a torch.nn.MultiheadAttention at .attn.
    # Parametrize the underlying matrices, never a possibly bypassed child.
    if not hasattr(module, "attn"):
        raise AssertionError(f"L88 expected MMCV MHA wrapper at {module_path}")
    _register(model, f"{module_path}.attn", "in_proj_weight", rank=rank, alpha=alpha, targets=targets)
    _register(model, f"{module_path}.attn", "out_proj.weight", rank=rank, alpha=alpha, targets=targets)


def inject_lora(model: nn.Module, *, rank: int = RANK, alpha: float = ALPHA,
                expected_fusion_indices: tuple[int, int] | None = None) -> LoRAInjector:
    """Attach the preregistered L88 target set to a live detector.

    Fusion targets are discovered from the actual encoder length, with the
    final two blocks selected by position.  The decoder target is layer zero
    only.  A missing authorized submodule is a contract error, not a reason to
    silently broaden the target set.
    """
    if int(rank) != RANK or float(alpha) != ALPHA:
        raise AssertionError(f"L88 rank/alpha are fixed at {RANK}/{ALPHA}")
    fusion = getattr(getattr(model, "encoder", None), "fusion_layers", None)
    if fusion is None or len(fusion) < 2:
        raise AssertionError("GroundingDINO fusion_layers contract unavailable")
    fusion_indices = (len(fusion) - 2, len(fusion) - 1)
    if expected_fusion_indices is not None and tuple(expected_fusion_indices) != fusion_indices:
        raise AssertionError(f"fusion index drift: expected {expected_fusion_indices}, got {fusion_indices}")
    targets: list[LoRATarget] = []
    fusion_names = ("v_proj", "l_proj", "values_v_proj", "values_l_proj", "out_v_proj", "out_l_proj")
    for index in fusion_indices:
        path = f"encoder.fusion_layers.{index}.attn"
        for name in fusion_names:
            _register(model, path, f"{name}.weight", rank=rank, alpha=alpha, targets=targets)

    _mha_targets(model, "decoder.layers.0.self_attn", rank=rank, alpha=alpha, targets=targets)
    _mha_targets(model, "decoder.layers.0.cross_attn_text", rank=rank, alpha=alpha, targets=targets)

    deform_path = "decoder.layers.0.cross_attn"
    deform = _get_module(model, deform_path)
    for name in ("sampling_offsets", "attention_weights", "value_proj", "output_proj"):
        if hasattr(deform, name) and isinstance(getattr(deform, name), nn.Linear):
            _register(model, deform_path, f"{name}.weight", rank=rank, alpha=alpha, targets=targets)
        elif name in ("sampling_offsets", "attention_weights", "value_proj", "output_proj"):
            raise AssertionError(f"authorized decoder deformable target unavailable: {deform_path}.{name}")

    ffn = _get_module(model, "decoder.layers.0.ffn")
    ffn_linears = [(name, value) for name, value in ffn.named_modules() if isinstance(value, nn.Linear)]
    if not ffn_linears:
        raise AssertionError("decoder layer 0 FFN has no linear weights")
    for relative, _value in ffn_linears:
        path = "decoder.layers.0.ffn" if not relative else f"decoder.layers.0.ffn.{relative}"
        _register(model, path, "weight", rank=rank, alpha=alpha, targets=targets)

    injector = LoRAInjector(model, targets)
    # No base parameter outside the adapter factors is allowed to train.
    for name, parameter in model.named_parameters():
        if ".parametrizations." in name and (name.endswith(".A") or name.endswith(".B")):
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(False)
    return injector


def adapter_grad_report(injector: LoRAInjector) -> dict[str, Any]:
    values = []
    nonzero = 0
    finite = True
    for name, parameter in ((f"{target.key}.A", target.lora.A) for target in injector.targets):
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        values.append((name, float(grad.norm())))
        nonzero += int(float(grad.norm()) > 0.0)
        finite = finite and bool(torch.isfinite(grad).all())
    for target in injector.targets:
        parameter = target.lora.B
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        values.append((f"{target.key}.B", float(grad.norm())))
        nonzero += int(float(grad.norm()) > 0.0)
        finite = finite and bool(torch.isfinite(grad).all())
    return {
        "parameter_count": injector.parameter_count,
        "gradient_entries": len(values),
        "nonzero_gradient_entries": nonzero,
        "gradient_norm_sum": float(sum(value for _name, value in values)),
        "finite": finite,
        "by_parameter": {name: value for name, value in values},
    }


__all__ = ["ALPHA", "RANK", "SCALE", "LoRAInjector", "LoRAWeight", "adapter_grad_report", "inject_lora"]
