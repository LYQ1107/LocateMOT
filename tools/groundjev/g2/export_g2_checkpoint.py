#!/usr/bin/env python3
"""Export the trained G2 projector and LoRA into a deployable DLM bundle.

The training checkpoint contains only trainable tensors.  This exporter applies
the projector tensors and the mathematically equivalent LoRA deltas to a fresh
copy of the published GroundAnything bundle.  The original bundle is never
modified, and the portable adapter is copied into the versioned output so the
LoRA weights are not lost after producing the merged model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = PROJECT_ROOT / "models/GroundAnything-DLM"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_bundle_without_weights(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for entry in source.iterdir():
        if entry.name == "model.safetensors":
            continue
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, symlinks=True)
        else:
            shutil.copy2(entry, target, follow_symlinks=False)


def apply_trainable_state(
    base: Dict[str, torch.Tensor],
    trainable: Dict[str, torch.Tensor],
    *,
    rank: int,
    alpha: float,
) -> dict[str, object]:
    projector_prefix = "base_model.model.visual.projector."
    lora_suffixes = (".lora_A.weight", ".lora_B.weight")
    replacements: list[str] = []
    merged_lora: list[str] = []
    changed_elements = 0

    for name, value in trainable.items():
        if name.startswith(projector_prefix):
            if name not in base:
                raise KeyError(f"projector tensor missing from base bundle: {name}")
            if tuple(base[name].shape) != tuple(value.shape):
                raise ValueError(f"projector shape mismatch for {name}: {tuple(value.shape)} vs {tuple(base[name].shape)}")
            replacement = value.detach().to(dtype=base[name].dtype).contiguous()
            changed_elements += int((base[name] != replacement).sum())
            base[name] = replacement
            replacements.append(name)

    for name, value in trainable.items():
        if not name.endswith(lora_suffixes[0]):
            continue
        stem = name[: -len(lora_suffixes[0])]
        b_name = stem + lora_suffixes[1]
        base_name = stem + ".weight"
        if b_name not in trainable:
            raise KeyError(f"LoRA B tensor missing for {name}")
        if base_name not in base:
            raise KeyError(f"base projection missing for {base_name}")
        a = value.detach().float()
        b = trainable[b_name].detach().float()
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"unexpected LoRA shapes for {stem}: A={tuple(a.shape)} B={tuple(b.shape)} rank={rank}")
        if tuple(b.shape[:1] + a.shape[1:]) != tuple(base[base_name].shape):
            raise ValueError(f"LoRA/base shape mismatch for {base_name}: A={tuple(a.shape)} B={tuple(b.shape)} base={tuple(base[base_name].shape)}")
        delta = (b @ a) * (float(alpha) / float(rank))
        merged = (base[base_name].float() + delta).to(dtype=base[base_name].dtype).contiguous()
        changed_elements += int((base[base_name] != merged).sum())
        base[base_name] = merged
        merged_lora.append(stem)

    if not replacements:
        raise RuntimeError("G2 adapter has no projector tensors")
    if len(merged_lora) * 2 != sum(name.endswith(suffix) for name in trainable for suffix in lora_suffixes):
        raise RuntimeError("G2 adapter contains an incomplete LoRA projection set")
    return {
        "projector_tensors_replaced": len(replacements),
        "projector_tensor_names": sorted(replacements),
        "lora_projections_merged": sorted(merged_lora),
        "lora_rank": int(rank),
        "lora_alpha": float(alpha),
        "changed_elements_after_bf16_cast": int(changed_elements),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    args = parser.parse_args()

    base = args.base.resolve(strict=True)
    adapter = args.adapter.resolve(strict=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {args.out}")
    if not (base / "model.safetensors").is_file():
        raise FileNotFoundError(base / "model.safetensors")

    payload = torch.load(adapter, map_location="cpu", weights_only=True)
    if payload.get("format") not in {
        "groundanything-g2-adapter-projector-v1",
        "groundanything-g2-adapter-projector-v2",
    }:
        raise RuntimeError(f"unexpected adapter format: {payload.get('format')!r}")
    if int(payload.get("step", -1)) != int(args.step):
        raise RuntimeError(f"adapter step mismatch: {payload.get('step')} != {args.step}")
    if payload.get("manifest_sha256") != args.manifest_sha256:
        raise RuntimeError("adapter/data manifest SHA mismatch")
    trainable = payload["state"]
    if not isinstance(trainable, dict):
        raise RuntimeError("adapter state is not a tensor mapping")

    # This is the only large allocation in the merge.  The released bundle has
    # 569 tensors; keeping the entire state in memory lets safetensors write an
    # atomic, complete native model without touching the source weights.
    merged_state = load_file(str(base / "model.safetensors"), device="cpu")
    merge = apply_trainable_state(merged_state, trainable, rank=args.rank, alpha=args.alpha)
    copy_bundle_without_weights(base, args.out)
    save_file(
        merged_state,
        str(args.out / "model.safetensors"),
        metadata={
            "base_model": str(base),
            "g2_training_step": str(args.step),
            "g2_manifest_sha256": args.manifest_sha256,
            "g2_export": "projector_plus_qwen3_lora_merged",
        },
    )
    # Keep an adapter-only artifact alongside the merged deployment bundle.
    # The full optimizer/RNG recovery checkpoint remains in the run directory.
    shutil.copy2(adapter, args.out / "g2_adapter_projector_final.pt")

    receipt = {
        "status": "G2_NATIVE_MERGED_BUNDLE_EXPORTED",
        "base_bundle": str(base),
        "base_model_safetensors_sha256": sha256(base / "model.safetensors"),
        "output_bundle": str(args.out.resolve()),
        "output_model_safetensors_sha256": sha256(args.out / "model.safetensors"),
        "adapter": str(adapter),
        "adapter_sha256": sha256(adapter),
        "step": int(args.step),
        "manifest_sha256": args.manifest_sha256,
        "native_tensor_count": len(merged_state),
        "source_bundle_overwritten": False,
        "merge": merge,
        "optimizer_recovery_checkpoint": str(adapter.parent / "checkpoints" / f"adapter-step{args.step:06d}.pt"),
        "lora_adapter_preserved_in_bundle": str((args.out / "g2_adapter_projector_final.pt").resolve()),
        "benchmark_improvement_claim": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
