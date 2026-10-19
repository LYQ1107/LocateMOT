#!/usr/bin/env python3
"""Export one immutable G2 recovery checkpoint for independent evaluation.

The formal G2 trainer stores a strict recovery payload containing trainable
projector/LoRA tensors together with Adam, scheduler, sampler and RNG state.
This tool validates that payload without changing it, extracts a deployable
adapter payload, and merges it into a fresh copy of the frozen
GroundAnything-DLM bundle.  It is intentionally separate from the final
``adapter_projector_final.pt`` export so an in-progress run can be measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


HERE = Path(__file__).resolve().parent
# ``.../<project>/tools/groundjev/g2`` -> project is parents[2].
PROJECT_ROOT = HERE.parents[2]
DEFAULT_BASE = PROJECT_ROOT / "models/GroundAnything-DLM"

sys.path.insert(0, str(HERE))
from export_g2_checkpoint import apply_trainable_state, copy_bundle_without_weights  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _manifest_sha(manifest_path: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_path = _resolve(manifest["rows_path"])
    observed = sha256(rows_path)
    declared = manifest.get("rows_sha256")
    if declared != observed:
        raise RuntimeError(
            f"G2 manifest rows SHA drift: declared={declared} observed={observed}"
        )
    return manifest, observed


def _validate_checkpoint(
    state: dict[str, Any],
    *,
    checkpoint_path: Path,
    base: Path,
    manifest_path: Path,
    expected_step: int | None,
    expected_world_size: int,
    expected_max_steps: int,
    expected_micro_batch: int,
    expected_grad_accum: int,
    expected_global_batch: int,
    expected_token_budget: int,
    expected_rank: int,
    expected_alpha: float,
    expected_dropout: float,
) -> tuple[dict[str, Any], str]:
    if state.get("format") != "groundanything-g2-projector-qwen3-lora-v2-strict":
        raise RuntimeError(f"unexpected strict checkpoint format: {state.get('format')!r}")
    required = {
        "step", "world_size", "base_checkpoint", "base_model_safetensors_sha256",
        "manifest_sha256", "model_trainable_state", "trainable_signature",
        "optimizer_state", "scheduler_state", "rng_state", "rank_rng_paths",
        "sampler", "lora_injection", "training_contract", "trainable_audit",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"strict checkpoint missing fields: {missing}")

    step = int(state["step"])
    if step <= 0 or (expected_step is not None and step != expected_step):
        raise RuntimeError(f"unexpected checkpoint step: {step} (expected {expected_step})")
    if int(state["world_size"]) != expected_world_size:
        raise RuntimeError(
            f"world-size drift: checkpoint={state['world_size']} expected={expected_world_size}"
        )

    manifest, manifest_sha = _manifest_sha(manifest_path)
    if state["manifest_sha256"] != manifest_sha:
        raise RuntimeError(
            f"checkpoint/manifest SHA mismatch: {state['manifest_sha256']} != {manifest_sha}"
        )
    if _resolve(state["base_checkpoint"]) != base:
        raise RuntimeError(
            f"checkpoint base path drift: {state['base_checkpoint']} != {base}"
        )
    base_sha = sha256(base / "model.safetensors")
    if state["base_model_safetensors_sha256"] != base_sha:
        raise RuntimeError(
            f"checkpoint/base SHA mismatch: {state['base_model_safetensors_sha256']} != {base_sha}"
        )

    injection = state["lora_injection"]
    if int(injection.get("rank", -1)) != expected_rank:
        raise RuntimeError(f"LoRA rank drift: {injection.get('rank')} != {expected_rank}")
    if float(injection.get("alpha", -1.0)) != float(expected_alpha):
        raise RuntimeError(f"LoRA alpha drift: {injection.get('alpha')} != {expected_alpha}")
    if float(injection.get("dropout", -1.0)) != float(expected_dropout):
        raise RuntimeError(
            f"LoRA dropout drift: {injection.get('dropout')} != {expected_dropout}"
        )
    names = list(injection.get("names", []))
    if not names or not all(name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"} for name in names):
        raise RuntimeError("strict checkpoint does not contain the registered Qwen3 q/k/v/o LoRA set")

    contract = state["training_contract"]
    expected_contract = {
        "max_steps": expected_max_steps,
        "micro_batch_size": expected_micro_batch,
        "grad_accum_steps": expected_grad_accum,
        "effective_global_batch": expected_global_batch,
        "token_budget_per_device": expected_token_budget,
        "world_size": expected_world_size,
    }
    contract_drift = {
        key: (contract.get(key), value)
        for key, value in expected_contract.items()
        if contract.get(key) != value
    }
    if contract_drift:
        raise RuntimeError(f"G2 training contract drift: {contract_drift}")
    objective = contract.get("objective")
    if objective != {"mdm_loss_weight": 0.5, "causal_loss_weight": 0.5}:
        raise RuntimeError(f"G2 objective drift: {objective!r}")

    trainable = state["model_trainable_state"]
    signature = state["trainable_signature"]
    if not isinstance(trainable, dict) or not trainable:
        raise RuntimeError("strict checkpoint has no trainable tensor mapping")
    if set(trainable) != set(signature):
        raise RuntimeError("trainable tensor/signature key drift")
    for name, value in trainable.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"trainable value is not a tensor: {name}")
        expected = signature[name]
        if list(value.shape) != list(expected["shape"]) or str(value.dtype) != expected["dtype"]:
            raise RuntimeError(
                f"trainable signature drift for {name}: shape={list(value.shape)} "
                f"dtype={value.dtype} expected={expected}"
            )
    projector = [name for name in trainable if name.startswith("base_model.model.visual.projector.")]
    lora = [name for name in trainable if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight")]
    if not projector or not lora or len(lora) % 2:
        raise RuntimeError("strict checkpoint has incomplete projector/LoRA state")

    rank_rng_paths = state["rank_rng_paths"]
    if set(rank_rng_paths) != {str(i) for i in range(expected_world_size)}:
        raise RuntimeError(f"rank RNG sidecars do not cover world size {expected_world_size}")
    for rank, entry in rank_rng_paths.items():
        sidecar = _resolve(entry["path"])
        if not sidecar.is_file() or sha256(sidecar) != entry["sha256"]:
            raise RuntimeError(f"rank RNG sidecar missing or changed: rank={rank} path={sidecar}")

    # Keep the original recovery payload out of the deployable bundle, but
    # prove it is a real file before any large model copy starts.
    checkpoint_sha = sha256(checkpoint_path)
    return manifest, checkpoint_sha


def _adapter_payload(state: dict[str, Any], manifest_sha: str) -> dict[str, Any]:
    return {
        "format": "groundanything-g2-adapter-projector-v2",
        "step": int(state["step"]),
        "state": state["model_trainable_state"],
        "trainable_audit": state["trainable_audit"],
        "manifest_sha256": manifest_sha,
        "base_checkpoint": state["base_checkpoint"],
        "training_contract": state["training_contract"],
        "source_strict_recovery_format": state["format"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=None)
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument("--expected-max-steps", type=int, default=18227)
    parser.add_argument("--expected-micro-batch", type=int, default=2)
    parser.add_argument("--expected-grad-accum", type=int, default=8)
    parser.add_argument("--expected-global-batch", type=int, default=32)
    parser.add_argument("--expected-token-budget", type=int, default=16384)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args()

    checkpoint = _resolve(args.checkpoint)
    manifest_path = _resolve(args.manifest)
    base = _resolve(args.base)
    out = _resolve(args.out)
    receipt_path = _resolve(args.receipt)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing midterm bundle: {out}")
    if not checkpoint.is_file() or not (base / "model.safetensors").is_file():
        raise FileNotFoundError("strict checkpoint or base model is missing")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    checkpoint_sha_before = sha256(checkpoint)
    base_sha_before = sha256(base / "model.safetensors")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError("strict checkpoint payload is not a mapping")
    manifest, checkpoint_sha = _validate_checkpoint(
        state,
        checkpoint_path=checkpoint,
        base=base,
        manifest_path=manifest_path,
        expected_step=args.expected_step,
        expected_world_size=args.expected_world_size,
        expected_max_steps=args.expected_max_steps,
        expected_micro_batch=args.expected_micro_batch,
        expected_grad_accum=args.expected_grad_accum,
        expected_global_batch=args.expected_global_batch,
        expected_token_budget=args.expected_token_budget,
        expected_rank=args.rank,
        expected_alpha=args.alpha,
        expected_dropout=args.dropout,
    )
    if checkpoint_sha != checkpoint_sha_before:
        raise RuntimeError("strict checkpoint changed while it was being read")

    adapter_path = out.with_name(out.name + ".adapter.pt")
    payload = _adapter_payload(state, manifest["rows_sha256"])
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, adapter_path)

    temporary = out.with_name(out.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse incomplete output: {temporary}")
    try:
        copy_bundle_without_weights(base, temporary)
        merged_state = load_file(str(base / "model.safetensors"), device="cpu")
        merge = apply_trainable_state(
            merged_state,
            payload["state"],
            rank=args.rank,
            alpha=args.alpha,
        )
        save_file(
            merged_state,
            str(temporary / "model.safetensors"),
            metadata={
                "base_model": str(base),
                "g2_training_step": str(state["step"]),
                "g2_manifest_sha256": manifest["rows_sha256"],
                "g2_export": "clean_midterm_projector_plus_qwen3_lora_merged",
            },
        )
        shutil.copy2(adapter_path, temporary / "g2_adapter_projector_midterm.pt")
        # Read the newly written safetensors once to prove all tensors are
        # loadable before exposing the directory as a completed model bundle.
        reloaded = load_file(str(temporary / "model.safetensors"), device="cpu")
        if set(reloaded) != set(merged_state):
            raise RuntimeError("midterm safetensors key set changed after write")
        temporary.replace(out)
    except Exception:
        # Keep no misleading receipt.  The incomplete directory is retained
        # for forensic inspection and cannot be mistaken for a valid bundle.
        raise

    if sha256(checkpoint) != checkpoint_sha_before or sha256(base / "model.safetensors") != base_sha_before:
        raise RuntimeError("source checkpoint/base changed during midterm export")
    receipt = {
        "status": "G2_CLEAN_MIDTERM_BUNDLE_EXPORTED",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "step": int(state["step"]),
        "base_bundle": str(base),
        "base_model_safetensors_sha256": base_sha_before,
        "output_bundle": str(out),
        "output_model_safetensors_sha256": sha256(out / "model.safetensors"),
        "adapter_payload": str(adapter_path),
        "adapter_sha256": sha256(adapter_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["rows_sha256"],
        "world_size": int(state["world_size"]),
        "training_contract": state["training_contract"],
        "lora_injection": state["lora_injection"],
        "merge": merge,
        "strict_recovery_checkpoint_untouched": True,
        "base_bundle_untouched": True,
        "optimizer_and_rng_preserved_at": str(checkpoint),
        "final_model_claim": False,
        "midterm_only": True,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
