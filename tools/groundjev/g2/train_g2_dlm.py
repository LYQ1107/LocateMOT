#!/usr/bin/env python3
"""Official-protocol GroundAnything G2 projector + Qwen3 LoRA trainer.

This is a small native DLM trainer, deliberately separate from the historical
OneJev SFT launcher.  It keeps the official 0.5 MDM + 0.5 causal objective,
freezes the vision tower and original language weights, and saves an atomic
adapter/projector/Adam/RNG/sampler receipt after the first update and at each
registered interval.  A deterministic source-round-robin sampler exposes all
available TRAIN rows while avoiding a first-epoch dominated by MOT17.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "external/GroundAnything"
sys.path[:0] = [str(ROOT), str(OFFICIAL)]
os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
# The released recipe prefers FlashAttention-2, but this environment does not
# ship the optional ``flash_attn`` wheel.  GroundAnything provides an official
# cuDNN varlen-SDPA fallback with the same packed stream contract; select it
# only when FlashAttention is genuinely unavailable.
if importlib.util.find_spec("flash_attn") is None:
    os.environ.setdefault("GAM_DLM_PACKED_CLEAN_ATTENTION_BACKEND", "cudnn_sdpa")
PACKED_LM_HEAD_LOSS_BACKEND = (
    "fused" if importlib.util.find_spec("fla") is not None else "checkpointed_chunk"
)

from tools.groundjev.g2.compat import patch_transformers_causal_mask  # noqa: E402
from tools.groundjev.g2.groundanything_lora import (  # noqa: E402
    adapter_state_dict,
    freeze_all_except_projector_and_lora,
    inject_qwen3_lora,
    trainable_audit,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JsonlDataset:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve(strict=True)
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise RuntimeError(f"empty G2 corpus: {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[int(index)])
            return json.loads(handle.readline())


class BalancedSourceSampler:
    """Round-robin all sources, repeating smaller sources deterministically."""

    def __init__(self, dataset: JsonlDataset, *, seed: int, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        source_indices: dict[str, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            # A lightweight source read is acceptable once at startup and
            # avoids retaining the 190-MB JSONL rows in every rank.
            source = str(dataset[index]["source"]).split(":", 1)[0]
            source_indices[source].append(index)
        self.source_indices = {key: value for key, value in sorted(source_indices.items())}
        self.sources = tuple(self.source_indices)
        self.max_source_rows = max(len(value) for value in self.source_indices.values())
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(self.max_source_rows * len(self.sources) / self.world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools: dict[str, list[int]] = {}
        for source, values in self.source_indices.items():
            values = list(values)
            rng.shuffle(values)
            pools[source] = values
        global_order: list[int] = []
        for position in range(self.max_source_rows):
            for source in self.sources:
                values = pools[source]
                global_order.append(values[position % len(values)])
        return iter(global_order[self.rank :: self.world_size])

    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": "g2-source-round-robin-pcg64-v1",
            "sources": list(self.sources),
            "source_rows": {key: len(value) for key, value in self.source_indices.items()},
            "max_source_rows": self.max_source_rows,
            "epoch": self.epoch,
            "rank": self.rank,
            "world_size": self.world_size,
        }


def load_model(checkpoint: Path, device: torch.device):
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from models.dlm.vlm import GAMQwen3DLM

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    raw = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    Config = get_class_from_dynamic_module("configuration_groundinganything.GroundAnythingVLMConfig", str(checkpoint), local_files_only=True)
    Model = get_class_from_dynamic_module("modeling_groundinganything.GroundAnythingVLMForConditionalGeneration", str(checkpoint), local_files_only=True)
    config = Config.from_dict(raw.to_dict())
    config._attn_implementation = "eager"
    with safe_open(checkpoint / "model.safetensors", framework="pt") as tensors:
        config.text_config.vocab_size = tensors.get_slice("base_model.model.language_model.embed_tokens.weight").get_shape()[0]
    if config.text_config.vocab_size != len(processor.tokenizer):
        raise RuntimeError("checkpoint vocabulary and tokenizer length disagree")
    mask = processor.tokenizer.encode("|<MASK>|", add_special_tokens=False)
    if len(mask) != 1:
        raise RuntimeError(f"GroundAnything mask token is not atomic: {mask}")
    with init_empty_weights(include_buffers=False):
        model = GAMQwen3DLM(Model(config), mask[0], processor.tokenizer.convert_tokens_to_ids("<|im_end|>"), block_size=32)
    model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True, assign=True)
    injection = inject_qwen3_lora(model, rank=16, alpha=32, dropout=0.05)
    model.to(device)
    model.base_model.model.visual.projector.float()
    freeze_all_except_projector_and_lora(model)
    # Match the released DLM general recipe for the long-sequence route.  The
    # first formal attempt used the legacy padded language loop and exhausted
    # the remaining memory on a 2,193-token MOT20 row.  Padding-free packing
    # and non-reentrant checkpointing bound the saved activations while keeping
    # the official 0.5 MDM + 0.5 causal objective unchanged.
    model.set_loss_contract(
        casuallossenable=True,
        mdm_loss_weight=0.5,
        causal_loss_weight=0.5,
    )
    model.set_language_checkpoint_stride(0)
    model.set_padding_free_packing(True)
    model.set_packing_language_checkpoint_stride(1)
    model.set_packing_linear_kernel("chunk")
    model.set_packing_lm_head_loss(PACKED_LM_HEAD_LOSS_BACKEND, 128)
    model.set_packing_activation_cpu_offload(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    audit = trainable_audit(model)
    if audit["vision_trainable_parameters"] != 0 or audit["original_language_base_trainable_parameters"] != 0:
        raise RuntimeError(f"frozen G2 policy failed: {audit}")
    if audit["projector_trainable_parameters"] <= 0 or audit["lora_trainable_parameters"] <= 0 or audit["other_unexpected_trainable_parameters"]:
        raise RuntimeError(f"G2 trainable audit failed: {audit}")
    model.train()
    model.config.use_cache = False
    return model, processor, injection, audit


def rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    import torch

    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    run_dir: Path,
    step: int,
    epoch: int,
    position: int,
    sampler: BalancedSourceSampler,
    manifest: dict[str, Any],
    base_checkpoint: Path,
    audit: dict[str, Any],
    injection: Any,
) -> Path:
    import torch

    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "format": "groundanything-g2-projector-qwen3-lora-v1",
        "step": int(step),
        "epoch": int(epoch),
        "sampler_position": int(position),
        "world_size": int(sampler.world_size),
        "base_checkpoint": str(base_checkpoint),
        "base_model_safetensors_sha256": sha256(base_checkpoint / "model.safetensors"),
        "manifest_path": manifest["rows_path"],
        "manifest_sha256": manifest["rows_sha256"],
        "trainable_audit": audit,
        "lora_injection": {"names": list(injection.names), "rank": injection.rank, "alpha": injection.alpha, "dropout": injection.dropout},
        "model_trainable_state": adapter_state_dict(raw_model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": rng_state(),
        "sampler": sampler.metadata(),
        "objective": {"mdm_loss_weight": 0.5, "causal_loss_weight": 0.5},
        "optimizer_contract": {"projector_lr": 2e-6, "lora_lr": 7e-6, "weight_decay": 0.1, "gradient_clip": 1.0, "scheduler": "cosine_with_min_lr", "min_lr_rate": 0.1, "warmup_ratio": 0.03},
    }
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    destination = checkpoints / f"adapter-step{step:06d}.pt"
    atomic_torch_save(state, destination)
    atomic_torch_save(state, checkpoints / "latest_training_state.pt")
    return destination


def make_scheduler(optimizer: Any, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / max(1, warmup_steps))
        progress = min(1.0, float(step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine

    return LambdaLR(optimizer, multiplier)


def main() -> int:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch._functorch import config as functorch_config
    from torch.utils.data import DataLoader
    from train.dlm.data import DLMDataCollator

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/GroundAnything-DLM")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--resume", type=Path, default=None)
    # The official DLM SFT contract uses 4096 text tokens and 8192 joint
    # tokens.  G2-A used a deliberately smaller 2048/4096 gate; formal rows
    # include crowded MOT20 answers that legitimately exceed that preflight
    # bound, so the formal default must match the official recipe.
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-joint-length", type=int, default=8192)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("G2 formal trainer requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    functorch_config.donated_buffer = False
    mask_compatibility = patch_transformers_causal_mask()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    manifest = json.loads(args.manifest.read_text())
    if manifest["rows_path"] != str(args.rows):
        raise RuntimeError("manifest rows path drift")
    if sha256(args.rows) != manifest["rows_sha256"]:
        raise RuntimeError("G2 rows SHA drift")
    dataset = JsonlDataset(args.rows)
    sampler = BalancedSourceSampler(dataset, seed=args.seed, rank=rank, world_size=world_size)
    model, processor, injection, audit = load_model(args.checkpoint, device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    raw_model = model.module if hasattr(model, "module") else model
    projector_params = list(raw_model.base_model.model.visual.projector.parameters())
    lora_params = [parameter for name, parameter in raw_model.named_parameters() if ".lora_A." in name or ".lora_B." in name]
    optimizer = torch.optim.AdamW(
        [{"params": projector_params, "lr": 2e-6}, {"params": lora_params, "lr": 7e-6}],
        weight_decay=0.1,
    )
    total_steps = int(args.max_steps)
    warmup_steps = max(1, int(round(total_steps * 0.03)))
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
    start_step = 0
    start_epoch = 0
    start_position = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["manifest_sha256"] != manifest["rows_sha256"] or int(state["world_size"]) != world_size:
            raise RuntimeError("resume checkpoint manifest/world-size drift")
        raw_model.load_state_dict(state["model_trainable_state"], strict=False)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        restore_rng(state["rng_state"])
        start_step = int(state["step"])
        start_epoch = int(state["epoch"])
        start_position = int(state["sampler_position"])
    collator = DLMDataCollator(processor, max_length=args.max_length, max_joint_length=args.max_joint_length, model_family="groundinganything_qwen3")
    loader = None
    completed = []
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (run_dir / "trainer_contract.json").write_text(json.dumps({"status": "G2_FORMAL_TRAINING_RUNNING", "created_localtime": time.strftime("%Y-%m-%d %H:%M:%S %z"), "rows": str(args.rows), "manifest": manifest, "mask_compatibility": mask_compatibility, "trainable_audit": audit, "world_size": world_size, "requested_physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"), "max_steps": total_steps, "save_every": args.save_every, "sampler": sampler.metadata(), "official_dlm_memory_contract": {"max_length": args.max_length, "max_joint_length": args.max_joint_length, "padding_free_packing": True, "gradient_checkpointing": True, "packing_language_checkpoint_stride": 1, "packing_linear_kernel": "chunk", "packing_lm_head_loss_backend": PACKED_LM_HEAD_LOSS_BACKEND, "packing_activation_cpu_offload": False, "packed_clean_attention_backend": os.environ.get("GAM_DLM_PACKED_CLEAN_ATTENTION_BACKEND")}}, indent=2, ensure_ascii=False) + "\n")
    for epoch in range(start_epoch, start_epoch + max(1, total_steps // max(1, len(sampler)) + 2)):
        sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=collator)
        for position, batch in enumerate(loader):
            if epoch == start_epoch and position < start_position:
                continue
            if start_step >= total_steps:
                break
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite G2 loss at step {start_step + 1}")
            loss.backward()
            trainable = [parameter for parameter in raw_model.parameters() if parameter.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite G2 gradient at step {start_step + 1}")
            optimizer.step()
            scheduler.step()
            start_step += 1
            start_position = position + 1
            completed.append({"step": start_step, "epoch": epoch, "position": start_position, "loss": float(loss.detach()), "gradient_norm": float(grad_norm), "lr_projector": float(optimizer.param_groups[0]["lr"]), "lr_lora": float(optimizer.param_groups[1]["lr"])})
            if rank == 0 and (start_step == 1 or start_step % args.save_every == 0 or start_step == total_steps):
                saved = save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, run_dir=run_dir, step=start_step, epoch=epoch, position=start_position, sampler=sampler, manifest=manifest, base_checkpoint=args.checkpoint, audit=audit, injection=injection)
                with (run_dir / "training_events.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({**completed[-1], "checkpoint": str(saved), "time": time.strftime("%Y-%m-%d %H:%M:%S %z")}, ensure_ascii=False) + "\n")
            if world_size > 1:
                dist.barrier()
        if start_step >= total_steps:
            break
        start_position = 0
    if rank == 0:
        raw_model = model.module if hasattr(model, "module") else model
        torch.save({"format": "groundanything-g2-adapter-projector-v1", "step": start_step, "state": adapter_state_dict(raw_model), "trainable_audit": audit, "manifest_sha256": manifest["rows_sha256"], "base_checkpoint": str(args.checkpoint)}, run_dir / "adapter_projector_final.pt")
        (run_dir / "final_receipt.json").write_text(json.dumps({"status": "G2_FORMAL_TRAINING_COMPLETE", "steps": start_step, "last_events": completed[-10:], "manifest_sha256": manifest["rows_sha256"], "world_size": world_size}, indent=2, ensure_ascii=False) + "\n")
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
