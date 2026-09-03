#!/usr/bin/env python3
"""Small L87-A launcher regression: verify the selected Python/NCCL path."""
from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def main() -> int:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world != 3:
        raise RuntimeError(f"expected three-process regression, got WORLD_SIZE={world}")
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("selected runtime does not provide CUDA/NCCL")
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", init_method="env://")
    value = torch.tensor([rank + 1], dtype=torch.int64, device=torch.device("cuda", local))
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) // 2
    if int(value.item()) != expected:
        raise AssertionError(f"all-reduce mismatch: {int(value.item())} != {expected}")
    dist.barrier()
    if rank == 0:
        print(json.dumps({
            "format": "locatemot-l87a-runtime-regression-v1",
            "status": "complete", "world_size": world,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "nccl": True, "all_reduce_sum": int(value.item()),
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
        }, sort_keys=True))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
