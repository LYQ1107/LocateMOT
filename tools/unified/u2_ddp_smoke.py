#!/usr/bin/env python3
"""Short DDP forward/backward smoke; no dataset or checkpoint is touched."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.models.unified_spec_track import UnifiedSpecTrack


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("U2 DDP smoke requires CUDA")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20261002 + rank)
    model = UnifiedSpecTrack().to(device).train()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    feature = torch.randn(2, 256, device=device)
    candidates = {
        "boxes": torch.tensor([[0., 0., 1., 1.], [.1, .1, 1.1, 1.1]], device=device),
        "scores": torch.tensor([.9, .8], device=device),
        "id_visual": feature,
        "semantic_visual": feature * .8,
        "decoder_query": feature * .4,
    }
    output = model(candidates=candidates, task_type="mot", training=True)
    logits = output["tracks"]["new_logits"]
    loss = logits.float().square().mean()
    loss.backward()
    finite = torch.tensor([float(torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))], device=device)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if rank == 0:
        payload = {
            "format": "locatemot-u2-ddp-smoke-v1",
            "status": "pass" if bool(finite.item()) else "fail",
            "world_size": dist.get_world_size(),
            "loss": float(loss.detach().cpu()),
            "same_task_all_ranks": True,
            "official_test_labels_read": False,
        }
        path = Path("outputs/unified/u2/eval/ddp_smoke.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
