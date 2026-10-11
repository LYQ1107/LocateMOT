#!/usr/bin/env python3
"""Check the complete real U2-P replay with native eight-rank CPU data loading."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/mmdetection"))

import torch
import torch.distributed as dist
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from tools.unified.train_u2_perception_adaptation import CONFIG, DIRECTORY, digest


def main() -> int:
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        raise ValueError("Replay verification requires its actual registered eight-rank sampler")
    dist.init_process_group("gloo")
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(CONFIG))
    manifest = json.loads((DIRECTORY / "replay_manifest.json").read_text())
    if digest(Path(manifest["replay_path"])) != manifest["replay_sha256"]:
        raise ValueError("Replay differs from the completed preparation artifact")
    # Only the check's worker count changes; sampling and batching use the
    # actual train configuration, with no model or optimizer created.
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    loader = Runner.build_dataloader(cfg.train_dataloader, seed=cfg.randomness.seed)
    dataset = loader.dataset
    if len(dataset) != 20000 or len(loader) != 1250:
        raise ValueError("Native loader changed the registered rows or exact per-rank step budget")
    indices = list(loader.sampler)
    if len(indices) != 2500:
        raise ValueError("Eight-rank sampler does not assign exactly 2,500 rows per rank")
    batch = next(iter(loader))
    if len(batch["inputs"]) != 2 or any(not torch.isfinite(image.float()).all() for image in batch["inputs"]):
        raise ValueError("Native batch-2 image packing failed")
    gathered = [None] * 8 if dist.get_rank() == 0 else None
    dist.gather_object(indices, gathered, dst=0)
    if dist.get_rank() == 0:
        all_indices = [index for partition in gathered for index in partition]
        if sorted(all_indices) != list(range(20000)):
            raise ValueError("Distributed epoch duplicates or omits a registered replay row")
        rows = [json.loads(line) for line in Path(manifest["replay_path"]).read_text().splitlines()]
        counts = Counter(row["stream"] for row in rows)
        if dict(counts) != manifest["allocations"]:
            raise ValueError("Actual epoch differs from the preregistered stream proportions")
        boxes, empty = Counter(), Counter()
        for index, row in enumerate(rows):
            annotations = dataset.get_data_info(index)["instances"]
            boxes[row["stream"]] += len(annotations)
            empty[row["stream"]] += not annotations
        report = {"state": "PASS_REAL_REPLAY_EIGHT_RANK_NATIVE_LOADER", "world_size": 8,
                  "rows": 20000, "rows_per_rank": 2500, "batch_per_rank": 2, "steps_per_epoch": 1250,
                  "duplicate_or_omitted_rows": 0, "stream_counts": dict(counts),
                  "native_positive_boxes": dict(boxes), "native_empty_positive_rows": dict(empty),
                  "native_box_filter_removals": {name: manifest["stream_counts"][name]["positive_boxes"] - boxes[name]
                                               for name in manifest["allocations"]},
                  "replay_sha256": manifest["replay_sha256"], "optimizer_created": False,
                  "gpu_model_created": False, "official_test_labels_read": False}
        (DIRECTORY / "native_replay_full_contract.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
