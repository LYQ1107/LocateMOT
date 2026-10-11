"""Base-only dense OVMOT labels kept outside frozen candidate shards."""
from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch

from .u2_mot_clips import CachedMOTClips, frame_tensor
from ..losses.u2_association import FrameSupervision


def joint_task_for_step(step: int, seed: int = 20261012) -> str:
    """One global task on all ranks, preserving registered .65/.35 weights."""
    return random.Random(seed + step * 1000003).choices(("mot", "ovmot"), weights=(.65, .35), k=1)[0]


class OriginalOVMOTSupervision:
    def __init__(self, group: dict, source: dict, max_age: int = 30) -> None:
        if group["dataset"] != "ovmot" or group["role"] != "train" or not group["video"].startswith("train/"):
            raise ValueError("OVMOT training cannot read validation/test identities")
        contents = Path(source["path"]).read_bytes()
        if hashlib.sha256(contents).hexdigest() != source["sha256"]:
            raise ValueError("Compiled original base supervision changed")
        data = json.loads(contents)
        if data["novel_class_or_identity_supervision"] or data["group_key"] != group["key"]:
            raise ValueError("OVMOT supervision does not match the frozen base-only group")
        if [frame["frame"] for frame in data["frames"]] != group["frames"]:
            raise ValueError("OVMOT supervision and selected candidate frames differ")
        appearances = defaultdict(list)
        for frame in data["frames"]:
            for identity in frame["identity_ids"]:
                appearances[identity].append(frame["source_frame_index"])
        self.frame_indices, self.frames = [], []
        negative_ids = set(data["negative_category_ids"])
        for frame in data["frames"]:
            physical = frame["source_frame_index"]
            trackable = set()
            for identity, times in appearances.items():
                next_index = bisect_left(times, physical)
                if next_index < len(times) and times[next_index] - physical <= max_age:
                    trackable.add(identity)
            self.frame_indices.append(physical)
            self.frames.append(FrameSupervision(frame_tensor(frame["boxes"]),
                                                 torch.tensor(frame["identity_ids"], dtype=torch.long),
                                                 frame_tensor(frame["ignore_boxes"]), trackable,
                                                 category_ids=torch.tensor(frame["category_ids"], dtype=torch.long),
                                                 negative_category_ids=negative_ids))
        self.positive_units = sum(len(frame.boxes) for frame in self.frames)
        self.track_classes = {track["identity_id"]: track["category_id"] for track in data["tracks"]}

    def __getitem__(self, position: int) -> FrameSupervision:
        return self.frames[position]


class CachedOVMOTClips(CachedMOTClips):
    """Reuse identical shard loading and the same shared core's input fields."""
    def __init__(self, cache: dict, supervision: dict, *, seed: int = 20261012,
                 shard_cache_size: int = 8, supervision_cache_size: int = 8) -> None:
        if (cache["status"] != "COMPLETE" or cache["spec"]["phase"] != "ovmot"
                or supervision["status"] != "COMPLETE" or supervision["novel_training_gt_units"] != 0):
            raise ValueError("Joint sampling requires complete frozen OVMOT candidates and base supervision")
        if cache["spec"]["inventory_sha256"] != supervision["inventory_sha256"]:
            raise ValueError("OVMOT features and labels differ in registered physical image scope")
        self.cache, self.seed = cache, seed
        self.supervision = {group["key"]: group for group in supervision["groups"]}
        self.shard_cache_size, self.supervision_cache_size = shard_cache_size, supervision_cache_size
        self.shards, self.gt = OrderedDict(), OrderedDict()
        self.groups, self.units = [], {}
        for item in cache["groups"]:
            group = item["group"]
            if group["dataset"] != "ovmot":
                raise ValueError("OVMOT cache phase includes another dataset")
            frame_units = {frame: (unit, index) for unit in item["units"] for index, frame in enumerate(unit["frames"])}
            if set(frame_units) != set(group["frames"]) or len(frame_units) != sum(len(unit["frames"]) for unit in item["units"]):
                raise ValueError("OVMOT selected cache omits or repeats registered frames")
            self.units[group["key"]] = frame_units
            if group["role"] == "train":
                self.groups.append(group)
        if {group["key"] for group in self.groups} != set(self.supervision):
            raise ValueError("OVMOT training groups differ from full dense+modal registered supervision")

    def sample(self, step: int, rank: int, device: torch.device | str, clips_per_rank: int = 2) -> list[list[dict]]:
        # U2-B starts from the selected U2-A state and retains eight frames.
        # Dense source strides remain 1..4; physical gaps come from original
        # source timestamps/filenames, never the reset interpolation index.
        length = 8
        rng = random.Random(self.seed + step * 1000003 + (rank + 1) * 9176)
        clips = []
        for _ in range(clips_per_rank):
            group = rng.choice(self.groups)
            stride_limit = min(4, (len(group["frames"]) - 1) // (length - 1))
            if stride_limit < 1:
                raise ValueError("Registered OVMOT train video is shorter than eight observations")
            stride = rng.randint(1, stride_limit)
            start = rng.randrange(len(group["frames"]) - (length - 1) * stride)
            key = group["key"]
            if key not in self.gt:
                self.gt[key] = OriginalOVMOTSupervision(group, self.supervision[key])
                if len(self.gt) > self.supervision_cache_size:
                    self.gt.popitem(last=False)
            self.gt.move_to_end(key)
            labels = self.gt[key]
            clips.append([{"frame_index": labels.frame_indices[position],
                           "candidates": self.load_candidates(group, position, device),
                           "supervision": labels[position].to(device),
                           "dataset": "ovmot", "video": group["video"], "frame": group["frames"][position]}
                          for position in range(start, start + length * stride, stride)])
        return clips
