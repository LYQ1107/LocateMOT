"""Frozen candidate clips with separate original ordinary-MOT supervision."""
from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from .bdd_tracking import frame_objects
from .legal_scope import assert_legal_video
from ..losses.u2_association import FrameSupervision
from ..runtime.u2_temporal_training import clip_length


DATASET_PROBABILITIES = {"bdd100k": .25, "dancetrack": .25, "mot17": .20, "mot20": .10,
                         "kitti_legal_development": .20}
# BDD annotations are 5 Hz, KITTI 10 Hz, DanceTrack 20 Hz; MOT rates
# come from each original seqinfo.ini. Sample gaps are about 0.2s or less.
DATASET_FPS = {"bdd100k": 5., "dancetrack": 20., "kitti_legal_development": 10.}


def source_frame_index(group: dict, position: int) -> int:
    return position if group["dataset"] == "bdd100k" else int(Path(group["frames"][position]).stem)


def frame_tensor(boxes: list) -> torch.Tensor:
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


class OriginalMOTSupervision:
    """Read exactly the frozen group frames; never union KITTI split versions.

    MOT uses mark/nonzero pedestrian and official distractor classes (including
    MOT20 class 6). KITTI follows pinned Car/Pedestrian occlusion <=2 and
    integer-cast truncation <=0, Van/Person_sitting distractors, DontCare regions
    and the unmatched 25px-height rule. BDD uses the validated raw toolkit rules.
    """

    def __init__(self, group: dict, source: dict, max_age: int = 30) -> None:
        self.group, self.max_age = group, max_age
        dataset = group["dataset"]
        path = Path(source["source_path"])
        if dataset == "kitti_legal_development":
            assert_legal_video(group["video"], dataset="refer_kitti_v2")
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != source["sha256"]:
            raise ValueError(f"Original supervision changed: {group['key']}")
        positive, regions, distractors = defaultdict(list), defaultdict(list), defaultdict(list)
        frame_indices = {source_frame_index(group, position) for position in range(len(group["frames"]))}
        if dataset == "bdd100k":
            by_name = {frame["name"]: frame for frame in json.loads(contents)}
            identities: dict[str, int] = {}
            for position, name in enumerate(group["frames"]):
                objects, ignored = frame_objects(by_name[name])
                for obj in objects:
                    identity = identities.setdefault(obj["track_id"], len(identities))
                    positive[position].append((identity, obj["bbox"]))
                regions[position].extend(obj["bbox"] for obj in ignored if "bbox" in obj)
        elif dataset == "kitti_legal_development":
            for line in contents.decode().splitlines():
                values = line.split()
                frame, identity, category = int(values[0]), int(values[1]), values[2].lower()
                if frame not in frame_indices:
                    continue
                box = list(map(float, values[6:10]))
                if category == "dontcare":
                    regions[frame].append(box)
                elif identity >= 0 and category in {"car", "pedestrian"}:
                    # Match the pinned public reader's integer conversion.
                    if int(float(values[4])) <= 2 and int(float(values[3])) <= 0:
                        positive[frame].append((identity, box))
                    else:
                        distractors[frame].append(box)
                elif identity >= 0 and category in {"van", "person_sitting"}:
                    distractors[frame].append(box)
        else:
            distracting_classes = {2, 7, 8, 12} | ({6} if dataset == "mot20" else set())
            for line in contents.decode().splitlines():
                values = [float(value) for value in line.split(",")]
                frame, identity, category = int(values[0]), int(values[1]), int(values[7])
                if frame not in frame_indices:
                    continue
                x, y, width, height = values[2:6]
                box = [x, y, x + width, y + height]
                if values[6] != 0 and category == 1:
                    positive[frame].append((identity, box))
                elif category in distracting_classes:
                    distractors[frame].append(box)
        self.frames: dict[int, FrameSupervision] = {}
        appearances: dict[int, list[int]] = defaultdict(list)
        for frame, objects in positive.items():
            for identity, _ in objects:
                appearances[identity].append(frame)
        for frames in appearances.values():
            frames.sort()
        # Loss labels use the full registered TRAIN sequence, including frames
        # between sampled observations, not truncated clip endpoints.
        for frame in sorted(frame_indices):
            objects = positive[frame]
            identities = [identity for identity, _ in objects]
            if len(set(identities)) != len(identities):
                raise ValueError(f"Duplicate original identity in {group['key']} frame {frame}")
            trackable = set()
            for identity, frames in appearances.items():
                next_index = bisect_left(frames, frame)
                if next_index < len(frames) and frames[next_index] - frame <= max_age:
                    trackable.add(identity)
            self.frames[frame] = FrameSupervision(
                frame_tensor([box for _, box in objects]), torch.tensor(identities, dtype=torch.long),
                frame_tensor(regions[frame]), trackable, frame_tensor(distractors[frame]),
                25. if dataset == "kitti_legal_development" else 0.)
        self.positive_units = sum(len(frame.boxes) for frame in self.frames.values())

    def __getitem__(self, position: int) -> FrameSupervision:
        return self.frames[source_frame_index(self.group, position)]


class CachedMOTClips:
    """Sample only complete ordinary TRAIN videos from the selected cache.

    Stateless step/rank seeds make sampling reproducible after a checkpoint
    resume. All ranks sample the same task and dataset per global optimizer
    step, with independent video/start/stride choices for their two clips.
    """

    def __init__(self, cache: dict, supervision: dict, *, seed: int = 20261012,
                 shard_cache_size: int = 8, supervision_cache_size: int = 8) -> None:
        self.cache, self.supervision, self.seed = cache, supervision, seed
        self.shard_cache_size, self.supervision_cache_size = shard_cache_size, supervision_cache_size
        self.shards: OrderedDict[str, dict] = OrderedDict()
        self.gt: OrderedDict[str, OriginalMOTSupervision] = OrderedDict()
        self.groups: dict[str, list[dict]] = defaultdict(list)
        self.units: dict[str, dict[str, tuple[dict, int]]] = {}
        for item in cache["groups"]:
            group = item["group"]
            if group["dataset"] not in DATASET_PROBABILITIES:
                continue
            if group["role"] == "train":
                self.groups[group["dataset"]].append(group)
            frame_units = {frame: (unit, index) for unit in item["units"] for index, frame in enumerate(unit["frames"])}
            if set(frame_units) != set(group["frames"]) or len(frame_units) != sum(len(unit["frames"]) for unit in item["units"]):
                raise ValueError(f"Candidate cache coverage differs: {group['key']}")
            self.units[group["key"]] = frame_units
        if set(self.groups) != set(DATASET_PROBABILITIES):
            raise ValueError("The MOT-only cache must contain all five registered training datasets")

    def dataset_for_step(self, step: int) -> str:
        rng = random.Random(self.seed + step * 1000003)
        return rng.choices(list(DATASET_PROBABILITIES), weights=list(DATASET_PROBABILITIES.values()), k=1)[0]

    def fps(self, group: dict) -> float:
        if group["dataset"] in DATASET_FPS:
            return DATASET_FPS[group["dataset"]]
        import configparser
        config = configparser.ConfigParser()
        config.read(Path(group["image_dir"]).parent / "seqinfo.ini")
        return float(config["Sequence"]["frameRate"])

    def load_candidates(self, group: dict, position: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        frame = group["frames"][position]
        unit, index = self.units[group["key"]][frame]
        path = unit["path"]
        if path not in self.shards:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            header = payload["header"]
            if (header["cache_spec_sha256"] != self.cache["cache_spec_sha256"]
                    or header["foundation_sha256"] != self.cache["spec"]["foundation_sha256"]
                    or header["group_key"] != group["key"] or header["frames"] != unit["frames"]):
                raise ValueError(f"Candidate shard belongs to a different frozen cache: {path}")
            self.shards[path] = payload
            if len(self.shards) > self.shard_cache_size:
                self.shards.popitem(last=False)
        self.shards.move_to_end(path)
        item = self.shards[path]["frames"][index]
        if item["frame"] != frame:
            raise ValueError("Candidate shard frame ordering differs")
        # BF16 storage is expanded to FP32 before the core's autocast boundary.
        return {name: value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
                for name, value in item["candidates"].items()}

    def sample(self, step: int, rank: int, device: torch.device | str, clips_per_rank: int = 2) -> list[list[dict]]:
        dataset, length = self.dataset_for_step(step), clip_length(step)
        rng = random.Random(self.seed + step * 1000003 + (rank + 1) * 9176)
        clips = []
        for _ in range(clips_per_rank):
            # The registered videos exceed eight frames. Short sources are an
            # explicit protocol error; never silently skip or pad samples.
            group = rng.choice(self.groups[dataset])
            max_stride = min(4, max(1, math.ceil(self.fps(group) / 7.5)))
            max_stride = min(max_stride, (len(group["frames"]) - 1) // (length - 1))
            if max_stride < 1:
                raise ValueError(f"Registered video too short for the temporal curriculum: {group['key']}")
            stride = rng.randint(1, max_stride)
            start = rng.randrange(len(group["frames"]) - (length - 1) * stride)
            key = group["key"]
            if key not in self.gt:
                self.gt[key] = OriginalMOTSupervision(group, self.supervision[key])
                if len(self.gt) > self.supervision_cache_size:
                    self.gt.popitem(last=False)
            self.gt.move_to_end(key)
            clip = []
            for position in range(start, start + length * stride, stride):
                clip.append({"frame_index": source_frame_index(group, position),
                             "candidates": self.load_candidates(group, position, device),
                             "supervision": self.gt[key][position].to(device),
                             "dataset": dataset, "video": group["video"], "frame": group["frames"][position]})
            clips.append(clip)
        return clips
