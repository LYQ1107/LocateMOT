"""Reusable loader for the Stage L17 frozen RN50 image-feature cache."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch


class RN50FeatureStore:
    def __init__(self, root, cache_size: int = 8):
        self.root = Path(root)
        self.cache_size = int(cache_size)
        self.cache = OrderedDict()

    def get(self, dataset: str, video: str) -> dict:
        key = (dataset, video)
        if key not in self.cache:
            path = self.root / dataset / f"{video}.pt"
            value = torch.load(path, map_location="cpu", weights_only=False)
            self.cache[key] = value
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]

    def frame_features(self, dataset: str, video: str, frame_index: int,
                       begin: int, end: int, device) -> dict:
        cache = self.get(dataset, video)
        local = cache["local"][begin:end].to(device, non_blocking=True)
        global_value = cache["global_frame"][frame_index].to(
            device, non_blocking=True)
        global_value = global_value.unsqueeze(0).expand(end - begin, -1)
        return {"ikun_local": local, "ikun_global": global_value}
