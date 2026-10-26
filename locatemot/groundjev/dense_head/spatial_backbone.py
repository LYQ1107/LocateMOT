"""Native MoonViT and checkpoint-specific projector, without answer/text tokens."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from locatemot.groundjev.native_features import NativeGroundFeatures


class SpatialBackbone(NativeGroundFeatures):
    def __init__(self, checkpoint, device="cuda:0", max_tokens=1024):
        super().__init__(checkpoint, device)
        # The same native resize policy and token budget apply to both arms.
        self.processor.max_output_tokens = max_tokens
        self.processor.media_proc_cfg["in_patch_limit"] = max_tokens * 4
        self.max_tokens = max_tokens

    def encode(self, image):
        # No GT, language answer, candidate boxes, identity or temporal input.
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        spatial, metadata = super().encode(image)
        torch.cuda.synchronize(self.device)
        metadata["seconds"] = time.perf_counter() - started
        metadata["image_only"] = True
        metadata["max_tokens"] = self.max_tokens
        # A normal tensor is needed by trainable projection backward, even
        # though the source vision forward uses inference_mode.
        return spatial.detach().clone(), metadata

    def differentiable_encode(self, image, train_projector=False):
        """Future projector gradient path; deliberately bypass frozen caches.

        This does not expose Qwen LoRA: language adaptation would require a
        separate answer-free causal language forward, not this vision adapter.
        """
        pixels = self.processor(images=[image], return_tensors="pt")
        grid = pixels["image_grid_thw"].to(self.device)
        self.visual.projector.requires_grad_(train_projector)
        with torch.no_grad():
            tower = self.visual.vision_tower(
                pixels["pixel_values"].to(self.device, dtype=self.visual.dtype), grid
            )
        tokens = torch.cat(self.visual.projector(tower), dim=0)
        _, h, w = grid[0].tolist()
        merge = self.visual.spatial_merge_size
        return tokens.reshape(h // merge, w // merge, -1).permute(2, 0, 1)[None]


class BoundedFeatureStore:
    """Per-arm bounded spatial cache; both registered arms together <=12 GiB.

    This is explicitly frozen-backbone head training, not joint G2 adaptation.
    Cache records contain image-only features, never GT or candidate RoIs.
    """
    def __init__(self, backbone, directory, budget_bytes=6 * 2**30):
        self.backbone = backbone
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.budget = budget_bytes
        self.entries = {p: p.stat().st_size for p in self.directory.glob("*.pt")}
        self.size = sum(self.entries.values())
        self.hits = self.misses = 0
        model = Path(backbone.checkpoint) / "model.safetensors"
        self.identity = [str(model.resolve()), model.stat().st_size,
                         model.stat().st_mtime_ns, backbone.max_tokens]

    def get(self, image_path, use_cache=True):
        from PIL import Image
        path = Path(image_path)
        stat = path.stat()
        key = hashlib.sha256(json.dumps([self.identity, str(path.resolve()),
                                       stat.st_size, stat.st_mtime_ns]).encode()).hexdigest()
        target = self.directory / (key + ".pt")
        if use_cache and target.exists():
            self.hits += 1
            target.touch()
            row = torch.load(target, map_location="cpu", weights_only=False)
            return row["spatial"].to(self.backbone.device), row["metadata"]
        self.misses += 1
        with Image.open(path) as image:
            features, metadata = self.backbone.encode(image.convert("RGB"))
        if use_cache:
            row = {"spatial": features.cpu(), "metadata": metadata}
            temporary = target.with_suffix(".tmp")
            torch.save(row, temporary)
            required = temporary.stat().st_size
            while self.entries and self.size + required > self.budget:
                oldest = min(self.entries, key=lambda p: p.stat().st_mtime_ns)
                self.size -= self.entries.pop(oldest)
                oldest.unlink()
            if required <= self.budget:
                temporary.replace(target)
                self.size += required
                self.entries[target] = required
            else:
                temporary.unlink()
        return features, metadata
