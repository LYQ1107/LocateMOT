"""L52 fit-only materialization and streaming CLIP crop/context tokens."""
from __future__ import annotations

import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
IMAGE_ROOT = ROOT / "data/kitti_tracking_training/image_02"
CLIP_WEIGHTS = Path("/home/lwr/.cache/clip/ViT-B-16.pt")


def crop_box(box, width, height, padding=0.10):
    x1, y1, x2, y2 = [float(x) for x in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return (max(0, int(math.floor(x1 - padding * bw))),
            max(0, int(math.floor(y1 - padding * bh))),
            min(width, int(math.ceil(x2 + padding * bw))),
            min(height, int(math.ceil(y2 + padding * bh))))


def frame_path(item, frame):
    return IMAGE_ROOT / str(item["video"]) / f"{int(frame):06d}.png"


class L52StreamingRegionEncoder:
    def __init__(self, device="cuda:0"):
        if not CLIP_WEIGHTS.is_file():
            raise FileNotFoundError(CLIP_WEIGHTS)
        import clip
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(str(CLIP_WEIGHTS), device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def _patch(self, pixels):
        visual = self.model.visual
        pixel = pixels.to(self.device, dtype=visual.conv1.weight.dtype)
        x = visual.conv1(pixel).reshape(pixel.shape[0], visual.conv1.out_channels, -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), 1) + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x).permute(1, 0, 2)
        x = visual.transformer(x).permute(1, 0, 2)[:, 1:]
        side = int(round(x.shape[1] ** 0.5))
        if side * side != x.shape[1]:
            raise AssertionError(f"unexpected patch token count {x.shape[1]}")
        x = F.adaptive_avg_pool2d(
            x.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side), (4, 4)
        ).flatten(2).transpose(1, 2)
        return x.float()

    @torch.inference_mode()
    def encode(self, item, chunk=32):
        if len(item["boxes"]) != len(item["frames"]):
            raise AssertionError("box/frame row drift")
        if not len(item["boxes"]):
            raise AssertionError("empty candidate set")
        image_tensors = []
        frame = int(item["frames"][0])
        path = frame_path(item, frame)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            full = self.preprocess(image)
            for box in item["boxes"].tolist():
                cb = crop_box(box, image.width, image.height)
                if cb[2] <= cb[0] or cb[3] <= cb[1]:
                    raise ValueError(f"invalid crop {path}: {box} -> {cb}")
                image_tensors.append(self.preprocess(image.crop(cb)))
        region_parts = []
        for start in range(0, len(image_tensors), chunk):
            region_parts.append(self._patch(torch.stack(image_tensors[start:start + chunk])))
        context = self._patch(full.unsqueeze(0))
        result = torch.cat(region_parts, 0)
        del image_tensors, region_parts, full
        gc.collect()
        return result, context, {"crop_count": len(item["boxes"]), "frame_id": frame, "image_path": str(path)}
