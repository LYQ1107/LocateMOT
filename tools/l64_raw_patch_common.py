"""Streaming OpenAI CLIP ViT-B/16 crop and token helpers for L64.

Pixels and encoder outputs are deliberately returned to the caller only for
the current unit.  This module never writes an image or feature cache.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
RAW_ROOT = ROOT / "data/kitti_tracking_training/image_02"
WEIGHTS = Path("/home/lwr/.cache/clip/ViT-B-16.pt").resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def image_path(video: str, frame: int) -> Path:
    return RAW_ROOT / str(video) / f"{int(frame):06d}.png"


def crop_box(box, width: int, height: int, padding: float = 0.10):
    x1, y1, x2, y2 = [float(x) for x in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= padding * bw; y1 -= padding * bh
    x2 += padding * bw; y2 += padding * bh
    return (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2)))


class StreamingOpenAIClip:
    def __init__(self, device="cuda:0", batch_size=32):
        import clip
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(str(WEIGHTS), device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.batch_size = int(batch_size)
        self.clip_module = clip

    @torch.inference_mode()
    def image_patch_tokens(self, pixel_batch: torch.Tensor) -> torch.Tensor:
        """Return [B,16,512] spatial ViT patch tokens, excluding CLS."""
        visual = self.model.visual
        x = pixel_batch.to(device=self.device, dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x).reshape(x.shape[0], visual.conv1.out_channels, -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), 1) + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x).permute(1, 0, 2)
        x = visual.transformer(x).permute(1, 0, 2)[:, 1:]
        side = int(round(x.shape[1] ** 0.5))
        if side * side != x.shape[1]:
            raise AssertionError(f"unexpected CLIP patch count {x.shape}")
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side)
        x = F.adaptive_avg_pool2d(x.float(), (4, 4)).flatten(2).transpose(1, 2)
        return x.float()

    @torch.inference_mode()
    def text_tokens(self, sentence: str):
        tokens = self.clip_module.tokenize([str(sentence)], truncate=True).to(self.device)
        model = self.model
        x = model.token_embedding(tokens).to(dtype=model.dtype)
        x = x + model.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)
        x = model.transformer(x)
        x = x.permute(1, 0, 2)
        x = model.ln_final(x).float()[0]
        valid = tokens[0].ne(0).bool()
        return x, valid

    @torch.inference_mode()
    def encode_unit(self, video: str, frame: int, boxes):
        path = image_path(video, frame)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            image = source.convert("RGB")
            pixels = []
            for box in boxes:
                cb = crop_box(box, image.width, image.height)
                if cb[2] <= cb[0] or cb[3] <= cb[1]:
                    raise ValueError(f"empty crop {path} box={box} crop={cb}")
                pixels.append(self.preprocess(image.crop(cb)))
            chunks = []
            for start in range(0, len(pixels), self.batch_size):
                batch = torch.stack(pixels[start:start + self.batch_size])
                chunks.append(self.image_patch_tokens(batch).cpu())
                del batch
        if not chunks:
            return torch.empty((0, 16, 512), dtype=torch.float32), path
        return torch.cat(chunks, 0), path
