"""Streaming OpenAI CLIP ViT-B/16 joint-space features for L65."""
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


def crop_box(box, width, height, padding=0.10):
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= padding * bw; y1 -= padding * bh; x2 += padding * bw; y2 += padding * bh
    return max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))


class StreamingClipJoint:
    def __init__(self, device="cuda:0", batch_size=32):
        import clip
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(str(WEIGHTS), device=self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.clip = clip
        self.batch_size = int(batch_size)

    @torch.inference_mode()
    def image_joint_tokens(self, pixel_batch):
        """Return normalized [B,17,512]: projected CLS + 4x4 patch tokens."""
        visual = self.model.visual
        x = pixel_batch.to(self.device, dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x).reshape(x.shape[0], visual.conv1.out_channels, -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), dim=1) + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x).permute(1, 0, 2)
        x = visual.transformer(x).permute(1, 0, 2)
        x = visual.ln_post(x)
        x = x @ visual.proj
        global_token = F.normalize(x[:, 0], dim=-1)
        spatial = x[:, 1:]
        side = int(round(spatial.shape[1] ** 0.5))
        if side * side != spatial.shape[1]:
            raise AssertionError(f"CLIP patch shape {spatial.shape}")
        spatial = spatial.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side)
        spatial = F.adaptive_avg_pool2d(spatial.float(), (4, 4)).flatten(2).transpose(1, 2)
        spatial = F.normalize(spatial, dim=-1)
        return torch.cat((global_token[:, None, :], spatial), dim=1).float()

    @torch.inference_mode()
    def text_joint_tokens(self, sentence):
        tokens = self.clip.tokenize([str(sentence)], truncate=True).to(self.device)
        model = self.model
        x = model.token_embedding(tokens).to(model.dtype)
        x = x + model.positional_embedding.to(x.dtype)
        x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = model.ln_final(x)
        x = x @ model.text_projection
        valid = tokens[0].ne(0).bool()
        token_features = F.normalize(x.float(), dim=-1)[0]
        eos = int(tokens[0].argmax())
        # Keep the frozen-to-trainable boundary explicit: callers clone again
        # if they attach autograd, while audit features remain CPU-only.
        return token_features.cpu(), valid.cpu(), token_features[eos].clone().cpu(), tokens[0].clone().cpu()

    @torch.inference_mode()
    def encode_unit(self, video, frame, boxes):
        path = image_path(video, frame)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            image = source.convert("RGB")
            pixels = []
            for box in boxes:
                crop = crop_box(box, image.width, image.height)
                if crop[2] <= crop[0] or crop[3] <= crop[1]:
                    raise ValueError(f"empty crop {path} {box} {crop}")
                pixels.append(self.preprocess(image.crop(crop)))
            chunks = []
            for start in range(0, len(pixels), self.batch_size):
                batch = torch.stack(pixels[start:start + self.batch_size])
                chunks.append(self.image_joint_tokens(batch).cpu())
                del batch
        return (torch.cat(chunks, 0) if chunks else torch.empty((0, 17, 512)), path)
