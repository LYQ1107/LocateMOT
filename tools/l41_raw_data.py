"""L41 streaming raw-image patch-token and pair helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l40_raw_data import (FIT_VIDEOS, HELDOUT_VIDEOS, WEIGHTS, crop_box, load_fragments,
                                make_pairs, sha256)
from tools.l40_raw_data import RAW_ROOT


class StreamingClipPatchEncoder:
    """Frozen CLIP ViT patch tokens, reduced to four spatial cells in RAM."""
    def __init__(self, device="cuda:0", weights=WEIGHTS, batch_size=32):
        import clip
        self.device = torch.device(device); self.batch_size = int(batch_size)
        self.model, self.preprocess = clip.load(str(weights), device=self.device)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)

    @torch.inference_mode()
    def _patches(self, pixel):
        visual = self.model.visual
        # OpenAI CLIP loads CUDA weights in fp16; preprocessing produces fp32.
        pixel = pixel.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(pixel)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), dim=1) + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x).permute(1, 0, 2)
        x = visual.transformer(x).permute(1, 0, 2)[:, 1:]
        side = int(round(x.shape[1] ** 0.5)); x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side)
        x = torch.nn.functional.adaptive_avg_pool2d(x.float(), (2, 2)).flatten(2).transpose(1, 2)
        return x

    @torch.inference_mode()
    def encode(self, fragments, ids):
        outputs = []
        for frag_id in ids:
            f = fragments[int(frag_id)]; pixels = []
            for ob in f["obs"]:
                path = Path(ob["image"])
                with Image.open(path) as image:
                    image = image.convert("RGB"); box = crop_box(ob["box"], image.width, image.height)
                    if box[2] <= box[0] or box[3] <= box[1]: raise ValueError(f"empty crop {path}")
                    pixels.append(self.preprocess(image.crop(box)))
            chunks = []
            for start in range(0, len(pixels), self.batch_size):
                batch = torch.stack(pixels[start:start + self.batch_size]).to(self.device)
                chunks.append(self._patches(batch).cpu().half()); del batch
            outputs.append(torch.cat(chunks, 0))
        return outputs


def pad_patches(fragments, ids, patch_map, device):
    b = len(ids); h = 8; d = int(patch_map[ids[0]].shape[-1])
    out = torch.zeros(b, h, 4, d, dtype=torch.float32); mask = torch.zeros(b, h, dtype=torch.bool)
    for j, i in enumerate(ids):
        k = min(h, len(fragments[i]["obs"])); st = h - k; out[j, st:] = patch_map[i][-k:].float(); mask[j, st:] = True
    return out.to(device), mask.to(device)


def relation_features(fa, fb):
    a = fa["obs"][-1]["numeric"].float(); b = fb["obs"][-1]["numeric"].float()
    ta, tb = fa["obs"][-1]["frame"], fb["obs"][-1]["frame"]
    return torch.cat((a - b, torch.abs(a - b), torch.tensor([(tb - ta) / 100.0], dtype=torch.float32)))


__all__ = ["FIT_VIDEOS", "HELDOUT_VIDEOS", "WEIGHTS", "load_fragments", "make_pairs", "sha256", "StreamingClipPatchEncoder", "pad_patches", "relation_features"]
