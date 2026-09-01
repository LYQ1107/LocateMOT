"""Streaming CLIP runtime and fit-unit helpers for L66; no persistent feature cache."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
RAW_ROOT = ROOT / "data/kitti_tracking_training/image_02"
UNITS = ROOT / "outputs/l49/data/train_units.jsonl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
CLIP_WEIGHTS = Path("/home/lwr/.cache/clip/ViT-B-16.pt").resolve()
L65_CHECKPOINT = ROOT / "outputs/l65/train/clip_joint_smoke100/checkpoint_l65_clip_joint_step100.pt"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
EXPECTED_CLIP = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def image_path(video, frame):
    return RAW_ROOT / str(video) / f"{int(frame):06d}.png"


def crop_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= .10 * bw; y1 -= .10 * bh; x2 += .10 * bw; y2 += .10 * bh
    return max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))


def fit_units():
    out = []
    for line in UNITS.read_text().splitlines():
        if not line.strip():
            continue
        u = json.loads(line)
        if u.get("split") == "fit" and u.get("dataset") in ("refer_kitti_v1", "refer_kitti_v2"):
            out.append(u)
    return out


def stratified(units, seed):
    rng = random.Random(seed)
    cats = ("positive", "multi_positive", "inactive", "present_uncovered")
    buckets = {(d, c): [] for d in ("refer_kitti_v1", "refer_kitti_v2") for c in cats}
    for u in units:
        buckets.setdefault((u["dataset"], u.get("category", "unknown")), []).append(u)
    for b in buckets.values(): rng.shuffle(b)
    order = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]: order.append(buckets[key].pop())
    return order


def numeric(t, begin, end):
    return torch.cat((t["geometry"][begin:end].float(), t["motion"][begin:end].float(),
                      t["lifecycle"][begin:end].float(), t["context"][begin:end].float(),
                      t["objectness"][begin:end].float().reshape(-1, 1)), 1)


class StreamingClipLora:
    def __init__(self, device="cuda:0", crop_batch=8):
        import clip
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(str(CLIP_WEIGHTS), device=self.device)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.clip = clip
        self.crop_batch = int(crop_batch)

    def image_joint_tokens(self, pixel_batch):
        """Visual forward is grad-enabled so attached LoRA receives real gradients."""
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
        side = int(round(spatial.shape[1] ** .5))
        if side * side != spatial.shape[1]: raise AssertionError(f"patch shape {tuple(spatial.shape)}")
        spatial = spatial.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side)
        spatial = F.adaptive_avg_pool2d(spatial.float(), (4, 4)).flatten(2).transpose(1, 2)
        spatial = F.normalize(spatial, dim=-1)
        return torch.cat((global_token[:, None, :], spatial), 1).float()

    @torch.no_grad()
    def text_joint_tokens(self, sentence):
        tokens = self.clip.tokenize([str(sentence)], truncate=True).to(self.device)
        m = self.model; x = m.token_embedding(tokens).to(m.dtype)
        x = x + m.positional_embedding.to(x.dtype)
        x = m.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = m.ln_final(x); x = x @ m.text_projection
        valid = tokens[0].ne(0).bool(); feat = F.normalize(x.float(), dim=-1)[0]
        eos = int(tokens[0].argmax())
        return feat.detach().cpu(), valid.detach().cpu(), feat[eos].detach().cpu(), tokens[0].detach().cpu()

    def encode_unit(self, video, frame, boxes):
        path = image_path(video, frame)
        if not path.is_file(): raise FileNotFoundError(path)
        chunks = []
        with Image.open(path) as source:
            image = source.convert("RGB")
            pixels = []
            for box in boxes:
                c = crop_box(box, image.width, image.height)
                if c[2] <= c[0] or c[3] <= c[1]: raise ValueError(f"empty crop {path} {box} {c}")
                pixels.append(self.preprocess(image.crop(c)))
            for start in range(0, len(pixels), self.crop_batch):
                batch = torch.stack(pixels[start:start + self.crop_batch]).to(self.device)
                chunks.append(self.image_joint_tokens(batch))
                del batch
        return torch.cat(chunks, 0) if chunks else torch.empty((0, 17, 512), device=self.device), path


def load_unit_features(unit, runtime, labels=True):
    bank = torch.load(Path(unit["bank_path"]), map_location="cpu", weights_only=False)
    t = bank["tensors"]; b, e = int(unit["begin"]), int(unit["end"]); n = e - b
    if n != int(unit["candidate_count"]): raise AssertionError(f"candidate count {unit['unit_key']}")
    patches, image = runtime.encode_unit(unit["video"], int(unit["frame_id"]), t["box"][b:e].float().tolist())
    words, mask, _, _ = runtime.text_joint_tokens(unit["sentence"])
    if tuple(patches.shape) != (n, 17, 512): raise AssertionError(f"patch shape {patches.shape}")
    nums = numeric(t, b, e)
    y = None
    if labels:
        y = torch.zeros(n, dtype=torch.bool)
        indices = [int(x) for x in unit.get("positive_indices", [])]
        if any(x < 0 or x >= n for x in indices): raise AssertionError(f"positive index {unit['unit_key']}")
        if indices: y[torch.as_tensor(indices, dtype=torch.long)] = True
    result = {"unit": unit, "patches": patches.clone(), "words": words.clone(), "mask": mask.clone(), "numeric": nums.clone(), "target": y, "image": str(image), "candidate_count": n}
    del bank, t, patches, words, mask, nums
    return result


def balanced(score, target):
    vals = []
    if target.any(): vals.append(F.binary_cross_entropy_with_logits(score[target], torch.ones_like(score[target])))
    if (~target).any(): vals.append(F.binary_cross_entropy_with_logits(score[~target], torch.zeros_like(score[~target])))
    return torch.stack(vals).mean() if vals else score.new_zeros(())


def loss_fn(output, target):
    s = output["relevance_logit"]; pos = torch.nonzero(target, as_tuple=False).flatten(); neg = torch.nonzero(~target, as_tuple=False).flatten(); z = s.new_zeros(())
    bce = balanced(s, target)
    if len(pos) and len(neg):
        hard = neg[torch.argsort(s.detach()[neg], descending=True)[:min(24, len(neg))]]
        pair = F.softplus(.2 + s[hard][None, :] - s[pos][:, None]).mean()
        listwise = torch.logsumexp(s, 0) - torch.logsumexp(s[pos], 0)
    else: hard, pair, listwise = neg, z, z
    minimum = F.binary_cross_entropy_with_logits(s[pos], torch.ones_like(s[pos])) if len(pos) else z
    inactive = balanced(s, torch.zeros_like(target)) if not len(pos) else z
    null = F.binary_cross_entropy_with_logits(output["null_logit"], s.new_tensor(float(not target.any())))
    brier = (torch.sigmoid(s) - target.float()).square().mean()
    total = bce + .5 * pair + .5 * listwise + .5 * minimum + inactive + null + .05 * brier
    return total, {"total": float(total.detach()), "bce": float(bce.detach()), "pairwise": float(pair.detach()), "listwise": float(listwise.detach()), "minimum_positive": float(minimum.detach()), "inactive": float(inactive.detach()), "null": float(null.detach()), "brier": float(brier.detach()), "positive_count": int(pos.numel()), "hard_count": int(hard.numel())}
