#!/usr/bin/env python3
"""L52 A2: small raw-image query/region forward feasibility check.

This is deliberately an availability/shape smoke, not a scorer evaluation.  It
loads one fit unit, decodes two candidate crops plus one full-frame context
image, and keeps all image tensors in memory only until the command exits.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
DATA = ROOT / "outputs/l49/data"
WEIGHTS = Path("/home/lwr/.cache/clip/ViT-B-16.pt")
OUT = ROOT / "outputs/l52/audit/forward_feasibility.json"
sys.path.insert(0, str(ROOT))
from locatemot.rmot.l49_data import load_bank, sha256_file  # noqa: E402


def crop_box(box, width, height, padding=0.10):
    x1, y1, x2, y2 = [float(x) for x in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return (max(0, int(math.floor(x1 - padding * bw))),
            max(0, int(math.floor(y1 - padding * bh))),
            min(width, int(math.ceil(x2 + padding * bw))),
            min(height, int(math.ceil(y2 + padding * bh))))


@torch.inference_mode()
def patch_tokens(model, pixels, device):
    visual = model.visual
    pixel = pixels.to(device)
    pixel = pixel.to(dtype=visual.conv1.weight.dtype)
    x = visual.conv1(pixel).reshape(pixel.shape[0], visual.conv1.out_channels, -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat((cls, x), 1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x).permute(1, 0, 2)
    x = visual.transformer(x).permute(1, 0, 2)[:, 1:]
    side = int(round(x.shape[1] ** 0.5))
    if side * side != x.shape[1]:
        raise AssertionError(f"non-square patch token count {x.shape[1]}")
    raw_shape = list(x.shape)
    pooled = F.adaptive_avg_pool2d(
        x.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side), (4, 4)
    ).flatten(2).transpose(1, 2).float()
    return raw_shape, pooled


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()}")
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("A2 requires authorized GPU0 for CLIP forward")
    units = [json.loads(x) for x in (DATA / "train_units.jsonl").read_text().splitlines() if x.strip()]
    fit = [x for x in units if x.get("split") == "fit"]
    if not fit:
        raise RuntimeError("no fit unit")
    unit = sorted(fit, key=lambda x: x["unit_key"])[0]
    bank = load_bank(str(unit["dataset"]), str(unit["video"]))
    tensors = bank["tensors"]
    begin, end = int(unit["begin"]), int(unit["end"])
    if end - begin != int(unit["candidate_count"]):
        raise AssertionError("candidate count drift")
    rows = list(range(begin, min(end, begin + 2)))
    frame = int(tensors["frame"][begin])
    image_path = ROOT / "data/kitti_tracking_training/image_02" / str(unit["video"]) / f"{frame:06d}.png"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    import clip
    started = time.time()
    model, preprocess = clip.load(str(WEIGHTS), device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        candidate_images = []
        candidate_boxes = []
        for row in rows:
            box = tensors["box"][row].tolist()
            cb = crop_box(box, image.width, image.height)
            if cb[2] <= cb[0] or cb[3] <= cb[1]:
                raise ValueError(f"invalid crop {box} -> {cb}")
            candidate_boxes.append(list(cb))
            candidate_images.append(preprocess(image.crop(cb)))
        # One whole-frame context image; it is not a persistent feature map.
        pixels = torch.stack(candidate_images + [preprocess(image)])
    if not torch.isfinite(pixels).all():
        raise AssertionError("nonfinite input pixels")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    raw_shape, tokens = patch_tokens(model, pixels, device)
    elapsed = time.time() - started
    text = torch.load(ROOT / "outputs/l48/data/text_cache.pt", map_location="cpu", weights_only=False)
    text_index = text["sentence_to_index"].get(unit["sentence"])
    if text_index is None:
        raise KeyError(unit["sentence"])
    text_tokens = text["token_hidden"][int(text_index)]
    text_mask = text["attention_mask"][int(text_index)]
    result = {
        "schema": "locatemot-l52-a2-forward-v1",
        "status": "pass",
        "cwd": str(Path.cwd().resolve()),
        "device": str(device),
        "unit_key": unit["unit_key"], "dataset": unit["dataset"], "video": unit["video"],
        "frame_id": frame, "candidate_rows_checked": len(rows),
        "candidate_count_full": int(end - begin), "candidate_set_truncated": False,
        "candidate_boxes": candidate_boxes, "image_path": str(image_path.resolve()),
        "crop_contract": {"padding": 0.10, "boundary": "clip", "box_source": "L19 observation box"},
        "candidate_patch_tokens_shape": raw_shape,
        "candidate_patch_tokens_shape_after_fixed_pool": [len(rows), 16, int(tokens.shape[-1])],
        "whole_frame_context_tokens_shape": [1, 16, int(tokens.shape[-1])],
        "text_tokens_shape": list(text_tokens.shape), "text_mask_shape": list(text_mask.shape),
        "text_valid_tokens": int(text_mask.bool().sum()),
        "image_encoder": "OpenAI CLIP ViT-B/16 visual patch encoder",
        "weights": str(WEIGHTS.resolve()), "weights_sha256": sha256_file(WEIGHTS),
        "finite": bool(torch.isfinite(tokens).all() and torch.isfinite(text_tokens).all()),
        "key_alignment": "pass: rows/frame/video/box decoded from same L19 bank unit",
        "streaming_no_persistent_raw_or_dense_cache": True,
        "official_test_labels_read": False, "screening_gt_used": False,
        "ordinary_mot_ovmot_touched": False, "elapsed_sec": elapsed,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "note": "CLIP is visual-only here; text tokens are the frozen L48 cache. This is a weak feasibility probe, not verified cross-modal grounding.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    del pixels, tokens, model, bank
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"status": "pass", "output": str(OUT), "elapsed_sec": elapsed, "shapes": {"patch": raw_shape, "text": list(text_tokens.shape)}}))


if __name__ == "__main__":
    main()
