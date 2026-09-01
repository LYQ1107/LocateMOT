#!/usr/bin/env python3
"""Label-free one-unit audit for the L64 real-PNG CLIP patch path."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WEIGHTS = Path("/home/lwr/.cache/clip/ViT-B-16.pt").resolve()
UNITS = ROOT / "outputs/l49/data/train_units.jsonl"
OUT = ROOT / "outputs/l64/audit/raw_patch_input"

import sys
sys.path.insert(0, str(ROOT))
from tools.l64_raw_patch_common import StreamingOpenAIClip, image_path, sha256


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    fit = [json.loads(x) for x in UNITS.read_text().splitlines() if x.strip()
           and json.loads(x).get("split") == "fit"
           and json.loads(x).get("dataset") in ("refer_kitti_v1", "refer_kitti_v2")]
    unit = fit[0]
    bank = torch.load(Path(unit["bank_path"]), map_location="cpu", weights_only=False)
    tensors = bank["tensors"]
    begin, end = int(unit["begin"]), int(unit["end"])
    if end - begin != int(unit["candidate_count"]):
        raise AssertionError("candidate count contract")
    boxes = tensors["box"][begin:end].float()
    image = image_path(unit["video"], int(unit["frame_id"]))
    start = time.time()
    encoder = StreamingOpenAIClip("cuda:0", batch_size=32)
    if any(p.requires_grad for p in encoder.model.parameters()):
        raise AssertionError("detector/image encoder is trainable")
    patches, path = encoder.encode_unit(unit["video"], int(unit["frame_id"]), boxes.tolist())
    text, mask = encoder.text_tokens(unit["sentence"])
    finite = bool(torch.isfinite(patches).all() and torch.isfinite(text).all())
    peak = int(torch.cuda.max_memory_allocated())
    payload = {
        "format": "locatemot-l64-raw-patch-input-audit-v1", "status": "complete",
        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
        "unit_key": unit["unit_key"], "dataset": unit["dataset"], "video": unit["video"],
        "frame_id": int(unit["frame_id"]), "candidate_count": int(end - begin),
        "candidate_rows_retained": True, "candidate_truncation": False,
        "image_path": str(path.resolve()), "image_exists": path.is_file(),
        "image_size": list(Image.open(path).size),
        "crop_rule": "L19 box, 10% padding, clip-to-image, OpenAI CLIP preprocess",
        "weights": str(WEIGHTS), "weights_sha256": sha256(WEIGHTS),
        "encoder": "frozen OpenAI CLIP ViT-B/16; real PNG crops",
        "patch_tokens_shape": list(patches.shape), "text_tokens_shape": list(text.shape),
        "text_valid_count": int(mask.sum()), "text_mask_shape": list(mask.shape),
        "feature_dtype": str(patches.dtype), "finite_patch_tokens": finite,
        "finite_text_tokens": bool(torch.isfinite(text).all()), "raw_cache_written": False,
        "labels_read_for_feature_construction": False, "gt_used_for_features": False,
        "token_span_alignment": "UNALIGNED", "static_motion_mask": "UNALIGNED",
        "detector_or_backbone_gradients": False, "peak_memory_bytes": peak,
        "elapsed_sec": time.time() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_patch_input.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({
        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
        "unit_source": str(UNITS), "unit_source_sha256": sha256(UNITS),
        "fit_only": True, "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "persistent_feature_cache": False,
    }, indent=2) + "\n")
    del encoder, patches, text, mask, boxes, tensors, bank
    print(json.dumps({"status": "complete", "output": str(OUT / "raw_patch_input.json"),
                      "patch_shape": payload["patch_tokens_shape"], "text_shape": payload["text_tokens_shape"],
                      "candidate_count": payload["candidate_count"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
