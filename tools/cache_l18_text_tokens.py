"""Cache token-level frozen CLIP text states for Stage L18."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))


def collect_sentences() -> list[str]:
    paths = (
        ROOT / "outputs/l11/data/rmot_kitti/expressions.json",
        ROOT / "outputs/l16/data/kitti_missing/records/expressions.json",
        ROOT / "outputs/l16/data/protocol/refer_dance_expressions.json",
        ROOT / "outputs/l8/data/rmot_train/expressions.json",
        ROOT / "outputs/l8/data/rmot_eval/expressions.json",
    )
    values = set()
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for entries in payload.values():
            for entry in entries:
                text = entry.get("sentence", entry.get("expression", ""))
                if text:
                    values.add(str(text))
    if not values:
        raise RuntimeError("no expressions found")
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--out", default="outputs/l18/data/text_cache")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import clip

    sentences = collect_sentences()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    tokens = clip.tokenize(sentences)
    hidden = []
    with torch.no_grad():
        for start in range(0, len(sentences), args.batch):
            chunk = tokens[start:start + args.batch].to(device)
            value = model.token_embedding(chunk).type(model.dtype)
            value = value + model.positional_embedding.type(model.dtype)
            value = value.permute(1, 0, 2)
            value = model.transformer(value)
            value = value.permute(1, 0, 2)
            value = model.ln_final(value).float().cpu()
            hidden.append(value)
    values = torch.cat(hidden).numpy().astype(np.float16)
    masks = (tokens.numpy() != 0)
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tokens.npy", values)
    np.save(out / "masks.npy", masks)
    (out / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False,
                                                    indent=1) + "\n")
    manifest = {
        "format": "locatemot-l18-text-cache-v1",
        "encoder": "OpenAI CLIP ViT-B/32 token transformer states",
        "token_shape": list(values.shape),
        "mask_shape": list(masks.shape),
        "sentences_sha256": hashlib.sha256(
            json.dumps(sentences, ensure_ascii=False).encode()).hexdigest(),
        "token_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
