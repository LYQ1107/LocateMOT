#!/usr/bin/env python3
"""Build a compact word-level text table for L48 fit/validation queries.

Existing V5 RoBERTa states are reused by sentence.  Only V1 sentences missing
from that frozen table are encoded with the same local RoBERTa checkpoint.  No
evaluation labels or images are read here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.rmot.l48_data import V5_TEXT, V5_TEXT_MANIFEST, sha256_file  # noqa: E402

QUERY_MANIFEST = ROOT / "outputs/l48/data/query_manifest.jsonl"
OUT = ROOT / "outputs/l48/data/text_cache.pt"
OUT_META = ROOT / "outputs/l48/data/text_cache_manifest.json"
ROBERTA = Path("/home/lwr/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b")


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [json.loads(line) for line in QUERY_MANIFEST.read_text().splitlines() if line.strip()]
    sentences = sorted({str(row["sentence"]) for row in rows})
    v5_manifest = json.loads(V5_TEXT_MANIFEST.read_text())["expressions"]
    v5_by_sentence = {}
    for row in v5_manifest:
        v5_by_sentence.setdefault(str(row["sentence"]), int(row["query_index"]))
    v5 = torch.load(V5_TEXT, map_location="cpu", weights_only=False)
    v5_hidden = v5["token_hidden"]
    v5_mask = v5["attention_mask"].bool()
    hidden_rows = []
    mask_rows = []
    reused = []
    missing = []
    for sentence in sentences:
        index = v5_by_sentence.get(sentence)
        if index is None:
            missing.append(sentence)
            hidden_rows.append(None)
            mask_rows.append(None)
        else:
            reused.append(sentence)
            hidden_rows.append(v5_hidden[index].cpu().half())
            mask_rows.append(v5_mask[index].cpu().bool())
    if missing:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA), local_files_only=True)
        model = AutoModel.from_pretrained(str(ROBERTA), local_files_only=True).eval()
        with torch.inference_mode():
            for start in range(0, len(missing), 64):
                text = missing[start:start + 64]
                enc = tokenizer(text, padding="max_length", truncation=True,
                                 max_length=64, return_tensors="pt")
                value = model(input_ids=enc["input_ids"],
                              attention_mask=enc["attention_mask"]).last_hidden_state
                for offset in range(len(text)):
                    sentence = text[offset]
                    slot = sentences.index(sentence)
                    hidden_rows[slot] = value[offset].cpu().half()
                    mask_rows[slot] = enc["attention_mask"][offset].cpu().bool()
        del model
    hidden = torch.stack(hidden_rows).contiguous()
    masks = torch.stack(mask_rows).contiguous()
    if hidden.ndim != 3 or hidden.shape[-1] != 768 or not bool(torch.isfinite(hidden.float()).all()):
        raise FloatingPointError(f"invalid text cache shape={tuple(hidden.shape)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sentences": sentences, "token_hidden": hidden,
               "attention_mask": masks,
               "sentence_to_index": {s: i for i, s in enumerate(sentences)}}
    torch.save(payload, str(OUT) + ".tmp")
    os.replace(str(OUT) + ".tmp", OUT)
    sources = {"reused_v5": len(reused), "encoded_local_roberta": len(missing)}
    meta = {
        "format": "locatemot-l48-word-level-text-cache-v1",
        "query_manifest": str(QUERY_MANIFEST.resolve()),
        "query_manifest_sha256": sha256_file(QUERY_MANIFEST),
        "sentence_count": len(sentences), "shape": list(hidden.shape),
        "sources": sources,
        "v5_text_cache": str(V5_TEXT.resolve()), "v5_text_cache_sha256": sha256_file(V5_TEXT),
        "roberta_snapshot": str(ROBERTA),
        "roberta_weights_sha256": sha256_file(ROBERTA / "model.safetensors"),
        "labels_read": False, "screening_labels_read": False,
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_mask": "UNALIGNED/not claimed",
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
