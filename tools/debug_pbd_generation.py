#!/usr/bin/env python
"""Stage L0-B: instrumented PBD generation on a small fixed image set.

Writes generation_events.jsonl, object_tokens.jsonl, mapping_integrity.json and
l0_b_runtime.csv under outputs/l0_b_token_debug.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.models.object_tokens.extractor import ObjectTokenExtractor  # noqa: E402


MODEL_COMMIT = "783f656d127ee498137b5ff52603ce36c292d317"


def sha256_of_model(model_dir: str) -> str:
    hashes = {}
    for f in sorted(os.listdir(model_dir)):
        if f.startswith("model-") and f.endswith(".safetensors"):
            hashes[f] = hashlib.sha256(
                open(os.path.join(model_dir, f), "rb").read()
            ).hexdigest()
    return json.dumps(hashes, sort_keys=True)


def parse_boxes(answer: str, w: int, h: int):
    boxes = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        boxes.append([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h])
    return boxes


def build_queries(images_dir: str):
    names = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
    queries = []
    for name in names:
        base = [
            ("ground_single", "Locate a single instance that matches the following description: person."),
            ("detect", "Locate all the instances that matches the following description: person</c>car</c>bicycle."),
            ("negative", "Locate all the instances that matches the following description: purple elephant."),
            ("point", "Point to: the traffic light."),
        ]
        if name in ("000000000139.jpg",):
            base.append(("detect_text", "Detect all the text in box format."))
        for qname, qtext in base:
            queries.append((name, qname, qtext))
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default="outputs/l0_b_token_debug")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--generation-mode", default="hybrid")
    ap.add_argument("--limit-images", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    ckpt_hash = sha256_of_model(args.model)
    extractor = ObjectTokenExtractor(
        model,
        tokenizer,
        processor,
        model_dir=args.model,
        model_commit=MODEL_COMMIT,
        checkpoint_hash=ckpt_hash,
        seed=args.seed,
    )

    queries = build_queries(args.images_dir)[: args.limit_images * 4]
    events_path = os.path.join(args.out, "generation_events.jsonl")
    tokens_path = os.path.join(args.out, "object_tokens.jsonl")
    runtime_path = os.path.join(args.out, "l0_b_runtime.csv")
    ev_f = open(events_path, "w")
    tok_f = open(tokens_path, "w")
    rt_f = open(runtime_path, "w", newline="")
    rt_w = csv.writer(rt_f)
    rt_w.writerow([
        "image", "query_name", "generation_mode", "seconds", "peak_mem_gb",
        "steps", "switch_to_ar", "accepted_blocks", "parsed_boxes", "tokens",
        "answer", "truncated",
    ])

    integrity = {"samples": [], "summary": {}}
    sample_id = 0
    for image_name, qname, qtext in queries:
        img_path = os.path.join(args.images_dir, image_name)
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        result = extractor.extract(
            image,
            question=qtext,
            semantic_label=qname,
            source_frame=image_name,
            generation_mode=args.generation_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=None,
            repetition_penalty=1.1,
        )
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        trace = result["trace"]
        tokens = result["object_tokens"]

        w, h = image.size
        parsed = parse_boxes(trace["answer"], w, h)
        accepted = [e for e in trace["events"] if e.accepted and e.block_type == "coord_box"]
        truncated = len(trace["answer"]) >= args.max_new_tokens * 4

        for ev in trace["events"]:
            row = ev.to_dict()
            row["sample_index"] = sample_id
            row["image"] = image_name
            row["query_name"] = qname
            ev_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for tok in tokens:
            row = tok.to_dict()
            row["sample_index"] = sample_id
            row["image"] = image_name
            row["query_name"] = qname
            tok_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        sample = {
            "sample_index": sample_id,
            "image": image_name,
            "query_name": qname,
            "accepted_blocks": len(accepted),
            "parsed_boxes": len(parsed),
            "object_tokens": len(tokens),
            "order_match": True,
            "box_match": True,
            "point_boxes": 0,
            "none": "<box>None</box>" in trace["answer"] or "<box>none</box>" in trace["answer"],
            "rejected_blocks": sum(1 for e in trace["events"] if e.block_type == "error_box"),
            "fallback_events": sum(1 for e in trace["events"] if e.fallback_occurred),
        }
        if len(accepted) != len(parsed) or len(tokens) != len(accepted):
            sample["order_match"] = False
            sample["box_match"] = False
        else:
            for tok, box in zip(tokens, parsed):
                if abs(tok.box_xyxy[0] - box[0]) > 1e-3 or abs(tok.box_xyxy[1] - box[1]) > 1e-3:
                    sample["box_match"] = False
        integrity["samples"].append(sample)

        rt_w.writerow([
            image_name, qname, args.generation_mode, f"{elapsed:.3f}", f"{peak:.2f}",
            trace["num_steps"], trace["switch_to_ar_count"], len(accepted),
            len(parsed), len(tokens), trace["answer"], truncated,
        ])
        print(
            f"[{sample_id}] {image_name} | {qname:14s} | steps={trace['num_steps']:3d} | "
            f"acc={len(accepted)} parsed={len(parsed)} tok={len(tokens)} | {elapsed:.2f}s | "
            f"peak={peak:.2f}GB | ar_fb={trace['switch_to_ar_count']}"
        )
        sample_id += 1

    ev_f.close()
    tok_f.close()
    rt_f.close()

    n = len(integrity["samples"])
    integrity["summary"] = {
        "samples": n,
        "images": len(set(s["image"] for s in integrity["samples"])),
        "total_accepted_blocks": sum(s["accepted_blocks"] for s in integrity["samples"]),
        "total_parsed_boxes": sum(s["parsed_boxes"] for s in integrity["samples"]),
        "total_object_tokens": sum(s["object_tokens"] for s in integrity["samples"]),
        "order_mismatches": sum(1 for s in integrity["samples"] if not s["order_match"]),
        "box_mismatches": sum(1 for s in integrity["samples"] if not s["box_match"]),
        "mapping_integrity": 1.0 if all(
            s["order_match"] and s["box_match"] and s["accepted_blocks"] == s["object_tokens"]
            for s in integrity["samples"]
        ) else 0.0,
        "model_commit": MODEL_COMMIT,
        "checkpoint_hash": ckpt_hash,
    }
    with open(os.path.join(args.out, "mapping_integrity.json"), "w") as f:
        json.dump(integrity, f, ensure_ascii=False, indent=2)
    print("mapping integrity summary:", json.dumps(integrity["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
