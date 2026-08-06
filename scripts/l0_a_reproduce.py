#!/usr/bin/env python
"""Stage L0-A minimal reproduction for official LocateAnything-3B.

Loads the official checkpoint from a local directory, runs a small fixed set of
queries (single grounding, multi-category detection, dense text detection,
pointing, negative/no-object), and saves raw outputs plus parsed boxes.
"""
import argparse
import json
import os
import time

import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nvidia/LocateAnything-3B")
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="outputs/l0_a_reproduction")
    p.add_argument("--device", default="cuda")
    p.add_argument("--generation-mode", default="hybrid")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--limit-images", type=int, default=8)
    p.add_argument("--greedy", action="store_true", help="use temperature=0 instead of official defaults")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device).eval()

    image_paths = sorted(
        os.path.join(args.images, f)
        for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )[: args.limit_images]

    queries = {
        "ground_single": "Locate a single instance that matches the following description: person.",
        "ground_multi": "Locate all the instances that match the following description: people wearing red shirts.",
        "detect": "Locate all the instances that matches the following description: person</c>car</c>bicycle.",
        "detect_text": "Detect all the text in box format.",
        "point": "Point to: the traffic light.",
        "negative": "Locate all the instances that matches the following description: purple elephant.",
    }

    records = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        for qname, question in queries.items():
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }]
            text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = processor.process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(args.device)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    generation_mode=args.generation_mode,
                    repetition_penalty=1.0 if args.greedy else 1.1,
                    verbose=False,
                )
                if args.greedy:
                    gen_kwargs.update(temperature=0.0, top_p=None, top_k=None)
                else:
                    gen_kwargs.update(temperature=0.7, top_p=0.9, top_k=None)
                response = model.generate(
                    pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws"),
                    tokenizer=tokenizer,
                    **gen_kwargs,
                )
            elapsed = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated()
            answer = response[0] if isinstance(response, tuple) else response

            boxes = []
            import re
            for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
                x1, y1, x2, y2 = (int(g) for g in m.groups())
                boxes.append({
                    "x1": x1 / 1000 * w, "y1": y1 / 1000 * h,
                    "x2": x2 / 1000 * w, "y2": y2 / 1000 * h,
                    "x1n": x1 / 1000, "y1n": y1 / 1000,
                    "x2n": x2 / 1000, "y2n": y2 / 1000,
                })
            points = []
            for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
                x, y = (int(g) for g in m.groups())
                points.append({"x": x / 1000 * w, "y": y / 1000 * h})
            no_object = "<box>none</box>" in answer.lower()
            truncated = answer.strip().endswith("<text_mask>") or len(answer) >= args.max_new_tokens * 4

            record = {
                "image": os.path.basename(img_path),
                "image_path": img_path,
                "image_size": [w, h],
                "query_name": qname,
                "prompt": question,
                "raw_answer": answer,
                "num_boxes": len(boxes),
                "boxes": boxes,
                "points": points,
                "no_object": no_object,
                "truncated": truncated,
                "inference_seconds": round(elapsed, 4),
                "peak_mem_bytes": peak_mem,
                "generation_mode": args.generation_mode,
                "max_new_tokens": args.max_new_tokens,
            }
            records.append(record)
            print(
                f"{os.path.basename(img_path)} | {qname:10s} | boxes={len(boxes):2d} | "
                f"none={no_object} | {elapsed:.2f}s | peak={peak_mem/1e9:.2f}GB"
            )

    out_json = os.path.join(args.out, "raw_outputs.jsonl")
    with open(out_json, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()
