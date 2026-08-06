#!/usr/bin/env python
"""Re-validates mapping integrity from saved events/tokens outputs."""
import argparse
import json
import os
import re
from collections import defaultdict


def parse_boxes(answer, w, h):
    boxes = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        boxes.append([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h])
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/l0_b_token_debug")
    args = ap.parse_args()

    events = [json.loads(l) for l in open(os.path.join(args.out, "generation_events.jsonl"))]
    tokens = [json.loads(l) for l in open(os.path.join(args.out, "object_tokens.jsonl"))]
    runtime = [json.loads(l) for l in open(os.path.join(args.out, "l0_b_runtime.csv"))] \
        if os.path.exists(os.path.join(args.out, "l0_b_runtime.csv")) else []

    by_sample = defaultdict(lambda: {"events": [], "tokens": []})
    for e in events:
        by_sample[e["sample_index"]]["events"].append(e)
    for t in tokens:
        by_sample[t["sample_index"]]["tokens"].append(t)

    summary = {
        "samples": len(by_sample),
        "accepted_blocks": sum(
            1 for e in events if e.get("accepted") and e.get("block_type") == "coord_box"
        ),
        "object_tokens": len(tokens),
        "rejected_blocks": sum(1 for e in events if e.get("block_type") == "error_box"),
        "fallback_events": sum(1 for e in events if e.get("fallback_occurred")),
        "point_events": sum(1 for e in events if e.get("block_type") == "point_box"),
        "empty_events": sum(1 for e in events if e.get("block_type") == "empty_box"),
    }
    mismatches = []
    for sid, data in by_sample.items():
        accepted = [e for e in data["events"] if e.get("accepted") and e.get("block_type") == "coord_box"]
        toks = data["tokens"]
        if len(accepted) != len(toks):
            mismatches.append({"sample": sid, "reason": "count_mismatch",
                               "accepted": len(accepted), "tokens": len(toks)})
            continue
        for e, t in zip(accepted, toks):
            if e.get("output_order") != t.get("object_index"):
                mismatches.append({"sample": sid, "reason": "order_mismatch"})
                break
            if e.get("parsed_box") != t.get("box_xyxy"):
                mismatches.append({"sample": sid, "reason": "box_mismatch",
                                   "event_box": e.get("parsed_box"), "token_box": t.get("box_xyxy")})
                break
    summary["mismatches"] = mismatches
    summary["mapping_integrity"] = 1.0 if not mismatches else 0.0
    with open(os.path.join(args.out, "mapping_integrity.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
