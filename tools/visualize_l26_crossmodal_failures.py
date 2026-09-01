#!/usr/bin/env python3
"""Score and render fixed representative L23 hard-negative cases with L26."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
V5 = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"
SOURCE = ROOT / "outputs/l23/eval/dense_hard_negative_visualization_corrected/representative_cases.json"
CKPT = ROOT / "outputs/l26/train/C1_crossmodal_adapter_S2000/checkpoint_c1_step2000.pt"
RAW = ROOT / "data/kitti_tracking_training/image_02"
OUT = ROOT / "outputs/l26/fallback/F7_hard_negative_visualization"


def box(draw, coords, color, width=4):
    xy = tuple(float(x) for x in coords)
    draw.rectangle(xy, outline=color, width=width)


def main():
    source = json.loads(SOURCE.read_text())
    text = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((V5 / "text_manifest.json").read_text())["expressions"]
    text_index = {(str(x["video"]), str(x["expression"])): int(x["query_index"]) for x in manifest}
    from locatemot.models.l26_crossmodal_adapter import L26CrossModalAdapter

    model = L26CrossModalAdapter(variant="token_region")
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"])
    model.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    with torch.inference_mode():
        for i, case in enumerate(source["cases"]):
            video = str(case["video"]); expression = str(case["expression"])
            data = torch.load(V5 / "kitti" / f"{video}.pt", map_location="cpu", weights_only=False)
            t = data["tensors"]
            frame_ids = t["frame_ids"].tolist()
            frame = int(case["frame"]); fi = frame_ids.index(frame)
            begin, end = int(t["frame_ptr"][fi]), int(t["frame_ptr"][fi + 1])
            ti = text_index[(video, expression)]
            out = model(text["token_hidden"][ti], text["attention_mask"][ti].bool(), t["dino_roi_tokens_v5"][begin:end], t["roi_coords_v5"][begin:end])
            scores = out["score"].float()
            order = torch.argsort(scores, descending=True).tolist()
            chosen = int(case["chosen"]["relative_index"])
            low = case.get("lowest_positive")
            low_idx = int(low["relative_index"]) if low is not None else None
            image_path = RAW / video / f"{frame:06d}.png"
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            box(draw, case["chosen"]["bbox"], "red")
            if low is not None:
                box(draw, low["bbox"], "lime")
            name = f"case_{i:02d}_{case['category']}_q{case['query_index']}_{video}_f{frame:06d}.png"
            image_out = OUT / name
            image.save(image_out)
            chosen_rank = order.index(chosen) + 1 if chosen in order else None
            low_rank = order.index(low_idx) + 1 if low_idx is not None and low_idx in order else None
            cases.append({
                "category": case["category"], "query_index": int(case["query_index"]), "text_index": ti,
                "video": video, "frame": frame, "expression": expression, "image": str(image_out),
                "selection": "fixed L23 representative case; no screening GT used for model selection",
                "annotation": "source GT/IoU metadata retained for audit annotation only",
                "candidate_count": int(end - begin), "positive_count_source": int(case.get("positive_count", 0)),
                "chosen": {**case["chosen"], "l26_score": float(scores[chosen]), "l26_rank": chosen_rank},
                "lowest_positive": ({**low, "l26_score": float(scores[low_idx]), "l26_rank": low_rank} if low is not None else None),
                "top5_relative_indices": order[:5],
                "l26_top1_is_source_positive": bool(low_idx is not None and order[0] == low_idx),
                "attention_entropy": float(out["attention_entropy"]),
                "text_word_tokens": int(text["attention_mask"][ti].sum()),
            })
    report = {"format": "locatemot-l26-crossmodal-failure-visualization-v1", "source_cases": str(SOURCE), "v5_root": str(V5), "checkpoint": str(CKPT), "screening_gt_used_for_selection": False, "audit_gt_used_for_annotation": True, "saved_cases": len(cases), "categories": sorted({x["category"] for x in cases}), "cases": cases}
    (OUT / "visualization.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "visualization.md").write_text("# L26 cross-modal hard-negative visualization\n\nRed is the fixed hard candidate; lime is the fixed lowest annotated positive. Selection is inherited from the fixed L23 cases and uses no screening GT for model selection.\n")
    print(json.dumps({"out": str(OUT), "saved_cases": len(cases), "categories": report["categories"]}, indent=2))


if __name__ == "__main__":
    main()
