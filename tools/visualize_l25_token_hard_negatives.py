#!/usr/bin/env python3
"""Make an independent L25 audit view of representative token hard negatives.

Case selection is inherited from the already completed L23 audit (which did
not use screening GT for model selection). GT is used only to annotate the
saved audit metadata, never to choose a training model or sampling point.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
CASES = ROOT / "outputs/l23/eval/dense_hard_negative_visualization_corrected/representative_cases.json"
V4 = ROOT / "outputs/l25/candidate_bank_v4"
OUT = ROOT / "outputs/l25/eval/token_hard_negative_visualization"
RAW = ROOT / "data/kitti_tracking_training/image_02"


def box(b):
    return tuple(int(round(x)) for x in b)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = json.loads(CASES.read_text())
    enriched = []
    for index, case in enumerate(src["cases"]):
        video = str(case["video"])
        frame = int(case["frame"])
        bank = torch.load(V4 / "kitti" / f"{video}.pt", map_location="cpu", weights_only=False)
        t = bank["tensors"]
        frame_ids = t["frame_ids"].numpy().astype(np.int64)
        ptr = t["frame_ptr"].numpy().astype(np.int64)
        frame_pos = int(np.flatnonzero(frame_ids == frame)[0])
        begin, end = int(ptr[frame_pos]), int(ptr[frame_pos + 1])
        def row_feature(item):
            return t["dense_points_v4"][begin + int(item["relative_index"])].numpy()
        chosen = case["chosen"]
        chosen_feat = row_feature(chosen)
        annotated = dict(case)
        annotated["l25_v4"] = {
            "bank": str(V4 / "kitti" / f"{video}.pt"),
            "row_range": [begin, end],
            "chosen_row": begin + int(chosen["relative_index"]),
            "chosen_token_shape": list(chosen_feat.shape),
            "chosen_token_l2_mean": float(np.linalg.norm(chosen_feat, axis=1).mean()),
            "chosen_dense_point_mean": chosen_feat.mean(axis=0).tolist(),
            "source": chosen["pool"],
            "gt_use": "audit_annotation_only",
        }
        if case.get("lowest_positive") is not None:
            positive = case["lowest_positive"]
            positive_feat = row_feature(positive)
            annotated["l25_v4"].update({
                "positive_row": begin + int(positive["relative_index"]),
                "positive_token_shape": list(positive_feat.shape),
                "positive_token_l2_mean": float(np.linalg.norm(positive_feat, axis=1).mean()),
                "chosen_vs_positive_token_cosine": cosine(chosen_feat.mean(0), positive_feat.mean(0)),
            })
        image_path = RAW / video / f"{frame:06d}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        # Green = annotated lowest positive; red = representative hard candidate.
        x1, y1, x2, y2 = box(chosen["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        if case.get("lowest_positive") is not None:
            x1, y1, x2, y2 = box(case["lowest_positive"]["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 3)
        label = f"{case['category']} red={chosen['pool']} track={chosen['track_id']}"
        cv2.putText(image, label[:110], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        out_image = OUT / f"case_{index:02d}_{case['category']}_q{case['query_index']}_{video}_f{frame:06d}.png"
        cv2.imwrite(str(out_image), image)
        annotated["image"] = str(out_image)
        annotated["feature_provenance"] = "L25 v4 CLIP ViT-B/16 projected 14x14 dense map; fixed candidate-box sampling"
        enriched.append(annotated)
    report = {
        "format": "locatemot-l25-token-hard-negative-visualization-v1",
        "manifest": src["manifest"],
        "source_cases": str(CASES),
        "v4_root": str(V4),
        "screening_gt_used_for_selection": False,
        "audit_gt_used_for_annotation": True,
        "required_categories": src["required_categories"],
        "saved_cases": len(enriched),
        "legend": {"red": "representative hard candidate", "green": "lowest annotated positive"},
        "cases": enriched,
    }
    (OUT / "visualization.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT / "visualization.md").write_text(
        "# L25 token hard-negative visualization\n\n"
        "Red boxes are representative hard candidates; green boxes are audit-only lowest positives. "
        "Selection was inherited from the prior GT-free model audit; GT is annotation-only.\n\n"
        + "\n".join(f"- `{c['category']}`: `{c['image']}`" for c in enriched) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(OUT), "saved_cases": len(enriched)}, indent=2))


if __name__ == "__main__":
    main()
