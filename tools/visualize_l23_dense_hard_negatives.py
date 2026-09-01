"""Create representative L23 dense hard-negative audit images.

GT is used only to annotate/stratify audit cases. It is never used to choose
features, train a model, or select a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.rmot_dense_correspondence_scorer import DenseQueryCorrespondenceScorer  # noqa: E402
from tools.train_l23_dense_correspondence import arrays_for, model_score  # noqa: E402
from tools.train_rmot_candidate_scorer import load_bank, load_metadata, make_refs  # noqa: E402


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1])), min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1])); bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-8, aa + bb - inter)


def record_for(video: str) -> dict:
    for path in (ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl", ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"):
        if path.exists(): return pickle.loads(path.read_bytes())
    raise FileNotFoundError(video)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json"); ap.add_argument("--v3-root", default="outputs/l23/candidate_bank_v3"); ap.add_argument("--checkpoint", default="outputs/l23/train/D0_dense_roi_query_cross_attention_S250/checkpoint_d0_step250.pt"); ap.add_argument("--raw-root", default="data/kitti_tracking_training/image_02"); ap.add_argument("--out-root", default="outputs/l23/eval/dense_hard_negative_visualization"); ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    def p(x: str) -> Path:
        x = Path(x); return x if x.is_absolute() else ROOT / x
    manifest, v3_root, checkpoint, raw_root, out_root = map(p, (args.manifest, args.v3_root, args.checkpoint, args.raw_root, args.out_root))
    if out_root.exists(): raise FileExistsError(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    data = json.loads(manifest.read_text()); queries = sorted(data["queries"], key=lambda x: int(x["query_index"])); query_by_index = {int(q["query_index"]): q for q in queries}; metadata = load_metadata(); videos = sorted({str(q["video"]) for q in queries}); banks = {v: load_bank(v3_root / "kitti" / f"{v}.pt") for v in videos}; refs = [r for r in make_refs(queries, metadata, banks) if r["split"] == "screening"]; records = {v: record_for(v) for v in videos}
    device = torch.device(args.device); model = DenseQueryCorrespondenceScorer(stage="D0").to(device); payload = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model"]); model.eval()
    needed = {"positive_below_model_hard": None, "same_dense_similar_candidate": None, "multi_positive_frame": None, "main_hard_negative": None, "reserve_hard_negative": None, "NULL_frame": None}
    for ref in refs:
        bank = banks[ref["video"]]; size = ref["end"] - ref["begin"]; labels = ref["positive"].astype(bool); rows = np.arange(size, dtype=np.int64)
        with torch.inference_mode(): scores = model_score(model, {k: torch.as_tensor(v, device=device) for k, v in arrays_for(ref, bank, rows).items()}).cpu().numpy()
        pos = np.flatnonzero(labels); neg = np.flatnonzero(~labels); objectness = bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1); pre = neg[np.argsort(-objectness[neg], kind="stable")[:min(96, len(neg))]]; hard = pre[np.argsort(-scores[pre], kind="stable")[:min(24, len(pre))]]
        dense = bank["tensors"]["dense_roi"][ref["begin"]:ref["end"]].float().numpy(); dense = dense / np.maximum(np.linalg.norm(dense, axis=1, keepdims=True), 1e-6); sim = (dense @ dense[pos].T).max(axis=1) if len(pos) else np.zeros(size, np.float32)
        pool = bank["tensors"]["pool_id"][ref["begin"]:ref["end"]].numpy().astype(int); candidates = []
        if len(pos) and len(hard) and needed["positive_below_model_hard"] is None and scores[pos].min() < scores[hard].max(): candidates.append("positive_below_model_hard")
        if len(pos) and len(neg) and needed["same_dense_similar_candidate"] is None: candidates.append("same_dense_similar_candidate")
        if len(pos) > 1 and needed["multi_positive_frame"] is None: candidates.append("multi_positive_frame")
        if len(hard) and np.any(pool[hard] == 0) and needed["main_hard_negative"] is None: candidates.append("main_hard_negative")
        if len(hard) and np.any(pool[hard] == 1) and needed["reserve_hard_negative"] is None: candidates.append("reserve_hard_negative")
        if not len(pos) and needed["NULL_frame"] is None: candidates.append("NULL_frame")
        for category in candidates:
            if category == "same_dense_similar_candidate": chosen = int(neg[np.argmax(sim[neg])])
            elif category in ("main_hard_negative", "reserve_hard_negative"): chosen = int(max((x for x in hard.tolist() if pool[x] == (0 if category.startswith("main") else 1)), key=lambda x: scores[x]))
            elif category == "NULL_frame": chosen = int(np.argmax(scores))
            else: chosen = int(hard[np.argmax(scores[hard])]) if len(hard) else int(np.argmax(scores))
            pos_for_display = int(pos[np.argmin(scores[pos])]) if len(pos) else None
            frame = int(ref["frame_id"]); image_path = raw_root / ref["video"] / f"{frame:06d}.png"; image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            expression = str(query_by_index[int(ref["query_index"])] ["expression"]); target_ids = {str(x) for x in metadata[(ref["video"], expression)].get("label", {}).get(str(frame), [])}; gt_boxes = {str(k): v for k, v in {int(f): fdata for fdata in records[ref["video"]]["frames"] for f in [fdata["frame"]]}.get(frame, {}).get("gt_boxes", {}).items()}
            target_boxes = [np.asarray(gt_boxes[x], np.float32) for x in target_ids if x in gt_boxes]
            def annotation(index: int | None) -> dict | None:
                if index is None: return None
                row = ref["begin"] + index; box = bank["tensors"]["box"][row].float().numpy().tolist(); overlap = max((iou(box, x) for x in target_boxes), default=0.0)
                return {"relative_index": index, "track_id": int(bank["tensors"]["track_id"][row]), "pool": "main" if int(pool[index]) == 0 else "reserve", "bbox": box, "score": float(scores[index]), "iou_to_target": overlap, "dense_feature_similarity_to_positive": float(sim[index]) if len(pos) else None}
            for index in range(size):
                box = bank["tensors"]["box"][ref["begin"] + index].float().numpy().astype(int); color = (0, 255, 0) if labels[index] else (150, 150, 150)
                if index == chosen: color = (0, 0, 255)
                if index == pos_for_display: color = (0, 255, 255)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2); cv2.putText(image, f"{index}:{scores[index]:.2f}", (box[0], max(14, box[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
            filename = f"case_{len([x for x in needed.values() if x is not None]):02d}_{category}_q{int(ref['query_index'])}_{ref['video']}_f{frame:06d}.png"; image_out = out_root / filename; cv2.imwrite(str(image_out), image)
            needed[category] = {"category": category, "query_index": int(ref["query_index"]), "video": ref["video"], "frame": frame, "expression": expression, "target_ids": sorted(target_ids), "candidate_count": size, "model": "D0_dense_roi_query_cross_attention_step250", "gt_use": "audit_annotation_only", "hard_rule": {"objectness_prefilter": 96, "model_topk": 24}, "chosen": annotation(chosen), "lowest_positive": annotation(pos_for_display), "positive_count": len(pos), "main_positive_count": int(np.sum(pool[pos] == 0)) if len(pos) else 0, "reserve_positive_count": int(np.sum(pool[pos] == 1)) if len(pos) else 0, "image": str(image_out)}
        if all(x is not None for x in needed.values()): break
    report = {"format": "locatemot-l23-dense-hard-negative-visualization-v1", "manifest": str(manifest), "v3_root": str(v3_root), "checkpoint": str(checkpoint), "screening_gt_used_for_selection": False, "audit_gt_used_for_annotation": True, "required_categories": list(needed), "category_available": {k: v is not None for k, v in needed.items()}, "saved_cases": sum(v is not None for v in needed.values()), "cases": [v for v in needed.values() if v is not None]}
    (out_root / "representative_cases.json").write_text(json.dumps(report, indent=2) + "\n"); (out_root / "representative_cases.md").write_text("# L23 dense hard-negative visualization\n\nGT is used only for audit annotation; the screening set was not used for checkpoint/model selection.\n\n" + "\n".join(f"- {k}: `{v is not None}`" for k, v in needed.items()) + "\n")
    print(json.dumps({"out_root": str(out_root), "saved_cases": report["saved_cases"], "category_available": report["category_available"]}, indent=2))


if __name__ == "__main__": main()
