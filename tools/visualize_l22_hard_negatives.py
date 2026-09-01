"""Create a compact, auditable visualization of frozen-bank hard negatives."""
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


def iou(a, b):
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    aa = max(0., float(a[2] - a[0])) * max(0., float(a[3] - a[1]))
    bb = max(0., float(b[2] - b[0])) * max(0., float(b[3] - b[1]))
    return inter / max(1e-8, aa + bb - inter)


def metadata(root):
    out = {}
    for p in (root / "outputs/l11/data/rmot_kitti/expressions.json",
              root / "outputs/l16/data/kitti_missing/records/expressions.json"):
        if not p.exists(): continue
        for video, entries in json.loads(p.read_text()).items():
            for e in entries:
                out[(str(video), str(e.get("expression", e.get("sentence", ""))))] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    ap.add_argument("--bank-root", default="outputs/l22/candidate_bank_v2")
    ap.add_argument("--raw-root", default="data/kitti_tracking_training/image_02")
    ap.add_argument("--out-root", default="outputs/l22/eval/hard_negative_visualization")
    args = ap.parse_args()
    manifest = Path(args.manifest); bank_root = Path(args.bank_root); raw_root = Path(args.raw_root); out = Path(args.out_root)
    if not manifest.is_absolute(): manifest = ROOT / manifest
    if not bank_root.is_absolute(): bank_root = ROOT / bank_root
    if not raw_root.is_absolute(): raw_root = ROOT / raw_root
    if not out.is_absolute(): out = ROOT / out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    md = metadata(ROOT); manifest_data = json.loads(manifest.read_text())
    cases = []
    for q in manifest_data["queries"]:
        if q["split"] != "screening": continue
        video, expression = str(q["video"]), str(q["expression"])
        bank = torch.load(bank_root / "kitti" / f"{video}.pt", map_location="cpu", weights_only=False)
        labels = json.loads((bank_root / "kitti" / f"{video}.labels.json").read_text())["candidate_gt"]
        t = bank["tensors"]; entry = md[(video, expression)]
        record_path = ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl"
        if not record_path.exists(): record_path = ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"
        record = pickle.loads(record_path.read_bytes()); record_frames = {int(f["frame"]): f for f in record["frames"]}
        for fi, frame in enumerate(t["frame_ids"].tolist()):
            frame = int(frame); begin, end = int(t["frame_ptr"][fi]), int(t["frame_ptr"][fi + 1])
            targets = {str(x) for x in entry.get("label", {}).get(str(frame), [])}
            if not targets: continue
            gt = record_frames[frame].get("gt_boxes", {})
            target_boxes = [np.asarray(gt[x], np.float32) for x in targets if x in gt]
            pos = np.asarray([labels[i] is not None and str(labels[i]) in targets for i in range(begin, end)], bool)
            neg = np.flatnonzero(~pos)
            if not len(pos.nonzero()[0]) or not len(neg): continue
            boxes = t["box"][begin:end].numpy().astype(np.float32); clips = t["clip"][begin:end].float().numpy()
            unit = clips / np.maximum(np.linalg.norm(clips, axis=1, keepdims=True), 1e-6)
            pos_i = np.flatnonzero(pos); app = np.max(unit @ unit[pos_i].T, axis=1)
            obj = t["objectness"][begin:end].float().numpy().reshape(-1)
            hard = neg[np.argsort(-obj[neg], kind="stable")[:min(12, len(neg))]]
            if not len(hard): continue
            hard_i = max(hard.tolist(), key=lambda i: (
                float(app[i]), max((iou(boxes[i], target) for target in target_boxes), default=0.)))
            hard_iou = max((iou(boxes[hard_i], b) for b in target_boxes), default=0.)
            hardness = float(app[hard_i] + hard_iou + obj[hard_i])
            cases.append({"query_index": int(q["query_index"]), "video": video, "expression": expression,
                          "frame": frame, "begin": begin, "positive_indices": pos_i.tolist(), "hard_index": int(hard_i),
                          "hard_objectness": float(obj[hard_i]), "hard_clip_similarity_to_positive": float(app[hard_i]),
                          "hard_iou_to_target": float(hard_iou), "hardness": hardness,
                          "target_ids": sorted(targets), "boxes": boxes.tolist()})
    cases.sort(key=lambda x: x["hardness"], reverse=True); cases = cases[:24]
    for number, case in enumerate(cases):
        image_path = raw_root / case["video"] / f"{case['frame']:06d}.png"; image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(image_path)
        for idx in case["positive_indices"]:
            b = np.asarray(case["boxes"][idx]); x1,y1,x2,y2 = [int(round(v)) for v in b]
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,210,0),2); cv2.putText(image,"POS",(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,210,0),1,cv2.LINE_AA)
        idx = case["hard_index"]; b=np.asarray(case["boxes"][idx]); x1,y1,x2,y2=[int(round(v)) for v in b]
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),3); cv2.putText(image,"HARD NEG",(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2,cv2.LINE_AA)
        title=f"q{case['query_index']} {case['video']} f{case['frame']} obj={case['hard_objectness']:.3f} app={case['hard_clip_similarity_to_positive']:.3f} iou={case['hard_iou_to_target']:.3f}"
        cv2.putText(image,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),2,cv2.LINE_AA); cv2.putText(image,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1,cv2.LINE_AA)
        cv2.imwrite(str(out / f"case_{number:02d}_q{case['query_index']}_{case['video']}_f{case['frame']:06d}.png"), image)
    summary={"format":"locatemot-l22-hard-negative-visualization-v1","manifest":str(manifest),"manifest_sha256":__import__('hashlib').sha256(manifest.read_bytes()).hexdigest(),"screening_cases_scanned":len(cases),"saved_cases":len(cases),"selection":"negative among frozen-objectness top-12, ranked by CLIP similarity to same-frame positives plus IoU/objectness for audit only","gt_used":"visualization labels only; no feature or training input","cases":cases}
    (out/"hard_negative_cases.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# L22 hard-negative visualization", "", "Red boxes are frozen-objectness hard negatives; green boxes are expression-positive bank rows. This is an audit visualization, not a model result.", "", f"Saved `{len(cases)}` screening cases.", ""]
    for i,c in enumerate(cases): lines.append(f"- `case_{i:02d}` q{c['query_index']} {c['video']} frame {c['frame']}: hard objectness `{c['hard_objectness']:.4f}`, positive-CLIP similarity `{c['hard_clip_similarity_to_positive']:.4f}`, target IoU `{c['hard_iou_to_target']:.4f}`")
    (out/"hard_negative_cases.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"output":str(out),"saved_cases":len(cases)},indent=2))


if __name__ == "__main__": main()
