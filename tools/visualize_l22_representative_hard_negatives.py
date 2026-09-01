"""Representative, transparent hard-negative cases for the L22 v2 bank.

The displayed score is frozen CLIP text-to-tight-crop cosine only.  It is not a
trained model and is used solely to make the failure modes inspectable.  GT is
read only to annotate IoU/positive state; it is never used to alter features or
select a training checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import defaultdict
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


def load_metadata():
    out = {}
    for p in (ROOT / "outputs/l11/data/rmot_kitti/expressions.json",
              ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"):
        if not p.exists(): continue
        for video, entries in json.loads(p.read_text()).items():
            for entry in entries:
                out[(str(video), str(entry.get("expression", entry.get("sentence", ""))))] = entry
    return out


def main():
    manifest_path = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
    bank_root = ROOT / "outputs/l22/candidate_bank_v2"
    raw_root = ROOT / "data/kitti_tracking_training/image_02"
    out = ROOT / "outputs/l22/eval/hard_negative_visualization_representative"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text()); metadata = load_metadata()
    banks, records = {}, {}
    for video in sorted({str(q["video"]) for q in manifest["queries"]}):
        banks[video] = torch.load(bank_root / "kitti" / f"{video}.pt", map_location="cpu", weights_only=False)
        rp = ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl"
        if not rp.exists(): rp = ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"
        records[video] = {int(f["frame"]): f for f in pickle.loads(rp.read_bytes())["frames"]}
    candidates = defaultdict(list); scanned = 0
    for q in manifest["queries"]:
        if q["split"] != "screening": continue
        video, expr = str(q["video"]), str(q["expression"]); bank = banks[video]; t = bank["tensors"]
        labels = json.loads((bank_root / "kitti" / f"{video}.labels.json").read_text())["candidate_gt"]
        entry = metadata[(video, expr)]; spec = np.asarray(entry["spec"], np.float32); spec /= max(1e-6, np.linalg.norm(spec))
        for fi, frame in enumerate(t["frame_ids"].tolist()):
            frame = int(frame); begin, end = int(t["frame_ptr"][fi]), int(t["frame_ptr"][fi + 1]); n = end - begin
            target_ids = {str(x) for x in entry.get("label", {}).get(str(frame), [])}; gt = records[video][frame].get("gt_boxes", {})
            target_boxes = [np.asarray(gt[x], np.float32) for x in target_ids if x in gt]
            box = t["box"][begin:end].numpy().astype(np.float32); pool = t["pool_id"][begin:end].numpy().astype(int); track = t["track_id"][begin:end].numpy().astype(int)
            tight = t["crop_tight"][begin:end].float().numpy(); unit = tight / np.maximum(np.linalg.norm(tight, axis=1, keepdims=True), 1e-6)
            score = unit @ spec; pos = np.asarray([labels[i] is not None and str(labels[i]) in target_ids for i in range(begin, end)], bool); neg = np.flatnonzero(~pos); pos_idx = np.flatnonzero(pos)
            best_neg = int(neg[np.argmax(score[neg])]) if len(neg) else None; best_pos = int(pos_idx[np.argmin(score[pos_idx])]) if len(pos_idx) else None
            if best_neg is None: continue
            hard_iou = max((iou(box[best_neg], b) for b in target_boxes), default=0.)
            hard_app = float(np.max(unit[best_neg] @ unit[pos_idx].T)) if len(pos_idx) else 0.
            row_common = {"query_index": int(q["query_index"]), "video": video, "expression": expr, "frame": frame, "target_ids": sorted(target_ids), "score_type": "frozen_CLIP_text_to_tight_crop_cosine", "candidate_count": n,
                          "hard_negative": {"relative_index": best_neg, "track_id": int(track[best_neg]), "pool": "main" if pool[best_neg] == 0 else "reserve", "bbox": box[best_neg].tolist(), "score": float(score[best_neg]), "iou_to_target": float(hard_iou), "feature_similarity_to_positive": hard_app, "objectness": float(t["objectness"][begin + best_neg])}}
            if best_pos is not None:
                pos_iou = max((iou(box[best_pos], b) for b in target_boxes), default=0.)
                row_common["lowest_positive"] = {"relative_index": best_pos, "track_id": int(track[best_pos]), "pool": "main" if pool[best_pos] == 0 else "reserve", "bbox": box[best_pos].tolist(), "score": float(score[best_pos]), "iou_to_target": float(pos_iou), "feature_similarity_to_positive": 1.0, "objectness": float(t["objectness"][begin + best_pos])}
                row_common["positive_below_model_hard"] = bool(score[best_pos] < score[best_neg])
            row_common["multi_positive"] = bool(len(pos_idx) >= 2)
            row_common["positive_count"] = int(len(pos_idx)); row_common["main_positive_count"] = int(np.sum(pos & (pool == 0))); row_common["reserve_positive_count"] = int(np.sum(pos & (pool == 1)))
            row_common["is_null"] = bool(not target_ids or not len(pos_idx)); scanned += 1
            hardness = float(score[best_neg] + hard_app + hard_iou)
            for category in ("positive_below_model_hard" if row_common.get("positive_below_model_hard") else "", "same_class_similar_candidate" if hard_app >= .85 else "", "multi_positive_frame" if row_common["multi_positive"] else "", "main_hard_negative" if pool[best_neg] == 0 else "reserve_hard_negative", "NULL_frame" if row_common["is_null"] else ""):
                if category: candidates[category].append((hardness, row_common, box.copy(), pos_idx.copy(), best_neg, pool.copy(), track.copy(), score.copy()))
    selected=[]
    required=("positive_below_model_hard","same_class_similar_candidate","multi_positive_frame","main_hard_negative","reserve_hard_negative","NULL_frame")
    for category in required:
        items=sorted(candidates[category], key=lambda x:x[0], reverse=True)
        if items: selected.append((category, items[0]))
    # Add diverse cases until the report has at least twelve images when data permits.
    used={(c, item[1]["query_index"], item[1]["video"], item[1]["frame"]) for c,item in selected}
    for category in required:
        for item in sorted(candidates[category], key=lambda x:x[0], reverse=True):
            key=(category,item[1]["query_index"],item[1]["video"],item[1]["frame"])
            if key in used: continue
            selected.append((category,item)); used.add(key)
            if len(selected)>=18: break
        if len(selected)>=18: break
    case_rows=[]
    for number,(category,item) in enumerate(selected):
        _, meta, boxes, pos_idx, hard, pools, tracks, scores = item; image_path=raw_root/meta["video"]/f"{meta['frame']:06d}.png"; image=cv2.imread(str(image_path));
        for idx in pos_idx.tolist():
            b=boxes[idx]; x1,y1,x2,y2=[int(round(v)) for v in b]; cv2.rectangle(image,(x1,y1),(x2,y2),(0,210,0),2); cv2.putText(image,f"P {tracks[idx]}",(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,210,0),1,cv2.LINE_AA)
        b=boxes[hard]; x1,y1,x2,y2=[int(round(v)) for v in b]; cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),3); cv2.putText(image,f"H {meta['hard_negative']['pool']} {tracks[hard]}",(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),2,cv2.LINE_AA)
        title=f"{category} q{meta['query_index']} {meta['video']} f{meta['frame']} score={meta['hard_negative']['score']:.3f} sim={meta['hard_negative']['feature_similarity_to_positive']:.3f} IoU={meta['hard_negative']['iou_to_target']:.3f}"
        cv2.putText(image,title,(6,18),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),2,cv2.LINE_AA); cv2.putText(image,title,(6,18),cv2.FONT_HERSHEY_SIMPLEX,.36,(20,20,20),1,cv2.LINE_AA)
        filename=f"case_{number:02d}_{category}_q{meta['query_index']}_{meta['video']}_f{meta['frame']:06d}.png"; cv2.imwrite(str(out/filename),image)
        meta={**meta,"category":category,"image":str(out/filename)}; case_rows.append(meta)
    result={"format":"locatemot-l22-representative-hard-negative-v1","manifest":str(manifest_path),"manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),"bank_root":str(bank_root),"gt_use":"audit annotation only; no training/model selection","score_type":"frozen CLIP text-to-tight-crop cosine","screening_units_scanned":scanned,"required_categories":list(required),"category_available":{k:bool(candidates[k]) for k in required},"saved_cases":len(case_rows),"cases":case_rows}
    (out/"representative_cases.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# L22 representative hard-negative cases","", "The score is frozen CLIP text-to-tight-crop cosine for transparent audit only. GT annotates IoU/positive state; it does not select training data.",""]
    for i,c in enumerate(case_rows): lines.append(f"- `{c['category']}` q{c['query_index']} {c['video']} frame {c['frame']}: hard pool `{c['hard_negative']['pool']}`, track `{c['hard_negative']['track_id']}`, score `{c['hard_negative']['score']:.4f}`, IoU `{c['hard_negative']['iou_to_target']:.4f}`, feature similarity `{c['hard_negative']['feature_similarity_to_positive']:.4f}`, NULL `{c['is_null']}`")
    (out/"representative_cases.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"output":str(out),"scanned":scanned,"saved_cases":len(case_rows),"category_available":result["category_available"]},indent=2))


if __name__ == "__main__": main()
