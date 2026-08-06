#!/usr/bin/env python
"""Stage L0-B feature sanity check on a small YouTube-VOS sample (read-only)."""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.evaluation.token_sanity import (  # noqa: E402
    cosine,
    feature_vector,
    load_youtube_vos_meta,
    mask_boxes_for_frame,
    match_tokens_to_gt,
    select_videos,
    write_sanity_outputs,
)
from locatemot.models.object_tokens.extractor import ObjectTokenExtractor  # noqa: E402

MODEL_COMMIT = "783f656d127ee498137b5ff52603ce36c292d317"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-root", default="/data3/testdata/vranlee/.MOTSynth.partial/YouTube-VOS-2019")
    ap.add_argument("--out", default="outputs/l0_b_token_debug")
    ap.add_argument("--num-videos", type=int, default=10)
    ap.add_argument("--frames-per-video", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()

    meta = load_youtube_vos_meta(args.data_root, "train")
    video_ids = select_videos(meta, n=args.num_videos, min_objects=2)
    print("selected videos:", video_ids)

    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    ckpt_hash = json.dumps({
        f: __import__("hashlib").sha256(open(os.path.join(args.model, f), "rb").read()).hexdigest()
        for f in sorted(os.listdir(args.model))
        if f.startswith("model-") and f.endswith(".safetensors")
    }, sort_keys=True)
    extractor = ObjectTokenExtractor(
        model, tok, proc, model_dir=args.model,
        model_commit=MODEL_COMMIT, checkpoint_hash=ckpt_hash, seed=args.seed,
    )

    # per-frame token cache: (video, frame, category) -> [(token, gt_id, iou)]
    frame_cache = {}
    candidate_stats = defaultdict(lambda: {"gt": 0, "recall": {0.3: 0, 0.5: 0, 0.7: 0}})

    for vid in video_ids:
        info = meta["videos"][vid]
        frames = sorted(info["objects"][next(iter(info["objects"]))]["frames"])
        if len(frames) < 2:
            continue
        chosen = frames[: args.frames_per_video]
        for frame in chosen:
            jpg = os.path.join(args.data_root, "train", "JPEGImages", vid, f"{frame}.jpg")
            mask_png = os.path.join(args.data_root, "train", "Annotations", vid, f"{frame}.png")
            if not os.path.exists(jpg) or not os.path.exists(mask_png):
                continue
            gt_boxes = mask_boxes_for_frame(mask_png)
            # group objects by category
            objs_by_cat = defaultdict(list)
            for obj_id, oinfo in info["objects"].items():
                if frame in oinfo["frames"] and int(obj_id) in gt_boxes:
                    objs_by_cat[oinfo["category"]].append(int(obj_id))
            image = Image.open(jpg).convert("RGB")
            for cat, obj_ids in objs_by_cat.items():
                q = f"Locate all the instances that matches the following description: {cat}."
                result = extractor.extract(
                    image, question=q, semantic_label=cat, source_frame=f"{vid}/{frame}",
                    generation_mode="hybrid", max_new_tokens=args.max_new_tokens,
                    temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.1,
                    in_token_limit=4096,
                )
                tokens = result["object_tokens"]
                matched = match_tokens_to_gt(tokens, gt_boxes, image.size, iou_thresh=0.5)
                frame_cache[(vid, frame, cat)] = matched
                for obj_id in obj_ids:
                    candidate_stats[(vid, cat)]["gt"] += 1
                    for thresh in (0.3, 0.5, 0.7):
                        hit = any(
                            m[1] == obj_id and m[2] >= thresh
                            for m in matched
                        )
                        if hit:
                            candidate_stats[(vid, cat)]["recall"][thresh] += 1
                print(f"  {vid}/{frame} [{cat}] gt={len(obj_ids)} matched={len(matched)}")

    # build pairs
    pairs = []
    rng = np.random.RandomState(args.seed)
    keys = sorted(frame_cache.keys())
    pos, neg_same, neg_cross = 0, 0, 0
    for i, (ka, items_a) in enumerate(frame_cache.items()):
        for ta, oid_a, _ in items_a:
            # positive: same video same object different frame
            for kb, items_b in frame_cache.items():
                if ka[0] == kb[0] and ka[1] != kb[1]:
                    for tb, oid_b, _ in items_b:
                        if oid_a == oid_b and pos < 60:
                            pairs.append(_pair("positive", ka, ta, oid_a, kb, tb, oid_b))
                            pos += 1
            # negatives
            for kb, items_b in frame_cache.items():
                if ka[0] == kb[0]:
                    for tb, oid_b, _ in items_b:
                        if oid_a != oid_b and neg_same < 60:
                            pairs.append(_pair("negative_same_video", ka, ta, oid_a, kb, tb, oid_b))
                            neg_same += 1
                else:
                    for tb, oid_b, _ in items_b:
                        if neg_cross < 60:
                            pairs.append(_pair("negative_cross_video", ka, ta, oid_a, kb, tb, oid_b))
                            neg_cross += 1
    rng.shuffle(pairs)

    features = ["pbd_box_end_feature", "pbd_coordinate_mean_feature", "pbd_full_block_mean_feature",
                "region_feature", "fused_feature"]
    scores = {name: {"pos": [], "neg": []} for name in features}
    for p in pairs:
        for name in features:
            va = feature_vector(p["token_a"], name)
            vb = feature_vector(p["token_b"], name)
            c = cosine(va, vb)
            p[name.replace("_feature", "")] = c
            if c is not None:
                group = "pos" if p["pair_type"] == "positive" else "neg"
                scores[name][group].append(c)

    metrics = {"num_videos": len(video_ids), "pairs": len(pairs)}
    for name in features:
        pos_v = scores[name]["pos"]
        neg_v = scores[name]["neg"]
        auc = None
        if len(pos_v) >= 2 and len(neg_v) >= 2:
            y = [1] * len(pos_v) + [0] * len(neg_v)
            x = pos_v + neg_v
            auc = float(roc_auc_score(y, x))
        metrics[name] = {
            "positive_mean": float(np.mean(pos_v)) if pos_v else None,
            "positive_median": float(np.median(pos_v)) if pos_v else None,
            "negative_mean": float(np.mean(neg_v)) if neg_v else None,
            "negative_median": float(np.median(neg_v)) if neg_v else None,
            "auc": auc,
            "valid_pos": len(pos_v),
            "valid_neg": len(neg_v),
        }
    # candidate recall
    recall = {f"recall@{t}": 0.0 for t in (0.3, 0.5, 0.7)}
    total_gt = 0
    for key, st in candidate_stats.items():
        total_gt += st["gt"]
        for t in (0.3, 0.5, 0.7):
            recall[f"recall@{t}"] += st["recall"][t]
    for t in (0.3, 0.5, 0.7):
        recall[f"recall@{t}"] = round(recall[f"recall@{t}"] / max(1, total_gt), 4)
    metrics["candidate_recall"] = recall
    metrics["gt_objects"] = total_gt
    metrics["data_root"] = args.data_root
    metrics["note"] = "sanity check only; GT used for prompt/pair labeling and matching, not candidate filtering"

    write_sanity_outputs(
        pairs,
        metrics,
        os.path.join(args.out, "sanity_pairs.csv"),
        os.path.join(args.out, "sanity_metrics.json"),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _pair(pair_type, ka, ta, oid_a, kb, tb, oid_b):
    return {
        "pair_type": pair_type,
        "video_a": ka[0], "frame_a": ka[1], "obj_a": oid_a,
        "video_b": kb[0], "frame_b": kb[1], "obj_b": oid_b,
        "token_a": ta, "token_b": tb,
    }


if __name__ == "__main__":
    main()
