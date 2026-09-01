"""Materialize L16 expression metadata and immutable video-level splits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
DANCE = ROOT / "data/refer_dance"
KITTI_TEST = {"0005", "0011", "0013", "0019"}
KITTI_VAL = {"0004", "0018"}


def encode_texts(texts, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    result = {}
    for start in range(0, len(texts), 256):
        batch = texts[start:start + 256]
        with torch.no_grad():
            value = model.encode_text(clip.tokenize(batch).to(device)).float()
        value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        for text, feature in zip(batch, value.cpu().numpy()):
            result[text] = feature.astype(np.float32)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="outputs/l16/data/protocol")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dance_raw = {}
    texts = set()
    for video_dir in sorted((DANCE / "expression").iterdir()):
        if not video_dir.is_dir():
            continue
        entries = []
        for path in sorted(video_dir.glob("*.json")):
            value = json.loads(path.read_text())
            sentence = value.get("sentence", path.stem.replace("-", " "))
            label = {str(key): [str(x) for x in ids]
                     for key, ids in value.get("label", {}).items()}
            entries.append({
                "expression": path.stem,
                "sentence": sentence,
                "label": label,
                "nonempty": any(bool(ids) for ids in label.values()),
            })
            texts.add(sentence)
        dance_raw[video_dir.name] = entries

    specs = encode_texts(sorted(texts), args.gpu)
    train_videos = sorted(video for video, entries in dance_raw.items()
                          if len(entries) == 39)
    official_eval = sorted(video for video, entries in dance_raw.items()
                           if len(entries) == 17)
    train_val = train_videos[::5]
    train = sorted(set(train_videos) - set(train_val))
    for video, entries in dance_raw.items():
        split = ("official_eval" if video in official_eval else
                 "train_val" if video in train_val else "train")
        for entry in entries:
            entry["split"] = split
            entry["spec"] = specs[entry["sentence"]].tolist()

    (out / "refer_dance_expressions.json").write_text(
        json.dumps(dance_raw, indent=1))

    all_kitti = [f"{index:04d}" for index in range(21)]
    kitti_train = sorted(set(all_kitti) - KITTI_TEST - KITTI_VAL)
    manifest = {
        "seed": 20260825,
        "selection_unit": "video",
        "kitti_v2": {
            "train": kitti_train,
            "train_val": sorted(KITTI_VAL),
            "official_eval": sorted(KITTI_TEST),
            "official_eval_queries": 862,
            "missing_records_materialized_by_l16": [
                "0016", "0017", "0018", "0020"
            ],
        },
        "refer_dance": {
            "train": train,
            "train_val": train_val,
            "official_eval": official_eval,
            "train_expressions": sum(len(dance_raw[v]) for v in train_videos),
            "train_nonempty_expressions": sum(
                sum(entry["nonempty"] for entry in dance_raw[v])
                for v in train_videos),
            "all_eval_expressions": sum(len(dance_raw[v]) for v in official_eval),
            "formal_nonempty_eval_queries": sum(
                sum(entry["nonempty"] for entry in dance_raw[v])
                for v in official_eval),
            "all_query_protocol": (
                "425 provenance-preserving expression/video pairs; 385 have "
                "empty GT and are not silently added to the 40-query HOTA row"
            ),
            "raw_images": {
                "train": 41796,
                "official_eval": 25508,
                "root": "/data1/LWR/vranlee/DATASETS/JDE/dancetrack",
            },
        },
        "external": {
            "storm_bench": {
                "annotation_lfs_oid": (
                    "65190803082fb9c2dee6fbb2bcbe6fbc51a98474e7dce095d8f4babc1bef9c2c"
                ),
                "annotation_bytes": 54027273,
                "raw_vidor_frames_local": False,
                "used_for_training": False,
            },
            "refer_bdd_local": False,
            "aerialmind_local": False,
        },
        "licenses": {
            "DanceTrack_annotations": "CC BY 4.0",
            "DanceTrack_media": "non-commercial research only",
            "Refer_Dance": "derived annotations from iKUN; upstream terms retained",
        },
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[l16-protocol] dance={sum(map(len, dance_raw.values()))} "
          f"train={len(train)} train_val={len(train_val)} "
          f"official_eval={len(official_eval)} out={out}", flush=True)


if __name__ == "__main__":
    main()

