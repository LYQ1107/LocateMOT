"""Build the small V1 RMOT evaluation metadata used by Stage L13.

The V1 expressions/labels are downloaded from the official RMOT release,
while detector/CLIP/PBD candidate caches are reused from the repaired L11
KITTI cache for the three overlapping evaluation videos.  This script only
creates expression specs and official V1 GT templates; it does not touch the
shared UIDM checkpoint or candidate cache.

Usage:
  python tools/build_l13_refer_kitti_v1.py --gpu 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
V1 = ROOT / "outputs" / "l13" / "data" / "refer_kitti_v1"
CAND = ROOT / "outputs" / "l11" / "data" / "rmot_kitti"
REF = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos")
SEQMAP = REF / "rmot_official" / "datasets" / "data_path" / "seqmap.txt"
EVAL_SEQS = {"0005", "0011", "0013"}


def load_expressions(seq: str):
    out = {}
    for p in sorted((V1 / "expression" / seq).glob("*.json")):
        obj = json.loads(p.read_text())
        out[p.stem] = {
            "expression": p.stem,
            "sentence": obj.get("sentence", p.stem),
            "label": {str(k): [str(x) for x in v]
                      for k, v in obj.get("label", {}).items()},
            "ignore": obj.get("ignore", []),
            "video_name": obj.get("video_name", ""),
        }
    return out


def load_labels(seq: str):
    out = {}
    for p in sorted((V1 / "labels_with_ids" / "image_02" / seq).glob("*.txt")):
        rows = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            vals = line.split()
            rows.append((int(float(vals[0])), int(float(vals[1])),
                         *[float(x) for x in vals[2:6]]))
        out[int(p.stem)] = rows
    return out


def make_specs(exps, device):
    import clip

    texts = sorted({e["sentence"] for seq in exps.values() for e in seq.values()})
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    cache = {}
    for start in range(0, len(texts), 256):
        batch = texts[start:start + 256]
        with torch.no_grad():
            z = model.encode_text(clip.tokenize(batch).to(device)).float()
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cache.update({t: x.cpu().numpy().astype(np.float32)
                      for t, x in zip(batch, z)})
    return cache


def write_gt(seq, expr, meta, labels, image_size):
    w, h = image_size
    ids_by_frame = meta["label"]
    rows = []
    for frame_s, ids in ids_by_frame.items():
        frame = int(frame_s)
        idset = {int(x) for x in ids}
        for _cls, tid, x, y, bw, bh in labels.get(frame, []):
            if tid not in idset:
                continue
            rows.append((frame + 1, tid, x * w, y * h, bw * w, bh * h,
                         1, 1, 1))
    rows.sort()
    dst = V1 / "gt_template" / seq / expr / "gt.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for row in rows:
            f.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x)
                             for x in row) + "\n")
    return len(rows)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seqs = sorted({x.split("+", 1)[0] for x in SEQMAP.read_text().splitlines()
                   if x.strip()})
    all_exps = {seq: load_expressions(seq) for seq in
                sorted({p.name for p in (V1 / "expression").iterdir()
                        if p.is_dir()})}
    specs = make_specs(all_exps, device)
    metadata = {
        seq: [{**e, "spec": specs[e["sentence"]].tolist()}
              for e in sorted(exps.values(), key=lambda x: x["expression"])]
        for seq, exps in all_exps.items()
    }
    (V1 / "expressions.json").write_text(json.dumps(metadata, indent=1))

    gt_rows = 0
    gt_queries = 0
    for line in SEQMAP.read_text().splitlines():
        if not line.strip():
            continue
        seq, expr = line.strip().split("+", 1)
        meta = all_exps[seq][expr]
        rec = pickle.load(open(CAND / f"{seq}.pkl", "rb"))
        rows = write_gt(seq, expr, meta, load_labels(seq), rec["image_size"])
        gt_rows += rows
        gt_queries += 1

    manifest = {
        "seqmap": str(SEQMAP),
        "eval_sequences": seqs,
        "queries": gt_queries,
        "gt_rows": gt_rows,
        "expression_zip_sha256": sha256(V1 / "raw" / "expression.zip"),
        "labels_zip_sha256": sha256(V1 / "raw" / "labels_with_ids.zip"),
        "candidate_source": str(CAND),
        "clip_model": "ViT-B/32",
    }
    (V1 / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[l13v1] specs={sum(len(x) for x in metadata.values())} "
          f"eval_queries={gt_queries} gt_rows={gt_rows} device={device}",
          flush=True)


if __name__ == "__main__":
    main()
