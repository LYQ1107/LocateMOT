"""Stage L1-B pilot split builder (video-level disjoint, seed=20260806).

Writes configs/l1_b/pilot_videos.json with per-dataset selected videos
and sampled frames.  Only metadata is read; no inference here.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
OUT = ROOT / "configs" / "l1_b"
SEED = 20260806

DANCETRACK = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
MOT17 = Path("/data1/LWR/vranlee/DATASETS/JDE/MOT17")
MOT20 = Path("/data1/LWR/vranlee/M4FTMoveOut4Doing/ByteTrack-mbt/datasets/"
             "mix_mot20_ch/mot20_train")
TAO = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
           "TAO-Amodal")
YTVOS = Path("/data3/testdata/vranlee/.MOTSynth.partial/YouTube-VOS-2019")
MOSE = Path("/data3/testdata/vranlee/.MOTSynth.partial/MOSEv2")
BDD_LABELS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/"
                  "annotations/box_track_20")
BDD_IMAGES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/"
                  "bdd100k/images/track")


def pick_dancetrack(rng, n_videos=6, max_frames=60):
    train = DANCETRACK / "train"
    cands = []
    for v in sorted(train.iterdir()):
        if not (train / v.name).is_dir():
            continue
        gt = train / v.name / "gt" / "gt.txt"
        if not gt.exists():
            continue
        ids = set()
        for line in gt.read_text().splitlines():
            p = line.split(",")
            if len(p) >= 9 and int(p[7]) == 1:
                ids.add(int(p[1]))
        frames = sorted((train / v.name / "img1").glob("*.jpg"))
        if len(ids) >= 5 and len(frames) >= max_frames:
            cands.append((v.name, len(ids), len(frames)))
    cands.sort(key=lambda x: x[0])
    rng.shuffle(cands)
    selected = cands[:n_videos]
    out = {}
    for name, _, nf in selected:
        frames = sorted(int(p.stem) for p in
                        (train / name / "img1").glob("*.jpg"))
        # deterministic strided sampling
        idx = [round(i * (len(frames) - 1) / (max_frames - 1))
               for i in range(max_frames)]
        out[name] = sorted({frames[i] for i in idx})
    return out


def pick_mot(seq_root, n_seqs, max_frames, base_names=None):
    seqs = sorted(x for x in seq_root.iterdir() if x.is_dir())
    if base_names:
        seqs = [s for s in seqs if s.name in base_names]
    out = {}
    for seq in seqs[:n_seqs]:
        gt = seq / "gt" / "gt.txt"
        if not gt.exists():
            continue
        frames = sorted(int(p.stem) for p in (seq / "img1").glob("*.jpg"))
        if not frames:
            continue
        idx = [round(i * (len(frames) - 1) / (max_frames - 1))
               for i in range(max_frames)]
        out[seq.name] = sorted({frames[i] for i in idx})
    return out


def pick_bdd(rng, n_videos=6, max_frames=40):
    lab_dir = BDD_LABELS / "train"
    img_dir = BDD_IMAGES / "train"
    cands = []
    for p in sorted(lab_dir.glob("*.json")):
        vname = p.stem
        if not (img_dir / vname).exists():
            continue
        v = json.loads(p.read_text())
        ids = set()
        for fr in v:
            for lab in fr.get("labels", []):
                if lab.get("box2d") is not None:
                    ids.add(str(lab.get("id")))
        if len(ids) >= 5 and len(v) >= max_frames:
            cands.append((vname, len(ids)))
    cands.sort(key=lambda x: x[0])
    rng.shuffle(cands)
    out = {}
    for vname, _ in cands[:n_videos]:
        v = json.loads((lab_dir / f"{vname}.json").read_text())
        idx = [round(i * (len(v) - 1) / (max_frames - 1))
               for i in range(max_frames)]
        out[vname] = sorted({int(v[i]["frameIndex"]) for i in idx})
    return out


def pick_tao(rng, n_videos=6, max_frames=40):
    d = json.loads((TAO / "annotations" / "train.json").read_text())
    ann_by_img = defaultdict(list)
    for a in d["annotations"]:
        ann_by_img[a["image_id"]].append(a)
    img_by_video = defaultdict(list)
    for im in d["images"]:
        img_by_video[im["video"]].append(im)
    cat_name = {c["id"]: c["name"] for c in d["categories"]}
    cands = []
    for vname, ims in img_by_video.items():
        ids = set()
        cats = set()
        for im in ims:
            for a in ann_by_img.get(im["id"], []):
                ids.add((im["video"], a["track_id"]))
                cats.add(a["category_id"])
        if len(ids) >= 3 and len(ims) >= max_frames:
            cands.append((vname, len(ids), sorted(ims, key=lambda x:
                          x["frame_index"])))
    cands.sort(key=lambda x: x[0])
    rng.shuffle(cands)
    out = {}
    for vname, _, ims in cands[:n_videos]:
        idx = [round(i * (len(ims) - 1) / (max_frames - 1))
               for i in range(max_frames)]
        frames = []
        cats = set()
        for i in sorted({idx[i] for i in range(max_frames)}):
            im = ims[i]
            frames.append({
                "frame_index": int(im["frame_index"]),
                "file_name": im["file_name"],
            })
            for a in ann_by_img.get(im["id"], []):
                cats.add(cat_name.get(a["category_id"], "object"))
        out[vname] = {"frames": frames,
                      "query": "Locate all the instances that matches the "
                               "following description: " +
                               ", ".join(sorted(cats)[:6]) + "."}
    return out


def pick_vos(meta, img_root, ann_root, n_videos, max_frames, rng,
             same_cat=True):
    cands = []
    for vid, info in meta["videos"].items():
        objs = info.get("objects", {})
        if isinstance(objs, list):
            continue
        cats = {o.get("category", "") for o in objs.values()}
        all_frames = set()
        for o in objs.values():
            all_frames.update(o.get("frames", []))
        if len(objs) >= 3 and len(all_frames) >= max_frames:
            if same_cat and len(cats) >= 2:
                continue
            cands.append((vid, sorted(all_frames)))
    cands.sort(key=lambda x: x[0])
    rng.shuffle(cands)
    out = {}
    for vid, frames in cands[:n_videos]:
        idx = [round(i * (len(frames) - 1) / (max_frames - 1))
               for i in range(max_frames)]
        out[vid] = sorted({frames[i] for i in idx})
    return out


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    split = {}
    split["dancetrack_train"] = pick_dancetrack(rng)
    split["mot17_train"] = pick_mot(
        MOT17 / "train", 3, 80,
        base_names={"MOT17-02-SDP", "MOT17-04-SDP", "MOT17-09-SDP"})
    split["mot20_train"] = pick_mot(MOT20, 2, 80,
                                    base_names={"MOT20-01", "MOT20-02"})
    split["tao_amodal_train"] = pick_tao(rng)
    split["bdd100k_train"] = pick_bdd(rng)
    yt = json.loads((YTVOS / "train" / "meta.json").read_text())
    split["ytvos_train"] = pick_vos(
        yt, YTVOS / "train" / "JPEGImages", YTVOS / "train" / "Annotations",
        12, 30, rng, same_cat=False)
    mo = json.loads((MOSE / "meta_train.json").read_text())
    # MOSE has object-id lists; choose videos with >=3 object ids
    cands = [(vid, info["objects"], len(info["frames"]))
             for vid, info in mo["videos"].items()
             if isinstance(info.get("objects"), list)
             and len(info["objects"]) >= 3 and len(info["frames"]) >= 50]
    cands.sort(key=lambda x: x[0])
    rng.shuffle(cands)
    selected = {}
    for vid, objs, nf in cands[:10]:
        frames = mo["videos"][vid]["frames"]
        idx = [round(i * (len(frames) - 1) / 49) for i in range(50)]
        selected[vid] = sorted({frames[i] for i in idx})
    split["mose_train"] = selected
    counts = {k: sum(len(v) for v in val.values())
              if isinstance(val, dict) else 0
              for k, val in split.items()}
    (OUT / "pilot_videos.json").write_text(
        json.dumps(split, indent=1))
    print("PILOT_SPLIT_DONE", counts, "total", sum(counts.values()))


if __name__ == "__main__":
    main()
