"""Stage L1-B0: multi-dataset identity audit.

Computes per-dataset statistics directly from local annotation files and
writes outputs/l1_b/dataset_statistics.json plus a markdown audit.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
OUT = ROOT / "outputs" / "l1_b"
DOC = ROOT / "docs" / "l1_b_dataset_identity_audit.md"

DS = {
    "dancetrack": Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack"),
    "mot17": Path("/data1/LWR/vranlee/DATASETS/JDE/MOT17"),
    "mot20": Path(
        "/data1/LWR/vranlee/M4FTMoveOut4Doing/ByteTrack-mbt/datasets/"
        "mix_mot20_ch/mot20_train"),
    "bdd100k": Path(
        "/data1/LWR/vranlee/SERVER_ONLY/avis/BDD100K/bdd100k_labels/"
        "bdd100k/labels/100k"),
    "ytvos2019": Path(
        "/data3/testdata/vranlee/.MOTSynth.partial/YouTube-VOS-2019"),
    "mosev2": Path("/data3/testdata/vranlee/.MOTSynth.partial/MOSEv2"),
    "ctao": Path("/data3/testdata/vranlee/.MOTSynth.partial/C-TAO"),
    "bdd_track_labels": Path(
        "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/annotations/"
        "box_track_20"),
    "bdd_track_images": Path(
        "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/bdd100k/"
        "images/track"),
}


def _num(x):
    return float(x)


def _summarize(track_lens, objects_per_frame):
    tl = track_lens
    opf = objects_per_frame
    return {
        "unique_identities": len(tl),
        "mean_track_length": round(statistics.mean(tl), 2) if tl else 0.0,
        "median_track_length": round(statistics.median(tl), 2)
        if tl else 0.0,
        "short_tracks_le4": sum(1 for x in tl if x <= 4),
        "long_tracks_gt64": sum(1 for x in tl if x > 64),
        "mean_objects_per_frame": round(statistics.mean(opf), 2)
        if opf else 0.0,
        "median_objects_per_frame": round(statistics.median(opf), 2)
        if opf else 0.0,
        "max_objects_per_frame": max(opf) if opf else 0,
    }


def audit_dancetrack():
    ann_dir = DS["dancetrack"] / "annotations"
    out = {"family": "same-class dense", "splits": {}}
    for split in ("train", "val", "test"):
        d = json.loads((ann_dir / f"{split}.json").read_text())
        img_video = {im["id"]: im["video_id"] for im in d["images"]}
        ann_by_img = defaultdict(list)
        for a in d["annotations"]:
            ann_by_img[a["image_id"]].append(a)
        track_len = Counter()
        has_id = bool(d["annotations"]) and "track_id" in d["annotations"][0]
        for a in d["annotations"]:
            if has_id:
                track_len[(img_video[a["image_id"]], a["track_id"])] += 1
        frames = len(d["images"])
        opf = [len(ann_by_img[i["id"]]) for i in d["images"]]
        videos = len(d["videos"])
        cats = {c["name"] for c in d.get("categories", [])}
        # same-category multi-instance videos: any frame with >=2 same cat
        smi = 0
        for im in d["images"]:
            anns = ann_by_img[im["id"]]
            if len(anns) >= 2 and len({a["category_id"] for a in anns}) == 1:
                smi += 1
                break
        s = _summarize(list(track_len.values()), opf)
        s.update({
            "videos": videos, "frames": frames,
            "categories": sorted(cats),
            "n_categories": len(cats),
            "same_category_multi_instance_videos": int(smi),
            "box_available": True, "mask_available": False,
            "stable_identity": has_id, "category_available": True,
            "identity_available": has_id,
            "visibility_available": False, "ignore_regions": False,
            "annotation_exhaustive": True, "annotation_sparse": False,
            "birth_annotation": False, "track_continuity": True,
        })
        out["splits"][split] = s
    return out


def audit_mot(seq_root, name):
    out = {"family": "same-class dense", "splits": {}}
    if not seq_root.exists():
        return {"status": "MISSING", "root": str(seq_root)}
    splits = {"train": seq_root / "train", "test": seq_root / "test"}
    if not splits["train"].exists() and \
            any(p.is_dir() and p.name.startswith("MOT20")
                for p in seq_root.iterdir()):
        splits = {"train": seq_root}
    for split, d in splits.items():
        if not d.exists():
            continue
        seqs = sorted(x for x in d.iterdir() if x.is_dir())
        # MOT17 has 3 detector variants per sequence; dedupe by stem
        seen = set()
        tl_all = []
        opf_all = []
        n_videos = 0
        frames = 0
        for seq in seqs:
            stem = seq.name
            base = "-".join(stem.split("-")[:2])
            if base in seen:
                continue
            seen.add(base)
            gt = seq / "gt" / "gt.txt"
            if not gt.exists():
                continue
            n_videos += 1
            track_len = Counter()
            frame_obj = Counter()
            for line in gt.read_text().splitlines():
                p = line.split(",")
                fid, tid = int(p[0]), int(p[1])
                if int(p[6]) == 0:  # ignore
                    continue
                track_len[tid] += 1
                frame_obj[fid] += 1
            frames += max(frame_obj) if frame_obj else 0
            tl_all.extend(track_len.values())
            opf_all.extend(frame_obj.values())
        s = _summarize(tl_all, opf_all)
        s.update({
            "videos": n_videos, "frames": frames,
            "categories": ["person"], "n_categories": 1,
            "box_available": True, "mask_available": False,
            "stable_identity": True, "category_available": True,
            "identity_available": True,
            "visibility_available": True, "ignore_regions": True,
            "annotation_exhaustive": True, "annotation_sparse": False,
            "birth_annotation": False, "track_continuity": True,
        })
        out["splits"][split] = s
    return out


def audit_bdd():
    root = DS["bdd100k"]
    out = {"family": "road multi-class",
           "note": "box_track_20 tracking labels; local images: 200 train "
                   "+ 200 val videos",
           "splits": {}}
    for split in ("train", "val"):
        lab_dir = DS["bdd_track_labels"] / split
        img_dir = DS["bdd_track_images"] / split
        if not lab_dir.exists():
            continue
        tl = []
        opf = []
        frames = 0
        videos = 0
        cats = set()
        smi_videos = 0
        for p in sorted(lab_dir.glob("*.json")):
            vname = p.stem
            img_vdir = img_dir / vname
            if not img_vdir.exists():
                continue  # labels without local frames
            videos += 1
            track_len = Counter()
            per_frame = Counter()
            vjson = json.loads(p.read_text())
            for fr in vjson:
                labels = fr.get("labels", [])
                frame_cats = Counter()
                for lab in labels:
                    b = lab.get("box2d")
                    if b is None:
                        continue
                    tid = lab.get("id")
                    if tid is None:
                        continue
                    track_len[str(tid)] += 1
                    per_frame[fr.get("name")] += 1
                    frame_cats[lab.get("category", "")] += 1
                    cats.add(lab.get("category", ""))
                if len(labels) >= 2 and len(frame_cats) == 1:
                    smi_videos += 1
            tl.extend(track_len.values())
            opf.extend(per_frame.values())
            frames += len(vjson)
        s = _summarize(tl, opf)
        s.update({
            "videos": videos, "frames": frames,
            "categories": sorted(cats), "n_categories": len(cats),
            "same_category_multi_instance_videos": smi_videos,
            "box_available": True, "mask_available": False,
            "stable_identity": True, "category_available": True,
            "identity_available": True,
            "visibility_available": True, "ignore_regions": False,
            "annotation_exhaustive": False, "annotation_sparse": False,
            "birth_annotation": False, "track_continuity": True,
        })
        out["splits"][split] = s
    return out


def _scan_vos_masks(args):
    ann_dir, frames = args
    track_len = Counter()
    opf = []
    masked = 0
    for fr in frames:
        p = ann_dir / f"{Path(fr).stem}.png"
        if not p.exists():
            continue
        masked += 1
        arr = np.asarray(Image.open(p))
        counts = np.bincount(arr.ravel(), minlength=256)
        vals = np.nonzero(counts)[0]
        vals = vals[vals != 0]
        if vals.size:
            opf.append(int(vals.size))
        for v in vals.tolist():
            track_len[int(v)] += 1
    return list(track_len.values()), opf, masked


def audit_vos(name, meta_path, ann_root, family):
    out = {"family": family, "splits": {}}
    for split, mp in meta_path.items():
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        videos = meta["videos"]
        tl, opf, frames, cats = [], [], 0, set()
        smi_videos, n_videos = 0, 0
        first = next(iter(videos.values()), {})
        list_style = isinstance(first.get("objects"), list)
        if not list_style:
            for vid, info in videos.items():
                n_videos += 1
                objs = info.get("objects", {})
                track_len = Counter()
                cat_set = set()
                all_frames = set()
                obj_frames = Counter()
                for oid, o in objs.items():
                    track_len[oid] = len(o.get("frames", []))
                    cat_set.add(o.get("category", ""))
                    cats.add(o.get("category", ""))
                    all_frames.update(o.get("frames", []))
                for o in objs.values():
                    for f in o.get("frames", []):
                        obj_frames[f] += 1
                tl.extend(track_len.values())
                opf.extend(obj_frames.values())
                frames += len(all_frames)
                if len(objs) >= 2 and len(cat_set) == 1:
                    smi_videos += 1
                if n_videos % 500 == 0:
                    print(f"[l1b-audit] {name}/{split} videos={n_videos} "
                          f"frames={frames}", flush=True)
        else:
            # VOS-style meta with object-id lists (e.g. MOSE): read mask
            # PNGs in parallel to count per-object frame presence.
            from concurrent.futures import ProcessPoolExecutor
            ann_root_split = ann_root / split / "Annotations"
            items = [(ann_root_split / vid, info.get("frames", []))
                     for vid, info in videos.items()]
            n_videos = len(videos)
            masked_frames = 0
            with ProcessPoolExecutor(max_workers=16) as ex:
                for done, (tl_i, opf_i, masked_i) in enumerate(ex.map(
                        _scan_vos_masks, items), 1):
                    tl.extend(tl_i)
                    opf.extend(opf_i)
                    masked_frames += masked_i
                    if done % 400 == 0:
                        print(f"[l1b-audit] {name}/{split} scanned "
                              f"videos={done}/{n_videos}", flush=True)
            frames = sum(len(info.get("frames", []))
                         for info in videos.values())
            identity_available = masked_frames > 0.5 * frames
        s = _summarize(tl, opf)
        s.update({
            "videos": n_videos, "frames": frames,
            "categories": sorted(cats), "n_categories": len(cats),
            "same_category_multi_instance_videos": smi_videos,
            "box_available": False, "mask_available": True,
            "stable_identity": True,
            "category_available": len(cats) > 0,
            "identity_available": identity_available if list_style else True,
            "hidden_gt": (list_style and not identity_available),
            "visibility_available": False, "ignore_regions": False,
            "annotation_exhaustive": False, "annotation_sparse": True,
            "birth_annotation": True, "track_continuity": True,
        })
        out["splits"][split] = s
    return out


def audit_ctao():
    root = DS["ctao"]
    out = {"family": "multi-class long-tail (TAO-format)", "splits": {}}
    for name in ("ctao_base.json", "ctao_base_and_novel.json"):
        p = root / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ann_by_img = defaultdict(list)
        for a in d["annotations"]:
            ann_by_img[a["image_id"]].append(a)
        track_len = Counter()
        cats = set()
        for a in d["annotations"]:
            track_len[(a["video_id"], a["track_id"])] += 1
            cats.add(a["category_id"])
        opf = [len(ann_by_img[i["id"]]) for i in d["images"]]
        videos = len(d["videos"])
        frames = len(d["images"])
        sparse = sum(1 for v in d["videos"]
                     if v.get("not_exhaustive_category_ids"))
        s = _summarize(list(track_len.values()), opf)
        s.update({
            "videos": videos, "frames": frames,
            "categories": len(cats), "n_categories": len(cats),
            "same_category_multi_instance_videos": None,
            "box_available": True, "mask_available": True,
            "stable_identity": True, "category_available": True,
            "identity_available": True,
            "visibility_available": False,
            "ignore_regions": True,
            "annotation_exhaustive": False, "annotation_sparse": True,
            "birth_annotation": True, "track_continuity": True,
            "videos_with_not_exhaustive": sparse,
        })
        out["splits"][name] = s
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    stats["dancetrack"] = audit_dancetrack()
    stats["mot17"] = audit_mot(DS["mot17"], "mot17")
    stats["mot20"] = audit_mot(DS["mot20"], "mot20")
    stats["bdd100k"] = audit_bdd()
    stats["ytvos2019"] = audit_vos(
        "ytvos2019",
        {"train": DS["ytvos2019"] / "train" / "meta.json",
         "valid": DS["ytvos2019"] / "valid" / "meta.json"},
        DS["ytvos2019"], "deformable / diverse object (sparse)")
    stats["mosev2"] = audit_vos(
        "mosev2",
        {"train": DS["mosev2"] / "meta_train.json",
         "valid": DS["mosev2"] / "meta_valid.json"},
        DS["mosev2"], "deformable / occlusion (sparse)")
    stats["ctao"] = audit_ctao()
    stats["motsynth"] = {"status": "FORBIDDEN_BY_SPEC", "note": "not used"}
    stats["tao_official"] = {"status": "MISSING_PUBLIC",
                             "note": "official TAO annotations not found "
                                     "locally; C-TAO (TAO-format) used"}
    (OUT / "dataset_statistics.json").write_text(
        json.dumps(stats, indent=1))
    print("L1_B_DATASET_AUDIT_DONE")
    print(json.dumps({k: list(v.get("splits", {}).keys())
                      for k, v in stats.items()}, indent=1)[:2000])


if __name__ == "__main__":
    main()
