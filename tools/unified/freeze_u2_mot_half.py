#!/usr/bin/env python3
"""Register the ByteTrack MOT17/MOT20 half split before U2 measurements."""
from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "outputs/unified/u2/protocol"
CONVERTER_SHA = {
    "mot17": "e10a109b51d0dd1278d59819060777ed8322979c49fd46c48b484ad4ffdf66c6",
    "mot20": "6e138e2cb7a278b471d5b50229ec7116b146ec67dd0330d8d63f2de3dd51f8b6",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(dataset: str) -> dict:
    root = ROOT / "data" / dataset.upper() / dataset.upper()
    converter = PROTOCOL / "sources" / f"convert_{dataset}_to_coco.py"
    if not converter.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url = f"https://raw.githubusercontent.com/ifzhang/ByteTrack/main/tools/{converter.name}"
        converter.parent.mkdir(parents=True, exist_ok=True)
        converter.write_bytes(opener.open(url, timeout=30).read())
    if sha256(converter) != CONVERTER_SHA[dataset]:
        raise RuntimeError(f"The reference converter differs from the registered source: {converter}")
    sequences = []
    for directory in sorted((root / "train").iterdir()):
        if not directory.is_dir() or (dataset == "mot17" and not directory.name.endswith("-FRCNN")):
            continue
        info = configparser.ConfigParser()
        info.read(directory / "seqinfo.ini")
        sequence = info["Sequence"]
        length = int(sequence["seqLength"])
        filenames = [f"{frame:06d}{sequence['imExt']}" for frame in range(1, length + 1)]
        images = directory / sequence["imDir"]
        actual = sorted(path.name for path in images.glob(f"*{sequence['imExt']}"))
        if actual != filenames:
            raise RuntimeError(f"Incomplete sequence: {directory.name}")
        sequences.append({
            "sequence": directory.name, "image_dir": str(images),
            "gt_path": str(directory / "gt/gt.txt"),
            "gt_sha256": sha256(directory / "gt/gt.txt"),
            "seqinfo_sha256": sha256(directory / "seqinfo.ini"),
            "width": int(sequence["imWidth"]), "height": int(sequence["imHeight"]),
            "fps": int(sequence["frameRate"]), "length": length,
            # The reference converter's inclusive zero-based train range is
            # [0, N//2], and its validation range is [N//2+1, N-1].
            "train_frames": filenames[:length // 2 + 1],
            "validation_frames": filenames[length // 2 + 1:],
        })
    result = {
        "format": "locatemot-u2-mot-half-v1", "dataset": dataset,
        "root": str(root), "version": dataset.upper(),
        "source": f"https://motchallenge.net/data/{dataset.upper()}/",
        "license": "MOTChallenge dataset terms",
        "split_reference": {
            "url": f"https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_{dataset}_to_coco.py",
            "local_source": str(converter), "sha256": sha256(converter),
            "one_based_train": "1..floor(N/2)+1 inclusive",
            "one_based_validation": "floor(N/2)+2..N inclusive",
        },
        "physical_video_policy": "FRCNN copy only" if dataset == "mot17" else "each release sequence once",
        "vocabulary": "person .", "custom_entities": True,
        "max_per_img": 900, "measured_top_k": [150, 300], "candidate_top_k": 150,
        "iou_threshold": 0.5,
        "gt_policy": {
            "positive": "mark == 1 and category == 1 (pedestrian)",
            "visibility_cutoff": None, "boxes": "source xywh converted to xyxy without clipping or coordinate shift",
            "recall_unit": "one positive physical GT object per frame",
        },
        "sequences": sequences,
        "train_images": sum(len(s["train_frames"]) for s in sequences),
        "validation_images": sum(len(s["validation_frames"]) for s in sequences),
        "official_test_labels_read": False,
    }
    path = PROTOCOL / f"{dataset}_half.json"
    contents = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != contents:
        raise RuntimeError(f"An existing frozen protocol differs: {path}")
    path.write_text(contents)
    return {"dataset": dataset, "protocol": str(path), "sha256": sha256(path),
            "train_images": result["train_images"], "validation_images": result["validation_images"]}


if __name__ == "__main__":
    print(json.dumps([freeze(dataset) for dataset in ("mot17", "mot20")], indent=2))
