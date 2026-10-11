#!/usr/bin/env python3
"""Prepare the fixed, bounded U2-P train-only replay through native ODVG."""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from locatemot.unified.data.bdd_tracking import frame_objects
from locatemot.unified.data.legal_scope import assert_legal_path, assert_legal_video

SEED = 20261012
ALLOCATIONS = {"u1_generic": 2000, "u1_v1": 3000, "u1_v2": 5000,
               "bdd100k": 2000, "dancetrack": 2000, "mot17": 2000, "mot20": 2000, "tao_base": 2000}
DIRECTORY = ROOT / "outputs/unified/u2/perception_adaptation"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def clean_name(name: str) -> str:
    # The installed MMDetection text transform uses this lexical normalization.
    return re.sub(r"\(.*\)", "", name).replace("_", " ").replace("  ", " ").strip()


def grounding(groups: dict[str, list[list[float]]], empty_caption: str) -> dict:
    caption, regions = "", []
    for name, boxes in sorted(groups.items()):
        phrase = clean_name(name)
        begin = len(caption)
        caption += phrase
        regions.append({"bbox": boxes, "phrase": phrase, "tokens_positive": [[begin, len(caption)]]})
        caption += " . "
    return {"caption": caption.strip() if regions else clean_name(empty_caption) + " .", "regions": regions}


def sample_video_pool(pool: list[dict], count: int, seed: int) -> list[dict]:
    videos = defaultdict(list)
    for row in pool:
        videos[str(row["video"])].append(row)
    rng = random.Random(seed)
    order = sorted(videos)
    rng.shuffle(order)
    for video in order:
        rng.shuffle(videos[video])
    return [videos[order[index % len(order)]][(index // len(order)) % len(videos[order[index % len(order)]])]
            for index in range(count)]


def ordinary_pool(dataset: str) -> tuple[list[dict], dict]:
    name = f"{dataset}_half.json" if dataset in ("mot17", "mot20") else f"{dataset}.json"
    path = ROOT / "outputs/unified/u2/protocol" / name
    protocol = json.loads(path.read_text())
    pool = [{"video": sequence["sequence"], "frame": frame,
             "filename": str(Path(sequence["image_dir"]) / frame),
             "width": sequence["width"], "height": sequence["height"], "gt_path": sequence["gt_path"]}
            for sequence in protocol["sequences"] for frame in sequence["train_frames"]]
    return pool, {"path": str(path), "sha256": digest(path), "frames": len(pool),
                  "videos": len({row["video"] for row in pool}), "split": "registered train frames only"}


def convert_ordinary(selected: list[dict], dataset: str) -> list[dict]:
    rows, annotations = [], {}
    for source in selected:
        path = source["gt_path"]
        if path not in annotations:
            by_frame = defaultdict(list)
            if dataset == "bdd100k":
                for frame in json.loads(Path(path).read_text()):
                    objects, _ = frame_objects(frame)
                    by_frame[frame["name"]] = objects
            else:
                for line in Path(path).read_text().splitlines():
                    values = [float(value) for value in line.split(",")]
                    if values[6] == 1 and values[7] == 1:
                        x, y, width, height = values[2:6]
                        by_frame[int(values[0])].append({"category": "person", "bbox": [x, y, x + width, y + height]})
            annotations[path] = by_frame
        key = source["frame"] if dataset == "bdd100k" else int(Path(source["frame"]).stem)
        groups = defaultdict(list)
        for obj in annotations[path][key]:
            groups[obj["category"]].append(obj["bbox"])
        rows.append({"filename": source["filename"], "width": source["width"], "height": source["height"],
                     "grounding": grounding(groups, "pedestrian" if dataset == "bdd100k" else "person"),
                     "stream": dataset, "video": source["video"], "frame": source["frame"]})
    return rows


def tao_rows(count: int, seed: int) -> tuple[list[dict], dict]:
    path = ROOT / "data/TAO-Amodal/annotations/train.json"
    data = json.loads(path.read_text())
    vocabulary_path = ROOT / "outputs/unified/u2/protocol/lvis_v1_vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text())
    by_synset = {category["synset"]: category for category in vocabulary["categories"]}
    categories = {category["id"]: category for category in data["categories"]}
    base = {category_id: by_synset[category["synset"]]["name"] for category_id, category in categories.items()
            if category["synset"] in by_synset and by_synset[category["synset"]]["frequency"] in ("f", "c")}
    pool = [{**image, "video": str(image["video_id"])} for image in data["images"]]
    if not all(image["file_name"].startswith("train/") for image in pool):
        raise RuntimeError("TAO perception adaptation attempted to use a non-training image")
    selected = sample_video_pool(pool, count, seed)
    by_image, excluded = defaultdict(list), Counter()
    for annotation in data["annotations"]:
        category_id = annotation["category_id"]
        if category_id in base:
            by_image[annotation["image_id"]].append(annotation)
        else:
            category = categories[category_id]
            excluded["unmapped" if category["synset"] not in by_synset else "novel"] += 1
    empty_class = next(category["name"] for category in vocabulary["categories"] if category["frequency"] in ("f", "c"))
    rows = []
    for image in selected:
        groups = defaultdict(list)
        for annotation in by_image[image["id"]]:
            x, y, width, height = annotation["bbox"]
            groups[base[annotation["category_id"]]].append([x, y, x + width, y + height])
        rows.append({"filename": str(ROOT / "data/TAO-Amodal/frames" / image["file_name"]),
                     "width": image["width"], "height": image["height"],
                     "grounding": grounding(groups, empty_class), "stream": "tao_base",
                     "video": image["video"], "frame": image["file_name"]})
    return rows, {"annotation_path": str(path), "annotation_sha256": digest(path),
                  "vocabulary_sha256": digest(vocabulary_path), "source_frames": len(pool),
                  "source_videos": len({row["video"] for row in pool}), "source_gt": len(data["annotations"]),
                  "class_and_identity_supervision_excluded_gt": dict(excluded),
                  "novel_semantic_supervised_gt": 0, "novel_identity_supervised_gt": 0,
                  "policy": "mapped LVIS-v1 f/c positives only; retain every selected frame including empty base positives"}


def wait_data(pid: int) -> None:
    status = ROOT / "outputs/unified/u2/acquisition/bdd_train_extraction.json"
    while not status.exists():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise RuntimeError("BDD extraction ended before complete original training coverage")
        time.sleep(10)
    if json.loads(status.read_text())["state"] != "COMPLETE":
        raise RuntimeError("BDD training extraction is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-bdd-pid", type=int)
    args = parser.parse_args()
    if args.wait_bdd_pid:
        wait_data(args.wait_bdd_pid)
    subprocess.run([sys.executable, str(ROOT / "tools/unified/audit_u2_data.py")], check=True, cwd=ROOT)
    inventory = json.loads((ROOT / "outputs/unified/u2/protocol/data_manifest.json").read_text())
    if any(item["status"] != "AVAILABLE" for item in inventory["datasets"]):
        raise RuntimeError("Prepare U2-P replay only after complete required source image coverage")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    streams, sources = {}, {}
    generic_map = json.loads((ROOT / "outputs/unified/u1r2/generic_label_map.json").read_text())
    for index, (stream, name) in enumerate([("u1_generic", "expression_mix_generic.jsonl"),
                                           ("u1_v1", "expression_mix_v1.jsonl"), ("u1_v2", "expression_mix_v2.jsonl")]):
        path = ROOT / "outputs/unified/u1r2" / name
        pool = [json.loads(line) for line in path.read_text().splitlines()]
        selected = sample_video_pool(pool, ALLOCATIONS[stream], SEED + index)
        rows = []
        for row in selected:
            assert_legal_video(row["video"])
            assert_legal_path(row["filename"])
            if stream == "u1_generic":
                groups = defaultdict(list)
                for instance in row["detection"]["instances"]:
                    groups[generic_map[str(instance["label"])]].append(instance["bbox"])
                payload = grounding(groups, generic_map["0"])
            else:
                payload = row["grounding"]
            rows.append({"filename": row["filename"], "width": row["width"], "height": row["height"],
                         "grounding": payload, "stream": stream, "video": row["video"]})
        streams[stream] = rows
        sources[stream] = {"path": str(path), "sha256": digest(path), "pool_rows": len(pool),
                           "selected_videos": len({row["video"] for row in rows}), "scope": "unchanged U1R2 legal-fit replay"}
    for index, dataset in enumerate(("bdd100k", "dancetrack", "mot17", "mot20"), start=3):
        pool, sources[dataset] = ordinary_pool(dataset)
        selected = sample_video_pool(pool, ALLOCATIONS[dataset], SEED + index)
        streams[dataset] = convert_ordinary(selected, dataset)
    streams["tao_base"], sources["tao_base"] = tao_rows(ALLOCATIONS["tao_base"], SEED + 7)
    rows = [row for stream in ALLOCATIONS for row in streams[stream]]
    random.Random(SEED).shuffle(rows)
    if len(rows) != 20000 or any(not Path(row["filename"]).is_file() for row in rows):
        raise RuntimeError("Replay has incomplete selected training-image coverage")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "weights/bert-base-uncased"), local_files_only=True)
    lengths = tokenizer([row["grounding"]["caption"] for row in rows], add_special_tokens=True, truncation=False)["input_ids"]
    max_tokens = max(map(len, lengths))
    if max_tokens > 256:
        raise RuntimeError(f"Registered replay exceeds the foundation token contract: {max_tokens}")
    output = DIRECTORY / "replay_train.jsonl"
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {"format": "locatemot-u2-bounded-perception-replay-v1", "seed": SEED,
               "rows_per_epoch": len(rows), "maximum_epochs": 2, "allocations": ALLOCATIONS,
               "sampling": "fixed deterministic round-robin shuffled video/frame pools; cycle small replay pools",
               "sources": sources, "replay_path": str(output), "replay_sha256": digest(output),
               "max_prompt_tokens": max_tokens, "native_dataset_mode": "ODVG visual grounding",
               "stream_counts": {name: {"rows": len(values), "videos": len({row['video'] for row in values}),
                                        "empty_positive_rows": sum(not row['grounding']['regions'] for row in values),
                                        "positive_boxes": sum(len(region['bbox']) for row in values for region in row['grounding']['regions'])}
                                 for name, values in streams.items()},
               "official_test_labels_read": False, "training_started": False}
    (DIRECTORY / "replay_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
