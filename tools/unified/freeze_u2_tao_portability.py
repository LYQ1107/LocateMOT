#!/usr/bin/env python3
"""Freeze full modal TAO validation and original-frame C-TAO training audit."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "outputs/unified/u2/protocol"
FOUNDATION_SHA = "4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_images(data: dict, selector: str) -> list[dict]:
    images = data["images"] if selector == "all_validation" else [image for image in data["images"] if image["is_ori"]]
    return sorted(images, key=lambda image: (image["file_name"], image["id"]))


def group_for(category: dict, vocabulary: dict[str, dict]) -> str:
    # Preserve every original TAO GT unit. Unmapped v0.5 synsets retain their
    # source frequency for this localization audit and are reported separately.
    frequency = vocabulary.get(category["synset"], category)["frequency"]
    return "base" if frequency in ("f", "c") else "novel"


def freeze(dataset: str, annotation: Path, selector: str, vocabulary: dict) -> dict:
    data = json.loads(annotation.read_text())
    images = selected_images(data, selector)
    assert len({image["id"] for image in images}) == len(images)
    assert len({image["file_name"] for image in images}) == len(images)
    ids = {image["id"] for image in images}
    categories = {category["id"]: category for category in data["categories"]}
    by_synset = {category["synset"]: category for category in vocabulary["categories"]}
    counts, unmapped = Counter(), Counter()
    for annotation_row in data["annotations"]:
        if annotation_row["image_id"] not in ids:
            continue
        category = categories[annotation_row["category_id"]]
        counts[group_for(category, by_synset)] += 1
        if category["synset"] not in by_synset:
            unmapped[category["name"]] += 1
    if dataset == "ctao":
        assert counts["novel"] == 0
        assert all(image["file_name"].startswith("train/") for image in images)
    else:
        assert all(image["file_name"].startswith("val/") for image in images)
    keys = [f"{dataset}/{image['id']}" for image in images]
    result = {
        "format": "locatemot-u2-tao-portability-protocol-v1", "dataset": dataset,
        "foundation_sha256": FOUNDATION_SHA, "annotation_path": str(annotation.resolve()),
        "annotation_sha256": sha256(annotation), "image_root": str(ROOT / "data/TAO-Amodal/frames"),
        "image_selector": selector, "image_order": "file_name, id ascending",
        "measurement_images": len(images), "frame_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "coverage_keys_sha256": hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(),
        "videos": len({image["video_id"] for image in images}),
        "gt_units_expected": {"base": counts["base"], "novel": counts["novel"]},
        "unmapped_lvis_v1_gt": dict(sorted(unmapped.items())),
        "unmapped_policy": "retain every original GT unit; group using original frequency if no LVIS-v1 synset match",
        "vocabulary_path": str(DIRECTORY / "lvis_v1_vocabulary.json"),
        "vocabulary_sha256": sha256(DIRECTORY / "lvis_v1_vocabulary.json"), "vocabulary_size": 1203,
        "chunked_size": 32, "max_per_img_per_chunk": 900, "merge_iou": 0.8,
        "candidate_top_k": 300, "iou_threshold": 0.5,
        "gt_policy": "all source modal xywh boxes on selected images; no clipping, visibility cut or category exclusion",
        "source_archive_required_state": "CRC-checked extraction complete",
        "interpolated_ctao_training_frames_retained_for_identity_training": True,
        "official_test_labels_read": False,
    }
    output = DIRECTORY / f"{dataset}_portability.json"
    contents = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != contents:
        raise RuntimeError(f"A frozen portability protocol differs: {output}")
    output.write_text(contents)
    return {"dataset": dataset, "images": len(images), "gt_counts": result["gt_units_expected"],
            "protocol_sha256": sha256(output), "unmapped_gt": sum(unmapped.values())}


if __name__ == "__main__":
    vocabulary = json.loads((DIRECTORY / "lvis_v1_vocabulary.json").read_text())
    print(json.dumps([
        freeze("tao", ROOT / "data/TAO-Amodal/annotations/validation.json", "all_validation", vocabulary),
        freeze("ctao", ROOT / "data/C-TAO/ctao_base.json", "all_original_training_frames", vocabulary),
    ], indent=2))
