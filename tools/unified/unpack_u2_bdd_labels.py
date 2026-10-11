#!/usr/bin/env python3
"""Extract raw BDD tracking annotations and validation images from a pinned mirror."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/unified/u2/acquisition"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-pid", type=int, required=True)
    args = parser.parse_args()
    manifest = ROOT / "outputs/unified/u2/protocol/bdd_labels_acquisition_manifest.json"
    entry, = json.loads(manifest.read_text())
    archive = Path(entry["target"])
    status = OUT / f"{entry['dataset']}_{archive.name}.json"
    while True:
        try:
            acquired = json.loads(status.read_text()) if status.exists() else {}
        except json.JSONDecodeError:
            acquired = {}
        if acquired.get("state") == "COMPLETE":
            break
        if acquired.get("state") == "FAILED":
            raise RuntimeError(f"BDD annotation download failed: {status}")
        try:
            os.kill(args.download_pid, 0)
        except ProcessLookupError:
            raise RuntimeError(f"BDD annotation downloader ended without a complete archive: {status}")
        time.sleep(10)
    if acquired["sha256"] != entry["expected_sha256"]:
        raise RuntimeError("Mirror archive differs from the pinned Hugging Face object")
    destination = archive.parent / "original_tracking_mirror"
    prefixes = ("bdd100k/labels/box_track_20/train/", "bdd100k/labels/box_track_20/val/",
                "bdd100k/images/track/val/")
    with zipfile.ZipFile(archive) as zipped:
        members = [item for item in zipped.infolist() if item.filename.startswith(prefixes)]
        for item in members:
            zipped.extract(item, destination)
    extracted = destination / "bdd100k"
    view = ROOT / "data/bdd100k"
    for relative in ("labels/box_track_20/train", "labels/box_track_20/val", "images/track/val"):
        link, target = view / relative, extracted / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"BDD project view points to another source: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    scopes = {}
    for split, expected in (("train", 1400), ("val", 200)):
        files = sorted((view / "labels/box_track_20" / split).glob("*.json"))
        if len(files) != expected:
            raise RuntimeError(f"Incomplete raw BDD {split} annotation videos: {len(files)} != {expected}")
        categories, frames, crowd = Counter(), 0, Counter()
        for path in files:
            video = json.loads(path.read_text())
            if not isinstance(video, list):
                raise RuntimeError(f"Expected original per-video Scalabel frames: {path}")
            frames += len(video)
            for frame in video:
                if split == "val" and not (view / "images/track/val" / path.stem / frame["name"]).is_file():
                    raise RuntimeError(f"Original validation frame missing: {path.stem}/{frame['name']}")
                for label in frame.get("labels") or []:
                    categories[label["category"]] += 1
                    if label.get("attributes", {}).get("crowd", False):
                        crowd[label["category"]] += 1
        scopes[split] = {"videos": len(files), "annotation_frames": frames,
                         "category_counts": dict(sorted(categories.items())), "crowd_counts": dict(sorted(crowd.items()))}
    payload = {"state": "COMPLETE", "source": acquired["source"], "revision": entry["revision"],
               "source_archive_sha256": acquired["sha256"], "crc_checked_during_extraction": True,
               "selected_member_prefixes": prefixes, "members_extracted": len(members),
               "raw_annotation_scopes": scopes, "official_test_labels_read": False,
               "train_images_complete": False}
    (OUT / "bdd_labels_extraction.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
