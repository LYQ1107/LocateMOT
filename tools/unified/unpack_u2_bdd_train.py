#!/usr/bin/env python3
"""Wait for verified original BDD train archives, extract and merge video views."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/unified/u2/acquisition"


def extract(entry: dict, download_pid: int) -> dict:
    archive = Path(entry["target"])
    state_path = OUT / f"{entry['dataset']}_{archive.name}.json"
    while True:
        try:
            acquired = json.loads(state_path.read_text()) if state_path.exists() else {}
        except json.JSONDecodeError:
            acquired = {}
        if acquired.get("state") == "COMPLETE":
            break
        if acquired.get("state") == "FAILED":
            raise RuntimeError(f"Original BDD archive download failed: {state_path}")
        try:
            os.kill(download_pid, 0)
        except ProcessLookupError:
            raise RuntimeError(f"BDD downloader ended before archive completion: {state_path}")
        time.sleep(10)
    if acquired["expected_md5"] != entry["expected_md5"]:
        raise RuntimeError("Original BDD archive differs from registered release")
    destination = archive.parent / "extracted" / archive.stem
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(destination)
    # The release archives store bdd100k/images/track/train/<video>.
    train = destination / "bdd100k/images/track/train"
    if not train.is_dir():
        raise RuntimeError(f"Unexpected original BDD archive layout: {archive}")
    view = ROOT / "data/bdd100k/images/track/train"
    view.mkdir(parents=True, exist_ok=True)
    videos, images = [], 0
    for directory in sorted(train.iterdir()):
        if not directory.is_dir():
            continue
        link = view / directory.name
        if link.exists() or link.is_symlink():
            if link.resolve() != directory.resolve():
                raise RuntimeError(f"BDD training video occurs in multiple source archives: {link}")
        else:
            link.symlink_to(directory, target_is_directory=True)
        videos.append(directory.name)
        images += sum(1 for _ in directory.glob("*.jpg"))
    payload = {"archive": archive.name, "state": "COMPLETE", "source": acquired["source"],
               "source_md5": entry["expected_md5"], "source_sha256": acquired["sha256"],
               "crc_checked_during_extraction": True, "videos": videos, "images": images}
    (OUT / f"bdd_extracted_{archive.name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"archive": archive.name, "state": "COMPLETE", "videos": len(videos), "images": images}), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-pid", type=int, required=True)
    args = parser.parse_args()
    entries = json.loads((ROOT / "outputs/unified/u2/protocol/bdd_acquisition_manifest.json").read_text())
    entries = [entry for entry in entries if "-train-" in entry["remote"]]
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = [future.result() for future in as_completed(
            executor.submit(extract, entry, args.download_pid) for entry in entries)]
    videos = [video for result in results for video in result["videos"]]
    if len(videos) != 1400 or len(set(videos)) != 1400:
        raise RuntimeError(f"Incomplete original BDD training video coverage: {len(videos)}")
    payload = {"state": "COMPLETE", "archives": sorted(results, key=lambda result: result["archive"]),
               "videos": len(videos), "images": sum(result["images"] for result in results)}
    (OUT / "bdd_train_extraction.json").write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
