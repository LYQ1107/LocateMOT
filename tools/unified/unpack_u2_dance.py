#!/usr/bin/env python3
"""Extract official DanceTrack train/val archives into one canonical view."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "outputs/unified/u2/acquisition"
VIEW = ROOT / "data/DanceTrack"


def extract(filename: str, download_pid: int) -> dict:
    record_path = STATUS / f"dancetrack_{filename}.json"
    while True:
        record = json.loads(record_path.read_text())
        if record["state"] == "COMPLETE":
            break
        if record["state"] == "FAILED":
            raise RuntimeError(f"Dance acquisition failed: {filename}")
        try:
            os.kill(download_pid, 0)
        except ProcessLookupError:
            raise RuntimeError(f"Dance acquisition ended before completion: {filename}")
        time.sleep(10)
    archive = Path(record["target"])
    extracted = archive.parent / "extracted"
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        zipped.extractall(extracted)
    part = extracted / filename.removesuffix(".zip")
    split = "val" if filename == "val.zip" else "train"
    sequences = sorted(path for path in part.iterdir() if path.is_dir() and path.name.startswith("dancetrack"))
    (VIEW / split).mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        target = VIEW / split / sequence.name
        if not target.exists():
            target.symlink_to(sequence, target_is_directory=True)
        elif target.resolve() != sequence.resolve():
            raise RuntimeError(f"Different release source already exists: {target}")
    result = {"archive": filename, "state": "COMPLETE", "source_sha256": record["sha256"],
              "crc_checked_during_extraction": True, "sequences": [path.name for path in sequences],
              "images": sum(member.filename.lower().endswith(".jpg") for member in members)}
    (STATUS / f"dancetrack_extracted_{filename}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-pid", type=int, required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = [future.result() for future in as_completed([
            executor.submit(extract, filename, args.download_pid) for filename in ("train1.zip", "train2.zip", "val.zip")])]
    (STATUS / "dance_extraction.json").write_text(json.dumps({"state": "COMPLETE", "archives": results}, indent=2) + "\n")
