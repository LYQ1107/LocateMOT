#!/usr/bin/env python3
"""Resume the public TempRMOT Refer-KITTI-V2 data folder download.

The folder contains only expression JSON and corrected labels_with_ids.  The
script deliberately does not request the author's checkpoints or KITTI image
archive; those are separate assets.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import gdown

FOLDER_ID = "1DnWcaf13tUnSJazwzNrnvc1bhx_abImM"
OUT = Path("data/refer_kitti_v2")
MANIFEST = Path("outputs/restore_20261001/v2_file_manifest.json")
VALIDATION = Path("outputs/restore_20261001/v2_download_validation.json")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = gdown.download_folder(
        id=FOLDER_ID,
        output=str(OUT),
        quiet=True,
        use_cookies=False,
        resume=True,
        timeout=60,
        retries=3,
    )
    expected = [str(getattr(item, "path", "")) for item in items]
    payload = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    if int(payload.get("count", len(expected))) != len(expected):
        raise AssertionError(f"V2 manifest count changed: {len(expected)}")
    missing = [path for path in expected if not (OUT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"V2 files missing after download: {missing[:5]} ({len(missing)} total)")
    counts = Counter(path.split("/", 1)[0] for path in expected)
    json_count = sum(1 for path in expected if path.startswith("expression/") and path.endswith(".json"))
    label_count = sum(1 for path in expected if path.startswith("labels_with_ids/") and path.endswith(".txt"))
    if json_count != counts["expression"] or label_count != counts["labels_with_ids"]:
        raise AssertionError("unexpected V2 file layout")
    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION.write_text(json.dumps({
        "status": "complete",
        "folder_id": FOLDER_ID,
        "file_count": len(expected),
        "counts": dict(counts),
        "expression_json_count": json_count,
        "labels_with_ids_count": label_count,
        "source": "https://drive.google.com/drive/folders/1eaxuRK-ewl0cpGshOxylSFZ5PPu3_WUT",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
