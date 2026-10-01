#!/usr/bin/env python3
"""Parallel direct downloader for the public TempRMOT V2 data files.

The author published this folder as individual Drive files.  Serial gdown
requests are impractically slow on the migrated server, so this uses a small
bounded worker pool against the public file IDs.  Existing files are kept and
skipped; each new file is written atomically through a ``.part`` path.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

MANIFEST = Path("outputs/restore_20261001/v2_file_manifest.json")
OUT = Path("data/refer_kitti_v2")
PROGRESS = Path("outputs/restore_20261001/v2_parallel_progress.json")
VALIDATION = Path("outputs/restore_20261001/v2_download_validation.json")
MAX_WORKERS = 32

_local = threading.local()


def session() -> requests.Session:
    value = getattr(_local, "session", None)
    if value is None:
        value = requests.Session()
        value.trust_env = False
        value.headers.update({"User-Agent": "LocateMOT-public-data-restore/2026-10-01"})
        _local.session = value
    return value


def download_one(item: dict[str, Any]) -> tuple[str, str]:
    rel = Path(str(item["path"]))
    target = OUT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return str(rel), "existing"
    part = target.with_name(target.name + ".part")
    url = f"https://drive.usercontent.google.com/download?id={item['id']}&export=download&confirm=t"
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with session().get(url, stream=True, timeout=(20, 120), allow_redirects=True) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type:
                    raise RuntimeError(f"unexpected HTML response for {rel}: {content_type}")
                with part.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        if chunk:
                            handle.write(chunk)
            if not part.is_file() or part.stat().st_size == 0:
                raise RuntimeError(f"empty response for {rel}")
            part.replace(target)
            return str(rel), "downloaded"
        except Exception as exc:  # retry transient Drive failures
            last_error = exc
            try:
                part.unlink()
            except FileNotFoundError:
                pass
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"failed {rel}: {last_error}")


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_items = list(payload["files"])
    # Drive exposes 20 duplicate expression paths in video 0007.  Their
    # independent IDs were fetched and SHA-audited separately; they are byte
    # identical, so one path is sufficient in the local dataset tree.
    items = list({str(item["path"]): item for item in raw_items}.values())
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"existing": 0, "downloaded": 0, "failed": 0}
    failures: list[str] = []
    lock = threading.Lock()
    total = len(items)
    started = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_one, item) for item in items]
        for index, future in enumerate(as_completed(futures), 1):
            try:
                _path, status = future.result()
                with lock:
                    counts[status] += 1
            except Exception as exc:
                with lock:
                    counts["failed"] += 1
                    failures.append(str(exc))
            if index % 250 == 0 or index == total:
                PROGRESS.parent.mkdir(parents=True, exist_ok=True)
                PROGRESS.write_text(json.dumps({
                    "status": "running" if index < total else "finished",
                    "completed_futures": index,
                    "total_futures": total,
                    "counts": counts,
                    "failures_sample": failures[:20],
                    "elapsed_seconds": time.time() - started,
                }, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"V2 parallel download failures: {len(failures)}; first={failures[0]}")
    missing = [str(item["path"]) for item in items if not (OUT / str(item["path"])).is_file()]
    if missing:
        raise FileNotFoundError(f"V2 missing files after parallel download: {missing[:5]}")
    expression_count = sum(1 for item in items if str(item["path"]).startswith("expression/") and str(item["path"]).endswith(".json"))
    labels_count = sum(1 for item in items if str(item["path"]).startswith("labels_with_ids/") and str(item["path"]).endswith(".txt"))
    VALIDATION.write_text(json.dumps({
        "status": "complete",
        "folder_id": payload["folder_id"],
        "file_count": total,
        "manifest_records": len(raw_items),
        "duplicate_path_records": len(raw_items) - total,
        "counts": counts,
        "expression_json_count": expression_count,
        "labels_with_ids_count": labels_count,
        "workers": MAX_WORKERS,
        "source": "https://drive.google.com/drive/folders/1eaxuRK-ewl0cpGshOxylSFZ5PPu3_WUT",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
