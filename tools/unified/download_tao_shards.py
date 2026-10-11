#!/usr/bin/env python3
"""Acquire the registered TAO shards with one bounded worker per volume."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def priority(entry: dict) -> tuple[int, str]:
    relative = entry["relative_path"]
    if relative in ("annotations/train.json", "annotations/validation.json", "README.md"):
        return 0, relative
    if relative.startswith(("frames/train/", "frames/val/")):
        return 1, relative
    return 2, relative


def download_shard(shard: int, files: list[dict], args: argparse.Namespace) -> dict:
    log = args.log_dir / f"tao_shard_{shard}.log"
    status_path = args.status_dir / f"tao_shard_{shard}.json"
    status = {"shard": shard, "pid": os.getpid(), "state": "RUNNING", "completed": [], "expected_files": len(files)}
    env = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(name, None)

    def save() -> None:
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(status_path)

    save()
    with log.open("a", encoding="utf-8") as output:
        for entry in sorted(files, key=priority):
            target = Path(entry["local_root"]) / entry["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            status["current_file"] = entry["relative_path"]
            save()
            output.write(f"BEGIN {entry['relative_path']} expected_bytes={entry['size']}\n")
            output.flush()
            command = [str(args.client), "d", entry["remote_path"], f"--saveto={target.parent}",
                       f"--mode={args.mode}", f"--retry={args.retries}", "-p", str(args.threads), "-l", "1"]
            result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
            actual_size = target.stat().st_size if target.is_file() else None
            if result.returncode or actual_size != entry["size"]:
                status.update(state="FAILED", returncode=result.returncode,
                              error="CLIENT_FAILED" if result.returncode else "INCOMPLETE_OUTPUT",
                              actual_bytes=actual_size, expected_bytes=entry["size"])
                save()
                return status
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            status["completed"].append({"relative_path": entry["relative_path"],
                                        "local_path": str(target), "bytes": actual_size,
                                        "sha256": digest.hexdigest()})
            output.write(f"COMPLETE {entry['relative_path']} bytes={actual_size}\n")
            output.flush()
            save()
    status.update(state="COMPLETE", returncode=0, current_file=None)
    save()
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/unified/u2/protocol/tao_shard_manifest.json")
    parser.add_argument("--client", type=Path, default=ROOT / ".tools/baidupcs/BaiduPCS-Go-v3.9.6-linux-amd64/BaiduPCS-Go")
    parser.add_argument("--log-dir", type=Path, default=ROOT / ".tools/baidupcs/logs/tao_20261011")
    parser.add_argument("--status-dir", type=Path, default=ROOT / "outputs/unified/u2/acquisition")
    parser.add_argument("--mode", choices=("locate", "pcs", "stream"), default="locate")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.status_dir.mkdir(parents=True, exist_ok=True)
    shards = {index: [] for index in range(len(manifest["roots"]))}
    for entry in manifest["files"]:
        shards[entry["shard"]].append(entry)
    print(json.dumps({"state": "STARTING", "pid": os.getpid(), "roots": manifest["roots"],
                      "files": len(manifest["files"]), "workers": len(shards)}, ensure_ascii=False), flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = [executor.submit(download_shard, shard, files, args) for shard, files in shards.items()]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"shard": result["shard"], "state": result["state"],
                              "completed_files": len(result["completed"]), "error": result.get("error")}), flush=True)
    summary = {"state": "COMPLETE" if all(result["state"] == "COMPLETE" for result in results) else "FAILED",
               "manifest": str(args.manifest), "shards": results}
    (args.status_dir / "tao_download.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return 0 if summary["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
