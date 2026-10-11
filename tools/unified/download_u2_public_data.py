#!/usr/bin/env python3
"""Download official DanceTrack train/val and base-only C-TAO annotations."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/unified/u2/acquisition"
FILES = [
    {"dataset": "dancetrack", "repo": "datasets/noahcao/dancetrack",
     "revision": "a0ba42ac690c41e9850a20e76a0b9450a6fb6a47", "remote": name,
     "target": str(ROOT / "data/downloads/DanceTrack" / name), "bytes": size,
     "license": "CC-BY-4.0", "split": "validation" if name == "val.zip" else "train"}
    for name, size in (("train1.zip", 3606300312), ("train2.zip", 3299320948), ("val.zip", 4209785614))
] + [{"dataset": "ctao", "repo": "clarkqian/COVTrack",
      "revision": "a6c1a4673a45425d9375f75d1427df7d22590837",
      "remote": "saved_models/ctao_dataset/ctao_base.json",
      "target": str(ROOT / "data/C-TAO/ctao_base.json"), "bytes": 546599277,
      "license": "C-TAO release; original TAO frame terms apply", "split": "base-only continuous training"}]


def acquire(entry: dict) -> dict:
    target = Path(entry["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    url = entry.get("source") or f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{entry['remote']}"
    result = {**entry, "source": url, "state": "RUNNING"}
    status_path = OUT / f"{entry['dataset']}_{target.name}.json"

    def save():
        status_path.write_text(json.dumps(result, indent=2) + "\n")

    save()
    env = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(name, None)
    log_path = OUT / f"{entry['dataset']}_{target.name}.log"
    if (not target.is_file() or target.stat().st_size != entry["bytes"]
            or Path(str(target) + ".aria2").exists()):
        partial = target.with_name(target.name + ".part")
        with log_path.open("a") as log:
            # OpenSSL curl succeeds on this host where aria2's GnuTLS
            # handshake terminates before receiving any bytes.
            command = ["curl", "--noproxy", "*", "--location", "--fail", "--continue-at", "-",
                       "--retry", "3", "--retry-all-errors", "--retry-delay", "2",
                       "--connect-timeout", "20", "--speed-time", "60", "--speed-limit", "1024",
                       "--silent", "--show-error", "--output", str(partial),
                       url + "?download=true&locatemot=" + str(int(time.time()))]
            if entry.get("transport") == "aria2":
                command = ["aria2c", "--continue=true", "--auto-file-renaming=false", "--file-allocation=none",
                           "--all-proxy=", "--max-connection-per-server=8", "--split=8", "--min-split-size=4M",
                           "--max-tries=3", "--retry-wait=5", "--connect-timeout=20", "--timeout=60",
                           "--console-log-level=warn", "--summary-interval=0", "--download-result=hide",
                           "--check-integrity=true", "--checksum=md5=" + entry["expected_md5"],
                           "--dir=" + str(target.parent), "--out=" + target.name, url]
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            result.update(state="FAILED", returncode=completed.returncode, log=str(log_path))
            save()
            return result
        if entry.get("transport") != "aria2":
            partial.replace(target)
    if target.stat().st_size != entry["bytes"]:
        result.update(state="FAILED", error="INCOMPLETE_OUTPUT", actual_bytes=target.stat().st_size)
        save()
        return result
    digest = hashlib.sha256()
    md5 = hashlib.md5()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            md5.update(block)
    if entry.get("expected_md5") and md5.hexdigest() != entry["expected_md5"]:
        result.update(state="FAILED", error="SOURCE_MD5_MISMATCH", actual_md5=md5.hexdigest())
        save()
        return result
    if entry.get("expected_sha256") and digest.hexdigest() != entry["expected_sha256"]:
        result.update(state="FAILED", error="SOURCE_SHA256_MISMATCH", actual_sha256=digest.hexdigest())
        save()
        return result
    result.update(state="COMPLETE", sha256=digest.hexdigest(), proxy_used=False)
    save()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summary", default="public_downloads.json")
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text()) if args.manifest else FILES
    OUT.mkdir(parents=True, exist_ok=True)
    if args.manifest is None:
        (ROOT / "outputs/unified/u2/protocol/public_acquisition_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(json.dumps({"state": "RUNNING", "pid": os.getpid(), "workers": args.workers, "files": len(entries)}), flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(acquire, entry) for entry in entries]):
            result = future.result()
            results.append(result)
            print(json.dumps({"dataset": result["dataset"], "file": result["remote"], "state": result["state"],
                              "returncode": result.get("returncode")}), flush=True)
    summary = {"state": "COMPLETE" if all(r["state"] == "COMPLETE" for r in results) else "FAILED", "files": results}
    (OUT / args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
