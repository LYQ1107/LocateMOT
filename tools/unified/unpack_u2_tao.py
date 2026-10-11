#!/usr/bin/env python3
"""Extract CRC-checked TAO archives on their registered volume and link a view."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/unified/u2/acquisition"
VIEW = ROOT / "data/TAO-Amodal"


def link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    destination.symlink_to(source, target_is_directory=source.is_dir())


def unpack_shard(shard: int, entries: list[dict], download_pid: int) -> dict:
    status_file = OUT / f"tao_shard_{shard}.json"
    output = OUT / f"tao_extraction_{shard}.json"
    result = json.loads(output.read_text()) if output.exists() else {"shard": shard, "state": "RUNNING", "archives": []}
    result["state"] = "RUNNING"
    completed = {entry["relative_path"] for entry in result["archives"]}
    pending = {entry["relative_path"]: entry for entry in entries if entry["relative_path"] not in completed}
    for entry in result["archives"]:
        link(Path(entry["extracted_path"]), VIEW / entry["relative_path"].removesuffix(".zip"))
    while pending:
        state = json.loads(status_file.read_text())
        ready = {entry["relative_path"]: entry for entry in state["completed"]}
        for relative in list(pending):
            if relative not in ready:
                continue
            entry = pending[relative]
            archive = Path(entry["local_root"]) / relative
            destination = archive.with_suffix("")
            result["current_archive"] = relative
            output.write_text(json.dumps(result, indent=2) + "\n")
            with zipfile.ZipFile(archive) as zipped:
                members = zipped.infolist()
                # ZipFile checks each member's CRC while extracting it. A
                # failure propagates; no partially extracted archive is ready.
                zipped.extractall(destination)
            image_count = sum(m.filename.lower().endswith((".jpg", ".jpeg", ".png")) for m in members)
            result["archives"].append({"relative_path": relative, "source_sha256": ready[relative]["sha256"],
                                       "extracted_path": str(destination), "members": len(members),
                                       "images": image_count, "uncompressed_bytes": sum(m.file_size for m in members),
                                       "crc_checked_during_extraction": True})
            link(destination, VIEW / relative.removesuffix(".zip"))
            del pending[relative]
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"archive": relative, "state": "EXTRACTED", "images": image_count}), flush=True)
        if not pending:
            break
        if state["state"] in ("FAILED", "COMPLETE"):
            raise RuntimeError(f"Missing completed archives from terminal shard {shard}: {list(pending)}")
        try:
            os.kill(download_pid, 0)
        except ProcessLookupError:
            raise RuntimeError(f"Download process ended with pending archives: {list(pending)}")
        time.sleep(10)
    result.update(state="COMPLETE", current_archive=None)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    manifest = json.loads((ROOT / "outputs/unified/u2/protocol/tao_shard_manifest.json").read_text())
    pid = json.loads((OUT / "tao_shard_0.json").read_text())["pid"]
    for relative in ("annotations/train.json", "annotations/validation.json", "README.md"):
        source = next(entry for entry in manifest["files"] if entry["relative_path"] == relative)
        link(Path(source["local_root"]) / relative, VIEW / relative)
    shards = {index: [] for index in range(len(manifest["roots"]))}
    for entry in manifest["files"]:
        if entry["relative_path"].startswith("frames/") and entry["relative_path"].endswith(".zip"):
            shards[entry["shard"]].append(entry)
    print(json.dumps({"state": "RUNNING", "pid": os.getpid(), "download_pid": pid, "view": str(VIEW)}), flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = [future.result() for future in as_completed([
            executor.submit(unpack_shard, shard, entries, pid) for shard, entries in shards.items()])]
    (OUT / "tao_extraction.json").write_text(json.dumps({"state": "COMPLETE", "shards": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
