#!/usr/bin/env python3
"""Record the immutable G2 environment and the GPU3/GPU5 co-residency audit.

This is intentionally read-only.  It does not stop processes, change CUDA
visibility, or create a training server.  The resulting JSON is the resource
and version receipt used by every G2 run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


# ``g2/audit_environment.py`` lives one level below the usual
# ``tools/groundjev`` scripts, so the checkout root is parents[3].
ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "external/GroundAnything"
BASE = ROOT / "models/GroundAnything-DLM"
EXPECTED_OFFICIAL_COMMIT = "e054ab329cfe3f1b88ca8cf279bcab3e2f12934e"
DATA_ROOTS = {
    "locatemot_data": Path("/data2/user/LocateMOT/data"),
    "full_data_v2": Path("/data3/user/LocateMOT_FullDataV2"),
    "full_data_v2_spill": Path("/data2/user/LocateMOT_FullDataV2_spill"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def gpu_audit() -> dict[str, Any]:
    query = (
        "index,name,memory.total,memory.used,memory.free,utilization.gpu,"
        "temperature.gpu"
    )
    rows = command(
        "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"
    ).splitlines()
    apps = command(
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ).splitlines()
    cards = {}
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) < 7:
            continue
        index = int(fields[0])
        cards[index] = {
            "name": fields[1],
            "memory_total_mib": int(fields[2]),
            "memory_used_mib": int(fields[3]),
            "memory_free_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
            "temperature_c": int(fields[6]),
        }
    return {
        "requested_physical_gpus": [3, 5],
        "cards": {str(index): cards.get(index) for index in (3, 5)},
        "compute_apps_raw": apps,
        "safe_initial_reserve_policy_mib": 8 * 1024,
        "interpretation": "instantaneous audit; full forward/backward/optimizer peak test is a separate gate",
    }


def dataset_inventory() -> dict[str, Any]:
    expected = {
        "MOT17": Path("/data2/user/LocateMOT/data/MOT17"),
        "MOT20": Path("/data2/user/LocateMOT/data/MOT20"),
        "DanceTrack": Path("/data2/user/LocateMOT/data/DanceTrack"),
        "SportsMOT": Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT"),
        "AnimalTrack": Path("/data3/user/LocateMOT_FullDataV2/datasets/AnimalTrack"),
        "Refer-KITTI-V2": Path("/data2/user/LocateMOT/data/refer_kitti_v2"),
        "AerialMind": Path("/data3/user/LocateMOT_FullDataV2/datasets/AerialMind"),
        "RMOT26": Path("/data3/user/LocateMOT_FullDataV2/datasets/RMOT26"),
        "TAO-Amodal": Path("/data2/user/LocateMOT/data/TAO-Amodal"),
        "BURST": Path("/data3/user/LocateMOT_FullDataV2/datasets/BURST"),
    }
    inventory = {}
    for name, path in expected.items():
        files = []
        if path.is_dir():
            # Avoid retaining a full file list in the receipt; count only real
            # files and retain a few deterministic examples for diagnosis.
            for child in sorted(path.rglob("*")):
                if child.is_file() and ".DS_Store" not in child.name:
                    files.append(child)
        inventory[name] = {
            "path": str(path),
            "exists": path.exists(),
            "file_count": len(files),
            "example_files": [str(item) for item in files[:5]],
            "has_train_keyword": any("train" in item.name.lower() for item in files),
        }
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    official_commit = command("git", "-C", str(OFFICIAL), "rev-parse", "HEAD")
    repo_status = command("git", "-C", str(ROOT), "status", "--short")
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    receipt = {
        "status": "G2_ENVIRONMENT_AUDIT_COMPLETE",
        "created_localtime": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "root": str(ROOT),
        "official_repo": str(OFFICIAL),
        "official_repo_commit": official_commit,
        "official_commit_expected_in_task": EXPECTED_OFFICIAL_COMMIT,
        "official_commit_matches_task_reference": official_commit == EXPECTED_OFFICIAL_COMMIT,
        "base_checkpoint": str(BASE),
        "base_model_safetensors_sha256": sha256(BASE / "model.safetensors"),
        "base_tokenizer_sha256": sha256(BASE / "tokenizer.json"),
        "base_chat_template_sha256": sha256(BASE / "chat_template.jinja"),
        "packages": packages,
        "repo_status_at_audit": repo_status,
        "cuda_visible_devices_at_audit": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_audit": gpu_audit(),
        "datasets": dataset_inventory(),
        "protected_process_policy": "G2 may use only physical GPU3/GPU5 after peak preflight; do not stop other users or J1/MOT20/RMOT owners",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
