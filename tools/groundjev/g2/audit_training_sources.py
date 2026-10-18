#!/usr/bin/env python3
"""Hash and summarize the official GroundAnything DLM training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "external/GroundAnything"
FILES = [
    "docs/TRAINING.md",
    "docs/DATA_PREPARATION.md",
    "configs/training_stages.yaml",
    "configs/train/general.yaml",
    "configs/train/specialist.yaml",
    "configs/train/sft1.yaml",
    "configs/train/sft2.yaml",
    "train/dlm/train_dlm.py",
    "models/dlm/vlm.py",
    "models/dlm/hybrid.py",
    "train/dlm/data.py",
    "train/data/row_contract.py",
    "train/release_parameters.py",
    "grounding_anything/protocol.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", "-C", str(OFFICIAL), *args), text=True).strip()


def yaml_summary(path: Path) -> dict[str, Any]:
    if path.name == "training_stages.yaml":
        value = yaml.safe_load(path.read_text())
        return {"stages": value.get("stages", []), "rl": value.get("rl", {})}
    value = yaml.safe_load(path.read_text())
    training = value.get("training", {})
    runtime = value.get("runtime", {})
    return {
        "mode": value.get("mode"),
        "experiment": value.get("meta", {}).get("experiment"),
        "sampled_rows": value.get("data", {}).get("sampled_rows"),
        "epochs": training.get("num_train_epochs"),
        "learning_rate": training.get("learning_rate"),
        "projector_learning_rate": training.get("projector_learning_rate"),
        "vision_learning_rate": training.get("vision_learning_rate"),
        "scheduler": training.get("lr_scheduler_type"),
        "min_lr_rate": training.get("lr_scheduler_kwargs", {}).get("min_lr_rate"),
        "warmup_ratio": training.get("warmup_ratio"),
        "block_size": training.get("block_size"),
        "mdm_loss_weight": training.get("mdm_loss_weight"),
        "causal_loss_weight": training.get("causal_loss_weight"),
        "freeze_vision_encoder": training.get("freeze_vision_encoder"),
        "freeze_projector": training.get("freeze_projector"),
        "max_length": runtime.get("max_length"),
        "max_joint_length": runtime.get("max_joint_length"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = {}
    for relative in FILES:
        path = OFFICIAL / relative
        record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            record["sha256"] = sha256(path)
            if path.suffix in {".yaml", ".yml"}:
                record["configuration_summary"] = yaml_summary(path)
        files[relative] = record
    receipt = {
        "status": "G2_OFFICIAL_TRAINING_SOURCE_AUDIT_COMPLETE",
        "official_repo": str(OFFICIAL),
        "official_commit": git("rev-parse", "HEAD"),
        "official_commit_date": git("show", "-s", "--format=%cI", "HEAD"),
        "official_remote": git("config", "--get", "remote.origin.url"),
        "files": files,
        "G2_interpretation": {
            "initialization": "published complete GroundAnything-DLM checkpoint",
            "native_loss": "0.5 * MDM + 0.5 * Causal",
            "frozen": ["vision tower", "original Qwen3 language weights", "unregistered modules"],
            "trainable": ["full visual projector", "Qwen3 q/k/v/o LoRA"],
            "data_contract": "id/messages/images/source plus official protocol and row-contract validation",
            "official_cache_contract": "sampling, selected lengths, image integrity, response audit, and topology-matched packing",
            "not_claimed": "G2 does not reproduce the official 64-GPU multi-million-row pretraining stages",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
