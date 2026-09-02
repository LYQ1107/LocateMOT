#!/usr/bin/env python3
"""Build one canonical L84 probe state for each registered seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l84_paired_probe import L84PairedProbe, L84PairedProbeConfig

SEEDS = (20260829, 20260830, 20260831)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l84/protocol")
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed in SEEDS:
        path = out / f"probe_init_seed{seed}.pt"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite canonical state: {path}")
        torch.manual_seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        random.seed(seed)
        probe = L84PairedProbe(L84PairedProbeConfig())
        state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
        torch.save({
            "format": "locatemot-l84-canonical-probe-init-v1",
            "seed": seed,
            "model_config": {"input_dim": 256, "hidden": 256, "dropout": 0.05},
            "state_dict": state,
            "parameter_count": int(sum(value.numel() for value in probe.parameters())),
        }, path)
        manifest.append({
            "seed": seed,
            "path": str(path),
            "sha256": sha256_file(path),
            "parameter_count": int(sum(value.numel() for value in probe.parameters())),
            "state_keys": list(state),
        })
        del probe
    payload = {
        "format": "locatemot-l84-canonical-probe-init-manifest-v1",
        "status": "complete",
        "seeds": list(SEEDS),
        "architecture": {"input_dim": 256, "hidden": 256, "dropout": 0.05, "parameter_count": 66561},
        "initializations": manifest,
        "command": " ".join([str(Path(__file__).resolve()), *[str(x) for x in []]]),
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "hota_trackeval_run": False,
        "ordinary_mot_ovmot_touched": False,
    }
    (out / "probe_initializations.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "out": str(out), "seeds": list(SEEDS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
