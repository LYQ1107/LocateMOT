#!/usr/bin/env python3
"""Repair only the dataset seqmaps of a completed L86 oracle output.

The original oracle wrote all GT/prediction sequence files but overwrote each
dataset seqmap when its last video was processed.  This targeted repair uses
the already completed, GT-privileged files through project-local symlinks;
it does not rerun the oracle, alter GT/predictions, or copy large artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); source = args.source.resolve(); out = args.out.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"refusing nonempty repair output: {out}")
    out.mkdir(parents=True, exist_ok=True); command = " ".join([sys.executable, *sys.argv])
    try:
        if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        summary = json.loads((source / "summary.json").read_text())
        if summary.get("status") != "complete" or not summary.get("full_video"):
            raise AssertionError("source oracle is not complete full-video output")
        repaired = []
        for dataset in DATASETS:
            src = source / dataset; dst = out / dataset
            if not (src / "gt").is_dir() or not (src / "trackers" / "l86_oracle" / "data").is_dir():
                raise FileNotFoundError(f"source oracle dataset incomplete: {src}")
            dst.mkdir(parents=True, exist_ok=True)
            os.symlink((src / "gt").resolve(), dst / "gt")
            os.symlink((src / "trackers").resolve(), dst / "trackers")
            sequences = sorted(path.name for path in (src / "gt").iterdir() if path.is_dir())
            tracker_files = sorted(path.name for path in (src / "trackers" / "l86_oracle" / "data").glob("*.txt"))
            if len(sequences) != len(tracker_files) or set(tracker_files) != {value + ".txt" for value in sequences}:
                raise AssertionError(f"source sequence file drift {dataset}: {len(sequences)} / {len(tracker_files)}")
            (dst / "seqmap.txt").write_text("name\n" + "\n".join(sequences) + "\n")
            repaired.append({"dataset": dataset, "sequence_count": len(sequences), "source": str(src),
                             "seqmap": str((dst / "seqmap.txt").resolve()), "gt_symlink": True, "tracker_symlink": True})
        payload = dict(summary)
        payload.update({"format": "locatemot-l86-semantic-oracle-repair-v1", "status": "complete",
                        "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                        "source_oracle": str(source), "source_oracle_summary_sha256": sha256(source / "summary.json"),
                        "seqmap_repaired_without_rerun": True, "large_files_copied": False,
                        "videos": repaired, "sequence_count": sum(x["sequence_count"] for x in repaired),
                        "next_action": "run L86 TrackEval wrapper on the repaired seqmaps"})
        write_json(out / "summary.json", payload); write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 0
    except Exception:
        trace = traceback.format_exc(); (out / "INCOMPLETE.md").write_text("# L86 oracle seqmap repair — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l86-semantic-oracle-repair-v1", "status": "incomplete",
                                         "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False,
                                         "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                         "hota_trackeval_run": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
