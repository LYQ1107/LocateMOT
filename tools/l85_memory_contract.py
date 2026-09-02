#!/usr/bin/env python3
"""Label-free GPU capacity audit for complete-candidate/query tiling."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from locatemot.models.l85_full_rmot import L85FullRMOT, L85Config  # noqa: E402
from locatemot.rmot.l85_fullvideo_bank import EXPECTED_MANIFEST_SHA, MANIFEST, sha256_file  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); command = " ".join([sys.executable, *sys.argv])
    try:
        if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA: raise AssertionError("manifest SHA drift")
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        device = torch.device(args.device); torch.cuda.set_device(device)
        # N is read from an already audited compact protocol JSON, never from labels.
        protocol = json.loads((ROOT / "outputs/l85/audit/protocol/protocol.json").read_text())
        max_n = max(max(int(frame["rows"]) for frame in row["frame_stats"]) for row in protocol["per_video"])
        model = L85FullRMOT(L85Config()).to(device=device, dtype=torch.float32).eval()
        rows = []
        for tile in (8, 16, 24, 32):
            torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); ok = True; error = None
            try:
                q = min(tile, 32); n = max_n; h = 8; d = 256; obs = 1432
                z = torch.randn(q, n, d, device=device); p = torch.randn(q, 512, device=device)
                current = torch.randn(n, obs, device=device); history = torch.randn(n, h, obs, device=device)
                mask = torch.ones(n, h, dtype=torch.bool, device=device); frames = torch.arange(h, device=device).repeat(n, 1)
                with torch.inference_mode():
                    output = model(z, p, current, history, mask, frames, int(h - 1), temporal_enabled=True)
                finite = all(bool(torch.isfinite(x.float()).all()) for x in output.values())
                del z, p, current, history, mask, frames, output
                if not finite: raise FloatingPointError("nonfinite memory contract output")
            except RuntimeError as exc:
                ok = False; error = f"{type(exc).__name__}: {exc}"
            rows.append({"query_tile": tile, "candidate_count": max_n, "ok": ok, "error": error,
                         "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "wall_sec": time.perf_counter() - started,
                         "candidate_set_complete": True, "labels_used": False})
            gc.collect(); torch.cuda.empty_cache()
        legal = [row for row in rows if row["ok"] and row["peak_memory_bytes"] < int(0.90 * torch.cuda.get_device_properties(device).total_memory)]
        if not legal: raise RuntimeError("no legal query tile under 90% GPU memory")
        selected = max(legal, key=lambda row: int(row["query_tile"]))
        value = {"format": "locatemot-l85-memory-contract-v1", "status": "complete", "command": command, "cwd": str(ROOT),
                 "candidate_count_max": max_n, "trials": rows, "selected_query_tile": selected["query_tile"],
                 "selection": "largest label-free tile with no OOM and peak memory <90%", "total_gpu_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
                 "candidate_deletion": False, "candidate_truncation": False, "screening_gt_used": False,
                 "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "training_run": False,
                 "hota_trackeval_run": False, "failure_root_cause": None, "next_action": "freeze tile in training config"}
        write_json(out / "memory_contract.json", value); write_json(out / "provenance.json", value); write_json(out / "status.json", value)
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text("# L85 memory contract — INCOMPLETE\n\n" + __import__("traceback").format_exc() + "\n")
        write_json(out / "status.json", {"format": "locatemot-l85-memory-contract-v1", "status": "incomplete", "command": command,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md", "next_action": "repair memory contract",
                                          "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        return 1


if __name__ == "__main__": raise SystemExit(main())
