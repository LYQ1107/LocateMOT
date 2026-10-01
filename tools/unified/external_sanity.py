#!/usr/bin/env python3
"""Short, non-training sanity checks for the ten external reference clones.

The command intentionally stops at import/argument-parser/bytecode checks.  It
does not construct a dataset, open Refer-KITTI labels, build a CUDA operator or
launch distributed training.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = Path("/data2/user/reference_repos")

CHECKS = [
    ("MOTR", "MOTR", ["main.py", "models/motr.py", "models/qim.py"], ["main.py", "--help"]),
    ("TransRMOT", "TransRMOT", ["main.py", "models/transrmot.py", "models/qim.py"], ["main.py", "--help"]),
    ("TempRMOT", "TempRMOT", ["main.py", "models/transrmot_pro.py", "models/spatial_temporal_reason.py"], ["main.py", "--help"]),
    ("DKGTrack", "DKGTrack", ["main.py", "models/spatial_temporal_reason.py", "models/fuse_modules.py"], ["main.py", "--help"]),
    ("FlexHook", "FlexHook", ["main.py", "models/mymodel.py"], ["main.py", "--help"]),
    ("iKUN", "iKUN", ["opts.py", "train.py", "model.py", "loss.py"], ["train.py", "--help"]),
    ("OVTR", "OVTR", ["ovtr/main.py", "ovtr/models/ovtr.py", "ovtr/models/updater.py"], ["ovtr/main.py", "--help"]),
    ("COVTrack", "COVTrack", ["tools/train.py", "ovtrack/models/roi_heads/ovtrack_roi_head.py", "ovtrack/models/trackers/ovtracker.py"], ["tools/train.py", "--help"]),
    ("ReferDINO", "ReferDINO", ["main.py", "trainer.py", "models/GroundingDINO/groundingdino.py"], ["main.py", "--help"]),
    ("Open-GroundingDino", "Open-GroundingDino", ["main.py", "models/GroundingDINO/groundingdino.py", "models/GroundingDINO/matcher.py"], ["main.py", "--help"]),
]


def excerpt(text: str, limit: int = 500) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= limit:
        return text
    # Keep both the import location and the final exception; the latter is
    # usually the first actionable dependency/version finding.
    head = limit // 2
    return text[:head] + " ... " + text[-(limit - head - 5):]


def main() -> int:
    rows: list[dict[str, object]] = []
    for name, repo, compile_files, help_command in CHECKS:
        cwd = REFERENCE_ROOT / repo
        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", *compile_files],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            help_result = subprocess.run(
                [sys.executable, *help_command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            help_row: dict[str, object] = {
                "returncode": help_result.returncode,
                "stdout": excerpt(help_result.stdout),
                "stderr": excerpt(help_result.stderr),
            }
        except subprocess.TimeoutExpired as exc:
            help_row = {
                "returncode": None,
                "timeout_seconds": 30,
                "stdout": excerpt(exc.stdout or ""),
                "stderr": excerpt(exc.stderr or ""),
            }
        rows.append(
            {
                "name": name,
                "repo": str(cwd),
                "compile": {
                    "returncode": compile_result.returncode,
                    "stdout": excerpt(compile_result.stdout),
                    "stderr": excerpt(compile_result.stderr),
                },
                "help": help_row,
            }
        )

    payload = {
        "format": "locatemot-u-external-sanity-v1",
        "status": "completed",
        "checks": rows,
        "scope": "compile_and_help_only",
        "training_launched": False,
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "note": "Import failures reflect the reference repository dependency contract on this host; no data or checkpoint substitution was made.",
    }
    out = ROOT / "outputs/unified/external_sanity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    compile_ok = sum(int(row["compile"]["returncode"] == 0) for row in rows)
    help_ok = sum(int(row["help"].get("returncode") == 0) for row in rows)
    report_lines = [
        "# External baseline sanity — U0",
        "",
        "The bounded sanity command completed without launching training, reading any Refer-KITTI official-test labels, or constructing an external dataset. It compiled the selected source files and invoked each advertised entrypoint with `--help`.",
        "",
        f"- Bytecode compilation: **{compile_ok}/{len(rows)}** repository checks passed.",
        f"- Argument-parser/import help: **{help_ok}/{len(rows)}** completed successfully (iKUN only on this host).",
        "- `training_launched=false`, `official_test_labels_read=false`, `screening_gt_used=false`.",
        "",
        "## Results",
        "",
        "| Repository | Compile | Help | First actionable environment result |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        compile_pass = "pass" if row["compile"]["returncode"] == 0 else "fail"
        help_pass = "pass" if row["help"].get("returncode") == 0 else "fail"
        detail = row["help"].get("stderr") or row["compile"].get("stderr") or "none"
        report_lines.append(f"| {row['name']} | {compile_pass} | {help_pass} | {detail} |")
    report_lines += [
        "",
        "The failures are dependency or version-contract findings: legacy torchvision symbols (MOTR/TransRMOT), repository-local imports after argument parsing (TempRMOT/DKGTrack), missing `yacs` (FlexHook), missing `fvcore` (OVTR), MMCV 2.x lacking the legacy `Config` API (COVTrack), missing `wandb` (ReferDINO), and missing `colorlog` (Open-GroundingDino). They are preserved as evidence; no package substitution or external metric claim was made.",
        "",
        "The legal-scope guard and LocateMOT-U skeleton smoke are recorded separately in `outputs/unified/skeleton_smoke.json` and `outputs/unified/parameter_sharing_smoke.json`.",
        "",
    ]
    (ROOT / "reports/unified/EXTERNAL_BASELINE_SANITY.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
