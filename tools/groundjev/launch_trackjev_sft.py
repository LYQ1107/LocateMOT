"""Launch one TrackJEV SFT stage (J0 / J1 / J2) through the resumable trainer.

Refuses to start when the registered prerequisites are missing, so a run can
never silently train on absent or illegal data.

Example:
  python tools/groundjev/launch_trackjev_sft.py --stage J2_full_language \
      --rows outputs/groundjev/full_data_v2/trackjev_sft_v1/rows.jsonl \
      --name trackjev_J2_v1 --steps 2000 --device cuda:6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "groundjev" / "trackjev_j_stages_v1.json"
CURRICULUM = ROOT / "configs" / "groundjev" / "trackjev_curriculum_v1.json"
RUNS = Path("/data3/user/LocateMOT-runs/TrackJEV_FullDataV2")
CKPT = Path("/data3/user/LocateMOT-checkpoints/TrackJEV_FullDataV2")
TRAINER = ROOT / "tools" / "groundjev" / "onejev_resumable_sft.py"
PY = "/data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["J0_frozen", "J1_lora", "J2_full_language"])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=32768)
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text())
    stage = registry["stages"][args.stage]

    if not args.rows.is_file():
        return _fail(f"row file missing: {args.rows} (run the candidate export and row export first)")
    if not CURRICULUM.is_file():
        return _fail(f"curriculum missing: {CURRICULUM}")
    curriculum = json.loads(CURRICULUM.read_text())
    rows = sum(1 for _ in args.rows.open())

    run_dir = RUNS / args.name
    ckpt_dir = CKPT / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [PY, "-u", str(TRAINER),
           "--model", registry["base_model"],
           "--train", str(args.rows),
           "--name", args.name,
           "--runs-dir", str(RUNS),
           "--device", args.device,
           "--epochs", str(args.epochs),
           "--max-steps", str(args.steps),
           "--max-len", str(args.max_len),
           "--batch-tokens", str(args.batch_tokens),
           "--max-rows", str(args.max_rows),
           "--save-every", str(args.save_every),
           "--log-every", "10"]
    # NOTE: never pass --seed here. The resumable wrapper defines --seed-global-rngs, so
    # argparse prefix-matches "--seed 0" onto it and leaves a stray "0" argument.
    # A non-zero seed is supplied through the upstream YAML config instead.
    if args.seed:
        config_path = run_dir / "seed_config.yaml"
        config_path.write_text(f"seed: {int(args.seed)}\n")
        cmd += ["--config", str(config_path)]
    flags = stage["flags"]
    if flags.get("multimodal"):
        cmd += ["--multimodal", "--media-root", str(ROOT)]
    if flags.get("full_finetune"):
        cmd += ["--full-finetune"]
    if flags.get("train_vision"):
        cmd += ["--train-vision"]
    if flags.get("gradient_checkpointing"):
        pass  # on by default in the trainer
    if flags.get("lora_r"):
        cmd += ["--lora-r", str(flags["lora_r"]), "--lora-alpha", str(flags["lora_alpha"]),
                "--lora-dropout", str(flags["lora_dropout"])]

    manifest = {
        "stage": args.stage,
        "name": args.name,
        "command": cmd,
        "rows_file": str(args.rows),
        "rows": rows,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "curriculum_ready_datasets": curriculum.get("ready_datasets"),
        "curriculum_pending_datasets": curriculum.get("pending_datasets"),
        "curriculum_coverage_complete": curriculum.get("coverage_complete"),
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("training data are the legal rows only; coverage_complete reflects whether every "
                 "planned dataset participated, and is copied here verbatim"),
    }
    (run_dir / "launch_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({"dry_run": args.dry_run, "manifest": str(run_dir / "launch_manifest.json"),
                      "rows": rows,
                      "coverage_complete": manifest["curriculum_coverage_complete"],
                      "pending": manifest["curriculum_pending_datasets"],
                      "command": " ".join(cmd)[:600]}, indent=1, ensure_ascii=False))
    if args.dry_run:
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'external' / 'OneJev'}"
    log_path = run_dir / "train.log"
    with log_path.open("w") as handle:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
    receipt = {"pid": proc.pid, "log": str(log_path), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "command": cmd, "stage": args.stage, "rows": rows}
    (run_dir / "launch_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1, ensure_ascii=False)[:1500])
    return 0


def _fail(message: str) -> int:
    print(json.dumps({"status": "REFUSED", "reason": message}, indent=1, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    sys.exit(main())
