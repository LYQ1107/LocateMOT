#!/usr/bin/env python3
"""Complete same-scope portability and U1 retention for one bounded U2-P epoch."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.unified.build_u2_portability_artifact import read_measurement
from tools.unified.select_u2_perception_adaptation import REFERENCES, SCOPE_FIELDS, verified_audit
from tools.unified.train_u2_perception_adaptation import CONFIG, DIRECTORY, digest

ORDINARY = ("bdd100k", "dancetrack", "mot17", "mot20", "kitti_legal_development")
FROZEN_PREFIXES = ("language_model.", "backbone.patch_embed.", "backbone.stages.0.",
                   "backbone.stages.1.", "backbone.norm0.", "backbone.norm1.")


def ordinary_protocol(name: str) -> Path:
    filename = "kitti_ordinary_portability.json" if name == "kitti_legal_development" else name + ("_half" if name in ("mot17", "mot20") else "") + ".json"
    return ROOT / "outputs/unified/u2/protocol" / filename


def run(command: list[str], log: Path | None = None) -> None:
    if log is None:
        subprocess.run(command, check=True, cwd=ROOT)
    else:
        with log.open("a") as handle:
            subprocess.run(command, check=True, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)


def checkpoint_contract(path: Path, epoch: int, sha: str) -> dict:
    import torch
    from tools.unified.u2_mot_portability import CHECKPOINT
    candidate = torch.load(path, map_location="cpu", weights_only=False)
    if candidate["meta"]["epoch"] != epoch or candidate["meta"]["iter"] != epoch * 1250:
        raise ValueError("Candidate is not the completed registered 1,250-step epoch")
    reference = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state, original = candidate["state_dict"], reference["state_dict"]
    keys = [key for key in original if key.startswith(FROZEN_PREFIXES)]
    if not keys or any(not torch.equal(state[key], original[key]) for key in keys):
        raise ValueError("BERT or frozen Swin stages differ from immutable U1R2")
    return {"state": "PASS", "epoch": epoch, "optimizer_steps": epoch * 1250,
            "foundation_sha256": sha, "frozen_tensors_compared": len(keys),
            "bert_and_swin_stage_0_1_exactly_retained": True}


def reference_shards(name: str) -> dict[int, dict]:
    result = json.loads(REFERENCES[name].read_text())
    shards = [json.loads(Path(path).read_text()) for path in result["shards"]]
    if len(shards) != 8 or any(row["worker_count"] != 8 for row in shards):
        raise ValueError("The original U1 audit must retain its eight-worker partition")
    return {row["worker_index"]: row for row in shards}


def validate_shard(path: Path, name: str, index: int, sha: str, reference: dict) -> None:
    result = json.loads(path.read_text())
    if (result["worker_index"] != index or result["worker_count"] != 8
            or result["runtime"]["checkpoint_sha256"] != sha
            or any(result[field] != reference[field] for field in SCOPE_FIELDS[name])):
        raise ValueError(f"Existing {name} shard has a different checkpoint or full scope: {path}")


def u1_audit(name: str, args: argparse.Namespace, sha: str) -> Path:
    scripts = {"generic": "u1_formal_generic_gate.py", "expression": "u1_formal_expression_gate.py",
               "counterfactual": "u1_expression_counterfactual_audit.py"}
    script = str(ROOT / "tools/unified" / scripts[name])
    folder = args.out / name
    folder.mkdir(parents=True, exist_ok=True)
    expected = reference_shards(name)
    paths = [folder / f"worker_{index}.json" for index in range(8)]
    processes = []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
    try:
        for index, gpu in enumerate(args.gpus):
            if paths[index].exists():
                validate_shard(paths[index], name, index, sha, expected[index])
                continue
            command = [sys.executable, "-u", script, "--worker-index", str(index), "--worker-count", "8",
                       "--gpu", str(gpu), "--checkpoint", str(args.checkpoint), "--config", str(CONFIG),
                       "--out", str(paths[index])]
            log = (folder / f"worker_{index}.log").open("a")
            processes.append((subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log))
        with ThreadPoolExecutor(max_workers=8) as executor:
            for future in as_completed([executor.submit(process.wait) for process, _ in processes]):
                code = future.result()
                if code:
                    raise RuntimeError(f"U1 {name} worker failed with return code {code}; inspect {folder}")
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()
    for index, path in enumerate(paths):
        validate_shard(path, name, index, sha, expected[index])
    result = folder / "result.json"
    run([sys.executable, script, "--aggregate", "--shards", *map(str, paths), "--out", str(result)])
    verified_audit(result, name, sha)
    return result


def evaluation_plan(args: argparse.Namespace, sha: str) -> dict:
    folder = ROOT / "outputs/unified/u2/protocol"
    ordinary_images = sum(json.loads(ordinary_protocol(name).read_text())["validation_images"]
                          for name in ORDINARY)
    tao_images = sum(json.loads((folder / f"{name}_portability.json").read_text())["measurement_images"] for name in ("tao", "ctao"))
    references = {name: json.loads(path.read_text()) for name, path in REFERENCES.items()}
    return {"state": "PLAN_ONLY_NOT_EXECUTED", "epoch": args.epoch,
            "checkpoint": str(args.checkpoint.resolve()), "foundation_sha256": sha,
            "gpus": args.gpus, "ordinary_images": ordinary_images, "tao_ctao_images": tao_images,
            "u1_generic_images": references["generic"]["image_count"],
            "u1_expression_query_frames": references["expression"]["query_frame_count"],
            "counterfactual_pairs": references["counterfactual"]["pair_count"], "ordinary_streams": list(ORDINARY),
            "maximum_candidate_k": 300, "crowded_override_minimum": .90,
            "all_original_scopes_retained": True, "official_test_labels_read": False,
            "track_training_launched": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epoch", type=int, choices=(1, 2), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--extraction-pid", type=int, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8:
        parser.error("Use eight distinct GPUs to retain the original U1 worker partition")
    args.out = args.out or DIRECTORY / f"eval_epoch_{args.epoch}"
    sha = digest(args.checkpoint)
    plan = evaluation_plan(args, sha)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    contract = checkpoint_contract(args.checkpoint, args.epoch, sha)
    (args.out / "checkpoint_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    measurements = args.out / "measurements"
    common = ["--checkpoint", str(args.checkpoint), "--expected-foundation-sha256", sha,
              "--measurements-dir", str(measurements), "--gpus", *map(str, args.gpus)]
    protocols = [ordinary_protocol(name) for name in ORDINARY]
    if any(read_measurement(measurements, name, sha) is None for name in ORDINARY):
        print(json.dumps({"state": "ORDINARY_FULL_AUDIT", "epoch": args.epoch}), flush=True)
        run([sys.executable, "-u", str(ROOT / "tools/unified/u2_mot_portability.py"),
             "--protocol", *map(str, protocols), "--out", str(args.out / "ordinary"), *common], args.out / "ordinary.log")
    if any(read_measurement(measurements, name, sha) is None for name in ("tao", "ctao")):
        print(json.dumps({"state": "TAO_CTAO_FULL_AUDIT", "epoch": args.epoch}), flush=True)
        run([sys.executable, "-u", str(ROOT / "tools/unified/u2_tao_portability.py"),
             "--out", str(args.out / "tao_ctao"), "--extraction-pid", str(args.extraction_pid),
             "--reuse-frozen-text", "--bulk-label-metadata", *common], args.out / "tao_ctao.log")
    results = {}
    for name in REFERENCES:
        print(json.dumps({"state": "U1_FULL_AUDIT", "audit": name, "epoch": args.epoch}), flush=True)
        results[name] = u1_audit(name, args, sha)
    crowded = args.out / "mot20_crowded_registration.json"
    mot20 = read_measurement(measurements, "mot20", sha)
    counts = mot20["counts"]
    if counts["hits_top150"] / counts["gt_total"] < .85 and counts["hits_top300"] / counts["gt_total"] >= .90:
        if not crowded.exists():
            run([sys.executable, str(ROOT / "tools/unified/register_u2_crowded_candidates.py"),
                 "--measurements-dir", str(measurements), "--foundation-sha256", sha, "--out", str(crowded)])
    portability = args.out / "portability.json"
    command = [sys.executable, str(ROOT / "tools/unified/build_u2_portability_artifact.py"),
               "--expected-foundation-sha256", sha, "--measurements-dir", str(measurements),
               "--u1-generic-result", str(results["generic"]), "--out", str(portability)]
    if crowded.exists():
        command += ["--crowded-registration", str(crowded)]
    run(command)
    selection = args.out / "selection.json"
    run([sys.executable, str(ROOT / "tools/unified/select_u2_perception_adaptation.py"),
         "--foundation-sha256", sha, "--epoch", str(args.epoch), "--portability", str(portability),
         "--out", str(selection), *[value for name, path in results.items() for value in ("--" + name, str(path))]])
    print(json.dumps({"state": "COMPLETE", "epoch": args.epoch, "selection": str(selection),
                      "status": json.loads(selection.read_text())["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
