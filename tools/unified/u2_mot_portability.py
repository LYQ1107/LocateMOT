#!/usr/bin/env python3
"""Measure the frozen U1R2 foundation on registered ordinary MOT dev frames."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FOUNDATION_SHA = "4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26"
CHECKPOINT = ROOT / "outputs/unified/u1r2/expression_adaptation_epoch1/epoch_1.pth"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scope(protocol_paths: list[Path]) -> tuple[dict, list[dict]]:
    protocols, records = {}, []
    for path in protocol_paths:
        protocol = json.loads(path.read_text())
        dataset = protocol["dataset"]
        protocols[dataset] = {"path": str(path.resolve()), "sha256": digest(path), "data": protocol}
        for sequence in protocol["sequences"]:
            if dataset == "kitti_legal_development":
                from locatemot.unified.data.legal_scope import assert_legal_video
                assert_legal_video(sequence["sequence"], dataset="refer_kitti_v2")
            if digest(Path(sequence["gt_path"])) != sequence["gt_sha256"]:
                raise RuntimeError(f"GT differs from registered source: {sequence['sequence']}")
            for frame in sequence["validation_frames"]:
                records.append({"dataset": dataset, "sequence": sequence["sequence"],
                                "frame": frame, "key": f"{dataset}/{sequence['sequence']}/{frame}",
                                "image": str(Path(sequence["image_dir"]) / frame),
                                "gt_path": sequence["gt_path"]})
    return protocols, records


def worker(args: argparse.Namespace) -> int:
    import torch
    from torchvision.ops import box_iou
    from locatemot.unified.runtime.grounding_inference import (
        load_grounding_runtime, prediction_arrays, run_grounding, runtime_metadata,
    )

    torch.set_num_threads(2)
    protocols, all_records = scope(args.protocol)
    records = all_records[args.worker_index::args.worker_count]
    if args.smoke:
        records = records[:1]
    prefix = args.out / f"worker_{args.worker_index}"
    metadata_path = prefix.with_suffix(".metadata.json")
    rows_path = prefix.with_suffix(".jsonl")
    metadata = {"foundation_sha256": args.expected_foundation_sha256,
                "protocols": {name: value["sha256"] for name, value in protocols.items()},
                "worker_index": args.worker_index, "worker_count": args.worker_count,
                "smoke": args.smoke, "expected_keys": [r["key"] for r in records]}
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise RuntimeError(f"Resume scope differs: {metadata_path}")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    previous = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    if [row["key"] for row in previous] != metadata["expected_keys"][:len(previous)]:
        raise RuntimeError(f"Resume rows are not an exact completed prefix: {rows_path}")
    runtime = load_grounding_runtime(checkpoint=args.checkpoint, device=f"cuda:{args.gpu}", max_per_img=900)
    if runtime.checkpoint_sha256 != args.expected_foundation_sha256:
        raise RuntimeError("The loaded foundation differs from the explicitly registered checkpoint")
    print(json.dumps({"worker": args.worker_index, "state": "LOADED", "runtime": runtime_metadata(runtime)}), flush=True)
    labels = {}
    started = time.perf_counter()
    with rows_path.open("a") as output:
        for record in records[len(previous):]:
            protocol = protocols[record["dataset"]]["data"]
            result = run_grounding(runtime, record["image"], protocol["vocabulary"], custom_entities=True)
            boxes, scores = prediction_arrays(result)
            if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                raise RuntimeError(f"Nonfinite prediction: {record['key']}")
            boxes = boxes[torch.argsort(scores, descending=True)]
            # Labels are evaluation-only: the inference call above sees just
            # the image and the preregistered vocabulary.
            if record["gt_path"] not in labels:
                by_frame = defaultdict(list)
                if record["gt_path"].endswith(".json"):
                    from locatemot.unified.data.bdd_tracking import frame_objects
                    for frame in json.loads(Path(record["gt_path"]).read_text()):
                        objects, _ = frame_objects(frame)
                        by_frame[frame["name"]] = [obj["bbox"] for obj in objects]
                elif record["dataset"] == "kitti_legal_development":
                    for line in Path(record["gt_path"]).read_text().splitlines():
                        fields = line.split()
                        # Match the frozen upstream Kitti2DBox GT preprocessing;
                        # do not replace original tracking IDs with Refer IDs.
                        if (fields[2].lower() in ("car", "pedestrian")
                                and int(float(fields[4])) <= 2 and int(float(fields[3])) <= 0):
                            by_frame[int(fields[0])].append([float(value) for value in fields[6:10]])
                else:
                    for line in Path(record["gt_path"]).read_text().splitlines():
                        fields = [float(value) for value in line.split(",")]
                        if fields[6] == 1 and fields[7] == 1:
                            x, y, width, height = fields[2:6]
                            by_frame[int(fields[0])].append([x, y, x + width, y + height])
                labels[record["gt_path"]] = by_frame
            frame_key = record["frame"] if record["gt_path"].endswith(".json") else int(Path(record["frame"]).stem)
            targets = torch.tensor(labels[record["gt_path"]][frame_key], dtype=torch.float32).reshape(-1, 4)
            counts = {"gt_total": len(targets)}
            overlaps = box_iou(targets, boxes[:300])
            for k in (150, 300):
                best = overlaps[:, :k].amax(dim=1) if overlaps.shape[1] else torch.zeros(len(targets))
                counts[f"hits_top{k}"] = int((best >= protocol["iou_threshold"]).sum())
            row = {"key": record["key"], "dataset": record["dataset"], "sequence": record["sequence"],
                   "frame": record["frame"], "proposals": len(boxes), "counts": counts}
            output.write(json.dumps(row) + "\n")
            output.flush()
    payload = {"state": "SMOKE_COMPLETE" if args.smoke else "COMPLETE", "metadata": metadata,
               "runtime": runtime_metadata(runtime), "elapsed_seconds": time.perf_counter() - started}
    prefix.with_suffix(".result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"worker": args.worker_index, "state": payload["state"], "images": len(records)}), flush=True)
    return 0


def aggregate(args: argparse.Namespace) -> dict:
    protocols, records = scope(args.protocol)
    rows = []
    for index in range(len(args.gpus)):
        prefix = args.out / f"worker_{index}"
        terminal = json.loads(prefix.with_suffix(".result.json").read_text())
        if terminal["state"] != "COMPLETE":
            raise RuntimeError(f"Incomplete measurement worker {index}")
        expected = {"foundation_sha256": args.expected_foundation_sha256,
                    "protocols": {name: value["sha256"] for name, value in protocols.items()},
                    "worker_index": index, "worker_count": len(args.gpus), "smoke": False,
                    "expected_keys": [r["key"] for r in records[index::len(args.gpus)]]}
        if terminal["metadata"] != expected:
            raise RuntimeError(f"Different measurement worker scope: {index}")
        worker_rows = [json.loads(line) for line in prefix.with_suffix(".jsonl").read_text().splitlines()]
        if [row["key"] for row in worker_rows] != expected["expected_keys"]:
            raise RuntimeError(f"Incomplete measurement frame coverage: {index}")
        rows.extend(worker_rows)
    summary = {}
    args.measurements_dir.mkdir(parents=True, exist_ok=True)
    for dataset, registered in protocols.items():
        selected = [row for row in rows if row["dataset"] == dataset]
        counts = {key: sum(row["counts"][key] for row in selected)
                  for key in ("gt_total", "hits_top150", "hits_top300")}
        sequence_counts = {}
        for sequence in registered["data"]["sequences"]:
            seq_rows = [row for row in selected if row["sequence"] == sequence["sequence"]]
            sequence_counts[sequence["sequence"]] = {
                "images": len(seq_rows),
                "counts": {key: sum(row["counts"][key] for row in seq_rows) for key in counts},
            }
        payload = {"format": "locatemot-u2-portability-measurement-v1", "status": "COMPLETE",
                   "dataset": dataset, "foundation_sha256": args.expected_foundation_sha256,
                   "protocol_path": registered["path"], "protocol_sha256": registered["sha256"],
                   "images_expected": registered["data"]["validation_images"], "images_evaluated": len(selected),
                   "candidate_top_k": registered["data"]["candidate_top_k"],
                   "counts": counts, "per_sequence": sequence_counts,
                   "coverage_keys_sha256": hashlib.sha256("\n".join(sorted(row["key"] for row in selected)).encode()).hexdigest(),
                   "worker_artifacts": str(args.out.resolve()), "official_test_labels_read": False}
        (args.measurements_dir / f"{dataset}.json").write_text(json.dumps(payload, indent=2) + "\n")
        summary[dataset] = {"images": len(selected), "gt": counts["gt_total"],
                            "top150": counts["hits_top150"] / counts["gt_total"],
                            "top300": counts["hits_top300"] / counts["gt_total"]}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, nargs="+", default=[ROOT / f"outputs/unified/u2/protocol/{name}_half.json" for name in ("mot17", "mot20")])
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--expected-foundation-sha256", default=FOUNDATION_SHA)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/unified/u2/perception/mot_half_frozen_foundation")
    parser.add_argument("--measurements-dir", type=Path, default=ROOT / "outputs/unified/u2/perception/measurements")
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.expected_foundation_sha256 != FOUNDATION_SHA:
        if (args.out.resolve() == parser.get_default("out").resolve()
                or args.measurements_dir.resolve() == parser.get_default("measurements_dir").resolve()):
            parser.error("Adapted foundations require separate --out and --measurements-dir; preserve F0 measurements")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker_index is not None:
        return worker(args)
    if args.aggregate_only:
        print(json.dumps(aggregate(args), indent=2))
        return 0
    if args.smoke:
        parser.error("--smoke requires --worker-index and a separate --out directory")
    processes = []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
    # One parent shell command owns launch, blocking waits and aggregation.
    # Workers resume exact completed frame prefixes and never drop failed frames.
    try:
        for index, gpu in enumerate(args.gpus):
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--protocol", *map(str, args.protocol),
                       "--checkpoint", str(args.checkpoint), "--expected-foundation-sha256", args.expected_foundation_sha256,
                       "--out", str(args.out), "--measurements-dir", str(args.measurements_dir),
                       "--worker-index", str(index), "--worker-count", str(len(args.gpus)), "--gpu", str(gpu)]
            log = (args.out / f"worker_{index}.log").open("a")
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((process, log))
        print(json.dumps({"state": "RUNNING", "parent_pid": os.getpid(), "gpus": args.gpus,
                          "worker_pids": [p.pid for p, _ in processes]}), flush=True)
        codes = [process.wait() for process, _ in processes]
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
    if any(codes):
        print(json.dumps({"state": "FAILED", "worker_returncodes": codes}), flush=True)
        return 1
    print(json.dumps({"state": "COMPLETE", "measurements": aggregate(args)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
