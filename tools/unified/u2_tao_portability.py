#!/usr/bin/env python3
"""Full-scope TAO/C-TAO localization audit with shared-backbone vocabulary chunks."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.unified.freeze_u2_tao_portability import FOUNDATION_SHA, group_for, selected_images, sha256

CHECKPOINT = ROOT / "outputs/unified/u1r2/expression_adaptation_epoch1/epoch_1.pth"


def load_scope(paths: list[Path]) -> tuple[dict, list[dict]]:
    scopes, records = {}, []
    for path in paths:
        protocol = json.loads(path.read_text())
        annotation_path = Path(protocol["annotation_path"])
        if sha256(annotation_path) != protocol["annotation_sha256"]:
            raise RuntimeError(f"Annotation differs from registration: {annotation_path}")
        data = json.loads(annotation_path.read_text())
        images = selected_images(data, protocol["image_selector"])
        dataset = protocol["dataset"]
        keys = [f"{dataset}/{image['id']}" for image in images]
        if (len(images) != protocol["measurement_images"] or
                hashlib.sha256("\n".join(keys).encode()).hexdigest() != protocol["frame_keys_sha256"]):
            raise RuntimeError(f"Image scope differs from registration: {path}")
        vocabulary_path = Path(protocol["vocabulary_path"])
        if sha256(vocabulary_path) != protocol["vocabulary_sha256"]:
            raise RuntimeError("Vocabulary differs from registration")
        vocabulary = json.loads(vocabulary_path.read_text())
        scopes[dataset] = {"path": str(path.resolve()), "sha256": sha256(path), "protocol": protocol,
                           "data": data, "vocabulary": vocabulary}
        for image in images:
            records.append({"dataset": dataset, "key": f"{dataset}/{image['id']}",
                            "image_id": image["id"], "file_name": image["file_name"],
                            "image": str(Path(protocol["image_root"]) / image["file_name"])})
    return scopes, records


def wait_for_extracted(record: dict, ready: set[str], extraction_pid: int) -> None:
    split, source = Path(record["file_name"]).parts[:2]
    archive = f"frames/{split}/{source}.zip"
    while archive not in ready:
        for path in (ROOT / "outputs/unified/u2/acquisition").glob("tao_extraction_[0-3].json"):
            # The already-running extractor writes these small records at
            # archive boundaries. Retry a concurrent write, never an image.
            try:
                state = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            ready.update(item["relative_path"] for item in state["archives"]
                         if item["crc_checked_during_extraction"])
        if archive in ready:
            break
        try:
            os.kill(extraction_pid, 0)
        except ProcessLookupError:
            raise RuntimeError(f"Extraction ended before registered source was ready: {archive}")
        time.sleep(10)
    if not Path(record["image"]).is_file():
        raise RuntimeError(f"Registered image missing from CRC-completed source: {record['image']}")


def worker(args: argparse.Namespace) -> int:
    import torch
    from torchvision.ops import box_iou
    from locatemot.unified.runtime.grounding_inference import (
        load_grounding_runtime, prediction_arrays, run_grounding_chunked, runtime_metadata,
    )
    from locatemot.unified.runtime.u2_frozen_text_cache import frozen_text_cache
    from locatemot.unified.runtime.u2_label_metadata import bulk_label_metadata

    torch.set_num_threads(2)
    if args.bulk_label_metadata:
        torch.cuda.set_device(args.gpu)
    scopes, all_records = load_scope(args.protocol)
    records = all_records[args.worker_index::args.worker_count]
    if args.smoke:
        records = [next(row for row in records if row["dataset"] == dataset) for dataset in scopes]
    prefix = args.out / f"worker_{args.worker_index}"
    metadata = {"foundation_sha256": args.expected_foundation_sha256, "protocols": {name: scope["sha256"] for name, scope in scopes.items()},
                "worker_index": args.worker_index, "worker_count": args.worker_count,
                "smoke": args.smoke, "expected_keys": [row["key"] for row in records]}
    meta_path, rows_path = prefix.with_suffix(".metadata.json"), prefix.with_suffix(".jsonl")
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise RuntimeError("Resume metadata differs from the frozen scope")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    previous = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    if [row["key"] for row in previous] != metadata["expected_keys"][:len(previous)]:
        raise RuntimeError("Resume rows are not an exact completed frame prefix")
    runtime = load_grounding_runtime(checkpoint=args.checkpoint, device=f"cuda:{args.gpu}", max_per_img=900)
    if runtime.checkpoint_sha256 != args.expected_foundation_sha256:
        raise RuntimeError("The actual foundation differs from the explicitly registered checkpoint")
    print(json.dumps({"worker": args.worker_index, "state": "LOADED", "runtime": runtime_metadata(runtime)}), flush=True)
    gt, by_group, unmapped_categories = {}, {}, {}
    for dataset, scope in scopes.items():
        gt[dataset] = defaultdict(list)
        categories = {category["id"]: category for category in scope["data"]["categories"]}
        vocabulary = {category["synset"]: category for category in scope["vocabulary"]["categories"]}
        by_group[dataset] = {category_id: group_for(category, vocabulary) for category_id, category in categories.items()}
        unmapped_categories[dataset] = {category_id for category_id, category in categories.items()
                                        if category["synset"] not in vocabulary}
        for annotation in scope["data"]["annotations"]:
            gt[dataset][annotation["image_id"]].append(annotation)
    ready, calls = set(), {"backbone": 0}
    def count_backbone(module, inputs, output):
        calls["backbone"] += 1
    hook = runtime.model.backbone.register_forward_hook(count_backbone)
    started = time.perf_counter()
    text_context = frozen_text_cache(runtime) if args.reuse_frozen_text else nullcontext(None)
    label_context = bulk_label_metadata(runtime) if args.bulk_label_metadata else nullcontext(None)
    with text_context as text_stats, label_context as label_stats, rows_path.open("a") as output:
        for index, record in enumerate(records[len(previous):], start=len(previous)):
            wait_for_extracted(record, ready, args.extraction_pid)
            scope = scopes[record["dataset"]]
            protocol = scope["protocol"]
            calls["backbone"] = 0
            result = run_grounding_chunked(runtime, record["image"], [category["name"] for category in scope["vocabulary"]["categories"]],
                                           chunked_size=protocol["chunked_size"], merge_iou=protocol["merge_iou"],
                                           top_k=protocol["candidate_top_k"])
            if calls["backbone"] != 1:
                raise RuntimeError("Vocabulary replay did not use exactly one image backbone extraction")
            boxes, _ = prediction_arrays(result)
            labels = gt[record["dataset"]][record["image_id"]]
            targets = torch.tensor([[annotation["bbox"][0], annotation["bbox"][1],
                                      annotation["bbox"][0] + annotation["bbox"][2],
                                      annotation["bbox"][1] + annotation["bbox"][3]] for annotation in labels],
                                   dtype=torch.float32).reshape(-1, 4)
            overlaps = box_iou(targets, boxes)
            hits = (overlaps.amax(dim=1) >= .5) if len(boxes) else torch.zeros(len(targets), dtype=torch.bool)
            counts = {"base_gt_total": 0, "base_hits_top300": 0, "novel_gt_total": 0, "novel_hits_top300": 0,
                      "unmapped_gt_total": 0, "unmapped_hits_top300": 0}
            for annotation, hit in zip(labels, hits.tolist()):
                group = by_group[record["dataset"]][annotation["category_id"]]
                counts[group + "_gt_total"] += 1
                counts[group + "_hits_top300"] += int(hit)
                if annotation["category_id"] in unmapped_categories[record["dataset"]]:
                    counts["unmapped_gt_total"] += 1
                    counts["unmapped_hits_top300"] += int(hit)
            output.write(json.dumps({"key": record["key"], "dataset": record["dataset"],
                                     "file_name": record["file_name"], "counts": counts,
                                     "proposals": len(boxes), "backbone_calls": calls["backbone"]}) + "\n")
            output.flush()
            if (index + 1) % 100 == 0:
                print(json.dumps({"worker": args.worker_index, "images": index + 1, "expected": len(records)}), flush=True)
    hook.remove()
    terminal = {"state": "SMOKE_COMPLETE" if args.smoke else "COMPLETE", "metadata": metadata,
                "runtime": runtime_metadata(runtime), "elapsed_seconds": time.perf_counter() - started}
    if args.reuse_frozen_text:
        terminal["frozen_text_cache"] = text_stats
    if args.bulk_label_metadata:
        terminal["bulk_label_metadata"] = label_stats
    prefix.with_suffix(".result.json").write_text(json.dumps(terminal, indent=2) + "\n")
    print(json.dumps({"worker": args.worker_index, "state": terminal["state"], "images": len(records)}), flush=True)
    return 0


def aggregate(args: argparse.Namespace) -> dict:
    scopes, records = load_scope(args.protocol)
    rows = []
    for index in range(len(args.gpus)):
        prefix = args.out / f"worker_{index}"
        result = json.loads(prefix.with_suffix(".result.json").read_text())
        expected_keys = [row["key"] for row in records[index::len(args.gpus)]]
        if (result["state"] != "COMPLETE" or result["metadata"]["smoke"]
                or result["metadata"]["foundation_sha256"] != args.expected_foundation_sha256
                or result["metadata"]["protocols"] != {name: scope["sha256"] for name, scope in scopes.items()}
                or result["metadata"]["expected_keys"] != expected_keys):
            raise RuntimeError(f"Worker {index} has incomplete or different measurement scope")
        worker_rows = [json.loads(line) for line in prefix.with_suffix(".jsonl").read_text().splitlines()]
        if [row["key"] for row in worker_rows] != expected_keys:
            raise RuntimeError(f"Worker {index} did not cover every registered frame")
        rows.extend(worker_rows)
    args.measurements_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for dataset, scope in scopes.items():
        selected = [row for row in rows if row["dataset"] == dataset]
        counts = {name: sum(row["counts"][name] for row in selected) for name in selected[0]["counts"]}
        for group in ("base", "novel"):
            if counts[group + "_gt_total"] != scope["protocol"]["gt_units_expected"][group]:
                raise RuntimeError(f"GT coverage differs from registration: {dataset}/{group}")
        if counts["unmapped_gt_total"] != sum(scope["protocol"]["unmapped_lvis_v1_gt"].values()):
            raise RuntimeError("Unmapped categories were not completely retained")
        payload = {"format": "locatemot-u2-portability-measurement-v1", "status": "COMPLETE", "dataset": dataset,
                   "foundation_sha256": args.expected_foundation_sha256, "protocol_path": scope["path"], "protocol_sha256": scope["sha256"],
                   "images_expected": scope["protocol"]["measurement_images"], "images_evaluated": len(selected),
                   "coverage_keys_sha256": hashlib.sha256("\n".join(sorted(row["key"] for row in selected)).encode()).hexdigest(),
                   "counts": counts, "proposals_per_frame": {
                       "mean": sum(row["proposals"] for row in selected) / len(selected),
                       "min": min(row["proposals"] for row in selected), "max": max(row["proposals"] for row in selected)},
                   "every_frame_backbone_calls_one": all(row["backbone_calls"] == 1 for row in selected),
                   "worker_artifacts": str(args.out.resolve()), "official_test_labels_read": False}
        (args.measurements_dir / f"{dataset}.json").write_text(json.dumps(payload, indent=2) + "\n")
        summary[dataset] = {"images": len(selected), "base_recall": counts["base_hits_top300"] / counts["base_gt_total"],
                            "novel_recall": counts["novel_hits_top300"] / counts["novel_gt_total"] if counts["novel_gt_total"] else None}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, nargs="+", default=[ROOT / f"outputs/unified/u2/protocol/{name}_portability.json" for name in ("tao", "ctao")])
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--expected-foundation-sha256", default=FOUNDATION_SHA)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/unified/u2/perception/tao_frozen_foundation")
    parser.add_argument("--measurements-dir", type=Path, default=ROOT / "outputs/unified/u2/perception/measurements")
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--extraction-pid", type=int, required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--reuse-frozen-text", action="store_true",
                        help="Exact worker-local native text reuse; default F0 commands remain unchanged")
    parser.add_argument("--bulk-label-metadata", action="store_true")
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
        parser.error("A smoke requires --worker-index and a separate --out directory")
    processes = []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
    try:
        for index, gpu in enumerate(args.gpus):
            log = (args.out / f"worker_{index}.log").open("a")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--protocol", *map(str, args.protocol),
                       "--checkpoint", str(args.checkpoint), "--expected-foundation-sha256", args.expected_foundation_sha256,
                       "--out", str(args.out), "--measurements-dir", str(args.measurements_dir),
                       "--extraction-pid", str(args.extraction_pid),
                       "--worker-index", str(index), "--worker-count", str(len(args.gpus)), "--gpu", str(gpu)]
            if args.reuse_frozen_text:
                command.append("--reuse-frozen-text")
            if args.bulk_label_metadata:
                command.append("--bulk-label-metadata")
            processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(json.dumps({"state": "RUNNING", "gpus": args.gpus, "worker_pids": [p.pid for p, _ in processes]}), flush=True)
        codes = []
        with ThreadPoolExecutor(max_workers=len(processes)) as executor:
            for future in as_completed([executor.submit(p.wait) for p, _ in processes]):
                code = future.result()
                codes.append(code)
                if code:
                    for process, _ in processes:
                        if process.poll() is None:
                            process.terminate()
                    break
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()
    if any(codes):
        print(json.dumps({"state": "FAILED", "worker_returncodes": [p.returncode for p, _ in processes]}), flush=True)
        return 1
    print(json.dumps({"state": "COMPLETE", "measurements": aggregate(args)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
