#!/usr/bin/env python3
"""Build complete label-free video shards only after full U2-P selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.unified.select_u2_perception_adaptation import verify_selected_foundation

DIRECTORY = ROOT / "outputs/unified/u2/perception_cache"
PLAN = DIRECTORY / "frame_inventory.jsonl"
PROTOCOL = ROOT / "outputs/unified/u2/protocol"
SHARD_FRAMES = 64


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inventory() -> dict:
    groups, sources, supervision = [], {}, {}
    names = {"bdd100k": "bdd100k.json", "dancetrack": "dancetrack.json",
             "mot17": "mot17_half.json", "mot20": "mot20_half.json",
             "kitti_legal_development": "kitti_legal_tracking.json"}
    for dataset, filename in names.items():
        path = PROTOCOL / filename
        protocol = json.loads(path.read_text())
        sources[dataset] = {"path": str(path), "sha256": digest(path)}
        for seq in protocol["sequences"]:
            for role, frame_field in (("train", "train_frames"),
                                      ("dev", "development_frames" if dataset == "kitti_legal_development" else "validation_frames")):
                frames = seq[frame_field]
                if not frames:
                    continue
                key = f"{dataset}/{role}/{seq['sequence']}"
                groups.append({"key": key, "dataset": dataset, "role": role,
                               "video": seq["sequence"], "image_dir": seq["image_dir"],
                               "frames": frames, "vocabulary": protocol["vocabulary"],
                               "protocol": str(path), "protocol_sha256": sources[dataset]["sha256"],
                               "split_component": seq.get("split", role), "storage_volume": None})
                supervision[key] = {"source_path": seq["gt_path"], "sha256": seq["gt_sha256"]}
    # Deduplicate physical images shared by C-TAO dense and modal TAO train.
    # Their annotations stay in the separate supervision source inventory.
    by_video: dict[tuple[str, str], set[str]] = defaultdict(set)
    for stream, role, path in (("ctao", "train", ROOT / "data/C-TAO/ctao_base.json"),
                              ("tao_train", "train", ROOT / "data/TAO-Amodal/annotations/train.json"),
                              ("tao_val", "dev", ROOT / "data/TAO-Amodal/annotations/validation.json")):
        data = json.loads(path.read_text())
        for image in data["images"]:
            relative = Path(image["file_name"])
            expected_prefix = "train" if role == "train" else "val"
            if relative.parts[0] != expected_prefix:
                raise ValueError(f"{stream} frame is outside its registered source split")
            by_video[(role, str(relative.parent))].add(relative.name)
        sources[stream] = {"path": str(path), "sha256": digest(path), "source_images": len(data["images"])}
        del data
    for (role, video), frames in sorted(by_video.items()):
        groups.append({"key": f"ovmot/{role}/{video}", "dataset": "ovmot", "role": role,
                       "video": video, "image_dir": str(ROOT / "data/TAO-Amodal/frames" / video),
                       "frames": sorted(frames), "vocabulary": "full_lvis_v1",
                       "split_component": "dense_ctao_plus_modal_tao_train" if role == "train" else "tao_validation"})
    volumes = [0] * 4
    for group in sorted((group for group in groups if group["dataset"] == "ovmot"),
                        key=lambda group: (-len(group["frames"]), group["key"])):
        volume = min(range(4), key=lambda index: (volumes[index], index))
        group["storage_volume"] = volume
        volumes[volume] += len(group["frames"])
    groups.sort(key=lambda group: group["key"])
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    contents = "".join(json.dumps(group, sort_keys=True) + "\n" for group in groups)
    if PLAN.exists() and PLAN.read_text() != contents:
        raise ValueError("The frozen cache frame inventory differs")
    PLAN.write_text(contents)
    counts = defaultdict(lambda: {"videos": 0, "frames": 0})
    for group in groups:
        count = counts[f"{group['dataset']}/{group['role']}"]
        count["videos"] += 1
        count["frames"] += len(group["frames"])
    vocabulary_path = PROTOCOL / "lvis_v1_vocabulary.json"
    summary = {"format": "locatemot-u2-label-free-cache-inventory-v1", "status": "PLAN_ONLY_NOT_EXECUTED",
               "inventory_path": str(PLAN), "inventory_sha256": digest(PLAN), "counts": dict(counts),
               "sources": sources, "shard_frames": SHARD_FRAMES,
               "ovmot_storage_volume_frames": volumes,
               "ovmot_storage_roots": [f"/data{index}/user/LocateMOT_U2_perception_cache" for index in range(4)],
               "ovmot_physical_image_deduplication": "C-TAO dense and modal TAO train use one cache per original file_name",
               "vocabulary_path": str(vocabulary_path), "vocabulary_sha256": digest(vocabulary_path),
               "feature_storage_dtype": "bfloat16", "box_score_storage_dtype": "float32",
               "gt_ids_in_feature_payload": False, "formal_cache_started": False,
               "official_test_labels_read": False}
    write_json(DIRECTORY / "inventory_summary.json", summary)
    write_json(DIRECTORY / "supervision_source_inventory.json", {"ordinary": supervision,
               "ovmot": {key: sources[key] for key in ("ctao", "tao_train", "tao_val")},
               "novel_class_or_identity_supervision_automatically_enabled": False})
    return summary


def group_directory(group: dict, spec: dict) -> Path:
    sha = spec["foundation_sha256"]
    if group["dataset"] == "ovmot":
        return Path(spec["ovmot_storage_roots"][group["storage_volume"]]) / sha / group["key"]
    return DIRECTORY / sha / group["key"]


def assigned_groups(groups: list[dict], count: int) -> list[list[dict]]:
    work, totals = [[] for _ in range(count)], [0] * count
    for group in sorted(groups, key=lambda group: (-len(group["frames"]) * (38 if group["dataset"] == "ovmot" else 1), group["key"])):
        rank = min(range(count), key=lambda index: (totals[index], index))
        work[rank].append(group)
        totals[rank] += len(group["frames"]) * (38 if group["dataset"] == "ovmot" else 1)
    return work


def worker(args: argparse.Namespace) -> int:
    import torch
    from locatemot.unified.runtime.grounding_inference import load_grounding_runtime
    from locatemot.unified.runtime.u2_grounding_candidates import run_u2_candidates
    from locatemot.unified.runtime.u2_frozen_text_cache import frozen_text_cache

    torch.set_num_threads(2)
    spec_path = args.spec
    spec = json.loads(spec_path.read_text())
    if digest(PLAN) != spec["inventory_sha256"]:
        raise ValueError("Cache worker frame inventory differs from selection")
    all_groups = [json.loads(line) for line in PLAN.read_text().splitlines()]
    groups = assigned_groups([group for group in all_groups if (group["dataset"] == "ovmot") == (spec["phase"] == "ovmot")],
                             spec["worker_count"])[args.worker_index]
    runtime = load_grounding_runtime(checkpoint=Path(spec["checkpoint"]), device=f"cuda:{args.gpu}", max_per_img=900)
    if runtime.checkpoint_sha256 != spec["foundation_sha256"] or runtime.missing_keys or runtime.unexpected_keys:
        raise ValueError("Cache worker loaded a different/incomplete foundation")
    categories = json.loads(Path(spec["vocabulary_path"]).read_text())["categories"]
    vocabulary = [category["name"] for category in categories]
    manifests = []
    with frozen_text_cache(runtime) as text_stats:
        for group in groups:
            directory = group_directory(group, spec)
            directory.mkdir(parents=True, exist_ok=True)
            k = spec["candidate_caps"].get(group["dataset"], 300)
            caption = vocabulary if group["dataset"] == "ovmot" else group["vocabulary"]
            units = []
            for start in range(0, len(group["frames"]), SHARD_FRAMES):
                frames = group["frames"][start:start + SHARD_FRAMES]
                path = directory / f"frames_{start:06d}.pt"
                expected = {"format": "locatemot-u2-label-free-video-shard-v1", "group_key": group["key"],
                            "foundation_sha256": spec["foundation_sha256"], "cache_spec_sha256": digest(spec_path),
                            "frames": frames, "max_candidates": k}
                if path.exists():
                    loaded = torch.load(path, map_location="cpu", weights_only=False)
                    if loaded["header"] != expected or [item["frame"] for item in loaded["frames"]] != frames:
                        raise ValueError(f"Existing cache unit has different weights or scope: {path}")
                    del loaded
                else:
                    payload = []
                    for frame in frames:
                        candidates = run_u2_candidates(runtime, Path(group["image_dir"]) / frame, caption, max_candidates=k)
                        # Semantic/decoder fields share the same serialized storage.
                        decoder = candidates["decoder_query"].to(device="cpu", dtype=torch.bfloat16)
                        cpu = {key: value.cpu() for key, value in candidates.items() if isinstance(value, torch.Tensor)
                               and key not in ("id_visual", "semantic_visual", "decoder_query")}
                        cpu.update(id_visual=candidates["id_visual"].to(device="cpu", dtype=torch.bfloat16),
                                   semantic_visual=decoder, decoder_query=decoder)
                        metadata = dict(candidates["source_metadata"])
                        metadata.pop("vocabulary")
                        payload.append({"dataset": group["dataset"], "video": group["video"], "frame": frame,
                                        "candidates": cpu, "source_metadata": metadata})
                    temporary = path.with_suffix(".tmp")
                    torch.save({"header": expected, "frames": payload}, temporary)
                    temporary.replace(path)
                units.append({"path": str(path), "sha256": digest(path), "frames": frames})
            manifest = {"group": group, "foundation_sha256": spec["foundation_sha256"], "units": units}
            write_json(directory / "manifest.json", manifest)
            manifests.append(manifest)
            print(json.dumps({"worker": args.worker_index, "video_complete": group["key"], "frames": len(group["frames"])}), flush=True)
    write_json(spec_path.parent / f"worker_{args.worker_index}.json", {"status": "COMPLETE", "worker_index": args.worker_index,
               "worker_count": spec["worker_count"], "cache_spec_sha256": digest(spec_path), "groups": manifests,
               "frozen_text_cache": text_stats})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--phase", choices=("ordinary", "ovmot"), default="ordinary")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.worker_index is not None:
        return worker(args)
    if args.plan_only:
        print(json.dumps(inventory(), indent=2))
        return 0
    if args.checkpoint is None or args.selection is None:
        parser.error("Full cache construction requires the passing selection and selected checkpoint")
    sha = digest(args.checkpoint)
    selected = verify_selected_foundation(args.selection, sha)
    summary = json.loads((DIRECTORY / "inventory_summary.json").read_text())
    if digest(PLAN) != summary["inventory_sha256"]:
        raise ValueError("The frozen frame inventory differs")
    for source in summary["sources"].values():
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("Cache source/protocol differs from its frozen inventory")
    if digest(Path(summary["vocabulary_path"])) != summary["vocabulary_sha256"]:
        raise ValueError("Full LVIS vocabulary differs from registration")
    portability = json.loads(Path(selected["sources"]["portability"]["path"]).read_text())
    caps = {name: item.get("candidate_top_k", 150) for name, item in portability["datasets"].items()
            if item["kind"] == "ordinary_mot"}
    spec = {"format": "locatemot-u2-selected-candidate-cache-v1", "foundation_sha256": sha,
            "checkpoint": str(args.checkpoint.resolve()), "selection": str(args.selection.resolve()),
            "selection_sha256": digest(args.selection), "phase": args.phase,
            "worker_count": len(args.gpus), "candidate_caps": caps,
            "frozen_text_cache": {"mode": "worker-local exact native BERT and token-prompt reuse; cloned return values",
                                  "implementation_sha256": digest(ROOT / "locatemot/unified/runtime/u2_frozen_text_cache.py"),
                                  "reference_validation_sha256": digest(ROOT / "outputs/unified/u2/eval/frozen_text_cache_contract.json")},
            **{key: summary[key] for key in ("inventory_sha256", "ovmot_storage_roots", "vocabulary_path", "vocabulary_sha256",
                                           "feature_storage_dtype", "box_score_storage_dtype", "shard_frames")}}
    directory = DIRECTORY / sha / args.phase
    spec_path = directory / "cache_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("An existing cache run uses a different registered foundation/scope")
    write_json(spec_path, spec)
    processes = []
    for index, gpu in enumerate(args.gpus):
        log = directory / f"worker_{index}.log"
        handle = log.open("a")
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker-index", str(index),
                                    "--gpu", str(gpu), "--spec", str(spec_path)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        processes.append((index, process, log))
    # Exactly one blocking parent owns these workers. A real failed worker
    # stops only this parent's other workers; every complete shard is retained.
    with ThreadPoolExecutor(max_workers=len(processes)) as executor:
        futures = {executor.submit(process.wait): (index, log) for index, process, log in processes}
        for future in as_completed(futures):
            index, log = futures[future]
            if future.result() != 0:
                for _, process, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                raise RuntimeError(f"Cache worker {index} failed; inspect its first error: {log}")
    groups = [json.loads(line) for line in PLAN.read_text().splitlines()]
    expected_groups = [group for group in groups if (group["dataset"] == "ovmot") == (args.phase == "ovmot")]
    completed = []
    text_reports = []
    for index in range(len(args.gpus)):
        report = json.loads((directory / f"worker_{index}.json").read_text())
        if report["status"] != "COMPLETE" or report["cache_spec_sha256"] != digest(spec_path):
            raise ValueError("Cache worker did not complete the current registered scope")
        completed.extend(report["groups"])
        text_reports.append({"worker_index": index, **report["frozen_text_cache"]})
    if sorted(group["group"]["key"] for group in completed) != sorted(group["key"] for group in expected_groups):
        raise ValueError("Candidate cache omits or duplicates registered videos")
    for result in completed:
        if ([frame for unit in result["units"] for frame in unit["frames"]] != result["group"]["frames"]
                or result["foundation_sha256"] != sha):
            raise ValueError("Candidate cache omits frames or mixes foundations")
    manifest = {"format": "locatemot-u2-complete-label-free-cache-v1", "status": "COMPLETE",
                "spec": spec, "cache_spec_sha256": digest(spec_path), "groups": completed,
                "frames": sum(len(group["frames"]) for group in expected_groups),
                "gt_ids_in_feature_payload": False, "official_test_labels_read": False,
                "frozen_text_cache_workers": text_reports}
    manifest_path = directory / "complete_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "manifest": str(manifest_path), "candidate_cache_sha256": digest(manifest_path),
                      "frames": manifest["frames"], "phase": args.phase}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
