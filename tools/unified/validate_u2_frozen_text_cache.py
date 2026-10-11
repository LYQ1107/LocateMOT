#!/usr/bin/env python3
"""Actual two-frame native/cached MOT and full-LVIS OVMOT parity/timing."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    from locatemot.unified.runtime.grounding_inference import load_grounding_runtime
    from locatemot.unified.runtime.u2_grounding_candidates import run_u2_candidates
    from locatemot.unified.runtime.u2_frozen_text_cache import frozen_text_cache
    from tools.unified.u2_mot_portability import CHECKPOINT, FOUNDATION_SHA
    from tools.unified.build_u2_candidate_cache import write_json

    torch.set_num_threads(2)
    runtime = load_grounding_runtime(checkpoint=CHECKPOINT, device=args.device, max_per_img=900)
    assert runtime.checkpoint_sha256 == FOUNDATION_SHA and not runtime.missing_keys and not runtime.unexpected_keys
    protocols = ROOT / "outputs/unified/u2/protocol"
    mot = json.loads((protocols / "mot17_half.json").read_text())
    sequence = mot["sequences"][0]
    mot_images = [Path(sequence["image_dir"]) / frame for frame in sequence["validation_frames"][:2]]
    tao = json.loads((protocols / "tao_portability.json").read_text())
    annotation = json.loads(Path(tao["annotation_path"]).read_text())
    tao_images = [Path(tao["image_root"]) / row["file_name"] for row in annotation["images"][:2]]
    del annotation
    entities = [c["name"] for c in json.loads((protocols / "lvis_v1_vocabulary.json").read_text())["categories"]]
    counters = {"bert": 0, "backbone": 0, "decoder": 0}

    def counter(name):
        def record(module, inputs, output):
            counters[name] += 1
        return record

    hooks = [runtime.model.language_model.language_backbone.body.model.register_forward_hook(counter("bert")),
             runtime.model.backbone.register_forward_hook(counter("backbone")),
             runtime.model.decoder.register_forward_hook(counter("decoder"))]
    fields = ("boxes", "scores", "id_visual", "semantic_visual", "decoder_query", "predicted_class", "query_index", "chunk_index")
    reports = []
    native_cases = []

    def extract(image, vocabulary, cap):
        torch.cuda.synchronize(runtime.device)
        start = time.perf_counter()
        result = run_u2_candidates(runtime, image, vocabulary, max_candidates=cap)
        torch.cuda.synchronize(runtime.device)
        seconds = time.perf_counter() - start
        return {name: result[name].detach().cpu() for name in fields}, seconds

    try:
        for task, images, vocabulary, cap, chunks in [("mot", mot_images, mot["vocabulary"], 150, 1),
                                                    ("ovmot", tao_images, entities, 300, 38)]:
            extract(images[0], vocabulary, cap)  # Equal native GPU warm-up before timing.
            counters.update(bert=0, backbone=0, decoder=0)
            baseline, native_times = [], []
            for image in images:
                tensors, elapsed = extract(image, vocabulary, cap)
                baseline.append(tensors)
                native_times.append(elapsed)
            native_counts = dict(counters)
            counters.update(bert=0, backbone=0, decoder=0)
            cached_times = []
            with frozen_text_cache(runtime) as stats:
                for image, native in zip(images, baseline):
                    cached, elapsed = extract(image, vocabulary, cap)
                    assert all(torch.equal(native[name], cached[name]) for name in fields), task
                    cached_times.append(elapsed)
                inference_counts = dict(counters)
                inference_stats = dict(stats)
                assert native_counts == {"bert": 2 * chunks, "backbone": 2, "decoder": 2 * chunks}
                assert inference_counts == {"bert": chunks, "backbone": 2, "decoder": 2 * chunks}
                assert inference_stats["bert_calls"] == chunks and inference_stats["bert_hits"] == chunks
                previous = runtime.model.test_cfg.get("chunked_size", -1)
                runtime.model.test_cfg.chunked_size = 32 if task == "ovmot" else -1
                try:
                    maps, captions, _, _ = runtime.model.get_tokens_positive_and_prompts(vocabulary, True)
                    original_maps, original_captions = copy.deepcopy(maps), copy.deepcopy(captions)
                    if task == "ovmot":
                        maps[0].clear()
                        captions[0] = "mutated"
                    else:
                        maps.clear()
                    fresh = runtime.model.get_tokens_positive_and_prompts(vocabulary, True)
                    assert fresh[0] == original_maps and fresh[1] == original_captions
                    prompt = original_captions[0] if task == "ovmot" else original_captions
                    with torch.inference_mode():
                        output = runtime.model.language_model([prompt])
                        unchanged = {name: value.clone() for name, value in output.items()}
                        for value in output.values():
                            value.zero_()
                        restored = runtime.model.language_model([prompt])
                        assert all(torch.equal(unchanged[name], restored[name]) for name in unchanged)
                    assert counters == inference_counts, "Mutation probes unexpectedly re-encoded BERT"
                finally:
                    runtime.model.test_cfg.chunked_size = previous
            assert all(name not in module.__dict__ for module, name in
                       [(runtime.model.language_model, "forward"), (runtime.model, "get_tokens_and_prompts"),
                        (runtime.model, "get_tokens_positive_and_prompts")])
            report = {"task": task, "images": [str(p) for p in images], "frames": 2, "candidate_cap": cap,
                      "chunks_per_frame": chunks, "all_candidate_features": "BIT_EXACT",
                      "native_calls": native_counts, "cached_calls": inference_counts,
                      "cache_counters_before_mutation_probe": inference_stats,
                      "tensor_and_positive_map_mutation_isolation": "PASS", "methods_restored": True,
                      "native_seconds_per_frame": native_times, "cached_seconds_per_frame": cached_times,
                      "observed_two_frame_speed_ratio": sum(native_times) / sum(cached_times),
                      "observed_warm_frame_speed_ratio": native_times[1] / cached_times[1]}
            reports.append(report)
            native_cases.append((task, images, vocabulary, cap, baseline))
            print(json.dumps(report), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    # Execute the real worker and serializer on an explicitly separate tiny
    # reference inventory, then verify completed-unit resume does no inference.
    from tools.unified import build_u2_candidate_cache as builder
    from locatemot.unified.runtime import grounding_inference, u2_grounding_candidates
    worker_cases = []
    with tempfile.TemporaryDirectory(prefix="locatemot-u2-text-worker-contract-") as directory:
        temporary = Path(directory)
        for task, images, vocabulary, cap, native in native_cases:
            phase = "ovmot" if task == "ovmot" else "ordinary"
            group = {"dataset": "ovmot" if task == "ovmot" else "mot17", "key": "REFERENCE_ONLY_" + task,
                     "video": images[0].parent.parent.name if task == "mot" else images[0].parent.name,
                     "image_dir": str(images[0].parent),
                     "frames": [image.name for image in images], "vocabulary": vocabulary}
            assert all(image.parent == images[0].parent for image in images)
            plan = temporary / (phase + ".jsonl")
            plan.write_text(json.dumps(group) + "\n")
            spec_path = temporary / phase / "cache_spec.json"
            spec_path.parent.mkdir()
            spec = {"checkpoint": str(CHECKPOINT), "foundation_sha256": FOUNDATION_SHA,
                    "inventory_sha256": builder.digest(plan), "phase": phase, "worker_count": 1,
                    "vocabulary_path": str(protocols / "lvis_v1_vocabulary.json"),
                    "candidate_caps": {group["dataset"]: cap}, "reference_only": True}
            write_json(spec_path, spec)
            worker_args = argparse.Namespace(spec=spec_path, worker_index=0, gpu=runtime.device.index)
            location = temporary / phase / "reference_only_units"
            with patch.object(builder, "PLAN", plan), \
                    patch.object(builder, "group_directory", return_value=location), \
                    patch.object(grounding_inference, "load_grounding_runtime", return_value=runtime):
                assert builder.worker(worker_args) == 0
                unit = torch.load(location / "frames_000000.pt", weights_only=False)
                assert unit["header"]["frames"] == group["frames"]
                assert unit["header"]["cache_spec_sha256"] == builder.digest(spec_path)
                for item, original in zip(unit["frames"], native):
                    stored = item["candidates"]
                    for name in fields:
                        expected = original[name].to(torch.bfloat16) if name in ("id_visual", "semantic_visual", "decoder_query") else original[name]
                        assert torch.equal(stored[name], expected), (task, name, "worker serialization")
                    assert stored["semantic_visual"].data_ptr() == stored["decoder_query"].data_ptr()
                    assert not {"gt_ids", "gt_classes", "gt_boxes", "track_ids"}.intersection(stored)
                first = json.loads((spec_path.parent / "worker_0.json").read_text())
                with patch.object(u2_grounding_candidates, "run_u2_candidates", side_effect=AssertionError("Completed shard was rerun")):
                    assert builder.worker(worker_args) == 0
                resumed = json.loads((spec_path.parent / "worker_0.json").read_text())
                assert first["groups"] == resumed["groups"]
                assert all(value == 0 for value in resumed["frozen_text_cache"].values())
            worker_cases.append({"task": task, "reference_only_frames": 2, "serialized_all_features": "BIT_EXACT",
                                 "semantic_decoder_storage_shared": True, "gt_labels_in_payload": False,
                                 "complete_shard_resume": "NO_RERUN", "first_run_text_counts": first["frozen_text_cache"]})
    report = {"format": "locatemot-u2-frozen-text-reuse-contract-v1", "status": "PASS",
              "foundation_sha256": FOUNDATION_SHA, "reference_only": True, "precision": "native FP32, no autocast",
              "cases": reports, "performance_scope": "two real frames per task on a shared GPU; not full-cache throughput",
              "actual_worker_serialization_and_resume": worker_cases,
              "image_and_multimodal_computations_unchanged": True, "vocabulary_or_candidate_caps_changed": False,
              "live_full_audits_modified_or_restarted": False, "formal_selected_candidate_cache_started": False,
              "benchmark_tracking_measured": False, "official_test_labels_read": False}
    write_json(ROOT / "outputs/unified/u2/eval/frozen_text_cache_contract.json", report)
    print(json.dumps({"status": "PASS", "report": "outputs/unified/u2/eval/frozen_text_cache_contract.json"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
