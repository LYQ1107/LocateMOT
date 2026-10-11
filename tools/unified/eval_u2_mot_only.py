"""All-five ordinary dev evaluation and frozen MOT-only checkpoint selection."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from tools.unified.build_u2_candidate_cache import assigned_groups, digest, write_json


def evaluate_checkpoint(core, sampler, cache: dict, supervision: dict, args, step: int, checkpoint: Path,
                        *, tracker_name: str = "U2_MOT_ONLY_BASELINE",
                        report_format: str = "locatemot-u2-mot-only-evaluation-v1") -> dict | None:
    import torch
    import torch.distributed as dist
    from locatemot.unified.data.u2_mot_clips import DATASET_PROBABILITIES, source_frame_index
    from locatemot.unified.runtime.u2_ordinary_evaluation import prepare_gt_view, write_predictions, evaluate_ordinary
    rank, world = dist.get_rank(), dist.get_world_size()
    directory = args.out / f"evaluation_{step:04d}"
    result_path = directory / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if (result["status"] != "COMPLETE" or result["checkpoint_sha256"] != digest(checkpoint)
                or result["candidate_cache_sha256"] != digest(args.cache_manifest)
                or set(result["datasets"]) != set(DATASET_PROBABILITIES)):
            raise ValueError("Existing evaluation belongs to a different scope/checkpoint")
        dist.barrier()
        return result if rank == 0 else None
    groups = [item["group"] for item in cache["groups"] if item["group"]["role"] == "dev"]
    tracker = tracker_name
    core.eval()
    with torch.no_grad():
        for group in assigned_groups(groups, world)[rank]:
            state, frames = None, []
            for position, name in enumerate(group["frames"]):
                candidates = sampler.load_candidates(group, position, next(core.parameters()).device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = core(candidates, state, task_type="mot", frame_index=source_frame_index(group, position))
                state, indices = output["state"], output["candidate_indices"]
                frames.append({"frame": name, "track_ids": output["track_ids"].tolist(),
                               "boxes": candidates["boxes"][indices].tolist(), "scores": candidates["scores"][indices].tolist(),
                               "predicted_class": candidates["predicted_class"][indices].tolist()})
            extension = ".json" if group["dataset"] == "bdd100k" else ".txt"
            path = directory / group["dataset"] / "predictions" / tracker / "data" / f"{group['video']}{extension}"
            write_predictions(group["dataset"], group, frames, path)
    core.train()
    dist.barrier()
    result = None
    if rank == 0:
        reports = {}
        for dataset in DATASET_PROBABILITIES:
            selected = [group for group in groups if group["dataset"] == dataset]
            gt_directory = directory / dataset / "gt_view"
            view = prepare_gt_view(dataset, selected, supervision, gt_directory)
            reports[dataset] = evaluate_ordinary(dataset, gt_directory, directory / dataset / "predictions",
                                                 view["sequence_lengths"], tracker)
            reference = json.loads(args.simple_baseline.read_text())["datasets"][dataset]
            if (reports[dataset]["gt_detections"] != reference["gt_detections"]
                    or reports[dataset]["gt_tracks"] != reference["gt_tracks"]):
                raise ValueError(f"Learned {dataset} evaluation changed the simple baseline GT scope")
            write_json(directory / dataset / "result.json", reports[dataset])
        macro = {key: sum(report["metrics"][key] for report in reports.values()) / 5
                 for key in ("HOTA", "DetA", "AssA", "IDF1")}
        result = {"format": report_format, "status": "COMPLETE", "step": step,
                  "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint),
                  "foundation_sha256": cache["spec"]["foundation_sha256"],
                  "candidate_cache_sha256": digest(args.cache_manifest), "task_type": "mot",
                  "datasets": reports, "macro": macro, "total_idsw": sum(report["metrics"]["IDSW"] for report in reports.values()),
                  "gt_tracks": sum(report["gt_tracks"] for report in reports.values()), "official_test_labels_read": False}
        write_json(result_path, result)
        print(json.dumps({"evaluation_step": step, "macro": macro}), flush=True)
    dist.barrier()
    return result


def select_mot_only(directory: Path, simple: dict, cache_manifest: Path) -> dict:
    from tools.unified.train_u2_mot_only import EVALUATION_STEPS
    from locatemot.unified.data.u2_mot_clips import DATASET_PROBABILITIES
    reports, decisions = [], []
    for step in EVALUATION_STEPS:
        report = json.loads((directory / f"evaluation_{step:04d}/result.json").read_text())
        if (report["status"] != "COMPLETE" or report["step"] != step
                or report["candidate_cache_sha256"] != digest(cache_manifest)
                or report["foundation_sha256"] != simple["foundation_sha256"]
                or set(report["datasets"]) != set(DATASET_PROBABILITIES)
                or report["checkpoint_sha256"] != digest(Path(report["checkpoint"]))):
            raise ValueError("MOT-only selection requires every original evaluation boundary on the same cache")
        for dataset in DATASET_PROBABILITIES:
            actual, reference = report["datasets"][dataset], simple["datasets"][dataset]
            if (actual["status"] != "COMPLETE" or any(actual[key] != reference[key] for key in
                    ("frames_evaluated", "sequences_evaluated", "gt_detections", "gt_tracks"))):
                raise ValueError(f"MOT-only selection differs from the registered {dataset} development scope")
        deltas = {dataset: {key: report["datasets"][dataset]["metrics"][key] - simple["datasets"][dataset]["metrics"][key]
                            for key in ("HOTA", "AssA")}
                  for dataset in DATASET_PROBABILITIES}
        # Compare the recorded decimal scores at the exact registered point
        # thresholds, without an epsilon or an extra rounding allowance.
        identity_gain = (Decimal(str(report["macro"]["AssA"])) >= Decimal(str(simple["macro"]["AssA"])) + Decimal(".03")
                         or report["macro"]["AssA"] >= .50)
        hota_retained = all(Decimal(str(report["datasets"][dataset]["metrics"]["HOTA"])) >=
                            Decimal(str(simple["datasets"][dataset]["metrics"]["HOTA"])) - Decimal(".02")
                            for dataset in DATASET_PROBABILITIES)
        eligible = identity_gain and hota_retained
        decisions.append({"step": step, "eligible": eligible, "identity_gain": identity_gain,
                          "all_dataset_hota_retained": hota_retained, "dataset_delta": deltas,
                          "macro": report["macro"]})
        if eligible:
            reports.append(report)
    selected = max(reports, key=lambda report: (report["macro"]["AssA"], report["macro"]["HOTA"],
                                               -report["total_idsw"] / max(1, report["gt_tracks"]), -report["step"])) if reports else None
    result = {"format": "locatemot-u2-mot-only-selection-v1", "status": "MOT_ONLY_SELECTED" if selected else "LOCATEMOT_U_BLOCKED_U2_IDENTITY_CORE",
              "decisions": decisions, "selected": selected, "optimizer_steps": 6000, "joint_training_launched": False,
              "negative_transfer_reference_frozen": selected is not None, "official_test_labels_read": False}
    write_json(directory / "selection.json", result)
    return result
