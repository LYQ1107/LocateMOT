"""Complete ordinary+TAO evaluation and one-point joint regression gates."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path

from tools.unified.build_u2_candidate_cache import assigned_groups, digest, write_json


def checkpoint_decision(ordinary: dict, ovmot: dict, reference: dict) -> dict:
    from locatemot.unified.data.u2_mot_clips import DATASET_PROBABILITIES
    if (ordinary["status"] != "COMPLETE" or ovmot["status"] != "COMPLETE"
            or set(ordinary["datasets"]) != set(DATASET_PROBABILITIES)
            or ordinary["candidate_cache_sha256"] != reference["candidate_cache_sha256"]
            or ordinary["foundation_sha256"] != reference["foundation_sha256"]):
        raise ValueError("Joint negative transfer requires complete same-candidate ordinary evaluations")
    deltas, retained_by_dataset = {}, {}
    for dataset in DATASET_PROBABILITIES:
        actual, frozen = ordinary["datasets"][dataset], reference["datasets"][dataset]
        if actual["status"] != "COMPLETE" or any(actual[key] != frozen[key] for key in
                ("frames_evaluated", "sequences_evaluated", "gt_detections", "gt_tracks")):
            raise ValueError(f"Joint {dataset} evaluation changed the frozen ordinary reference scope")
        deltas[dataset] = {key: actual["metrics"][key] - frozen["metrics"][key] for key in ("HOTA", "AssA")}
        # Exact recorded decimal thresholds, without tolerance or gate drift.
        retained_by_dataset[dataset] = all(Decimal(str(actual["metrics"][key])) >=
                                           Decimal(str(frozen["metrics"][key])) - Decimal(".01")
                                           for key in ("HOTA", "AssA"))
    ordinary_retained = all(retained_by_dataset.values())
    overall = ovmot["metrics"]["overall"]
    main = overall["TETA"] >= 34. and overall["AssocA"] >= 37.
    strong = overall["TETA"] >= 36. and overall["AssocA"] >= 40.
    refinement = ordinary_retained and not main and overall["TETA"] >= 31. and overall["AssocA"] >= 34.
    return {"eligible": ordinary_retained and main, "ordinary_retained": ordinary_retained,
            "ordinary_delta": deltas, "ovmot_gate_pass": main, "u2_strong": ordinary_retained and strong,
            "bounded_u2_c_eligible": refinement,
            "failed_ordinary_datasets": [name for name, retained in retained_by_dataset.items() if not retained],
            "ordinary_max_degradation": .01, "ovmot_required": {"TETA": 34., "AssocA": 37.}}


def evaluate_checkpoint(core, mot_sampler, ov_sampler, ordinary_cache: dict, ovmot_cache: dict,
                        mot_supervision: dict, class_text, reference: dict, args, step: int,
                        checkpoint: Path) -> dict | None:
    import torch
    import torch.distributed as dist
    from locatemot.unified.runtime.u2_teta_evaluation import evaluate_tao_teta, ovmot_frame_prediction, write_ovmot_predictions
    from tools.unified.eval_u2_mot_only import evaluate_checkpoint as ordinary_evaluate
    rank, world = dist.get_rank(), dist.get_world_size()
    directory = args.out / f"evaluation_{step:04d}"
    result_path = directory / "result.json"
    provenance = {"checkpoint_sha256": digest(checkpoint), "foundation_sha256": ordinary_cache["spec"]["foundation_sha256"],
                  "ordinary_cache_sha256": digest(args.cache_manifest), "ovmot_cache_sha256": digest(args.ovmot_cache),
                  "class_text_sha256": digest(args.class_text), "mot_only_checkpoint_sha256": reference["checkpoint_sha256"],
                  "tao_gt_view_sha256": digest(args.tao_gt_view)}
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result["status"] != "COMPLETE" or any(result[key] != value for key, value in provenance.items()):
            raise ValueError("Joint evaluation differs from its frozen checkpoint/reference/features")
        dist.barrier()
        return result if rank == 0 else None
    ordinary_args = argparse.Namespace(out=directory / "ordinary", cache_manifest=args.cache_manifest,
                                       simple_baseline=args.simple_baseline)
    ordinary = ordinary_evaluate(core, mot_sampler, ordinary_cache, mot_supervision, ordinary_args, step, checkpoint,
                                 tracker_name="U2_JOINT_ORDINARY", report_format="locatemot-u2-joint-ordinary-evaluation-v1")
    gt_view = json.loads(args.tao_gt_view.read_text())
    gt_path = Path(gt_view["view_path"])
    # The original image IDs/timeline are evaluation metadata, not labels fed
    # into the causal core. No base/novel GT boxes/classes are read as features.
    images = {image["file_name"]: image for image in json.loads(gt_path.read_text())["images"]}
    groups = [item["group"] for item in ovmot_cache["groups"] if item["group"]["role"] == "dev"]
    if {f"{group['video']}/{frame}" for group in groups for frame in group["frames"]} != set(images):
        raise ValueError("Joint TAO evaluation must cover every original validation image exactly")
    reports = []
    core.eval()
    with torch.no_grad():
        for group in assigned_groups(groups, world)[rank]:
            path = directory / "ovmot/predictions" / f"{group['video']}.json"
            contract_path = path.with_suffix(".contract.json")
            if contract_path.is_file():
                contract = json.loads(contract_path.read_text())
                if (contract["provenance"] != provenance or contract["frames"] != len(group["frames"])
                        or contract["sha256"] != digest(path)):
                    raise ValueError("Completed OVMOT evaluation unit differs; no silent replacement/skip")
                reports.append(contract)
                continue
            state, frames = None, []
            for position, name in enumerate(group["frames"]):
                candidates = ov_sampler.load_candidates(group, position, next(core.parameters()).device)
                physical = images[f"{group['video']}/{name}"]["frame_index"]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = core(candidates, state, task_type="ovmot", class_text=class_text, frame_index=physical)
                    frame = ovmot_frame_prediction(core, output, candidates, class_text)
                state = output["state"]
                frames.append({"frame": name, **frame})
            contract = write_ovmot_predictions(group, frames, images, path)
            contract["provenance"] = provenance
            write_json(contract_path, contract)
            reports.append(contract)
    core.train()
    write_json(directory / f"ovmot/worker_{rank}.json", {"provenance": provenance, "groups": reports})
    dist.barrier()
    result = None
    if rank == 0:
        completed = []
        for worker in range(world):
            report = json.loads((directory / f"ovmot/worker_{worker}.json").read_text())
            if report["provenance"] != provenance:
                raise ValueError("TAO worker inference used another checkpoint")
            completed.extend(report["groups"])
        if (sorted(report["video"] for report in completed) != sorted(group["video"] for group in groups)
                or sum(report["frames"] for report in completed) != len(images)):
            raise ValueError("TAO evaluation omits or repeats an original validation unit")
        # Stream the complete official prediction array instead of holding all
        # ~10M detections plus every GPU rank's feature cache in parent RAM.
        prediction_path = directory / "ovmot/predictions.json"
        with prediction_path.open("w") as handle:
            handle.write("[")
            separator = ""
            for report in sorted(completed, key=lambda report: report["video"]):
                if digest(Path(report["path"])) != report["sha256"]:
                    raise ValueError("A completed TAO prediction unit changed")
                for row in json.loads(Path(report["path"]).read_text()):
                    handle.write(separator + json.dumps(row))
                    separator = ","
            handle.write("]\n")
        ovmot = evaluate_tao_teta(gt_path, prediction_path, directory / "ovmot/teta", "U2_JOINT_OVMOT", workers=8)
        if (ovmot["frames_evaluated"] != gt_view["view_images"] or ovmot["videos_evaluated"] != gt_view["view_videos"]
                or ovmot["gt_units"] != gt_view["view_gt_units"]):
            raise ValueError("Official TETA changed the registered full original v1 validation scope")
        decision = checkpoint_decision(ordinary, ovmot, reference)
        result = {"format": "locatemot-u2-joint-evaluation-v1", "status": "COMPLETE", "step": step,
                  "checkpoint": str(checkpoint.resolve()), **provenance, "ordinary": ordinary, "ovmot": ovmot,
                  "decision": decision, "source_gt_mapping": gt_view,
                  "official_test_labels_read": False, "foundation_weights_modified": False}
        write_json(result_path, result)
        print(json.dumps({"joint_evaluation_step": step, "ordinary_macro": ordinary["macro"],
                          "ovmot": ovmot["metrics"], "eligible": decision["eligible"]}), flush=True)
    dist.barrier()
    return result


def select_joint(directory: Path, reference: dict, metadata: dict) -> dict:
    from tools.unified.train_u2_joint import EVALUATION_STEPS
    reports = []
    for step in EVALUATION_STEPS:
        report = json.loads((directory / f"evaluation_{step:04d}/result.json").read_text())
        if (report["status"] != "COMPLETE" or report["step"] != step
                or report["checkpoint_sha256"] != digest(Path(report["checkpoint"]))
                or any(report[key] != metadata[key] for key in ("foundation_sha256", "ordinary_cache_sha256",
                        "ovmot_cache_sha256", "class_text_sha256", "mot_only_checkpoint_sha256", "tao_gt_view_sha256"))):
            raise ValueError("Joint selection requires all four complete original evaluation boundaries")
        if report["decision"] != checkpoint_decision(report["ordinary"], report["ovmot"], reference):
            raise ValueError("Joint checkpoint decision differs from its measured individual dataset gates")
        reports.append(report)
    eligible = [report for report in reports if report["decision"]["eligible"]]
    order = lambda report: (report["ovmot"]["metrics"]["overall"]["AssocA"],
                            report["ovmot"]["metrics"]["overall"]["TETA"],
                            report["ordinary"]["macro"]["AssA"], -report["step"])
    selected = max(eligible, key=order) if eligible else None
    refinement = [report for report in reports if report["decision"]["bounded_u2_c_eligible"]]
    if selected:
        status = "U2_B_JOINT_SELECTED"
    elif refinement:
        status = "U2_C_BOUNDED_REFINEMENT_ELIGIBLE"
    elif any(report["decision"]["ordinary_retained"] for report in reports):
        status = "LOCATEMOT_U_BLOCKED_U2_OVMOT"
    else:
        status = "LOCATEMOT_U_BLOCKED_U2_NEGATIVE_TRANSFER"
    result = {"format": "locatemot-u2-joint-selection-v1", "status": status, "optimizer_steps": 4000,
              "selected": selected, "refinement_initialization": max(refinement, key=order) if not selected and refinement else None,
              "decisions": [{"step": report["step"], **report["decision"]} for report in reports],
              "u2_strong": selected["decision"]["u2_strong"] if selected else False,
              "mot_only_reference_sha256": reference["checkpoint_sha256"],
              "ready_for_u3": False, "pending": "actual self-contained checkpoint and integration retention audit" if selected else status,
              "foundation_weights_modified": False, "official_test_labels_read": False}
    write_json(directory / "selection.json", result)
    return result
