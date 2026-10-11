#!/usr/bin/env python3
"""Freeze the U2-P portability decision without inventing missing metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "outputs/unified/u2/protocol/data_manifest.json"
U1_RESULT = ROOT / "outputs/unified/u1r2/formal_generic_expression_epoch1/result.json"
FOUNDATION_SHA = "4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26"


def read_measurement(directory: Path, dataset: str, expected_foundation_sha256: str = FOUNDATION_SHA) -> dict | None:
    path = directory / f"{dataset}.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["dataset"] != dataset or result["foundation_sha256"] != expected_foundation_sha256:
        raise ValueError(f"Measurement provenance mismatch: {path}")
    if (result.get("status") != "COMPLETE" or result["images_evaluated"] <= 0
            or result["images_evaluated"] != result.get("images_expected")):
        raise ValueError(f"Incomplete measurement scope: {path}")
    protocol_path = Path(result["protocol_path"])
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != result["protocol_sha256"]:
        raise ValueError(f"Measurement protocol differs: {path}")
    protocol = json.loads(protocol_path.read_text())
    if dataset == "kitti_legal_development" and result["counts"]["gt_total"] != protocol["positive_gt_boxes"]:
        raise ValueError("Original KITTI audit does not retain every registered ordinary GT unit")
    expected_images = protocol["measurement_images"] if "measurement_images" in protocol else protocol["validation_images"]
    if protocol["dataset"] != dataset or expected_images != result["images_expected"]:
        raise ValueError(f"Measurement differs from registered frame scope: {path}")
    if dataset in ("mot17", "mot20", "dancetrack", "bdd100k", "kitti_legal_development"):
        keys = sorted(f"{dataset}/{sequence['sequence']}/{frame}"
                      for sequence in protocol["sequences"] for frame in sequence["validation_frames"])
        coverage_sha = hashlib.sha256("\n".join(keys).encode()).hexdigest()
        if result["coverage_keys_sha256"] != coverage_sha:
            raise ValueError(f"Incomplete registered frame coverage: {path}")
    elif dataset in ("tao", "ctao"):
        if (result["coverage_keys_sha256"] != protocol["coverage_keys_sha256"]
                or not result["every_frame_backbone_calls_one"]):
            raise ValueError(f"Incomplete TAO scope or backbone-reuse contract: {path}")
    return result


def recall(hits: int, total: int) -> float | None:
    if total == 0:
        return None
    if not 0 <= hits <= total:
        raise ValueError(f"Invalid recall numerator/denominator: {hits}/{total}")
    return hits / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--measurements-dir", type=Path, default=ROOT / "outputs/unified/u2/perception/measurements")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/unified/u2/perception/portability.json")
    parser.add_argument("--expected-foundation-sha256", default=FOUNDATION_SHA)
    parser.add_argument("--u1-generic-result", type=Path, default=U1_RESULT)
    parser.add_argument("--crowded-registration", type=Path)
    args = parser.parse_args()
    if args.expected_foundation_sha256 != FOUNDATION_SHA:
        if (args.out.resolve() == parser.get_default("out").resolve()
                or args.measurements_dir.resolve() == parser.get_default("measurements_dir").resolve()
                or args.u1_generic_result.resolve() == U1_RESULT.resolve()):
            parser.error("Adapted foundations require their own measurements, U1 generic result, and output")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    u1 = json.loads(args.u1_generic_result.read_text(encoding="utf-8"))
    if (u1["status"] != "FORMAL_COMPLETE"
            or u1["runtime"]["checkpoint_sha256"] != args.expected_foundation_sha256):
        raise ValueError("KITTI development metrics do not belong to the measured foundation")
    metrics = u1["metrics"]

    kitti = {
        "source": str(args.u1_generic_result),
        "protocol": "legal Refer-KITTI calibration/validation scope; V1 and V2 are reported separately",
        "domains": {
            "refer_kitti_v1": {
                "top150_recall_iou050": metrics["refer_kitti_v1"]["150"]["0.5"]["recall"],
                "top300_recall_iou050": metrics["refer_kitti_v1"]["300"]["0.5"]["recall"],
            },
            "refer_kitti_v2": {
                "top150_recall_iou050": metrics["refer_kitti_v2"]["150"]["0.5"]["recall"],
                "top300_recall_iou050": metrics["refer_kitti_v2"]["300"]["0.5"]["recall"],
            },
        },
        "top150_minimum_pass": all(metrics[domain]["150"]["0.5"]["recall"] >= .85
                                   for domain in ("refer_kitti_v1", "refer_kitti_v2")),
        "top150_preferred_pass": all(metrics[domain]["150"]["0.5"]["recall"] >= .90
                                     for domain in ("refer_kitti_v1", "refer_kitti_v2")),
        "metrics_status": "MEASURED_U1_REFERENCE" if args.expected_foundation_sha256 == FOUNDATION_SHA else "MEASURED_ADAPTED_FOUNDATION",
        "top300_registered": False,
    }
    original_kitti = read_measurement(args.measurements_dir, "kitti_legal_development", args.expected_foundation_sha256)
    kitti["u1_reference_top150_minimum_pass"] = kitti["top150_minimum_pass"]
    kitti["original_gt_metrics_status"] = "NOT_MEASURED" if original_kitti is None else "MEASURED"
    kitti.update(original_gt_measurement=original_kitti,
                 top150_recall_iou050=None if original_kitti is None else recall(original_kitti["counts"]["hits_top150"], original_kitti["counts"]["gt_total"]),
                 top300_recall_iou050=None if original_kitti is None else recall(original_kitti["counts"]["hits_top300"], original_kitti["counts"]["gt_total"]))
    kitti["top150_minimum_pass"] = (kitti["top150_minimum_pass"] and original_kitti is not None
                                    and kitti["top150_recall_iou050"] >= .85)
    kitti["top150_preferred_pass"] = (kitti["top150_preferred_pass"] and original_kitti is not None
                                     and kitti["top150_recall_iou050"] >= .90)
    kitti["measurement"] = original_kitti
    datasets: dict[str, dict] = {}
    for item in manifest["datasets"]:
        name = item["dataset"]
        if name == "kitti_legal_development":
            datasets[name] = {**item, **kitti}
        else:
            datasets[name] = {
                **item,
                "top150_recall_iou050": None,
                "top300_recall_iou050": None,
                "metrics_status": "NOT_MEASURED" if item["status"] == "AVAILABLE" else "NOT_MEASURED_MISSING_LOCAL_DATA",
            }
            measured = read_measurement(args.measurements_dir, name, args.expected_foundation_sha256)
            if measured is not None:
                datasets[name].update(metrics_status="MEASURED", measurement=measured)
                if "candidate_top_k" in measured:
                    datasets[name]["candidate_top_k"] = measured["candidate_top_k"]
                counts = measured["counts"]
                if name not in ("tao", "ctao"):
                    datasets[name]["top150_recall_iou050"] = recall(counts["hits_top150"], counts["gt_total"])
                    datasets[name]["top300_recall_iou050"] = recall(counts["hits_top300"], counts["gt_total"])
                else:
                    datasets[name]["base_recall_iou050"] = recall(counts["base_hits_top300"], counts["base_gt_total"])
                    datasets[name]["novel_recall_iou050"] = recall(counts["novel_hits_top300"], counts["novel_gt_total"])
    if args.crowded_registration:
        registration = json.loads(args.crowded_registration.read_text())
        measurement_path = args.measurements_dir / "mot20.json"
        if (registration["foundation_sha256"] != args.expected_foundation_sha256
                or registration["measurement_sha256"] != hashlib.sha256(measurement_path.read_bytes()).hexdigest()
                or registration["selected_top_k"] != 300
                or registration["minimum_recall"] != .90
                or not registration["before_track_training"]
                or datasets["mot20"]["top300_recall_iou050"] < .90):
            raise ValueError("Crowded Top300 registration does not satisfy the preregistered 0.90 gate")
        recipe = Path(registration["recipe_path"])
        if hashlib.sha256(recipe.read_bytes()).hexdigest() != registration["recipe_sha256"]:
            raise ValueError("Crowded candidate recipe differs from its registration")
        rule = json.loads(recipe.read_text())["mot20_crowded_candidate_rule"]
        if rule["conditional_k"] != 300 or rule["top300_acceptance_min"] != .90:
            raise ValueError("Crowded candidate decision was not preregistered at 0.90")
        datasets["mot20"]["candidate_top_k"] = 300
        datasets["mot20"]["crowded_registration"] = str(args.crowded_registration)
    ordinary = [datasets[name] for name in ("bdd100k", "dancetrack", "mot17", "mot20", "kitti_legal_development")]
    for item in ordinary:
        if item.get("candidate_top_k", 150) != 150:
            if item["dataset"] != "mot20" or not args.crowded_registration or item["candidate_top_k"] != 300:
                raise ValueError("Ordinary candidate overrides require the registered MOT20 Top300 audit")
    missing_ordinary = [item["dataset"] for item in ordinary if item["status"] != "AVAILABLE"]
    tao_missing = any(datasets[name]["status"] != "AVAILABLE" for name in ("tao", "ctao"))
    unmeasured_ordinary = [item["dataset"] for item in ordinary
                          if item["dataset"] != "kitti_legal_development"
                          and item.get(f"top{item.get('candidate_top_k', 150)}_recall_iou050") is None]
    if original_kitti is None:
        unmeasured_ordinary.append("kitti_legal_development")
    failed_ordinary = [item["dataset"] for item in ordinary
                       if item["dataset"] != "kitti_legal_development"
                       and item.get(f"top{item.get('candidate_top_k', 150)}_recall_iou050") is not None
                       and item[f"top{item.get('candidate_top_k', 150)}_recall_iou050"] < (.90 if item.get('candidate_top_k', 150) == 300 else .85)]
    if not kitti["u1_reference_top150_minimum_pass"] or (original_kitti is not None and not kitti["top150_minimum_pass"]):
        failed_ordinary.append("kitti_legal_development")
    hard_stop_ordinary = [item["dataset"] for item in ordinary
                          if item["dataset"] != "kitti_legal_development"
                          and item.get(f"top{item.get('candidate_top_k', 150)}_recall_iou050") is not None
                          and item[f"top{item.get('candidate_top_k', 150)}_recall_iou050"] < .80]
    if original_kitti is not None and kitti["top150_recall_iou050"] < .80:
        hard_stop_ordinary.append("kitti_legal_development")
    tao_gates = {}
    for name in ("tao", "ctao"):
        item = datasets[name]
        base, novel = item.get("base_recall_iou050"), item.get("novel_recall_iou050")
        # C-TAO is the official base-only release: zero novel GT is an
        # inapplicable group, while its measured base recall is still required.
        novel_required = name == "tao"
        unmeasured = base is None or (novel_required and novel is None)
        passed = not unmeasured and base >= .70 and (not novel_required or novel >= .60)
        tao_gates[name] = {"base_recall_iou050": base, "novel_recall_iou050": novel,
                           "novel_required": novel_required, "unmeasured": unmeasured, "pass": passed}
    tao_unmeasured = any(gate["unmeasured"] for gate in tao_gates.values())
    tao_pass = all(gate["pass"] for gate in tao_gates.values())
    if missing_ordinary or tao_missing:
        status = "BLOCKED_MISSING_LOCAL_DATA"
    elif unmeasured_ordinary or tao_unmeasured:
        status = "BLOCKED_NOT_MEASURED"
    elif failed_ordinary or not tao_pass:
        status = "FAIL_PORTABILITY_GATES"
    else:
        status = "PASS"
    payload = {
        "format": "locatemot-u2-perception-portability-v1",
        "stage": "U2-P",
        "foundation_sha256": args.expected_foundation_sha256,
        "measurements_dir": str(args.measurements_dir.resolve()),
        "top_k_default": 150,
        "top_k_crowded_override": 300,
        "crowded_override_minimum": .90,
        "ordinary_target_minimum": 0.85,
        "ordinary_target_preferred": 0.90,
        "ordinary_hard_stop": 0.80,
        "tao_target_base": 0.70,
        "tao_target_novel": 0.60,
        "tao_insufficient_below": 0.50,
        "datasets": datasets,
        "ordinary_missing_local_data": missing_ordinary,
        "tao_missing_local_data": tao_missing,
        "ordinary_unmeasured": unmeasured_ordinary,
        "ordinary_below_minimum": failed_ordinary,
        "ordinary_below_hard_stop": hard_stop_ordinary,
        "tao_unmeasured": tao_unmeasured,
        "tao_gate_pass": tao_pass,
        "tao_stream_gates": tao_gates,
        "status": status,
        "training_eligible": status == "PASS",
        "candidate_cache_built": False,
        "official_test_labels_read": False,
        "refer_kitti_official_eval_labels_read": False,
        "notes": [
            "Both immutable U1 Refer-KITTI references and the complete original-KITTI legal development GT audit must pass; no reserved official labels are opened.",
            "Missing public datasets remain null rather than being replaced by synthetic or fabricated metrics.",
            "Track training requires available registered data and measured portability gates; availability alone is insufficient.",
        ],
    }
    output = args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "ordinary_missing": missing_ordinary, "tao_missing": tao_missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
