#!/usr/bin/env python3
"""Select a bounded U2-P checkpoint using full portability and unchanged U1 audits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.unified.build_u2_portability_artifact import read_measurement

REFERENCES = {
    "generic": ROOT / "outputs/unified/u1r2/formal_generic_expression_epoch1/result.json",
    "expression": ROOT / "outputs/unified/u1r2/formal_expression_adaptation_epoch1/result.json",
    "counterfactual": ROOT / "outputs/unified/u1r2/expression_counterfactual_epoch1/result.json",
}
SCOPE_FIELDS = {
    "generic": ("image_count", "target_count", "unique_physical_objects"),
    "expression": ("query_frame_count", "target_reference_count", "valid_target_count",
                   "missing_source_target_count", "valid_target_count_by_dataset", "missing_source_target_count_by_dataset"),
    "counterfactual": ("pair_count",),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_audit(path: Path, name: str, sha: str) -> dict:
    result = json.loads(path.read_text())
    reference = json.loads(REFERENCES[name].read_text())
    if (result["status"] != reference["status"] or result["format"] != reference["format"]
            or result["runtime"]["checkpoint_sha256"] != sha or result["protocol"] != reference["protocol"]
            or any(result[field] != reference[field] for field in SCOPE_FIELDS[name])):
        raise ValueError(f"U1 {name} audit differs from the original full legal scope or candidate weights")
    shards = [json.loads(Path(shard).read_text()) for shard in result["shards"]]
    count = shards[0]["worker_count"]
    if (len(shards) != count or sorted(shard["worker_index"] for shard in shards) != list(range(count))
            or any(shard["worker_count"] != count or shard["runtime"]["checkpoint_sha256"] != sha for shard in shards)):
        raise ValueError(f"U1 {name} audit contains missing, duplicate, or mixed-checkpoint shards")
    if name == "counterfactual":
        keys = ("dataset", "video", "frame", "pair_index", "true_expression", "shuffled_expression",
                "true_target_ids", "shuffled_target_ids", "target_count")
        # Retain all fields that identify the original deterministic pair set.
        reference_pairs = [{key: row[key] for key in keys} for row in reference["pairs"]]
        result_pairs = [{key: row[key] for key in keys} for row in result["pairs"]]
        if result_pairs != reference_pairs:
            raise ValueError("Counterfactual audit differs from the original fixed 500 pairs")
    return result


def selection_payload(args: argparse.Namespace) -> dict:
    portability = json.loads(args.portability.read_text())
    if (portability["foundation_sha256"] != args.foundation_sha256
            or portability["status"] not in ("PASS", "FAIL_PORTABILITY_GATES")):
        raise ValueError("Selection requires the complete portability audit of the same adapted checkpoint")
    for name in ("bdd100k", "dancetrack", "mot17", "mot20", "kitti_legal_development", "tao", "ctao"):
        measured = portability["datasets"][name]["measurement"]
        verified = read_measurement(Path(portability["measurements_dir"]), name, args.foundation_sha256)
        if verified != measured:
            raise ValueError("Portability selection differs from its full measurement artifact")
        if measured["status"] != "COMPLETE" or measured["foundation_sha256"] != args.foundation_sha256:
            raise ValueError("Incomplete or mixed-checkpoint portability selection")
    results = {name: verified_audit(getattr(args, name), name, args.foundation_sha256) for name in REFERENCES}
    for domain in ("refer_kitti_v1", "refer_kitti_v2"):
        for k in (150, 300):
            if (portability["datasets"]["kitti_legal_development"]["domains"][domain][f"top{k}_recall_iou050"]
                    != results["generic"]["metrics"][domain][str(k)]["0.5"]["recall"]):
                raise ValueError("KITTI portability and retention use different generic measurements")
    thresholds = {"generic": {"refer_kitti_v1": .966, "refer_kitti_v2": .939},
                  "expression": {"refer_kitti_v1": .97, "refer_kitti_v2": .93}}
    retained = {}
    for name, domains in thresholds.items():
        k = "150" if name == "generic" else "100"
        retained[name] = {domain: {"recall": results[name]["metrics"][domain][k]["0.5"]["recall"],
                                  "minimum": minimum,
                                  "pass": results[name]["metrics"][domain][k]["0.5"]["recall"] >= minimum}
                          for domain, minimum in domains.items()}
    cf = results["counterfactual"]
    language_pass = cf["true_ap50_top300_mean"] > cf["shuffled_ap50_top300_mean"]
    retention_pass = language_pass and all(cell["pass"] for domains in retained.values() for cell in domains.values())
    status = "REJECT_RETENTION" if not retention_pass else "SELECTED" if portability["status"] == "PASS" else "REJECT_PORTABILITY"
    payload = {"format": "locatemot-u2-perception-adaptation-selection-v1", "epoch": args.epoch,
               "foundation_sha256": args.foundation_sha256, "status": status,
               "retention_pass": retention_pass, "retention": retained,
               "language_conditioning_pass": language_pass,
               "true_ap50": cf["true_ap50_top300_mean"], "shuffled_ap50": cf["shuffled_ap50_top300_mean"],
               "portability_status": portability["status"], "track_training_eligible": status == "SELECTED",
               "next_epoch_eligible": args.epoch == 1 and status != "SELECTED",
               "sources": {name: {"path": str(getattr(args, name).resolve()), "sha256": digest(getattr(args, name))}
                           for name in ("portability", *REFERENCES)},
               "official_test_labels_read": False}
    return payload


def verify_selected_foundation(path: Path, sha: str) -> dict:
    """Recompute the existing full gate before cache or track training."""
    selected = json.loads(path.read_text())
    if (selected["status"] != "SELECTED" or not selected["track_training_eligible"]
            or selected["foundation_sha256"] != sha):
        raise ValueError("Cache/tracking requires a selected foundation with passing full gates")
    sources = selected["sources"]
    for source in sources.values():
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("The selected foundation's full audit artifact has changed")
    args = argparse.Namespace(foundation_sha256=sha, epoch=selected["epoch"],
                              **{name: Path(source["path"]) for name, source in sources.items()})
    if selection_payload(args) != selected:
        raise ValueError("The recorded foundation selection differs from the actual full audits")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-sha256", required=True)
    parser.add_argument("--epoch", type=int, choices=(1, 2), required=True)
    parser.add_argument("--portability", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--counterfactual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = selection_payload(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
