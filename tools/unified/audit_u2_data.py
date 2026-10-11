#!/usr/bin/env python3
"""Audit U2 ordinary-MOT and OVMOT data without opening reserved KITTI labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
U1_DATA = PROJECT_ROOT / "outputs" / "unified" / "protocol" / "data_manifest.json"


DATASETS: dict[str, dict[str, Any]] = {
    "bdd100k": {
        "kind": "ordinary_mot",
        "roots": [
            Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd"),
            Path("/data2/user/datasets/bdd100k"),
            PROJECT_ROOT / "data" / "bdd100k",
        ],
        "license": "BDD100K terms; local acquisition required",
        "split": "official tracking train/val; full raw annotation and image coverage required",
    },
    "dancetrack": {
        "kind": "ordinary_mot",
        "roots": [
            Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack"),
            Path("/data2/user/datasets/DanceTrack"),
            PROJECT_ROOT / "data" / "DanceTrack",
        ],
        "license": "DanceTrack research release; local acquisition required",
        "split": "existing project protocol configs/data/l1_a_dancetrack_{train,val}.json",
    },
    "mot17": {
        "kind": "ordinary_mot",
        "roots": [
            Path("/data1/LWR/vranlee/DATASETS/JDE/MOT17"),
            Path("/data2/user/datasets/MOT17"),
            PROJECT_ROOT / "data" / "MOT17" / "MOT17",
            PROJECT_ROOT / "data" / "MOT17",
        ],
        "license": "MOTChallenge terms; local acquisition required",
        "split": "ByteTrack train-half; frozen outputs/unified/u2/protocol/mot17_half.json; FRCNN copies only",
    },
    "mot20": {
        "kind": "ordinary_mot",
        "roots": [
            Path("/data1/LWR/vranlee/M4FTMoveOut4Doing/ByteTrack-mbt/datasets/mix_mot20_ch/mot20_train"),
            Path("/data2/user/datasets/MOT20"),
            PROJECT_ROOT / "data" / "MOT20" / "MOT20",
            PROJECT_ROOT / "data" / "MOT20",
        ],
        "license": "MOTChallenge terms; local acquisition required",
        "split": "ByteTrack train-half; frozen outputs/unified/u2/protocol/mot20_half.json",
    },
    "kitti_legal_development": {
        "kind": "ordinary_mot",
        "roots": [PROJECT_ROOT / "data" / "kitti_tracking"],
        "license": "KITTI tracking; U1 legal development scope",
        "split": "Refer-KITTI legal calibration/validation only; reserved videos are excluded",
    },
    "tao": {
        "kind": "ovmot",
        "roots": [
            Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal"),
            Path("/data2/user/datasets/TAO-Amodal"),
            PROJECT_ROOT / "data" / "TAO-Amodal",
        ],
        "license": "TAO research terms; local acquisition required",
        "split": "TAO train for learning, TAO validation for evaluation",
    },
    "ctao": {
        "kind": "ovmot",
        "roots": [
            Path("/data3/testdata/vranlee/.MOTSynth.partial/C-TAO"),
            Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/C-TAO"),
            Path("/data2/user/datasets/C-TAO"),
            PROJECT_ROOT / "data" / "C-TAO",
        ],
        "license": "C-TAO continuous annotation release; local acquisition required",
        "split": "C-TAO train base annotations; TAO validation remains held out",
    },
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    files = 0
    dirs = 0
    image_files = 0
    annotation_files = 0
    sample_files: list[str] = []
    for current, dirnames, filenames in os.walk(path, followlinks=True):
        dirs += len(dirnames)
        files += len(filenames)
        image_files += sum(name.lower().endswith((".jpg", ".jpeg", ".png")) for name in filenames)
        annotation_files += sum(name == "gt.txt" or name.lower().endswith(".json") for name in filenames)
        if len(sample_files) < 5:
            sample_files.extend(str(Path(current) / name) for name in sorted(filenames)[: 5 - len(sample_files)])
    return {
        "path": str(path),
        "exists": True,
        "files": files,
        "image_files": image_files,
        "annotation_files": annotation_files,
        "directories": dirs,
        "sample_files": sample_files,
    }


def audit_dataset(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    roots = [root_summary(path) for path in spec["roots"]]
    existing = [item for item in roots if item["exists"] and item["files"] > 0
                and (item["image_files"] > 0 or name == "ctao")
                and (item["annotation_files"] > 0 or name == "kitti_legal_development")]
    status = "AVAILABLE" if existing else "MISSING_LOCAL_DATA"
    selected = existing[0]["path"] if existing else None
    evidence: dict[str, Any] = {}
    if name == "bdd100k":
        view = PROJECT_ROOT / "data/bdd100k"
        scopes = {}
        for split, expected_videos in (("train", 1400), ("val", 200)):
            labels = sorted((view / "labels/box_track_20" / split).glob("*.json"))
            frames = missing = 0
            for label_path in labels:
                video = json.loads(label_path.read_text())
                frames += len(video)
                missing += sum(not (view / "images/track" / split / label_path.stem / frame["name"]).is_file()
                               for frame in video)
            scopes[split] = {"videos": len(labels), "videos_expected": expected_videos,
                             "annotation_frames": frames, "missing_images": missing}
        if any(scope["videos"] for scope in scopes.values()):
            selected = str(view)
            evidence = {"raw_tracking_coverage": scopes,
                        "annotation_acquisition_manifest": "outputs/unified/u2/protocol/bdd_labels_acquisition_manifest.json"}
            status = "AVAILABLE" if all(scope["videos"] == scope["videos_expected"] and not scope["missing_images"]
                                         for scope in scopes.values()) else "INCOMPLETE_LOCAL_DATA"
    protocol = PROJECT_ROOT / f"outputs/unified/u2/protocol/{name}_half.json"
    if name == "dancetrack":
        protocol = PROJECT_ROOT / "outputs/unified/u2/protocol/dancetrack.json"
    if name in ("mot17", "mot20", "dancetrack") and protocol.exists():
        registered = json.loads(protocol.read_text())
        missing = [str(Path(sequence["image_dir"]) / frame)
                   for sequence in registered["sequences"]
                   for frame in sequence["train_frames"] + sequence.get("calibration_frames", []) + sequence["validation_frames"]
                   if not (Path(sequence["image_dir"]) / frame).is_file()]
        gt_matches = all(sha256(Path(sequence["gt_path"])) == sequence["gt_sha256"]
                         for sequence in registered["sequences"])
        evidence = {"protocol_path": str(protocol), "protocol_sha256": sha256(protocol),
                    "train_images": registered["train_images"], "validation_images": registered["validation_images"],
                    "missing_registered_images": len(missing), "registered_gt_matches": gt_matches,
                    "source": registered["source"], "version": registered["version"]}
        if name == "dancetrack":
            evidence.update(calibration_images=registered["calibration_images"],
                            metadata_frame_differences=registered["metadata_frame_differences"])
        selected = registered["root"]
        status = "AVAILABLE" if not missing and gt_matches else "INCOMPLETE_LOCAL_DATA"
    if name == "tao":
        view = PROJECT_ROOT / "data/TAO-Amodal"
        annotations = [view / f"annotations/{filename}.json" for filename in ("train", "validation")]
        if all(path.is_file() for path in annotations):
            selected = str(view)
            scopes = {}
            for path in annotations:
                data = json.loads(path.read_text())
                missing = sum(not (view / "frames" / image["file_name"]).is_file() for image in data["images"])
                scopes[path.stem] = {"annotation_sha256": sha256(path), "images_expected": len(data["images"]),
                                    "missing_images": missing}
            evidence = {"modal_annotation_coverage": scopes, "source": "/我的资源/TAO-Amodal",
                        "shard_manifest": "outputs/unified/u2/protocol/tao_shard_manifest.json"}
            status = "AVAILABLE" if all(s["missing_images"] == 0 for s in scopes.values()) else "INCOMPLETE_LOCAL_DATA"
    if name == "ctao":
        base = PROJECT_ROOT / "data/C-TAO/ctao_base.json"
        if base.is_file():
            selected = str(base.parent)
            data = json.loads(base.read_text())
            frames = PROJECT_ROOT / "data/TAO-Amodal/frames"
            missing = sum(not (frames / image["file_name"]).is_file() for image in data["images"])
            nontrain = sum(not image["file_name"].startswith("train/") for image in data["images"])
            evidence = {"annotation_sha256": sha256(base), "images_expected": len(data["images"]),
                        "missing_images": missing, "non_training_images": nontrain,
                        "source": "https://huggingface.co/clarkqian/COVTrack", "variant": "ctao_base.json"}
            status = "AVAILABLE" if not missing and not nontrain else "INCOMPLETE_LOCAL_DATA"
    return {
        "dataset": name,
        "kind": spec["kind"],
        "status": status,
        "selected_root": selected,
        "candidate_roots": roots,
        "license": spec["license"],
        "split_protocol": spec["split"],
        "availability_evidence": evidence,
        "portability_training_allowed": status == "AVAILABLE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/unified/u2/protocol/data_manifest.json"))
    args = parser.parse_args()
    payload = {
        "format": "locatemot-u-u2-data-manifest-v1",
        "stage": "U2-P",
        "source_commit": "56507f4aa9a12e76d6f9f6702f0f0b38163666ec",
        "u1_foundation_sha256": "4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26",
        "datasets": [audit_dataset(name, spec) for name, spec in DATASETS.items()],
        "kitti_reserved_videos_opened": [],
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "notes": [
            "Missing public datasets are recorded without fabricating metrics.",
            "Refer-KITTI official evaluation labels remain outside the U2 loader contract.",
            "Long tracking training is not eligible until all required portability inputs exist.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status_counts": {
            status: sum(item["status"] == status for item in payload["datasets"])
            for status in sorted({item["status"] for item in payload["datasets"]})
        },
        "out": str(args.out),
        "official_test_labels_read": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
