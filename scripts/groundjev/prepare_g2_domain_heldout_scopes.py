#!/usr/bin/env python3
"""Register independent G2 Clean domain-held-out MOT scopes.

This prepares only scopes whose source split is outside the terminal G2 TRAIN
manifest: DanceTrack validation video ``dancetrack0004`` and the AnimalTrack
test video ``chicken_1``.  It performs an exact image-path audit plus a
video-level source audit before writing manifests consumed by ``run_video.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
TRAIN_ROWS = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/"
    "g2_clean_train_rows.jsonl"
)
TRAIN_MANIFEST = TRAIN_ROWS.with_name("g2_clean_train_manifest.json")
OUT_ROOT = RUNTIME_ROOT / "outputs/groundjev/heldout_inputs/g2_domain_v1"
PROTOCOL_ROOT = Path("/data2/user/LocateMOT/outputs/unified/u2/protocol")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def images_in(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or "0"),
    )


def make_manifest(
    *,
    dataset: str,
    sequence: str,
    scope: str,
    image_paths: list[Path],
    gt_path: Path,
    fps: float,
    vocabulary: list[str],
    protocol: Path,
    note: str,
) -> dict[str, Any]:
    if not image_paths:
        raise FileNotFoundError(f"no images found for {scope}")
    frame_numbers = [int("".join(c for c in path.stem if c.isdigit())) for path in image_paths]
    if frame_numbers != list(range(frame_numbers[0], frame_numbers[-1] + 1)):
        raise ValueError(f"non-contiguous frame numbers for {scope}")
    with Image.open(image_paths[0]) as image:
        width, height = image.size
    return {
        "dataset": dataset,
        "sequence": sequence,
        "task": "mot",
        "images": [str(path) for path in image_paths],
        "frame_numbers": frame_numbers,
        "fps": fps,
        "image_size": {"width": width, "height": height},
        "protocol": str(protocol),
        "gt_path_for_evaluation_only": str(gt_path),
        "gt_sha256": sha256(gt_path),
        "gt_policy": "original release MOT-style rows; evaluation access only",
        "vocabulary": vocabulary,
        "scope": scope,
        "scope_note": note,
        "training_on_this_scope": False,
        "official_test_labels_read": False,
        "runtime_GT_fields": False,
    }


def normalized(path: str) -> str:
    return str(Path(path))


def audit(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    targets = {
        name: {normalized(path) for path in manifest["images"]}
        for name, manifest in manifests.items()
    }
    video_targets = {
        name: {
            manifest["sequence"],
            f"{manifest['dataset']}_{manifest['sequence']}",
        }
        for name, manifest in manifests.items()
    }
    exact: dict[str, set[str]] = {name: set() for name in manifests}
    videos: dict[str, set[str]] = {name: set() for name in manifests}
    for line in TRAIN_ROWS.open():
        row = json.loads(line)
        row_paths = {
            normalized(image["path"] if isinstance(image, dict) else str(image))
            for image in row.get("images", [])
        }
        row_video_names = set()
        for image in row.get("images", []):
            path = Path(image["path"] if isinstance(image, dict) else str(image))
            row_video_names.update(parent.name for parent in list(path.parents)[:4])
        for name in manifests:
            exact[name].update(row_paths & targets[name])
            videos[name].update(row_video_names & video_targets[name])
    return {
        "status": "G2_DOMAIN_HELDOUT_AUDIT_COMPLETE",
        "train_manifest": str(TRAIN_MANIFEST),
        "train_manifest_sha256": sha256(TRAIN_MANIFEST),
        "train_rows": str(TRAIN_ROWS),
        "scopes": {
            name: {
                "manifest": str(OUT_ROOT / f"{name}.json"),
                "frames": len(manifest["images"]),
                "dataset": manifest["dataset"],
                "sequence": manifest["sequence"],
                "exact_image_overlap": sorted(exact[name]),
                "train_video_overlap": sorted(videos[name]),
                "independent": not exact[name] and not videos[name],
            }
            for name, manifest in manifests.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    dance_root = Path("/data2/user/LocateMOT/data/downloads/DanceTrack/extracted/val/dancetrack0004")
    animal_frames = RUNTIME_ROOT / "outputs/groundjev/heldout_inputs/animaltrack_test_chicken_1/frames"
    animal_gt = Path("/data2/user/LocateMOT_FullDataV2_spill/datasets/AnimalTrack/gt_all/chicken_1_gt.txt")
    manifests = {
        "dancetrack0004_val1203": make_manifest(
            dataset="dancetrack",
            sequence="dancetrack0004",
            scope="dancetrack0004_val1203",
            image_paths=images_in(dance_root / "img1"),
            gt_path=dance_root / "gt/gt.txt",
            fps=20.0,
            vocabulary=["person"],
            protocol=PROTOCOL_ROOT / "dancetrack.json",
            note="complete legal DanceTrack validation video; no G2 TRAIN video overlap",
        ),
        "animaltrack_chicken1_test306": make_manifest(
            dataset="animaltrack",
            sequence="chicken_1",
            scope="animaltrack_chicken1_test306",
            image_paths=images_in(animal_frames),
            gt_path=animal_gt,
            fps=30.0,
            vocabulary=["chicken"],
            protocol=PROTOCOL_ROOT / "animaltrack.json",
            note="complete AnimalTrack official test video decoded from release MP4",
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
        (args.out / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = audit(manifests)
    if not all(scope["independent"] for scope in report["scopes"].values()):
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    (args.out / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
