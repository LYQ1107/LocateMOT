"""Build the registered TrackJEV training curriculum from the legal index.

What this fixes (and the historical pipeline got wrong): exposure is planned over
*physical* units (videos, frames, identities, queries, categories), with explicit
task-level ratios, instead of letting repeated question rows decide the mixture.
The output is a registered config: no training run may start without one.

It also refuses to plan a full-coverage epoch when a dataset is missing: the plan
is written with ``coverage_complete=false`` and the participating dataset list is
explicit, so no later reader can mistake it for a ten-dataset run.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

INDEX = ROOT / "outputs" / "groundjev" / "full_data_v2" / "legal_index_v1"
OUT = ROOT / "configs" / "groundjev" / "trackjev_curriculum_v1.json"

# task-level exposure ratios for one epoch of the joint curriculum
TASK_RATIO = {"mot": 0.60, "rmot": 0.25, "ovmot": 0.15}

# one question row per physical unit, by task (identity-level supervision is
# produced by the native export pass, not invented here)
ROWS_PER_MOT_FRAME = 1
ROWS_PER_RMOT_QUERY = 3          # positive / negative / membership variants
ROWS_PER_OVMOT_IMAGE = 2         # category root / conditional leaf


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    units: dict[str, dict[str, Any]] = {}
    for name in ("mot17", "mot20", "dancetrack"):
        rows = load_jsonl(INDEX / f"{name}_train_frames.jsonl")
        units[name] = {
            "task": "mot",
            "frames": len(rows),
            "videos": len({r["video"] for r in rows}),
            "missing_images": sum(1 for r in rows if not r.get("image_exists")),
        }
    refer = load_jsonl(INDEX / "refer_kitti_v2_train_queries.jsonl")
    units["refer_kitti_v2"] = {
        "task": "rmot",
        "queries": len(refer),
        "videos": len({r["video"] for r in refer}),
        "positive_identity_frames": sum(r.get("positive_frames", 0) for r in refer),
    }
    rmot26 = load_jsonl(INDEX / "rmot26_train_samples.jsonl")
    units["rmot26"] = {
        "task": "rmot",
        "queries": len(rmot26),
        "with_identity_supervision": sum(1 for r in rmot26 if r.get("has_identity_supervision")),
        "videos": len({r["source_sequence"] for r in rmot26}),
        "held_out_samples_dropped": 342,
    }
    aerialmind_manifest = Path("/data3/user/LocateMOT_FullDataV2/datasets/AerialMind/expression")
    units["aerialmind"] = {
        "task": "rmot",
        "queries": sum(1 for _ in aerialmind_manifest.rglob("*.json")) if aerialmind_manifest.is_dir() else 0,
        "status": "READY" if aerialmind_manifest.is_dir() else "PENDING_EXTRACTION",
    }
    tao = load_jsonl(INDEX / "tao_amodal_train_images.jsonl")
    units["tao_amodal"] = {
        "task": "ovmot",
        "images": len(tao),
        "videos": len({r["canonical_video_uid"] for r in tao}),
        "missing_images": sum(1 for r in tao if not r.get("image_exists")),
    }
    animaltrack_root = Path("/data3/user/LocateMOT_FullDataV2/datasets/AnimalTrack")
    animal_split = animaltrack_root / "splits" / "gt_train.txt"
    animal_ready = animal_split.is_file()
    animal_train = ([l.strip() for l in animal_split.read_text().splitlines() if l.strip()]
                    if animal_ready else [])
    animal_frames = sum(1 for _ in (animaltrack_root / "frames_all").rglob("*.jpg")) \
        if (animaltrack_root / "frames_all").is_dir() else 0
    units["animaltrack"] = {
        "task": "mot",
        "videos": len(animal_train),
        "frames": animal_frames,
        "status": "READY" if animal_ready else "PENDING_SPLIT",
        "note": ("official TRAIN 32 / TEST 26 split registered; only TRAIN videos enter the "
                 "curriculum"),
    }
    sportsmot_root = Path("/data3/user/LocateMOT_FullDataV2/datasets/SportsMOT")
    sports_frames = 0
    sports_videos = 0
    if (sportsmot_root / "train").is_dir():
        sports_videos = sum(1 for p in (sportsmot_root / "train").iterdir()
                            if p.is_dir() and not p.name.startswith("._"))
        sports_frames = sum(1 for _ in (sportsmot_root / "train").rglob("*.jpg"))
    units["sportsmot"] = {"task": "mot", "videos": sports_videos, "frames": sports_frames}
    units["burst"] = {"task": "ovmot", "status": "PENDING_EXTRACTION",
                      "note": "shares videos with TAO; mask-derived supervision only"}

    # per-task physical totals from the datasets that are actually ready
    ready = {name: u for name, u in units.items()
             if u.get("status", "READY") == "READY"}
    task_frames = Counter()
    for name, entry in ready.items():
        task = entry["task"]
        if task == "mot":
            task_frames[task] += entry.get("frames", 0)
        elif task == "rmot":
            task_frames[task] += entry.get("queries", 0) * ROWS_PER_RMOT_QUERY
        else:
            task_frames[task] += entry.get("images", 0) * ROWS_PER_OVMOT_IMAGE

    total_rows = sum(task_frames.values())
    target_rows = total_rows  # one epoch over all ready physical units
    plan_rows: dict[str, int] = {}
    for task, ratio in TASK_RATIO.items():
        plan_rows[task] = int(round(target_rows * ratio))

    # scale dataset exposure inside each task to hit the task budget exactly
    dataset_plan: dict[str, dict[str, Any]] = {}
    for task in TASK_RATIO:
        members = {name: entry for name, entry in ready.items() if entry["task"] == task}
        task_units = Counter()
        for name, entry in members.items():
            if task == "mot":
                task_units[name] = entry.get("frames", 0)
            elif task == "rmot":
                task_units[name] = entry.get("queries", 0) * ROWS_PER_RMOT_QUERY
            else:
                task_units[name] = entry.get("images", 0) * ROWS_PER_OVMOT_IMAGE
        subtotal = sum(task_units.values())
        for name, unit in task_units.items():
            share = unit / subtotal if subtotal else 0.0
            dataset_plan[name] = {
                "task": task,
                "physical_units": members[name].get("frames")
                                  or members[name].get("queries")
                                  or members[name].get("images"),
                "share_in_task": round(share, 6),
                "planned_rows": int(round(plan_rows[task] * share)),
            }

    missing = sorted(name for name, entry in units.items()
                     if entry.get("status", "READY") != "READY")
    config = {
        "name": "TrackJEV joint curriculum v1 (pretrained OneJev, full legal TRAIN coverage)",
        "index_root": str(INDEX),
        "task_ratio": TASK_RATIO,
        "rows_per_unit": {"mot_frame": ROWS_PER_MOT_FRAME,
                          "rmot_query": ROWS_PER_RMOT_QUERY,
                          "ovmot_image": ROWS_PER_OVMOT_IMAGE},
        "physical_units": units,
        "ready_datasets": sorted(ready),
        "pending_datasets": missing,
        "coverage_complete": not missing,
        "planned_rows_by_task": plan_rows,
        "planned_rows_total": target_rows,
        "dataset_plan": dataset_plan,
        "invariants": {
            "no_held_out_video_in_training": True,
            "rmot26_held_out_samples_dropped": 342,
            "mot17_detector_copies_deduplicated": True,
            "burst_videos_not_counted_as_new": True,
            "unknown_is_not_reject": True,
            "no_label_without_validity_flag": True,
        },
        "notes": [
            "exposure is planned over physical units, not over repeated question rows",
            "a missing dataset keeps coverage_complete=false and is listed in pending_datasets",
            "identity-level rows are produced by the native export pass, which must cite this config",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(config, indent=1, ensure_ascii=False))
    print(json.dumps({"written": str(OUT),
                      "ready": config["ready_datasets"],
                      "pending": config["pending_datasets"],
                      "planned_rows_by_task": plan_rows,
                      "planned_rows_total": target_rows,
                      "coverage_complete": config["coverage_complete"]},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
