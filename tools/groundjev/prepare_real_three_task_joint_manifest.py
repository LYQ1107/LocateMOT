#!/usr/bin/env python3
"""Build a streamed, auditable three-task OneJev source manifest.

This preparation deliberately does not launch an optimizer.  It replaces the
historical full-source manifest whose MOT20-05 rows overlap the independent
held-out scope.  MOT and RMOT rows come from the registered native candidate
inventories. OVMOT Base rows remain an explicit teacher-control by default;
a complete native TAO TRAIN join manifest can be supplied explicitly once its
candidate stream has been exported.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, TextIO


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path("/data2/user/LocateMOT_GroundJEV")
MOT = Path("/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/trackjev_sft_v1/rows.jsonl")
RMOT = RUNTIME / "outputs/groundjev/training_data/onejev_native_rmot_membership_v2_registered_release_root/train.jsonl"
OVMOT = RUNTIME / "outputs/groundjev/training_data/onejev_base_categories_v1/train.jsonl"
OVMOT_NATIVE_ARCHIVE = RUNTIME / "outputs/groundjev/multitask_runs/tao_base_novel63_full_v2_expression_none/native_candidates/frame_candidates.jsonl"
OVMOT_NATIVE_RECEIPT = OVMOT_NATIVE_ARCHIVE.parent / "receipt.json"

PROTOCOL = "immutable-native-sources-family-filter-full-exposure-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_rows(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")  # type: ignore[return-value]
    return path.open("r", encoding="utf-8")


def audit_source(path: Path, *, families: set[str], expected_rows: int,
                 source_name: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    counts = Counter()
    ids: set[str] = set()
    media_paths: set[str] = set()
    forbidden: list[str] = []
    rows = 0
    with open_rows(path) as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("split", "train") != "train":
                raise ValueError(f"{source_name}: non-train row at {line_no}")
            task = str(row.get("task"))
            if task not in families:
                raise ValueError(f"{source_name}: unexpected task {task!r} at {line_no}")
            row_id = str(row.get("id"))
            if row_id in ids:
                raise ValueError(f"{source_name}: duplicate id {row_id!r}")
            ids.add(row_id)
            text = json.dumps(row, ensure_ascii=False)
            if "MOT20-05" in text or "test" in str(row.get("split", "")).lower():
                forbidden.append(row_id)
            for media in row.get("media", []):
                if isinstance(media, dict) and media.get("path"):
                    media_paths.add(str(media["path"]))
            counts[task] += 1
            rows += 1
    if rows != expected_rows:
        raise ValueError(f"{source_name}: expected {expected_rows}, got {rows}")
    if forbidden:
        raise ValueError(f"{source_name}: forbidden held-out/test markers: {forbidden[:3]}")
    return {
        "path": str(path),
        "rows": rows,
        "families": sorted(families),
        "task_counts": dict(sorted(counts.items())),
        "unique_row_ids": len(ids),
        "unique_media_paths": len(media_paths),
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=RUNTIME / "outputs/groundjev/training_data/real_three_task_joint_v1",
    )
    parser.add_argument(
        "--ovmot-native-manifest", type=Path, default=None,
        help="manifest.json emitted by prepare_native_ovmot_train_join.py; must cover all TAO TRAIN images",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    native_ovmot_manifest = None
    native_ovmot_rows = False
    if args.ovmot_native_manifest is not None:
        native_ovmot_manifest = json.loads(args.ovmot_native_manifest.read_text())
        if (native_ovmot_manifest.get("native_ovmot_training_rows") is not True
                or native_ovmot_manifest.get("ovmot_training_mode") != "native"
                or native_ovmot_manifest.get("scope", {}).get("full_train_image_scope") is not True):
            raise ValueError("native OVMOT manifest is not a complete native TRAIN join")
        native_rows_path = Path(native_ovmot_manifest["train_rows"])
        if not native_rows_path.is_file():
            raise FileNotFoundError(native_rows_path)
        native_ovmot_rows = True
        ovmot_source = (native_rows_path, {"category_group", "category_leaf"},
                        int(native_ovmot_manifest["rows"]), "ovmot_native")
    else:
        ovmot_source = (OVMOT, {"category_group", "category_leaf"}, 430,
                        "ovmot_base_teacher_control")
    sources = [
        (MOT, {"identity"}, 131214, "mot_native"),
        (RMOT, {"query_membership", "empty_query_presence"}, 346, "rmot_native"),
        ovmot_source,
    ]
    audits = [audit_source(path, families=families, expected_rows=rows, source_name=name)
              for path, families, rows, name in sources]
    manifest_sources = [
        {"path": item["path"], "families": item["families"], "rows": item["rows"],
         "source_name": name, "source_sha256": item["sha256"]}
        for (path, _families, _rows, name), item in zip(sources, audits)
    ]
    if native_ovmot_manifest is None:
        native_receipt = json.loads(OVMOT_NATIVE_RECEIPT.read_text())
        if native_receipt.get("status") != "COMPLETE_LEGAL_NATIVE_VOCABULARY_AND_ROI_STREAM":
            raise ValueError(f"unexpected native OVMOT receipt: {native_receipt.get('status')}")
        if native_receipt.get("frames") != 40 or native_receipt.get("candidates") != 35228:
            raise ValueError("native OVMOT archive receipt does not match registered 40-frame/35228-candidate scope")
    else:
        native_receipt = {
            "status": "NATIVE_TRAIN_JOIN_MANIFEST",
            "manifest": str(args.ovmot_native_manifest),
            "manifest_sha256": sha256(args.ovmot_native_manifest),
        }

    total = sum(x["rows"] for x in audits)
    task_counts = Counter()
    for item in audits:
        task_counts.update(item["task_counts"])
    manifest = {
        "protocol": PROTOCOL,
        "manifest_version": "real-three-task-joint-v1-native-ovmot" if native_ovmot_rows else "real-three-task-joint-v1",
        "rows": total,
        "sources": manifest_sources,
        "rows_per_task": dict(sorted(task_counts.items())),
        "native_ovmot_training_rows": native_ovmot_rows,
        "ovmot_training_mode": "native" if native_ovmot_rows else "base_teacher_control",
        "heldout_exclusion": {
            "MOT20-05": "excluded; no source contains MOT20-05",
            "official_eval_labels": "not read by this preparation",
        },
        "candidate_provenance": {
            "mot": "actual native GroundAnything candidate rows; 7 MOT17 TRAIN sequences",
            "rmot": "actual native GroundAnything candidates; 9 Refer-KITTI fit videos / 72 frames",
            "ovmot": "native TAO TRAIN candidate/annotation join rows" if native_ovmot_rows else "Base category teacher-control rows; native TAO archive registered separately",
            "ovmot_native_archive": str(OVMOT_NATIVE_ARCHIVE),
            "ovmot_native_archive_sha256": sha256(OVMOT_NATIVE_ARCHIVE),
            "ovmot_native_receipt": native_receipt,
            "native_ovmot_training_rows": native_ovmot_rows,
            "native_ovmot_join_manifest": str(args.ovmot_native_manifest) if native_ovmot_manifest is not None else None,
        },
        "optimizer_updates": 0,
        "gpu_launched": False,
        "full_goal_complete": False,
    }
    (out / "train.sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    receipt = {
        "status": "REAL_THREE_TASK_SOURCE_MANIFEST_PREPARED_NOT_TRAINED",
        "manifest": str(out / "train.sources.json"),
        "manifest_sha256": sha256(out / "train.sources.json"),
        "source_audits": audits,
        "rows": total,
        "rows_per_task": dict(sorted(task_counts.items())),
        "native_ovmot_rows_pending": not native_ovmot_rows,
        "optimizer_updates": 0,
        "gpu_launched": False,
        "heldout_overlap_audit": "MOT20-05 absent by streamed row scan; official eval labels not read",
    }
    (out / "manifest_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
