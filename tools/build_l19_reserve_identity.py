"""Materialize L19 long reserve identity and observation groups.

This consumes the frozen L18 dual bank and writes a new RMOT-only bank.  It
does not rerun the detector or alter the L16/L18 inputs, which makes the
identity comparison directly attributable to the linker and feature-memory
change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.rmot.l19_reserve_identity import (  # noqa: E402
    long_reserve_track_ids, observation_groups, reserve_identity_features,
)


def as_frame_values(tensors: dict, name: str) -> list[np.ndarray]:
    values = []
    ptr = tensors["frame_ptr"].tolist()
    for index in range(len(ptr) - 1):
        values.append(tensors[name][ptr[index]:ptr[index + 1]].numpy())
    return values


def build_one(source: Path, destination: Path, max_gap: int,
              preserve_source_ids: bool = False) -> dict:
    bank = torch.load(source, map_location="cpu", weights_only=False)
    tensors = bank["tensors"]
    frame_ids = [int(value) for value in tensors["frame_ids"].tolist()]
    pool = tensors.get("pool_id", torch.zeros(len(tensors["track_id"]), dtype=torch.long))
    main_boxes, reserve_boxes = [], []
    main_clips, reserve_clips = [], []
    reserve_source_ids = []
    ptr = tensors["frame_ptr"].tolist()
    for index in range(len(frame_ids)):
        frame = slice(ptr[index], ptr[index + 1])
        source_pool = pool[frame].numpy()
        boxes = tensors["box"][frame].numpy().astype(np.float32)
        clips = tensors["clip"][frame].numpy().astype(np.float32)
        main = source_pool == 0
        reserve = source_pool == 1
        main_boxes.append(boxes[main])
        reserve_boxes.append(boxes[reserve])
        main_clips.append(clips[main])
        reserve_clips.append(clips[reserve])
        reserve_source_ids.append(tensors["track_id"][frame].numpy()[reserve])
    image_size = tuple(bank["metadata"].get("image_size", [1242, 375]))
    reserve_ids = long_reserve_track_ids(
        reserve_boxes, reserve_clips, frame_ids, image_size, max_gap=max_gap)
    identity_ids = (reserve_source_ids if preserve_source_ids else reserve_ids)
    identity = reserve_identity_features(
        reserve_boxes, reserve_clips, identity_ids, frame_ids, image_size)
    groups, duplicates = observation_groups(
        main_boxes, main_clips, reserve_boxes, reserve_clips, frame_ids)

    # Clone only tensor metadata and replace reserve-side views.  The main
    # observations remain byte-identical to the L18 source bank.
    output_tensors = {name: value.clone() for name, value in tensors.items()}
    reserve_rows = []
    for index in range(len(frame_ids)):
        start, end = ptr[index], ptr[index + 1]
        source_pool = pool[start:end].numpy()
        reserve_rows.append(np.flatnonzero(source_pool == 1) + start)
        rows = reserve_rows[-1]
        if not len(rows):
            continue
        if not preserve_source_ids:
            output_tensors["track_id"][rows] = torch.as_tensor(
                reserve_ids[index].astype(np.int64) + 1_000_000,
                dtype=output_tensors["track_id"].dtype)
        for name in ("history_clip", "pbd", "uidm_h", "uidm_ref_pbd",
                     "uidm_anchor_pbd", "motion", "lifecycle"):
            output_tensors[name][rows] = torch.from_numpy(identity[name][index])
    output_tensors["observation_group_id"] = torch.from_numpy(
        np.concatenate(groups, axis=0).astype(np.int64))
    output_tensors["cross_pool_duplicate"] = torch.from_numpy(
        np.concatenate(duplicates, axis=0).astype(np.uint8))

    reserve_rows_flat = np.concatenate(reserve_rows) if reserve_rows else np.zeros(0, np.int64)
    nonzero = {}
    for name in ("pbd", "uidm_h", "uidm_ref_pbd", "uidm_anchor_pbd"):
        value = output_tensors[name][reserve_rows_flat].float() if len(reserve_rows_flat) else torch.zeros(0)
        nonzero[name] = {
            "rows": int(len(value)),
            "nonzero_rows": int((value.abs().sum(dim=1) > 1e-6).sum()) if value.ndim == 2 else 0,
            "mean_l2": float(value.norm(dim=1).mean()) if value.ndim == 2 and len(value) else 0.0,
        }
    metadata = dict(bank["metadata"])
    metadata.update({
        "format": "locatemot-l19-dual-bank-v1",
        "parent_bank": str(source.resolve()),
        "parent_bank_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reserve_tracker": ("frozen L18 causal IoU/CLIP linker with L19 identity views"
                             if preserve_source_ids else
                             "causal long-memory IoU/appearance linker"),
        "reserve_tracker_max_gap": int(max_gap),
        "reserve_identity_feature_schema": "clip/history/delta/abs_delta",
        "reserve_identity_features": "nonzero equivalent identity views; not official UIDM",
        "observation_group_schema": "same-frame IoU>=.50 or IoU>=.30 plus CLIP cosine>=.82",
        "cross_pool_duplicate_rows": int(output_tensors["cross_pool_duplicate"].sum()),
        "reserve_identity_nonzero": nonzero,
        "causal": True,
        "rmot_only_reserve_namespace": True,
        "preserve_source_ids": bool(preserve_source_ids),
    })
    out_bank = {"metadata": metadata, "tensors": output_tensors}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(out_bank, temporary)
    os.replace(temporary, destination)
    for suffix in (".audit.json", ".complete"):
        path = destination.with_suffix(suffix)
        if suffix == ".audit.json":
            path.write_text(json.dumps(metadata, indent=2) + "\n")
        else:
            path.write_text("ok\n")
    labels = source.with_suffix(".labels.json")
    if labels.exists():
        destination.with_suffix(".labels.json").write_text(labels.read_text())
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="outputs/l18/dual_banks/kitti")
    parser.add_argument("--out-root", default="outputs/l19/dual_banks/kitti")
    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preserve-source-ids", action="store_true")
    args = parser.parse_args()
    source_root = (ROOT / args.source_root).resolve()
    out_root = (ROOT / args.out_root).resolve()
    names = sorted(source_root.glob("*.pt"))
    if args.videos:
        allowed = set(args.videos)
        names = [path for path in names if path.stem in allowed]
    rows = []
    for index, source in enumerate(names):
        destination = out_root / source.name
        if destination.with_suffix(".complete").exists() and not args.force:
            bank = torch.load(destination, map_location="cpu", weights_only=False)
            rows.append(bank["metadata"])
            print(f"[l19-bank] skip {source.stem}", flush=True)
            continue
        metadata = build_one(source, destination, args.max_gap,
                             preserve_source_ids=args.preserve_source_ids)
        rows.append(metadata)
        print(f"[l19-bank] {source.stem} {index + 1}/{len(names)} "
              f"reserve={metadata.get('reserve_observations', 0)} "
              f"duplicates={metadata['cross_pool_duplicate_rows']}", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps({
        "format": "locatemot-l19-dual-bank-manifest-v1",
        "source_root": str(source_root), "max_gap": args.max_gap,
        "videos": rows,
    }, indent=2) + "\n")
    print(f"[l19-bank] done videos={len(rows)} out={out_root}", flush=True)


if __name__ == "__main__":
    main()
