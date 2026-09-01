#!/usr/bin/env python3
"""L69 wrapper for a new budget-40 RMOT-only bank.

The dual-bank construction delegates the proposal/linker implementation to
the frozen L18 builder functions, while explicitly loading only the
train-pool DINO cache and the L16 main-only record path.  The wrapper adds a
compact raw-rank tensor as provenance and never writes to an L16/L18/L19 path.
The feature stage delegates the already audited L19 transformation to a new
output root with max_gap=2 and preserved source IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DINO_PATH = ROOT / "outputs/l18/cache/dino_kitti_trainval.pkl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MAIN_ROOT = ROOT / "outputs/l16/track_banks_dedup/kitti"
L16_RECORD_ROOT = ROOT / "outputs/l16/data/kitti_missing/records"
L11_RECORD_ROOT = ROOT / "outputs/l11/data/rmot_kitti"
CLIP_PATH = Path("/home/lwr/.cache/clip/ViT-B-32.pt")
RECORD_COMPAT_PYTHON = Path("/home/lwr/anaconda3/envs/masaenv_debug/bin/python")
EXPECTED_DINO = "ce0cc5b342ecf0cd7195fd4f67fcc6ec1c915b170d501e3969b3b2e1c25e1c9d"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
VIDEOS = ("0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018", "0020")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pickle_compat(path: Path):
    try:
        return pickle.load(path.open("rb"))
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        import numpy as np_local
        sys.modules["numpy._core"] = np_local.core
        sys.modules["numpy._core.numeric"] = np_local.core.numeric
        return pickle.load(path.open("rb"))


def load_l16_record(video: str):
    # Missing-record materializations are the required source for the target
    # V2 validation videos.  Other train-pool videos use the existing L11
    # train-pool record; no official-eval record is admitted here.
    path = L16_RECORD_ROOT / f"{video}.pkl"
    if not path.exists():
        path = L11_RECORD_ROOT / f"{video}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"no train-pool record for {video}")
    # The non-missing L11 train-pool pickles were written with NumPy's
    # ``numpy._core`` module path.  Injecting that path into the Torch 1.10 /
    # NumPy 1.24 builder process reproducibly segfaults while unpickling.  Read
    # only the small frame/GT-box portion needed by the frozen L18 builder in
    # the verified NumPy 1.26 environment, and return JSON-compatible values;
    # the source pickle is never rewritten and no feature/cache is produced.
    if path.parent == L11_RECORD_ROOT:
        if not RECORD_COMPAT_PYTHON.is_file():
            raise FileNotFoundError(f"record compatibility interpreter missing: {RECORD_COMPAT_PYTHON}")
        extract = r'''
import json, pickle, sys
import numpy as np

with open(sys.argv[1], "rb") as handle:
    source = pickle.load(handle)
frames = []
for frame in source.get("frames", []):
    gt_boxes = {}
    for key, value in frame.get("gt_boxes", {}).items():
        gt_boxes[str(key)] = np.asarray(value, dtype=np.float32).reshape(-1).tolist()
    frames.append({"frame": int(frame["frame"]), "gt_boxes": gt_boxes})
print(json.dumps({"frames": frames}, separators=(",", ":")))
'''
        result = subprocess.run(
            [str(RECORD_COMPAT_PYTHON), "-c", extract, str(path)],
            check=False, capture_output=True, text=True, timeout=120,
            env={key: value for key, value in os.environ.items()
                 if key not in {"PYTHONPATH", "PYTHONHOME"}},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"compatible L11 record read failed rc={result.returncode}: "
                f"{result.stderr[-2000:]}")
        return json.loads(result.stdout)
    return load_pickle_compat(path)


def add_raw_rank(bank_path: Path, dino_video: dict) -> dict:
    bank = torch.load(bank_path, map_location="cpu")
    tensors = bank["tensors"]
    frame_ids = [int(x) for x in tensors["frame_ids"].tolist()]
    pool = tensors["pool_id"].long()
    boxes = tensors["box"].float().numpy().astype(np.float32)
    raw_rank = np.full(len(boxes), -1, np.int32)
    ptr = tensors["frame_ptr"].long().tolist()
    reserve_count = 0
    for fi, frame in enumerate(frame_ids):
        start, end = int(ptr[fi]), int(ptr[fi + 1])
        entry = dino_video.get(frame)
        if entry is None:
            raise AssertionError(f"DINO frame missing for raw rank {frame}")
        raw_boxes = np.asarray(entry.get("boxes", []), np.float32).reshape(-1, 4)
        scores = np.asarray(entry.get("scores", []), np.float32).reshape(-1)
        order = np.argsort(-scores, kind="stable")[:40]
        seen, unique_raw = set(), []
        for raw_index in order.tolist():
            key = np.asarray(raw_boxes[raw_index], np.float32).tobytes()
            if key in seen:
                continue
            seen.add(key)
            unique_raw.append(int(raw_index))
        reserve_rows = [i for i in range(start, end) if int(pool[i]) == 1]
        if len(reserve_rows) != len(unique_raw):
            raise AssertionError(f"reserve count/raw dedup mismatch {frame}: {len(reserve_rows)} != {len(unique_raw)}")
        for row, raw_index in zip(reserve_rows, unique_raw):
            if np.asarray(boxes[row], np.float32).tobytes() != np.asarray(raw_boxes[raw_index], np.float32).tobytes():
                raise AssertionError(f"reserve row order/box mismatch {frame}/{row}")
            raw_rank[row] = int(raw_index) + 1
        reserve_count += len(reserve_rows)
    if (raw_rank[pool.numpy() == 0] != -1).any() or (raw_rank[pool.numpy() == 1] < 1).any():
        raise AssertionError("raw_rank pool contract failed")
    tensors["raw_rank"] = torch.from_numpy(raw_rank)
    metadata = dict(bank.get("metadata", {}))
    metadata.update({
        "format": "locatemot-l18-dual-track-bank-v1",
        "reserve_budget": 40,
        "raw_rank_schema": "main=-1; reserve=1-based rank in stable raw DINO array after top-40 selection and exact dedup",
        "l69_wrapper": "budget40; train-pool-only; L18 builder implementation reused",
        "l69_source_record_policy": "L16 missing-record materializations first; L11 train-pool records only for non-missing train videos",
        "reserve_observations": int(reserve_count),
        "observations": int(len(tensors["track_id"])),
        "causal": True, "query_independent": True, "rmot_only_reserve_namespace": True,
    })
    bank["metadata"] = metadata
    temporary = bank_path.with_suffix(".l69tmp")
    torch.save(bank, temporary)
    os.replace(temporary, bank_path)
    bank_path.with_suffix(".audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def write_manifest(out_root: Path, rows: list[dict], stage: str, command: str):
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps({
        "format": "locatemot-l69-budget40-manifest-v1", "status": "complete",
        "stage": stage, "project_root": str(ROOT), "cwd": str(Path.cwd()),
        "command": command, "budget": 40, "videos": rows,
        "dino_cache": str(DINO_PATH), "dino_cache_sha256": sha256(DINO_PATH),
        "clip_weight": str(CLIP_PATH), "clip_sha256": sha256(CLIP_PATH),
        "record_compatibility_interpreter": str(RECORD_COMPAT_PYTHON),
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "official_eval_loaded": False, "screening_loaded": False,
    }, indent=2) + "\n")


def dual_stage(videos: list[str], out_root: Path, command: str):
    if sha256(DINO_PATH) != EXPECTED_DINO or sha256(MANIFEST) != EXPECTED_MANIFEST:
        raise AssertionError("frozen input hash mismatch")
    if any(v not in VIDEOS for v in videos):
        raise AssertionError("video outside fixed train-pool union")
    dino = load_pickle_compat(DINO_PATH)
    # Importing the L18 implementation is allowed; its main() is not called,
    # so the official DINO cache and official-eval split are never loaded.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    import build_l18_dual_track_bank as l18
    # The approved ovtr runtime is Torch 1.10, while the frozen L18 source
    # was also run under newer Torch and passes weights_only=False.  Adapt only
    # this subprocess call; do not edit the frozen builder or source banks.
    original_torch_load = torch.load
    def load_without_weights_only(*args, **kwargs):
        kwargs.pop("weights_only", None)
        return original_torch_load(*args, **kwargs)
    l18.torch.load = load_without_weights_only
    l18.load_record = load_l16_record
    try:
        import clip
    except Exception as exc:
        raise RuntimeError(f"verified local CLIP API unavailable: {exc}")
    if not CLIP_PATH.is_file():
        raise FileNotFoundError(CLIP_PATH)
    if sha256(CLIP_PATH) != "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af":
        raise AssertionError("ViT-B/32 weight hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("L69 dual materialization requires the approved single GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model, _preprocess = clip.load(str(CLIP_PATH), device=device, jit=False)
    model.eval()
    rows = []
    for index, video in enumerate(videos):
        destination = out_root / f"{video}.pt"
        complete = destination.with_suffix(".complete")
        if complete.exists():
            blob = torch.load(destination, map_location="cpu")
            if int(blob.get("metadata", {}).get("reserve_budget", -1)) != 40:
                raise AssertionError(f"existing output is not budget40: {destination}")
            rows.append(blob["metadata"])
            print(f"[l69-dual] reuse {video}", flush=True)
            continue
        meta = l18.build_one(video, 40, dino, {}, model, device, out_root)
        meta = add_raw_rank(destination, dino[video])
        rows.append(meta)
        print(f"[l69-dual] {video} {index + 1}/{len(videos)} rows={meta['observations']} reserve={meta['reserve_observations']}", flush=True)
        torch.cuda.empty_cache()
    del model, dino
    torch.cuda.empty_cache()
    write_manifest(out_root, rows, "dual", command)


def feature_stage(videos: list[str], source_root: Path, out_root: Path, command: str):
    if any(v not in VIDEOS for v in videos):
        raise AssertionError("video outside fixed train-pool union")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    import build_l19_reserve_identity as l19
    rows = []
    for index, video in enumerate(videos):
        source = source_root / f"{video}.pt"
        destination = out_root / f"{video}.pt"
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.with_suffix(".complete").exists():
            blob = torch.load(destination, map_location="cpu")
            rows.append(blob["metadata"])
            print(f"[l69-features] reuse {video}", flush=True)
            continue
        meta = l19.build_one(source, destination, max_gap=2, preserve_source_ids=True)
        if int(meta.get("reserve_budget", -1)) != 40 or not meta.get("preserve_source_ids"):
            raise AssertionError(f"feature metadata contract failed {video}")
        rows.append(meta)
        print(f"[l69-features] {video} {index + 1}/{len(videos)} rows={meta['observations']}", flush=True)
    write_manifest(out_root, rows, "features", command)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("dual", "features"), required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--source-root", type=Path)
    args = ap.parse_args()
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_root = out_root.resolve()
    command = " ".join([sys.executable] + sys.argv)
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd {Path.cwd()}")
        videos = list(dict.fromkeys(args.videos))
        if args.stage == "dual":
            dual_stage(videos, out_root, command)
        else:
            if args.source_root is None:
                raise ValueError("--source-root required for feature stage")
            source_root = args.source_root if args.source_root.is_absolute() else ROOT / args.source_root
            feature_stage(videos, source_root.resolve(), out_root, command)
    except Exception:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "INCOMPLETE.md").write_text("# L69 INCOMPLETE\n\n```text\n" + traceback.format_exc() + "\n```\n")
        raise


if __name__ == "__main__":
    main()
