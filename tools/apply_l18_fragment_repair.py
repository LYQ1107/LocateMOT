"""Apply causal RMOT-only fragment repair to an L18 prediction tree."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.rmot.l18_fragment_repair import Fragment, repair_fragment  # noqa: E402
from tools.train_l18_carr import BankStore  # noqa: E402


def read_prediction(path: Path):
    rows = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if row:
                rows.append([float(value) for value in row])
    return rows


def write_prediction(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(",".join(
                f"{value:.6f}" if isinstance(value, float) else str(value)
                for value in row) + "\n")


def bank_lookup(bank: dict, dance: bool):
    tensors = bank["tensors"]
    out = {}
    frame_ids = tensors["frame_ids"].tolist()
    for fi, raw_frame in enumerate(frame_ids):
        begin = int(tensors["frame_ptr"][fi])
        end = int(tensors["frame_ptr"][fi + 1])
        frame = int(raw_frame) if dance else int(raw_frame) + 1
        for index in range(begin, end):
            out[(frame, int(tensors["track_id"][index]))] = index
    return out


def repair_query(rows, bank: dict, dance: bool, lookup=None,
                 repair_threshold: float = 0.62):
    lookup = bank_lookup(bank, dance) if lookup is None else lookup
    tensors = bank["tensors"]
    active: list[Fragment] = []
    repaired = []
    diagnostics = {"input_rows": len(rows), "merges": 0, "new_fragments": 0,
                   "lookup_misses": 0}
    rows_by_frame = {}
    for row in rows:
        rows_by_frame.setdefault(int(row[0]), []).append(row)
    for frame in sorted(rows_by_frame):
        for row in rows_by_frame[frame]:
            track_id = int(row[1])
            index = lookup.get((frame, track_id))
            if index is None:
                diagnostics["lookup_misses"] += 1
                box = np.asarray((row[2], row[3], row[2] + row[4],
                                  row[3] + row[5]), np.float32)
                appearance = np.zeros(512, np.float32)
                embedding = np.zeros(384, np.float32)
                source = 0
            else:
                box = np.asarray(tensors["box"][index], np.float32)
                appearance = np.asarray(tensors["clip"][index], np.float32)
                embedding = np.asarray(tensors["uidm_h"][index], np.float32)
                pool = tensors.get("pool_id")
                source = int(pool[index]) if pool is not None else 0
            identity, label, _score = repair_fragment(
                active, frame, float(row[6]), appearance, embedding,
                box, source, threshold=repair_threshold)
            diagnostics["merges" if label == "merge" else "new_fragments"] += 1
            repaired.append([
                int(row[0]), identity, float(row[2]), float(row[3]),
                float(row[4]), float(row[5]), float(row[6]), -1, -1, -1,
            ])
    return repaired, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bank-root", default="outputs/l18/dual_banks")
    parser.add_argument("--repair-threshold", type=float, default=0.62)
    args = parser.parse_args()
    input_root = (ROOT / args.input_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    store = BankStore((ROOT / args.bank_root).resolve(), cache_size=1)
    dance = args.dataset in ("dance", "trainval_dance")
    grouped = {}
    for prediction in input_root.glob("uidm18/*/*/predict.txt"):
        video = prediction.parts[-3]
        expression = prediction.parts[-2]
        grouped.setdefault(video, []).append((expression, prediction))
    diagnostics = []
    for video, entries in sorted(grouped.items()):
        bank = store.get("dance_eval" if dance else "kitti", video)
        lookup = bank_lookup(bank, dance)
        for expression, source in sorted(entries):
            rows = read_prediction(source)
            repaired, diag = repair_query(
                rows, bank, dance, lookup, args.repair_threshold)
            destination = output_root / "uidm18" / video / expression / "predict.txt"
            write_prediction(destination, repaired)
            gt = source.parent / "gt.txt"
            if gt.exists():
                target_gt = destination.parent / "gt.txt"
                if not target_gt.exists():
                    target_gt.symlink_to(gt.resolve())
            diagnostics.append({"video": video, "expression": expression, **diag})
    for name in ("seqmap_l18.txt",):
        source = input_root / name
        if source.exists():
            shutil.copy2(source, output_root / name)
    (output_root / "fragment_repair_manifest.json").write_text(
        json.dumps({"dataset": args.dataset, "queries": diagnostics}, indent=2) + "\n")
    print(json.dumps({"dataset": args.dataset, "queries": len(diagnostics),
                      "input_rows": sum(x["input_rows"] for x in diagnostics),
                      "merges": sum(x["merges"] for x in diagnostics),
                      "new_fragments": sum(x["new_fragments"] for x in diagnostics),
                      "lookup_misses": sum(x["lookup_misses"] for x in diagnostics)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
