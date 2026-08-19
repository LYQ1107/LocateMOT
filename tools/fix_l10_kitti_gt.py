"""Regenerate all Refer-KITTI-V2 GT templates (1-based frames).

The official seqmap contains 19 entries whose expression_id is a
comma-separated list of expression files (e.g. 0005+on-the-right,
silver-cars-are-located).  This writes one merged gt.txt per combined
query under outputs/l10/data/rmot_kitti/gt_template.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
OUT = ROOT / "outputs" / "l10" / "data" / "rmot_kitti"
KITTI_IMGS = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/KITTI_tracking"
                  "/training/image_02")
V2_ROOT = Path("/data1/LWR/vranlee/MFT2025/REFER-MFT25/refer-kitti-v2")
SEQMAP = (Path("/data1/LWR/vranlee/SERVER_ONLY/avis/"
               "LocateMOT_reference_repos") / "temp_rmot" /
          "datasets" / "data_path" / "seqmap.txt")


def load_labels(seq):
    d = V2_ROOT / "labels_with_ids" / "image_02" / seq
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.txt")):
        frame = int(p.stem)
        rows = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            rows.append((int(float(parts[0])), int(float(parts[1])),
                         float(parts[2]), float(parts[3]),
                         float(parts[4]), float(parts[5])))
        out[frame] = rows
    return out


def main():
    exp = json.loads((OUT / "expressions.json").read_text())
    n = 0
    for line in SEQMAP.read_text().splitlines():
        if not line.strip():
            continue
        seq, expr = line.strip().split("+", 1)
        e = next((x for x in exp.get(seq, [])
                  if x["expression"] == expr), None)
        if e is None:
            continue
        labels = load_labels(seq)
        img_dir = KITTI_IMGS / seq
        first = sorted(img_dir.glob("*.png"))[0]
        arr = cv2.imread(str(first), cv2.IMREAD_COLOR)
        H0, W0 = arr.shape[:2]
        d = OUT / "gt_template" / seq / expr
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for frame_s, ids in e["label"].items():
            frame = int(frame_s)
            idset = {int(x) for x in ids}
            for cls, tid, x1, y1, w, h in labels.get(frame, []):
                if tid in idset:
                    rows.append((frame + 1, tid, x1 * W0, y1 * H0,
                                 w * W0, h * H0, 1, 1, 1))
        rows.sort()
        with open(d / "gt.txt", "w") as f:
            for r in rows:
                f.write(",".join(
                    f"{v:.3f}" if isinstance(v, float) else str(v)
                    for v in r) + "\n")
        n += 1
    print(f"[fixkitti] gt templates regenerated: {n}")


if __name__ == "__main__":
    main()
