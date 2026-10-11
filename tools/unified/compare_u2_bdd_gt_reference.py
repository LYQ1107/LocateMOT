#!/usr/bin/env python3
"""Compare all BDD validation GT objects with the cached original toolkit functions."""
from __future__ import annotations

import ast
import __future__
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from locatemot.unified.data.bdd_tracking import frame_objects


def main() -> None:
    sources = ROOT / "outputs/unified/u2/protocol/sources"
    namespace = {"IGNORED": "ignored"}
    for filename, names in (
        ("bdd_to_scalabel.py", ["deal_bdd100k_category"]),
        ("scalabel_transforms.py", ["box2d_to_bbox"]),
        ("scalabel_label_utils.py", ["check_crowd", "check_ignored"]),
    ):
        source = ast.parse((sources / filename).read_text())
        nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
    sections, current = {}, None
    for raw in (sources / "bdd_mot_config.txt").read_text().splitlines():
        line = raw.strip()
        if line in ("[name_mapping]", "[ignored_mapping]"):
            current = line[1:-1]
            sections[current] = {}
        elif line.startswith("["):
            current = None
        elif current and "=" in line:
            key, value = line.split("=", 1)
            sections[current][key.strip().strip('"')] = ast.literal_eval(value.strip())
    config = SimpleNamespace(**sections, remove_ignored=False, ignored_as_class=False)
    classes = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    counts, frames = Counter(), 0
    paths = sorted((ROOT / "data/bdd100k/labels/box_track_20/val").glob("*.json"))
    for path in paths:
        for frame in json.loads(path.read_text()):
            expected = []
            for raw in frame.get("labels") or []:
                label = SimpleNamespace(category=raw["category"], attributes=dict(raw.get("attributes") or {}),
                                        id=raw["id"], box2d=SimpleNamespace(**raw["box2d"]) if raw.get("box2d") else None)
                label = namespace["deal_bdd100k_category"](label, config, {name: i for i, name in enumerate(classes)})
                if label.box2d is None or namespace["check_crowd"](label) or namespace["check_ignored"](label):
                    continue
                x, y, width, height = namespace["box2d_to_bbox"](label.box2d)
                expected.append({"track_id": str(label.id), "category": label.category,
                                 "bbox": [x, y, x + width, y + height]})
            actual, _ = frame_objects(frame)
            if actual != expected:
                raise RuntimeError(f"BDD GT differs from official reference: {path.name}/{frame['name']}")
            counts.update(obj["category"] for obj in actual)
            frames += 1
    if len(paths) != 200 or frames != 39973:
        raise RuntimeError("Incomplete original BDD validation reference comparison")
    payload = {"status": "PASS", "validation_videos": len(paths), "validation_frames": frames,
               "positive_gt_boxes": sum(counts.values()), "positive_category_counts": dict(sorted(counts.items())),
               "reference": "cached official BDD category converter and Scalabel crowd/ignore/Box2D functions",
               "scope": "every raw validation frame and object", "coordinates_equal_exactly": True,
               "official_test_labels_read": False}
    path = ROOT / "outputs/unified/u2/protocol/bdd_gt_reference_comparison.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
