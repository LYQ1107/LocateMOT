"""Original BDD MOT categories and positive/ignore semantics from its toolkit."""
from __future__ import annotations

CLASSES = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
NAME_MAPPING = {"bike": "bicycle", "caravan": "car", "motor": "motorcycle", "person": "pedestrian", "van": "car"}
IGNORED_MAPPING = {"other person": "pedestrian", "other vehicle": "car", "trailer": "truck"}


def frame_objects(frame: dict) -> tuple[list[dict], list[dict]]:
    positives, ignored = [], []
    for label in frame.get("labels") or []:
        category = NAME_MAPPING.get(label["category"], label["category"])
        attributes = label.get("attributes") or {}
        reason = "ignored_category" if category in IGNORED_MAPPING else None
        category = IGNORED_MAPPING.get(category, category)
        if category not in CLASSES:
            raise ValueError(f"Category outside the original BDD MOT protocol: {category}")
        box = label.get("box2d")
        if box is None:
            ignored.append({"category": category, "reason": "no_box2d"})
            continue
        # Scalabel box2d_to_bbox uses inclusive x2/y2 (+1 width/height).
        width, height = box["x2"] - box["x1"] + 1, box["y2"] - box["y1"] + 1
        obj = {"track_id": str(label["id"]), "category": category,
               "bbox": [box["x1"], box["y1"], box["x1"] + width, box["y1"] + height]}
        if reason or attributes.get("ignored", False) or attributes.get("crowd", False):
            obj["reason"] = reason or ("ignored_attribute" if attributes.get("ignored", False) else "crowd")
            ignored.append(obj)
        else:
            positives.append(obj)
    return positives, ignored
