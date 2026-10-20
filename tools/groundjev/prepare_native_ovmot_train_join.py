#!/usr/bin/env python3
"""Join native GroundAnything TAO TRAIN candidates with official Base labels.

The detector stream and the TAO annotation reader are deliberately separate.
The model sees only the native candidate crop sheet during SFT; TAO TRAIN
annotations are read offline to assign a category target to a candidate whose
box overlaps one annotation.  This script never reads validation/test labels,
never replaces a candidate box with a GT box, and refuses a partial stream as
the final native OVMOT source.

The output is a gzip JSONL category component using the same hierarchical
category questions as the existing OneJev Base component.  Crop sheets are
recipes, so real pixels are read later by ``streamed_category_media`` rather
than copied into a cache.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
ONEJEV_ROOT = Path("/data2/user/LocateMOT_GroundJEV/external/OneJev")
if not ONEJEV_ROOT.is_dir():
    ONEJEV_ROOT = ROOT / "external/OneJev"
sys.path[:0] = [str(ROOT), str(ONEJEV_ROOT)]

DEFAULT_ANNOTATION = Path("/data1/user/TAO-Amodal/annotations/train.json")
DEFAULT_VOCABULARY = Path("/data2/user/LocateMOT/outputs/unified/u2/protocol/lvis_v1_vocabulary.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_key(path: str) -> str:
    """Return TAO's ``train/...`` key and reject non-TRAIN paths."""
    normal = Path(path).as_posix()
    marker = "/frames/train/"
    if marker not in normal:
        raise ValueError(f"native candidate path is not TAO TRAIN: {path}")
    suffix = normal.split(marker, 1)[1]
    if not suffix or "/val/" in suffix or "/test/" in suffix:
        raise ValueError(f"native candidate path contains a forbidden split: {path}")
    return "train/" + suffix


def xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    x, y, width, height = [float(value) for value in box]
    return [x, y, x + width, y + height]


def validate_xyxy(box: Any, width: int, height: int) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"candidate box must have four coordinates: {box!r}")
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"candidate box contains a non-finite coordinate: {box!r}")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"candidate box is outside the image without clipping: {box!r}")
    return values


def iou_xyxy(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def vocabulary_index_map(
    categories: list[dict[str, Any]], vocabulary: list[dict[str, Any]]
) -> tuple[dict[int, int], list[int]]:
    """Map TAO category IDs to the exact 0-based LVIS input vocabulary.

    Synset is the primary key.  A unique normalized name/synonym fallback is
    allowed only when it identifies exactly one vocabulary entry; otherwise
    the category is excluded and reported instead of being guessed.
    """
    by_synset = {str(item.get("synset")): index for index, item in enumerate(vocabulary)}
    by_alias: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(vocabulary):
        aliases = [item.get("name", ""), *item.get("synonyms", [])]
        for alias in aliases:
            by_alias[str(alias).replace("_", " ").strip().lower()].append(index)
    mapping: dict[int, int] = {}
    missing: list[int] = []
    for category in categories:
        if category.get("frequency") not in ("c", "f"):
            continue
        synset = str(category.get("synset"))
        index = by_synset.get(synset)
        if index is None:
            aliases = [category.get("name", ""), *category.get("synonyms", [])]
            candidates = {candidate for alias in aliases for candidate in by_alias.get(str(alias).replace("_", " ").strip().lower(), [])}
            if len(candidates) == 1:
                index = candidates.pop()
        if index is None:
            missing.append(int(category["id"]))
        else:
            mapping[int(category["id"])] = index
    return mapping, missing


def greedy_matches(candidates: list[list[float]], annotations: list[dict[str, Any]], minimum_iou: float) -> list[dict[str, Any]]:
    """Produce a deterministic one-to-one candidate/annotation assignment."""
    possible: list[tuple[float, int, int]] = []
    for candidate_index, candidate_box in enumerate(candidates):
        for annotation_index, annotation in enumerate(annotations):
            score = iou_xyxy(candidate_box, annotation["box_xyxy"])
            if score >= minimum_iou:
                possible.append((score, candidate_index, annotation_index))
    possible.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_candidates: set[int] = set()
    used_annotations: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, candidate_index, annotation_index in possible:
        if candidate_index in used_candidates or annotation_index in used_annotations:
            continue
        used_candidates.add(candidate_index)
        used_annotations.add(annotation_index)
        matches.append({
            "candidate_index": candidate_index,
            "annotation_index": annotation_index,
            "iou": score,
        })
    matches.sort(key=lambda item: item["candidate_index"])
    return matches


def _recipe_head():
    from locatemot.groundjev.category import NativeCategoryHead
    from tools.groundjev.streamed_category_media import RECIPE_KEY

    class RecipeHead(NativeCategoryHead):
        def crop_sheet(self, image: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "type": "image",
                "path": str(image),
                "w": 4 * 192,
                "h": max(1, math.ceil(len(candidates) / 4)) * 216,
                RECIPE_KEY: [list(candidate["box"]) for candidate in candidates],
            }

    return RecipeHead(None, group_size=128, batch_candidates=8)


def _build_question_rows(
    *, image_path: str, video_id: int, frame_index: int, matches: list[dict[str, Any]],
    candidates: list[dict[str, Any]], annotations: list[dict[str, Any]], vocabulary: list[dict[str, Any]],
    output: Any, counts: Counter[str],
) -> None:
    from qev.prompt import render_question
    from train import common

    head = _build_question_rows.head
    names = [str(category["name"]).replace("_", " ") for category in vocabulary]
    for offset in range(0, len(matches), head.batch_candidates):
        batch_matches = matches[offset:offset + head.batch_candidates]
        batch = [{"box": candidates[item["candidate_index"]]["box"]} for item in batch_matches]
        targets = [annotations[item["annotation_index"]]["vocabulary_index"] for item in batch_matches]
        request, media, groups = head.build_root_request(image_path, batch, names)
        roots = {index: str(target // head.group_size) for index, target in enumerate(targets)}
        leaf = head.build_leaf_request(request.state, groups, roots)
        if leaf is None:
            raise ValueError("a matched Base category unexpectedly produced no leaf question")
        recipe = media[0]["image"]
        for local, target in enumerate(targets):
            match = batch_matches[local]
            annotation = annotations[match["annotation_index"]]
            for task, question_set, key, answer in (
                ("category_group", request, f"group_{local}", roots[local]),
                ("category_leaf", leaf, f"class_{local}", str(target % head.group_size)),
            ):
                question = question_set.questions[key]
                common.target_vector({answer: 1.0}, render_question("q", question).labels)
                row = {
                    "id": f"native_ovmot_train_{annotation['image_id']}_{offset}_{key}",
                    "split": "train",
                    "source": f"tao_train_{video_id}",
                    "task": task,
                    "state": request.state,
                    "question": question.model_dump(exclude_none=True),
                    "target": {answer: 1.0},
                    "media": [recipe],
                    "supervision": {
                        "frame": frame_index,
                        "image_id": annotation["image_id"],
                        "video_id": video_id,
                        "annotation_id": annotation["id"],
                        "candidate_index": match["candidate_index"],
                        "candidate_iou_to_train_annotation": match["iou"],
                        "Base_vocabulary_index": target,
                        "candidate_source": "native GroundAnything candidate box; official TAO TRAIN label joined offline",
                        "native_candidate_box": True,
                        "native_region_feature_present": True,
                        "GT_class_or_identity_in_state": False,
                        "candidate_query_label_is_not_training_target": True,
                        "GroundAnything_prediction_claim": False,
                        "all1203_input_categories_retained": True,
                        "Novel_GT_supervision": False,
                        "unknown_supervision": False,
                        "temporal_or_actor_history_claim": False,
                        "input_io_protocol": "gzip-jsonl-and-native-category-crop-recipes-v1",
                    },
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[task] += 1
                counts["rows"] += 1
            counts["matched_objects"] += 1
        counts["crop_sheet_recipes"] += 1


_build_question_rows.head = None  # type: ignore[attr-defined]


def build(args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = args.annotation.resolve()
    vocabulary_path = args.vocabulary.resolve()
    candidate_path = args.candidates.resolve()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if not candidate_path.is_file() or not annotation_path.is_file() or not vocabulary_path.is_file():
        raise FileNotFoundError("candidate archive, TAO TRAIN annotation, or vocabulary is missing")

    annotation_source = json.loads(annotation_path.read_text())
    vocabulary_source = json.loads(vocabulary_path.read_text())
    vocabulary = vocabulary_source["categories"]
    if len(vocabulary) != 1203:
        raise ValueError(f"expected the registered 1203-category vocabulary, got {len(vocabulary)}")
    annotation_version = str(annotation_source.get("info", {}).get("version", ""))
    if annotation_version and not annotation_version.startswith("0.1"):
        raise ValueError(f"unexpected TAO annotation version: {annotation_version}")
    category_to_vocab, missing_categories = vocabulary_index_map(annotation_source["categories"], vocabulary)
    category_by_id = {int(item["id"]): item for item in annotation_source["categories"]}
    image_by_key: dict[str, dict[str, Any]] = {}
    annotation_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image in annotation_source["images"]:
        key = str(image["file_name"])
        if not key.startswith("train/"):
            raise ValueError(f"TAO TRAIN annotation contains a non-train image key: {key}")
        if key in image_by_key:
            raise ValueError(f"duplicate TAO TRAIN image path: {key}")
        image_by_key[key] = image
    for annotation in annotation_source["annotations"]:
        category = category_by_id[int(annotation["category_id"])]
        if category.get("frequency") not in ("c", "f") or int(annotation["category_id"]) not in category_to_vocab:
            continue
        annotation_by_image[int(annotation["image_id"])].append({
            **annotation,
            "box_xyxy": xywh_to_xyxy(annotation["bbox"]),
            "vocabulary_index": category_to_vocab[int(annotation["category_id"])],
        })

    args.out.mkdir(parents=True)
    counts: Counter[str] = Counter()
    seen_keys: set[str] = set()
    frame_scope = (args.out / "frame_scope.jsonl").open("w", encoding="utf-8")
    output = gzip.open(args.out / "train.jsonl.gz", "wt", encoding="utf-8", compresslevel=6)
    try:
        for line_no, line in enumerate(candidate_path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            path = str(record.get("image_path", ""))
            key = image_key(path)
            if key in seen_keys:
                raise ValueError(f"duplicate native candidate frame path at line {line_no}: {key}")
            seen_keys.add(key)
            if args.max_frames is not None and len(seen_keys) > args.max_frames:
                break
            image = image_by_key.get(key)
            if image is None:
                raise ValueError(f"native candidate path is absent from TAO TRAIN annotations: {key}")
            perception = record.get("perception", {})
            if perception.get("GT_read") is not False:
                raise ValueError(f"candidate stream does not prove GT-free model inference at line {line_no}")
            candidates = record.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(f"candidate list missing at line {line_no}")
            boxes: list[list[float]] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError(f"candidate is not an object at line {line_no}")
                if not candidate.get("feature_source") or "native" not in str(candidate["feature_source"]).lower():
                    raise ValueError(f"candidate is missing native feature provenance at line {line_no}")
                if "native_region_feature" not in candidate:
                    raise ValueError(f"candidate is missing the native region feature at line {line_no}")
                boxes.append(validate_xyxy(candidate["box"], int(image["width"]), int(image["height"])))
                candidate["box"] = boxes[-1]
            annotations = annotation_by_image[int(image["id"])]
            matches = greedy_matches(boxes, annotations, args.minimum_iou)
            _build_question_rows.head = _recipe_head()  # type: ignore[attr-defined]
            _build_question_rows(
                image_path=path, video_id=int(image["video_id"]), frame_index=int(image["frame_index"]),
                matches=matches, candidates=candidates, annotations=annotations, vocabulary=vocabulary,
                output=output, counts=counts,
            )
            matched_annotation_indices = {item["annotation_index"] for item in matches}
            frame_scope.write(json.dumps({
                "image_id": image["id"], "video_id": image["video_id"], "frame_index": image["frame_index"],
                "image_path": path, "split": "train", "native_candidates": len(candidates),
                "eligible_base_annotations": len(annotations), "matched_objects": len(matches),
                "unmatched_base_annotations": len(annotations) - len(matched_annotation_indices),
                "minimum_iou": args.minimum_iou,
            }, ensure_ascii=False) + "\n")
            counts["frames"] += 1
            counts["native_candidates"] += len(candidates)
            counts["eligible_base_annotations"] += len(annotations)
            counts["unmatched_base_annotations"] += len(annotations) - len(matched_annotation_indices)
    finally:
        output.close()
        frame_scope.close()

    full_scope = args.max_frames is None and len(seen_keys) == len(image_by_key)
    if args.max_frames is None and not full_scope:
        raise ValueError(f"native candidate stream is incomplete: {len(seen_keys)} / {len(image_by_key)} TRAIN images")
    if not counts["matched_objects"]:
        raise ValueError("no native candidate was joined to a Base TAO TRAIN annotation")
    rows_path = args.out / "train.jsonl.gz"
    manifest = {
        "protocol": "native-ovmot-train-candidate-annotation-join-v1",
        "manifest_version": "native-ovmot-train-v1" if full_scope else "native-ovmot-preflight-v1",
        "split": "train",
        "rows": counts["rows"],
        "rows_per_task": {key: counts[key] for key in ("category_group", "category_leaf")},
        "train_rows": str(rows_path),
        "frame_scope": str(args.out / "frame_scope.jsonl"),
        "native_ovmot_training_rows": full_scope,
        "ovmot_training_mode": "native" if full_scope else "preflight",
        "candidate_provenance": {
            "native_ovmot_training_rows": full_scope,
            "candidate_archive": str(candidate_path),
            "candidate_archive_sha256": sha256(candidate_path),
            "tao_train_annotation": str(annotation_path),
            "tao_train_annotation_sha256": sha256(annotation_path),
            "input_vocabulary": str(vocabulary_path),
            "input_vocabulary_sha256": sha256(vocabulary_path),
            "minimum_iou": args.minimum_iou,
            "assignment": "greedy one-to-one geometry match; candidate boxes unchanged; labels from TAO TRAIN only",
            "model_inference_GT_read": False,
            "native_per_box_score_required": False,
        },
        "scope": {
            "frames_seen": counts["frames"], "tao_train_images": len(image_by_key),
            "full_train_image_scope": full_scope, "missing_categories_without_vocabulary_index": missing_categories,
        },
        "counts": dict(counts),
        "optimizer_updates": 0,
        "gpu_launched": False,
        "full_goal_complete": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--minimum-iou", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None, help="preflight only; output cannot pass the native training gate")
    args = parser.parse_args()
    if not (0.0 < args.minimum_iou <= 1.0):
        parser.error("--minimum-iou must be in (0, 1]")
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
