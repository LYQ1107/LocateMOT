"""Independent L78 raw full-frame/complete-L69-row helpers.

The helper deliberately keeps feature construction label-free.  L69 sidecar
labels are loaded only through ``attach_labels`` after a caller has materialized
the complete native row list and raw feature representation.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L49_ROOT = ROOT / "outputs/l49/data"
L62_RECORDS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
TEXT_CACHE = ROOT / "outputs/l48/data/text_cache.pt"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
RAW_ROOT = ROOT / "data/kitti_tracking_training/image_02"
CLIP_WEIGHTS = Path("/home/lwr/.cache/clip/ViT-B-16.pt").resolve()
EXPECTED_CLIP_SHA = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
L69_VIDEOS = (
    "0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008",
    "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018", "0020",
)
FORBIDDEN_LABEL_FIELDS = (
    "target_ids", "positive_indices", "positive_count", "category", "labels",
    "target_present", "candidate_gt",
)
L29_VALIDATION_CONTROL = {
    "recall": 0.7333333333333333,
    "precision": 0.0830188679245283,
    "fp_per_frame": 10.125,
    "predictions_per_positive": 8.833333333333334,
    "hard_violation": 0.9166666666666666,
    "multi_positive_recall": 0.8194444444444443,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def unit_key(unit: dict[str, Any]) -> str:
    if unit.get("unit_key"):
        return str(unit["unit_key"])
    return f"{unit['dataset']}|{unit['video']}|{int(unit['query_id'])}|{int(unit['frame_id'])}"


def image_path(video: str, frame: int) -> Path:
    return RAW_ROOT / str(video) / f"{int(frame):06d}.png"


def fixed_key_order() -> list[str]:
    rows = []
    for line in L62_RECORDS.read_text().splitlines():
        if line.strip():
            rows.append(str(json.loads(line)["unit_key"]))
    if len(rows) != 40 or len(set(rows)) != 40:
        raise AssertionError(f"fixed L62 order must contain 40 unique keys, got {len(rows)}")
    return rows


def fixed_key_metadata() -> list[dict[str, Any]]:
    """Join only non-label metadata to the immutable L62 key order."""
    allowed = ("dataset", "video", "query_id", "frame_id", "sentence", "expression", "unit_key")
    lookup: dict[str, dict[str, Any]] = {}
    for filename in ("calibration_units.jsonl", "validation_units.jsonl"):
        for row in read_jsonl(L49_ROOT / filename):
            value = {key: row[key] for key in allowed if key in row}
            key = unit_key(value)
            if any(field in value for field in FORBIDDEN_LABEL_FIELDS):
                raise AssertionError(f"label field retained in key-only metadata {key}")
            if key in lookup:
                raise AssertionError(f"duplicate fixed metadata {key}")
            lookup[key] = value
    result = []
    for order, key in enumerate(fixed_key_order()):
        if key not in lookup:
            raise KeyError(f"fixed key missing from L49 metadata: {key}")
        row = dict(lookup[key])
        row["fixed_eval_order"] = order
        row["fixed_eval_split"] = "calibration" if order < 16 else "validation"
        result.append(row)
    return result


def authorized_fixed_labels(orders: Iterable[int]) -> dict[int, dict[str, Any]]:
    """Load labels for explicit calibration/validation orders only."""
    selected = sorted({int(x) for x in orders})
    if any(x < 0 or x >= 40 for x in selected):
        raise ValueError(selected)
    keys = fixed_key_order()
    result: dict[int, dict[str, Any]] = {}
    for role, filename in (("calibration", "calibration_units.jsonl"), ("validation", "validation_units.jsonl")):
        role_orders = [x for x in selected if (x < 16) == (role == "calibration")]
        wanted = {keys[x] for x in role_orders}
        found = {str(row["unit_key"]): row for row in read_jsonl(L49_ROOT / filename) if str(row.get("unit_key")) in wanted}
        if set(found) != wanted:
            raise KeyError(f"missing authorized {role} labels: {sorted(wanted - set(found))}")
        for order in role_orders:
            result[order] = found[keys[order]]
    if set(result) != set(selected):
        raise AssertionError("authorized fixed labels incomplete")
    return result


def fit_units() -> list[dict[str, Any]]:
    rows = read_jsonl(L49_ROOT / "train_units.jsonl")
    rows = [row for row in rows if row.get("split") == "fit" and row.get("dataset") in DATASETS]
    if len(rows) != 5314:
        raise AssertionError(f"expected 5314 fit rows, got {len(rows)}")
    return rows


def make_fit_schedule(steps: int, seed: int = 20260829) -> list[dict[str, Any]]:
    rows = fit_units()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["dataset"]), str(row["category"]))].append(row)
    required = {(dataset, category) for dataset in DATASETS for category in
                ("positive", "multi_positive", "inactive", "present_uncovered")}
    if set(buckets) != required:
        raise AssertionError(f"fit strata mismatch: {sorted(set(buckets) ^ required)}")
    rng = np.random.default_rng(int(seed))
    keys = sorted(required)
    shuffled: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key in keys:
        values = sorted(buckets[key], key=lambda x: (str(x["video"]), int(x["query_id"]), int(x["frame_id"]), unit_key(x)))
        order = rng.permutation(len(values)).tolist()
        shuffled[key] = [values[int(i)] for i in order]
    result = []
    for position in range(int(steps)):
        key = keys[position % len(keys)]
        value = dict(shuffled[key][(position // len(keys)) % len(shuffled[key])])
        value["schedule_position"] = position
        value["train_step"] = position + 1
        result.append(value)
    if {str(x["dataset"]) for x in result} != set(DATASETS):
        raise AssertionError("schedule missed domain")
    if {str(x["category"]) for x in result} != {"positive", "multi_positive", "inactive", "present_uncovered"}:
        raise AssertionError("schedule missed category")
    return result


class L78Bank:
    """One-video native L69 bank; sidecar labels are lazy."""
    def __init__(self, video: str):
        self.video = str(video)
        if self.video not in L69_VIDEOS:
            raise ValueError(f"video outside L69 train pool: {video}")
        self.path = L69_ROOT / f"{self.video}.pt"
        self.label_path = self.path.with_suffix(".labels.json")
        self.blob = safe_torch_load(self.path)
        self.tensors = self.blob["tensors"]
        self.metadata = dict(self.blob.get("metadata", {}))
        self.labels: list[str | None] | None = None
        self.frame_ranges: dict[int, tuple[int, int, int]] = {}
        self._validate()

    def _validate(self) -> None:
        t = self.tensors
        required = {"frame", "frame_ids", "frame_ptr", "candidate_index", "track_id", "box", "pool_id", "raw_rank"}
        missing = required.difference(t)
        if missing:
            raise KeyError(f"{self.path}: missing {sorted(missing)}")
        count = int(t["track_id"].numel())
        ptr = t["frame_ptr"].long().tolist()
        ids = t["frame_ids"].long().tolist()
        if len(ptr) != len(ids) + 1 or ptr[-1] != count:
            raise AssertionError(f"{self.path}: frame pointer contract")
        for name in ("frame", "candidate_index", "track_id", "box", "pool_id", "raw_rank", "objectness"):
            if name in t and int(t[name].shape[0]) != count:
                raise AssertionError(f"{self.path}: row mismatch {name}")
        if t["box"].ndim != 2 or tuple(t["box"].shape[1:]) != (4,):
            raise AssertionError(f"{self.path}: box shape")
        if not torch.isfinite(t["box"].float()).all():
            raise AssertionError(f"{self.path}: nonfinite boxes")
        frames = t["frame"].long()
        for index, frame_id in enumerate(ids):
            begin, end = int(ptr[index]), int(ptr[index + 1])
            if begin < 0 or end < begin or not torch.equal(frames[begin:end], torch.full((end - begin,), int(frame_id), dtype=frames.dtype)):
                raise AssertionError(f"{self.path}: frame slice mismatch {frame_id}")
            self.frame_ranges[int(frame_id)] = (begin, end, index)

    @property
    def count(self) -> int:
        return int(self.tensors["track_id"].numel())

    def label_values(self) -> list[str | None]:
        if self.labels is None:
            if not self.label_path.is_file():
                raise FileNotFoundError(self.label_path)
            values = json.loads(self.label_path.read_text()).get("candidate_gt", [])
            if len(values) != self.count:
                raise AssertionError(f"{self.label_path}: label length mismatch")
            self.labels = [None if value is None else str(value) for value in values]
        return self.labels

    def label_free_record(self, unit: dict[str, Any]) -> dict[str, Any]:
        if str(unit["video"]) != self.video:
            raise AssertionError("unit/video mismatch")
        frame_id = int(unit["frame_id"])
        if frame_id not in self.frame_ranges:
            raise KeyError(f"frame missing {unit_key(unit)}")
        begin, end, frame_index = self.frame_ranges[frame_id]
        rows = list(range(begin, end))
        t = self.tensors
        row_keys = [[str(unit["dataset"]), self.video, int(unit["query_id"]), frame_id, str(self.path), int(row)] for row in rows]
        if [int(key[-1]) for key in row_keys] != rows:
            raise AssertionError(f"row order drift {unit_key(unit)}")
        boxes = t["box"][begin:end].float().clone()
        if boxes.shape != (len(rows), 4) or not torch.isfinite(boxes).all():
            raise AssertionError(f"box contract {unit_key(unit)}")
        return {
            "format": "locatemot-l78-label-free-row-v1",
            "status": "complete",
            "unit_key": unit_key(unit),
            "dataset": str(unit["dataset"]), "video": self.video,
            "query_id": int(unit["query_id"]), "frame_id": frame_id,
            "sentence": str(unit.get("sentence", unit.get("expression", ""))),
            "expression": str(unit.get("expression", "")),
            "frame_index": int(frame_index), "bank_path": str(self.path),
            "row_offsets": rows, "row_keys": row_keys,
            "candidate_index_provenance": t["candidate_index"][begin:end].long().tolist(),
            "track_id_provenance": t["track_id"][begin:end].long().tolist(),
            "pool_id_provenance": t["pool_id"][begin:end].long().tolist(),
            "raw_rank_provenance": t["raw_rank"][begin:end].long().tolist(),
            "candidate_count": len(rows), "boxes": boxes,
            "image_size_bank": list(self.metadata.get("image_size", [])),
            "candidate_deletion": False, "candidate_truncation": False,
            "old_l49_ranges_used": False, "old_l49_positive_indices_used": False,
            "ids_are_provenance_only": True,
        }

    def attach_labels(self, record: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
        values = self.label_values()
        begin = int(record["row_offsets"][0]) if record["row_offsets"] else int(self.frame_ranges[int(record["frame_id"])][0])
        end = begin + int(record["candidate_count"])
        targets = {str(value) for value in (unit.get("target_ids") or [])}
        sidecar = values[begin:end]
        membership = [value is not None and str(value) in targets for value in sidecar]
        positives = [int(i) for i, value in enumerate(membership) if value]
        category = "multi_positive" if len(positives) > 1 else "positive" if positives else "present_uncovered" if targets else "inactive"
        result = dict(record)
        result.update({
            "target_ids": sorted(targets), "sidecar_candidate_gt": sidecar,
            "labels": [int(x) for x in membership], "positive_indices": positives,
            "positive_count": len(positives), "target_present": bool(targets),
            "candidate_present": bool(positives),
            "coverage_mask": not (bool(targets) and not bool(positives)),
            "null_target": int(not bool(targets)), "category": category,
            "label_source": str(self.label_path),
        })
        if len(result["labels"]) != int(record["candidate_count"]):
            raise AssertionError(f"label length drift {record['unit_key']}")
        return result

    def close(self) -> None:
        self.blob = None
        self.tensors = {}
        self.metadata = {}
        self.labels = None
        self.frame_ranges = {}


def padding_box(box: Iterable[float], width: int, height: int, padding: float = 0.10) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = max(0.0, x1 - padding * bw); y1 = max(0.0, y1 - padding * bh)
    x2 = min(float(width), x2 + padding * bw); y2 = min(float(height), y2 + padding * bh)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty candidate box after padding: {box}")
    return x1, y1, x2, y2


def letterbox_image(image: Image.Image, size: int = 224) -> tuple[Image.Image, dict[str, float | int]]:
    width, height = image.size
    scale = min(float(size) / max(1, width), float(size) / max(1, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    mean = (int(round(0.48145466 * 255)), int(round(0.4578275 * 255)), int(round(0.40821073 * 255)))
    canvas = Image.new("RGB", (size, size), mean)
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas, {"original_width": width, "original_height": height, "scale": scale,
                    "resized_width": resized_width, "resized_height": resized_height,
                    "offset_x": offset_x, "offset_y": offset_y, "output_size": size}


def boxes_to_normalized(boxes: torch.Tensor, geometry: dict[str, float | int], padding: float = 0.10) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    width, height = int(geometry["original_width"]), int(geometry["original_height"])
    scale, ox, oy, size = float(geometry["scale"]), float(geometry["offset_x"]), float(geometry["offset_y"]), float(geometry["output_size"])
    normalized = []
    details = []
    for value in boxes.float().tolist():
        px = padding_box(value, width, height, padding)
        mapped = [px[0] * scale + ox, px[1] * scale + oy, px[2] * scale + ox, px[3] * scale + oy]
        norm = [max(0.0, min(1.0, x / size)) for x in mapped]
        if norm[2] < norm[0] or norm[3] < norm[1]:
            raise AssertionError("normalized box inversion")
        normalized.append(norm)
        details.append({"original_box": value, "padded_clipped_box": list(px), "resized_box": mapped, "normalized_box": norm,
                        "empty": False, "padding": padding})
    result = torch.tensor(normalized, dtype=torch.float32)
    if result.shape != (len(boxes), 4) or not torch.isfinite(result).all():
        raise AssertionError("normalized boxes finite/shape contract")
    return result, details


class StreamingOpenAIClipFullFrame:
    """Frozen OpenAI CLIP ViT-B/16 exposing a full 14x14 patch map."""
    def __init__(self, device: str = "cuda:0"):
        import clip
        self.device = torch.device(device)
        if not CLIP_WEIGHTS.is_file():
            raise FileNotFoundError(CLIP_WEIGHTS)
        if sha256_file(CLIP_WEIGHTS) != EXPECTED_CLIP_SHA:
            raise AssertionError("CLIP ViT-B/16 SHA mismatch")
        self.clip = clip
        self.model, self.preprocess = clip.load(str(CLIP_WEIGHTS), device=self.device, jit=False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dtype = self.model.visual.conv1.weight.dtype

    @torch.inference_mode()
    def image_map(self, image_path_value: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if not image_path_value.is_file():
            raise FileNotFoundError(image_path_value)
        with Image.open(image_path_value) as source:
            image = source.convert("RGB")
            canvas, geometry = letterbox_image(image)
            pixel = self.preprocess(canvas).unsqueeze(0).to(self.device, dtype=self.dtype)
            visual = self.model.visual
            x = visual.conv1(pixel)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
            x = torch.cat((cls, x), dim=1) + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x).permute(1, 0, 2)
            x = visual.transformer(x).permute(1, 0, 2)
            x = visual.ln_post(x)
            if visual.proj is None:
                raise AssertionError("OpenAI CLIP visual.proj is unexpectedly None")
            projected = x @ visual.proj
            side = int(round((projected.shape[1] - 1) ** 0.5))
            if side * side != projected.shape[1] - 1:
                raise AssertionError(f"unexpected CLIP patch shape {tuple(projected.shape)}")
            global_token = projected[:, 0].float().squeeze(0).clone()
            spatial = projected[:, 1:].float().transpose(1, 2).reshape(1, 512, side, side).clone()
            geometry.update({"image_path": str(image_path_value), "spatial_side": side,
                             "patch_count": side * side, "pixel_shape": list(pixel.shape),
                             "visual_dtype": str(self.dtype)})
            del pixel, projected, x, canvas, image
        if not torch.isfinite(spatial).all() or not torch.isfinite(global_token).all():
            raise FloatingPointError(f"nonfinite CLIP full-frame output {image_path_value}")
        return spatial, global_token, geometry

    @torch.inference_mode()
    def text_tokens(self, sentence: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.clip.tokenize([str(sentence)], truncate=True).to(self.device)
        model = self.model
        x = model.token_embedding(tokens).to(model.dtype)
        x = x + model.positional_embedding.to(x.dtype)
        x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = model.ln_final(x)
        if model.text_projection is None:
            raise AssertionError("OpenAI CLIP text_projection is unexpectedly None")
        x = x @ model.text_projection
        valid = tokens[0].ne(0).bool()
        return x.float()[0].clone(), valid.cpu().clone(), tokens[0].cpu().clone()

    def frozen_contract(self) -> dict[str, Any]:
        return {"all_parameters_requires_grad_false": all(not p.requires_grad for p in self.model.parameters()),
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "dtype": str(self.dtype), "weight": str(CLIP_WEIGHTS), "weight_sha256": sha256_file(CLIP_WEIGHTS),
                "text_projection": list(self.model.text_projection.shape),
                "visual_projection": list(self.model.visual.proj.shape)}


def select_label_free_audit_units() -> list[dict[str, Any]]:
    """Select declared fit strata before any candidate sidecar is loaded."""
    rows = fit_units()
    chosen = []
    for dataset in DATASETS:
        for category in ("positive", "multi_positive", "inactive", "present_uncovered"):
            match = sorted((x for x in rows if x["dataset"] == dataset and x["category"] == category),
                           key=lambda x: (str(x["video"]), int(x["query_id"]), int(x["frame_id"]), unit_key(x)))
            if not match:
                raise AssertionError(f"no audit unit for {dataset}/{category}")
            chosen.append(dict(match[0]))
    return chosen


def dist(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"count": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max())}


def simple_svg_boxes(path: Path, geometry: dict[str, Any], boxes: torch.Tensor, scores: Iterable[float] | None = None) -> None:
    """Small diagnostic SVG; it contains no source pixels or labels."""
    size = int(geometry["output_size"])
    score_values = list(scores) if scores is not None else [0.0] * len(boxes)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">']
    for box, score in zip(boxes.tolist(), score_values):
        x1, y1, x2, y2 = [float(x) * size for x in box]
        svg.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{max(0,x2-x1):.2f}" height="{max(0,y2-y1):.2f}" fill="none" stroke="#2c7" stroke-width="0.7"><title>{float(score):.5f}</title></rect>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n")
