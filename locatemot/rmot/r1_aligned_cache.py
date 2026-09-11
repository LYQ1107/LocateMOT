"""R1 data/index and aligned-state cache helpers.

The cache builder is the only component that talks to the local frozen
GroundingDINO runtime.  Its payload is query-conditioned but label-free:
Z0/Z1/Z4 and compact provenance are written, never targets or scores.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from locatemot.models.l82_grounding_reference import (
    boxes_to_reference_points,
    boxes_xyxy_to_normalized,
    candidate_seed_with_reference,
    pool_memory_by_box,
)


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[2]
R0A_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R0A").resolve()
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L49_DATA = ROOT / "outputs/l49/data"
R0_VISUAL_MANIFEST = Path("/data2/usr_for_deadline/locatemot_r0a_visual_tokens_retry3_final").resolve()
LANGUAGE_ROOTS = (
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/cache/language_tokens_retry1").resolve(),
    Path("/data2/usr_for_deadline/locatemot_r0a_language_tokens_retry1").resolve(),
)
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
OBS_FIELDS = ("clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle", "objectness")
OBS_DIM = 1432

if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
            "sha256": sha256_file(path) if path.is_file() else None}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def image_path(video: str, frame_id: int) -> Path:
    path = ROOT / "data/kitti_tracking_training/image_02" / str(video) / f"{int(frame_id):06d}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def row_digest(row_keys: Iterable[Iterable[Any]]) -> str:
    data = "\n".join("|".join(str(value) for value in key) for key in row_keys).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def observation_from_rows(tensors: dict[str, torch.Tensor], offsets: list[int]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for field in OBS_FIELDS:
        value = tensors[field][offsets].float()
        if field == "objectness":
            value = value.reshape(-1, 1)
        if value.ndim != 2:
            raise ValueError(f"R1 observation field is not [N,D]: {field} {tuple(value.shape)}")
        pieces.append(value)
    result = torch.cat(pieces, dim=-1)
    if result.shape != (len(offsets), OBS_DIM) or not bool(torch.isfinite(result).all()):
        raise FloatingPointError("R1 observation shape/finite contract failed")
    return result


class R1BankStore:
    """Streaming L69 reader whose GT sidecar is opened only on attachment.

    The existing R0 reader is intentionally useful for R0 experiments, but it
    eagerly loads ``candidate_gt`` while loading a video.  R1 keeps the
    feature/cache boundary stricter: bank tensors and row structure are
    enough for feature construction; the sidecar is read only by the
    explicit ``attach_frame_labels`` call after that construction returns.
    """

    def __init__(self) -> None:
        self._video: str | None = None
        self._path: Path | None = None
        self._blob: dict[str, Any] | None = None
        self._track_rows: dict[int, list[int]] = {}

    @property
    def tensors(self) -> dict[str, Any]:
        if self._blob is None:
            raise RuntimeError("R1 bank store has no loaded video")
        return self._blob["tensors"]

    @property
    def bank_path(self) -> Path:
        if self._path is None:
            raise RuntimeError("R1 bank store has no loaded video")
        return self._path

    @property
    def track_rows(self) -> dict[int, list[int]]:
        return self._track_rows

    def load_video(self, video: str) -> None:
        if str(video) == self._video:
            return
        from locatemot.rmot.r0_dense_data import load_l69_bank

        path, blob = load_l69_bank(str(video))
        tensors = blob["tensors"]
        frames = [int(value) for value in tensors["frame"].long().tolist()]
        tracks = [int(value) for value in tensors["track_id"].long().tolist()]
        rows: dict[int, list[int]] = {}
        for offset, track in enumerate(tracks):
            rows.setdefault(track, []).append(offset)
        for values in rows.values():
            values.sort(key=lambda offset: (frames[offset], offset))
        self._video, self._path, self._blob, self._track_rows = str(video), path, blob, rows

    def build_frame(self, dataset: str, video: str, query_id: int, frame_id: int, sentence: str) -> Any:
        from locatemot.rmot.r0_dense_data import R0FrameBatch, native_frame_slice

        self.load_video(str(video))
        tensors = self.tensors
        frame_ids = [int(value) for value in tensors["frame_ids"].long().tolist()]
        if int(frame_id) not in frame_ids:
            raise KeyError(f"R1 frame {frame_id} missing from {video}")
        position = frame_ids.index(int(frame_id))
        frame, begin, end = native_frame_slice(tensors, position)
        offsets = list(range(begin, end))
        if frame != int(frame_id):
            raise AssertionError("R1 frame pointer identity drift")
        candidate_indices = [int(value) for value in tensors["candidate_index"][begin:end].tolist()]
        track_ids = [int(value) for value in tensors["track_id"][begin:end].tolist()]
        pool_ids = [int(value) for value in tensors["pool_id"][begin:end].tolist()]
        boxes = tensors["box"][begin:end].float().clone()
        width, height = [int(value) for value in self._blob["metadata"]["image_size"][:2]]
        keys = [(str(dataset), str(video), int(query_id), int(frame_id), str(self.bank_path), int(offset)) for offset in offsets]
        if keys != sorted(keys, key=lambda value: value[-1]) or len(set(keys)) != len(keys):
            raise AssertionError("R1 immutable row key/order drift")
        return R0FrameBatch(str(dataset), str(video), int(query_id), int(frame_id), str(sentence), str(self.bank_path),
                            offsets, keys, candidate_indices, track_ids, pool_ids, boxes, (width, height))

    def attach_frame_labels(self, batch: Any, target_ids: Iterable[object]) -> dict[str, Any]:
        """Open the bank sidecar only at the post-feature supervision boundary."""
        label_path = Path(batch.bank_path).with_suffix(".labels.json")
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        values = payload.get("candidate_gt")
        if not isinstance(values, list) or len(values) != int(self.tensors["track_id"].numel()):
            raise ValueError(f"R1 candidate sidecar/row mismatch: {label_path}")
        target_set = {str(value) for value in target_ids}
        selected = [None if values[offset] is None else str(values[offset]) for offset in batch.row_offsets]
        labels = torch.tensor([value is not None and value in target_set for value in selected], dtype=torch.bool)
        candidate_targets = {value for value in selected if value is not None}
        covered = target_set & candidate_targets
        target_present = bool(target_set)
        candidate_present = bool(covered)
        if not target_present:
            category = "inactive"
        elif not candidate_present:
            category = "present_uncovered"
        elif len(covered) > 1:
            category = "multi_positive"
        else:
            category = "positive"
        return {
            "labels": labels, "candidate_gt": selected, "target_ids": sorted(target_set),
            "covered_target_ids": sorted(covered), "category": category,
            "target_present": target_present, "candidate_present": candidate_present,
            "present_uncovered": bool(target_present and not candidate_present),
            "positive_row_count": int(labels.sum()), "positive_count": int(labels.sum()),
            "membership_mask": torch.full_like(labels, not (target_present and not candidate_present)),
            "labels_attached_after_feature_construction": True,
        }


def build_causal_history(store: Any, current_batch: Any, history_length: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return [N,8,1432], mask and frame IDs using only rows <= current frame."""
    if history_length != 8:
        raise ValueError("R1 history length is fixed at 8")
    tensors = store.tensors
    offsets = [int(value) for value in current_batch.row_offsets]
    frames = tensors["frame"].long()
    tracks = tensors["track_id"].long()
    current_frame = int(current_batch.frame_id)
    all_by_track = store.track_rows
    history = torch.zeros((len(offsets), history_length, OBS_DIM), dtype=torch.float32)
    mask = torch.zeros((len(offsets), history_length), dtype=torch.bool)
    frame_ids = torch.full((len(offsets), history_length), -1, dtype=torch.long)
    for output_index, offset in enumerate(offsets):
        track = int(tracks[offset])
        eligible = [int(row) for row in all_by_track.get(track, [offset]) if int(frames[row]) <= current_frame]
        if offset not in eligible:
            eligible.append(offset)
        eligible = sorted(set(eligible), key=lambda row: (int(frames[row]), int(row)))
        chosen = eligible[-history_length:]
        values = observation_from_rows(tensors, chosen)
        start = history_length - len(chosen)
        history[output_index, start:] = values
        mask[output_index, start:] = True
        frame_ids[output_index, start:] = frames[chosen]
        if bool((frame_ids[output_index][mask[output_index]] > current_frame).any()):
            raise AssertionError("R1 history contains a future observation")
    if not bool(torch.isfinite(history).all()) or int((frame_ids[mask] > current_frame).sum()) != 0:
        raise FloatingPointError("R1 causal history finite/future contract failed")
    return history, mask, frame_ids


def build_geometry(store: Any, batch: Any) -> torch.Tensor:
    from locatemot.rmot.r0_geometry import build_track_geometry
    result = build_track_geometry(store, batch, batch.image_hw, history_length=4)
    if result.shape != (batch.candidate_count, 5, 10) or not bool(torch.isfinite(result).all()):
        raise AssertionError("R1 geometry contract drift")
    return result.float()


@dataclass
class R1FrameInput:
    dataset: str
    video: str
    query_id: int
    frame_id: int
    sentence: str
    bank_path: str
    row_offsets: list[int]
    row_keys: list[list[Any]]
    candidate_indices: list[int]
    track_ids: list[int]
    pool_ids: list[int]
    boxes: torch.Tensor
    boxes_norm: torch.Tensor
    image_size: tuple[int, int]
    image_path: str
    raw_visual_tokens: torch.Tensor
    text_tokens: torch.Tensor
    text_mask: torch.Tensor
    text_global: torch.Tensor
    frame_global: torch.Tensor
    geometry: torch.Tensor
    history_observations: torch.Tensor
    history_mask: torch.Tensor
    history_frame_ids: torch.Tensor
    z0: torch.Tensor
    z1: torch.Tensor
    z4: torch.Tensor
    base: dict[str, torch.Tensor] | None = None
    supervision: dict[str, Any] | None = None

    @property
    def candidate_count(self) -> int:
        return len(self.row_offsets)


class R1AlignedCacheIndex:
    """Random access over compact cache manifests; tensors are loaded per query."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        self.entries: dict[tuple[str, str, int, int], tuple[Path, int]] = {}
        self.group_paths: dict[tuple[str, str, int], Path] = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            path = Path(str(item["path"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            group_key = (str(item["dataset"]), str(item["video"]), int(item["frame_id"]))
            if group_key in self.group_paths and self.group_paths[group_key] != path:
                raise AssertionError(f"R1 aligned group path drift: {group_key}")
            self.group_paths[group_key] = path
            keys = item.get("query_unit_keys", [])
            query_ids = item.get("query_ids", [])
            for index, (unit_key, query_id) in enumerate(zip(keys, query_ids)):
                dataset, video, found_query, frame = str(unit_key).split("|", 3)
                if int(found_query) != int(query_id):
                    raise AssertionError("R1 cache query key mismatch")
                key = (dataset, video, int(query_id), int(frame))
                if key in self.entries:
                    raise AssertionError(f"duplicate R1 aligned cache key: {key}")
                self.entries[key] = (path, index)
        if not self.entries:
            raise AssertionError("empty R1 aligned cache manifest")

    def read_group(self, dataset: str, video: str, frame_id: int) -> dict[str, Any]:
        """Read one frame payload once for efficient full-video replay."""
        key = (str(dataset), str(video), int(frame_id))
        path = self.group_paths.get(key)
        if path is None:
            raise KeyError(f"R1 aligned cache group miss: {key}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = ("z0", "z1", "z4", "query_unit_keys", "query_ids", "row_offsets", "candidate_indices")
        if not isinstance(payload, dict) or any(field not in payload for field in required):
            raise AssertionError(f"invalid R1 cache group payload: {path}")
        if str(payload.get("dataset")) != str(dataset) or str(payload.get("video")) != str(video) or int(payload.get("frame_id", -1)) != int(frame_id):
            raise AssertionError(f"R1 cache group identity drift: {path}")
        if int(payload.get("candidate_count", -1)) != len(payload["row_offsets"]):
            raise AssertionError(f"R1 cache group candidate count drift: {path}")
        for name in ("z0", "z1", "z4"):
            value = payload[name]
            expected = (len(payload["query_ids"]), len(payload["row_offsets"]), 256)
            if not torch.is_tensor(value) or tuple(value.shape) != expected or not bool(torch.isfinite(value.float()).all()):
                raise AssertionError(f"R1 cache group {name} shape/finite drift: {path}")
        for name in ("text_global", "frame_global"):
            value = payload.get(name)
            if value is not None:
                expected = (len(payload["query_ids"]), 256)
                if not torch.is_tensor(value) or tuple(value.shape) != expected or not bool(torch.isfinite(value.float()).all()):
                    raise AssertionError(f"R1 cache group {name} shape/finite drift: {path}")
        return payload

    def get(self, dataset: str, video: str, query_id: int, frame_id: int) -> dict[str, Any]:
        key = (str(dataset), str(video), int(query_id), int(frame_id))
        found = self.entries.get(key)
        if found is None:
            raise KeyError(f"R1 aligned cache miss: {key}")
        path, index = found
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = ("z0", "z1", "z4", "query_unit_keys", "row_offsets", "candidate_indices")
        if not isinstance(payload, dict) or any(field not in payload for field in required):
            raise AssertionError(f"invalid R1 cache payload: {path}")
        unit_key = f"{dataset}|{video}|{int(query_id)}|{int(frame_id)}"
        if str(payload["query_unit_keys"][index]) != unit_key:
            raise AssertionError("R1 cache unit key drift")
        result = dict(payload)
        # Only these fields have a leading query dimension.  Do not infer
        # that dimension from shape equality: a frame can legitimately have
        # as many candidates as query records, and boxes/row metadata must
        # remain frame-level values.
        for key in ("z0", "z1", "z4", "text_global", "frame_global", "query_audits"):
            if key in payload:
                result[key] = payload[key][index]
        for name in ("z0", "z1", "z4"):
            value = result[name]
            if not torch.is_tensor(value) or value.shape != (len(payload["row_offsets"]), 256) or not bool(torch.isfinite(value.float()).all()):
                raise AssertionError(f"R1 cache {name} shape/finite drift: {path}")
        return result


class R1FeatureAssembler:
    """Combine immutable L69/R0 visual/language rows with one aligned cache item."""

    def __init__(self, aligned: R1AlignedCacheIndex, device: torch.device,
                 visual_roots: Iterable[Path] | None = None) -> None:
        from tools.r0_common import MergedLanguageCache, VisualCacheIndex

        self.aligned = aligned
        self.device = device
        self.language = MergedLanguageCache(LANGUAGE_ROOTS)
        roots = list(visual_roots) if visual_roots is not None else [R0_VISUAL_MANIFEST]
        if not roots:
            raise ValueError("R1 requires at least one visual cache root")
        self.visuals = [VisualCacheIndex(Path(root).resolve()) for root in roots]
        self.store = R1BankStore()

    def _visual_get(self, dataset: str, video: str, frame_id: int) -> dict[str, Any]:
        found: list[dict[str, Any]] = []
        for visual in self.visuals:
            try:
                found.append(visual.get(dataset, video, frame_id))
            except KeyError:
                continue
        if not found:
            raise KeyError(f"R1 visual cache miss: {dataset}|{video}|{frame_id}")
        first = found[0]
        for other in found[1:]:
            for field in ("row_offsets", "candidate_indices", "track_ids", "pool_ids"):
                if list(other.get(field, [])) != list(first.get(field, [])):
                    raise AssertionError(f"R1 duplicate visual cache row drift: {dataset}|{video}|{frame_id}")
        return first

    def prepare(self, record: dict[str, Any], attach_labels: bool = False) -> R1FrameInput:
        dataset, video = str(record["dataset"]), str(record["video"])
        frame_id, query_id = int(record["frame_id"]), int(record["query_id"])
        sentence = str(record["sentence"])
        self.store.load_video(video)
        batch = self.store.build_frame(dataset, video, query_id, frame_id, sentence)
        cache = self.aligned.get(dataset, video, query_id, frame_id)
        visual = self._visual_get(dataset, video, frame_id)
        expected = list(range(int(visual["row_offsets"][0]), int(visual["row_offsets"][-1]) + 1)) if visual["row_offsets"] else []
        if batch.row_offsets != expected or list(cache["row_offsets"]) != expected:
            raise AssertionError(f"R1 native row-offset drift: {record.get('unit_key')}")
        if int(cache["candidate_count"]) != batch.candidate_count or int(record.get("candidate_count", batch.candidate_count)) != batch.candidate_count:
            raise AssertionError(f"R1 candidate count drift: {record.get('unit_key')}")
        if [int(x) for x in cache["candidate_indices"]] != batch.candidate_indices:
            raise AssertionError("R1 candidate-index order drift")
        rows = [list(key) for key in batch.row_keys]
        if len(rows) != batch.candidate_count or len({tuple(key) for key in rows}) != len(rows):
            raise AssertionError("R1 row-key completeness drift")
        # Immutable cache values cross into the sidecar only through an
        # explicit clone and device transfer.  This keeps frozen-cache data
        # out of accidental autograd aliases.
        raw = torch.cat((visual["inner_tokens"].float(), visual["context_tokens"].float()), dim=1).to(self.device).clone()
        if raw.shape != (batch.candidate_count, 72, 256) or not bool(torch.isfinite(raw).all()):
            raise AssertionError("R1 raw visual token shape/finite drift")
        language = self.language.get(dataset, video, query_id, sentence)
        text = language.tokens.float().clone()
        mask = language.mask.bool().clone()
        text_global = (text * mask[:, None].float()).sum(dim=0) / mask.sum().clamp_min(1)
        history, history_mask, history_frames = build_causal_history(self.store, batch)
        history = history.to(self.device).clone()
        history_mask = history_mask.to(self.device).clone()
        history_frames = history_frames.to(self.device).clone()
        geometry = build_geometry(self.store, batch).to(self.device).clone()
        result = R1FrameInput(
            dataset=dataset, video=video, query_id=query_id, frame_id=frame_id, sentence=sentence,
            bank_path=str(batch.bank_path), row_offsets=list(batch.row_offsets), row_keys=rows,
            candidate_indices=list(batch.candidate_indices), track_ids=list(batch.track_ids), pool_ids=list(batch.pool_ids),
            boxes=batch.boxes.float().to(self.device).clone(), boxes_norm=torch.as_tensor(visual["boxes_normalized"]).float().to(self.device).clone(),
            image_size=tuple(int(x) for x in batch.image_size), image_path=str(image_path(video, frame_id)),
            raw_visual_tokens=raw, text_tokens=text.to(self.device).clone(), text_mask=mask.to(self.device).clone(), text_global=text_global.float().to(self.device).clone(),
            frame_global=cache["frame_global"].float().to(self.device).clone(),
            geometry=geometry, history_observations=history, history_mask=history_mask,
            history_frame_ids=history_frames, z0=cache["z0"].float().to(self.device).clone(), z1=cache["z1"].float().to(self.device).clone(),
            z4=cache["z4"].float().to(self.device).clone(),
        )
        if (result.boxes_norm.shape != (result.candidate_count, 4) or result.frame_global.shape != (256,) or
                not bool(torch.isfinite(result.boxes_norm).all()) or not bool(torch.isfinite(result.frame_global).all())):
            raise FloatingPointError("R1 normalized box contract failed")
        if attach_labels:
            raw_label = record.get("_label")
            if raw_label is None:
                raise ValueError("R1 label attachment requires explicit _label")
            result.supervision = self.store.attach_frame_labels(batch, raw_label["target_ids"])
            result.supervision["label"] = raw_label
        return result

    def close(self) -> None:
        self.store._blob = None
        gc.collect()


def fixed_reference_multistage_batch(model: Any, seed: torch.Tensor, references: torch.Tensor, event: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Exact first four fixed-reference decoder layers: layer 0=Z1, layer 3=Z4."""
    from mmdet.models.layers.transformer.utils import coordinate_to_encoding

    if seed.ndim != 3 or references.ndim != 3 or seed.shape[:2] != references.shape[:2]:
        raise ValueError("R1 fixed-reference batch shape drift")
    query = seed
    reference_batch = references
    decoder = model.decoder
    states: list[torch.Tensor] = []
    if len(decoder.layers) < 4:
        raise AssertionError("GroundingDINO decoder has fewer than four layers")
    for layer in decoder.layers[:4]:
        reference_input = reference_batch[:, :, None] * torch.cat([event["valid_ratios"], event["valid_ratios"]], dim=-1)[:, None]
        query_sine = coordinate_to_encoding(reference_input[:, :, 0, :])
        query_pos = decoder.ref_point_head(query_sine)
        query = layer(query, query_pos=query_pos, value=event["memory"], key_padding_mask=event["memory_mask"],
                      self_attn_mask=None, spatial_shapes=event["spatial_shapes"],
                      level_start_index=event["level_start_index"], valid_ratios=event["valid_ratios"],
                      reference_points=reference_input, memory_text=event["memory_text"],
                      text_attention_mask=event["text_attention_mask"])
        states.append(decoder.norm(query))
    result = {"z1": states[0], "z4": states[3]}
    for value in result.values():
        if value.shape != seed.shape or not bool(torch.isfinite(value.float()).all()):
            raise FloatingPointError("R1 fixed-reference Z state shape/finite drift")
    return result


class R1GroundingRuntime:
    """Native one-frame visual pass plus batched text replays for Z0/Z1/Z4."""

    def __init__(self, device: torch.device) -> None:
        from locatemot.rmot.l82_grounding_runtime import GroundingCandidateReferenceRuntime, install_clip_torchvision_compat

        install_clip_torchvision_compat()
        self._base = GroundingCandidateReferenceRuntime(device)
        self.device = device
        self.model = self._base.model
        self.model_info = self._base.model_info
        self.inference_detector = self._base.inference_detector
        self.encoder_events = self._base.encoder_events
        self.capture = self._base.capture
        self.set_sample_text = self._base.set_sample_text
        self.original_forward_transformer = self._base.original_forward_transformer
        self.make_text_batch_dict = __import__("tools.l82_audit_grounding_interface", fromlist=["make_text_batch_dict"]).make_text_batch_dict
        self.native_visual_forward_count = 0
        self.replay_count = 0

    def extract_group(self, batches: list[Any], query_batch_size: int = 8) -> dict[str, Any]:
        if not batches:
            raise ValueError("empty R1 group")
        first = batches[0]
        self.encoder_events.clear(); self.capture.clear()
        started = time.perf_counter()
        with torch.inference_mode():
            native = self.inference_detector(self.model, str(image_path(first.video, first.frame_id)),
                                             text_prompt=str(first.sentence), custom_entities=True)
        if len(self.encoder_events) != 1:
            raise AssertionError("R1 native encoder event count drift")
        visual_feats = self.capture.get("visual_feats")
        templates = self.capture.get("sample_template")
        if visual_feats is None or not isinstance(templates, (list, tuple)) or len(templates) != 1:
            raise AssertionError("R1 reusable native visual contract missing")
        self.native_visual_forward_count += 1
        image_shape = tuple(int(x) for x in native.metainfo["img_shape"][:2])
        scale_factor = native.metainfo["scale_factor"]
        boxes = first.boxes.to(self.device)
        boxes_norm = boxes_xyxy_to_normalized(boxes, image_shape, scale_factor)
        refs = boxes_to_reference_points(boxes_norm)
        from mmdet.models.layers.transformer.utils import coordinate_to_encoding
        enc = coordinate_to_encoding(refs.unsqueeze(0), num_feats=128)
        reference_position = self.model.decoder.ref_point_head(enc).squeeze(0)
        z0_values: list[torch.Tensor] = []
        z1_values: list[torch.Tensor] = []
        z4_values: list[torch.Tensor] = []
        text_globals: list[torch.Tensor] = []
        frame_globals: list[torch.Tensor] = []
        query_audits: list[dict[str, Any]] = []
        query_rows = batches
        for start in range(0, len(query_rows), int(query_batch_size)):
            chunk = query_rows[start:start + int(query_batch_size)]
            captions: list[str] = []; maps: list[Any] = []
            old_pad = bool(self.model.language_model.pad_to_max); self.model.language_model.pad_to_max = True
            try:
                for batch in chunk:
                    token_map, caption, _positive_map, _entities = self.model.get_tokens_positive_and_prompts(str(batch.sentence), True, None, None)
                    captions.append(caption); maps.append(token_map)
                text_dict = self.make_text_batch_dict(self.model, captions, self.device, force_pad_to_max=True)
            finally:
                self.model.language_model.pad_to_max = old_pad
            samples = []
            for caption, token_map in zip(captions, maps):
                sample = copy.deepcopy(templates[0]); self.set_sample_text(sample, caption, token_map); samples.append(sample)
            visual_batch = tuple(value.expand(len(chunk), *value.shape[1:]) for value in visual_feats)
            self.encoder_events.clear()
            with torch.inference_mode():
                self.original_forward_transformer(visual_batch, text_dict, samples)
            self.replay_count += len(chunk)
            if len(self.encoder_events) != 1:
                raise AssertionError("R1 batched encoder replay count drift")
            event = self.encoder_events[-1]
            for index, batch in enumerate(chunk):
                replay: dict[str, Any] = {}
                for key, value in event.items():
                    if key in {"memory", "memory_text", "memory_mask", "valid_ratios", "text_attention_mask", "text_token_mask"} and torch.is_tensor(value):
                        replay[key] = value[index:index + 1]
                    else:
                        replay[key] = value
                seed, audit = pool_memory_by_box(replay["memory"], replay["spatial_shapes"], replay["level_start_index"], boxes_norm,
                                                 replay["memory_mask"], grid_size=4)
                fixed = fixed_reference_multistage_batch(self.model, (seed + reference_position).unsqueeze(0), refs.unsqueeze(0), replay)
                z0_values.append(seed.detach().cpu().half())
                z1_values.append(fixed["z1"][0].detach().cpu().half())
                z4_values.append(fixed["z4"][0].detach().cpu().half())
                text_mask = replay.get("text_token_mask")
                if text_mask is None:
                    text_mask = torch.ones(replay["memory_text"].shape[:2], dtype=torch.bool, device=self.device)
                text = replay["memory_text"].float()
                text_globals.append(((text * text_mask[:, :, None].float()).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1)[:, None])[0].detach().cpu().half())
                visual = replay["memory"].float()
                valid = ~replay["memory_mask"].bool() if replay.get("memory_mask") is not None else torch.ones(visual.shape[:2], dtype=torch.bool, device=self.device)
                frame_globals.append(((visual * valid[:, :, None].float()).sum(dim=1) / valid.sum(dim=1).clamp_min(1)[:, None])[0].detach().cpu().half())
                query_audits.append({"unit_key": f"{batch.dataset}|{batch.video}|{batch.query_id}|{batch.frame_id}",
                                     "z0_shape": list(z0_values[-1].shape), "z1_shape": list(z1_values[-1].shape),
                                     "z4_shape": list(z4_values[-1].shape), "memory_shape": list(replay["memory"].shape),
                                     "memory_text_shape": list(replay["memory_text"].shape), "text_valid_tokens": int(text_mask.sum()),
                                     "roi_audit": audit})
            self.encoder_events.clear()
            del visual_batch, text_dict, samples, event
        payload = {
            "format": "locatemot-r1-aligned-z0-z1-z4-v1", "group_key": f"{first.dataset}|{first.video}|{first.frame_id}",
            "dataset": str(first.dataset), "video": str(first.video), "frame_id": int(first.frame_id),
            "query_unit_keys": [f"{x.dataset}|{x.video}|{x.query_id}|{x.frame_id}" for x in query_rows],
            "query_ids": [int(x.query_id) for x in query_rows], "sentences": [str(x.sentence) for x in query_rows],
            "z0": torch.stack(z0_values), "z1": torch.stack(z1_values), "z4": torch.stack(z4_values),
            "text_global": torch.stack(text_globals), "frame_global": torch.stack(frame_globals),
            "candidate_count": int(first.candidate_count), "row_offsets": list(first.row_offsets),
            "row_keys_digest": row_digest(first.row_keys), "candidate_indices": list(first.candidate_indices),
            "track_ids": list(first.track_ids), "pool_ids": list(first.pool_ids), "boxes": first.boxes.float().clone(),
            "boxes_norm": boxes_norm.float().cpu().clone(), "image_size": list(first.image_size), "query_audits": query_audits,
            "native_image_shape": list(image_shape), "native_scale_factor": np.asarray(scale_factor).reshape(-1).tolist(),
            "native_seconds": float(time.perf_counter() - started), "all_rows_retained": True,
            "candidate_deletion": False, "candidate_truncation": False, "features_persistent": True,
            "persistent_payload": "FP16 Z0/Z1/Z4 and summary vectors only; no labels/GT/scores",
            "labels_in_cache": False, "query_independent": False, "token_span_region_alignment": "UNALIGNED",
        }
        for key in ("z0", "z1", "z4", "text_global", "frame_global"):
            if not bool(torch.isfinite(payload[key].float()).all()):
                raise FloatingPointError(f"R1 cache nonfinite {key}")
        return payload

    def close(self) -> None:
        self._base.close()


__all__ = [
    "L49_DATA", "L69_ROOT", "MANIFEST", "MANIFEST_SHA", "OBS_DIM", "OBS_FIELDS", "R0_VISUAL_MANIFEST",
    "R1AlignedCacheIndex", "R1BankStore", "R1FeatureAssembler", "R1FrameInput", "R1GroundingRuntime", "ROOT", "THREAD",
    "build_causal_history", "build_geometry", "file_meta", "fixed_reference_multistage_batch", "image_path",
    "observation_from_rows", "row_digest", "sha256_file", "write_json",
]
