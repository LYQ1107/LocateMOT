"""Global candidate competition for TrackJEV.

GroundAnything emits several boxes for the same physical object.  What the model
must learn is not "is this box similar to another box" but "which of these boxes
is the one admissible representative of a physical entity, and which are
duplicates that must not create a second identity".

This module builds that evidence and its supervision:

* :func:`sparse_edges` — IoU / centre distance / appearance cosine over all
  candidate pairs, but stored sparsely (no NxN feature bank).
* :func:`entity_clusters` — one-to-one cluster growth seeded by the lowest index
  and extended by highest remaining IoU to *any* member, capped at
  ``MAX_GROUP``.  Deliberately **not** unconstrained connected components: a
  chain A~B~C must not merge two real neighbours that never overlap.
* :func:`admission_targets` — per-candidate label: duplicated / background /
  valid / unknown, from frame-level GT, with the reason recorded.
* :func:`competition_rows` — compact row strings (integer milli-units) that go
  into the OneJev state, following the historical v5/v6 encoding convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

MAX_GROUP = 16
MILLI = 1000
BOX_FORMATS = ("xywh", "xyxy")


@dataclass
class Edge:
    i: int
    j: int
    iou: float
    center_distance: float
    cosine: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _box_bounds(box: Sequence[float], box_format: str = "xywh") -> tuple[float, float, float, float]:
    """Return ``(x1, y1, x2, y2)`` for an explicit box convention.

    The legacy TrackJEV data contract uses top-left ``xywh``.  Native
    GroundAnything/runtime candidates use pixel ``xyxy``.  Keeping the
    conversion at this boundary prevents silently mixing the two conventions.
    """
    if box_format not in BOX_FORMATS:
        raise ValueError(f"unsupported box_format={box_format!r}; expected one of {BOX_FORMATS}")
    x1, y1, a, b = (float(v) for v in box[:4])
    if box_format == "xywh":
        return x1, y1, x1 + a, y1 + b
    return x1, y1, a, b


def _box_iou(a: Sequence[float], b: Sequence[float], box_format: str = "xywh") -> float:
    ax1, ay1, ax2, ay2 = _box_bounds(a, box_format)
    bx1, by1, bx2, by2 = _box_bounds(b, box_format)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + \
        max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a: Sequence[float], b: Sequence[float], scale: float,
                     box_format: str = "xywh") -> float:
    ax1, ay1, ax2, ay2 = _box_bounds(a, box_format)
    bx1, by1, bx2, by2 = _box_bounds(b, box_format)
    dx = ((ax1 + ax2) / 2) - ((bx1 + bx2) / 2)
    dy = ((ay1 + ay2) / 2) - ((by1 + by2) / 2)
    return math.hypot(dx, dy) / max(scale, 1e-6)


def _cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / max(na * nb, 1e-12)


def sparse_edges(boxes: Sequence[Sequence[float]],
                 features: Optional[Sequence[Optional[Sequence[float]]]] = None,
                 *, iou_threshold: float = 0.0, cosine_threshold: Optional[float] = None,
                 max_neighbors: int = 8,
                 scale: Optional[float] = None,
                 box_format: str = "xywh") -> list[Edge]:
    """All candidate pairs worth showing to the model, capped per candidate."""
    if box_format not in BOX_FORMATS:
        raise ValueError(f"unsupported box_format={box_format!r}; expected one of {BOX_FORMATS}")
    n = len(boxes)
    if n < 2:
        return []
    if scale is None:
        width = max((_box_bounds(b, box_format)[2] for b in boxes), default=1.0)
        height = max((_box_bounds(b, box_format)[3] for b in boxes), default=1.0)
        scale = math.hypot(width, height)
    edges: list[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            iou = _box_iou(boxes[i], boxes[j], box_format)
            cosine = _cosine(features[i] if features else None,
                             features[j] if features else None)
            overlap_hit = iou > 0.0 and iou >= iou_threshold
            appearance_hit = (cosine is not None and cosine_threshold is not None
                              and cosine >= cosine_threshold)
            if not (overlap_hit or appearance_hit):
                continue
            edges.append(Edge(i, j, round(iou, 4),
                              round(_center_distance(boxes[i], boxes[j], scale, box_format), 4),
                              None if cosine is None else round(cosine, 4)))
    # keep the strongest few neighbours per candidate, deterministic order
    by_candidate: dict[int, list[Edge]] = {}
    for edge in edges:
        by_candidate.setdefault(edge.i, []).append(edge)
        by_candidate.setdefault(edge.j, []).append(edge)
    keep: set[tuple[int, int]] = set()
    for candidate, items in by_candidate.items():
        items.sort(key=lambda e: (-(e.iou), e.center_distance, e.j if e.i == candidate else e.i))
        for edge in items[:max_neighbors]:
            keep.add((edge.i, edge.j))
    return [edge for edge in edges if (edge.i, edge.j) in keep]


def entity_clusters(boxes: Sequence[Sequence[float]], edges: Sequence[Edge], *,
                    max_group: int = MAX_GROUP,
                    min_iou: float = 0.0) -> list[list[int]]:
    """Grow one-to-one entity clusters from the strongest remaining edge.

    Ordering rule: seed with the lowest unused candidate index, then repeatedly
    add the unused candidate with the highest IoU to *any* current member; ties
    fall back to original index.  This keeps the policy deterministic and stops a
    single bridging box from merging two real neighbours: every candidate joins
    at most one cluster and a cluster never exceeds ``max_group``.
    """
    adjacency: dict[int, dict[int, float]] = {i: {} for i in range(len(boxes))}
    for edge in edges:
        if edge.iou <= 0.0 or edge.iou < min_iou:
            continue
        adjacency[edge.i][edge.j] = max(adjacency[edge.i].get(edge.j, 0.0), edge.iou)
        adjacency[edge.j][edge.i] = max(adjacency[edge.j].get(edge.i, 0.0), edge.iou)
    unused = set(range(len(boxes)))
    clusters: list[list[int]] = []
    while unused:
        seed = min(unused)
        unused.discard(seed)
        cluster = [seed]
        while len(cluster) < max_group and unused:
            best_candidate, best_iou = None, 0.0
            for member in cluster:
                for neighbour, iou in adjacency.get(member, {}).items():
                    if neighbour not in unused or iou <= 0.0:
                        continue
                    if iou > best_iou or (iou == best_iou and best_candidate is not None
                                          and neighbour < best_candidate):
                        best_candidate, best_iou = neighbour, iou
            if best_candidate is None:
                break
            unused.discard(best_candidate)
            cluster.append(best_candidate)
        clusters.append(sorted(cluster))
    return clusters


@dataclass
class AdmissionTarget:
    candidate: int
    label: str                  # valid_entity | duplicate | background | unknown
    matched_gt: Optional[int] = None
    matched_iou: float = 0.0
    duplicate_of: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def admission_targets(boxes: Sequence[Sequence[float]],
                      gt_boxes: Sequence[Sequence[float]],
                      gt_ids: Sequence[int],
                      *, match_iou: float = 0.5,
                      annotation_exhaustive: bool = True,
                      clusters: Optional[Sequence[Sequence[int]]] = None) -> list[AdmissionTarget]:
    """Per-candidate admission label from frame-level GT.

    ``annotation_exhaustive=False`` marks unmatched candidates as ``unknown``
    instead of ``background``: a partial annotation must never be turned into a
    background negative.
    """
    best_gt: dict[int, tuple[Optional[int], float]] = {}
    for index, box in enumerate(boxes):
        best_id, best_iou = None, 0.0
        for gt_box, gt_id in zip(gt_boxes, gt_ids):
            iou = _box_iou(box, gt_box)
            if iou > best_iou:
                best_id, best_iou = int(gt_id), iou
        best_gt[index] = (best_id, best_iou)

    # within a cluster, the candidate with the highest IoU to a GT owns it
    owner: dict[int, int] = {}
    if clusters:
        for cluster in clusters:
            for gt_id in {best_gt[i][0] for i in cluster if best_gt[i][0] is not None}:
                members = [i for i in cluster if best_gt[i][0] == gt_id]
                members.sort(key=lambda i: (-best_gt[i][1], i))
                high = members[0]
                if best_gt[high][1] >= match_iou:
                    owner[gt_id] = high

    targets: list[AdmissionTarget] = []
    for index in range(len(boxes)):
        gt_id, iou = best_gt[index]
        if gt_id is not None and iou >= match_iou:
            if owner.get(gt_id, index) == index:
                targets.append(AdmissionTarget(index, "valid_entity", gt_id, round(iou, 4),
                                               reason="highest-IoU representative of its GT"))
            else:
                targets.append(AdmissionTarget(index, "duplicate", gt_id, round(iou, 4),
                                               duplicate_of=owner[gt_id],
                                               reason="same GT already represented by a better box"))
            continue
        if not annotation_exhaustive:
            targets.append(AdmissionTarget(index, "unknown", gt_id, round(iou, 4),
                                           reason="partial annotation: unmatched is not background"))
            continue
        # a box that overlaps some GT below threshold and is not the best one for
        # any GT is still ambiguous in crowded frames -> unknown rather than background
        if gt_id is not None and iou > 0.3:
            targets.append(AdmissionTarget(index, "unknown", gt_id, round(iou, 4),
                                           reason="overlaps a GT below the match threshold"))
        else:
            targets.append(AdmissionTarget(index, "background", gt_id, round(iou, 4),
                                           reason="no GT support in an exhaustive frame"))
    return targets


def competition_rows(boxes: Sequence[Sequence[float]], clusters: Sequence[Sequence[int]],
                     edges: Sequence[Edge], *, scale: Optional[float] = None,
                     box_format: str = "xywh") -> dict[str, Any]:
    """Compact, token-efficient competition context for the OneJev state.

    Matrices are emitted as CSV row strings in integer milli-units (the encoding
    already validated in evidence formats v5/v6), never as pretty JSON.
    """
    if box_format not in BOX_FORMATS:
        raise ValueError(f"unsupported box_format={box_format!r}; expected one of {BOX_FORMATS}")
    if scale is None:
        width = max((_box_bounds(b, box_format)[2] for b in boxes), default=1.0)
        height = max((_box_bounds(b, box_format)[3] for b in boxes), default=1.0)
        scale = math.hypot(width, height)
    iou = [[0 for _ in boxes] for _ in boxes]
    distance = [[0 for _ in boxes] for _ in boxes]
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                iou[i][j] = MILLI
                distance[i][j] = 0
                continue
            iou[i][j] = round(_box_iou(boxes[i], boxes[j], box_format) * MILLI)
            distance[i][j] = round(_center_distance(boxes[i], boxes[j], scale, box_format) * MILLI)
    group_of = {candidate: group_index for group_index, cluster in enumerate(clusters)
                for candidate in cluster}
    return {
        "competition_group_of_candidate": [group_of.get(i, -1) for i in range(len(boxes))],
        "competition_groups": [list(cluster) for cluster in clusters],
        "competition_iou_rows_milli": [",".join(str(v) for v in row) for row in iou],
        "competition_center_distance_rows_milli": [",".join(str(v) for v in row) for row in distance],
        "competition_edge_count": len(edges),
        "competition_box_format": box_format,
        "competition_numeric_encoding": ("CSV rows in candidate order, integer/1000 units, "
                                         "diagonal IoU=1000, no truncation of candidates"),
    }


def cluster_consistency(clusters: Sequence[Sequence[int]], edges: Sequence[Edge]) -> dict[str, Any]:
    """Diagnostics: how connected are the groups, and are there bridging chains?"""
    lookup = {(min(e.i, e.j), max(e.i, e.j)): e for e in edges}
    chain_pairs = 0
    for cluster in clusters:
        for a in cluster:
            for b in cluster:
                if a >= b:
                    continue
                edge = lookup.get((a, b))
                if edge is None or edge.iou <= 0.0:
                    chain_pairs += 1
    return {"clusters": len(clusters), "max_cluster": max((len(c) for c in clusters), default=0),
            "non_overlapping_pairs_inside_clusters": chain_pairs}
