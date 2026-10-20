"""Lossless-for-training compacting of dense TrackJEV competition context.

The native MOT export stores two full ``N x N`` CSV matrices in every row.
Those matrices duplicate geometry already present in ``state.candidates`` and
make crowded frames exceed the OneJev context window.  This transform keeps
the exact candidate boxes and every non-zero IoU edge (with its rounded
center-distance value), while removing only the reconstructible dense rows.

It is selected explicitly by a source-manifest field.  The immutable native
JSONL is never changed, so historical manifests and receipts remain replayable.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


TRANSFORM = "compact_competition_v1"


def _parse_csv_rows(value: Any, n: int, field: str) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{field}: expected {n} CSV rows")
    rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, str):
            raise ValueError(f"{field}: row is not a CSV string")
        values = [int(item) for item in row.split(",")]
        if len(values) != n:
            raise ValueError(f"{field}: expected square row of {n} values")
        rows.append(values)
    return rows


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a compact equivalent of one native TrackJEV state.

    Candidate geometry and cluster assignments remain byte-for-byte equal.
    For every pair with positive IoU, the rounded native IoU and center
    distance are retained as ``[i, j, iou_milli, distance_milli]``.  Pairs
    with zero IoU are exactly derivable from the retained candidate boxes and
    therefore do not need another textual matrix.
    """
    if not isinstance(state, dict):
        raise TypeError("TrackJEV state must be an object")
    dense_iou = state.get("competition_iou_rows_milli")
    dense_distance = state.get("competition_center_distance_rows_milli")
    if dense_iou is None and dense_distance is None:
        # Idempotent for rows that have already been transformed.
        return deepcopy(state)
    candidates = state.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("TrackJEV state has competition matrices but no candidates")
    n = len(candidates)
    iou = _parse_csv_rows(dense_iou, n, "competition_iou_rows_milli")
    distance = _parse_csv_rows(dense_distance, n, "competition_center_distance_rows_milli")
    edges: list[list[int]] = []
    for i in range(n):
        if iou[i][i] != 1000 or distance[i][i] != 0:
            raise ValueError("native competition diagonal is inconsistent")
        for j in range(i + 1, n):
            if iou[i][j] > 0:
                edges.append([i, j, iou[i][j], distance[i][j]])
            if iou[i][j] != iou[j][i] or distance[i][j] != distance[j][i]:
                raise ValueError("native competition matrix is not symmetric")
    compact = deepcopy(state)
    compact.pop("competition_iou_rows_milli", None)
    compact.pop("competition_center_distance_rows_milli", None)
    compact["competition_overlap_edges_milli"] = edges
    compact["competition_numeric_encoding"] = (
        "sparse-overlap-edge-v1; [candidate_i,candidate_j,iou_milli,center_distance_milli]; "
        "all positive-IoU pairs retained; zero-IoU geometry is derivable from exact candidate boxes"
    )
    compact["competition_dense_rows_omitted"] = True
    return compact


def transform_row(row: dict[str, Any], transform: str | None) -> dict[str, Any]:
    """Apply an explicitly registered source-row transform."""
    if not transform:
        return row
    if transform != TRANSFORM:
        raise ValueError(f"unknown native source transform {transform!r}")
    transformed = dict(row)
    transformed["state"] = compact_state(row.get("state"))
    return transformed


__all__ = ["TRANSFORM", "compact_state", "transform_row"]
