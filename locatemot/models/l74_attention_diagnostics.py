"""Small, dependency-light diagnostics for L74 attention localization.

The functions in this module operate on already detached attention summaries.
They do not choose a model, filter candidates, or read labels.
"""
from __future__ import annotations

from typing import Any, Iterable

import math
import numpy as np


def finite_stats(values: Iterable[float | int | None]) -> dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if array.size == 0:
        return {"count": 0, "finite": True, "mean": None, "median": None, "std": None,
                "min": None, "max": None, "q05": None, "q25": None, "q75": None, "q95": None}
    return {
        "count": int(array.size),
        "finite": bool(np.isfinite(array).all()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def safe_corr(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return None
    x, y = x[valid], y[valid]
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def concentration(values: Iterable[float]) -> dict[str, Any]:
    raw = np.asarray(list(values), dtype=np.float64)
    if raw.size == 0 or not np.isfinite(raw).all():
        return {"finite": False, "count": int(raw.size)}
    positive = np.clip(raw, 0.0, None)
    total = float(positive.sum())
    if total <= 1e-12:
        probabilities = np.full(raw.size, 1.0 / max(1, raw.size), dtype=np.float64)
    else:
        probabilities = positive / total
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    normalized = entropy / max(math.log(max(2, raw.size)), 1e-12)
    sorted_prob = np.sort(probabilities)
    gini = float((2.0 * np.arange(1, raw.size + 1) - raw.size - 1.0).dot(sorted_prob)
                 / max(1.0, raw.size * sorted_prob.sum()))
    return {
        "finite": True,
        "count": int(raw.size),
        "entropy": entropy,
        "normalized_entropy": normalized,
        "gini": gini,
        "herfindahl": float((probabilities ** 2).sum()),
        "peak_to_uniform": float(probabilities.max() * raw.size),
        "peak_fraction": float(probabilities.max()),
        "negative_value_count": int((raw < 0).sum()),
    }


def cell_centers(
    grid_shape: tuple[int, int], processed_size: tuple[int, int]
) -> np.ndarray:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    proc_w, proc_h = float(processed_size[0]), float(processed_size[1])
    cell_w, cell_h = proc_w / max(1, width), proc_h / max(1, height)
    centers = []
    for row in range(height):
        for col in range(width):
            centers.append([(col + 0.5) * cell_w, (row + 0.5) * cell_h])
    return np.asarray(centers, dtype=np.float64)


def candidate_attention_summary(
    image_values: Iterable[float],
    candidate_indices: Iterable[int],
    overlap_areas: Iterable[float],
    grid_shape: tuple[int, int],
    processed_size: tuple[int, int],
    scaled_box: Iterable[float],
) -> dict[str, Any]:
    """Summarize one candidate against all image-token attention values."""
    values = np.asarray(list(image_values), dtype=np.float64)
    indices = [int(value) for value in candidate_indices]
    areas = np.asarray(list(overlap_areas), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return {"finite": False, "token_count": len(indices)}
    if not indices:
        return {
            "finite": True,
            "token_count": 0,
            "empty": True,
            "candidate_mean": None,
            "candidate_mass": 0.0,
            "candidate_mass_fraction": 0.0,
            "uniform_baseline": float(1.0 / max(1, values.size)),
            "peak_cell": None,
            "centroid": None,
            "box_center_distance": None,
        }
    if len(indices) != len(areas):
        raise ValueError("candidate index/area length mismatch")
    if min(indices) < 0 or max(indices) >= values.size:
        raise ValueError("candidate image-cell index out of range")
    selected = values[np.asarray(indices, dtype=np.int64)]
    centers = cell_centers(grid_shape, processed_size)
    selected_centers = centers[np.asarray(indices, dtype=np.int64)]
    positive = np.clip(values, 0.0, None)
    selected_positive = np.clip(selected, 0.0, None)
    image_mass = float(positive.sum())
    candidate_mass = float(selected_positive.sum())
    if float(areas.sum()) > 0:
        area_mean = float(np.dot(selected, areas) / areas.sum())
    else:
        area_mean = float(selected.mean())
    candidate_mean = float(selected.mean())
    cell_norm_mass = candidate_mass / max(1, len(indices))
    uniform = float(1.0 / max(1, values.size))
    image_mean, image_std = float(values.mean()), float(values.std())
    distribution = concentration(selected)
    if candidate_mass > 1e-12:
        probabilities = selected_positive / candidate_mass
        centroid = (probabilities[:, None] * selected_centers).sum(axis=0)
        peak_local = int(np.argmax(selected_positive))
    else:
        probabilities = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
        centroid = (probabilities[:, None] * selected_centers).sum(axis=0)
        peak_local = int(np.argmax(selected))
    scaled = [float(value) for value in scaled_box]
    box_center = np.asarray([(scaled[0] + scaled[2]) * 0.5, (scaled[1] + scaled[3]) * 0.5])
    box_diag = max(1e-6, float(np.linalg.norm(np.asarray([scaled[2] - scaled[0], scaled[3] - scaled[1]]))))
    peak_index = indices[peak_local]
    grid_width = int(grid_shape[1])
    return {
        "finite": True,
        "empty": False,
        "token_count": int(len(indices)),
        "candidate_mean": candidate_mean,
        "overlap_area_weighted_mean": area_mean,
        "candidate_mass": candidate_mass,
        "image_mass": image_mass,
        "background_mass": float(max(0.0, image_mass - candidate_mass)),
        "candidate_mass_fraction": float(candidate_mass / max(1e-12, image_mass)),
        "background_mass_fraction": float(max(0.0, image_mass - candidate_mass) / max(1e-12, image_mass)),
        "cell_count_normalized_mass": float(cell_norm_mass),
        "uniform_baseline": uniform,
        "peak_to_uniform_image": float(selected.max() / max(1e-12, uniform)),
        "candidate_mean_to_uniform": float(candidate_mean / max(1e-12, uniform)),
        "candidate_mean_z_vs_image": float((candidate_mean - image_mean) / max(1e-12, image_std)),
        "image_entropy": concentration(values),
        "candidate_concentration": distribution,
        "peak_cell": peak_index,
        "peak_cell_rc": [int(peak_index // grid_width), int(peak_index % grid_width)],
        "spatial_centroid_processed": [float(value) for value in centroid.tolist()],
        "box_center_processed": [float(value) for value in box_center.tolist()],
        "box_center_distance_processed": float(np.linalg.norm(centroid - box_center)),
        "box_center_distance_normalized": float(np.linalg.norm(centroid - box_center) / box_diag),
        "candidate_area_fraction_processed": None,
    }


def flatten_panel_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and isinstance(value, (float, int)) and math.isfinite(float(value)):
            values.append(float(value))
    return finite_stats(values)

