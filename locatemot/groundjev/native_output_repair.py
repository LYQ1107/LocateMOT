"""Conservative recovery of complete GroundAnything native bbox pairs.

The normal protocol parser is deliberately strict and rejects a whole model
response when one decoded token is malformed.  For diagnosis we sometimes
need to retain the complete ``object_ref``/``box`` pairs that are still
present in that response.  This module never invents coordinates, reorders a
box, or fills a missing token: an incomplete pair is dropped and recorded.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Callable, Iterable


OBJECT_REF_START = "<|object_ref_start|>"
OBJECT_REF_END = "<|object_ref_end|>"
BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
_COORDINATE = re.compile(r"(?:<([0-9]{1,3})>){4}")
_RESERVED = (OBJECT_REF_START, OBJECT_REF_END, BOX_START, BOX_END)


def _valid_bbox(component: str) -> tuple[int, int, int, int] | None:
    """Parse one exact four-coordinate component without changing it."""

    match = _COORDINATE.fullmatch(component)
    if match is None:
        return None
    values = tuple(int(value) for value in re.findall(r"<([0-9]{1,3})>", component))
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    if not (x1 < x2 and y1 < y2):
        return None
    return values  # type: ignore[return-value]


def _preview(text: str, limit: int = 120) -> str:
    return text.replace("\n", "\\n")[:limit]


def repair_gam_bbox_output(
    text: object,
    *,
    label_validator: Callable[[str], str] | None = None,
) -> tuple[list[tuple[str, tuple[int, int, int, int]]], dict[str, object]]:
    """Recover only complete, independently valid native bbox pairs.

    ``label_validator`` should be the official GroundAnything phrase validator
    when it is available.  A validator that canonicalizes a label to a
    different string is treated as a rejection; this function never rewrites
    model text.  The returned report is intentionally compact enough to store
    in per-frame provenance.
    """

    if not isinstance(text, str):
        return [], {
            "version": "complete_pair_salvage_v1",
            "status": "REJECTED_NON_STRING",
            "candidate_count": 0,
            "duplicate_count": 0,
            "dropped_fragment_count": 1,
            "dropped_reasons": {"non_string": 1},
        }

    reasons: Counter[str] = Counter()
    fragments: list[dict[str, object]] = []

    def drop(reason: str, start: int, end: int, label: str = "") -> None:
        reasons[reason] += 1
        if len(fragments) < 32:
            fragments.append(
                {
                    "reason": reason,
                    "start": start,
                    "end": end,
                    "label_prefix": _preview(label),
                }
            )

    recovered: list[tuple[str, tuple[int, int, int, int]]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    duplicate_count = 0
    cursor = 0
    text_length = len(text)

    while cursor < text_length:
        start = text.find(OBJECT_REF_START, cursor)
        if start < 0:
            tail = text[cursor:]
            if tail.strip(" ,"):
                drop("unparsed_trailing_fragment", cursor, text_length)
            break

        separator = text[cursor:start]
        if separator not in ("", ", "):
            drop("unexpected_separator", cursor, start)

        label_end = text.find(OBJECT_REF_END, start + len(OBJECT_REF_START))
        if label_end < 0:
            drop("missing_object_ref_end", start, text_length)
            break
        label = text[start + len(OBJECT_REF_START) : label_end]

        box_start = text.find(BOX_START, label_end + len(OBJECT_REF_END))
        next_object = text.find(OBJECT_REF_START, label_end + len(OBJECT_REF_END))
        if box_start < 0 or (next_object >= 0 and next_object < box_start):
            drop("missing_box_start", start, next_object if next_object >= 0 else text_length, label)
            cursor = next_object if next_object >= 0 else text_length
            continue

        box_end = text.find(BOX_END, box_start + len(BOX_START))
        nested_object = text.find(OBJECT_REF_START, box_start + len(BOX_START))
        if box_end < 0:
            drop("missing_box_end", start, text_length, label)
            cursor = nested_object if nested_object >= 0 else text_length
            continue
        if nested_object >= 0 and nested_object < box_end:
            # A second object begins before the first box closes.  Preserve
            # neither incomplete pair; rescan at the complete nested object.
            drop("nested_object_before_box_end", start, nested_object, label)
            cursor = nested_object
            continue

        if label_validator is not None:
            try:
                canonical = label_validator(label)
            except Exception:
                drop("invalid_label", start, box_end + len(BOX_END), label)
                cursor = box_end + len(BOX_END)
                continue
            if canonical != label:
                drop("noncanonical_label", start, box_end + len(BOX_END), label)
                cursor = box_end + len(BOX_END)
                continue
        elif not label or any(token in label for token in _RESERVED):
            drop("invalid_label", start, box_end + len(BOX_END), label)
            cursor = box_end + len(BOX_END)
            continue

        payload = text[box_start + len(BOX_START) : box_end]
        components = payload.split(",")
        parsed: list[tuple[int, int, int, int]] = []
        for component in components:
            coordinate = _valid_bbox(component)
            if coordinate is None:
                parsed = []
                break
            parsed.append(coordinate)
        if not parsed:
            drop("invalid_coordinate_payload", start, box_end + len(BOX_END), label)
            cursor = box_end + len(BOX_END)
            continue

        for coordinate in parsed:
            item = (label, coordinate)
            if item in seen:
                duplicate_count += 1
                continue
            seen.add(item)
            recovered.append(item)
        cursor = box_end + len(BOX_END)

    report: dict[str, object] = {
        "version": "complete_pair_salvage_v1",
        "status": "RECOVERED" if recovered else "NO_COMPLETE_PAIR",
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "candidate_count": len(recovered),
        "duplicate_count": duplicate_count,
        "dropped_fragment_count": sum(reasons.values()),
        "dropped_reasons": dict(reasons),
        "dropped_fragments": fragments,
        "coordinates_reordered": False,
        "missing_coordinates_filled": False,
    }
    return recovered, report


__all__ = ["repair_gam_bbox_output"]
