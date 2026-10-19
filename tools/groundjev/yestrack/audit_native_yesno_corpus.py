#!/usr/bin/env python3
"""Audit the native multimodal YesTrack corpus before model loading."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit(path: Path) -> dict:
    errors: list[str] = []
    ids: set[str] = set()
    source_splits: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, str, str]] = Counter()
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(f"line{line_no}:json:{exc}")
                continue
            rows.append(row)
            prefix = f"line{line_no}"
            required = ("id", "split", "source", "task", "state", "question", "target", "media", "supervision")
            errors.extend(f"{prefix}:missing_{key}" for key in required if key not in row)
            row_id = str(row.get("id", ""))
            if not row_id:
                errors.append(f"{prefix}:empty_id")
            elif row_id in ids:
                errors.append(f"{prefix}:duplicate_id:{row_id}")
            ids.add(row_id)
            split = str(row.get("split", ""))
            source = str(row.get("source", ""))
            source_splits[source].add(split)
            task = str(row.get("task", ""))
            target = row.get("target")
            if target not in ({"yes": 1.0}, {"no": 1.0}):
                errors.append(f"{prefix}:target_not_binary_yes_no")
            if not isinstance(row.get("state"), dict):
                errors.append(f"{prefix}:state_not_object")
            question = row.get("question")
            if not isinstance(question, dict) or question.get("type") != "noul":
                errors.append(f"{prefix}:question_not_noul")
            else:
                instructions = str(question.get("instructions", "")).lower()
                if any(token in instructions for token in ("ground truth", "ground-truth", "gt label", "oracle")):
                    errors.append(f"{prefix}:ground_truth_word_in_question")
            media = row.get("media")
            if not isinstance(media, list) or not media:
                errors.append(f"{prefix}:missing_media")
            if "mot20-05" in source.lower():
                errors.append(f"{prefix}:forbidden_MOT20-05")
            supervision = row.get("supervision")
            if not isinstance(supervision, dict):
                errors.append(f"{prefix}:supervision_not_object")
            else:
                if supervision.get("gt_used_for_prompt") is not False:
                    errors.append(f"{prefix}:gt_used_for_prompt")
                if supervision.get("future_evidence") is not False:
                    errors.append(f"{prefix}:future_evidence")
            counts[(split, task, "yes" if target == {"yes": 1.0} else "no")] += 1
    leakage = {source: sorted(splits) for source, splits in source_splits.items() if len(splits) > 1}
    errors.extend(f"source_split_leakage:{source}" for source in sorted(leakage))
    return {
        "status": "NATIVE_YESTRACK_CORPUS_AUDIT_PASS" if not errors else "NATIVE_YESTRACK_CORPUS_AUDIT_FAILED",
        "format": "locatemot-native-yesno-v1",
        "rows": str(path.resolve()),
        "rows_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "source_split_leakage": leakage,
        "errors": errors,
        "gt_in_model_input": any("ground_truth" in e for e in errors),
        "future_evidence": any("future" in e for e in errors),
        "official_test_labels_read": False,
        "MOT20-05_included": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "row_count": report["row_count"], "errors": len(report["errors"])}, ensure_ascii=False))
    return 0 if report["status"].endswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
