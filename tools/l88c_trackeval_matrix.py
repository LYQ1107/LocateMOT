#!/usr/bin/env python3
"""TrackEval wrapper for L88C corrected full-video outputs.

The local TrackEval invocation is the already audited implementation.  The
wrapper makes the provenance of this corrected, zero-training replay explicit
without modifying the historical L88 TrackEval helper or its outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from l88_eval_common import MANIFEST, MANIFEST_SHA, THREAD, sha256, write_json
import l88_trackeval_matrix as legacy


WORK_ROOT = Path(__file__).resolve().parents[1]


def _rewrite(path: Path, values: dict[str, Any]) -> None:
    payload = json.loads(path.read_text())
    payload.update(values)
    write_json(path, payload)


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    # The legacy runner already refuses nonempty output paths and writes the
    # complete TrackEval provenance.  It only consumes L88C prediction roots.
    result = legacy.run(args)
    out = args.out.resolve()
    update = {
        "format": "locatemot-l88c-trackeval-matrix-v1",
        "base_l88_sha": "c9b44c07b9b977de9d0f839fb2ff6363abb0386e",
        "zero_training": True,
        "corrected_candidate_vs_null": True,
        "corrected_emission_contract": "candidate_energy-null_logit >= null_margin AND presence_logit >= presence_threshold",
        "l88c_code_sha": None,
        "luna_thread": THREAD,
        "manifest_sha256": MANIFEST_SHA,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "no_hota_or_trackeval": False,
    }
    for name in ("trackeval_matrix.json", "provenance.json"):
        _rewrite(out / name, update)
    _rewrite(out / "status.json", {
        "format": "locatemot-l88c-trackeval-status-v1",
        "zero_training": True,
        "corrected_candidate_vs_null": True,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": True,
        "no_hota_or_trackeval": False,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
