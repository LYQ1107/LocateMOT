#!/usr/bin/env python3
"""Run the registered GroundJEV video evaluator.

The current reproducible video path is the fixed-candidate evaluator in
evaluate_fixed_candidate_clip.py. It consumes one immutable native
GroundAnything region cache, runs the typed GroundJEV readout and causal
memory/assignment runtime, and writes TrackEval HOTA/CLEAR/Identity results.
The scope is explicit: this is not a second VLM and not a terminal G2 Clean
benchmark claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.groundjev.coupled.evaluate_fixed_candidate_clip import main as _evaluate_main


def main() -> int:
    return int(_evaluate_main())


if __name__ == "__main__":
    raise SystemExit(main())
