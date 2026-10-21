#!/usr/bin/env python3
"""Run the registered GroundJEV video evaluator.

The current reproducible video path is the fixed-candidate evaluator in
``evaluate_fixed_candidate_clip.py``.  It consumes one immutable native
GroundAnything region cache, runs the typed GroundJEV readout and causal
memory/assignment runtime, and writes TrackEval HOTA/CLEAR/Identity results.
The entry point deliberately keeps that scope explicit: it is not a second
VLM, a CPU mock, or an online claim that the final G2 Clean checkpoint has
already been exported.

All command-line options are forwarded to the registered evaluator, for
example::

    python tools/groundjev/coupled/run_coupled_video.py \\
        --clip /path/to/clip --checkpoint /path/to/readout.pt \\
        --output /path/to/new-receipt --device cpu
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.groundjev.coupled.evaluate_fixed_candidate_clip import main as _evaluate_main


def main() -> int:
    """Delegate to the tested fixed-candidate video implementation."""
    return int(_evaluate_main())


if __name__ == "__main__":
    raise SystemExit(main())
