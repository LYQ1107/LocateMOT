# G2 held-out receipt collection

Use `scripts/groundjev/collect_g2_heldout_receipts.py` to generate a current
status/metrics collection for the G0, G2 Clean, DanceTrack, and AnimalTrack
owners. The collector records the model and TRAIN manifest SHA, but promotes a
scope into `accepted` only when its owner state contains complete result,
frame-contract, scorer, and official-metrics evidence.

The first collection on 2026-10-22 found all three MOT20 owners still running
and both domain continuations waiting, so `accepted_scope_count` was zero.
