# G2 event-integrity audit — step 12650

This audit reads the formal `training_events.jsonl` without touching the live
trainer. At the audit cutoff it contained 12,650 rows covering every integer
step from 1 through 12,650 exactly once.

- Missing steps: `0`.
- Duplicate step IDs: `0`.
- Non-finite values across loss, gradient norm, peak memory and both learning
  rates: `0`.
- Explicit checkpoint events: `26`; the latest registered numbered checkpoint
  is `adapter-step012500.pt`.
- `final_receipt.json`: absent at the audit cutoff; G2 is not terminal.

The event log advanced to step `12651` immediately afterward (`2026-10-21
05:18:46 +08:00`, loss `0.20978708984330297`, finite gradient norm
`10.187045097351074`). That newer event is a progress observation; the
integrity counts above intentionally remain tied to the completed audit cutoff.

This evidence supports resumability and numeric integrity only. It is not a
detector or MOT/RMOT/OVMOT performance result.
