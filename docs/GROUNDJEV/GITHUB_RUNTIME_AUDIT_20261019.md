# GroundJEV runtime and GitHub audit — 2026-10-19

This audit was performed from the LocateMOT YesTrack worktree at
`/data2/user/LocateMOT_YesTrack`, whose base is the clean G2 research head
`a3fc905ee4f1450dc155dc40f84d86030a8e079f`. The active training directories
were read once and were not stopped, restarted, or modified.

| Component | Observed state | Consequence |
| --- | --- | --- |
| G2 clean formal coverage | `G2_FORMAL_TRAINING_RUNNING`, 2 ranks, 18,227 planned steps, global batch 32, clean manifest SHA `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71` | Running dependency; no G2 quality result yet |
| Historical `formal_g2_1200` / G2-PL | Contains MOT20-05 rows | Engineering evidence only; rejected by clean evaluation preflight |
| J1 three-task training | Durable checkpoint observed at step 700, `groundjev-full-fp32-adam-rng-v1`, FP32/Adam/scheduler/RNG receipt present | Still running; no held-out metric claim |
| J1 held-out gate | Existing owner is `WAITING_J1_FINAL` | New gate contract is prepared in an independent worktree; active owner was not replaced |
| G1 GPU5 queue | `WAITING_G2_CLEAN_SUCCESS` | No G1 checkpoint or metric yet |
| G2 bundle export | `WAITING_G2_AND_G1_TERMINAL` | No clean bundle exists, so no GPU evaluation was started |

The historical G2 JSONL contains 92 events at the time of the audit. Every
event is pre-v2 and logged `samples_global=16` on a two-rank run. The read-only
accounting report reconstructs the global value as 32 and leaves the raw log
SHA unchanged (`bc2d4021d4f37d50fe4d84648f6743ef87213f55da66bca4f8478aa8281736a8`).
New trainer events use the explicit
`g2-training-event-v2-global-accounting` schema; the running old process was
not restarted to adopt it.

The clean evaluation chain now requires a fresh export receipt, the clean
manifest SHA, the G0 base SHA, the 569-tensor bundle contract, and a result
receipt that proves every fixed-scope frame was processed. Runtime GT/future
frame access is rejected, and rejected perception frames remain in the frame
denominator and latency/receipt records. The denominator example is
`183/1658 = 11.04%`: 1,658 is the full fixed manifest frame count, including
the 183 rejected requests.

Status of this document is `IMPLEMENTED` for the audit/preflight contract and
`WAITING_DEPENDENCY` for the actual clean G2 bundle and downstream evaluation.
No scientific improvement claim is made.
