# GroundJEV G2 Clean live progress — step 13361

- Run: `G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final`
- Scope: step **13,361 / 18,227**, exact-once two-rank training on physical GPUs 3/5.
- Latest event (2026-10-21 07:35:07 +0800): loss `0.2484929720`, gradient norm `11.54479408`, peak event memory `12.828843 GiB`; all finite.
- The parent PID `1799359` was live at the audit. The terminal receipt and terminal adapter are absent.
- Clean TRAIN manifest SHA-256: `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`.
- Durable recovery remains `adapter-step013000.pt` (SHA-256 `08cf28930eced61b2086785494728221e113ac75cd7dc6aefe02ab1bc2ebfe59`) and `latest_training_state.pt` (SHA-256 `4a79cfb3e64e69ffc56aaf964754d0209382b87c2124d46f13506325ca52fd22`).
- G1, native TAO/OVMOT joining, three-task OneJev optimization and independent held-out evaluation remain gated by the terminal G2 receipt.
- This is training-health and resumability evidence only; it is **not** a MOT/RMOT/OVMOT performance result.

Machine-readable record: [G2_PROGRESS_LIVE_20261021_STEP13361.json](./G2_PROGRESS_LIVE_20261021_STEP13361.json).
