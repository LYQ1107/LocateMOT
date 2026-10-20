# G2 terminal finalization contract audit — 2026-10-21

- **Observed event:** step 13,167/18,227 at 06:57:55 +08:00.
- **Trainer contract:** `tools/groundjev/g2/train_g2_dlm.py` writes `adapter_projector_final.pt` and `final_receipt.json` only after the terminal step on rank 0.
- **Required receipt:** status `G2_EXACT_COVERAGE_TRAINING_COMPLETE`, step `18,227`, and Clean TRAIN manifest SHA `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`.
- **Bundle gate:** `queue_g2_clean_bundle_export.py` rejects a missing receipt, wrong status/step/manifest, or missing final adapter, and refuses to overwrite the source bundle.
- **Current result:** static contract audit passed; the runtime remains `G2_FORMAL_TRAINING_RUNNING` with no terminal receipt. No GPU, checkpoint or model was modified.

This is a completion-contract audit, not a tracking metric or a claim of G2 performance.
