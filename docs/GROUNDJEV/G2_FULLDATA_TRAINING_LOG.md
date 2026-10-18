# G2 full-data training log

## 2026-10-19 — corpus and DDP gate

- Audited all ten requested dataset roots with symlink following and TRAIN-only boundaries.
- Prepared 199,053 validated JSONL rows from nine sources. BURST remains annotation-only and is explicitly excluded; RMOT26 retains its 20-entry failed-download receipt.
- G2-A passed dense/referring/OVMOT native forward/backward/AdamW on one card.
- A one-step two-card DDP smoke on physical GPU3/GPU5 passed in `.venv-onejev-py310`; the first system-Python attempt failed before model loading because `accelerate` was absent and is preserved as an environment failure.

## Formal run

The formal run is launched after the above gates with two ranks on physical GPU3/GPU5, master port `29619`, and `max_steps=1200`. It uses the same rows SHA, base checkpoint, projector/LoRA policy, official mixed loss, and atomic checkpoint contract. Every checkpoint and event is written beneath `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_PROJECTOR_QWEN3_LORA_FULLDATA/formal_g2_1200/`. Independent held-out evaluation remains blocked until this run produces a valid terminal receipt.

## 2026-10-19 — official memory route repair

The first 4,096/8,192 recovery reached `mot20_MOT20-05_001167_dense` but exhausted GPU memory because the custom trainer still used the legacy padded language loop while GPU3/GPU5 each had an unrelated 18.5-GiB resident workload. The valid step-1 checkpoint was preserved. The trainer now enables the official DLM general memory route: padding-free packing, non-reentrant language gradient checkpointing at every packed layer, chunk linear kernel, and no activation CPU offload.

The environment has neither optional `flash_attn` nor `fla`; the trainer records and selects GroundAnything's supported `cudnn_sdpa` packed clean-attention fallback and `checkpointed_chunk` LM-head fallback. A two-rank long-row regression with one 2,193-token MOT20 sample per rank completed one optimizer step on physical GPU3/GPU5 (loss 2.0606513, gradient norm 5.1535301) and wrote atomic checkpoint/final receipts. This is a memory/continuation gate only; no G2 quality result is claimed.

## 2026-10-19 — formal recovery relaunched

The repaired formal run was relaunched from the preserved step-1 checkpoint with the unchanged 199,053-row manifest (SHA `b8df7365a1b1cda47489f8044d86c030be9ab17c48724b5b5620befb5c9eb76e`), `max_steps=1200`, save interval 50, seed `20261019`, and two DDP ranks on physical GPU3/GPU5 (master port `29633`). The packed/checkpointed route and recorded `cudnn_sdpa`/`checkpointed_chunk` fallbacks are active. The command is still running; no G2 quality metric or completion claim is made before its terminal receipt.

## 2026-10-19 — formal recovery evidence checkpoint

The repaired formal process is live and has written atomic checkpoints through step 650. Recent events remain finite (for example step 650 loss `1.5408508778`, gradient norm `4.7876658430`); the run continues toward step 1200 with the unchanged corpus and objective. This is training-progress evidence only, not a detector or tracking quality result.

## 2026-10-19 — historical G2 terminal receipt and clean V2 launch

The historical `formal_g2_1200` owner completed step 1200 with finite loss
`1.7369554043` and gradient norm `14.4138050079`. Its terminal receipt is
`G2_FORMAL_TRAINING_COMPLETE`, but its 199,053-row manifest contains 3,317
MOT20-05 rows. It is therefore retained as a contaminated engineering
checkpoint, not an independent held-out result.

The corrected `G2_CLEAN_FULL_COVERAGE_V2` manifest contains 583,251 unique
rows and excludes MOT20-05, TAO Novel/sparse negatives and placeholder RMOT
queries. A two-rank BS2/accumulation smoke and a strict step-1 resume replay
passed. The formal clean owner starts from G0 `GroundAnything-DLM` on physical
GPU3/GPU5 with global batch 32 and a planned 18,227 optimizer updates. It is
running under the V2 report; no quality metric is available before its
terminal checkpoint and independent evaluation.

## 2026-10-19 — clean manifest contract re-audit

The read-only contract audit completed against the exact manifest used by the
formal owner. It found 583,251 JSONL records, 583,251 unique row IDs, zero
MOT20-05 rows, zero `the selected target` placeholder rows, and no cross-source
absolute media-path intersections. The audit artifact is
`/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/g2_clean_data_contract_audit_v2.json`;
the rows SHA remains
`f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`.
The audit also scans the raw MOT20 directory, so its inventory lists the
excluded MOT20-05 directory; that raw-tree observation is not a clean-row
inclusion. No active owner or checkpoint was modified.

## 2026-10-19 — latest submission audit receipt

The read-only audit entrypoint tools/groundjev/audit_latest_submission.py recorded
the live state without touching any owner. The clean G2 event log is finite at
the observed step, and its recorded/actual manifest SHA and base-model SHA
match. The J1 log has an observed step 680 while its durable FP32/Adam/RNG
checkpoint is step 650; these are reported separately. The G1 continuation,
G2 bundle export, and J1 held-out gate are still dependency waiters, so no
quality or promotion claim is made.
