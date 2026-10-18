# G2 full-data training log

## 2026-10-19 — corpus and DDP gate

- Audited all ten requested dataset roots with symlink following and TRAIN-only boundaries.
- Prepared 199,053 validated JSONL rows from nine sources. BURST remains annotation-only and is explicitly excluded; RMOT26 retains its 20-entry failed-download receipt.
- G2-A passed dense/referring/OVMOT native forward/backward/AdamW on one card.
- A one-step two-card DDP smoke on physical GPU3/GPU5 passed in `.venv-onejev-py310`; the first system-Python attempt failed before model loading because `accelerate` was absent and is preserved as an environment failure.

## Formal run

The formal run is launched after the above gates with two ranks on physical GPU3/GPU5, master port `29619`, and `max_steps=1200`. It uses the same rows SHA, base checkpoint, projector/LoRA policy, official mixed loss, and atomic checkpoint contract. Every checkpoint and event is written beneath `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_PROJECTOR_QWEN3_LORA_FULLDATA/formal_g2_1200/`. Independent held-out evaluation remains blocked until this run produces a valid terminal receipt.

## 2026-10-19 — official memory route repair

The first 4,096/8,192 recovery reached mot20_MOT20-05_001167_dense but exhausted GPU memory because the custom trainer still used the legacy padded language loop while GPU3/GPU5 each had an unrelated 18.5-GiB resident workload. The valid step-1 checkpoint was preserved. The trainer now enables the official DLM general memory route: padding-free packing, non-reentrant language gradient checkpointing at every packed layer, chunk linear kernel, and no activation CPU offload.

The environment has neither optional flash_attn nor fla; the trainer records and selects GroundAnything's supported cudnn_sdpa packed clean-attention fallback and checkpointed_chunk LM-head fallback. A two-rank long-row regression with one 2,193-token MOT20 sample per rank completed one optimizer step on physical GPU3/GPU5 (loss 2.0606513, gradient norm 5.1535301) and wrote atomic checkpoint/final receipts. This is a memory/continuation gate only; no G2 quality result is claimed.
