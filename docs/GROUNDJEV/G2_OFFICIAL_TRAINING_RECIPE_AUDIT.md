# GroundAnything G2 official recipe audit

Status: the source and runtime contract audit is complete; formal ten-dataset training is running only after the one-step DDP gate passes.

- Official checkout: `external/GroundAnything`, commit `8a3b43a1b42952e117960e85d63ff173bb9fba0a`.
- The task text named `e054ab329...`; that revision is not the checkout's actual HEAD. The pinned HEAD, source-file hashes, and configuration summaries are in `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_PROJECTOR_QWEN3_LORA_FULLDATA/official_training_source_audit.json`.
- The native DLM objective remains `0.5 * MDM + 0.5 * causal`. The official SFT1 values used by G2 are language LR `7e-6`, projector LR `2e-6`, vision LR `7e-7`, cosine schedule with minimum rate `0.1`, warmup `0.03`, block size `32`, and gradient clip `1.0`.
- G2 changes the trainable policy only as specified by the task: the vision tower and original Qwen3 language weights are frozen; the complete visual projector and q/k/v/o LoRA modules are trainable. G2 does not claim to reproduce the official 64-GPU multi-million-row pretraining stages.
- Transformers 5.17 removed the `cache_position` keyword from the no-cache causal-mask helper used by the pinned source. `tools/groundjev/g2/compat.py` applies a narrow process-local compatibility shim and records it in each receipt; it does not alter the DLM loss or token protocol.

Evidence:

- `g2_a_gradient_gate.json`: dense MOT20, referring RMOT, and TAO-Amodal OVMOT all completed native forward, backward, finite gradients, AdamW update, and checkpoint gate.
- `g2_dataset_coverage_audit.json`: resolved symlinked roots and the four-volume TAO frame layout without opening held-out media.
- `g2_train_manifest.json`: 199,053 validated rows with absolute media paths and `bytes=null`; 2,246 explicit rejects are retained separately.
