# G2 projector and Qwen3 LoRA implementation

`tools/groundjev/g2/groundanything_lora.py` keeps the official DLM wrapper intact and replaces only the exact Qwen3 `q_proj`, `k_proj`, `v_proj`, and `o_proj` linear leaves with a frozen base projection plus a low-rank residual. LoRA A is Kaiming initialized, LoRA B is zero initialized, and the scaling is `alpha / rank` (`rank=16`, `alpha=32`, dropout `0.05`). This preserves the G0 function before an optimizer update and avoids wrapping the non-`PreTrainedModel` DLM with PEFT's generation-only interface.

The policy is applied after injection:

- vision tower: 0 trainable parameters;
- original Qwen3 language weights: 0 trainable parameters;
- visual projector: 27,267,584 trainable parameters;
- q/k/v/o LoRA: 11,796,480 trainable parameters across 36 layers;
- unexpected trainable modules: 0.

`train_g2_dlm.py` saves only the projector/LoRA trainable state together with AdamW, cosine-with-minimum-rate scheduler, Python/NumPy/Torch/CUDA RNG, sampler cursor, base checkpoint hash, source manifest hash, world size, and the official loss/optimizer contract. The first optimizer step and every registered interval are atomically checkpointed. `adapter_projector_final.pt` is written at normal completion.

The actual one-step two-GPU smoke receipt is under `.../ddp_smoke_1step/`: both ranks completed native loss/backward/AdamW, a finite gradient norm, and `adapter-step000001.pt` with `world_size=2`.
