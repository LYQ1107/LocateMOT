# OneJev multimodal efficiency status

- The correctness-first trainer remains active for the three-task SFT; it does not use a feature cache and saves FP32 masters, Adam moments, scheduler, and RNG state.
- The official `--share-prefix` path was tested on this Qwen3.5 multimodal checkpoint and rejected: the hybrid linear-attention layers cannot consume the branch mask (`1024` versus `17496` shape mismatch). It must not be enabled by assumption.
- `flash-linear-attention` and `causal_conv1d` are unavailable in the current runtime, so the reference PyTorch fallbacks explain the long per-step time. This is recorded as a kernel/runtime limitation, not a reason to change the data mix or metrics.
- The current SFT uses the verified old implementation while any shared-frame optimization remains a separate, non-blocking investigation. No detached hidden-state cache is used in the formal run.

`full_goal_complete=false`.
