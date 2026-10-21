# Phase C real G2 memory prefill — adapter step 17000

- Status: `PHASE_C_REAL_G2_MEMORY_PREFILL_COMPARISON_COMPLETE`
- Backbone: one official GroundAnything G2 DLM, adapter step `17000`.
- Scope: MOT20-03 TRAIN frame 82 as causal previous image and frame 83 as current image.
- Physical GPU: `7`; formal G2 run remained untouched.
- Visual embedding calls: `2` total (previous + current); the current embedding was reused for all three language-prefill conditions.
- Independent VLM instances: `0`.

| Condition | Generated tokens | Prefill calls | Grammar-forced `box_end` | Recovered complete pairs | Output |
|---|---:|---:|---:|---:|---|
| C0_disabled | 9 | 1 | 1 | 1 | `[['person', [1, 428, 40, 556]]]` |
| C2_correct_previous_memory | 10 | 1 | 1 | 1 | `[['person', [1, 428, 40, 556]]]` |
| C2_shuffled_previous_memory | 10 | 1 | 1 | 1 | `[['person', [1, 428, 40, 556]]]` |

The first attempt failed before prefill because BF16 multimodal embeddings entered a restored FP32 LoRA projection without the formal G2 autocast boundary. The real call now uses the same CUDA BF16 autocast boundary. The next attempt exposed that unconstrained greedy decoding continued beyond the four-coordinate payload without emitting `box_end`; the bounded runner now uses GroundAnything's official single-box structural termination rule, which constrains only the coordinate-token slots and emits the structural boundary. It does not invent, reorder, or fill coordinates.

Both C2 conditions produced the same recovered box as C0 in this run. The memory projector is untrained and uses previous-frame image-token summaries, so this is a negative/neutral engineering control, not evidence of identity-conditioned grounding improvement. No HOTA, IDF1, IDSW, recall, or independent generalization claim is made.

Machine-readable receipt: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_memory_prefill_step17000_v3/receipt.json`.
