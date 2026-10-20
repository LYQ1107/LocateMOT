# Native G2 clean context readout pilot — 2026-10-21

This bounded C1 pilot used the protected G2 Clean adapter at step 12000 and eight real MOT17-02-FRCNN frames. The official G2 DLM remained frozen; each frame was encoded once, producing region tokens and same-frame clean-language context for the GroundJEV typed readout. The first four frames were train and the last four were a same-video temporal development split. Previous-frame GT-matched region tokens supplied teacher-forced causal memory slots, so this is a learnability control rather than a deployment or generalization evaluation.

| Measure | Before | After |
|---|---:|---:|
| Train identity accuracy | 3.2258% | 72.5806% |
| Temporal-dev identity accuracy | 3.2258% | 40.3226% |
| Train admit accuracy | 91.1290% | 91.1290% |
| Temporal-dev admit accuracy | 79.0323% | 91.1290% |

The run used 248 candidate rows, eight native DLM forwards and 12 downstream optimizer steps. The same single-backbone path fused the contextual token through a trainable projection before all typed heads. The protected formal G2 directory was read-only.

The first attempt completed its forward/training loop but failed while writing the final receipt because the output directory was missing. That attempt is excluded from the result; the rerun created the directory and used the same data, split, epochs and optimizer protocol.

This result does not claim HOTA, IDF1, AssA, TETA, tracking improvement or independent-video generalization. The teacher-forced memory condition must be removed or compared with a learned causal memory course before scientific claims.

Machine-readable receipt: `PHASE_C_NATIVE_CONTEXT_PILOT_20261021.json`.
