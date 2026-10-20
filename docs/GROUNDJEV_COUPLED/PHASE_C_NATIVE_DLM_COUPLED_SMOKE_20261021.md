# Native G2 clean DLM + GroundJEV coupling smoke — 2026-10-21

This bounded run loaded the protected GroundAnything G2 Clean base and `adapter-step012000.pt`, then rendered one real MOT17 TRAIN image. The new provider used one native clean DLM forward for both streams:

- image hidden tokens: `1008 x 2560`
- non-image clean context: `[1, 2560]` from 44 valid text positions
- candidate boxes: 2 real boxes from MOT17-02-DPM frame 000001
- bridge forward count: 1
- native DLM forward count: 1
- independent VLM instances: 0
- adapter step: 12000
- total typed loss: `5.5338025`
- clipped gradient norm: `73.3002243`
- peak allocated memory: `12,058,940,928` bytes (~11.24 GiB)
- elapsed time: `16.91` seconds

The clean contextual token was projected into the GroundJEV candidate readout, and the typed identity/admit/Yes-No/lifecycle/memory losses completed a real GPU backward and optimizer step. The formal G2 run directory and checkpoint files were read-only.

PyTorch emitted a BF16/FP32 RMSNorm fallback warning in the frozen native language path. The result remained finite and the warning is preserved in the machine-readable receipt; it is a runtime performance/dtype note, not evidence of a metric failure.

This establishes the single-backbone visual-plus-language coupling interface only. It does not establish tracking performance, identity association gain, grounding recall, HOTA, IDF1, AssA, TETA, or independent generalization.
