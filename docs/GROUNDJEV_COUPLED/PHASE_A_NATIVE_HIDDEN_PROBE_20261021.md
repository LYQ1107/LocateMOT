# Native G2 Clean hidden-state probe — 2026-10-21

The protected GroundAnything G2 Clean base and the step-12000 adapter were loaded in the registered Python 3.10 runtime. One real row from the 583,251-row G2 TRAIN manifest was rendered:

- row: `mot17_MOT17-02-DPM_000001_dense_clean_v2`
- image: MOT17-02-DPM frame 000001
- adapter: `adapter-step012000.pt`, step 12000
- manifest SHA: `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`
- clean embedding shape: `[1, 1152, 2560]`
- hybrid language hidden shape: `[1, 1152, 2560]`
- image hidden vectors: `1008 x 2560`
- visual forward count: one
- independent VLM instances: zero
- elapsed time: 25.21 seconds

The probe follows the same BF16 autocast boundary as formal G2 training. Transformers 5.17 does not accept the `cache_position` keyword used by the pinned GroundAnything source, so the existing process-local compatibility shim was applied; no model weights or formal training files were modified. The probe also exposed and fixed a direct-eager dtype mismatch in the visual projector before the successful run.

This is an interface and provenance result only. It demonstrates that the native G2 visual tokens and clean language hidden state can be obtained from one real DLM forward. It does not measure HOTA, IDF1, AssA, TETA, grounding recall, identity accuracy, or any scientific improvement, and it does not yet complete the identity-conditioned decoder/grounding training path.

Machine-readable receipt: `PHASE_A_NATIVE_HIDDEN_PROBE_20261021.json`.
