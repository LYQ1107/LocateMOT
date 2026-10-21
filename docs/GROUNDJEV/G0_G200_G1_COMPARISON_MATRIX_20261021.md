# GroundJEV G0/G200/G1 comparison matrix

The requested mainline is **GroundJEV single-backbone coupling**: GroundAnything perception, typed JEV identity decisions, and persistent identity memory are trained and evaluated as one system. G0 and G200 below are preserved controls. The new GPU5 G1 continuation has not started because its dependency owner is correctly waiting for the terminal G2 Clean receipt.

| Arm | Current state | Scope and evidence | Scientific boundary |
|---|---|---|---|
| **G0** | Complete reference | Frozen GroundAnything-DLM on 1,658 legal MOT20-05 TRAIN frames: HOTA 0.433441, DetA 0.638348, AssA 0.294655, IDF1 0.441085, IDSW 6,239, FP 6,325, FN 98,471. | Reference scope only; not MOT20 TEST and not an independent G2 result. |
| **G200** | Complete isolated control | 200-step GPU5 localization adaptation, 27,267,584 trainable parameters. Registered controls cover MOT17-02 first 4 frames, RMOT single8 and RMOT multi12; the corresponding HOTA values are 0.840987, 0.187761 and 0.233548. | Short functional controls for the historical isolated projector arm; no full-video or held-out claim, and not the coupled model. |
| **G1** | Waiting | `native_tracking200_gpu5_g1_v1/queue_owner_state.json` is `WAITING_G2_CLEAN_SUCCESS`; optimizer updates 0 and GPU5 training has not started. | No G1 checkpoint or metric claim exists. |

The coupling implementation itself has passed bounded real-data engineering runs: the official dense decoder, typed identity/lifecycle readout, memory-token projector, and fixed-candidate video replay are present. The bounded DanceTrack run and MOT20-05 cross-sequence replay remain diagnostic evidence only. They do not establish a benchmark gain; the independent held-out gate is still required.

Authoritative machine-readable details are in [`G0_G200_G1_COMPARISON_MATRIX_20261021.json`](G0_G200_G1_COMPARISON_MATRIX_20261021.json). The original G0 receipt and G200 artifacts remain untouched.
