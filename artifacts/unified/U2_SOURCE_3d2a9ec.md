# Original U2 source history — 3d2a9ec

Original local commit: `3d2a9ec790aea938570fc176476be9c19234ad70`. Source tree: `3f92b5abe7a15eb0a1b922abdd86eb286f7b4ee5`.
The adjacent incremental bundle requires `30c7a560e13c9e0a367444d442ab7d4c33e820c4` (preserved in the earlier original history artifacts).

This stage compiles all 500 TRAIN videos / 491,353 dense+modal images into separate base-only supervision with 1,489,990 unique source GT units. All nine zero-area original annotations remain. No novel or validation identity labels enter training. Physical filename/camera timestamps retain missing-frame gaps. Actual shared F0 BERT provides all 1,203 textual class features; same-core semantic loss and official unmodified TETA export/evaluation now execute. Actual eight-GPU BF16 MOT and OVMOT proofs, causal/save-load checks and official TETA component fixtures passed.

These are source/data/contract results, not benchmark tracking results. Complete selected candidate caches, simple metrics, learned MOT-only and joint TETA remain unmeasured. Formal joint remains conditional on eligible selected MOT-only; no RMOT/prompt tracking or U3 readiness is claimed. Frozen original perception and downstream queue owners continue.
