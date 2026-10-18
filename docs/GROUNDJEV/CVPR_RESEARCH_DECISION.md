# CVPR research decision — 2026-10-18

1. Protect the live MOT20-05 G0 export and preserve its causal prefix.
2. Continue the real MOT/RMOT/OVMOT OneJev SFT on the balanced birth-aware rows; the old MOT-only long training is already stopped and has no usable checkpoint.
3. Keep G1 as a measured negative detector result rather than spending another GPU allocation on a repeat.
4. Complete the expression-conditioned RMOT native pilot before claiming native RMOT supervision. Teacher-forced RMOT/OVMOT rows remain explicitly marked `GT_teacher` until native rows pass the match-rate gate.
5. Evaluate only from a saved valid checkpoint on independent held-out videos. No HOTA, detector gain, or full unified result is declared before that evaluation.

The active goal remains `IN_PROGRESS`.
