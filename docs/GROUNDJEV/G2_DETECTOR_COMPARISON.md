# G2 detector comparison boundary

G0 is the published `GroundAnything-DLM` base. G1-200 is the prior isolated projector-only adaptation over the smaller legal MOT17/MOT20/DanceTrack/Refer-KITTI mix; its measured G0/G1 controls were mixed and mostly negative, so it remains a preserved baseline. G2 is a new experiment: full projector plus Qwen3 q/k/v/o LoRA, a larger legal TRAIN-only multidomain corpus, and the official native DLM objective.

No G2 detector recall or MOT/RMOT/OVMOT benchmark score is claimed from the preflight receipts. The comparison report will be populated only after a frozen G2 checkpoint is evaluated on independent held-out scopes with unchanged evaluator contracts. G1, G200, and G0 artifacts remain untouched.

The GPU5 G1 continuation is registered in
`tools/groundjev/queue_g1_gpu5_after_g2.py`. Its owner waits for a successful
`G2_CLEAN_FULL_COVERAGE_V2` terminal receipt, then trains a fresh 200-step
projector from G0, exports a separate `GroundAnything-DLM-TrackProjG1-GPU5-V2`
bundle, and runs the fixed 96-frame G0/G200/G1 controls. A failed G2 receipt
causes a recorded refusal; the current G2 owner and all prior G1/G200 artifacts
are never modified.
