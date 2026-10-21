# G2 Clean checkpoint step 17000

The protected exact-once two-rank G2 Clean run produced
`checkpoints/adapter-step017000.pt` and `checkpoints/latest_training_state.pt`.
Both were loaded successfully on CPU with `weights_only=False` while the live
trainer continued at step 17003/18227.

The two files report step `17000`, world size `2`, the registered Clean TRAIN
manifest SHA
`f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`, the
strict training contract, optimizer and scheduler state, sampler position,
and Python/NumPy/Torch/CUDA RNG state. The terminal receipt and terminal
adapter are not yet present, so downstream G1/TAO/OVMOT/joint/held-out gates
remain closed.

This is checkpoint/recovery evidence only; it makes no MOT, RMOT, OVMOT or
coupled tracking performance claim.
