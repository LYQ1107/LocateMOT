# L83 four-GPU training/resource report

The faithful target-bag probe used four DDP workers (`gpu_world_size=4`) in
`outputs/l83/train/faithful_bag_attempt1/`.  The decoder sharpness diagnostic
also used four workers in authoritative attempt9.  These are bounded frozen
representation diagnostics, not the planned large task-composition training.

The faithful run had 524 fit groups, 138 video-disjoint dev groups, ten
epochs, 5,240 finite/nonzero-gradient trace entries per representation, and
reloadable compact probe checkpoints.  No raw/dense feature cache was
persisted.  The later four-GPU task-composition phase was **not run** because
the faithful gate failed.  No 500/1000/2000/5000 task-composition schedule,
historical 16/24 semantic run, screening, official test, TrackEval, HOTA,
ordinary MOT, or OVMOT run is present.
