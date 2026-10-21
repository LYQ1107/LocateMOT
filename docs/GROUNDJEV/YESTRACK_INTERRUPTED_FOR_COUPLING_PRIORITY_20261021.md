# YesTrack paused for the GroundJEV coupling mainline

The latest GroundJEV direction prioritizes the single-backbone coupling path:
G2 Clean completion, GPU5 G1 adaptation, and native MOT/RMOT/OVMOT OneJev
joint optimization. YesTrack is an auxiliary identity-verifier study, not a
dependency of that mainline.

The real YesTrack retry therefore was stopped cleanly after it reached step
`20/273` (`160` examples) on physical GPU7. Its partial training log and all
existing queue/config records remain preserved under
`/data3/user/LocateMOT-checkpoints/yestrack_native_v2_lora` and the retry queue.
No adapter or validation receipt was promoted, and no YesTrack metric is
claimed. The GPU7 resource is available for the independent held-out gate after
G2 completes.

The protected G2 Clean trainer remains the active mainline optimizer; no G2
checkpoint or historical G0/G200/G1 artifact was modified.
