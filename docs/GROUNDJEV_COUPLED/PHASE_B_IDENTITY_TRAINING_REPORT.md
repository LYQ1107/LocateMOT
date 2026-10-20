# Phase B real native-G2 readout pilot

The Phase-B pilot is a bounded learnability gate for the coupled identity
readout. It is not a MOT, RMOT, OVMOT, or YesTrack benchmark and it does not
establish a scientific gain.

## Contract

- Backbone: the protected GroundAnything G2 visual path with the durable
  step-11000 projector (`f2b7199f57dae156a05e1efcadfa04b275a7f0399187b264e0b21ddc1b58409b`).
- Backbone state: frozen; only the typed GroundJEV readout was optimized.
- Data: six legal TRAIN examples selected from the 100-row internal joint
  corpus; native GroundAnything candidates and causal past-track memory were
  used. No MOT20-05 rows, official TEST labels, future evidence, or GT teacher
  injection entered the model input.
- Workload: 20 AdamW updates, one real visual forward per source image (six
  calls total), choice and Yes/No losses, finite-gradient check, and a saved
  readout checkpoint.

## Observed result

The run completed with finite values. Total loss decreased from `3.1731844`
at step 1 to `2.1330864` at step 20; identity-choice loss decreased from
`2.2314405` to `1.8834045`, and Yes/No loss decreased from `0.9417437` to
`0.2496820`. The final clipped gradient norm was `0.2682035`. The six
examples comprised native RMOT/KITTI and DanceTrack rows. Peak allocated
memory was 3,072,477,696 bytes and elapsed time was 5.0875 seconds after
model construction.

The checkpoint is recorded outside the repository at
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_b_yesno_pilot_v4/readout_step_final.pt`,
SHA256 `4707d52812cc8ce5d3ebecd35049b161a81726686456c9c2a940330e8336b101`.
The complete machine-readable receipt is the same directory's
`phase_b_receipt.json`.

## Interpretation and next gate

This proves that real G2 region evidence, causal dynamic slots, the typed
choice head, and the Yes/No head can participate in a finite training loop.
It does not prove identity association, calibration, HOTA, IDF1, IDSW, TETA,
or generalization. The next valid step is a larger TRAIN-only corpus with a
held-out development split and fixed candidate/association evaluation before
any claim or promotion.

