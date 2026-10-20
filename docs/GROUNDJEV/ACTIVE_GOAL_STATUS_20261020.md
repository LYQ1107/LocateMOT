# GroundJEV-MOT active execution audit — 2026-10-20

This audit uses runtime receipts and live process handles, not stale owner
labels alone.

## Protected and queued work

- **G2 Clean:** live two-rank torchrun on physical GPUs 3 and 5, event step
  `10654 / 18227` at `2026-10-20 22:48:18 +0800`; the final receipt does not
  exist yet. The latest durable adapter remains `adapter-step010500`. The
  original optimizer, sampler, RNG, world size, and checkpoint contract are
  unchanged.
- **GPU5 G1:** `queue_g1_gpu5_after_g2.py` is live with state
  `WAITING_G2_CLEAN_SUCCESS`. It will train the registered fresh 200-step G1
  projector only after the protected G2 receipt, then export and run isolated
  controls.
- **OneJev full source joint:** the historical four-GPU run really logged
  through step `3525`, but has no terminal `done` record. Its last atomic full
  FP32/Adam/scheduler/all-rank-RNG state is step `3000`. The new queue owner
  `full_joint_source_epoch1_resume_queue_v2` is live and waiting for valid G2,
  successful G1, and exclusive physical GPUs 4/5/6/7. It will resume from
  step3000 and never overwrite the historical output. Its new owner also
  waits for the independent G2 held-out queue to reach a terminal state before
  claiming GPUs 4/5/6/7.

- **Independent G2 held-out queue:** live on GPU7, with a completed exact
  image-path overlap audit (0 overlaps for both RMOT scopes and independent
  TAO video105). It is waiting for the terminal G2 Clean deployment bundle;
  no final held-out metric is claimed yet.

## Completed evidence

- **G0 MOT20-05 long export:** the registered 1,657-frame validation-half
  export and fixed memory controls are complete. It is video-disjoint from
  Clean TRAIN; it is not a full MOT20 TEST claim.
- **G2 step3000 independent MOT:** the same-contract MOT20-05 comparison is
  complete and separately reported in
  `G2_MOT20_05_INDEPENDENT_VALIDATION_20261020.{md,json}`.
- **J1 held-out gate:** the registered DT1203, RMOT single8/multi12, and OVMOT
  video4/video63 receipts are complete. The zero/empty outputs and malformed
  decoder rejections are retained as failure evidence, not performance gains.
- **Existing G1 control:** the older GPU5 fresh-projector controls are
  complete for 4 MOT + 8 RMOT-single + 12 RMOT-multi frames. They remain short
  controls; the new G2-gated GPU5 run is a separate artifact.
- No live independent MOT-only optimizer process is present. Historical
  MOT-only outputs and failed owners remain preserved.

The audit does not claim that the overall research goal is complete. A valid
final G2 receipt, fresh G1 receipt, resumed full-source OneJev terminal
receipt, and official independent held-out comparisons are still required.

The recent G2 event windows measure 11.44–11.88 seconds per step, leaving
approximately 24.1–25.0 hours of training from step 10654. This estimate
excludes the queued G1, bundle export, and held-out stages.
