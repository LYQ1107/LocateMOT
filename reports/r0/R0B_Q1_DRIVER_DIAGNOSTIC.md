# R0B Q=1 driver diagnostic

Status: `preserved_driver_diagnostic_not_primary`.

The previous R0A formal invocation was stopped gracefully on 2026-09-09 after
the R0B instruction identified that it was using the protocol-inconsistent
Q=1 driver.  Only the exact Python process launched from this worktree was
terminated; its output was not deleted or modified.

## Invocation and output

- Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R0A`
- Branch: `codex/r0a-frame-specific-target-contract-repair-20260909`
- Code at stop: `1f6a73efb3320466d1c2442b891cbfbcd983dda7`
- Command:
  `/home/lwr/anaconda3/envs/locatemot/bin/python tools/r0_train.py --dataset refer_kitti_v1 --dense-root outputs/r0/data/v1_dense_train_index_retry2 --visual-cache /data2/usr_for_deadline/locatemot_r0a_visual_tokens_retry3_final --out outputs/r0/train/v1_formal_retry1 --epochs 12 --steps-per-epoch 4000 --device cuda:0`
- Output root: `outputs/r0/train/v1_formal_retry1/`
- Stop signal: graceful `SIGTERM` to the identified PID only.

## Observed progress

- Completed loss records: `8307`.
- Last recorded global step: `8307`.
- Last complete epoch: `2` (the epoch-02 checkpoint was written).
- Partial epoch at stop: epoch `3`, step `307/4000`.
- Existing checkpoint: `checkpoints/checkpoint_r0_refer_kitti_v1_epoch02.pt`
  (format `locatemot-r0-tcgh-checkpoint-v1`, global step `8000`, unchanged).
- The interrupted process did not write a final status/provenance package;
  the preserved loss trace and checkpoint are the available evidence.
- Peak memory was not recorded by the interrupted v1 driver.  A post-stop
  `nvidia-smi` snapshot showed GPU0 at about 2038 MiB, which is not a peak
  measurement.

## Eligibility decision

The old checkpoint and trace are retained for diagnostics only and are marked
here as `PRIMARY_SELECTION_ELIGIBLE=false`.  They are excluded because this
run used one query per optimizer step, one GPU, no gradient accumulation,
the old query-centric category cycle, and no grouped same-frame query tiles.
It therefore does not implement the registered R0 protocol and must not enter
formal checkpoint selection or be compared as a primary R0 result.  The model,
loss equations, visual cache, dense labels, and all frozen MOT/OVMOT assets
remain unchanged.

