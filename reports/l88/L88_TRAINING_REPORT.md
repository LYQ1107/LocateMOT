# L88 Training Report

Date: 2026-09-04  
Project: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`  
L88 worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L88`  
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`

## Evidence class and scope

This is a completed RMOT-only fit experiment. It is not a public test result,
screening result, or final RMOT claim. The L69 budget-40 candidate bank, L86/L87
sidecar, ordinary MOT/OVMOT/TAO paths, old checkpoints, fixed manifest, and
TrackEval implementation were kept outside the L88 model changes.

The registered input cache was the query-independent GroundingDINO encoder
input cache: 2,520 entries and 58,812,203,520 bytes on `/data2`. It contains
no labels, query IDs as semantic features, or semantic scores. No raw image,
dense attention, or full-model checkpoint was added by the training run.

## Configuration and execution

The executed command was:

```text
/home/lwr/anaconda3/envs/masaenv_debug/bin/python3.11 tools/l88_train_full_rmot.py --epochs 40 --seed 20260829 --effective-frame-batch 8 --query-tile 4 --device cuda:0 --out outputs/l88/train/joint40_world4_retry1
```

The completed retry used world size 4, effective frame-group batch 8,
accumulation 2, 2,640 optimizer steps, and 20 even-epoch checkpoints (2
through 40). The registered BF16 path was not usable with the local MMCV
deformable-attention kernel, so the run used FP32 adapter/LoRA computation;
this is recorded as `precision_deviation`, not silently presented as BF16.

The final training record reports:

- 40/40 epochs complete and finite;
- V1/V2 counts 266/262 in the recorded training aggregation;
- category counts inactive 367, multi-positive 565, positive 545, and
  present-uncovered 198;
- 8,827 masked missing-target instances and 346 temporal identity pairs;
- all 24,288 recorded gradient entries finite and nonzero in the final
  aggregate;
- candidate deletion and truncation false, and peak resident GPU memory
  16,938,205,184 bytes.

The final epoch-40 checkpoint is:

```text
/data2/usr_for_deadline/locatemot_l88/project_outputs/train/joint40_world4_retry1/checkpoint_l88_epoch040.pt
SHA256 8714816470c88e23960251f7f27e653049112470ae7ccd87838e44eb31cb646e
```

The checkpoint selected later from development evidence is epoch 20:

```text
/data2/usr_for_deadline/locatemot_l88/project_outputs/train/joint40_world4_retry1/checkpoint_l88_epoch020.pt
SHA256 7012706140cfa94278ce15bb8da3e2318eb3efcfb8d26d3b1dbc5206f1145538
```

## Contract and failure history

The label-free differentiable smoke, distributed regression, and temporal
regression passed. The first formal 40-epoch attempt stopped because a causal
history key was taken from a cross-query store; it is retained with its
`INCOMPLETE.md`. The minimal fix uses the current frame's query ID. The retry
then completed without key drift.

The L88 LoRA contract remained restricted to the final two GroundingDINO
fusion layers and decoder layer 0, rank 16, alpha 32, zero-initialized B
factors. GroundingDINO backbone/text body, bbox head, decoder layers 2--6, the
L69 bank, and the sidecar protocol remained frozen as registered. No token/span
to region or static/motion supervision was available; both remain `UNALIGNED`.

## Status

Fit training is complete and reproducible from the recorded artifacts. It
does not by itself establish expression correspondence or ordinary-level
RMOT performance. The dev selection, fixed semantic gate, and internal
TrackEval evidence are reported separately.

