# LocateMOT-U final goal

## LOCATEMOT_U_FINAL_GOAL

Build **one unified specification-conditioned multi-object tracking
checkpoint**, named `UnifiedSpecTrack`, with one shared visual foundation, one
object/track representation, one persistent identity dynamics core, one track
memory, one online state format, and task-conditioned specifications.

The release checkpoint must support:

- closed-set ordinary MOT (`TASK_MOT`),
- open-vocabulary MOT (`TASK_OVMOT` plus class text),
- referring MOT (`TASK_RMOT` plus a natural-language expression),
- point, box, and mask prompt tracking (`TASK_POINT`, `TASK_BOX`,
  `TASK_MASK`).

Task-specific modules may contain task tokens, a specification encoder, small
adapters/LoRA, prompt encoders, and output heads. MOT, OVMOT, and RMOT may not
receive separate backbones or separate tracker cores. At least 90% of
non-head parameters must be shared in the final audit.

## Target gates

| Family | Target |
|---|---:|
| Refer-KITTI V1 | HOTA >= 45 |
| Refer-KITTI V2 | HOTA >= 40 |
| Refer-Dance | HOTA >= 42 |
| OVMOT novel | TETA >= 34; novel AssocA >= 40 target |
| Ordinary MOT | HOTA and AssA regression <= 1.0 point versus reconstructed single-task baselines |
| Prompt tracking | overall persistence >= 0.60; point/box/mask each >= 0.55 |

## Non-negotiable protocol

1. Development loaders use `locatemot/unified/data/legal_scope.py` as the
   single forbidden-video guard. Refer-KITTI V1 videos `0005,0011,0013` and
   V2 videos `0005,0011,0013,0019` remain unread until a final post-development
   decision.
2. The 157 legal V2 source target-ID anomalies are represented as
   `SOURCE_TARGET_ID_MISSING`. They do not create positive or false-negative
   box/tracking loss and are never silently dropped, filled, guessed, or
   relabelled.
3. Candidate and expression grounding gates precede tracking. The current
   Swin-T GroundingDINO probe has IoU@0.50 `0.1463`, so it cannot be used as a
   final RMOT candidate bank and no semantic head will be trained on that
   stream before the proposal gate is repaired.
4. Historical L69/L89E/R1 artifacts are unavailable. New measurements are
   reconstruction experiments and must carry their own checkpoint, manifest,
   split, and code hashes.
5. The final result is eligible only when one serialized checkpoint passes the
   ordinary MOT, OVMOT, RMOT, prompt, shared-parameter, and online-state gates.

## Staged route

`U0` freezes scope and audits external baselines. `U1` establishes a
shared MM-GroundingDINO-B foundation and passes generic proposal and
expression grounding gates. `U2` trains the shared MOT/OVMOT track core with
2/4/6/8-frame curriculum. `U3` adds semantic routing, conditioned visual
sampling, motion/relation reasoning, reliability fusion, rescue queries and
RMOT heads. `U4` adds point/box/mask prompt adapters. `U5` performs low-LR
all-task consolidation and selects one checkpoint by a pre-registered
composite score.

The current delivery is the U0 specification and skeleton. It is not a claim
that the final goal has been reached.
