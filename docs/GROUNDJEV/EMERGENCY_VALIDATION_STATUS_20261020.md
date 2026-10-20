# GroundJEV emergency validation status — 2026-10-20

This receipt covers the latest repair pass without restarting J1 or altering the
protected G2 Clean owner.

## Delivered code and audit chain

- The YesTrack branch contains the complete TrackJEV dependency tree,
  including `decision_schema.py`, `candidate_competition.py`,
  `tracker_decision.py` and `global_identity_runtime.py`.
- A clean detached worktree passed `compileall`, TrackJEV imports and self-tests,
  and loaded a real OneJev/J1 checkpoint for one real MOT17 frame.
- The Clean V2 evaluator labels the adapted arm `G2-CLEAN-V2`; `G2-PL` remains
  reserved for the historical projector-only result.
- The integrated branch is `research/groundjev-emergency-repair-20261020`;
  this status update is the next commit after the previously published
  `be16ed80c1a2aee32ce2aac391d9f392d6cc95d5`.

## G2 Clean training

The protected two-rank owner remains on physical GPUs 3 and 5 and is still
running. The latest accounting event is step **10540 / 18227** at
`2026-10-20 22:26:42 +0800`, with loss `0.334050` and peak allocation
`12.8289 GiB`; the latest durable checkpoint is `adapter-step010500`. The last
500 events average `11.46 s/step` (last 1,000: `11.69 s/step`), giving a linear
remainder estimate of about **24.5–25.0 hours**. This is an estimate, not a
completion time; the active process and checkpoint cadence were verified after
the independent evaluation.

The Clean TRAIN manifest has 583,251 rows. The overlap audit found exact
TRAIN-scope overlap for MOT17-02 frames 302–305 and 602 rows from the same
video; Refer-KITTI 0007 RMOT controls had no exact TRAIN path match.

## G2 step3000 expanded MOT result

A complete 600-frame MOT17-02-FRCNN run finished for both G0 and the preserved
G2 step3000 arm under the same parser, decoder, association, evaluator and
input manifest. The official metrics are:

| arm | HOTA | DetA | AssA | IDF1 | IDSW | FP | FN | rejected | outputs | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 0.357098 | 0.667771 | 0.194362 | 0.330325 | 822 | 1621 | 3212 | 48/600 | 17,121 | 0.1455 |
| G2 step3000 | **0.502578** | **0.716588** | **0.353559** | **0.495423** | **413** | **1664** | **2780** | 63/600 | 17,524 | 0.1574 |

This is a positive full-video result on the registered scope, including a
409-IDSW reduction and 432-fewer FN, but it is **not independent generalization**:
MOT17-02 is in the Clean TRAIN scope. The detailed report and hashes are in
`G2_FULL_MOT17_02_EVAL_20261020.{md,json}`. The video-disjoint registered validation-half result is reported below; it remains
a validation-half and is not a full MOT20 TEST or paper-level test claim.

## J1 and OVMOT

J1 step 1200 is immutable. On the repaired DanceTrack 1203-frame scope the full
parser plus `NO_MATCH=only_if_omitted` arm reports HOTA `0.158562`, DetA
`0.197692`, AssA `0.127243`, IDF1 `0.128277`, IDSW `0`, FP `110`, FN `3560`.
Strict ablation A/B/C remain zero; D/E are the only nonzero arms, so parser
repair alone is not an identity gain. OVMOT video4 has 40/40 rejected frames,
0 predictions and TETA/LocA/AssocA/ClsA all zero; video63 remains blocked by the
same malformed-output/very-slow-frame boundary.

## YesTrack

The real GPU7 native Yes/No run completed 100/100 steps. The 4,516-row causal
corpus is video-disjoint between train and validation and has no GT in prompts,
no future evidence, no MOT20-05 and no official TEST labels. On the same 137
valid held-out rows (23 of 160 over the fixed 16,384-token limit):

| arm | accuracy | ROC-AUC | PR-AUC | Brier | ECE | uncertain reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OneJev base | 0.6277 | 0.8031 | 0.8224 | 0.1777 | 0.2842 | 0.3066 |
| J1 step 1200 | 0.5401 | 0.6694 | 0.7466 | 0.2763 | 0.4648 | 0.1752 |
| YesTrack fine-tuned | **0.7810** | **0.8880** | **0.8963** | **0.1761** | 0.3374 | **0.0949** |

Fine-tuning improves discrimination and reduces uncertain rejection; ECE is
worse than base, so no calibration improvement is claimed. The real physical
replay hook passed on four exact MOT frames: disabled replay reproduced the
source, the fine-tuned arm made six real verifier calls through the same global
solve, and ran at 1.379 FPS. Four frames are a wiring receipt, not a HOTA/IDF1/
IDSW result; full physical comparison remains pending a complete causal
candidate/history/media archive.


## G2 step3000 video-disjoint MOT20-05 validation half

The registered MOT20-05 validation half completed for both G0 and G2: 1,657 continuous frames (1659–3315), video-disjoint from Clean TRAIN. Official metrics are HOTA/DetA/AssA/IDF1 `G0=0.332021/0.617396/0.178873/0.292386`, `G2=0.478449/0.797046/0.287713/0.420743`; IDSW `11289 -> 11117`, FP `16056 -> 20737`, FN `100036 -> 28100`. Rejection was `117/1657` for G0 and `66/1657` for G2. G0 end-to-end FPS was `0.054891`; G2 resumed successful frame-pipeline FPS was `0.043444`, and a full end-to-end FPS is intentionally not claimed because the run restored a 91-frame causal prefix after an interrupted first attempt.

This is a registered validation-half result, not full MOT20 TEST and not an isolated JEV identity claim. The detailed receipt is `G2_MOT20_05_INDEPENDENT_VALIDATION_20261020.{md,json}`; raw results remain under `/data3/user/LocateMOT-runs/GroundJEV/g2_step3000_full_eval_mot20_05`.
