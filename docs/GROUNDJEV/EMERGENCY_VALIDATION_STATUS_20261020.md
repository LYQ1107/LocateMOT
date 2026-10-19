# GroundJEV emergency validation status — 2026-10-20

This is the current cross-branch receipt after the latest repair pass.  It
does not restart J1 or alter the running G2 owner.

## Delivered code and audit chain

- The YesTrack branch now contains the complete TrackJEV dependency tree,
  including `decision_schema.py`, `candidate_competition.py`,
  `tracker_decision.py` and `global_identity_runtime.py`.  A clean detached
  worktree passed `compileall`, TrackJEV imports and self-tests, and loaded a
  real OneJev/J1 checkpoint for one real MOT17 frame.
- The Clean V2 evaluator labels the adapted arm `G2-CLEAN-V2`; `G2-PL` remains
  reserved for the historical projector-only result.
- The integrated research branch is
  `research/groundjev-emergency-repair-20261020` at commit
  `85b40c066f93dfcd9bca91a05a42d9f7b573ab5c`.  The source YesTrack branch is
  at `39274d25722506505d838f67c356489de40d3a3e`.

## G2 Clean training

The protected two-rank owner is still running on physical GPUs 3 and 5:

- step **5117 / 18227**, latest loss `1.312698`, peak allocation `17.371 GiB`;
- latest event `2026-10-20 00:19:27 +0800`;
- latest durable checkpoint is `adapter-step005000` with both rank RNG files;
- the last 1,000 event interval averaged 16.67 seconds/step, giving a linear
  remaining estimate of about **60.7 hours**.  This is an estimate, not a
  completion time; the last 100 steps were slower because of variable load.

The 24 short controls were audited against all 583,251 Clean TRAIN rows.
MOT17-02 frames 302–305 are exact TRAIN-scope overlaps and are retained only
as diagnostics.  Refer-KITTI 0007 RMOT controls had no exact TRAIN path match.
No independent full-video G2 metric is claimed until the clean bundle and a
larger legal development scope are complete.

## J1 and OVMOT

J1 step 1200 is immutable.  On the repaired DanceTrack 1203-frame scope the
full parser plus `NO_MATCH=only_if_omitted` arm reports HOTA `0.158562`, DetA
`0.197692`, AssA `0.127243`, IDF1 `0.128277`, IDSW `0`, FP `110`, FN `3560`.
The strict ablation shows A/B/C all remain zero; D/E are the only non-zero arms,
so parser repair alone is not an identity gain.  The OVMOT video4 receipt has
40/40 rejected frames, 0 predictions and TETA/LocA/AssocA/ClsA all zero;
video63 remains blocked by the same malformed-output/very-slow-frame boundary.

## YesTrack

The real GPU7 native Yes/No run completed 100/100 steps.  The 4,516-row causal
corpus is video-disjoint between train and validation, has no GT in prompts,
no future evidence, no MOT20-05 and no official TEST labels.  On the same 137
valid held-out rows (23 of 160 over the fixed 16,384-token limit):

| arm | accuracy | ROC-AUC | PR-AUC | Brier | ECE | uncertain reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OneJev base | 0.6277 | 0.8031 | 0.8224 | 0.1777 | 0.2842 | 0.3066 |
| J1 step 1200 | 0.5401 | 0.6694 | 0.7466 | 0.2763 | 0.4648 | 0.1752 |
| YesTrack fine-tuned | **0.7810** | **0.8880** | **0.8963** | **0.1761** | 0.3374 | **0.0949** |

Fine-tuning improves discrimination and reduces uncertain rejection; ECE is
worse than base, so no calibration improvement is claimed.  The real physical
replay hook also passed on four exact MOT frames: disabled replay was identical
to the source, the fine-tuned arm made six real verifier calls through the same
global solve, and ran at 1.379 frames/s.  Four frames are a wiring receipt, not
a HOTA/IDF1/IDSW result; the complete-video comparison still requires a shared
candidate/history/media key.

The detailed receipts are `YESTRACK_REAL_GPU_VERIFICATION_20261020.*` and
`YESTRACK_PHYSICAL_REPLAY_SMOKE_20261020.*` on the integrated branch.
