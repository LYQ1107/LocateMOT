# LocateMOT Research Log

Concise experimental log (latest first).

## 2026-10-01 — latest R1 restoration completed

- Hypothesis: restore only the newest R1 execution boundary and its legal
  public-data/runtime inputs on `/data2/user/LocateMOT`; do not pretend that
  unavailable L69/L89E/R1 payloads were reproduced.
- Restored: official KITTI training images, Refer-KITTI V1, Refer-KITTI V2,
  MMDetection 3.3.0/GroundingDINO, BERT, and the direct R1 source dependency
  modules. V1 audit is complete. V2 has 157 legal expression-frame target
  references (11 repeated physical video/frame/ID pairs) pointing to IDs
  absent from corrected labels; missing frames are zero, and the anomaly is
  retained for the next adapter instead of silently filtered. The new public
  query manifest records every affected row.
- Runtime evidence: GroundingDINO loads the SHA-verified official checkpoint
  with 172,977,693 frozen parameters; one real KITTI frame/expression GPU
  inference returned 300 finite candidate boxes and finite scores. R1 imports
  and compile checks pass after path migration. The safe source loader also
  reads all 8,397 legal V1/V2 expression records from the restored
  per-expression layout without opening official-evaluation videos.
- Failure/limit: historical R1 checkpoints, L69 bank, dense indexes,
  language/visual/aligned caches and fixed manifest are unavailable. No old
  HOTA or score is claimed as reproduced; the old R1 sidecar is not extended
  with threshold/layer/learning-rate sweeps.
- Next action: construct a new legal public-data manifest/index and then a
  freshly generated candidate bank before any new correspondence training.
  The manifest is now complete. A 40-row midpoint-frame GroundingDINO probe
  found mean best IoU `.3188`, IoU@.25 `.6585`, IoU@.50 `.1463`, and mean best
  score `.0107` over 123 target boxes; the frame-zero attempt is preserved as
  a separate artifact. This supports a new bank audit but is not an RMOT
  score. Official-evaluation videos remain unread.

## 2026-08-19 Stage L11 — OVMOT temporal pseudo-track + KITTI front-end

- Hypothesis: unmatched DLA detections on expanded TAO-train OVMOT
  stream must receive high-precision temporal pseudo-track supervision
  instead of "every detection is NEW"; KITTI RMOT DetPr ~1% is a
  candidate front-end problem, fixable with category whitelist + NMS +
  query-conditioned CLIP top-k.
- Implementation:
  - Pseudo-track generator (forward + backward cycle consistency,
    appearance/motion/category filters, confidence gating) for all 500
    TAO train videos; sidecars in `outputs/l11/data/pseudo_tracks`.
  - Class-A GT upgraded from C-TAO base @0.5 IoU to base_and_novel
    @0.3 IoU (~10% -> ~28% coverage).
  - Trainer: no_unmatched_new, confidence-weighted identity/relevance
    losses, pseudo ids in the same slot/lifecycle machinery.
  - KITTI: whitelist + cross-NMS + top-30 -> ~15.3 cands/frame (was 50);
    CLIP top-12 calibration: query precision 10.2%, target recall 65%.
- Quality: pseudo same-ID precision 99.26% (8-video audit, target 90%);
  cycle pass 0.997; NEW rate 0.110 vs L10 collapse ~1.0.
- Running: 4-GPU repair training (from L9-ovmot, ~6-8 s/step),
  repaired-KITTI eval (L9-ovmot ckpt), 10-video OVMOT over-birth
  baseline: unique/rows 0.211 (L9), reused IDs 467.
- Pending: step-7000 OVMOT quick eval, full TAO TETA, KITTI TrackEval,
  L12 prompt seeding.

## 2026-08-20 continued

- Low-LR continuation (10000->11000, LR~6e-6): RMOT-Dance 32.49 ->
  34.05 HOTA; ordinary Macro 0.4855 -> 0.4924.  OVMOT full TETA on
  s11k running.
- Fresh-optimizer balance run (p_rmot 0.45/p_ovmot 0.20) REJECTED at
  step 1000 (RMOT 32.06 / ordinary 0.4663).
- L12 frozen prompt-seeded (DAVIS 2017, 10 multi-object videos):
  mask/point > box for identity robustness; joint fine-tune not
  launched (mixed signal).  Infrastructure complete.

## 2026-09-04 Stage L88 — RMOT-aware GroundingDINO LoRA adaptation

- Hypothesis: a zero-initialized rank-16/alpha-32 LoRA on the final two
  GroundingDINO fusion layers and decoder layer 0, with the frozen L86/L87
  sidecar and causal history, would improve held-out target-bag and
  expression-to-candidate correspondence without changing the L69 bank.
- Contract: label-free, distributed, and temporal regressions passed; full
  V1/V2 fit used seed `20260829`, 40 epochs, 2,640 optimizer steps, world 4,
  effective group batch 8, and 20 even checkpoints. Local MMCV required FP32
  rather than the registered BF16 autocast. Candidate deletion/truncation was
  false and all frozen-base/LoRA gradient and reload checks passed.
- Dev: all 20 checkpoints were scored on 138 video-disjoint groups; the fixed
  selection chose epoch20/Rule B before fixed validation. Checkpoint SHA is
  `7012706140cfa94278ce15bb8da3e2318eb3efcfb8d26d3b1dbc5206f1145538`.
- Fixed semantic validation: immutable L29 was
  `recall=.7333333, precision=.0830189, FP/frame=10.125,
  pred/positive=8.8333, hard=.9166667, multi=.8194444`. L88 candidate-only
  was `recall=.4193548, precision=.1830986, FP/frame=2.4167,
  pred/positive=2.2903, hard=.8461538, multi=.3333333`; final Rule B was
  `recall=.1935484, precision=.1363636, FP/frame=1.5833,
  pred/positive=1.4194, hard=.8461538, multi=.1666667`, with empty rate
  `.5416667` and inactive false acceptance `.5`. The semantic gate failed on
  recall and multi-positive preservation; the lower volume is not a fix.
- Internal TrackEval: selected Rule B produced HOTA `26.0914` (V1) and
  `20.2386` (V2), below L86 `29.1663/21.6467` and L87-A
  `28.5752/22.1300`. No screening or official-test labels were read.
- Preserved failures/fixes: first formal causal-key drift, sparse seqinfo,
  merge, device, and score-shape retries remain in new directories with their
  provenance; no old bank/checkpoint/production entrypoint was overwritten.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`; no L88 long continuation,
  screening, official test, TrackEval beyond internal validation, or
  ordinary MOT/OVMOT change is authorized. Unique next action is supervisor
  review of one new RMOT correspondence/proposal design.

## 2026-09-05 Stage L88C — corrected candidate-vs-NULL replay

- Hypothesis: the prior L88 semantic failure may include an implementation
  error in the candidate-vs-NULL gate. Corrected emission is
  `candidate_energy >= candidate_threshold` AND
  `candidate_energy-null_logit >= null_margin` AND
  `presence_logit >= presence_threshold`; candidate-only uses only the first
  condition. No training, backward, optimizer, checkpoint update, bank
  change, top-k/NMS or NULL suppression was performed.
- Implementation: refit all 20 even checkpoints on the registered B/R/P
  grids, replayed the corrected shortlist (epochs 8/20/40/30/4) over complete
  V1/V2 dev video, ran the corrected internal TrackEval matrix, froze epoch
  30 Rule B before fixed validation, and ran the fixed 16-calibration/
  24-validation replay. A first TrackEval provenance alias failure was
  preserved; the minimal `scope_key/full_video` alias fix was pushed in
  `c5a6685` and the corrected matrix is attempt2.
- Result: L88C corrected final on validation was
  `recall=.354839, precision=.207547, FP/frame=1.75,
  pred/positive=1.7097, hard=.846154, multi=.305556,
  empty=.375, inactive-FA=.666667`. Immutable L29 remains
  `.733333/.083019/10.125/8.8333/.916667/.819444` for
  recall/precision/FP/pred-positive/hard/multi. Candidate-only recall was
  `.419355`; the corrected NULL comparison removed additional positives.
- Internal dev TrackEval for the frozen epoch30/Rule B was V1
  HOTA/DetA/AssA `.296615/.216472/.408302` and V2
  `.265682/.156622/.451938`; these are dev-only, not screening or official
  results. The semantic gate failed on recall and multi-positive recall.
- Root cause: existing candidate scores have useful average pairwise ranking,
  but the lowest-scoring positives in multi-target bags do not survive the
  fixed emission boundary; V2 is worse. This is a correspondence and
  multi-positive recall failure, not a universal NULL-acceptance,
  coverage, finite/reload, or implementation-smoke failure.
- Preserved outputs: corrected replay, two TrackEval attempts, frozen
  selection, fixed semantic JSON/gate, and offline A–I diagnosis under
  `outputs/l88c/`. Code was pushed to branch
  `codex/l88c-candidate-null-corrected-replay-20260905` at `e8f5f62`.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`. No screening/official-test
  labels, HOTA claim beyond internal dev TrackEval, ordinary MOT/OVMOT/TAO
  change, or new L88C training was run. The only next action is the single
  supervisor approval request in `reports/l88c/NEXT_TEST_APPROVAL_REQUEST.md`.

## 2026-09-06 — L88C recoverable storage cleanup

- While the authorized zero-training L88C final-internal replay was running,
  its open paths were checked. Three superseded L1 intermediate LoRA
  checkpoint directories (`outputs/l1_c/checkpoints/lora/checkpoint-100/200/300`,
  about 8.0G each) were not open by any process and were moved, not deleted,
  to `/data2/usr_for_deadline/locatemot_cleanup_20260906/`.
- Frozen banks, formal reports, current L88 epoch30 checkpoint/cache, and all
  ordinary MOT/OVMOT assets were left in place. `/data1` free space increased
  from about 76G to 114G. The recoverable inventory is documented in
  `reports/l88c/L88C_RESOURCE_CLEANUP_20260906.md`.

## 2026-09-07 Stage L89 — query-conditioned set correspondence decoder

- Hypothesis: a fresh QSC-D head over the frozen L84/L87-A Z1 candidate states,
  with candidate self-attention followed by cross-attention to pure frozen
  GroundingDINO/BERT language tokens, could improve same-frame correspondence
  without changing the bank, detector, tracker, or ordinary MOT/OVMOT.
- Implementation: isolated worktree at
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89`, base
  `ece9fc5549a9210424eb127fb10430c3fed7b7ba`; two set layers, eight heads,
  FFN 1024, hidden 256, history 8, 4,169,829 trainable parameters. Pure
  language cache has 3,418 label-free entries. Contract smoke passed after
  preserving two package/field failures. No token/span or static/motion
  supervision is verified (`UNALIGNED`).
- Fit: seed `20260829`, one GPU, BF16, 40 epochs/2,640 optimizer steps,
  finite loss and nonzero gradients, all 20 even checkpoints plus final.
  Only L49 fit supervision was used; no screening or official-test labels.
- Dev/selection: 20 checkpoints scored over 138 legal dev groups; five-item
  shortlist and full-video B/R/P matrix selected epoch 4, Rule R before fixed
  validation. Checkpoint SHA is
  `5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`.
- Fixed semantic: authoritative retry4 failed. L29 was
  `recall=.7333333, precision=.0830189, FP/frame=10.125,
  pred/positive=8.8333, hard=.9166667, multi=.8194444`; L89 was
  `recall=.8709677, precision=.0794118, FP/frame=13.0417,
  pred/positive=10.9677, hard=.6923077, multi=.9027778`, with inactive
  false acceptance `1.0`. Hard-negative ranking moved, but deployable volume,
  precision, and no-match behavior failed.
- Internal TrackEval: frozen epoch4/Rule R on V1 validation (86 sequences)
  gave HOTA `2.2906`, V2 validation (537 sequences) HOTA `0.5894`, versus
  L87-A `28.5752/22.1300`. Full-video post-hoc target-row recall was only
  `.0131/.0026` and inactive acceptance was `1.0`. This is internal
  validation TrackEval evidence, not screening or official-test evidence.
- Root cause judgment: query-conditioned set representation gave limited
  frame-level ranking signal but failed calibrated persistent emission and
  absence handling. The branch is stopped pending supervisor review; no L89
  continuation, screening, official test, or ordinary MOT/OVMOT change is
  authorized. All L89 outputs are preserved under `outputs/l89/` and the new
  source/reports are on branch `codex/l89-query-conditioned-set-correspondence-20260907`.

## 2026-09-08 Stage L89C — corrected candidate-vs-NULL replay

- Hypothesis/repair: L89 training and the registered loss use `candidate_energy`
  against `null_logit`, but the old L89 evaluation/deployment path emitted
  using `presence_logit - null_logit`. L89C changed only the evaluation,
  selection, inference, and TrackEval contract to the shared
  `corrected_emission_mask`: candidate threshold, candidate-minus-NULL margin,
  and presence threshold. No training, scorer forward, bank/cache rebuild, or
  model/ordinary-MOT change was made.
- Inputs/selection: immutable L89 raw dev records (9,960; SHA recorded in
  `outputs/l89c/dev/corrected_shortlist_attempt1/`), 20 existing checkpoints,
  and the fixed manifest SHA
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
  Corrected dev TrackEval evaluated 5 checkpoints × B/R/P (15 results), then
  froze epoch 2 / Rule R, checkpoint SHA
  `f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b`, with
  candidate/presence/NULL thresholds `-0.75/-1.0/0.0` before fixed validation.
- Fixed semantic evidence: authoritative retry2 has 40 ordered records (16
  calibration, 24 validation), with preselection forbidden-label fields
  absent. Validation is recall `.8387097`, precision `.0869565`, FP/frame
  `11.375`, predictions/positive `9.6452`, hard violation `.6923077`,
  multi-positive recall `.8333333`, inactive false acceptance `1.0`.
  The gate is `semantic_gate_fail`: hard/recall/precision/multi floors pass,
  but volume and inactive/no-match conditions fail.
- Internal evidence: frozen epoch-2/Rule-R full-video inference covered V1
  `0004,0018` (86 sequences) and V2 `0016,0017,0020` (537 sequences).
  Corrected TrackEval HOTA is V1 `2.244718%`, V2 `0.583059%`; no screening or
  official-test labels were read. These are internal validation-scope metrics,
  not a final RMOT result. Historical buggy L89 was `2.2906%/0.5894%` and
  L87-A was `28.5752%/22.1300%` on V1/V2.
- First actionable root cause: corrected candidate-vs-NULL accounting still
  leaves no-match acceptance universal and output volume too high. All rows,
  keys, finite checks, and frozen boundaries passed; the initial GPU OOM is
  preserved as an implementation/resource attempt and the exact CPU retry
  completed. Unique next action: supervisor review and authorization of one
  new RMOT branch explicitly addressing absence/volume calibration with
  candidate correspondence. Do not extend L89/L89C or tune thresholds.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`; `zero_training=true`,
  `screening_gt_used=false`, `official_test_labels_read=false`,
  `ordinary_mot_ovmot_touched=false`.

## 2026-09-08 Stage L89D — true full-video timeline repair

- Hypothesis/scope: repair only the L89C sparse-prediction versus dense-GT
  timeline mismatch. No training, model, loss, bank, tracker, threshold, or
  ordinary MOT/OVMOT change was made. New code is confined to `tools/l89d_*`
  and new evidence is under `reports/l89d/` and `outputs/l89d/`.
- Contract repair: native L69 `frame_ids` and frame pointers now drive every
  video timeline; all query/frame pairs and all candidate rows are scored.
  The multi-video query-map overwrite was fixed and covered by a targeted
  regression. Dev inference attempt3 covers 1,152,690/1,152,690 candidate
  pairs across 2,385 frame groups; internal attempt1 covers
  243,550/243,550 pairs across 1,844 frame groups. Both passed full-video and
  pair-completeness checks. The dev and internal dense Z1 supplements were
  built from frozen query-independent caches and finalized with no labels.
- Dev evidence: the valid 5-checkpoint × B/R/P TrackEval matrix selected
  epoch 8 / Rule P on the preregistered dev tuple. The frozen checkpoint is
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch008.pt`,
  SHA256 `28d4c35a189474d446a26fc096bbafbad77e1298429674383ce7ce8b14677d99`.
  The fixed 16-calibration/24-validation semantic replay then failed: recall
  `.5483871`, precision `.1517857`, FP/frame `3.9583333`,
  predictions/positive `3.6129032`, hard violation `.6923077`,
  multi-positive recall `.4861111`, inactive false acceptance `.6666667`.
  Recall and multi-positive floors failed; no screening/test labels were read.
- Internal evidence: fixed epoch 8 / Rule P true full-video TrackEval gives
  combined HOTA `23.8586078862%`, DetA `15.1294290119%`, AssA
  `37.9461647913%`, DetRe `57.2559333985%`, DetPr `16.9954368707%`, IDF1
  `18.6931722536%`, and IDSW `6337.5`; V1 HOTA `25.0798607896%`, V2 HOTA
  `22.6373549828%`. These are internal validation-scope TrackEval results,
  not screening, official-test, or final ordinary-RMOT evidence.
- Attempts preserved: dev ENOSPC, internal exit-137/OOM retries, the expected
  smoke full-video rejection, the selector directory-argument error, and the
  earlier multi-video GT-map mismatch. The final boundary guard passed after
  the path contract correction. The fixed manifest SHA remained
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
- Status: `L89D_COMPLETE_ZERO_TRAINING_TRUE_FULLVIDEO_REPAIR /
  semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`.
  Flags: `zero_training=true`, `screening_gt_used=false`,
  `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
  Unique next action: supervisor review and authorization of one separate
  absence/volume-calibration study; do not extend L89/L89C or alter the
  production MOT/OVMOT paths.

## 2026-09-09 Stage L89E — phase-consistent temporal/history replay

- Hypothesis/scope: L89D evaluated every checkpoint with temporal history
  enabled even though L89 training used Stage-S epochs 1–8 with temporal off
  and zero history. L89E changed only this replay contract: S epochs 2/4/6/8
  use `temporal_enabled=false` and zero history; T epochs 10–20 and J epochs
  22–40 use causal last-four history. No training, model, loss, bank, tracker,
  threshold science, UIDM, ordinary MOT or OVMOT source was changed.
- Implementation: isolated branch/worktree based on L89D commit
  `d0960e1d1679414765f79203bef444f0f9968928`; phase policy and wrappers are
  in `tools/l89e_*`. The required CPU equality check passed for S and T
  `L86ClipStore` history construction. The first replay failure was a
  `FrameExample`/`_clip_history` interface mismatch; the minimal adapter fix
  passed a targeted S/T causality regression. A subsequent historical-score
  merge check found that the immutable sparse file contains S+T/J rows; the
  wrapper now filters only the historical T/J epochs and preserves their
  values. Launch-path and shortlist argument mistakes were preserved as
  `INCOMPLETE`/operator evidence; no affected model run was treated as data.
- Stage-S evidence: fresh phase-correct replay of epochs 2/4/6/8 completed
  `4 x 498 = 1,992` records, all finite, complete and zero-history. Merged
  phase-consistent fit/dev pool contains 20 epochs and 9,960 records. The
  registered maximum-five fit/dev shortlist was frozen at epochs
  `8(S),20(T),40(J),32(J),4(S)`.
- Dev selection: full native dev replay retained `11,003,527/11,003,527`
  candidate rows for all five shortlist candidates. The 15-result TrackEval
  matrix selected epoch 4 / phase S / Rule B by the registered tuple; dev
  HOTA was `0.296512` (internal fit/dev selection evidence only).
- Fixed semantic evidence: the new 16-calibration/24-validation replay used
  the frozen epoch-4/S zero-history policy and Rule B. Immutable L29 remains
  recall/precision/FP-frame/pred-positive/hard/multi
  `.7333333/.0830189/10.125/8.8333/.9166667/.8194444`. L89E validation is
  recall/precision/FP-frame/pred-positive/hard/multi
  `.4516129/.1458333/3.4166667/3.0967742/.7692308/.3611111`, with inactive
  false acceptance `.8333333` and empty rate `.2083333`. The semantic gate
  fails recall and multi-positive floors despite lower volume and higher
  precision; no threshold rescue was performed.
- Internal TrackEval: after the frozen selection, exact internal timeline
  replay covered `243,550/243,550` pairs for V1 0004/0018 and V2 0016/0017/0020.
  Phase-consistent HOTA/DetA/AssA/DetRe/DetPr were V1
  `28.7628/18.7680/44.3927/80.3045/19.5312%` and V2
  `21.8385/13.3358/35.9990/45.1160/15.8179%`; IDSW was 1,997 and 9,915.
  These are internal validation-scope TrackEval results, not screening or
  official-test results.
- Root cause: correcting S-phase temporal/history evaluation does not
  restore expression correspondence; the fixed emission remains recall- and
  multi-positive-limited, with substantial inactive acceptance. Status is
  `L89E_COMPLETE_ZERO_TRAINING_PHASE_CONSISTENT_REPLAY /
  semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`.
- Preserved outputs: Stage-S attempts 1–4, merge attempts 1–2, shortlist
  attempts 1–4, dev/internal full-video and TrackEval outputs, fixed semantic
  output, and all reports under `reports/l89e/` and `outputs/l89e/`. The fixed
  manifest SHA remains
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
  Flags remain `zero_training=true`, `screening_gt_used=false`,
  `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
- The only next action is supervisor review of a separately authorized
  absence/volume-calibration study. Do not extend L89/L89C/L89E, alter the
  bank, or touch production MOT/OVMOT.

## R0A — safe target contract, dense native indexes, and implementation smoke — 2026-09-09

- Hypothesis: isolate only allowlisted V1/V2 target records, repair the
  frame-specific target contract, and construct native-frame R0 data without
  parsing forbidden official-eval payloads.
- Safe-source retry3 passed: lexical monolithic-JSON isolation deserialized
  only allowlisted video values and recorded
  `forbidden_payload_deserialized=false` and
  `official_test_labels_read=false`.  The exact 5,314 sparse fit rows matched
  the authoritative frame map with zero sentence/target/missing-query/invalid-
  frame mismatches.
- Dense native-frame index retry2 and its contract audit passed for V1/V2;
  all L69 rows and duplicate candidate indices were retained.  The geometry
  `image_hw` and visual-cache dataset/group-key contracts were corrected in
  R0-only files.  Retry1 dense/smoke failures remain preserved with their
  first trace; retry2 is authoritative.
- Real R0 contract smoke retry2 passed on one V1 fit frame: loss
  `3.555824041366577`, 108/108 finite nonzero trainable gradients, frozen
  GroundingDINO, no tracker optimizer parameters, strict reload max diff `0`,
  and temporary visual item removed.  This is implementation evidence only;
  no semantic, screening, official-test, HOTA, or production MOT/OVMOT result.
- Next action: commit/freeze the R0A safe code and construct the registered
  query-independent visual cache on `/data2`; no old asset or production
  entrypoint is modified.

## R0B — grouped DDP driver repair and formal legal evidence — 2026-09-09

- Hypothesis: the old Q1 query-at-a-time driver was the actionable R0 training
  contract bottleneck. A frame-centric grouped driver with complete candidate
  sets, Q=8, deterministic strata, and DDP accumulation should make the
  registered R0 experiment executable without changing the frozen bank,
  tracker, detector, or ordinary MOT/OVMOT.
- Preserved/disqualified Q1: PID 39571 was stopped gracefully; the old partial
  `outputs/r0/train/v1_formal_retry1/` remains historical and is not
  selection-eligible.
- Repairs: grouped Q8 schedule, video-local tiles, world4 DDP, accumulation 4,
  frozen detector/language inputs, complete candidate rows, present-uncovered
  masking, and resumable `locatemot-r0-tcgh-checkpoint-v2`. The legal GT
  materialization gap was repaired only in the R0 wrapper by recording and
  skipping missing annotation rows; frozen GT was not edited.
- Formal training: V1 completed 12 epochs/3,058 optimizer steps and V2
  completed 12 epochs/3,046 steps. Both were finite with nonzero gradients,
  strict reload, all four strata and both domains. Checkpoints 2/4/6/8/10/12
  are retained. Formal jobs used the free physical GPUs 4–7; physical GPUs
  0–3, which had unrelated processes, were not touched.
- Legal dev selection: V1 shortlist epochs 12/6 and V2 12/10; full dev
  TrackEval selected epoch12 for both. Dev HOTA was V1 22.8681% and V2
  19.5801% for the selected checkpoints.
- Internal validation-scope TrackEval: V1 0004/0018 HOTA 26.4582%, V2
  0016/0017/0020 HOTA 15.6097%. These are not screening or official-test
  results. No screening/official labels were read, and ordinary MOT/OVMOT,
  UIDM, L69 and production entrypoints were untouched.
- Outputs: training remains under `outputs/r0/`; large dev/inference/TrackEval
  products are under `/data2/usr_for_deadline/locatemot_r0b_*` because `/data1`
  was nearly full. No raw/dense debug cache or detector copy was created.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`. This is valid grouped R0
  implementation plus legal dev/internal TrackEval evidence, not final
  ordinary-RMOT completion. The only next action is supervisor review and a
  separately authorized follow-up.

## R1 — aligned track conditioning legal-dev contract — 2026-09-10

- Hypothesis/scope: test a zero-initialized RMOT-only residual conditioned on
  aligned frozen L89E Stage-S Z0/Z1/Z4 states and causal observation history;
  the anchor, L69 bank, ordinary MOT/OVMOT, UIDM and all production paths stay
  frozen. This is not final RMOT completion.
- Read-only legal-dev audit: frozen epoch-004 Rule-B anchor selection was
  rechecked. The epoch-004 presence-only miss is `3/400=.0075`, below the
  preregistered `.05` activation threshold, so the formal R1 config freezes
  `presence_residual=false`. Existing full-video artifacts have no candidate
  score arrays for a valid EXACT_INDEX_DEDUP replay; the conservative RAW
  policy is therefore frozen without claiming a dedup comparison.
- Preserved evidence: `outputs/r1/audit/legal_dev_selection_attempt1/` and
  `reports/r1/R1_LEGAL_DEV_SELECTION.md`; no training or new checkpoint was
  created by the audit. The next action is the bounded wiring/reload smoke;
  formal fit remains conditional on that implementation gate.

## R1 — formal fit and legal-development replay — 2026-09-12

- Hypothesis/scope: a zero-initialized aligned track-conditioning residual
  over frozen L89E stage states, causal observation history, language-directed
  R0 detail and complete L69 candidate sets could improve expression-to-track
  correspondence without changing the anchor, bank, tracker or ordinary
  MOT/OVMOT paths.
- Formal fit completed in isolated worktree
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R1`: six epochs, 2,250
  optimizer steps, all finite with nonzero gradients, strict reload for eight
  V1/V2 checkpoints, frozen anchor, and no persistent raw/dense debug cache.
- Prediction-only replays for legal-dev V1 videos `0008/0010/0020` and V2
  videos `0000/0008/0009` completed before legal GT was opened. The aggregate
  contains 2,385 groups and eight internal TrackEval results. Registered
  selection chose epoch 4 separately for V1 and V2. Selected HOTA was
  `24.2345%` (V1) and `23.6542%` (V2), versus the L89E anchor's `28.7628%`
  and `21.8385%`; this is not a stable breakthrough and remains legal-dev
  evidence only.
- Fixed 16-calibration/24-validation diagnostic retained all 2,435 L69 rows
  and 40 keys. R1 candidate-only recall/precision/FP-frame/pred-positive/
  hard/multi were `0.1000/0.1842/0.7750/0.5429/0.7727/0.1818`; the frozen
  Rule-B replay emitted zero rows for both anchor and R1 on this slice. The
  lower volume is a recall/multi-positive collapse, not a semantic repair;
  no threshold or NULL rescue was attempted.
- Status: `R1_NO_BREAKTHROUGH / STOPPED_PENDING_SUPERVISOR_REVIEW`.
  Outputs and failed technical attempts are preserved under `outputs/r1/`
  and `reports/r1/`. Flags remain
  `screening_gt_used=false`, `official_test_labels_read=false`, and
  `ordinary_mot_ovmot_touched=false`. The only next action is supervisor
  review and explicit authorization of one separate follow-up; no automatic
  continuation, screening, official test, or production merge is allowed.

## R1 — fixed semantic diagnostic closeout — 2026-09-12

- The fixed-slice evaluator used the native L62 order and attached labels only
  after row/feature construction and scoring. It used the frozen Rule-B values
  `candidate_threshold=1.0`, `presence_threshold=0.5`, `null_margin=0.0` and
  did not fit or tune a new threshold.
- This is a diagnostic, not a preregistered semantic gate and not HOTA. The
  authoritative files are in `outputs/r1/eval/fixed_semantic_attempt1/`; the
  corresponding report is `reports/r1/R1_FIXED_SEMANTIC_DIAGNOSTIC.md`.
# 2026-10-01 — new-server restoration started

- Hypothesis: restoring public V1/V2 data and exact base inference dependencies enables a new, attributable correspondence experiment; extending the failed R1 sidecar is unjustified.
- Evidence: fetched all remote branches; latest remains R1 `99090c8`. Read chronological historical endings and R1 reports: V1 legal-dev regression, limited V2 gain, fixed-slice score mismatch/output collapse. Old trained checkpoints are unavailable.
- Changes: started resumable official KITTI S3 **direct** download (15,813,146,295 bytes); downloaded and CRC-validated V1 expression/labels archives; isolated local data-tools environment. Full inventory/short plan: `reports/RESTORE_STATUS_20261001.md`.
- Failure/repair: shell sandbox lacks bubblewrap; execution requires tool escalation. Standard venv lacks ensurepip; local virtualenv bootstrap succeeded. Neither is a scientific result.
- Results: public V1 annotations recovered; KITTI/V2/runtime/project-generated assets pending. No historical checkpoint/bank reproduction claimed. Keep recovery scripts and new provenance; preserve migration AGENTS.md edit.

## LocateMOT-U U0 — unified specification and reference audit — 2026-10-01

- Hypothesis: a single specification-conditioned foundation can absorb the
  reusable persistent-query, temporal-language, open-vocabulary and reliability
  ideas from the ten requested repositories, provided the legal video boundary
  and one-checkpoint/shared-state contract are frozen before training.
- Code: cloned and shallow-audited MOTR, TransRMOT, TempRMOT, DKGTrack,
  FlexHook, iKUN, OVTR, COVTrack, ReferDINO and Open-GroundingDino; created
  `UnifiedSpecTrack` U0 shape skeleton, shared legal-scope guards, task/sample
  contracts, loss/runtime interfaces, parameter-sharing audit and reproducible
  audit/sanity tools. Exact source recipes and commit IDs are in
  `reports/unified/OPEN_SOURCE_TRAINING_AUDIT.md` and
  `outputs/unified/open_source_audit.json`.
- Data/protocol: froze V1/V2 fit/calibration/validation splits and reserved
  V1 `0005,0011,0013` plus V2 `0005,0011,0013,0019`; retained all 157 V2
  source target-ID-missing rows as unlabeled records. No official-test labels
  or screening ground truth were opened.
- Checkpoint/GPU: no LocateMOT-U checkpoint and no training run yet; GPU
  allocation was zero. External sanity was compile/help only (about 23 s wall
  time): 10/10 selected source files compiled, 1/10 entrypoints reached help,
  and the nine remaining failures are recorded dependency/version findings.
- Results: skeleton CPU forward finite with score shape `[1,100]`; legal guard
  smoke passed; skeleton non-head parameter fraction `0.9998259786` (not a
  final-model claim). No benchmark metric was produced. U0 scope/audit/sanity
  passed; U1 MM-GroundingDINO-B acquisition and proposal/expression gates are
  next.

## LocateMOT-U U1 — Swin-B foundation acquisition and adapter smoke — 2026-10-01

- Hypothesis: the audited GroundingDINO Swin-B CogCoOR weights can serve as the
  shared visual/specification foundation after selecting the checkpoint format
  compatible with the local MMDetection GroundingDINO implementation.
- Assets: acquired and hashed both the original release
  (`46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7`,
  938,057,991 bytes) and the OpenMMLab conversion
  (`55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`,
  935,971,702 bytes). The MMDetection asset is the runtime candidate because
  its key layout matches the adapter; provenance is in
  `outputs/unified/protocol/foundation_asset.json`.
- Code/smoke: built the local Swin-B model (232,996,763 parameters), loaded the
  converted checkpoint with 8 expected initialization keys and 3 BERT metadata
  keys reported as non-fatal mismatch, then ran one CUDA label-free forward on
  legal V1 video `0004` frame `000063.png` using `car . pedestrian . cyclist .`.
- Results: 300 finite predictions, finite boxes and finite scores;
  `outputs/unified/u1_foundation_smoke.json` records the exact config,
  checkpoint hash, device and legal frame. This is a compatibility smoke only;
  no proposal recall, expression grounding metric or tracking result is
  claimed. Official-test labels and screening ground truth remain unread.
- Checkpoint/GPU/walltime: no LocateMOT-U checkpoint; one A800-equivalent GPU;
  model build/forward about 9 s. Next gate is label-free proposal and
  expression grounding on legal development frames.

## LocateMOT-U U1 — proposal gate diagnostic — 2026-10-01

- Hypothesis: the loaded Swin-B foundation would provide sufficiently localized
  open-vocabulary candidates for the shared semantic/tracking path on a fixed
  legal development slice.
- Protocol: 16 V1/V2 calibration/validation expression-frame rows, 50 target
  boxes, 300 candidates per row; model inference completed before each
  corresponding label file was opened. No threshold fitting, training,
  official-test labels or screening ground truth.
- Result after correcting the MMDetection BGR input contract: best-IoU
  `>=0.25` was `0.5400`, best-IoU `>=0.50` was `0.0400`, mean best IoU
  `0.2756`, and mean score at best IoU `0.01312`. Boxes and scores were finite.
  This remains below a usable proposal gate, so semantic-head/tracking training
  from this stream is disallowed. The initial RGB/BGR-swapped diagnostic was
  discarded and is not used as evidence.
- Diagnosis/next change: inspect prompt tokenization/category assignment, box
  decoding and proposal selection on this same legal slice. Preserve the split,
  rows, checkpoint and labels; do not tune a threshold or delete hard cases.
  Authoritative output: `outputs/unified/u1_proposal_gate.json` and
  `reports/unified/U1_PROPOSAL_GATE.md`.
- Selection-only check: retaining all 900 decoder queries raised IoU@0.50 to
  `0.1000` and mean best IoU to `0.3341`, so top-k selection contributes but
  does not repair localization. The top-900 output is retained separately and
  is not used as a training bank.
- A fixed class prompt (`car . pedestrian . cyclist .`) with all 900 queries
  reached IoU@0.25 `0.9000`, IoU@0.50 `0.1800`, mean best IoU `0.3635`; this is
  a useful direction for the repair but still below the stage gate and remains
  excluded from training. Output:
  `outputs/unified/u1_proposal_gate_generic_classes_top900.json`.

## LocateMOT-U U1.3 — proposal-selection repair diagnostics finalized — 2026-10-01

- Change: made the canonical 300-query output reproducible with explicit
  `max_per_img` and prompt metadata, and synchronized the report/status values
  with the corrected BGR run. Re-ran the 300-query baseline, the 900-query
  selection diagnostic, and the fixed class prompt (`car . pedestrian .
  cyclist .`) diagnostic on the identical legal 16-row slice.
- Results: the baseline stayed at IoU@0.25 `0.5400`, IoU@0.50 `0.0400`, and
  mean best IoU `0.2756`; retaining 900 queries reached `0.1000` at IoU@0.50;
  the fixed class prompt with 900 queries reached `0.1800` at IoU@0.50 and
  `0.9000` at IoU@0.25. All boxes/scores were finite, no rows were skipped,
  and no official-test labels or screening ground truth were read.
- Interpretation: prompt/category assignment and top-k selection are concrete
  repair directions, but the proposal gate remains below the stage criterion.
  No semantic-head/tracking training bank was created and no training was
  launched. The next experiment must keep this split/checkpoint fixed while
  inspecting prompt tokenization, category assignment and box decoding.
- Verification: unified modules/tools compile; CPU skeleton smoke is finite;
  parameter-sharing audit remains skeleton-only with non-head fraction
  `0.9998259786`; legal guard accepts V1 `0004` and rejects reserved `0013`.

## LocateMOT-U U0R/U1R measurement-contract repair — 2026-10-01

- Scope: resumed from `014b8a5ec81bb83c5993b56da171089f8a440a11` on
  `codex/u0r-u1r-foundation-repair-20261001`; no U2 work and no training was
  launched. The project root was verified as `/data2/user/LocateMOT`.
- Reproducibility hypothesis: a fresh checkout should import the unified data
  package using tracked source only. The first clone failed because the broad
  `data/` ignore rule also hid `locatemot/unified/data/`. Added a narrow ignore
  exception, committed the package source, pushed it, and re-cloned the remote
  branch at `26f9c989c8b13773d681bbe6962b77e73a0026bd`. Imports, skeleton smoke,
  and foundation smoke passed in the clone. Decision: keep the repair and mark
  U0 legal-scope reproducibility PASS.
- Checkpoint hypothesis: the MMDetection Swin-B conversion must have only
  documented non-core load mismatches. `tools/unified/audit_foundation_checkpoint.py`
  now requires the exact one training-only missing key
  `dn_query_generator.label_embedding.weight`, the exact three BERT metadata
  unexpected keys, and zero backbone/encoder/decoder/bbox-head/language-fusion
  missing keys. Audit result: `PASS_CONDITIONAL_TRAINING_ONLY_MISSING`.
- Preprocessing hypothesis: the formal runtime must be equivalent to the
  official MMDetection path. Ten deterministic legal images were compared with
  the same fixed generic vocabulary; all produced 300 finite predictions,
  maximum box difference `0.0`, and maximum score difference `0.0`. Decision:
  preprocessing contract PASS; the prior hand-built tensor smoke is not formal
  evidence.
- Split hypothesis: a deterministic video-disjoint split balancing expression
  volume, physical-target volume, and motion proxy is adequate before formal
  evaluation. `tools/unified/build_legal_split.py` froze v2 with seed `20261001`
  and split SHA
  `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`.
  V1 fit/calibration/validation are `9/3/3` videos and V2 are `9/3/3`; V2
  source-target-ID-missing references remain `157` (fit only in this split),
  with no replacement or silent drop. Decision: freeze this split before any
  U1 adaptation or formal gate.
- Hardware/wall time: one CUDA device (A800-equivalent) was used for the
  checkpoint audit, smoke, and equivalence check; these short checks completed
  in under a minute each. Official-test labels and screening ground truth were
  not opened. External baseline execution remains incomplete and is the next
  blocking U0 task.

## LocateMOT-U U0R — TempRMOT legal-development execution — 2026-10-01

- Hypothesis: at least one published external RMOT method can be executed on a
  frozen legal-development subset without reading official-test labels or
  mixing its environment into LocateMOT-U.
- Protocol: copied TempRMOT at commit
  `6a65640d849fdee4a32bb055945ee34c3b0edeb1` into the isolated environment
  `/data2/user/reference_envs/temprmot`, loaded the released checkpoint
  `checkpoint_rk.pth` (SHA-256
  `55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8`), and
  ran the adapter on V1 validation videos `0012` and `0014`, one legal
  expression per video. TrackEval then evaluated the two generated sequences
  with HOTA, CLEAR, and Identity.
- Result: prediction and evaluation completed with 94 prediction rows and 93
  ground-truth rows. Combined HOTA was `94.679`, MOTA `98.925`, and IDF1
  `99.465`. Official-test labels, screening ground truth, and training were not
  used. The raw outputs and environment remain outside the repository;
  `outputs/unified/external_baselines/temprmot_legal_v1.json` records the
  reproducible metadata and `reports/unified/EXTERNAL_BASELINE_LEGAL_DEV.md`
  records the bounded scope.
- Compatibility work was confined to the isolated copy: a torchtext shim,
  current PyTorch `torch.load` behavior, disabling a redundant pretrained
  download, a CUDA scalar-type API update, and NumPy aliases in TrackEval.
  These changes do not alter the LocateMOT-U model or the frozen split.
- Decision: keep the TempRMOT run as real external-baseline evidence, while
  retaining `U0_EXTERNAL_BASELINE=INCOMPLETE` until a second independent
  method also produces predictions and evaluation. DKGTrack and FlexHook
  remain pending because their published weight links are not yet available
  through the isolated download path.

## LocateMOT-U U1R — formal generic and expression measurements — 2026-10-01

- Measurement contract: the frozen split SHA is
  `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`; the
  foundation checkpoint SHA is
  `55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`; every
  forward used `LoadImageFromFile`, `FixScaleResize(800,1333)`,
  `PackDetInputs`, and `DetDataPreprocessor` with BGR-to-RGB conversion.
- Generic protocol: one fixed query-independent vocabulary, all legal
  calibration/validation frames with materialized labels, physical units
  `(dataset, video, frame, track_id)`, Top100/150/300/900 diagnostics, and
  small/medium/large height strata. The run covered 4,201 unique images,
  33,526 physical GT units, and 5,992 unique physical objects. Top150 recall
  was V1 `0.0803` and V2 `0.0669` at IoU 0.50 (V1/V2 IoU .25: `.5309/.4940`,
  IoU .75: `.0045/.0062`). Decision: formal generic gate FAIL; this is below
  the `.80` adaptation early-pass, so a detection adaptation probe is required
  before any grounding adaptation.
- Expression protocol: every legal calibration/validation query contributed
  one deterministic middle labeled frame and, when available, one first empty
  target frame. Units were `(dataset, video, query_id, frame, target_id)`;
  source-missing references stayed in the source denominator. The run covered
  7,160 query-frame rows and 7,199 valid target units with zero source-missing
  units in this calibration/validation sample. Top1 IoU .50 recall was V1
  `0.0030` and V2 `0.0028`; AP50 and empty-frame FP/query are recorded in
  `reports/unified/U1_EXPRESSION_GROUNDING_GATE.md`. Decision: formal
  expression gate FAIL; no expression adaptation has started.
- No official-test labels or screening ground truth were read. No U2 or
  universal track core work was started. The external-baseline state remains
  incomplete because only TempRMOT has real predictions/evaluation; DKGTrack
  and FlexHook public Baidu links require an authenticated download path.

## LocateMOT-U U1-A — protocol correction, valid adaptation, and stop — 2026-10-01

- The first eight-process launch was invalid because MMEngine's default
  `--launcher none` left the processes outside DDP; it was stopped before any
  formal evidence. A later exploratory FP32 run also used `frozen_stages=1`.
  MMDetection's Swin implementation shows that value freezes patch embedding
  and stage0 only, while the registered contract requires stages 0 and 1.
  Those exploratory metrics are excluded from the final record.
- Corrected hypothesis/protocol: a fixed legal-fit generic stream can repair
  the foundation proposal localization while freezing Swin stages 0/1 and
  BERT. The legal-fit stream has 3,606 rows and fixed `label 0 -> car`; the
  split, vocabulary, official `FixScaleResize(800,1333)` pipeline, and all
  denominators stayed unchanged. The valid run used
  `frozen_stages=2`, BF16 AMP, eight-GPU PyTorch DDP, batch 2/GPU, and no
  accumulation (global batch 16). The logged peak was 5,945 MB/GPU. A two-step
  smoke passed before the full run.
- A batch-2/two-step-accumulation attempt passed a one-step smoke but failed in
  the full run with PyTorch's `expect_autograd_hooks_` DDP reducer assertion.
  The smallest working fix was accumulation 1; this kept the recommended
  microbatch 2 and did not alter the official input resolution. This failure
  and the corrected smoke are retained in the adaptation report.
- The valid schedule completed epochs 1, 2, 4, and 6, each formally evaluated
  on the unchanged 4,201-image, 33,526-physical-unit scope. Top150
  Recall@IoU0.50 (V1/V2) was foundation `0.0803/0.0669`, then
  `0.8729/0.8515` (epoch1), `0.8873/0.8766` (epoch2),
  `0.8945/0.8843` (epoch4), and `0.8938/0.8815` (epoch6). IoU0.25 reached
  about `.99` in both domains, while epoch-6 IoU0.75 was `0.7215/0.6119`.
  Size-stratum IoU0.50 was small `0.9136`, medium `0.9445`, and large
  `0.7085`.
- Training logs contain occasional `grad_norm: nan` values, but loss values
  remained finite, all six checkpoints were written, and every formal shard
  produced finite predictions. This telemetry is reported rather than
  filtered. The final epoch-6 checkpoint SHA-256 is
  `a1ff39f9c0bd59a5fd2b14968a528fc11e882b8fa6a3e85a9b64253a1cdc5ccc`.
- Decision: the valid V1/V2 generic gate did not reach `.90/.90` after the
  registered six-epoch maximum. Expression adaptation was not started because
  it is downstream of the generic gate. Added the corrected adaptation report,
  failure decomposition, and four valid formal JSONs. Final status is
  `LOCATEMOT_U_BLOCKED_GENERIC_GATE_AFTER_MAX_ADAPTATION` with
  `STOPPED_PENDING_SUPERVISOR_REVIEW`; U2 was not launched.

## LocateMOT-U U1R2 — measurement repair, class-aware refinement, and foundation completion — 2026-10-01

- Contract hypothesis: the remaining U1 failures were caused by a measurement/data contract mismatch rather than a need to replace the MM-GroundingDINO-B foundation. The source `labels_with_ids` fields are normalized top-left `xywh`; the repaired formal gates preserve source extents and use Top150 IoU.50 for generic proposals and Top100 IoU.50 target coverage for expressions. The previous U1-A-v1 files remain preserved as historical artifacts and were not overwritten.
- Coordinate/source audit: legal calibration/validation contains 33,526 physical GT boxes. 195 V1 boxes extend below the image at `y1>height` (166 large, 27 medium, 2 small); no x-side overflow exists. Same-frame geometry matching to official KITTI labels gives mean IoU `0.7414883`; Refer and official track IDs are separate namespaces. Decision: retain source extents and recover classes by geometry; do not clip evaluation GT.
- Data hypothesis: a class-aware physical-frame union can repair localization while removing V1/V2 duplicate training. The fit stream contains 2,545 images and 10,502 boxes; 3,424 V1/V2 overlaps were removed, 580 conflicting boxes and 21 missing class matches were recorded, and observed classes are car 6,787, pedestrian 3,700, cyclist 13, van 2. Decision: retain the stream with fixed class policy and explicit low-confidence audit.
- Numerical stability: the prior BF16 log has three finite-loss `grad_norm:nan` records (epoch/iter 1/50, 3/100, 6/100). A fresh FP32 one-step audit from epoch4 found 981 trainable parameters, total loss `40.27210998535156`, zero nonfinite gradients, and zero no-gradient parameters. Decision: R2 uses FP32/TF32, no AMP/BF16.
- Generic refinement: one bounded FP32 epoch from the frozen epoch4 checkpoint passed the corrected gate: V1/V2 Top150 IoU.50 `0.982716/0.971837`; small/medium/large `0.937467/0.989984/0.997825`. Decision: freeze the R2 epoch1 generic checkpoint and skip epoch2/mixed fallback.
- Expression hypothesis: whole-expression positive spans with all referred boxes, legal empty frames, and preserved V2 missing-target references can add language-conditioned candidate coverage without erasing generic proposals. One FP32 epoch completed from the selected generic checkpoint. Expression Top100 IoU.50 is V1/V2 `1.000000/0.961367`; the same checkpoint's generic regression is V1/V2 Top150 IoU.50 `0.986592/0.958739`.
- Language audit: on exactly 500 legal validation same-image pairs with a same-video sentence from a different target set, true versus shuffled Top10 recall is `0.959333/0.871133` and AP50 is `0.736901/0.491159`. Both strict comparisons pass, so language conditioning is measurable rather than generic-box-only behavior.
- External evaluator: synthetic perfect/empty/ID-switch and cross-evaluator arrays pass with maximum difference `0.0`. TempRMOT evaluated videos 0012/0014 occur in its declared training pool; its HOTA 94.679 is reclassified as `PIPELINE_SANITY_ONLY_TRAINING_OVERLAP`.
- Decision: the final shared expression checkpoint satisfies all U1 hard gates. `official_test_labels_read=false`, `screening_gt_used=false`, V2 source-target-ID-missing references remain explicit (157, silently dropped=false), and no U2/U3/U4 work started. Set `LOCATEMOT_U_READY_FOR_U2` and stop pending supervisor review.

## LocateMOT-U U2 — shared persistent track core and portability stop — 2026-10-02

- Provenance: working branch `codex/u2-shared-persistent-track-core-20261001`,
  authoritative local base `56507f4aa9a12e76d6f9f6702f0f0b38163666ec`, and
  immutable U1 foundation SHA
  `4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.
- Implementation: replaced the U0 tracking placeholders with one shared U2
  core (`PerceptionBridge`, independent identity/semantic candidate encoders,
  `UnifiedTrackState`, four-layer identity set interaction, GRU identity
  memory/anchor/EMA, association/new/no-match, lifecycle, motion, Hungarian
  rollout, and bounded OVMOT residual). The runtime is
  `locatemot/unified/runtime/u2_online_tracker.py`; no RMOT or prompt tracking
  was launched.
- Contract evidence: `tools/unified/u2_causal_contract.py` passed birth,
  occlusion, no-future, MOT specification invariance, OVMOT isolation, and
  online-state save/reload. Actual-model sharing audit passed with
  `shared_non_head_fraction=0.9378111093950003`.
- U2-P data manifest SHA-256 is
  `575bd349e00bab23634c6929c04513149afec838de419bb3f1a53469fafc7fda`.
  KITTI legal development portability passes (V1/V2 Top150 @.50
  `0.986592/0.958739`), while BDD100K, DanceTrack, MOT17, MOT20, TAO, and
  C-TAO are missing locally. Candidate-cache SHA is `not built`.
- Training command: not launched; GPU count `0`, optimizer steps `0`, wall
  clock training runtime `0`. Simple MOT, MOT-only, and joint OVMOT metrics
  are intentionally `null` because the registered portability protocol is
  incomplete.
- Decision: `LOCATEMOT_U_BLOCKED_U2_PERCEPTION_PORTABILITY`; stop pending local
  acquisition/audit of the missing streams. Official test labels and
  Refer-KITTI official evaluation labels remain unread.
- Validation note: the 2-process CUDA forward/backward smoke passed with the
  same MOT task on both ranks. An 8-process smoke could not initialize because
  all eight A800 devices were occupied by an unrelated process; it was
  terminated before any training step and is not counted as a pass.

## LocateMOT-U U2 — acquisition recovery — 2026-10-11

- Revalidated local descendant `666ab44` and the actual U1R2 checkpoint SHA
  `4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.
  U1 references and the checkpoint remain unchanged.
- Observed MOT17/MOT20 extraction counts of 33,810/13,430 files, with nested
  release roots. TAO's four volume roots exist but contain zero files; no
  training or download worker was live. The old download module spawned
  workers while being imported by each child, recursively launching itself.
- Replaced that launcher with one guarded entry point and four thread workers
  running bounded download clients. Keep the registered four-volume assignment,
  prioritize modal train/validation inputs, validate downloaded sizes, and
  record actual per-file SHA-256 for source provenance. A fresh direct-network
  README transfer passed: 9,189/9,189 bytes, so existing Baidu authentication is
  usable. This does not imply that the dataset or portability gate is complete.
- Identified a separate eligibility bug: the old portability artifact could
  return PASS solely from directory availability while all new-domain metrics
  were null. Correct the audit and eligibility calculation before any training;
  metric definitions, split rules, thresholds, and seeds remain unchanged.

## U2 frozen-foundation MOT portability — 2026-10-11

Hypothesis: the selected U1R2 foundation may need bounded adaptation on crowded/non-road pedestrian domains. Registered ByteTrack inclusive train-half ranges, FRCNN physical-video deduplication, `person .`, positive mark-1/class-1 GT and unmodified source box geometry before measurement. The loaded checkpoint SHA matched the immutable U1R2 reference, with no missing/unexpected weights.

Ran all 7,115 dev frames on GPUs 1–7 in one parent launch/wait command, leaving an externally occupied GPU 0 alone. MOT17: 2,652 images / 53,890 GT; Top150 0.787233, Top300 0.806253. MOT20: 4,463 images / 615,137 GT; Top150 0.731010, Top300 0.884673. Exact coverage and monotonic count checks passed; partial measurements are rejected. MOT17-02/04 and crowded MOT20-03/05 dominate the shortfall. Retain the measured failure; do not alter visibility filtering, vocabulary, frame scope or gates to improve the reported values.

Track training is held. At most two epochs of mixed U2-P perception adaptation are justified after the missing domains/protocols are ready, with U1 replay and full retention gates. MOT20 Top300 has not met the preferred 0.90 or been accepted as a runtime override. TAO four-volume acquisition/extraction continues. Cloud BDD resources are static detection rather than MOT 2020. Started official DanceTrack train/val and C-TAO base-only acquisition, pinning release revisions; aria2 TLS failed, curl direct HTTP-206 probe succeeded and curl replaces it. No tracker or RMOT/prompt training was launched.

U2-P follow-up: C-TAO base-only source downloaded and inspected: 490,210 train images, 1,489,637 base-category boxes, 500 videos; supervision frequencies f/c only and no validation paths. LVIS v1 full vocabulary frozen (1,203 names; 866 f/c base, 337 rare novel). Native MMDetection chunking passed a real TAO-image smoke with 38 chunks and exactly one backbone call, followed by class-agnostic IoU-0.8 merge and global Top300. This is runtime evidence, not a substitute for full TAO coverage. Original modal TAO validation has 36,375 images / 113,112 boxes; 314 boxes lack a LVIS-v1 synset mapping, so the benchmark conversion/audit must explicitly account for them. Original BDD MOT 2020 image host dl.yf.io is live; started eight train/val archives (38,886,924,336 bytes) with original published MD5 checks, four bounded workers and eight connections/file. Official 2021 tracking-label S3 URL returns 404; keep BDD labels missing while locating a valid original release.

## U2 TAO/C-TAO portability registration — 2026-10-12

Continue from the published full MOT-half failure measurements, preserving the immutable U1R2 checkpoint. Plan: freeze all official TAO validation images plus every original annotated C-TAO training-audit frame; preserve the full dense C-TAO set for later identity training. Use all 1,203 LVIS-v1 names, the already verified 32-name chunks, one image backbone extraction, class-agnostic IoU-0.8 merge and global Top300. Count all original modal TAO GT including the 314 annotations without a v1 synset mapping; report those separately with source-frequency grouping rather than dropping them. Wait for CRC-completed source archives internally in the single evaluation launcher while downloads/extraction continue. In parallel, finish Dance extraction and locate original BDD tracking labels. No track training is eligible yet.

## U2 data completion and full portability launch — 2026-10-12

Hypothesis: recovering the original data and measuring complete dev scopes will distinguish foundation domain failure from incomplete measurement. DanceTrack download/CRC extraction completed; preserved 32/8/25 project videos and exact 33,772/8,024/25,508 frames. TAO modal train/val and all 490,210 C-TAO training images now pass availability coverage (six available streams, BDD pending). Frozen full TAO validation (36,375 images, 110,055 base / 3,057 novel GT) and original C-TAO train audit (17,131 images, 53,263 base GT); retain all 314 unmapped original TAO GT units. Immutable-foundation real smokes passed in both chunked domains and DanceTrack; full audits run on GPUs 1–6 and 7 respectively. No partial recall promoted; MOT gates still fail and track training remains held.

BDD root cause: old official annotation URLs are dead and the user cloud BDD folder is static detection. Found a pinned HF archive whose directory contains all 1,400/200 raw tracking JSON videos and validation images. Download/hash/selected-member CRC extraction is active; source labels have not yet established availability. Added original toolkit category/ignored/crowd and inclusive Box2D handling; raw-label full-scope validation will precede BDD measurement. Original seven train-image archives continue downloading, with an internal blocking extractor. Keep these recovery changes; no learned model, seeds or gates changed.

## U2 BDD raw tracking recovery validated — 2026-10-12

Recovered and SHA/CRC-validated the pinned mirror: all 1,400 train and 200 val raw JSON videos; every 39,973 validation image matches the original release ZIP member size/CRC. Frozen 278,079 train / 39,973 val frames and the fixed eight-category vocabulary before inference. Full official-reference comparison covers all validation objects: 442,814 positive boxes, 18,625 crowd and 4,073 official ignored-category objects. First strict comparison exposed a 2.84e-14 rounding difference from direct x2+1; changed arithmetic to the official xywh reconstruction order, then every positive box/category/ID matched exactly (no tolerance relaxation). Keep the correction. Full BDD portability is queued after the DanceTrack worker on GPU 7 with a real-image smoke first. The first original train-image archive is CRC-extracted (200 videos, 39,418 frames); remaining train-image downloads/extraction continue. Numerical MOT gate failures and track-training hold remain unchanged.

## U2-P bounded perception adaptation preparation — 2026-10-12

Plan before any adaptation result: use a fixed 20,000-anchor replay epoch, maximum two epochs, seed 20261012. U1 generic/V1-expression/V2-expression proportions 0.10/0.15/0.25; BDD/Dance/MOT17/MOT20/TAO base detection each 0.10. Stratified video sampling covers every training video where the allocated budget permits; cycle deterministically when replay pools are smaller than their allocations. Preserve original U1 replay boxes/phrases; use only official train or registered MOT-half train frames. TAO class supervision uses mapped LVIS-v1 f/c categories only, with novel/unmapped annotations excluded from class and identity supervision. Encode detection classes as positive phrase spans through the existing native ODVG grounding path. Empty positives retain a fixed absent base-class caption. Keep FP32, decoder/multimodal LR 1e-5, Swin stages 2/3 LR 1e-6, BERT and Swin stages 0/1 frozen. Do not start optimization until all source images and full F0 domain audits are complete. Any adapted checkpoint must pass the unchanged U1 retention and complete portability gates; no tracker/cache before that.

Implemented the fixed replay preparator, checkpoint-compatible frozen-BERT model, guarded eight-GPU trainer, explicit candidate-SHA evaluation, crowded-K registration, and full-audit checkpoint selection. Config parsing exposed an inherited-variable NameError; use MMEngine's `_base_.train_pipeline`. Disabled aspect-ratio buckets because odd bucket tails can add singleton steps; ordinary batching preserves exactly 1,250 steps/epoch at global batch 16. Actual immutable checkpoint CPU loading matches every state key, BERT stays eval/frozen, Swin stages 0/1 stay frozen and stages 2/3 train. Native ODVG packing passed on eight real training-stream images and one empty-positive image; first check assumed a box wrapper but the native pipeline returns tensors, so corrected that check and reran. Reaggregated all 7,115 original MOT frames: every result remains identical. Mixed-checkpoint measurement/worker aggregation is rejected; F0 MOT20 Top300 0.8846728452 correctly fails the mandatory 0.90 override threshold. Original generic/expression/500-pair full U1 reference scopes and shard provenance validate. Keep these changes. Replay preparation session 81929 waits internally for actual BDD extractor PID 686873; training preflight reports incomplete BDD images, incomplete full F0 audits and pending full replay. No perception optimizer, track training, or candidate cache launched.

Scientific source backups use `/data2/user/locatemot_git_backups/u2_latest.bundle`; the adjacent `u2_latest.commit` records the exact original local HEAD preserved in each updated bundle. Original U1R2 history and the BDD recovery commits remain in this history.

Scope correction before any optimization: an epoch-1 retention failure rejects that checkpoint for tracking, but the specification still permits the second bounded perception epoch. Allow that optimizer continuation after all epoch-1 audits complete; both retention and portability acceptance thresholds remain unchanged, and no rejected candidate becomes track-eligible. The two-epoch/2,500-step cap remains fixed.

## U2 DanceTrack full measurement and bounded continuation — 2026-10-12

DanceTrack's original blocking audit completed: all 25,508 dev images, 225,148 positive source GT; Top150 224,318/225,148 = 0.996313536, Top300 224,627/225,148 = 0.997685967. Full frame coverage, registered GT count and immutable foundation provenance pass. Keep the complete measurement; it shows that the current foundation's failure is not universal across ordinary MOT domains. BDD's queued real-image check passed and its complete 39,973-frame audit began on GPU 7; the second original training-image archive completed CRC extraction (200 videos, 39,990 frames). No partial BDD/TAO values are promoted.

Plan: connect the already registered two-epoch trainer to complete ordinary/TAO portability, original eight-shard U1 generic/expression and fixed 500-pair audits, conditional MOT20 K300 registration, and checkpoint selection. One blocking controller will wait for the existing F0 audit/data-preparation parents, train only after all prerequisites hold, pause at each saved epoch for full audits, and stop after selection or the absolute two-epoch cap. Preserve exact worker partitions and resume only unfinished units; no metric, seed, frame scope or acceptance threshold changes.

All original BDD train-image extraction and replay preparation completed successfully during this stage. Required data availability is now 7/7; 1,400 train videos and every 278,079 training annotation frame have an image. The fixed replay is complete (SHA `1be4a6e2d03fd3f59aeeca8b931c3f638a2ca1db015ea7d54afa2ca725970947`), all 1,400 BDD and 500 TAO train videos represented; actual prompts use at most 21 tokens. Native eight-rank Gloo verification used the real 20,000-row ODVG dataset and actual configured sampler/batching: 2,500 rows and 1,250 batch-2 steps per rank, globally no duplicate/omitted row, all 300,070 positive boxes retained, 1,385 empty-positive rows retained, zero native box-filter removals. No GPU model or optimizer was created by that check. The first launch exposed a missing `.venv/bin/torchrun` entry point; use the same virtual-environment Python's `torch.distributed.run` module, rerun the targeted eight-rank check, and apply the identical launcher fix to the continuous trainer. Keep this fix; no package, seed, precision, LR or budget changes. FP32 training keeps the already registered U1 TF32 flags. All original eight-shard U1 scopes validate, and the original U1 checkpoint is correctly rejected as a completed 1,250-step U2-P candidate. Only incomplete full F0 BDD/TAO audits now hold optimization. The controller validates each existing parent's full artifact as it exits, then owns training and all audits; it never terminates unrelated jobs or advances through a failed/incomplete measurement.

Scientific source commit `cf41d7312e7046c10ebc802aa624d1be81e1a021` is backed up in the latest local bundle. Launched the continuous command with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/run_u2_perception_adaptation.py --tao-audit-pid 670232 --bdd-audit-pid 781721 --replay-pid 816267 --extraction-pid 625610`. Controller PID 854970 / exec session 11636 is confirmed live and has already validated the completed replay parent; it still waits for the two original full F0 audit parents. GPU count 8; optimizer steps 0; candidate cache not built. Clarified the existing native resume semantics after reading installed MMEngine: optimizer, scheduler and sampler epoch resume, while separately launched epochs use the same fixed base seed 20261012; worker RNG streams are not restored. This documents the configured epoch-boundary behavior, without changing seeds or training implementation. End-to-end training wall time is pending the blocking controller's completion.

## U2 real feature bridge and eight-GPU temporal validation — 2026-10-12

Hypothesis: attaching final-decoder query provenance before native chunk concatenation/NMS and sampling pre-text neck maps can supply correctly aligned identity/semantic features without changing the frozen detector's candidate distribution. Implemented `runtime/u2_grounding_candidates.py`: original boxes remain original-coordinate xyxy; identity sampling scales them into resized/padded image coordinates, preserving preprocessor padding rather than normalizing against only original image extent. The extractor reads no annotations and stores no GT identities. Actual checkpoint-backed GPU-0 validation (`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/validate_u2_candidate_features.py --device cuda:0`, terminal session 84086, exit 0) passed native box/score/class bit-exact parity for MOT17-02 frame 000302 Top150 and one TAO validation frame under the complete 1,203-category/38-chunk Top300 protocol. Selected decoder features remain bit-exact after cross-chunk NMS; backbone executes once; fixed-box identity features remain bit-exact after changing vocabulary. Scope is an alignment contract, not a replacement for full-dataset recall. Keep the bridge; full cache awaits passing U2-P selection.

The eight-GPU temporal contract (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 tools/unified/u2_ddp_temporal_contract.py`, terminal session 18873, exit 0) passed both MOT and OVMOT. Two four-frame clips/rank (global 16) retain temporal gradients; identity GRU, set interaction and association gradients are nonzero, with semantic GRU/text gradients also nonzero on OVMOT. Parameters remain bit-exact across all eight distinct GPU ranks after scratch updates. This synthetic probe does not modify the foundation or constitute a formal association run; no tracking metrics are claimed. Keep the full-clip-in-one-DDP-forward contract for the eventual trainer. Evidence: `eval/ddp_eight_rank_core.json` and `eval/real_candidate_feature_contract.json`.

Confirmed GitHub publication of original source `0b6f1d4e20faed17bc105b67aef4505bcdc87ea2` at remote `5d79bfcc0b0b3f58594817dc727d5707bb1d6331` (verified branch and commit tree through the connected app). Corrected stale final-status fields that still called BDD missing and Dance unmeasured: all seven sources are available, Dance complete, BDD/TAO full audits remain pending. The existing blocking continuous controller remains live; no gates, seeds, candidates or active audit jobs were changed.

## U2 candidate-cache and ordinary-baseline execution plan — 2026-10-12

Read the real candidate bridge, frozen ordinary/TAO protocols, KITTI legal scope and TrackEval readers. The TempRMOT fork changes MOT file loading into expression-specific prediction paths; do not use that reader for ordinary MOT evaluation. Original KITTI tracking labels already exist under `/data2/user/reference_envs/kitti_tracking_labels_20261001/training/label_02`; no extra dataset download is required. Plan: pin an unmodified upstream TrackEval, freeze KITTI V2 legal fit/calibration/validation frame lists with original tracking IDs, implement a label-free video-sharded cache that requires the existing complete perception selection, and implement the IoU/motion/visual-cosine simple baseline with fixed parameters before any benchmark results. Keep supervision in separate source adapters, preserve all existing gates and running full F0 jobs, and test only the contracts needed for valid tracking/evaluation.

Pinned direct upstream TrackEval at `12c8791b303e0a0b50f753af204249e622d0281a`, verified unmodified, and recorded its six relevant readers/metrics. Frozen original KITTI V2 legal fit/calibration/validation covers 15 videos, 1,814/2,087/1,385 frames; never opened the four reserved video labels. Original KITTI and Refer tracking namespaces differ, so the earlier U1 generic result alone was incomplete evidence for ordinary tracking GT. Added the missing full original-GT portability audit over 3,472 dev images / 15,875 eligible car/pedestrian units with unchanged pinned TrackEval occlusion/truncation handling. Existing U1 references, all old domain scopes, .85/.90/.80 ordinary thresholds, .70/.60 OVMOT thresholds and the two-epoch cap remain intact. Actual F0 KITTI image smoke passed (session 37939 exit 0, native state keys all matched). Launched the full original-KITTI F0 audit on GPU 0: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/u2_mot_portability.py --protocol outputs/unified/u2/protocol/kitti_ordinary_portability.json --out outputs/unified/u2/perception/kitti_original_gt_f0_full --gpus 0`; parent 864092, worker 864093, blocking session 95238. The old full BDD/TAO jobs remain untouched. U2-P candidate evaluation now includes 76,068 ordinary images, plus all original 53,506 TAO/C-TAO and U1 scopes.

Implemented full label-free cache inventory and selected-checkpoint-gated 64-frame video shards. C-TAO dense and modal TAO train union contains 491,353 unique train images (including 1,143 modal images outside the C-TAO set), plus 36,375 TAO dev images. OVMOT storage assignment balances four volumes at 131,934/131,931/131,931/131,932 frames. Ordinary train/dev cache covers 396,865 images. Actual inventory verification (session 32148 exit 0) proves all 3,182 groups and all 924,593 physical images occur exactly once, with no train/dev image overlap and exact eight-worker group coverage. Cache generation remains conditional on recomputing the original complete selection; features are BF16, original boxes/scores FP32, semantic/decoder storage is shared, and supervision stays separate with no automatic novel labels.

Implemented the frozen simple baseline and official full-sequence metric/export path. Registered parameters before benchmark results: IoU/motion-IoU/id-cosine .25/.35/.40, margin .45, distance 2, birth .10, appearance EMA .90, max_age 30. All input candidates participate in matching, low scores can recover IDs, and no motion-only box is emitted. Synthetic causal/export/metric validation (session 40233 exit 0) passed all five upstream readers: perfect HOTA/DetA/AssA/IDF1 1 and IDSW 0, while swapped MOT17 identities give DetA 1, AssA .4, IDF1 2/3, IDSW 2. BDD inclusive extents/aliases/ignored attributes and KITTI DontCare/MOT distractors are honored. No benchmark metric is claimed. The optional unrelated BURST import warns about missing tabulate; all required ordinary readers and metrics execute successfully, so no unrelated package repair is needed. Keep the cache and simple pipeline.

Launched one blocking follow-on queue: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_ordinary_baseline.py --wait-perception-pid 854970 --kitti-audit-pid 864092 --extraction-pid 625610`, controller 864319 / session 35256. It waits for genuine completion of the original-KITTI audit and the existing bounded perception owner, then builds the selected ordinary cache and measures all five simple-baseline dev sets. If the existing owner genuinely terminates at preflight before any optimizer because the newly supplied KITTI audit was still incomplete, the queue resumes only after that audit completes and the full unchanged preflight is READY. No live job is restarted on an observation timeout. Formal cache and formal track training are still unlaunched; the research goal remains active.

## U2-A temporal supervision implementation plan — 2026-10-12

The complete original-KITTI F0 audit terminated successfully (session 95238 exit 0): 3,472 images, 15,875 source GT units, Top150/Top300 recall .928503937/.940535433. The original perception owner and ordinary follow-on queue are confirmed live. Read the U2 temporal curriculum/teacher forcing/loss specification and actual core: query drop/injection is currently diagnostic-only, and motion/lifecycle assume consecutive processed frames despite required stride 1–4. Implement real training-state augmentation from past unmatched candidates, current-frame Hungarian GT targets and scheduled model-in-loop state assignment, the six registered identity/motion/lifecycle losses with explicit false-birth suppression, and physical frame-gap-aware state propagation. Use the same shared core, keep GT bookkeeping outside its runtime state, retain original gates and do not launch formal association optimization before selected caches and simple metrics exist. Validate crossing/occlusion, negative proposals, mixed teacher/model rollouts, connected temporal gradients, causality and task isolation before the 6,000-step run.

Implemented actual query-state drop and previous-frame hard-negative injection; external first-identity bookkeeping does not hide model switches or duplicate births. Registered six losses include positive column CE plus NEW reject BCE so false proposals (including the first frame) are not blindly taught births. Current-frame positive/distractor Hungarian and separate crowd IoA/min-height rules preserve official ignore semantics. Physical source-frame gaps drive motion, lifecycle and memory decay; expired IDs are retired before current matching. Batched association data transfers, GRU/birth updates and switch margins avoid per-track GPU/CPU synchronization without changing core parameters. Crossing/occlusion, false-birth/ignore gradients, mixed teacher/model rollout and connected temporal gradients pass; all existing causal/isolation/save-load checks pass.

Actual eight-GPU BF16 registered-loss proof initially failed at the Hungarian NumPy conversion (`TypeError: unsupported ScalarType BFloat16`, session 51784 exit 1; reproduced single GPU session 68632). Smallest fix: cast detached logits to FP32 only at the CPU assignment boundary. Single-GPU regression session 30093 and eight-GPU session 2469 passed: global16 clips, finite/nonzero candidate projection/identity GRU/pair/set/motion/alive/birth gradients, identical post-update rank parameters. Subsequent batched-transition/expiry edits trigger fresh targeted and dual-task/eight-GPU checks before publication. The first original-GT oracle check (53170 exit1) compared MOT reader XYWH to core XYXY; verified native source/reader format and fixed the oracle only. Session 16135 passed every positive box on five complete actual train videos: BDD203/1126 units, Dance703/4912, MOT17 301/8701, MOT20 215/9060, KITTI154/234. Scope is these five videos, not all train labels or benchmark results.

Added frozen-cache causal clip sampling, 6,000-step eight-GPU MOT-only entry, registered curriculum/teacher probabilities/seed20261012/dataset probabilities, full all-five dev evaluations at1000/2500/4000/6000, 500-step resume snapshots with optimizer and per-rank CPU/CUDA RNG, unchanged identity-benefit/HOTA gates and four-level checkpoint selection. Synthetic sampler/resume/selection proof passes: next stochastic update bit-exact after restore, train-only rank/step sampling, >2-point individual HOTA regressions rejected. Preflight (55952 exit2) correctly holds without the complete selected cache and all-five simple metrics; zero formal track updates. Launched one blocking follow-on queue: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_mot_only.py --wait-ordinary-pid 864319`; actual PID870157/session38396 waits for original ordinary owner, then launches exactly 6,000 MOT-only steps when prerequisites pass, without OVMOT. Original full F0/perception owners remain live and untouched. Keep implementation; U2/U3 and all trained metrics remain unproven.

The fresh batched-transition eight-GPU command (76774 exit1) passed both MOT/OVMOT temporal tasks first, then exposed a BF16 vector-index assignment mismatch: GRU output BF16 into FP32 persistent memory. Added explicit differentiable casts to destination identity/semantic memory dtype; no loss or parameter change. Run the actual single-GPU BF16 backward/active-state inference regression, then the full eight-GPU registered-loss proof again before publishing. The failure is explicitly recorded; old PASS artifacts do not certify the new code until the fresh proof succeeds.

Final current-source targeted regressions completed: session21978 exit0 (six-loss crossing/expiry, five actual-video GT comparison, sampler/RNG/selection), session1677 exit0 (post-dtype-fix temporal/causal/save-load and resume/selection), session87550 exit0 (fresh actual eight-GPU BF16 registered losses, all necessary gradients nonzero, both updates identical across ranks). The preceding dual-task eight-GPU FP32 phase of76774 passed for MOT and OVMOT; only its BF16 phase failed and was repaired. Single-GPU session57241 exit0 proves batched BF16 backward and nonempty-state causal inference. Keep these final implementations and proofs. No formal feature cache, learned MOT baseline or joint benchmark is claimed; the research goal remains active.

During verified publication/follow-on waits, original perception owner11636 reported BDD audit completion. The original BDD blocking session36974 terminated exit0; strict full measurement validation confirms all39,973 images,442,814 GT, coverage/protocol/F0 SHA. Hits363,947 Top150 (.8218958750175017),377,519 Top300 (.8525453124788286). BDD fails .85 minimum but exceeds .80 hard stop; no BDD K300 exception is allowed. All five ordinary F0 audits are now measured; bounded U2-P still waits for original full TAO/C-TAO audit. Updated canonical portability/status/reports and retain the same registered 2-epoch repair and six-thousand-step downstream gates. No formal track optimizer launched; keep result and publish this new full measurement.

## U2-B supervision/semantic/evaluation implementation plan — 2026-10-12

Previous goal turn made concrete progress: complete original KITTI/BDD F0 measurements, actual temporal supervision plus gated 6k MOT-only implementation, current eight-GPU BF16/causal/GT/resume proofs, source commits/backups and verified GitHub publications. Current source30c7a56 is clean; original bounded perception owner11636 and MOT follow-on38396 are confirmed live. Read OVMOT adapter, shared language bridge and U2 sections66–82. The OVMOT report incorrectly says TAO/C-TAO absent; all assets are present. Raw C-TAO interpolation resets frame_index (e.g. original frame0176=175, interpolated frame0177=1), and C-TAO/TAO numeric IDs occupy different source namespaces; physical filename chronology and original annotation UUID alignment are required. Plan: audit all dense/base supervision with no automatic rare/unmapped class or identity labels; implement shared-U1 BERT/projected full-vocabulary class features, semantic classification/consistency/gate losses and same-core OVMOT temporal rollout; pin official upstream TETA and implement complete original validation export/overall/base/novel reporting with explicit category mapping counts. Keep all ordinary loss weights and gates. Formal joint optimization requires selected MOT-only eligibility and is not authorized by contract fixtures or incomplete measurements; do not launch it now.

Full compiler failures were diagnosed sequentially without dropping units: camera timestamps require original index anchors (66320 exit1), LaSOT uses numeric filenames (second command terminal exit1), dense ArgoVerse frame_id compresses missing-image gaps (22205 exit1; 17/37 camera videos have nonconstant anchor offsets), and official dense annotations include nine zero-area boxes (97023 exit1). Retain source geometry/counts, use numbered filenames or timestamp interpolation between original TAO anchors, and leave zero-area GT at zero IoU. Final compiler29355 exit0 covers all500 train videos/491,353 images/1,489,990 unique base units; 52,911 base units deduplicate across sources, 353 modal-only base units remain. Novel/unmapped modal1,156/219 units never provide class/identity labels. The initial assumption of differing overlapping numeric IDs was not observed: all shared UUIDs have equal numeric IDs. Keep UUID alignment with original IDs retained separately. No validation/test labels enter training.

Actual F0 shared text extraction63671 exit0 yields1203x256 features,866 base/337 novel,38 native BERT chunks/max104 tokens/backbone0; F0 remains reference-only, not selected for formal training. Implemented existing-parameter semantic classification, final pair supervision, persistent-memory classification, consistency and gate regularization, with unknown candidate/row/contrast labels masked and certified negatives retained. Single80792 exit0 and actual eight-GPU7759 exit0 prove BF16 connected identity/semantic gradients, unknown birth-label exclusion, ordinary text isolation, full novel inference vocabulary and identical post-update ranks. This is one scratch update on synthetic track candidates with actual text features, not joint training. Ordinary temporal regression20603 and32602 exit0 passes unchanged six-loss values. Actual AVA/ArgoVerse/LaSOT full-video loaders67140 exit0 cover976/451/1169 frames and4074/1372/1169 units, all base-only/strictly increasing time. Deterministic joint task draws give6518 MOT/3482 OVMOT in10,000 steps; registered probabilities remain.65/.35.

Pinned unchanged official TETA b498aa87 and prepared complete LVIS-v1 TAO validation10545 exit0: all36,375 images/988 videos;112,798/113,112 original GT units and5473/5485 tracks map to v1;314 units/12 tracks explicitly unmapped by the official OVTrack synset conversion. Original v0.5 full-localization scope remains unchanged. TETA fixture22487 failed because OUTPUT_SUMMARY=False leaves upstream returned threshold uninitialized; use the official default summary=True with no source/metric modification. Fixture67046 exposed our bit-packed export IDs allocating huge dense upstream ID maps; stopped that owned synthetic command (exit143), removed the packing, and let official TAO resolve video-local IDs. Final61184 exit0 passes perfect100, wrong-class localization/association separation, identity-switch sensitivity and non-exhaustive unlabeled-object neutrality, with empty frames retained. No benchmark TETA/ordinary tracking score claimed, no formal joint optimizer or U3 launch. Keep all changes; original perception and downstream queues continue under unchanged gates.

Fresh ordinary eight-GPU BF16 regression71729 exit0 after the OVMOT changes again confirms all six registered loss gradients and identical rank parameters. Existing gate recipes, candidate caps, seeds, metrics and training scopes stay unchanged.

Current-source causal/state save-reload regression58539 exit0 passes on nonempty states, including ordinary text invariance and OVMOT raw identity/GRU isolation. No full candidate cache, simple tracking metric, learned MOT checkpoint or joint benchmark yet; continue the active U2 goal after publishing this completed source/supervision/semantic/TETA stage.

## U2-B gated joint execution plan — 2026-10-12

Previous goal turn made progress: full base-only dense supervision, actual shared F0 class features, same-core semantic/partial-label losses, official TETA validation conversion/export and actual eight-GPU proofs were committed3d2a9ec and published/verified as aff3b00. Current project identity/worktree are correct/clean; perception854970, ordinary864319 and MOT-only870157 are confirmed live. Read current baseline trainer/evaluator, both cache phases and U2 sections64–85. Implement exact4,000-step joint entry initialized with every selected MOT-only core tensor; global task.65/.35, all8 ranks/two8-frame clips, frozen foundation/BERT and separate5e-5/1e-4 core/adapter groups. At each1,000-step boundary evaluate every ordinary dev sequence plus all TAO validation through official TETA; reject any individual HOTA or AssA loss>.01 and enforce TETA34/AssocA37. Add exact RNG/optimizer resume and one blocking downstream owner that waits for real MOT-only completion/eligibility before OV feature-cache/class-text construction and joint launch. Actual semantic fixtures do not authorize a formal run. The full rank-zero metrics can exceed default NCCL collective timeout; extend only the collective wait timeout, with no change to data/metrics/optimizer/evaluation boundaries. Validate gate rejection, checkpoint initialization/resume and alternating-task eight-GPU execution before queuing.

Implemented the complete4,000-step joint trainer/evaluator and gated downstream owner. Registration/preflight55888 and29449 correctly return exit2 HELD with zero formal updates while selected caches/MOT-only are absent. Joint boundary85960 exit1 exposed .59-.60 floating comparison falsely rejecting the inclusive1pt threshold; MOT boundary4594 exit1 also exposed .4+.03 rejecting exact3pt gain. Compare the recorded decimal scores at the same literal.01/.02/.03 thresholds, with no epsilon/extra rounding tolerance or gate weakening. Final joint62479 and ordinary22178 exit0 pass inclusive boundaries, individual HOTA/AssA rejection, both OVMOT gates, refinement conditions and four-checkpoint selection. Original ordinary14478 regression also passed before this targeted numeric fix.

Actual eight-GPU BF16 alternating-task/init/optimizer proof14038 exit0 passed; strengthened1629 exit0 additionally restores the real scheduled teacher Bernoulli RNG for a bit-exact next update after actual joint checkpoint save/load. Both use all256-D core tensors,16 global clips/eight frames, actual reference-only F0 class text and synthetic tracking candidates; no formal optimizer exists. First whole eight-rank joint-evaluation fixture84831 exit1 failed at BDD predicted category person: the fixture incorrectly used the raw GT alias as detector vocabulary. Correct only the fixture to the existing registered pedestrian vocabulary; keep the actual BDD normalization, official reader, metrics and original scope unchanged. Run the whole entry again before queuing/publication.

Final whole joint-evaluation proof99228 exit0 executes the actual eight-rank entry, all five official ordinary readers, both base/novel TAO fixture videos and official TETA with8 CPU workers. Simulated interruption after completed video units resumes with zero core forward calls and identical full metrics. These scores are explicitly synthetic and are not benchmark evidence; no real tracking/foundation optimization or U3 readiness is claimed. Keep exact whole-entry implementation and all gates. The current source stage now has a single blocking joint downstream owner waiting for genuine completion of MOT-only870157; only a recomputed eligible selected reference permits full balanced OVMOT feature caching, selected actual class text, and the4k joint run. No live source jobs were restarted.

Launched that single blocking owner: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_joint.py --wait-mot-only-pid 870157`, actualPID883611/session32672. State confirmedWAITING_ELIGIBLE_MOT_ONLY; fullOVcache_started=False and formal_joint_training_launched=False. Keep all new implementations/proofs and publish the completed joint execution stage; the full U2 goal remains active with actual perception/cache/simple/learned/joint measurements unproven.

## U2 self-contained native checkpoint implementation plan — 2026-10-12

Previous goal turn made progress: scientific355c152 and exact eight-GPU joint execution/resume/evaluation proofs were committed, backed up and verified at GitHub c16cd98. Project root is correct and original perception/cache/baseline/joint owners are confirmed live. Read actual native runtime, U2 core and specification sections83–86. Final format must embed actual MM-GroundingDINO/shared BERT, tokenizer/config, every persistent core/head and task embeddings; the shape-only fallback is insufficient. Current core has no task embedding tensor. Add zero-initialized task embeddings only to the bounded semantic gate input; ordinary MOT gate remains exactly zero and raw identity/GRU remains task isolated. No formal track optimization has begun, so register this missing required component before selected caches/checkpoints. Implement native model reconstruction from embedded BERT architecture/tokenizer and strict embedded weights, with no external checkpoint/HF pretrained load. Validate actual F0 candidate/class-text bit-exact reload, causal state resume and same-core parameter sharing with a clearly reference-only random core pack; never treat it as a selected final checkpoint or U3 readiness. Actual final packing must recompute joint and perception eligibility and retain all original metadata/scopes.

Native reference-pack32496 exit1 exposed bbox-head constructor-derived fields saved after native configuration mutation;64857 exit1 then exposed the analogous denoising-query fields. Remove only those six constructor-derived entries before serialization; native construction recreates them.41966 exit1 passed strict complete foundation/core tensor reload but failed shared-BERT feature equality. Targeted31560 exit0 proves native from_pretrained uses SDPA while bare embedded BERT construction uses eager: max feature discrepancy1.144409e-5; current native output remains bit-exact with the immutable F0 text cache. Store and restore the actual native attention implementation rather than weaken comparison tolerance. Task-embedding joint eight-GPU60299 exit0 retains bit-exact next stochastic optimizer update, nonzero embedding gradient and ordinary adapter gradNone.35642 exit0 passes causal/temporal and CPU semantic regressions; strengthened38946 exit0 explicitly perturbs task embeddings and proves ordinary MOT unchanged/raw OVMOT identity isolation plus unchanged eligibility/initialization checks. No formal optimizer, reference, live owner, seed or gate was changed.

Attention attempt51064 exit1 still used eager because HF BertConfig.from_dict mutates the supplied dictionary and overwrites attn_implementation from kwargs;15925 verifies the actual packed class remained BertSelfAttention vs native BertSdpaSelfAttention. Pass a copied config plus explicit attention kwarg. Final native19361 exit0 proves all actual F0 foundation/core tensors,1203 class features, native MOT150/OVMOT300 candidate features and nonempty online state reload bit-exact, with external tensor/config/pretrained-HF calls prohibited. V1/V2 small deterministic generic/expression/true/shuffled integration outputs and metrics are bit-exact; this is a reference-only random core pack, not a selected final artifact or tracking benchmark. Task embeddings now use supplied zero weights without consuming RNG, preserving all subsequent shared-core initialization draws at the original seed. Final joint52310, causal/temporal/gate8930 and whole eight-rank official evaluation33800 exit0 pass. Targeted32182 recomputes actual parameter sharing after excluding shared detection/association/motion/lifecycle heads:236,751,992 shared non-head vs329,475 conservatively task-specific, fraction.9986102878>=.90. Only count metadata changed in the reference file; actual restored foundation/core tensors remain bit-exact. Formal packer recomputes original perception/MOT/joint eligibility, embeds original source/protocol/sampler/step metadata and remains pending actual selected-runtime retention. Keep all changes. No selected cache, real tracking benchmark, formal optimizer or U3 readiness yet.

Current CPU semantic18032 exit0 confirms task-embedding gradients and unknown-row/candidate masks after preserving initialization RNG. Store CPU evidence separately as ovmot_training_contract_cpu.json, retaining the historical GPU single-frame report. Updated eight-GPU joint evidence covers BF16 gradients and exact restore with current embeddings.

## U2 frozen text reuse for upcoming candidate caches — 2026-10-12

Previous goal turn made progress: actual native one-file F0/core reconstruction, task-embedding isolation and real parameter-sharing/retention proofs were committed2b775b7, backed up and verified at GitHub9c12586. Project root/clean worktree verified; original F0/perception and downstream owners are live and ordinary cache is still unlaunched. Read native GroundingDINO.predict: every full LVIS frame re-encodes the same38 BERT prompts although the language backbone is frozen; native prompt/token-positive-map preprocessing is also repeated. Implement worker-local fixed-text memoization by wrapping native language forward and prompt construction, preserving all native image/backbone/encoder/decoder/NMS operations and cloning returned tensors/maps before mutable native consumers. Apply only to upcoming full selected-foundation cache workers, without changing any running full audit or perception-training process. Measure actual FP32 F0 two-frame MOT and full1203-class OVMOT candidate/selected-feature equality, mutation isolation and actual BERT-call counts/timing; no approximate precision, cap, chunk, vocabulary, seed or scope change. Keep only if bit-exact outputs hold.

Initial frozen-text38589 exit0 proves actual FP32 F0 native/cached MOT150 and complete1203-class OVMOT300 two-frame candidate/feature parity, tensor/map mutation isolation and exact method restoration. Warm OVMOT elapsed5.208873→4.619842s (1.1275x observed throughput), with two-frame BERT76→38 calls and unchanged2 backbones/76 decoder calls. Implement only in upcoming selected-foundation cache workers; record source/proof provenance in spec and actual counters in all worker/full reports. Integrated80238 exit0 reconfirms full parity and executes the actual worker/serializer on explicitly separate two-frame reference inventories for both phases: serialized FP32/BF16 fields bit-exact, semantic/decoder storage shared, zero GT labels, completed-unit resume with no inference and identical unit manifests. Final warm OVMOT5.255846→4.638013s (1.1332x), MOT.188991→.180778s (1.0454x); shared-GPU two-frame observations, not full-cache speed estimates. Keep optimization; all actual source frames, chunk/candidate limits, precision, eight-rank assignment, four-volume distribution, original gates and running full jobs are unchanged. No formal selected cache, track optimizer, benchmark tracking score or U3 readiness exists yet.

## U2 selected-artifact finalization execution plan — 2026-10-12

Previous goal turn made progress: bit-exact real F0 fixed-text reuse and actual cache-worker serialization/resume were committed1ad7563, backed up and verified at GitHub4306013. Current project/worktree are correct/clean. Original joint blocking handle32672 is confirmed live and waits eligible MOT-only; no formal OV cache/joint has begun. Read U2 sections83–89/98, actual packer, small retention audit and joint owner. Implement a separate single blocking finalization owner waiting original883611 completion, then recompute full perception/MOT/joint eligibility, package the actual selected U2-B foundation/core, audit actual packed retention and unmodified selected-core causal/isolation/save-load behavior, and generate final readiness solely from current full artifacts/provenance. Reference-only F0 packs and scratch proofs cannot fulfill selected-artifact gates. Reject or retain exact held/blocked status when earlier gates fail; conditional U2-C remains a separate bounded task if eligible, never substitute a U2-B pack. Preserve all running owners and budgets. Use deterministic real candidate probes for actual learned heads; never modify selected weights to make runtime contract checks pass.

Existing historical two-GPU smoke had only one backward, while the requested two-step contract was missing. Updated the existing joint proof entry to accept expected world size2 or8 without changing its default eight-rank behavior; actual CUDA0/7 two-rank26053 exit0 now covers MOT/OVMOT updates and exact stochastic next-update checkpoint/RNG/optimizer restore. Fresh single-GPU8844 exit0 includes nonzero current task-embedding gradient and unchanged BF16 identity/semantic loss values; existing current-eight-GPU evidence stays intact. Reference selected-core-audit entry99248 exit1 exposed native candidate metadata strings being cloned as tensors; clone only tensor fields and retain metadata unchanged, then rerun targeted entry. This is an audit-interface repair, not a selected-weight, seed, gate or training-protocol change.

Selected-core entry48760 exit0 now passes four actual native F0 MOT frames and unchanged reference-core weights:150 births/repeat matches, one-frame occlusion/recovery, future-state independence, arbitrary MOT text invariance, raw OVMOT/fixed-pair GRU isolation and bit-exact state resume for both tasks on one core object. Readiness69971 exit0 tests the independent complete requirement set/material failure cases and the actual CLI returns2 HELD with missing full cache/simple measurements. Actual selected-audit CLI75746 exit0 proof rejects a reference-only random pack before cache read. Review found old small generic retention used only car rather than U1 original10-entity vocabulary; use original VOCABULARY in final audit and require that exact vocabulary in readiness. Full-vocabulary native/reference-pack96632 exit0 is bit-exact on both V1/V2; all original full U1 references are unchanged, and historical reference metadata now names the restricted car-only scope. Missing actual selected audits never become readiness evidence. Keep all repairs/entries.

Launched one blocking finalization owner with OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_finalize.py --wait-joint-pid 883611. ActualPID896659/session18355 confirms WAITING_ORIGINAL_JOINT_OWNER, pack_started=False, ready_for_u3=False. It waits real original completion, recomputes current full selection, packs actual frozen U2-B tensors, runs original-vocabulary retention and unmodified selected-core audits, then verifies every final gate against bound artifacts. Completed units resume by actual checkpoint receipt; reference fixtures and conditional U2-C cannot substitute for selected U2-B. No original owner was restarted, no additional formal optimizer launched and no U3 readiness claimed.

## U2 native image-batch admissibility probe — 2026-10-12

Previous goal turn made progress: selected-artifact finalization and bound real-audit/readiness entries were committed4102740, backed up and verified at GitHub592d097. Current worktree/project are correct/clean; original full TAO parent63615 is confirmed live through its existing blocking handle, and finalization waits original883611. No actual formal baseline/cache is ready. Read native inference_detector and fusion attention before considering future throughput: list-image API still loops singleton test_step, whereas the multimodal fusion uses a scalar max over the full batch. Probe actual FP32 native batch2 versus singleton on the two fixed original MOT17 frames, plus a second-image perturbation while keeping the first fixed. This is a small admissibility measurement, not a new formal audit or source/model change. Reject batching if candidate parity or first-frame independence fails; keep running jobs, candidate policy and original native singleton protocol unchanged.

Native image-batch probe10904 exit0 yields a genuine negative optimization result: same native preprocessing shape on two fixed MOT17 frames, but batch2 boxes/scores differ from singleton on both. Max same-rank score differences.000565775/.002606869; large coordinate differences1033.97/1865.55 reflect changed sorted proposal alignment and are not a matched-box localization error measurement. All labels remain equal. Actual six fusion stable_softmax_2d flags are False; the suspected scalar max is inactive, and changing only second-image pixels leaves the first batch output bit-exact in this probe. Reject native batch2 on candidate parity; no full OVMOT batch or semantic-feature integration is attempted, no TF32/precision tolerance or candidate policy is changed. Keep native singleton plus already proven fixed-text reuse. Original full TAO/perception/baseline/finalization jobs remain under their original blocking owners and gates; no tracking benchmark or U3 readiness is claimed.

## U2 upcoming perception-audit fixed-text reuse — 2026-10-12

Previous goal turn made progress: actual FP32 native image-batch parity failure was measured and committed24be26a/published64cb0ff; retain native singleton. Current project/worktree are correct/clean, and existing full TAO63615 blocking handle is confirmed live. Read current full TAO worker and bounded adapted-checkpoint evaluation caller. The already bit-exact fixed native text reuse can reduce repeated38-caption BERT work in upcoming adapted TAO/C-TAO audits too. Add an explicit optional CLI flag with unchanged default F0 behavior; only future adapted-checkpoint evaluator requests it. Keep identical singleton/native NMS, protocols, GT denominators, checkpoint/source fingerprints, resume rows and gates. Validate actual unchanged two-source smoke rows/recall counts with and without the flag before retaining this integration. Do not modify or restart the current loaded F0 workers.

Actual paired portability-worker37402 exit0 confirms optional text reuse: original two-source smoke metadata/rows identical,300 proposals each, TAO2base/1novel GT and C-TAO10base GT hit counts unchanged, one backbone/image, BERT38calls/38hits across two complete1203-class passes. Native/reuse elapsed11.545474/10.907659s are limited two-image cold/warm observations, not full throughput. Flag defaultFalse preserves F0 metadata/commands and resumable exact frame prefixes; only future adapted-checkpoint evaluation explicitly passes it to new workers. Keep this verified integration; all original full processes, protocols, GT denominators, weights, budgets, seeds and acceptance criteria remain unchanged. Actual adapted full audit and all tracking results remain unmeasured.

## U2 full F0 completion and native checkpoint load repair — 2026-10-12

Original blocking TAO audit63615 returned exit0/COMPLETE: all36,375 TAO validation and17,131 C-TAO original training frames measured. TAO base105,408/110,055=.957775658, novel2,777/3,057=.908406935; C-TAO base50,975/53,263=.957043351, novel inapplicable. Both OV portability streams pass; BDD/MOT17/MOT20 ordinary F0 gates still fail, so bounded U2-P is required. Its original owner11636 genuinely terminated at native MMEngine checkpoint loading before any optimization, because PyTorch's default weights_only=True rejects the original checkpoint's HistoryBuffer. Plan: retain original full measurements and exact eight-GPU replay/protocol; apply the existing U1 process-local TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD convention, verify the actual native loader on immutable U1R2, then resume the unfinished epoch and terminal downstream queues with updated owner PIDs. No metric, seed, budget or acceptance change.

Targeted regression94105 exit0: all six terminal shards exactly cover53,506 unique registered frames, every frame has one backbone call and300 native proposals, both source GT denominators and measurement totals match. The native MMEngine loader now loads the immutable SHA-verified actual U1R2 state_dict1,069 tensors plus HistoryBuffer/optimizer; full unchanged preflight is READY. No epoch checkpoint or first optimization update existed at failure. All four original downstream handles returned terminal HELD states; recover their owners only after this evidence. Keep the localized compatibility fix.

Recovery blocking supervisor23776 launched genuine new owners after all old handles terminated: perception1054069, ordinary1054070, MOT-only1054071, joint1054072, finalization1054074. Original failed states preserved in outputs/unified/u2/recovery_20261012/*.original; original training log retained. The single supervisor waits these exact processes; downstream unchanged gates prevent cache or tracking launch before a fully selected passing foundation. Full budget remains1,250 steps/epoch, at most2epochs, followed by6,000 MOT-only and eligible4,000 joint steps. Actual selected pack and U3 readiness remain unproven.

## U2-P epoch1 complete; adapted audit worker argument repair —2026-10-12

Recovery23776 returned genuine terminal exit1 after epoch1 completed all1,250 updates and saved epoch_1.pth (SHA1d9077ae603f40de0e955a11a88083f914f98faa967bc46cdd06bcdfb5129688). Actual checkpoint contract PASS compares265 immutable BERT/Swin0/1 tensors bit-exactly with original U1R2. First actionable evaluation failure: all ordinary workers exit2 before inference because the parent omits --measurements-dir, so their adapted-checkpoint parser sees the default F0 directory. The TAO parent has the same omission and would fail at its next stage. Plan: propagate the already explicitly supplied independent directory in both existing worker commands; verify actual adapted native smoke workers through both parent launchers, preserving guards and scopes; resume unfinished full evaluation from the completed epoch1 checkpoint, never rerun epoch1 optimization. All recovery downstream owners are terminal HELD, and will be restored behind the new perception owner.

Storage rebalance requested by user —2026-10-12: move only11 completed/CRC-extracted TAO archives from /data0/user (87.981GiB,49.54% of prior usage) and /data1/user (85.328GiB,48.72%) to matching /data3/user/TAO-Amodal/frames paths. Destination has789GiB free; extracted train/val/test image paths and scientific protocols remain unchanged. Two parallel volume transfers verify each destination against its existing download SHA before atomically replacing the original archive with a symlink; no source data is removed before verified copying. Plan recorded in outputs/unified/u2/storage_rebalance_20261012/plan.json.

User superseded archive relocation: original transfer69601 stopped with exit143; deleted all21 CRC-extracted completed TAO archives across allfour volumes (345.674GiB) plus cancelled relocation temporary copies (68.760GiB). New plan migrates approximately half of cleaned /data0/user and /data1/user image storage, using only test/validation directories; train directories remain in place. rsync validates transferred data, then a dry-run checks complete matching trees; Linux renameat2 exchange atomically replaces each original directory with a target symlink before removing its verified redundant original copy. Scientific image paths, original source annotations and all protocols stay valid. New explicit plan: outputs/unified/u2/storage_rebalance_20261012/image_migration_plan.json.

Post-cleanup image migration completed:95 directory units, /data0/user44.814GiB and /data1/user44.907GiB relocated to /data3/user, each49.999% of cleaned initial occupancy. rsync full-tree parity passed before atomic symlink exchange; all original image paths remain resolvable, only test/validation storage moved, train paths and experimental contracts unchanged. Final actual directory sizes and df recorded in outputs/unified/u2/storage_rebalance_20261012/image_migration_result.json.

Adapted launcher regression96599 exit0: through the actual ordinary and TAO parent launchers, each actual child receives the separate measurements directory and real epoch1 checkpoint SHA; one original legal KITTI frame and original two-source TAO/C-TAO smoke frames execute native inference successfully. Test-only smoke injection and collector do not produce formal measurements or weaken aggregate completeness; report is explicitly reference_only. Full recovery supervisor95952 resumes epoch1 evaluation from existing epoch_1.pth, new perception1089957/ordinary1089958/MOT-only1089959/joint1089960/finalization1089961 owners retain unchanged budgets and gates. All previous owners returned terminal, their failure states retained. Full U1 retention and adapted portability remain pending; no tracking or U3 claim.

User-requested inference sharding throughput plan —2026-10-12: full audits already use8 physical GPU shards, while native full1203-class TAO performs38 multimodal transformer passes per image. User asks to shorten runtime. Temporarily SIGSTOP exact evaluator1090197 and its existing TAO parent1238269/workers1238270..1238277, retaining all completed ordinary measurements and existing TAO exact frame prefixes; no original source inference re-started. Measure1/2/4 independent singleton processes on GPU1, real adapted epoch1 SHA1d9077a, identical12 reference frames spanning original TAO source domains, full vocabulary/native FP32/static text reuse and exact boxes/scores/classes parity. This is a bounded throughput probe, not a metric or scope change. Adopt higher concurrency only if actual throughput improves with unchanged outputs; otherwise retain the fastest measured concurrency and resume the existing owner. Probe shell60967; hold receipt outputs/unified/u2/perception_adaptation/throughput_probe_hold.json.

Same-GPU probe60967 completes: identical12 actual native full-vocabulary reference frames take158.996s with1 process,251.286s with2 and220.054s with4; all proposal boxes/scores/classes are bit-exact. Reject extra processes: single worker per GPU is fastest on this sample. Initial default-device memory counters were on GPU0 instead of model GPU1; omit those zero counters from published aggregate, retaining actual wall-clock/parity evidence.

Native metadata optimization85924 completes: explicit target CUDA binding plus one bulk integer-label copy for native label_names construction reduces the same12-frame sample to103.870s (observed1.531x), boxes/scores/classes bit-exact for all12, native full label_names identical on the first full-vocabulary frame; native methods restored. Actual model GPU1 peak allocation1,664,202,752bytes. This comparison includes CUDA binding and changed background load (seven formal audit workers resumed), so it is not an isolated causal label-copy benchmark or a full-run speed guarantee. Keep integer-only optimization provisionally; verify actual two-source native worker rows before resuming formal prefixes with it. Vocabulary1203, chunks32/38, NMS.8, Top300, precision/weights/protocol/GT denominators remain unchanged.

Actual bulk native worker63758 exit0: unchanged two-source smoke rows/GT hit counts and resume metadata are exact,300 candidates per frame, one backbone per frame,76 head calls/68,400 integer class labels; native/bulk CLI wall68.842/51.966s includes initialization and two frames, not formal throughput. After verified controlled interruption,95952 returned terminal exit1 (expected interrupted TAO command) and all four gated followers returned HELD without tracking launch. Every existing exact TAO prefix parsed successfully:440 completed frames preserved across initialized shards; shard1 had not yet written metadata or rows. All76,068 ordinary frames and epoch1 original1,250 updates retained. New supervisor57530 launches perception1313649/ordinary1313650/MOT-only1313651/joint1313652/finalization1313653. Full evaluation skips complete ordinary measurements and resumes unchanged eight TAO partitions with explicit GPU binding plus bulk integer metadata; labels alone temporarily use CPU for native name construction and return to the original device before NMS. Both flags preserve full1,203 vocabulary,38 native chunks, FP32, backbone counts, GT scope and all gates.

Actual epoch1 ordinary summary: BDD Top150.936574, Dance.999036, MOT17.878066, MOT20 Top150.819136/Top300.973281, original legal KITTI.946394. All five full ordinary streams are measured; MOT20 satisfies the registered Top300 crowded threshold but formal registration/selection still waits complete TAO and U1 retention. No selected foundation, candidate cache, tracking benchmark or U3 readiness is claimed.

### 2026-10-13 — U2 perception epoch 1 selected

- Completed the unchanged full TAO/CTAO portability audit: 36,375 / 17,131 images, Top300 base recall 0.9742855845 / 0.9729080225; TAO novel recall 0.9460255152 (2,892 / 3,057). Unmapped TAO GT retained separately (305 / 314 hits).
- Epoch 1 foundation `1d9077ae603f40de0e955a11a88083f914f98faa967bc46cdd06bcdfb5129688` passed all seven portability streams and U1 retention: generic V1/V2 0.9918663023 / 0.9672269380; expression Top100 V1/V2 1.0 / 0.9785374828. Fixed 500-pair language audit true/shuffled AP50 0.7301285093 / 0.4067468580. Generic scope 4,201 images; expression scope 7,160 representative query frames, 7,199 target references.
- Selection source receipts and their registered digests agree; retain epoch 1, no second adaptation epoch required. Existing recovery owner exited successfully and advanced the eight-GPU ordinary candidate cache automatically. Formal tracking scores and U3 readiness remain pending.
- Evidence: `outputs/unified/u2/perception_adaptation/eval_epoch_1/selection.json`, seven `measurements/*.json`, and generic/expression/counterfactual `result.json`.

### 2026-10-13 — user-directed GroundJEV research pivot

- User terminated full U2 candidate caching and authorized GroundAnything DLM + pretrained OneJev tracking research on at most four GPUs. Goal changed; this is not algorithm failure/NO_GO.
- Verified /proc ancestry/start ticks and stopped 13 old U2 processes with SIGTERM; no old queue/cache worker remains. Other project GPU jobs preserved. Removed 4,748 verified feature shards (39.89 GiB allocated), retaining manifests/data/results and loadable epoch-1 checkpoint SHA 1d9077ae… (1,909,880,505 bytes).
- New worktree /data2/user/LocateMOT_GroundJEV and branch codex/groundjev-4gpu-20261013 inherit freshly fetched remote 51614cb. Official GroundAnything/OneJev code cloned directly without proxy; isolated environments/weights setup started. No new training or tracking metric claimed.
- Correct inherited status terminology: generic .9787925789 is Top150 recall IoU .5, not AP; expression Top100 V1/V2 recall 1/.9785374828, while .7237566558/.6556441180 are query AP diagnostics. Raw receipts unchanged.
- Next: real DLM/OneJev loading, own-state short clip → full legal MOT video, fixed-candidate Simple/Full comparison, tracking-SFT and shared RMOT/OVMOT evaluation. Keep evidence-based failure repair and finite training budgets.

### 2026-10-13 — GroundJEV legal inputs and initial online interfaces

- Verified real legal MOT17-02 dev299, DanceTrack0004 full-val1203, MOT20-01 dev214 frames. No new data download or test-GT access needed for first video work.
- Implemented official-grid DLM HTTP candidates with missing native confidence explicitly preserved, pretrained OneJev batched Choice/Noul/Score identity and lifecycle adapter, Hungarian global competition, reliable long-memory updates and causal save/restore. Real crop evidence is explicitly distinct from pending native GroundAnything features. Replay control preserves identical candidates and reports no end-to-end FPS for cached perception.
- A two-frame real annotated CPU fixture passes identity persistence, exact resumed state and rejected past timestamps; GT is used only in this explicit fixture, not as a claimed runtime/benchmark result. Real model integration remains pending dependencies and weights.
- Installed independent Python3.12.15. Separate OneJev/SGLang installs active. PyYAML shared uv lock failure fixed by dedicated setup cache. Direct model transport hit aria2 TLS and curl HTTP2 cancellation; retained byte prefixes. Eight concurrent official1MiB ranges complete in7.397s (1.134MB/s aggregate); adopt sixteen8MiB ranges/large model on HTTP1.1 with official final weight digests. No proxy, no GPU usage yet.

### 2026-10-13 — official video evaluation and runtime setup progress

- Added complete-scope ordinary MOT export using unchanged TrackEval12c8791; all299 legal MOT17-02 dev frames in a labeled GT oracle fixture yield expected perfect metrics/zero IDSW/FP/FN. This validates format/scope handling, not GroundJEV performance. Runtime replay now checks actual physical image paths.
- To avoid repeated multi-GB dependency transfers, copied matching CPython3.10 installed framework files into an independent OneJev environment, excluding old Transformers/MMDetection and source console scripts. First copy attempt used the assumed venv package directory and copied zero files; corrected to actual distribution roots, then recorded nonzero distributions. Source environment untouched, no shared symlinks. Torch2.10+cu128/vision0.25 imports and CUDA availability on GPU6 pass; OneJev/Transformers5.5.4 installed and all69 package requirements compatible.
- Prefetched official pinned large PyPI wheels via verified byte ranges. Exact-version py3-none NVIDIA dependencies reused for GroundAnything's separate Python3.12/Torch2.9.1 setup. Controlled package transport interruptions preserve caches/bootstrap/custom vendor source; remaining official installation/ABI check continues. Universal flashinfer-cubin wheel was initially excluded by the platform selector; corrected from actual PyPI filename and completed the missing wheel.
- One blocking readiness queue owns the actual pretrained MOT-frame check, gated on full official OneJev weight receipt. Models still downloading; no real-model tracking or training result claimed. Optimizer steps zero, at most four physical GPUs enforced by common launcher; other projects preserved.

### 2026-10-14 — first actual OneJev frame and checkpoint-version repair

- Official OneJev weights complete and LFS digest matches: model.safetensors2,214,590,296bytes SHA cb4b7456703baa26a0cfacf177067200b50021df5a6ba2a2cd1e6f6c868f2244. No weight/model substitution.
- FirstGPU load reads474 weight entries but default official fork self-check fails (.0857 rounded probability difference > .05). Initial OneJev5.5.4 pin incorrectly reused GroundAnything's requirement; actual checkpoint config declaresTransformers5.17.0. Update only isolated OneJev to5.17.0/tokenizers.23.3, keep all official checks and .05 tolerance; same pretrained weights then pass auto validation and use batched mode.
- Real MOT17-02 frame000302 processed on GPU6 with Choice/Noul/Score: NEW.36689/IGNORE.63311, visible people.978877, quality expected level.919935. Preserve the negative generic choice; this is a load/interface check, not association accuracy. Actual three-question request.286737s, allocator peak3,119,928,832bytes, steps0.
- GroundAnything's independent Python3.12/Torch2.9.1/Transformers5.5.4 custom serve environment completes and official ABI check reports only its expected cuDNN9.16 metadata override, no unexpected conflicts. Single DLM owner waits for verified bundle and available GPU6 memory before starting official denoising with static memory fraction.4. GroundAnything download remains active. No full online video or fine-tuning claimed yet.

## 2026-10-14 — official DLM service and GPU ownership

GroundAnything DLM download completed at the pinned official revision with
its 9,687,419,736-byte weight matching the official LFS SHA256. The single
service owner launched official custom SGLang denoising on physical GPU6
(static memory fraction .4); the model endpoint responds. A single blocking
clip owner starts four real MOT17 dev frames using native DLM candidates,
pretrained OneJev, global assignment and its own causal identity memory.
Completion, output validity and tracking quality are still pending.

The user asked why physical4/5/6/7 still showed processes. A one-time GPU
process/cwd/argv audit identified UDMT training/inference on4/5/7, the two
GroundJEV model processes on6, and an unrelated user process on6. This is
not four-GPU GroundJEV usage and not old U2 caching. Other project processes
were preserved. Keep the explicit project device allowlist and ownership
attribution; no metric, seed or protocol changed.

## 2026-10-14 — tracking SFT teacher curriculum hypothesis

Hypothesis: the pretrained generic decision model needs explicit tracking
identity, lifecycle and memory-update supervision before own-state rollout.
Prepare real 16-observation legal training clips from MOT17-02, DanceTrack0001
and MOT20-01, mined for overlap, low visibility and missing observations.
Use a single shared runtime question/state builder and randomized persistent
identity aliases per clip. Teacher GT proposals/history are allowed only in
this initial curriculum; they are not GroundAnything output or benchmark
predictions. Missing shortlist identity and withheld correct-option rows are
explicit NO_MATCH supervision, never mislabeled as NEW. Full completion also
requires verified false-candidate supervision and actual model-state rollout.

Start with official multimodal language-model full SFT, frozen vision,
100 optimizer steps, one-row microbatches and four-row accumulation. Retain
the original seed0 and5e-6 recipe, verify all rows fit the token plan, save
every25steps, inspect real gradients/loss and reload the checkpoint. The
subsequent finite first training budget remains500–1000steps including
model rollout; this100-step check does not satisfy formal training completion.
The first native online clip is unexpectedly slow, so profile that exact live
process once before proposing a speed change.

## 2026-10-14 — native video failure, official parsing and resume

The native DLM + OneJev loop completed real frames302–304:31 candidates
each,30/31/31 output tracks,18.194/19.647/19.763s per frame. OneJev
created51 persistent IDs by the third frame, evidence of unadapted identity
fragmentation, not tracking success. The fourth frame failed the strict
client parser with non-canonical coordinate ordering/repeated phrase blocks.
The runner then deadlocked during cleanup: its parent waited for ffmpeg while
ffmpeg waited for an unclosed raw-frame pipe. A one-time targeted process
diagnosis identified this, the exact owned encoder received SIGTERM and
then SIGKILL after its verified10s timeout; no model/other-project process
was signalled. The original traceback returned and is preserved.

Repeating the exact frame/request three times yielded the same strict parse
error (17.009s total); resampling is not a fix and default inference still
uses one request per frame with unchanged model/decoder/sampling parameters.
The pinned official evaluation scoring parser preserves35 geometrically
valid boxes in this response. Add an explicit official_scoring adapter policy
using that unmodified function, retaining strict_wire_valid=false, raw output
and error. A regression on the observed failure matches every label/coordinate
with the official scorer and still rejects truncated generations. Client
strict mode remains selectable; no GT, coordinate repair or metric change.

Fix encoder cleanup by closing stdin before waiting, persist the first
failure, and reconstruct completed causal identity states from saved native
Choice/Noul/Score decisions. Real prefix reconstruction reproduced all three
frames' outputs/events/state exactly without GT or further model calls.
A refactoring mistake (len(questions) after moving request construction)
caused the first resumed attempt to fail at frame305; use len(request.questions)
and retain that failed attempt. The second resume starts from the same three
verified completed frames. Resumed legacy timing cannot account for the
original failed request/interruption, so no complete end-to-end FPS is claimed.

Teacher curriculum:18 legal16-observation clips,8964rows,59,665,520 history
PNG bytes,3 real reappearance observations in DanceTrack. All8964 official
question/target schemas verified;1728 Noul targets corrected from true/false
to the official yes/no labels before any training. No development/test GT
used. Official plan-only: p50/p90/p99/max5553/7201/7637/7858tokens,0 rows
dropped at16384, maximum18options (actual API limit255). Full pretrained
language-model100-step check launched on physicalGPU4 with frozen vision;
GroundJEV now uses two physical cards4and6, other jobs preserved. The training
owner is unique and blocking. This teacher phase still lacks complete false
candidate/termination/long-gap/model-rollout coverage, explicitly pending
for the next curriculum and formal500–1000step budget.

## 2026-10-14 — first native online output and negative association control

Four native MOT17 frames completed and passed prediction/one-to-one/memory
and ffprobe integrity checks:31/31/31/35 candidates,30/31/31/35 tracks,62
persistent IDs,next_id63. Official same-candidate4-frame TrackEval: Full
HOTA.421629,DetA.650042,AssA.281686,IDF1.427481,60IDSW,16FP,24FN; Simple
HOTA.748813,DetA.642304,AssA.895161,IDF1.836502,1IDSW,17FP,24FN. Retain
this negative generic-pretraining result; it supports tracking adaptation,
not a full-video/paper claim or global NO_GO. The resumed Full invocation
allocator peak8,350,338,048bytes covers that invocation only. Complete
end-to-end FPS remains null because the legacy interrupted run lacks failed
attempt timing; Simple10.8387FPS is association replay only.

The actual100-step SFT check is running: all474weights loaded,752.39M/852.99M
parameters trainable, first5 losses finite and nonzero gnorm, peak15.1GiB.
A separate single blocking owner starts all1,203DanceTrack0004 validation
frames with native DLM + pretrained Full, then same-candidate Simple and
unchanged TrackEval. Internal GPU monitoring excludes foreign jobs.
GroundJEV uses physical4and6 only; no queue/task was restarted implicitly.

## 2026-10-14 — complete-video generic decoder failure and official dense repair

DanceTrack native attempt completed20frames then failed frame21. The raw
output repeats the same box and ends with a trailing separator; even the
unmodified official scorer returns no valid predictions. Preserve all20
real frames, the valid latest memory, failure JSON and internally recorded
resource peaks. Encoder cleanup now returns the real failure promptly.
No frame is skipped or accepted as a valid prediction.

Root cause hypothesis: generic client uses greedy inner/outer sampling rather
than the officially documented dense request profile. The running server
YAML fixes sampling_seed42; retain42. Probe the same frame with official
DecodeV4 shared fields plus dense temperature.1/top_p.95/max8192 in both
outer sampling and nested gam_dlm_decode. Actual probe3.345s produces5
official-scoring-valid boxes; strict duplicate-coordinate error still logged.
Do not relax geometry/truncation/metric validation or switch DLM to VLM/AR.
Run a distinct, explicitly registered dense_v2 full video from its first
frame because request numerics changed; the old20frames remain a separate
negative attempt. This repair is retained pending full-video evidence.

SFT owner remains live and has reached observed step40 with finite losses,
nonzero gradients and15.2GiB allocator peak. Full ckpt-step25 is saved; no
completed100-step or reload result yet. Disk check:365GiB free on/data2,
enough for the finite model saves; feature-cache budget unchanged.

## 2026-10-14 — checkpoint validation and same-candidate SFT check plan

Previous goal turn made concrete progress: native model tracking, official
four-frame baseline metrics, actual SFT and verified GitHub publication.
Current single training handle is live at observed step80; dense full-video
owner is live at observed frame14. Continue those exact owners.

After the100-step owner exits, validate its done record and finite losses/
gradients; compare saved language-weight probes against the original at the
same BF16 precision, then load the checkpoint twice and check the same real
tracking Choice/Noul/Score probability distributions. Follow with the exact
four-frame candidate replay before/after control and unchanged TrackEval.
No evaluation GT enters inference and no full-video result is inferred from
this short scope. Then create real legal-train detector/model-history rollout
supervision for the finite formal adaptation budget; teacher-only training
does not complete the target.

## 2026-10-14 —100-step negative result and complete-frame scoring policy

Official SFT completed100steps/400examples in1509.262s. All474weight names
retained; three BF16-aligned language MLP probes changed1.882M/1.623M/1.083M
elements. Two reloads yield identical real tracking probabilities (maxdiff0),
unchanged official fork auto check passes. Four same-candidate official
metrics worsen: HOTA.328338,AssA.172383,IDF1.311787,75IDSW,17FP,24FN vs
pretrained HOTA.421629,60IDSW. Keep the100-step negative checkpoint; do not
claim completed formal adaptation or improvement from a falling training loss.

Dense request full-video attempt also stops at frame21 with repeated/malformed
wrappers, despite the isolated frame21 probe passing earlier. The request
contract repair does not guarantee valid output and must not be described
as a complete fix. Preserve both failures and20 valid native states.

For complete-video measurement, register explicit empty_and_record policy:
the same unchanged official scorer rejects malformed/truncated output; use
its empty prediction set, record raw error and latency, and run the actual
causal tracker on zero candidates. Every image remains in the official
metric denominator. Mark completion WITH_REJECTED_PERCEPTION, never all-valid
model success. Only GroundingProtocolError is handled; file/service/runtime
failures still stop. Reuse the verified20-frame prefix under unchanged
model/request/core settings; no skipped sample or reconstructed fake box.
This is an explicit measurement policy, not a GroundAnything quality fix.
Analyze correct-option coverage, selected identity and memory contamination
to determine why GT-teacher SFT is not sufficient before formal rollout SFT.

## 2026-10-14 — identity error diagnosis and evidence-format controlled repair

Post-hoc fixed-anchor diagnostics on the same four frames show correct-option
coverage100% for both pretrain and SFT100. Existing identity accuracy falls
from14/78(17.95%) to6/74(8.11%); native correct-option mass.09925 to.04026.
Candidate GT recall is the same111/135(.82222) for all three controls. The
shortlist is not the immediate cause on this scope. SFT100 holds all93
existing memory observations, while pretrain has7 wrong/unmatched memory
updates and only1 identity-consistent update. Preserve diagnostics as
post-hoc analysis, not official metric replacements or inference GT input.

Two concrete exposure mismatches need repair: inference candidates have no
explicit crop identifiers, relying on pixel-coordinate arithmetic in the
full image; teacher rows use6candidates/16historical options vs16/32 at
runtime, and miss wrong detections/model histories. Preregister optional
v2_crop_motion evidence: current candidate crop sheet plus immutable
anchor/long/recent history sheet, explicit causal IoU/center-distance
matrices as features, and a genuine NO_MATCH option. Hungarian still uses
native OneJev probabilities. Keep v1 unchanged for the ongoing full-video
baseline. Run pretrain and SFT100 with identical v2 evidence/candidate lists
as a controlled diagnosis before forming the new own-history curriculum.

Evidence v2 pretrain produced no tracks: all128 candidate actions were
NO_MATCH, including the first frame with no past identity. The new criterion
was semantically overlapping with NEW. Retain that version/results and finish
the matched SFT100 control. Register v3 as a minimum semantic repair: NO_MATCH
requires a previously observed object and is not offered with empty history.
Current crop/motion evidence otherwise identical. Do not silently overwrite
v2 or alter its model scores.

Evidence v2 pretrain and SFT100 both collapse to NO_MATCH on all128
candidates (HOTA0,FN135); retain this actual semantic-error result.
V3 removes NO_MATCH from empty history and distinguishes previously observed
objects from NEW. Pretrain produces10/29/24/35tracks,HOTA.343944,47IDSW,48FN.
SFT100 produces31/0/0/0tracks,HOTA.197395,0IDSW,110FN: zeroIDSW here reflects
missing outputs, not good association. No improvement or full-video claim.
The initial16-option teacher curriculum never exposes production35-option
questions; together with teacher-only states/absent wrong detections this
requires a matched-density own-state curriculum.

Start actual checkpoint100 V3 rollouts on three legal16-observation train
clips,32GT proposals plus two explicitly synthetic box-fragment negatives
per dense frame, actual model states/decisions and no GT state correction.
Reuse the observed real DanceTrack reappearance clip. Export supervised
requests with correctly permuted IDs and crop-sheet labels,35-option
exposure where actual memory permits, false-candidate IGNORE and real
wrong-history/lifecycle/memory labels. This isolates decision learning;
GT proposals are declared, not mislabeled as GroundAnything detections.
Actual DLM-proposal training rollout remains a further required stage.

The actual SFT100 V3 student rollout completed48legal observations and
6792supervised rows,38,165,373bytes crop evidence,max35options. It has96
IGNORE negatives,139NEW labels,87withheld-option NO_MATCH cases,206
TERMINATE and2620REACTIVATE supervision. Actor after the first frame chooses
NO_MATCH and keeps lost memory, so this branch does not supply enough wrong
identity updates. Preserve that collapsed real rollout, then generate the
same registered clips with the real pretrained V3 actor to obtain diverse
wrong association/memory states. Both branches use GT proposals plus explicitly
synthetic fragments, with GT labels only in the post-hoc supervisor.
Formal adaptation will mix both branches and increase identity row weight
4x (explicit repeat sampling) because the initial curriculum overweights
lifecycle relative to association. Keep seed0,optimizer/lr and all registered
frames; do not claim a completed formal500-step run until it actually ends.

Pretrained V3 rollout completed6709rows/48observations/44,023,768bytes
and526 contaminated history observations. Combined with the6792-row SFT100
rollout gives13,501unique rows,20,881 after explicit4x identity repeats.
Official complete plan: max22817/p50 15520/p90 22158/p99 22544,0 dropped
at32768 context; actual35-option exposure. Finite500-step official full
language-model SFT began from checkpoint100 with a deliberate new curriculum
optimizer, seed0/lr5e-6 unchanged. First two losses3.2479/8.0658 and nonzero
gradients299.979/513.188; peak18.1GiB,75.8/76.2s per step. Formal completion
and before/after remain pending. Source proposals remain declared GT plus
synthetic fragments; no claim of completed GroundAnything-proposal rollout.

## 2026-10-14 — proposal-aware lifecycle label regression and resume

A targeted audit found203 wrong ACTIVE/REACTIVATE lifecycle targets:164/3015
student100 and39/2936pretrain rows label visibility despite no matching
current proposal. Lifecycle requires association, so these targets are LOST.
Signal only the verified exact500-step trainer (PID3993066, own cwd/config,
SIGTERM), preserve its first5logged steps and alloriginal artifacts. No
checkpoint was saved in this aborted stage; resume from validated100-step
weights, seed0 unchanged. Old owner state STOPPED_FOR_VERIFIED_LABEL_REPAIR.

The repair initially parsed life_357 as _357; fix the delimiter, preserve
the partial derived repair, and regenerate from original real actor traces.
The prematurely started repaired preflight correctly refused incomplete
corpora and did not train. Source corpora, allstates/media, identity aliases
and non-lifecycle targets stay identical. Assert all5951 lifecycle targets
consistent with supplied current proposals and exactly164+39labelchanges.
A new corrected500-step owner starts with unchanged weights/lr/seed/context/
identity-repeat budget. The stopped run is not declared formal completion
or silently combined into an improved result.

## 2026-10-14 — native DLM visual feature adapter plan

Previous goal turn made verified progress: real SFT/reload/negative controls,
real actor histories, lifecycle label repair/regression, resumed finite500
training and exact GitHub publication. Continue existing live owners, not
new copies. Add a native visual adapter using DLM checkpoint visual/projector
weights and the unmodified official eager MoonViT/K3 preprocessing path.
Compare the official reference visual module with the SGLang visual adapter
on the same real image, preserve feature-grid geometry through resize/padding,
then ROI pool candidates. Only this small validation uses physicalGPU5,
within the project allowlist/4card ceiling; live training4 and video6 unchanged.
This provides genuine native features for temporal adaptation/Compact, not
a substituted crop encoder or a full-dataset cache.


## 2026-10-14 — Native feature adaptation and legal task loops

Plan after source inspection: verify the actual DLM visual/projector output,
train a finite frozen-backbone temporal adapter, wire these features into
causal memory, then run legal Refer-KITTI/TAO loops with the existing OneJev
checkpoint and exact-candidate Simple controls. Keep formal SFT500 and the
full DanceTrack owner unchanged; use physical5 for new checks within4/5/6/7.

Native DLM:428,482,048 visual parameters/170weights; real MOT regions31x2560
and local tokens31x2560x3x3. Official serving visual function and immutable
checkpoint reference both match exactly; independent ROI mapping differs
9.54e-7. Initial config.hidden_size lookup failed after real forward; fixed
only the adapter to use vision_config.out_hidden_size and retained failure.

Temporal adapter:200steps, seed0,693,632 changed parameters, loss4.2086 to
.09104, nonzero finite gradients; exact reload difference0. Frozen native
backbone and decoder unchanged. Independent TRAIN calibration retrieval
131/134=.977612, identical to unadapted features (random adapter130/134).
No tracking-improvement or decoder-fine-tuning claim. Features are transient;
training-device/CPU-training regions11,612,160bytes each, calibration1,464,320.
Checkpoint receipt records absolute path,8,331,829bytes and SHA256.

RMOT:legal current v2 validation0007, single+empty8frames and multi+empty
12frames, GT-free inference/shared V1 core. Single Full HOTA0/IDF1=0/FN5;
Simple HOTA.084894/FP11/FN5/IDF1=0. Multi Full HOTA.072117/FP16/FN30; Simple
HOTA.081860/FP16/FN30; both IDF1=0. Proposal target coverage at IoU.5 is0/5
and0/30. First bottleneck is referring proposals, not evidence of an identity
head improvement. All rejected/empty frames remain in denominators. Multi
full video-processing FPS.579804 including serialization/encoding; preloaded
remote DLM initialization excluded. Single legacy frame-compute FPS.627761
is explicitly not complete end-to-end speed. Exact fixed-candidate controls
never claim detector FPS.

Evaluation first failed on GT_LOC_FORMAT, then on omitted YOLO files in empty
query frames. Set the correct path format, and look up boxes only when the
release expression has targets; nonempty targets still require real labels.
Original failures and predictions remain; completed multi Full/Simple units
were reused by a new evaluation owner. No GT boxes, thresholds or scopes altered.

TAO:registered first video40annotation frames and4frame functional scope,
all1203fixed LVIS-v1 names in96-class native requests. Failure in any chunk
rejects the whole frame explicitly; no class or sample silently dropped.
Source physical FPS unavailable, so this functional scope declares annotation
step time and does not claim a verified physical video clock. Full TETA with
all original modal GT/category/frequency metadata is required; currently pending.

Memory:real native proposals/features on2frames pass exact serialization,
anchor immutability, held-long-feature consistency and past-time rejection,
with10,682,927byte memory. V1 crops-only state remains backward compatible.
New memory adds actual region/temporal anchors, recent/reliable-long features,
16-observation feature history, native lifecycle survival probability, identity
entropy and semantic observation counts. These are used only with native input.
V4 JEV native-matrix request failed at56,426tokens on frame303; preserve it.
V5 explicitly changes numeric evidence to CSV milli-units, retaining every
candidate/identity pair and the original32768limit. Its real4frame test is
running; no performance gain claimed. Compact/full-task and retention studies
remain required. Previous goal work is progress, not goal completion.


### Completed native-feature control and measured limitations

V5 still exceeded context at34313tokens. V6 compresses native-history rows
in addition to all-pair similarity matrices; frame304 then reached34419tokens.
The smallest further fix is context-adaptive candidate grouping only when the
unchanged official native preflight rejects a group before inference. Every
proposal and up-to32 shortlisted identity options remains. The two valid
V6 prefix frames restore exact native outputs/state; resume completes304/305.
Original V4/V5/V6 failed artifacts remain, no sample skipped, no token-limit
relaxation and no retraining/checkpoint change.

MOT17 same128native candidates,4frames, unchanged TrackEval:

| Decision | HOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|
| Simple IoU | .748813 | .836502 | 1 | 17 | 24 |
| Fixed IoU + native2560cosine | .738898 | .813688 | 4 | 17 | 24 |
| Fixed IoU + trained temporal128cosine | .746248 | .828897 | 2 | 17 | 24 |
| Pretrained OneJev V6 native/crop/motion | .359328 | .330709 | 61 | 17 | 33 |

Native/temporal fixed controls use exactly the exported features, coefficients
.5/.5, NEWscore.3,32-identity geometry shortlist and16-candidate groups. They
are untrained association controls, not JEV/Compact. Temporal-vs-native is
also a dimensionality change; random128 and larger video controls remain
required before attributing the difference solely to training. Native-feature
OneJev still underperforms Simple; no isolated feature or foundation gain
claim. Existing-ID diagnostic accuracy9/75=.12, option coverage72/75=.96,
mean correct native mass.062495; candidate GT coverage remains111/135.
Its8,504,115,712byte allocator peak and.042028association replay FPS exclude
live detection; resumed timing does not support an end-to-end speed claim.

TAO first4frames have zero predictions and TETA/LocA/AssocA/ClsA0 under the
unmodified official source b498aa87b252bfb75d7cc0d8d6435c3e1e74e4de. There
are18GT boxes, source canonical class baby/person.n.01, Base. Novel is null
because this scope has no Novel GT; never report it as a Novel benchmark.
Two evaluator formatting configuration errors were repaired by restoring
its official defaults OUTPUT_PER_SEQ_RES=True and OUTPUT_SUMMARY=True;
metrics/source/GT/predictions unchanged, failures retained.

The first vocabulary run registered1203classes but stopped on a rejected
96-class chunk:192/192/192/288classes actually requested across the4frames.
This is recorded in vocabulary_coverage.json, not claimed as exhaustive
1203-class inference. A new finite owner requests ALL13chunks on every frame,
records every native outcome, and only then applies the same explicitly
registered whole-frame-rejection policy. It is active at publication.
First-prefix run processing FPS.020682 is not full-vocabulary inference FPS.
Original raw failures show many class-conditioned hallucinated boxes. The
next grounding research must address proposal quality and negative classes
through the official spatial/DLM training route, with dense/referring retention.

Visualization integrity verified8/12/4frames; KITTI display edge padded from
1242x375 to1242x376, predictions/GT coordinates unchanged. Formal500-step
SFT and fullDanceTrack original blocking owners continue. Pilot and final
unified research goal remain incomplete.


## Pre-publication RMOT GT-contract repair (authoritative results)

The new RMOT evaluator incorrectly read release top-left x/y as centers.
This was found before updating GitHub's branch. Existing established
read_gt in tools/unified/u1_formal_expression_gate.py defines normalized
top-left xywh. Corrected transport agrees for all70target box comparisons
to5.68e-14pixels. Predictions, target sets, frames, source extents, seeds,
metrics and thresholds unchanged. Original metric receipts are preserved
as invalid_center_format_metrics.json and explicitly marked invalid;
local invalid GT views remain but raw GT is excluded from GitHub.

Authoritative corrected results:

| RMOT scope / backend | HOTA | IDF1 | IDSW | FP | FN | Proposal target recall IoU.5 |
|---|---:|---:|---:|---:|---:|---:|
| Single/empty8frames Full | 0 | 0 | 0 | 0 | 5 | 4/5 |
| Exact candidates Simple | .242284 | .375000 | 2 | 7 | 1 | 4/5 |
| Multi/empty12frames Full | .194792 | .217391 | 0 | 11 | 25 | 5/30 |
| Exact candidates Simple | .243151 | .217391 | 0 | 11 | 25 | 5/30 |

The earlier zero-proposal-coverage diagnosis is invalid. On the single
query, four correct observations are present but V1 OneJev suppresses them
as IGNORE; association/birth/input evidence is the first failure. On the
multiple-target query, proposal coverage5/30 still limits every tracker.
Next controlled check should compare V1 against current-crop V3 on the
same RMOT proposals, while preserving the DLM multi-target recall deficit.
Dense/referring adaptation and original retention remain required, but
not every RMOT failure can be attributed to the perception front end.

The new all-class TAO owner remains active. A800-SXM4-80GB hardware was
verified for4/5/6; foreign UDMT processes are unchanged. Original OneJev
500step and fullDanceTrack owners remain responsible for their jobs.


## Exhaustive TAO functional control completed

All52native requests completed:13chunks and1203fixed categories for each
of4registered annotation frames. Unchanged TETA yields0TETA/LocA/AssocA/
ClsA and0predictions under the recorded whole-frame-rejection policy; Base0,
Novelnull because this video's GT contains no Novel class. No full-video
or complete-dataset result is claimed. Actual complete processing
FPS.0037577955 includes all calls, association, file/video writes and writer
completion; preloaded remote DLM startup and9s-class OneJev loading excluded.

Rejected/scored chunks per frame:11/2,10/3,7/6,10/3. Independently scored
chunks contain201/441/777/354boxes. Evaluation-only class-agnostic target
coverage is10/18 atIoU.5 before whole-frame rejection. This is a localization
diagnostic, not TETA or classification correctness. Whole-frame rejection
discards useful native outputs; retain this negative control and preregister
a query-level empty-prediction control that retains independently scored
chunks, all class requests, all GT and frames. Never label invalid chunks
valid or change the official parser/metrics. Raw native responses remain local.

At this completion check, the original blocking SFT500 owner is live, last
reported50steps/loss2.8577/20.6GiB allocator; original fullDanceTrack owner
is live, completed769of1203frames. Dataset physical clocks independently
agree with seqinfo:MOT17=30Hz,DanceTrack=20Hz,MOT20=25Hz. TAO sourceFPS
remains unavailable and annotation-step units explicitly declared.
The69-file native-feature/GT-contract stage was verified on GitHub at
1724df9de3465fce5fd1de96ccf81510029cd7f1 with exact local tree equality.
No Pilot or final-goal completion claim.


## 2026-10-14 — Next controlled task decisions

Previous goal turn:progress, with real native adaptation/task controls and
verified GitHub publication. Current project is the authorized independent
GroundJEV worktree. Plan after code inspection:fix proposals/checkpoints/data
scope, compare RMOT V1 vs V3 current-crop evidence for pretrained/SFT100;
then retain independent scored TAO chunks, keeping failed chunks empty and
all1203classes/every GT/frame. Preserve whole-frame rejection as a separate
negative control. No official parser/metric or seed changes. Reconstruct
only already-recorded native responses, with no GT access or new detector
requests; all resulting timings are association replay, not end-to-end.
A single blocking owner executes registered finite controls on physical5;
original SFT4/DanceTrack6 owners continue. Peak monitoring stays internal.


## 2026-10-14 — Registered task and inference controls

All six controls completed on A800 GPU5 with original OneJev weights or
actual SFT100. Predictions use exactly the same saved native candidates;
GT uses verified top-left Refer-KITTI release coordinates and unchanged TETA.
Official scopes remain8/12RMOT frames and4TAO annotation frames.

| RMOT / V3 | HOTA (fraction) | IDF1 | FP | FN |
|---|---:|---:|---:|---:|
| Single pretrained | 0 | 0 | 0 | 5 |
| Single SFT100 | 0 | 0 | 3 | 5 |
| Multi pretrained | .222720 | .177778 | 10 | 25 |
| Multi SFT100 | 0 | 0 | 1 | 30 |

Current crops alone do not fix cold IGNORE. MOT-only SFT100 has negative
transfer; it is not evidence of useful RMOT fine-tuning. Original finite
MOT SFT500 continues, and mixed-task/actual-proposal training remains needed.

| TAO scored-query retention | TETA (%) | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|
| Fixed native candidates Simple | 24.171 | 29.452 | 43.061 | 0 |
| Same native candidates OneJev Full | 16.815 | 27.459 | 22.987 | 0 |

Failed query chunks remain empty; independently scored chunks retained,
without re-running the detector, filtering classes/GT, repairing invalid
syntax, altering metrics or dropping frames. This is a registered combiner
control, not an output-format pass. Full remains slower and less accurate.
Both classification terms0 expose missing/wrong class decisions; the native
shared JEV class head is still required. Novel is null in this Base-only scope.
No detector-inclusive or full-dataset speed claim:replay FPS.838284Simple
and.00352281Full, with all1773label/box rows kept.

Exact-coordinate object-hypothesis grouping preserves every distinct native
box and all label alternatives:201/441/777/354rows become54/58/109/85.
No overlap threshold/NMS/topK/GT filtering. Independent Simple/Full controls
are active; first native label remains an explicitly unlearned class baseline.
A future genuine typed classification head must use the full dynamic vocabulary.

Isolated FLA0.5.2 + einops0.8.2 addon (original environments untouched):
GDN kernels2.29/3.24/8.43x faster at257/1024/4096tokens after both sides'
warmup, maxoutput difference.0004883, finite gradients. First import lacked
einops and was fixed only in the addon. First unbalanced warmup is preserved
and does not support63x speed. Actual3multimodal OneJev requests differ up
to.021211 in typed answers, exceeding the preregistered.001. Choices unchanged
but probabilities/material memory gates can change. Accelerator rejected;
original automatic fork check/.05 tolerance remains unchanged, training
and original video owners stay on the reference backend. No end-to-end gain.

Native DLM training preparation reconstructs the actual official
GAMQwen3DLM wrapper, MDM.5+causal.5/block32, with all569tensor keys/shapes
matched. The released JSON has152670AR rows; real checkpoint/tokenizer have
152671including MASK152670. Native reconstruction uses real header size; no
weight is added or randomized, no new AR download required. Initial mismatch
retained. This is a key audit, not gradient proof.

A finite3step native projector-only gradient check on legalMOT17TRAIN209/210
combines original DLM spatial loss and same-ID contrastive ROI loss. Language/
ViT frozen, projectorFP32masters, seed42, all32targets each frame with no
clipping or dropping. It is a custom native-adapter policy, not the stock full
language SFT recipe or completed adaptation. Meta nonpersistent rotary buffers
initially caused allocation failure; native Accelerate initialization fixes
that without creating learned values. The strict native cached-image schema
then rejected a path-only dict; real original bytes/path are now supplied.
Both failed units preserved, new real-gradient check active onGPU7. Four
physical GPUs4/5/6/7max; foreign UDMT jobs untouched. Dense/referring retention,
exported bundle inference and formal adaptation remain required.


Exact-geometry controls completed:Simple TETA33.042/LocA48.539/AssocA50.587,
Full23.320/49.095/20.866, both ClsA0. Simple replay9.8226FPS, Full.0197772,
about5.6x the ungrouped Full. No detector-inclusive speed claim, no learned
category assignment and no Full advantage. Native geometry set/all label
alternatives retained; every original GT and frame evaluated.

Native3-step adaptation then exposed projector output as a per-image tensor
list, which was corrected using native concatenation. The first native
backward hit CUDA-graph lifetime replay; first failed units remain unchanged.
The next unit compiles the identical native FlexAttention operator without
CUDA graphs and probes native spatial-loss and temporal-loss gradients
independently before optimizer updates. Seed, block masks/loss weights/GT
and all target instances unchanged. PreserveFP32nonpersistent RoPE buffers.
No native decoder/localization adaptation success is claimed until actual
finite loss/gradients, changed BF16weights and checkpoint reload are verified.


### Actual native DLM spatial and temporal gradient proof

Final v6 native training unit completed3optimizer steps on legalMOT17TRAIN
209/210, all32targets each frame and32same-ID positives. Official native
GAMQwen3DLM MDM.5+causal.5/block32 retained, with.1identity ROI contrastive
loss. Language and native ViT frozen;27,267,584native projector parameters
useFP32masters. Independent dense-gradient norm8.75947 and temporal-gradient
norm2.13338, finite combined norms8.77141/9.25557/9.02194. Native losses
1.61708/1.66780/1.65477 include different sampled noise; do not claim training
improvement from this warmup.26,766,531FP32elements and1,968,676BF16elements
changed. Delta reload weight difference0,327,218,423bytes, SHA256
bf8537a97675e465d80dfb08cb5b21454ee37711f622510fe8181b86370e2b3e,
peak18,455,674,368GPUbytes. This proves actual localization and identity
gradients, not formal adaptation or tracking gain.

Checkpoint:/data2/user/LocateMOT_GroundJEV/outputs/groundjev/ground_adaptation/
native_tracking_grad3_v6_native_runtime_flags/projector_delta.pt. Frozen
nonpersistent rotary buffers remainFP32. Native official CUDA-graph flag0
and task-local donated_buffer=False accommodate checkpointed backward and
independent gradient probes; math/masks/loss weights/seed42unchanged.
Failed meta-buffer/image-schema/list-output/CUDAgraph/donation units retained.

Exported complete DLM variant models/GroundAnything-DLM-TrackProj3 replaces
only5native projector tensors, retaining all569keys and every other native
weight/tokenizer/code/config unchanged.9,687,419,704bytes, SHA256
a626aad29c5bbc17e3a465a519e93681aaedacd81cbb21805cce40b7122777e9.
Original model/server untouched. A distinct GPU7replica/8102 and launch bundle
are used for48same-profile native frames:4dense+8single-ref+12multi-ref,
original/updated, fixed Simple association. Requesteddecoderseed42, same
unchanged official parsers and evaluator. These functional retention probes
do not stand in for the full original GroundAnything benchmark suite.
The one blocking owner stops only its owned adapted replica on completion.


Adapted replica first launch passed real weight loading butKV profiling at
mem-fraction-static.25produced a negative cache budget. Owned replica
was stopped and failed launch preserved. New isolated replica uses.4(the
original service's working fraction), same checkpoint/decoder/seed/scopes.
No other process stopped and no GPU beyond4/5/6/7.


### Verified native projector retention controls (2026-10-14)

All six fresh inference and official evaluation units passed scope and encoded
video frame checks:48frames total, no runtime GT or future access, every
rejected/empty frame retained. Original versus projector3 HOTA/IDF1 fractions:

| Scope | Original HOTA | Projector3 HOTA | Original IDF1 | Projector3 IDF1 | IDSW before/after |
|---|---:|---:|---:|---:|---:|
| MOT17 4frames | .745083 | .755029 | .824427 | .847328 | 3/1 |
| Single RMOT 8frames | .299355 | .303853 | .375000 | .400000 | 1/1 |
| Multi RMOT 12frames | .229517 | .227592 | .222222 | .212766 | 0/1 |

Both single variants reject frame152; both multi variants reject frame34.
These frames remain in official denominators. Decoder tracking_dense,
requested seed42, official_scoring parser, unchanged geometry Simple and
legal validation scopes are identical. Short dense and single controls
improve slightly, multi-referring declines. This is not formal adaptation,
full original grounding/referring benchmark retention or a broad gain claim.
Actual detector-inclusive FPS are in aggregate_results.json; concurrent
workloads and original GPU6 versus replica GPU7 preclude an isolated speed
comparison. The first replica failed KV allocation at mem-fraction.25;
its failure is retained. The completed replica used.4 and stopped its own
process group. Original server/model and foreign UDMT are unchanged.

Authoritative receipt:
outputs/groundjev/ground_adaptation/projector3_retention_v2_memory_budget/aggregate_results.json.
OneJev formal500step owner remains live, latest returned report100/500;
full1203frame DanceTrack owner remains live, latest returned frame1183.
No completion claim for either. Current persistent feature service on5
uses about4GiB; no GroundJEV process remains on7 after replica cleanup.
Next research hypothesis: exact object geometry grouping improves localization,
but copying a native query label leaves ClsA0. A real typed OneJev category
head using every supplied vocabulary class, separate from identity and absent
detection confidence, is required. It must preserve all candidates and GT.


### Native shared JEV category hypothesis (registered before execution)

Exact-geometry Simple/Full ClsA0 exposes first-label copying. Add an opt-in
native typed category head on the same OneJev parameters, with all1203input
labels represented in deterministic128class root groups, selected conditional
leaf and explicitUNKNOWN. Hierarchical routing is approximate, not a flat
1203class probability distribution. Preserve all306geometries and unknown
predictions; no GT/class/box filtering or metric changes. First run fixes
Simple identity to isolate semantics, then same-policy Full uses the same
engine for identity/category. Keep class confidence separate from identity
and missing detector confidence. Plan:configs/groundjev/native_category_control_20261014.json.


### Actual GroundAnything proposal own-history rollout

The current500step training corpus uses GT proposals and declared synthetic
fragments. To close that gap, register48legal training frames from the same
three MOT sources, original DLM candidates with no topK/NMS/GT substitution,
and the valid100step OneJev actor. Ground failures remain empty frames.
Record only successful context groups, retain every native candidate, keep
GT matching/aliases outside uncorrected actor state, and supervise lifecycle
from targets matched to current actual proposals. This creates training
supervision, not validation predictions or a new500step result.
Plan:configs/groundjev/native_candidate_rollout_20261014.json.


### Real-video memory export performance repair

Native Full1203predictions completed, but Simple control spends seconds per
frame re-encoding unchanged historical images. Actual155track profile:3.283s
state serialization,3.256s inside465PNG saves. Cache at most one exact PNG
encoding per live track/image field, keyed by image object identity. Native
tracking replaces images rather than mutating saved crops. Encodings remain
byte-equivalent and are released with each track; no persistent feature cache.
Verify actual32frame decision replay, cold/warm/full-state reload equality,
then preserve complete Full and the interrupted valid Simple prefix in a
fresh owner. No detector rerun, new seed, metric/filter/state change or E2E
speed claim. Source failure/controlled interruption remains recorded.


### Native class decisions, actual detector rollout, and serialization repair

Opt-in native_hierarchical category inference now shares the actual OneJev
engine/parameters with Full identity decisions. All1203input categories are
represented in128class root groups; selected-group leaf and explicit UNKNOWN.
This is approximate hierarchical routing, not a flat global MAP classifier.
Native path probability remains separate from identity probability and absent
native detector confidence. Every306exact native geometry is retained.

Completed Simple/Full TAO controls:33.042/23.320TETA, bothClsA0, unchanged
localization/association compared to native-label controls.299/306objects
choose rootUNKNOWN; seven receive vehicle labels. Full and Simple category
decisions are exactly equal; all identity probability differences are0.
Actual four-frame videos, six-decimal probability rounding, all candidates,
GT/empty/rejected frames and unmodified TETA verified. Native category
inference is implemented but does not solve classification; it remains
opt-in. No OVMOT gain or Novel result claim. Next step is legal category
supervision and targeted visual/group-routing diagnosis, not GT relabeling.
Receipt:outputs/groundjev/task_controls/native_category_v1/aggregate_results.json.

Actual GroundAnything proposals now feed a finite48frame legal own-history
training exporter on physical7, original DLM service6. Actor is the valid
OneJev100step checkpoint, V3 crops/motion. No GT proposal substitution,
synthetic fragments, topK/NMS or corrected actor state. Independent GT
matching supplies training targets only; lifecycle labels use targets matched
to actual current proposals. Record only accepted context groups and retain
all candidates including empty/rejected native frames. This closes a genuine
training-data gap; corpus owner remains active, not completed training.
Plan:configs/groundjev/native_candidate_rollout_20261014.json.

Full native DanceTrack0004 predictions completed1203frames,27rejected frames
(2.244%) retained. Original20frame resume means end_to_end_fps remains null.
Official full-video metrics remain pending the exact-candidate Simple control.
Simple recompressed all historical PNG crops every frame:actual155track
profile3.283s,3.256s in465PNG saves. Exact per-field cached encoding retains
at most3encodings per live track and releases them with the track. Actual
223track snapshot:4.487s cold/.003874s unchanged warm call;20,884,556cached
encoded bytes.32frame recorded causal replay, cold/warm payload and fresh
reload are exact. These are serializer timings, not whole-video speed gains.

Only verified owned Simple child4127910 was interrupted after its valid
200frame prefix; original failed owner and interruption receipt preserved.
Fresh owner reuses all complete native Full outputs and restores the entire
unchanged Simple prefix. First reuse preflight wrongly expected the newer
multitask specification field in the older ordinary-MOT receipt; retained
failure and fixed using actual registered paths/frame numbers/native prompts,
not synthesized metadata. v5 owner is active with cached PNG export, no new
native detector calls or changed experimental protocol.
Full/Compact/matched trained controls, mixed task supervision, full original
GroundAnything retention and formal500step checkpoint remain required.


### Fresh full-video V3 evidence measurement

Full V1 actual1203frames:HOTA.0214546,IDF1.00482694,IDSW3792; identical
Simple:.0852976,.0395616,746. Native candidate recall4341/4586=.946577,
Full existing-identity accuracy234/4330=.054042 and option coverage.847113.
Retain this negative baseline. Register a fresh complete1203frame V3 run:
real current/history crops and geometry, native NO_MATCH, all proposals,
unchanged native decoder/official metrics, adaptive context groups without
option/candidate removal. Use verified cached PNG persistence; no resumption
or candidate replay in Full, so a complete E2E record can finally be measured.
Still pretraining baseline, not trained500checkpoint or a completed Pilot.
Plan:configs/groundjev/dancetrack_fresh_v3_20261014.json.


### Full DanceTrack0004 baseline is now official and validated

Both1203frame video artifacts and exact candidate equality passed. Native
Full V1 HOTA.0214546/IDF1.00482694/IDSW3792/FP9300/FN241; same-candidate
Simple HOTA.0852976/IDF1.0395616/IDSW746/FP9334/FN256.27native rejected
frames (2.244%) stay in every official denominator. GT4586observations,
native13664candidates, one-to-one IoU.5 coverage4341/4586=.946577.
Full correct surviving identity associations234/4330=.054042; correct
option coverage3668/4330=.847113.9323unmatched candidate diagnostics
include official ignore regions and are not themselves official FP labels.
These negative results show both excessive candidate output and weak
pretrained identity decisions; detection recall alone is insufficient.

The complete200frame Simple prefix matches its preserved source row for row.
Original Full20frame resumption and Simple200frame resumption preclude a
continuous E2E claim. Simple32.62198replayFPS excludes native detection and
full persistence/rendering costs. Scalar diagnostics never modify output.
Receipt:outputs/groundjev/pilot_runs/dancetrack0004_pretrained_dense_v5_cached_memory_control/aggregate_results.json.

A fresh complete1203frame V3 native run is now active on physical6. Same
pretrained OneJev/original DLM, decoder and official evaluator, real current
candidate/history crops and motion evidence, explicitNO_MATCH, no resumed
prefix or Full candidate replay. Native context splitting retains all boxes
and up-to32identity options. Exact cached PNG export preserves checkpoint
payloads. Completion will measure actual complete pipeline E2E speed; no
performance or full-goal completion claim while active. Formal500step
training remains the original live owner; no restart or shortened protocol.


### Actual native detector own-history corpus completed and validated

48declared legal training frames,1184actual native proposals,5661supervised
rows:1184identity,1184update,1184reliability,69withheldNO_MATCH,2040lifecycle.
All typed targets/options/media and training split membership validated,
no GT IDs in model state, no GT state correction or synthetic fragments.
Native GT matches:474/526MOT17,110/110DanceTrack,496/806MOT20. Uncovered
GT remains explicitly counted; never invented as native proposals. This is
decision supervision, not localization supervision for missing boxes.
Actor outputs only74across48frames (31/12/31first-frame births, subsequent
collapse), exposing the valid100step actor's negative transfer rather than
hiding it. All raw own histories are retained.31,045,329bytes of crop evidence,
no full source-dataset feature cache. Corpus completion is not weight update
or formal training completion. Receipt:
outputs/groundjev/training_data/native_ground_own_history_v1/corpus_validation.json.


### Compact native evidence/distillation study (registered)

Full baseline is weak: use actual recorded OneJev100 distributions as a
.2auxiliary term with GT supervision dominant. Pack all48legal train frames
using native frozen DLM ROI features and the actual actor anchor/recent/long
events, without correcting its state. Compare structured set Compact, the
identical architecture with GT-only supervision, and a parameter-matched
feedforward control with the same heads/evidence/distillation. Keep dynamic
IDs, NEW/IGNORE/NO_MATCH, memory update, reliability and lifecycle. Declared
option-withholding moves removed native teacher mass toNO_MATCH; it is an
augmentation, not a new native teacher request. InitialMOT-only stage, not
full unified-task or long-memory proof. Plan:configs/groundjev/compact_native_20261014.json.


### Actual trained Compact and matched baselines (2026-10-14)

All48legal native detector/OneJev100own-history train frames are packed as
80base groups plus69declared option-withholding augmentations,149examples.
Native DLM ROI2560and uncorrected anchor/recent/long feature history; noGT
state correction. The removed teacher identity mass is projected toNO_MATCH
only in the declared training augmentation. Native pack131,123,485bytes,
not a whole-source-dataset feature cache.

Three actual300step/seed0 models completed with finite nonzero gradients,
changed weights, exact checkpoint reloads. Compact distilled and GT-only
set control each930,829parameters; feedforward distilled control929,805,
ratio1.001101. Two set blocks provide interactions; feedforward control has
no set attention but shares mean context, evidence and structured heads.
All support dynamic identities/NEW/IGNORE/NO_MATCH, update, reliability and
lifecycle. They are trained structured decision heads, not direct pretrained
OneJev decoding. GT supervision plus.2native recorded Full teacher loss;
identical set GT-only control omits only that loss. Auxiliary visual backbone
remains original frozen DLM. Parameters for all models changed, reload0,
candidate/history permutation error<=7.63e-6. Allocation peaks about40.5MB
are core-only, excluding CUDA runtime and shared native visual/DLM services.
Initial Compact training is MOT-only: task-index conditioning exists, but
joint expression/class training and full unified-task/long-memory proof remain
required. Never present natural set capabilities as JEV-exclusive novelty.

Same real128native proposals/ROIs,4MOT17development frames and unchanged
TrackEval, actual4frame videos validated:

| Core | HOTA fraction | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|
| Compact distilled set | .669111 | .738197 | 1 | 9 | 46 |
| Same set, GT-only | .576415 | .638889 | 8 | 3 | 57 |
| Matched distilled feedforward | .634360 | .703196 | 3 | 3 | 54 |
| Existing plain Simple reference | .748813 | .836502 | 1 | 17 | 24 |

Distillation helps the same architecture in this tiny single-seed control;
Compact still trails Simple and has more missed targets. ReplayFPS4.358/
3.959/4.512exclude detector/native feature extraction and full export costs;
no E2E or full-video accuracy claim. Receipts in
outputs/groundjev/compact/native300_controls_v1/aggregate_results.json;
trained checkpoints and SHA receipts in outputs/groundjev/compact/native300_v1.

Registered full1203frame DanceTrack0004 Compact and two controls reuse every
original native detector candidate, unchanged official GT and scorer. Stream
ROI once through physical5native feature service; controls reuse identical
ROI values on physical7. Source frames are read progressively rather than
loading the complete replay into memory; actual4/1203row equivalence passed.
These are candidate/feature replay controls, not detector-inclusive E2E.
First run failed at actual empty-proposal frame21: empty ROI pool returned
BF16 whereas nonempty interpolation and optional temporal layer useFP32.
Preserved20complete frames and failure; fixed only empty output dtype.
Only verified owned native feature service4043767 was restarted on5. A
real frame21 empty RPC/lifecycle probe passes and real frame20nonempty ROI
max difference after restart is0. A probe initially used the wrong timestamp
constant; it was corrected to the registered20Hz clock, with no data/clock
protocol changes. Fresh v2 owner restores all20native prediction/state rows
and resumes frame21, without dropping it. Original OneJev500/Ground/freshV3
and foreign UDMT remain untouched. Full metrics are pending its terminal
owner. First model frame latency includes ROI RPC; later controls read fixed
ROIs, so do not claim a direct comparable speed ranking across those latencies.


### Shared semantic Compact adaptation (registered)

MOT-only Compact cannot establish task transfer or category accuracy. Keep
all native MOT149groups, add48frozen-fitKITTI teacher-GT-proposal frames
with expression membership targets and64TAOtrain Base-only GT-region frames
with category targets. Clearly separate teacher proposals/history from native
MOT rollout; never present GT boxes as evaluation predictions. No NovelGT
supervision and no non-exhaustive TAO unmatched-as-negative labeling.
Use original real OneJev frozen token embedding means for text; add shared
learned expression/category heads to the same visual/identity/set core.
All1203classes remain available at inference. Balanced1200step seeded joint
stage; unchanged evaluation scopes/metrics and before-after native proposals.
Plan:configs/groundjev/compact_joint_semantics_20261014.json.


### Lost original trainer and exact-protocol recovery

Authoritative/proc audit finds no original trainer or model services. FullV3
and Compact full-video owners have terminal success receipts. Original
OneJev500has only275last log,250BF16weight checkpoint, no final/done or
optimizer/FP32master/RNG state. External termination cause is not established.
Do not call this complete or silently reset Adam from250. Preserve that
intermediate; replay the unfinished original500step unit from its original
100step initialization with unchanged seed/corpus/lr/schedule/steps. A small
wrapper extends official checkpoint handling only: atomic FP32masters,
Adam moments, scheduler, RNG and exact planned-step cursor. Single latest
training state is a checkpoint, not a feature cache. Original loss/rendering/
row plans/native fork checks remain unchanged. New config:
configs/groundjev/onejev_own_history_v3_exact500_replay_stateful.json.

### Completed joint semantic controls and full-video evidence

Joint1200 truly trained1,146,511 shared parameters with exact3task reload.
Pack149nativeMOT+45RMOT+62TAO teacher groups,112declaredteacherframes,
5emptyframeswithoutregionexamples. TeacherGT/idealupdate proxies remain
explicit. Same candidates/nativeROI and decodedframes validated for all four
short scopes. MOT HOTA66.911->71.556%, butIDSW1->3. SingleRMOT8.932->0%,
multi0->0%; TAOBaseTETA14.013->39.272%,ClsA0->58.286%,AssocA21.350->17.971%.
Reject a unified-success claim. Posthoc evaluation finds all6single/all8multi
GT-covered candidate observations have membership below.008 and no emitted
coveredboxes. Keep threshold.5, allGT/frames/negative results unchanged.
Hypothesis: limited-fit teacher region supervision and frozen token means do
not transfer expression membership to native validation proposals; covered
regions show semantic suppression, not an absence of proposals. Preserve
stage as an ablation, not production promotion. Next training must use broader
legal fit/native proposal supervision and verified query sensitivity.

Full1203DanceTrack Compact300/setGT/MLP and freshFullV3 controls are terminal
and validated. Original same-candidate HOTA20.157/5.977/11.197% vsSimple8.530%
andFullV1 2.145%; single-video/seed only. FreshV3 HOTA2.389% vs its exactSimple
8.257%, detector-inclusive uninterrupted E2EFPS.118560 excluding model/server
initialization; do not compare fresh responses toV1 as identical candidates.
Register/execute joint1200 retention on all1203sameCompact300candidate+ROI
rows, unchangedGT/TrackEval and rejectiondenominator, onefiniteowner GPU7.

Stateful original500protocol replay latest returned50/500 onGPU4, not old250
exactAdam resumption. Resume-state functional probe nextloss/params difference0.
Old250/last275/interruption retained; completion and modelgain remain unproved.

Training-fit membership96/96correct(45positive/51negative). Swapping other
fitexpression vectors changes probability mean.10595/max.99159; this is
input sensitivity only, not a semantic correctness score. Native validation
covered positives remain suppressed. Hypothesis strengthened: limited-fit
teacher/domain/query generalization rather than failure to update head weights.

Joint1200full1203retention terminal and validated: HOTA.22103185,IDF1.21413276,
IDSW346,FP1653,FN1485 vsCompact300.20156623/.17140097/588/2129/951.
All1203candidate+ROIrows match, video1203decoded, all27rejectedframes kept.
No GT/parser/metric/candidate changes. Keep as positive complete-video MOT
retention plus negative RMOT/OVassociation ablation; not unified-success.
GPU7owner exited. ReplayFPS22.747excludes detector/ROI costs.

### RMOT fit breadth and native own-history intervention (registered)

Current membership fits96teacher regions but suppresses every covered native
validation target. Register72frames from all9fit-v2 videos, exact18primary/
counter expressions. Compare broader teacher curriculum with same broadened
teacher+actual nativeGroundproposals/uncorrectedCompact ownhistory; original
MOT/OVgroups, initialization,seed,optimizer,1200steps and allvalidation protocols
remain fixed. Alternate-expression supervised snapshots add query contrast;
no alternate detector request or actor update is claimed. AllGT mappings and
quality proxies remain outside model inputs. Never replace a rejected native
proposal frame byGT. Plan:configs/groundjev/rmot_native72_20261014.json.

Native72owner has confirmed owned original Ground API readiness onGPU6
and is building72fit frames through actualCompact own histories onGPU7.
Existing conservative feature/crop/readable-result bytes6,288,115,739(5.856GiB),
models/optimizer/videos excluded; newtensorserialization checkedbeforewrite.
Empty-candidate lifecycle check fromactualnative train history31slots passes
finite loss3.8286855,lifegradient12.4010; no droppedemptyframe rule needed.

### Original Ground dense/referring retention data preparation

Original nine dense/referring benchmark inputs are absent locally. Official
GroundAnything README links Skywalker0410/Grounding-EvalData. Direct no-proxy
HF API succeeds; pin772e34bb1505053ed4857bba4405ee7b308bbfb6. Fetch only
complete archives/annotations needed by4dense+5referring task entries, not
GUI/OCR/etc. Rawdata/model checkpoints are outside25GiBfeaturebudget.
Parallel3downloads, noGPU. Original full30helper requires allotherdata, so
reuse its verified unmodified archive extraction/path functions and explicitly
label this a nine-task data subset. No evaluation/retention score is claimed
until original unmodified scorer runs on all declared examples.

### Broad fit/native own-history outcomes and CUDA recovery

Both1200additional-step models genuinely trained fromjoint1200, all3task
reload differences0. Same params/initialization/seed/lr/taskcycle; not equal
training-budget/parameter comparison againstoldCompact300. Broadteacher:
MOT4HOTA71.359%,singleRMO20.921%,multi17.740%,TAOBaseTETA46.279/ClsA67.0%.
Matched teacher+native:63.077/0/14.846/56.733/100.0%. Allpairedcandidates/ROIs
and videos validated. Native data165candidates/49GTaligned,8rejectedframes,
all72retained;28contaminatedactorhistoryobservations; noGTstate repair.
Native mixture restores6/6covered singlequeryprobabilities below.5(.362-.498),
versusteacher2/6. Keep threshold.5. Hypothesis: weak native proposal support
and negative-heavy own-history/counter samples shift query decisions; mixing
not universally beneficial. Preserve bothmethods/negative transfer, no promotion.

Full1203GT/candidate/ROI/video controls: teacherHOTA.2040093/IDF.2383171,
IDSW151/FP615/FN2276;nativeHOTA.2631573/IDF.3063294,165/579/1962.
Native run failedCUDAunspecifiedlaunch at799, preserved798rows/memory39.9.
NVMLECC0/recoveryNone; privilegedkernel unavailable, externalcauseunknown.
FreshsameGPU7 actual799probe finite; CPU/GPUprob delta.000115(referenceonly).
No reset ormodelmath/data change. Strictsource798prefixreplayexact, resume799,
all1203candidate/ROIrows anddecodedvideoframes validated,all27rejections kept.
Originalfailureowner retained; separateterminalrecoveryreceipt. E2Enull.

Native Ground original retentioninputs direct3parallel downloads complete:
6.929GB archives, all4dense+5ref canonicaltasks, 91,124rows. Officialhelper
hash/extraction unchanged. Originalevaluator full/limitnull dry-run reports
no missinginputs. Initialrecipe-path resolution and Pythonvenv-symlink mistakes
fixed locally; no metric/source modifications. Benchmark scores still pending.

OneJevstatefuloriginal500replay reached125/500 with500examples, actualcomplete
FP32/Adam/scheduler/RNG/cursor checkpoint9,029,140,249bytes. Genuineintermediate
125BF16MMcheckpoint reload is executing onGPU7; notformal500completion.

ActualOne125MMreload terminal:474weights, BF16changed2,306,395/2,035,222/
1,627,638elements in threeMLPprobes vsinit100; independent nativeMMreload
probabilitydifference0(autoFork unchanged). Onlyintermediatecorrectness, no
formal500or trackinggain/exactFullFP32nextstep claim. GPU7probe exited.

### Native Ground formal200 tracking adaptation (registered)

Advance3step projector proof to200seed42steps on32dense train images plus
40fit-v2KITTIimages with2expressions each (112native SFT rows/56pairs).
Alternate dense/ref; allpairs visited. Preflight proves entireselected target
sets grid-representable, not clipped: MOT20window has69out-of-grid targets
and4otherKITTIwindows haveviolations, so theyareoutside thisnewregistered
native-token corpus, not droppedfromoldprotocols/evaluation. Frozen native
vision/language, trainoriginal27Mprojector FP32; officialMDM.5/causal.5 plus
.1GTidentity/.1actualnative-proposalidentity contrastive. Native boxes come
fromcompletedoriginal decoderrollouts, GTonlylabelalignment; adaptedlive
decoderrollout notclaimed. CompleteAdam/RNG/cursorcheckpoint every25steps.
Fulloriginaldense/ref9 inputs are ready; adaptation benefit remains unproved
until original unmodified before/after scoring. Newplan native_ground_tracking200_20261014.json.

Native200v1all112rows/56pairs fit native collator; actual0negative ref rows
(so absent-target supervision remains a later required stage). Firststep1
completed nativeMLM+teacher/native temporal gradients,27,267,584parameters;
step2cleanBF16attention cuBLAS internal failure onGPU7 before next update.
No validoptimizer/RNG checkpoint yet. Preserve v1failure/step1log. New GPU6
executionvariant holds actualdata/seed42/200steps/loss/optimizer unchanged,
adds checkpoint atstep1 toexisting25interval. SameA800architecture, noGPU
reset orforeignjob mutation. Externalrootcause notestablished; verify actual
secondbatch in freshGPU6context while replayingunfinished initialunit.

GPU6same protocol progressed3steps then failedcuBLAS;1fullstate saved.
Sync nativeFlexAttention beforecleanattention diagnostic resumedexact1state,
reached20steps then exposedillegalmemoryaccess immediatelyafterNoisyFlex
(step21), so clean cuBLAS was downstream/asynchronous symptom. NoGPUreset,
no samples/masks/seeds/loss changes. Native trainingprofile pinsTorch2.10,
whereassmoke usedServeTorch2.9.1. PrepareindependentPy312Torch2.10cu128
(A800/driver570-compatible CUDA family) /nativeTF5.7/Triton3.6; no serving/
One env edits. Resume native200 fromvalid1withall optimizer/RNG/cursor.
Any runtime numerical differences are to be recorded, not silently ignored.

NativeCPUevalenvironment officialsetup completed (Torch2.5.1CPU, shipped
lmms snapshot; serving/training environments unchanged). Startoneowned
originalGroundservice GPU7 + originalfull9canonicalbenchmark evaluator
CPU-only, limitnull,91,124declaredrows. No8sample-smoke orretentiongain claim.
Nativeevalresponsecache is originalscorer output, not U2candidatefeatures.

Originalfull9preflight/service ready; all9clients failedbeforefirstinference
while constructingHTTPproxy mappings in SDK/httpx0.23. Task-inherited proxy
configuration conflictswithlocalAPIclient; cleartasklocalHTTP/HTTPS/ALLproxy
vars only, preserveglobalenv/source/metric. Fresh v2runid retainsold9failure
logs/caches; all9unfinishedcanonicaltasks re-executed, no sample/seed changes.

IsolatedTorch2.10cu128/TF5.7/Triton3.6 installed; missingSciPy preflight
fixed withoutchangingdata. Actualstate1 resumed, validthrough57 with55full
state. Earlierbad-runtimegradientnorm at20=387471 vsnew=13.49, numerical
unsafe-runtimebehavior retained, notclaimed equivalent. StillillegalnoisyFlex
access at58, so runtimeupgrade alone is insufficient. Concrete-shape compile
specializes SAMEoriginalnative_flex_attention_kernel/masks/GQA; no score/
loss/model changes, no dense-reference fallback. Resume55in freshv6out.
Recompileguards raisedonlyforthefinite declaredvariable-shape corpus.

### Native Ground200 concrete-shape failure and targeted diagnosis

Concrete-shape v6 completed185updates and saved full FP32/Adam/allRNG/cursor185, then failed before186update with CUDA illegal access. Static shapes alone are insufficient; no metrics or samples are changed. Previous output handle is absent and owner terminal. Next action: replay from185 with CUDA_LAUNCH_BLOCKING and a persisted Q/K/V boundary plus full traceback, same operator/masks/data/seed/loss. This localizes the actual first failing kernel before choosing a repair; failing attempts remain preserved. Full original dense/ref9 baseline handle73408 was confirmed live, so it is not restarted.

### Native Ground200 resumed completion and full original-protocol retention

v7 synchronous replay from185 completed186–200, keeping exact native operator,
masks, seed42, data/loss/optimizer/schedule. Asynchronous kernel root cause is
still unestablished; all v1–v6 failures remain. Complete checkpoint chain:
v2step1, v5steps2–55, v6steps56–185, v7steps186–200. All200loss/gradient rows
finite;100dense+100ref, all56pairs visited. Native200reload maximum delta0;
27,267,353FP32 and16,204,590BF16 elements changed. Exported complete569tensor
9,687,419,704byte model; only five projector tensors replaced. These are
training/execution results, not benchmark or tracking accuracy improvements.

Original full9baseline handle73408 remains live. New adapted full9owner uses
GPU6/API8102 and separate execution rendezvous ports22345+, preserving all
canonical tasks, full rows, original scoring and native decode profiles. No
benchmark GT training or limit reductions. Four project GPUs maximum; other
projects are untouched. Real absent-target40rows remain a separate next stage.

Fresh nativeGround200 MOT4/RMOTsingle8/RMOTmulti12 before/after controls also launched through the two existing owned retention services, same Simple associator/parser/tracking_dense/seed42, no new GPU service or benchmark-limit changes. This48frame functional comparison is separate from original full91,124-row dense/ref9 scoring. All registered empty/rejected frames remain.

### Ground200 short-control negative transfer and registered absence120

All48frames and six rendered videos validated. Same fresh Simple associator:
MOT4 HOTA75.2498->84.5532%, IDSW2->0,FP16->2,FN24->13.
RMO single8 HOTA26.8665->25.2769%, empty-frame outputs3->8; multi12
HOTA24.7800->23.9589%, IDSW0->2. All empty/rejected frames retained, official
GT/scorers unchanged. These are single-seed short scopes, not full datasets.
Hypothesis: missing absent-target supervision contributes to referring false
positives; not established as the sole cause. Keep original200 and bothfull9
retention owners unchanged. Register a separate additional120step curriculum:
initialize exactFP32 projector200, explicitlyresetAdamW/lr1e-6, seed42;
all112existing rows+40previouslyregistered actualfit absences=152rows/76pairs,
all76visited. Same native loss/teacher and native identity losses. No synthetic
emptyimages, no evaluation labels in training. Verify first real native noisy
attention forward/QKV gradients for dense, positive referring and absence
against PyTorch eager mathematical reference; diagnostics consume no RNG.
Idle ownedROIservice5 stopped after no established clients; all consumers were
terminal. GPU5 is reused for this declared training, no other project stopped.

Absence120v1 preflight verifies152rows/76pairs/40realabsences, but stops beforefirstupdate: detached reference-probe backward inherits native activation-checkpoint saved-tensor hooks and recursively re-enters itself (_StopRecomputationError). This is a diagnostic integration failure, not a measured attention discrepancy. Small repair: isolate only detached diagnostic saved tensors with identity pack/unpack hooks; production model checkpointing and forward remain unchanged. Preserve failedv1. Freshv2 replays unfinished stage from exactTrackProj200, identical seed/data/loss/criterion.

Actual v2 native attention reference checks pass for first real dense/positive-ref/absent-ref layer: maxBF16 output delta.005859/.003906/.005859; Q-gradient relativeL2.043767/.033754/.036040, K/V below.007. Criteriaunchanged(output rtol/atol.02; QKVrelativeL2<.05), no diagnosticRNG consumption. This targeted check covers three real input types, not every layer/step. Training remains owned by the same v2blocking command.

### Real absence120 completion and unchanged native200/320 tracking comparison

Additional120 completed: all76pairs/152rows visited, including20actualabsent
pairs/40fit-negative rows; finite losses/gradients, FP32changed27,266,797,
BF16changed8,256,449 versus initialization200, native reload delta0, peak
allocated18,414,616,064bytes. Complete200model/full9owners retained.
Export full569tensor model320 (explicit200+120; five projector tensors only).
Next test: fresh200-versus320 native MOT4/RMO8/RMO12, same Simple associator,
parser/tracking_dense/seed42/GT/empty+rejected frames. New ownedGPU5/API8103
service only, auto-clean at terminal; existing200/API8102 onGPU6 untouched.
Short-scope result is not fulloriginal9retention or a completeRMOTclaim.

### Absence320 short negative result and native sampling-counter confound

Fresh shared200vs320 controls finish48frames/sixvideos. MOT HOTA83.737->83.402,
RMO single18.406->20.046, multi23.496->24.817%. Empty outputs unchanged7and3;
IDSWsingle1->2/multi2->4. No successful absence correction is established.
Originalnativecode inspection finds sampling seed depends on monotonically
incremented request_index, not just suppliedsampling_seed42. Shared full9
service consumes many intervening requests, so preliminary checkpoint
comparisons do not pair effective stochastic seeds. Preserve their real scores,
but do not attribute small changes solely to adaptation. New explicitly
registered control: all original/200/200repeat/320 on fresh isolatedGPU5
services, identical ordered24requests each, native source/math/seed42/scorers
unchanged. Require repeated200raw wire/candidates/outputs identical. Keep
both originalfull9owners unaffected. This is a small controlled protocol
verification, not a repetition of full91k benchmarks or threshold tuning.

Registered one finite blocking OneJev500completion owner: wait exactlive originaltrainer and verify terminal500receipt, actual474weight nativeMM2reload, then same candidate/GT/core4way Simple/pretrained/init100/500 controls acrossMOT/RMOT/TAOBase. GPU4 reused only after originaltraining exits. No original500restart or math/seed/corpus change. No detector E2E or Novelclaim from replay/Base-only clip. Fullvideo500before/after remains subsequent required work.

Isolated96v1 original24frame unit completed, then next immediate replica bind failedEADDRINUSE after ownedserverparentexit. Later targetedlistener/bind check shows portfree; sourceparentwait does not guarantee immediate TCPport reuse. Small execution fix: SO_REUSEADDR probe plus internal30second port-release wait, no model/math/seed changes or other process signals. Preserve originalfailedowner; newv2reuses complete original24rows/metrics and resumes only200/200repeat/320fresh replicas.

### Isolated native96 completed: confirmed positiveMOT/negativeRMOT transfer

All96frames/12videos validated;200repeat rawwire/candidates/causaloutputs
identical on24frames. Original/200/320 HOTA: MOT75.0687/83.4673/83.4016,
single30.3853/24.8762/20.0457, multi22.6643/23.6754/24.8173%. Singleempty
outputs3/7/7; multiple2/3/3. MultiIDSW0/2/4 and AssocA41.678/33.457/30.669.
Keep adaptedcheckpoints as research controls, do not promote320as unified
foundation. No successful absencecorrection. FixedSimple means observedMOT
benefit is front-end adaptation, not a claimedJEVassociation gain. Shared
service metrics remain preserved but not causal checkpoint evidence. Both
full9owners unaffected; isolatedservices all stopped. Next major stage is
actual500nativeMMreload and exact-proposal Fullbefore/after/multitask controls;
completion owner already waits verifiedcurrenttrainer, no restart/sample skip.

Originalfull9 v2terminalpartial: COCO allcanonical rows completed; other8clients HTTP503 queuefull. Native service defaultsmaxrunning1/maxqueued2 versus9clients*concurrency2. Small execution repair: task-local boundedqueue32, unchangedsource/scorers/data/limits/concurrency/genprofiles/seed42. Actual18parallel HTTPregression in a separate ownedservice, then freshoriginalresume. NativeCLIrejectsreusedoutput; wrapper invokes same nativefunctions in a freshresultroot, copies all-sample completedCOCO result and SQLitebackup partialcaches, spawns onlyunfinished8tasks. Preserve originalfailure/results; no exact stochasticcounter-resume or causalretentiongain claim.

Confirmed sameHTTP503 failure in8of9adapted200tasklogs. Verified nativeevaluator1053747 command/cwd and requested its supported cooperativeSIGTERM; originalowner97074 then failedterminal and stopped only its ownedservice. Preserve allpartialcaches/failedlogs/nativecancelledsummary; resumeall9unfinished after queue32regression, same200checkpoint/parser/scorer/data/seed42. No unrelatedprocess signals. Avoid waitinghours for a doomedpartialnine-task unit.

Queue32regressionv1timeout: newserverlog actuallyEADDRINUSE, while HTTPmodelsprobe accepted oldorphanAPI775339/1051641 after their launchers exited. Verified commands/cwd/model and originalisolatedPGIDs775337/1051638; priorTERM/grace hadfailed, targetedSIGKILLonlytheseownedgroups. Rootcleanupbug: waitlauncher returns before orphanAPI exits. Newhelper verifies/waits allownedPGIDmembers, forcekillsafter20secondfailedTERM, checksnonealive; add portbindguard to regression and recordgroupPID. Preserve oldfailedregression and recoverygate; rerun same18client queuecriterion in freshv2, no nativebenchmark protocol changes.

Freshclean-port queue32v2 actual18parallel callbacks allHTTP200; fullownedgroup1339199 verifiedemptyaftercleanup, old775337/1051638 verifiedempty. Recoveryv2 hadlocalargparseNamespace overwrittenbyprocargvlist beforeanychildrenlaunch; repairloopvariablename, preservefailure, v3runs samevalidregression gate/no retest or changedcriterion.

Queue32v2passed all18HTTP200; verifiedfullgroupcleanup and currentoldAPIports released. Full9recoveryv3startsoriginal8remaining plusadapted200all9 with validatedCOCO4952/4952reuse andSQLitecache backups through unchangednativefunctions/scorers. Keep allfailedstates/cache prefixes; no samples skipped/metric changes/base seed changes or exactstochasticcontinuation claim.

### Fixed-checkpoint full-video memory intervention (registered)

Read actualCompact/tracker: neuralhistory merges immutableanchor/recent/long
native2560features, learnedupdate>=.5, 3secondlifespan with learnedtermination.
Hypotheses: persistentanchors/gating improve naturalgap recovery, while always
updates can contaminate;30secondretention may recover longer gaps but increase
competition. Keep exactbroadnative1200checkpoint/1203originalV1candidates/
realnativeROIs/GT/scorer/thresholds. Six frozenmodel inferenceinterventions:
referenceanchor+gated3s, immediatepreviousframe noLostmemory/recentinput,
latest-observationinput withnormal3sretention, unconditionalEMA(.1)input,
anchor+alwaysreplace3s, anchor+gated30s. EMA is a convex mixture of realfeatures,
not a synthetic image encoder; lastreliabletimestamp still uses learnedreliability.
These are input/policy interventions, not independently retrained competitors.
No nativeDLMquery/newcache; reuse790,667,425byte sharedfullvideo featurelog.
Only featurearrays omitted from new logs, retain source reference/allboxes/
labels/scores/decisions; actualNNinputs unchanged. Evaluate allframes/rejected
frames with originalTrackEval and posthocfixed-birth-anchor recovery/NoMatch
counts (never inputGT/replaceofficialmetrics). Actualnativefeature2frame EMA
memorysave/load regression precedes fullruns; defaultprediction retention also
checked against preserved existing baseline, recording any numerical variance.

Actualnative two-frame EMAprobe passed:2560wide features, one matchedpriortrack expectedconvexupdate, originalanchor unchanged, forcedwrites leave lastreliabletime honest, restorednextoutputs/state exact. Sixfull1203frame memoryowner42123launchedGPU5; featurearrays omitted only in duplicatedlogs, exactsharedsource retained for inference/rehydration. Originalfull9owner39013 and One500/500completion owners remain active without restart.

Prior fullvideo posthoc: GTobservations4586/nativeone-to-one matches4341,
existingbirth-anchor associations575/1768, modelcorrectbeforematching651,
globalcompetition changes94correct->wrong and18wrong->correct. MatchedGT
NO_MATCH2100predictions with0option-missinggoldpositives under declared
birth-anchor rule; this diagnostic excludesunmatched/ignoredregions and is
notofficialNoMatchtruth beyond its rule. Candidate-gap returns194, ≤1s recovery
9/177,1–3s0/4, >3ssupport0/null. This video cannot establish long>3srecovery;
keep30sintervention asretention/competition sensitivity, not invented support.

Novel evaluation breadth: select lowestexistingcompletevalidationvideo with
bothofficialBase andNovel: TAOvideo63,40annotationframes/160GTboxes (Base120,
Novel40peeler), allpresentwithcorrectimagegeometry. Fullfixed1203vocabulary
unchanged, GTonlyscopeinventory/evaluation, noNoveltraininglabels. Register
40fullannotation-frame online nativeDLM520classqueries plus realnativeROIs,
then fixedproposalSimple/sharedCompact controls with unmodifiedTETA. Explicit
per-queryrejection contributesempty for thatquery only, concatenates remaining
original-parserboxes, noNMS/topK/GTselection; matches existing scored-query
control policy, not old whole-frame-rejection scores. One blocking owner waits
currentGPU5memoryjobexit beforeowning temporaryDLM/feature services; stops
bothcompletegroups beforecorecontrols. Timestampannotationsteps, notphysical
seconds/FPS. No rawdatasets/modelweights uploaded or all-datasetfeaturecache.

Prepare explicit nativeOne200curriculum for proposal-domain mismatch: retainall20881existingteacher-weighted rows; native5661unique realDLM/uncorrectedown-history rows identityweight4=>9213, deterministiccyclicexpand to20881, total41762equal-source rows. No eval/Novellabels, no state correction or omittednativeunit. Register matchedteacher-only200 andnative-mixture200 fromsame eventualOne500FP32 initialization, same lr2e-6/seed0/budget/officialloss. These are additional200 stages, not changes to current500. Run officialCPUplan-only with existing100tokenizer/config (same released tokenizer/visiongeometry) before anytraining; reject anydroppedrow. Stage launch requires actual500completion/evaluation and correctFP32 initialization.

Beforetraining, strengthen domaincomparison: preparedv1eligible familyratios differed. Preserveunusedv1; v2pairs eachteacherrow withnative row ofidenticalquestionfamily (9840identity/2460update/2460reliability/5951lifecycle/170withheld per source), equal50/50sources41762rows andall5661unique native rows. No label/dataframe or metricchange. Preflightv1stilldiagnoses allunique originals, newv2willget its own officialplan preflight; current500untouched.

Realfull375stage initialization probe passed: all320trainableparameter tensors/752393024parameters copiedfromactualFP32trainingstate withmaxdelta0;498430914elements differfromBF16inference-export-derived initialization. CPUonly, no old375weights/state/live500modification. Newstageinitializerexplicitreset is distinctfrom exactresume. v1officialplan-only41762rows max26028/cap32768, zero drops; exactfamilymatchedv2officialpreflight nowrunning separately. Stage200launch remains pending real500completion/controlreview, no blind extra training started.

### Full7218frame memory interventions complete

Allsix1203frame traces/videos/GT/scorers/samecandidate+ROI sources validated;
all27native rejectionframes eachretain denominator. Freshdefault matches
preservedGPU7/resumed baseline exactly, maxprobdelta0. OfficialHOTA/IDF1:
referencegated3=26.3157/30.6329%; frameonly11.4408/8.4615;
recentobs3=16.3943/18.2023; EMAalways3=14.6426/16.8764;
anchoralways3=24.2449/29.8179; anchorgated30=18.5523/19.8490.
ReferenceIDSW165 vsalways154, but referenceFN1962 vs2143 andFP579vs604;
no claim that gating improves everymetric. Keep default3s; longerretention
worsens thisvideo and haszero >3snaturalrecovery support, so do not claim
long-term recovery. Anchor/update evidence is frozenmodel interventions only,
not optimalEMA/newarchitecture superiority. MatchedGTNoMatch excessive;
nextnativeOne data-domain intervention remains pendingreal500results.

Selectedv2family-matched native mixture officialCPUpreflight passed41762rows, max26028/cap32768, zero drops andPLAN_OK,119seconds,723724879tokens. Alloriginalunique native rows retained. RealFP32stage initialization proved on375fullmodel; no additional200training started before500result review.

TAO63v1 failedbeforefirstnativequery: stream call omitted requiredexpression parameter (OVMOTNone). Bothownedservicegroups stoppedverified. SmallfixexplicitNone; preservev1empty prefix/failure; v2replays entireunfinished40frame unit withidenticalvocab/parser/scorer/seed/scope, actualfirstquery is targeted regression beforeremaining519requests.

Adapted200full9 child terminal separately: schedulerGPUlaunchfailure at00:04:41 inrawforward end sync, all9clientconnection failures downstream. Queue32isnotcurrentroot. No completedtask metrics; caches/logspreserved. Firstactionablediagnostic: task-localCUDA_LAUNCH_BLOCKING1 in freshGPU6service, resume9fromlatestv2responsecaches withsameoperator/model/data/seed/scorers/concurrency/queue. RootGPUkernel stillunknown, no reset/driver/globalenv/foreignprocess changes. Originalnative8taskowner and parent39013 remain live; newadaptedrecovery has its ownowner without duplicateoldservice.

2026-10-15 publication: all39files for completed six full1203 memory interventions, official metrics, predictions, SVG and matched real-native One continuation preparation remotely verified: GitHub085125569af96d47b89f2fd269c6501f02b0b28f tree3ffa25f12188b6190992de0baad740735fc9c43c matcheslocal034bc056e812d45f66eea34d54b1ca70cdb817e8. Existingblockinghandles confirmedOne500450/500, TAOBaseNovel63v2live, adapted200syncv3live. Retain failedadaptedv2 and originalfull9owner; no additional200launch or globalGPUreset.

Hypothesis before CPU RMOT diagnostic: validation membership failures may reflect weak expression response or proposal/history domain mismatch despite fitted training examples. Keep three existing semantic checkpoints, exact20registerednative validationframes/realROI vectors and unchanged0.5threshold; causal primary-query actor, compare None/counter expression only on captured same core input, allGT posthoc. Quantify fit membership and real validation query response plus inverse-aligned candidate/history permutations. No new training, synthetic features, altered official metrics or GPU consumption.

CPU real-query diagnostic complete on3existing semantic Compact checkpoints,20registerednativevalidationframes/27actualcandidateobservations. AllCPU replay boxes/IDs identical to recordedGPU outputs afterexcluding floatingprobabilities; maxmembershipGPUCPUdelta7.152557373046875e-5. Within captured candidategroups/shortlists inverse-aligned6headpermutationerror<=6.556510925292969e-7, not fullpipelinecrossgroupproof. Broadnative redcoatwoman6primary-onlyGTcovered observations havecorrectquerymean.432221 vs arrangingautomobilescounter.772220,0/6correctabove.5 vs6/6wrongcounterabove.5. Broadteacher same6wrongcounteralso6/6; language response is present butwronglydirected, not just orderbias. Fitoriginaljoint45positive/51negative perfect; broadnative282/316positive and460/588negative correct atunchangedhalf. No thresholdtuning/GTfeedback/officialmetricreplacement. FirstCPUv1diagnostic compared floatingqueryprobability in outputgeometry; v2 excludes both association/queryprobabilities and separately reports numericaldelta, preservesv1.

ActualOne SFTscope audit: original50020881rows andpreparednative41762rows ALLstate.task=mot,personvocabulary,noexpressions. Familymix is not tracking-taskmix. Preserveoriginal500 and matchedMOT-domaincontinuationcontrol, but these cannot prove unifiedFullsemantic adaptation. Nextimplementation: opt-in genuine nativeOneJev Noul query membership, defaultbehavior unchanged; export allactualDLM candidates for144primary/counterquery contexts across72legalfitRMOTframes withrealcropmedia, GTonlyoutsideinput and instantaneousquerysupervisionexplicit. Run officialCPU plan-only before anynewtrain; actualnativekernelruntime/semantictraining remainsrequired.

Fullsemantic preparation retained: genuineOneJev Noul membership opt-in implemented; all346nativefitRMOTrows officialCPUplanOK,max2293,0drops,457637tokens. v1data-rootfailure beforefirstrow preserved; v2usesregisteredreleasepath, all72frames/144contexts/all165nativecandidates twice/58positivemember targets retained plus16presencequestions on8empty/rejectedframes. Basecategory430questions from215GTteacher crops/64registeredtrainframes/9Baseclasses, all1203vocabchoices retained, noNovelGT orunknownnegativesupervision, explicitconditionalrootteacherforcing; officialCPUplanOK,max4233,0drops,1283259tokens. Sharedcategorybuilders actualsix-object root/leafrequest+mediapixels exactlyequalto8b880source undercapture-onlyrouting, notnativeinference. One blockingnativeOne500Noul4frameprobe owner waitsverifiedexistingcontrollerPID1209639/startticks thenrequiresitscomplete500controlsreceipt beforeGPU4reuse; session50119. No additionaltraining launched; actualOnekernelprobe/multitaskSFT and completevalidation remainrequired.

Plan for genuine sharedFull semantic adaptation: preserve original500MOTstage and existingMOT-only200study as separate protocols, neither is unifiedSFT. Prepare explicitadditional600matched arms from actual500FP32masters: teacher-onlyMOT vs50/50teacher/nativeMOT, identical common346nativefitRMOTmembership/presence and430trainBasecategoryroot/leafrows. EqualeligibleMOT/RMOT/OVMOTbucket counts bydeclaredcycling, everyexistingeligibleMOTrow andsemanticrow retained; exactMOTquestionfamilycounts matched. Sameofficialsampler/loss/seed0/lr2e-6/4rowsperstep/frozenvision/752393024trainablelanguage params/32768context; budgetmatchedoptimizersteps, notidenticaltokenFLOPsexposures. NativeRMOinstantaneous(noOne500historyclaim) andBaseGTcropteacherstage explicit; noNovel/validationGTtraining. Computeconservative25GiBfeaturebudget beforegeneration, officialCPUplan-only all125286rowsperarm; actualtraining launch awaits500matchedcontrolsreview and nativeNoulprobe. New600stage is registeredresearch, not silent alteration of existing500/200protocol.

Unified600v1 bothofficialplans completed125286each,zeroDropped,max22817teacher/26028native. Actual600prefixes2400rows haveMOT842/847,RMO799/801,OV759/752; only686/1558teachersemanticrepetitionIDs shared. Native length sort changes actualsemantic exposure despiteequaleligiblebuckets. Beforeanytraining registerv2: adjacentteacherMOTduplication aligns everyMOTquestionfamily withnative mixture; explicitpaired-index-shuffle-v1 underseed0 makes botharms selectidenticalrowindices/stepsemanticquestions/families, retainsallrows/nativebatchmax1/native4rowstep/loss/AMP. Optionalnew-stageflag only; defaultold500/200sampler and checkpointprotocol unchanged. V2checkpointprotocol recordssampler onresume. Preservev1data/plans; count bothversions conservativelyunder25GiB.

ActualOne500complete at2026-10-15 01:30:56:500updates/2000examples/33469.233s, valid9,029,140,249byte fullFP32AdamRNGcursor state. RealnativeMMtworeloads maxprobabilitydelta0, all474exportentries retained. CompleteCPU500masterinit320/752393024 restored0,498431025F32elements retainedvsBF16init; originaldefaults checkpointprotocol exactlycompatible. Fullfrozenvisual153tensors unchanged;BF16uniquechanged333632768afterdeduplicatingtiedembedding+lmhead (254279680duplicatedexportelements). Actualunique0.852985920Bparams incl0.100592896Bfrozenvision,0.752393024Blanguage. Matched16controller stillfinishingTAOunits; completeMOT/RMO review mixed:pretrained/100/500MOT HOTA.343944/.197395/.308680 vsSimple.748813,IDSW47/0/57 vs1;singleRMOp500.248862 vsoriginal0/Simple.242284;multi.214246 vsoriginal.222720/Simple.243151. No promotedmodel ormultiTaskSuccessclaim.

Unified600v2all125286rowsperarm passnativeCPUplan,0dropped,max22817teacher/26028native. Bothactual600prefixes2400rows exactly811MOT/749RMO/840OV; allper-stepquestionfamilies and1589semanticrepetitionIDs exact. DirectactualJSONpayloadcomparison all83524eligiblecommonsemanticrows (1589selected) byte-identical inclstate/question/media/target/provenance. Prefixtrainingtokens15774003teacher vs17216498native, notFLOPmatched. Conservativeexisting+newv2text/features16,289,535,076bytes<25GiB, v1data/failedcomparisonproofretained. Registeredfinitepaired600owner waits verifiednativeNoulprobe/500controls then startsGP4teacher;GP5native onlyafterexactownedTAOparentexit+emptyservicegroups, independentTAOfailuredoesnotglobalNoGo. Bothreadimmutable500masters/newstatepaths; streamactualchildlogs; nativeofficialcontrolsvs500andcompletevideoE2E remainrequiredaftertraining.

NativeOne500Noulprobe actualfourframescompleted (150..153, candidatecounts3/1/0/2) andsixactualprobabilities.172542..176372; ownerfailedonlyKeyErrorGT_read becausevalidatorlookedinsideOneJevmetadata ratherthanexistingrun-levelgt_read_by_runtime=False/future_frames_read_by_runtime=False. Preservev1nativeoutputs/failure and prelaunchpairedownerv2whichcorrectlyfailedbeforeanytraining. Fixvalidatorfieldlocation, add futuremetadataGT_read consistency, freshv2validation reusesentirecompleted4frame run withoutnewinference/skippedframes orweakenedcriteria; no accuracyclaim.

Original500matchedcontroller nowterminal0, all16officialcontrols/112decodedframes validated. TAOBase4pretrained13.243/10011.732/50011.981TETA, allClsA0 underlegacylabelpolicy; no consistentfoundationgain. NativeNoulv2validated all4actualcompletedframes/6probabilities via correctrun-levelGT/futureflags, no inference rerun or criterionweakening. Resource/GTvalidator firstGPU-memoryfield lookup assumedSimple hadlearnedGPUallocation; fixedSimpleunmeasured value=null, not0, targetedall16videos rerunpasses. Pairedownerv3actualteacherprocess launchedGPU4 (72945MiBfree atlaunch), actual500FP32copy320/752393024 difference0 andpaired_index_shuffle=True inrealargs.json. Native5arm waits exactliveTAOowner1444656/start37238782 andemptyownedgroups. Blockingpairedowner session33936; no newtrainingpoll oroldU2queue. Cumulativebudget andper-taskofficialbefore/afterevaluation remainpending.

Research plan: current3legalReferKITTIvalidation label scopes0007/0015/0016 have0naturalGTidentitygap>3s; do not infer longrecovery support from this. Audit onlyregisteredMOTvalidation frames for naturalannotationgaps/lowvisibility intervals, nevermaskpredictions/createocclusion. Extendposthoc physicalidentity/memory/NoMatchdiagnostics toRMO(allreleasephysicalGT, querylabels separate) andTAO(nonexhaustiveoriginalGT, unknownFPSannotationunits explicit). RuncompletedrealOne500all16scopes onCPU withoutGTfeedback/officialmetricreplacement; preserveoldMOTdefinitions. Thenregistersame-native-candidate native-semantic500/600 evaluation owner afterpairedtraining ratherthanmixlegacydefault andsemanticflags.

2026-10-15 13:05audit: oldpairedowner1557613 andchildren1557618/1584409 all/procabsent; unifiedhandlesmissing. No capturedPython/math/CUDAexception, logsabruptlystopteacher475(native250). ValidfullAdam/FP32/RNGcursors teacher450/native250 retained. Terminationsignal/origin notrecorded; uptime disproves assumingnewboot. OldchildPopeninheritedownerPGID/stdoutpipe; modifyexecutiononlydetachedsessions/directfilelogs tosurviveownercleanup. Freshdetachedresumeowner preserves originalunsaved475log beforetrimmingmainlogto450, restoresfull450/250 state andsampler unchanged, nooptimizerreset orBF16restart.

NativeTAO63streamcompletedall40frames/520requests:317scored/203originalparserrejectedqueries,35228boxes,1885903934bytes,10850.079sperception+ROI(noE2Eclaim). SimplecontrolfailedstatePNGserializationframe755 after28completecausaldecisionlogrows/27encodedframes. Actualnativebox[92.18018,522.51652,151.07307,523.23624] yields59x0PILcrop fromsubpixelheight. PreserveboxinclFP/noNMS/noGTclipping. Newmemoryv3sentinelexactemptydimensions/mode andloadroundtrip, noinventedpixels; nativecategorysheetmarksno-pixelsforactualemptycrop. Actualframe755probe passes unchangedbox/output count. FreshCPUcontrolowner resumes28validdecisions, all40frames required; sharedCompact CPU sameweights/ROI/source withdevice explicit, noGPU/E2Espeedclaim.

Registered same-semantic-policy before500/teacher600/native600 evaluation on GPU4 after verifiedteacherchildexit. Exact existing MOT4/SingleRMO8/MultiRMO12/TAOBase4 native candidate and frame sequences preflightpassed; unchangedMOT500 reused, newnativeNoul.5 andhierarchical1203category policy applied tobothbefore/after. Actual600MMreload comparesBF16exports against500 and repeatsnativeMMtwice withunchanged1e-6criterion. This12unitshortcomparison doesnotclaimNovel orfullvideos/E2E. Naturalvalidationaudit completed36sequences/647annotation-definedchallenges; selectedcompleteMOT20-05registeredhalfvalidation1659..3315 includes29.76s lowvisibilityinterval, notactualcandidateabsence/recoveryproof. No artificialocclusion, candidatefilter, testGT orofficialmetricchange.

B2 research plan: run originalU2SharedTrackCore unchanged with newlearned2560to256GroundROIbridge on same48legalnativeMOTtrainingframes, originalsixlosses/GTforcing/queryaugmentation andoriginalGTignoreloader. Newseed0training300updates; no existingtrainedU2trackingcheckpoint. CPUonly/noextraGPU/nofeaturecache/newperception/oldU2queue. ExactsameV1DTnativecandidates for4framefunctional memory roundtrip thenfull1203officialTrackEval. Same300updates asCompact butdifferentparams/losses/optimizer/frameexposure/maxage, thereforearchitecturebaseline notisolatedJEVeffect orparameter/FLOPmatch. MOTonly; nofabricatedRMOT/OVMOTcapability.

TAOCPUrecovery Simpleterminalvalidated:40decodedframes/35228nativepredictions/160GT; officialTETA18.640 Base23.3725 Novel9.1737 NovelClsA0. Alloriginal520queryoutcomes preserved. SharedCompactsamecandidateCPUcontrol active; noveldetection/localizationdoesnotmeanNovelrecognition. B2preflightreconstructedexactnativeboxes/features onall48frames, three16frame sequences,45adjacentobservationpairs (includingactualsourcegaps); noallocatedfeaturecache. DetachedB2ownerlaunched withall300updates/fourframefunctional/full1203officialcontrol inoneshellwait.

B2actual300trainingcomplete17.325s CPU,5942795params/4361991changed elements/nativecheckpointreload0. Fourframevideo/eval/memoryroundtrippassed. Fullvideo failed199 after198validframes: actualfinite retainedtrackcoordinates~7.6e19 andvelocity145.9 cause FP32norms inf/inf incenter-distance; allmodelweights/currentfeatures/statevaluesfinite. NotGPUbackend/GT/NaNfeature. Firstcandidatefix alloverflowingnorms altered previouslydefinedframe198pairlogits by.12063676 andwasREJECTED beforevideoresume. Smallacceptedfix recomputesonlyundefinedcenter-distance ratios inFP64, leaves everydefinedFP32distanceexact. Full198frame replay reproducesallpair/new/noMatch/alive logits andIDs exactly(maxdelta0), actual199passesfinite. Savevalidatedstate198; freshv2owner reusesreal300checkpoint/completedfunctional4 andstrictall198sourceprefix, resumes199..1203, no retraining/weight/seed/candidate/threshold/memory-lifetimechange. Originalfailure/rejectedregression retained; no shortened metric scope.

B2v2full1203terminalvalidated: all1203decodedframes/same13664nativecandidates/all27rejectedframes; originalTrackEvalHOTA.0124341105732 IDF1.00625459896983 IDSW250 FP455 FN4191. Completegeometry/order/imageframeagreement andperframeIDuniqueness checked. Strongnegativebaseline(majorGTmisses), noIDSWorJEVcausalitysuccessclaim. Actual17.325secondtraining/5942795params/4361991updates/reload0 retained. Resumed198prefix bypassescorecompute; removedinvalid standaloneassociationFPS metadata (raw26.915 retainedasresumedscopeencodingtimingonly), preservedrawreceipts andreranunmodifiedofficialscorer: allaccuracymetricsexact. NoE2Espeedclaim. Naturalchallenge manifest nowlinksitsalreadyregisteredMOT20protocol forunchangedofficialevaluation; images/GT/scopeunchanged.

Researchplan fullRMOT: twoexistinglegal0007full800frame expression scopes, firstgenuinefreshsharedCompactsinglepass throughoriginalDLM+actualROI+memory+video; include sidearchivewrites inactualE2Etime, remoteinitializationexcludedasinpriorFull. Newoptional run_video currentnativearchive preservesfeaturesonce andallowsgeometryonlylogs; olddefault/replayprotocolunchanged. Exactsame archived candidates thenSimple/pretrainedOne/500/teacher600/native600 nativeNoul.5/v3controls, officialqueryTrackEval andphysicalIDposthoc. SingleGPU5owner waits bothverified600childrenexit+valid600master/done beforeallocation, independentGPU4shortcontrols/nativefull9 unaffected. Budgetnew2GiB acrossbothsources, conservativeprior16.289GB+TAO1.886GB+new2.147GB=20.323GB<25GiB; no duplicatedfeaturesincontrol logs/datasetcache/oldU2queue. Actualnative archive/replay/cropmemory roundtrip targetedregression onrealDT4frameROI source beforequeue. No fullRMOTdataset/CompactFullreasoning/standalonereplayE2Eclaims.

Currentnativearchive regressionpassedactualDT4frames/13nativeROI2560candidates/742680bytes: botharchival and replay geometry/output/fullmemory states exactlyequal, GT/futurereadfalse, replayE2EFPS correctlynull. BothregisteredfullRMOT0007single/multi manifests validateall800images0..799/currentlegalv2validation withoutopeningGT. NewfullRMOTowner preconditionbindsthisactualreport; featurebudgetwithprobe20323665338bytes<25GiB. QueuewaitsGPU5traininghandoff, nochangescurrenttraining/nativefull9/TAOCPU owners.

RawTAO40bothcontrolscomplete: SimpleTETA18.640/AssocA28.618/ClsA20.290 vs sharedCompact14.517/5.1478/30.314. BothNovelClsA0; CompactBaseTETA18.93545/Novel5.679 <Simple23.3725/9.1737. CandidateGTrecall.91875 both, butcorrectphysicalanchor32optioncoverage.156716 (Simple1). No promotion. GTfreeexactcoordinateaudit:35228rawnativecandidates but7343uniquegeometry,27885aliasobservations(79.1558%),max188samebox; actualduplication isactionableroot. Preregister separateproposalpolicycontrol: exactxyxyequality, preserveall35228aliasindices/labels/scores andall520queries/40frames, representativefirstnativeboxwithidenticalROIsasserted; originalrawsource/resultsunchanged. SameSimple/CompactCPUweights/officialTETA, noIoUthreshold/topK/NMS/GTselection andnoassociation-only/JEVcausalclaim. Newrawsourcecap1GiB alongside2GiBRMO sources; cumulativeconservative21397407162bytes<25GiB. NoextraGPU/training/perception.

Researchplan tocompleteexplicit14.5task-sharingmatrix: foursemanticCompactarms MOT/MOT+RMO/MOT+OV/allthree, sameMOT-onlynative300initialstem andseed0newsemanticheads (notjointpretrained), sameexistingbroadnative536pack, same1200CPUFP32updates/loss/optimizer/params. Optionaltrainerdevice/tasks/per-task-stream flags leaveolddefaults unchanged. Independenttaskseed0+index samplers makecommonselectedexample-IDprefixes identical; exposure1200MOTvs600pairs/400triple andtokensdiffer, sofixedbudget comparisonnotisolatednegativeinterference. Existingolderactorhistoriesexplicitlyoffpolicy; no newfeaturecache/GPU/GTvaltraining/NovelGT. PlannedofficialcompleteDT1203/rawTAO40+shortRMO8/12, thencomplete800RMOafterexistingfreshsourcesready; preserveoriginalrawcandidateprotocol, notaliasconfound. CPUfullstate/Adam/RNG/sampler100step snapshots forunfinished-unitresume.

Actualsubsetpackpreflight536groups (MOT149/RMO325/OV62),142actualnativeRMOgroups,1203classes. Fourarms readoneexactsemanticinitialization checkpoint constructedfromMOT300+seed0heads, eliminatingpriorjointtraining. Separateper-task drawprefixes verifiedfromactualsampleIDs aftertraining. Evaluateall1203DT/all40rawTAO +RMO8/12 beforeblockingforfresh800sources; latterwaitscompleteactualsource, notcompletionofslowerGPUFullcomparators. No sourcefeaturefilescopied; onlysmallmodel/Adam/RNGcheckpoints added.

Tasksubsetallfouractual1200CPUtrainingscompleted/1146511params each/three-taskcheckpointreload0; actualsampleprefixesMOT400/RMO400/OV400exact, same sharedMOT300+seed0semantic initialization. FirstMOTonlyfull1203officialHOTA.1963864227 IDF1.2078625235 IDSW348 FP1089 FN1765; othercohortarms stillpending, notcomparedtolegacy2400budget26.316. Configurationerror discoveredbeforeunfinishedshortRMO scoring: v1pointedoldFullrawlogswithoutnativeROI. Existingjoint1200beforelogs haveexactsame8/12frames11/16boxes/labels/scores andreal2560ROIs; targetedall-rowmappingpasses. Preservev1badplan/oldmanagerstate, SIGTERMonlyverifiedCPUmanager1915707, leaveactualMOTinference1925936+ffmpeg1926238alivewithsameoutputs; newv2manager waitsoldgroupfinish andusesallfourcompletedcheckpoints/completedunits, no trainingorvalidinference restart. Sourcecontractproof saved.

ExactaliasTAO40bothcontrolscomplete: SimpleTETA22.556/LocA22.871/AssocA26.472/ClsA18.325; sharedCompact21.773/23.273/9.6266/32.419 vsraw14.517/8.088/5.1478/30.314. Compactcorrectbirthanchoroptioncoverage.156716->.530973, existingaccuracy.052239->.088496; stillunderSimpleTETA22.556/AssocA26.472. BothNovelClsA0; no promotion. Rawvs7343exactgroups physicalproposalIoU.5 one-to-one recallboth147/160=.91875 (GTposthocaftergrouping), allboxsetsequal; source402723754bytes withall35228aliases. Separateproposal/representativelabelpolicy, notpureJEVassociationclaim. Nextscientificgaps retain: conventionalmatchedvisual/semanticinformation baseline, Compact32optionretrieval/contextscaling underactualmissing-option failure, samevideo/hardware genuineFull-vs-CompactE2E, Groundvisual-only/jointtracking-aware factorial andnativefull9retention, furtherown-historymultiTaskresearch. Do notdeclarefullgoalcomplete.

SharedFullteacher600trainingterminalverified byexistingGPU4controlowner:600retainedupdates/2400examples, latest9,029,141,401byteFP32AdamRNGstate. Actualrestored450mastersdifference0; newtail150updates/600examples took3595.081s. Nativeforkdonecounterdividescumulative2400bytailtime(.66758) so notvalidtail/whole-stageFPS; curateactualtail.166895examples/s, nofull-stagewallclaim. Original475logloss1.404405526 vsreplayed4751.402670712 delta.001734814,slotaccsame.78/gradnormdelta.226344; restorestateexact doesnotprovebitwiseCUDAtrajectory. Causeunestablished; source deterministicrowseededrender/sampler andkernels/tolerances unchanged, no masking/reset/retraining. Original25unretainedloggedupdates replayed,totalobservedwork>=625updates; rawlogs preserved. Newcompletionvalidator recordslimitations. Native600arm andthree-taskofficial600scores pending. Same-policybefore500RMO8/12bothHOTA0 duetoexplicitnativequerygate, notcomparabletolegacyungated24.886/21.425.

E2Eresearchplan addressesactualmissingcomparison: run Fullnative600 andcurrent sharedCompact onexact samelegalRMO0007single800frames/GPU5, freshisolatedoriginalDLMserviceforeach andsingleclientrequestcounterreset (priorisolated96repeatvalidated). FullactualV3crops vsCompactactualnativeROI path, notcached/syntheticlatencies; timeallIO/nativeDLM/ROI/memory/sourcearchive/render/encode excludespreloadedmodel/serviceinitializationboth. Strictactualdetectorbox/label/score/parserstatuspairingguard, every800frame included; ifdifferent preservefailure/nopairclaim. FixedFullthenCompactorder/singlerepetition/concurrentotherwork limitations explicit. Wait completeoldfullRMOowner exit+ownedservicegroups empty, no overlapGPU5/newtraining. Newsourcecap1GiB, cumulativeconservative22471148986<25GiB; sampledownedtotalVRAM/coreallocatorpeak/processRSS andfrozenOneCPUtable footprint included.

TasksubsetDTfull1203fourarms nowcomplete HOTA/IDF1/IDSW/FN: MOT19.6386/20.7863/348/1765; M+R6.7597/9.8843/80/3865; M+OV21.2285/20.0809/414/1006; triple19.3119/17.8267/422/1372. Mixedoutcome, M+Rmisscollapse butMOTexposure600vs1200confoundsinterference. PreregistercomplementaryMOT400/600CPUcontrols sameactualsharedinitialization/source/per-taskrng/constantoptimizer/loss, assertactualsampleIDs+loggedloss exactprefixofMOT1200; thenfull1203fixednativeTrackEval. MatchMOTexposureofpairs/triple whiletotalupdatesdifferent, keeporiginalfixedtotal-budgetmatrixunchanged. No newGPU/queries/featurefiles; reportallfixedbudgets notselectbest.

PairedE2E canonicalcontract onactual4nativeframes/13ROIobjects passes acrossfeatures+geometrylogs; resourceprobeactualCompactCPU4frames records0GPUalloc andcorrect1146511core plus248320x1024BF16frozenCPUtable508559360nominalbytes (notresidentRSS). Newruntime peakRSS/UTC/gitcommit fields tested; no freshnativeSpeed claim fromreplayprobe. Queuedpairedowner1953808/session87824 waits priorGPU5fullRMOcomplete, ports8106/8114, onefreshserviceperarm. Optionaladditionalmetadata enrichments afterownerlaunch do notchangeinference/pairingprotocol; rawchildresultreports areauthoritativeforcomponentmemory. Legal0007v2validation800manifest checkedagainwithoutopeninglabels/images.

2026-10-15：四组完整MOT评测均完成，固定总1200步下 HOTA 为19.639/6.760/21.228/19.312。M+RMOT漏检3865，不能把IDSW减少当成改善；M+OV提升HOTA但IDF1/IDSW变差。因MOT实际更新量1200/600/600/400不同，补充MOT400/600同前缀训练量对照，保持原矩阵不变。已启动单一阻塞CPU owner1957205/session71241；真实端到端同视频/同GPU对照 owner1953808/session87824 仅排队等待GPU5，不与现有任务争用。新增RSS/UTC/git记录已用真实4帧Compact重放校验；未将该重放当成E2E。

研究计划：按任务书14.2，在同一已训练共享Compact、同一候选/三份native外观/几何/文本语义输入下，比较几何固定关联、几何+三记忆cos固定关联、固定关联+学习辅助头、学习关联+固定辅助头及原学习参考。所有语义分类/成员判断函数权重相同；状态由各策略自身逐帧生成，不做GT回填。固定分数明确非概率，常数沿用已有.5/.5与NEW=.3；这是一组冻结策略干预，不冒充重新训练最优传统模型或JEV独有证明。计划完整DT1203、原始/精确合并TAO40、RMO8/12，CPU、不新增feature源。已观察完成MOT耗时408.52s中350.69s、原始TAO1152.83s中773.39s在已测关联块之外，不能全部归因于状态导出。新CPU对照可选逐帧决策trace+最终无损二进制状态，保留所有状态字段，先验证序列化/下一帧一致；原默认JSON/每帧导出和已注册E2E保持不变，不比较这组存储不同的速度。

MOT400/600补充对照已验证完整1203帧/13664候选/27拒绝帧：HOTA13.3173/13.9788，IDF1 17.7572/13.4107，IDSW174/312，FN3483/2566。实际样本ID及已记录loss与MOT1200前缀完全一致(delta0)，各1146511参数/三任务reload0。同MOT更新量下，M+R600为6.7597、M+OV600为21.2285、三任务400为19.3119；总更新量仍不同，不能直接归因纯任务梯度干扰。无模型提升/全数据集/E2E声明。

固定关联及存储回归完成：真实4帧全预测/状态、完整序列化字段和恢复后真实第5帧一致；三个任务在同状态下语义/保留辅助输出一致，固定外观分数与现有FeatureDecision差1.11e-16。新注册五种冻结策略×完整DT1203/短RMO8与12/TAO原始及alias40的CPU序列对照；使用同一checkpoint/原始native2560三记忆/同一文本语义头，逐帧自身历史，无GT修正。所有新组都用binary-final存储且不报告跨存储速度。run_video最终完整状态导出移入pipeline计时，修正旧版重复最终导出未计时的问题；既有速度原样保留，未来配对双方均按修正口径。

冻结关联对照owner1986817/session48561已启动，单一阻塞shell等待所有25个单位；所有CPU比较均新跑同设备参考。逐观察posthoc诊断各约5MB保留本地，Git发布精简统计/官方全指标/实际运行结果和代码，不复制大逐观察列表；完整逐帧trace、视频、状态仍在原输出目录。

研究计划：针对Compact32身份短名单遗漏，先在原始/alias TAO40已完成轨迹上真实因果重放，核对全部原始输出/状态和32-ID排序一致，再做同一轨迹的32/64/128排序覆盖诊断；GT只读取已完成posthoc birth-anchor归属，绝不输入tracker修正历史。覆盖上限不冒充更大context实际精度。后续固定同checkpoint/16候选组，预注册32/64/128实际完整DT与aliasTAO对照；上下文增大超出原训练32选项分布，会改变softmax/语义/状态，必须如实报告。

已验证aliasTAO40因果重放全部7343候选/原始输出/状态，113个已有正确birth-anchor事件下32/64/128覆盖53.097/60.177/74.336%，最大420活跃身份、正确最小rank最大395。说明128也非完整覆盖，不能用覆盖率当精度；原始TAO排名审计owner2000016/session86629仍按原单一阻塞命令运行。真实2帧高密度检查：第2帧前251live IDs，64/128各候选选项数量正确，默认32与显式32全预测/状态一致。后续完整DT1203与aliasTAO40、学习/固定cos两策略×64/128共8新单位，CPU owner2003663/session5404等待当前affinity全部完成并退出，复用并重新核验32参考，不新增feature/GPU。

Full teacher600四个同策略短测均完成：MOT HOTA30.868->30.601，单RMO0->0，多RMO0->9.945，BaseTAO TETA11.981->52.181/AssocA18.399->28.308/ClsA0->100。四个before与四个after都保留原候选/帧并完整decode，RMO双方native Noul.5一致；TAO都输入1203类但4帧只有Base person/baby，不作Novel/广泛分类结论。实际native600训练/后续控制仍待完成，教师600恢复非bitwise限制未更改，无整体成功/模型提升/E2E声明。

原paired600阻塞训练句柄68568已exit0，teacher/native均600更新/2400例，两个最新FP32/Adam/scheduler/RNG文件均9029141401bytes。两个600真实nativeMM重复加载均delta0/474entries；native从250完整state精确恢复，后350更新1400例耗8767.184s，尾段.159686例/s，不能使用累计2400除尾段耗时的.273748当全程速度。native无250之后旧日志重叠，无法证明bitwise重放；teacher475已有差异仍保留。GPU4原owner继续native600评测；GPU5原fullRMOTowner已完成训练交接，实际状态single800_FRESH_COMPACT_NATIVE800，禁止重复启动。后续pairedE2E仍等待其全部结束。

补齐关键完整MOT证据：此前Full微调前后只有短MOT4，不能据此替代完整视频。新注册原V1 DT1203固定候选上Full native600/before500/teacher600三组、固定顺序，各用原生Choice/Noul/Score+V3真实crop/motion、动态分组不增token上限、全部27拒绝帧保留。GPU4新owner只在原短控制owner1870814成功并退出后执行，不与其重叠；GPU5完整RMOT继续。复用已验证600真实MM加载结果，三组统一binary-final存储/共享输入feature只读/逐帧决策trace，无新features或E2E速度声明。通用控制runner流式核对源候选，避免整源重复常驻内存；默认短测行为不变。

原始TAO40排名审计session86629已exit0：全35228候选/原轨迹逐帧输出和状态精确恢复，最大2670live IDs，134已有正确anchor事件，32/64/128覆盖15.672/31.343/44.776%。不把覆盖反事实当精度，也不排除已终止/从未正确初始化身份这一诊断局限。Full DT1203新owner2024544/session88330已验证等待原GPU4短测owner1870814/start41851012，未提前占卡。

研究计划：原始TAO32选项覆盖15.67%，扩大128也仅44.78%（冻结轨迹诊断），已有正确anchor常被纯几何排序挤出。新增独立候选协议：保持32总选项、16候选分组/同权重/同输入，先取16几何，再按实际anchor/recent/long原生2560特征平均cos（对组内候选取最大）补足至32，稳定以lost_age和track_id解同分；不读GT，不更改身份状态或训练。旧geometry默认保持。完整DT1203与raw/aliasTAO40，学习关联及固定cos两策略，四个nativeROI输入仍共享，不新建featurefiles；CPU新owner等既有contextowner完成后串行跑。与原geometry32分别报告，非纯JEV唯一收益或最优检索声明。先检查真实251-ID片段的候选合法性与完整前缀可执行性。

完成并核验任务组合16现有范围：DT1203 HOTA19.639/6.760/21.229/19.312；rawTAO40 TETA12.421/19.284/12.850/10.081；单RMO8 HOTA0/0/26.494/37.265；多RMO12 4.096/3.513/9.622/3.513。所有实际结果/官方指标/候选完整性一致、空失败表，full800仍待源。无普遍最优，三任务单query受益但多query IDF1=0/FN30且TAO退化；四组NovelClsA0。共享总1200步但每任务暴露不同，保留MOT400/600对照，不宣称纯梯度干扰。

5个同输入完整MOT策略完成：learned/learned26.316，IoU/fixed7.488，cos/fixed13.311，cos/learnedaux13.009，learned/fixedaux24.283。学习辅助只在学习关联组合有增益；learned FP579/FN1962 vsfixedcos9328/250，不能将总体收益全归因身份排序。各轨迹内生不同，不是唯一JEV证明。Hybrid32真实251身份片段检查通过：几何默认与旧版全输出状态精确一致，每个新候选32个独特live IDs、前16几何一致，230候选的身份集合实际变化。新CPU owner2034367/session35730只等待已有contextowner2003663完成+退出，再跑3视频范围×2关联策略；保留原默认、不改训练、GT阈值和指标。

研究计划（补齐任务书14.4必需控制）：原Ground200含native MDM/AR+.1GT身份+.1decoded-proposal身份，缺少只视觉定位对照。新localization200从原569张量初始化，同72图/112native行/56pair、seed42/200步/AdamWlr2e-6、相同采样/参数27M，只将两个身份loss系数置0，仍记录非加权分量，不改变native算子/掩码/GT。原track200有中断恢复，声明相同注册配方/采样，不声称bitwise轨迹配对。先CPU比较全部样本payload与schedule，GPU6 owner仅等已有adapted200 full9客户全部结束、该服务组已被原owner释放后才训练，不要求GPU7也结束。随后导出新bundle，并用同GPU6/私有新service/requestcounter同序列比较原始/只定位/跟踪视觉适配；同各自原生候选再比较原始One与native600，组成受控视觉×JEV组合，不称端到端联合梯度训练。无新featurecache，保留全部原测评。

localization CPU预检v1失败：比较最终Ground200 checkpoint与最初GPU7登记的整个protocol，实际差异仅GPU6、step1保存和execution_variant。全部200采样索引已匹配，loss/data/seed/lr未改变。改以实际完成的native_ground_tracking200_gpu6_20261014.json为参考，仍要求实际protocol精确相同及全部112行/200索引相同，fresh v2重跑；保留v1诊断，不放宽数值标准。

发现并修复Full1203排队driver句柄变量遮蔽：upstream registered list/None覆盖teacher/native dict。原等待owner2024544在短测owner1870814正常exit0交接时已实际报TypeError退出（88330 exit1），completed={}，未启动GPU模型；试图替换前的状态断言阻止了任何误杀。修复为独立upstream_registered。真实已完成owner+两份实际600记录的CPU调度回归通过，不启动推理、不冒充评测。原failure保留，fresh v2同1203协议恢复，无已有训练/有效推理重跑。

Full三分支12同策略短测已终端验证：before500/teacher600/native600 MOT HOTA30.868/30.601/32.236、单RMO全0、多RMO0/9.945/11.214、Base4 TETA11.981/52.181/47.682。native多RMO IDF10.256/IDSW1/FP6/FN27，Base4ClsA100只涵盖person/baby单一Basecanonical类，不泛化Novel。native历史效果混合、不晋升模型。Full1203 freshv2 owner2049900/session70182实际进入native_mot_full1203_INFERENCE，原v1错误/无模型执行事实保留。

localization200实际CPU预检v2通过112行图像bytes/prompt/targets/nativealign及与完成checkpoint全部200draw完全相同，72图/56pairs、884nativeproposals/628alignedregions/24支持身份pairs；只定位两个身份coeff0，原图像标签和native核不变。新owner2052936/session76692已等待GPU6 native200 full9真实clients+服务组释放，不中止其原测评；随后200步训练+导出+原始/只定位/跟踪视觉三前端及localization repeat共96帧，再与pretrainedOne/native600组合18短测，明确非联合梯度训练，不新增featurefiles。
