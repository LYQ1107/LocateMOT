Warning: truncated output (original token count: 112419)
Total output lines: 2871

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

Launched one blocking finalization owner with OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_finalize.py --wait-joint-pid 883611. ActualPID896659/session18355 confirms WAITING_ORIGINAL_JOINT_OWNER, pack_started=False, ready_for_u3=False. It waits real original completion, recomputes current full selection, packs actual frozen U2-B tensors, runs original-vocabulary retention and unmodified selected-core audits, then verifies every final gate against bound artifacts. Completed units resume by actual checkpoint receipt; reference fixtures and conditional U2-C cannot substitute for selected U2-B. No original owner was restarted, no additional formal optimizer launche…62419 tokens truncated…实际测评再决定受控续训。

2026-10-16全量CPU规划效率修复计划：已读原full-source owner，459027条原生MM长度检查固定world1，输入仍在检查而非已开始四卡optimizer。新增独立CPU-only Gloo4入口，调用完全相同train.sft.token_lengths/原render seed/media counter/paired index shuffle和global16 plan；先对64个immutable source各首/中/末实际行做serial-vs4长度逐值相等gate，再对全部行原生规划且0drop，保存int32长度约1.84MB供后续显式复用研究。全量检查通过前不切换原owner；原GPU训练/测评、原serial检查均继续，0新GPU或模型权重。小报告与长度另预留4MiB，总feature上界26583180254<26843545600；不将CPU计划/短probe当大规模联合训练或长视频BPTT。

四CPU原192条native MM数值gate已实际通过：覆盖64源各首/中/末，所有token长度serial与Gloo4逐值相同，全459027检查仍运行。新增显式--validated-native-lengths入口：只允许同source manifest/immutable源文件、tokenizer checkpoint、seed/epoch0、augmentation/media/length与global batch配置，原生batch/index shuffle/rank deal保留；默认入口不变。独立真实8条MM regression实际exit0，所有长度/微批/四rank step分配完全相同，改seed拒绝；0model/GPU/optimizer。full-source owner新增可选已完成plan重验证路径，尚未切换原owner或宣称full pass。报告/新probe沿本轮4MiB预留，原6000最新4225正常。

下一自动接续计划：新增单一blocking coordinator，等待原49885实际exit0/full459027原生长度完成，复用入口另做全量CPU重验证并要求所有summary完全相同；仅原owner仍在CPU且GPU训练路径不存在时，先冻结并确认原parent状态T，再确认其exact CPU child/CUDA为空，保留handoff intent和全部旧日志，终止尚未完成CPU子单元。原parent记录真实-15/assertion退出不伪造成功；同原459027/seed0/global16/原生600FP32初始化/评测协议的新owner接续，省去重复tokenize，原GPU训练与测评不停止。若原owner已进入GPU则不切换。新独立first5 observer显式绑定新owner/config，旧observer保留原终止证据。额外元数据+observer共1MiB，feature upper26584228830<25GiB；不改变样本/metric/seed/门槛，不将等待/数据计划当训练完成。

4500实际里程碑核验计划：原64133已输出4500/18000及9029140633bytes完整FP32/Adam/RNG保存。仅CPU读取同一个atomic latest state（不复制9GB），核验320 FP32 masters/两Adam全向量finite及全部moment step/scheduler/cursor4500，读取冻结24k语料按原seed0 index-shuffle计算真实已消费任务/unique source row/首媒体路径曝光，明确重复问题不等于独立帧，仍无6000终评。小报告另预留512KiB使feature upper26584753118<25GiB；原全部训练/测评/CPU队列继续。

4500 checkpoint CPU完整核验原78686 actual exit0：全部752393024 FP32参数、1504786048 Adam moment元素finite，320optimizer step/scheduler/cursor4500与seen18000一致；185条实际日志loss/grad finite。真实sampler消费MOT5957/RMOT5977/OV6066问题，实际首媒体路径MOT980/RMO72/OV3064。首版曝光metadata把课程重新编号ID当source ID，原记录错误计数保留在exposure_count_correction；已改用supervision.source_row_id单独重算（不重复扫描9GB状态、不改训练），课程ID与source ID分列。实际unique source rows为{"rmot": 5677, "mot": 5957, "ovmot": 5624}。不称独立轨迹数量或6000完成/跟踪增益；完整latest4500已验证但未来5000会原子替换，编号BF16权重保留。

全量CPU completion核验计划：原94290已先通过serial full plan并进入FULL_JOINT_FOUR_GPU_TRAINING，原17189实际返回同阶段等待更新；并行49885后完成actual exit0，coordinator2858 actual exit0/status ORIGINAL_ALREADY_ADVANCED_NO_HANDOFF，无任何signal/重启。比较两份completed native plan全部summary（除不同计时定义），核验int32 459027长度及int64总和/原生global16同seed0计划全行守恒和tail3；读取原RMOT source refs/group first-media，追溯raw1814与semantic1796实际曝光差18，保留全部raw/原行，不更改正在训练数据。报告与已完成lengtharray沿原4MiB报告预留，不新增GPU/feature档案；实测CPU计划tokens5639441074/28690updates是计划，不是已消费tokens/训练结果。

原35297 actual exit0：serial与4CPU全部完整数值summary相同，459027长度array int32/全int64总5639441074核验，原生index-shuffle/global16恰28690updates、末step rank行数1/1/1/0，全部行一次不漏不重。native allreduce_grads原实现为无本地梯度参数补零再SUM，末rank空不丢样本。实际输入首媒体MOT1056/RMO1796/OV17023；RMO raw1814的18张差额在两query原保留监督中都仅有lifecycle，登记semantic-only不选这些proxy，原始帧/row完整保留，不造新presence标签、不改当前数据。原serial CPU计时7943.0s仅capture后全plan；4CPU2415.9s含加载及192probe，不作严格受控速度基准。原四卡owner94290仍保留，原17189已实证阶段切换但首5实际optimizer仍待观测；无handoff/重启/信号，三任务普通水平未证明。

原17189 actual exit0：first5四卡真实训练已验证，actual step1..5/seen16,32,48,64,80，MOT49/RMO12/OV19；5条loss/CE/Brier/gradient全finite且gradient>0，原生600初始化320FP32/752393024参数maxdiff0，exact4rank分别在物理4/5/6/7，原owner94290未重启。新phase28690步/459027问题/vision frozen/seed0/确定性FA，非running6000续训、非Ground-One联合梯度、非长视频BPTT；首5仅启动证明不是完整epoch/跟踪收益。前5实际global输入吞吐约13.1k..15.0k tokens/s、每步9.7..19.3秒，仅极短早期/同GPU共用记录，不作稳定速度或结束时间预测。原6000另已至少5300updates，5000完整9029140633byte恢复state/save receipt与编号BF16存在已核验；不重复4500已完成的全部数值扫描。更新TRAINING_LOG/CURRENT_STATUS顶部，旧active状态标历史，保存first5实际args和5000save小receipt；沿原observer1MiB及既有future-curriculum metadata预留，0新feature/model/optimizer。所有tracking终评与ordinary-level目标仍未完成。

连续观测计划：原四卡训练实际first5已完成并发布，但94290原shell句柄不转发逐步stdout，实际日志在training/log.jsonl。新增单一blocking只读跟随器，核对原owner PID/starttick/cwd后用pidfd内部等待，持续转发新完整JSON日志/phase变化并检查loss/gradient finite；从已证明step5之后接续，不重启训练、不写新feature或模型、不进行agent-level ps/tail轮询。优化器done仅表示训练步骤结束，最终reload/跟踪指标仍交原owner校验。

原76060单一blocking follower实际接上原94290的同一训练，立即转发实际step25/400rows，loss2.044512/grad208.539均finite，最近窗口12.097s/update、14576.845global planned-input tokens/s；原样记录共用GPU的极早期样本，不当稳定速度/结束时间/推理E2E或质量收益。原日志原样读不改，first400实际seed0 sampler task counts={"mot": 203, "rmot": 98, "ovmot": 99}；注册follower句柄供后续直接write_stdin，直到optimizer done交原parent完成reload/scorers，0新增模型/feature/optimizer。

6000完成核验准备：复用已实际验证4500的CPU完整状态/曝光freezer，仅将结果字段拆为complete6000_training与new_tracking_metrics_available，避免6000实际结束时仍把二者合并标false；不改验证门槛、训练、采样或metric。原owner负责保存全部6000 FP32/Adam/RNG、final BF16与native-MM reload后固定四scope对照。待原阻塞训练输出完成/阶段事件后才运行CPU验证，不重复扫描仍在进行的5500状态。

Compact扩展最小计划：现有共享Compact只有48MOT/72RMO/64OV小pack，而已有43MOT1056帧+18RMO3628query观察的真实Compact actor原生Full-v3导出，输入内含确切pre-write geometry与anchor/recent/long像素来源。做只读FP32引用loader：按source archive(frame,candidate)重建当前与三份历史2560向量，保存小geometry和reference元数据，不重复保存每group约1MB历史向量；原监督行独立保留，assignment/opaque alias只作真实输入条件不引入GT。先CPU检验与原Compact.forward实际输入逐tensor相等、source outputs/states一致；保留全部空/拒绝帧，空n0/m0明确原Compact不调用forward，不能伪称新增presence head。此为未来三任务Compact扩展输入组件，175类ROI与新train recipe尚未准备/启动；0GPU/optimizer、仅32MiB metadata预留，feature upper26618307550<25GiB。当前两轮OneJev训练及测评不变。

Compact input组件原35764 actual exit0：全部61unit/4684query-frame contexts、7071ordinary输入group与5870真实assigned条件，45空n0m0group保留；MOT1056/RMO1814unique物理图。仅21265696byte metadata，FP32 arrays全部引用原归档不复制；8920次重复box引用在真实数据中逐值确认原F32向量相同，未猜测匹配。原85147 actual exit0：3真实source各前3帧、36实际forward groups/22带历史、6784728输入tensor元素torch.equal，所有native模型forward/actual source outputs-state完全相同，0GT修正/0GPU/optimizer。全61数据准备与九帧数值regression范围分开记录，仍不是完成三任务Compact训练pack，OV175native features未采集、新model训练未启动；后续共享训练需补齐该组件及独立登记recipe。完整21MB可再生metadata cache server-only，发布loader/builder/validator/config/完成证据，不把缓存当tracking结果。原两个OneJev训练和所有评测继续，goal未完成。

Compact OV175补齐计划：仅复用原正跑24k课程中已登记4000个root对象draw及其leaf对应，不另选GT难度/类别或读eval/Novel标签。CPU逐一核对train-Base annoID/vocab index/image/GT bbox与原crop recipe；去重观察向量按annotationID索引，保留rare-class原draw重复。新私有GPU4 ROI-only视觉服务与现有任务共用4/5/6/7总4卡，不停止任何既有进程，原生MoonViT/projector无语言解码/新DLM请求，RPC只给原图+框。单一FP32memmap bank约36MiB及逐图完成记录支持未完成单位恢复，无新PNG/整集缓存。64MiB预留使upper26685416414<25GiB；此组件不启动Compactoptimizer，头/采样recipe另登记。

## 2026-10-16 — shared6000 completion and joint-training scope

Balanced shared OneJev MOT/RMOT/OVMOT training completed6000updates/24000questions (8000per task),29432.9s. Final full FP32/Adam/RNG state validates all752393024parameters and1504786048moment elements finite, all320Adam counters/scheduler/cursor6000; native multimodal repeated reload probability difference0. Retain final and numbered checkpoint; downstream controls have started and tracking gains remain pending. Actual first-media paths:1056MOT/72RMOT/3227OVMOT, distinct from question counts. Independent four-GPU full-source epoch latest observed425/28690updates,6800questions,finite loss/gradient. Ground and OneJev are separately optimized; no end-to-end joint foundation gradient or long-video BPTT. Evidence: outputs/groundjev/training/shared6000_actual6000_snapshot_v1/validation.json.

## 2026-10-16 — broader native Compact supervised adaptation plan

Hypothesis: the previous Compact corpus (48MOT/72RMOT/64OV images) limits transfer; first isolate broader data with the unchanged semantic Compact architecture and existing runtime. Build a fresh explicitly registered question-level curriculum from43MOT native units,18RMOT native query units and the selected175Base native teacher ROI bank. Route only supported original heads (identity/observation-update/reliability/lifecycle/Membership/flat category); inventory unsupported presence and final-ID questions rather than relabeling them. No RMOT unknown-birth auxiliary labels or OV teacher identity/update targets are invented. Train one common final checkpoint on CPU, retain original actor as before control, and run unchanged causal evaluation before judging gains. This is supervised Compact adaptation, not native OneJev decoder reasoning or a completed large-scale epoch.

Compact curriculum first target validation failed onMOT20-05_766_batch0_match_13: no-history native questions have onlyNEW/IGNORE, while existing Compact keeps a third hard-maskedNO_MATCH logit. Encoded target already correctly padded zero; fixed only the validator legal-option expectation, preserving all source labels/questions and the existing model. Failure receipt retained. Base175 native-input validation meanwhile passed all175class representatives/216205500tensor elements with exact inputs and forward outputs.

Broader Compact curriculum now concretely prepared and all24000selected labels checked against original native questions/option order. Balanced MOT8000 (5000identity/2000life/500observation-update/500reliability), native RMOT8000membership (4000yes/4000no), OV8000flat-class exposures preserve the4000originalBase draws twice. Actual eligible first-media coverage1053MOT/1795RMOT/3227OVimages; all226query-frame contexts without a selected supported head explicitly inventoried, original1056/1814source frames preserved. This new question-level balanced recipe also changes sampling/loss aggregation, so resulting comparison is not an isolated data-only ablation. Real three-task8-update continuous versus4+4restored CPU trajectories match every model/Adam/counter/RNG value exactly. Registered8000updates/24000questions unchanged-architecture training and subsequent five unchanged causal controls launched onCPU, no additional GPU. Native OneJev large epoch and6000model controls continue separately.

Broader Compact actual first100updates/300questions verified from its complete log and live PID/starttick, all first logged losses/gradients finite; this is real training, not only registration. Single Base bank component42,597,901bytes<64MiB and curriculum metadata547,051bytes<16MiB; registered caps retained conservatively. CPU before/after scope protocol uses nonoverlapping MOT20train766–893 versus validation1659–3315. Final8000 and five-scope scores pending; no training-gain claim.

Frozen fit affinity diagnosis (1250selected unique correct-existing questions) completed before training: native2560cosine retrieves correct listed identity1050/1250=84.00%, projected128cosine1026/1250=82.08%, IoU986/1250=78.88%, learned existing-only logits893/1250=71.44%; adding existing action competition leaves521/1250=41.68% full correct decisions, with589NO_MATCH choices despite a correct existing option. Thus this fit evidence points more strongly to association/action learning than a wholesale loss of appearance in compression. It is neither held-out accuracy nor causal tracking gain. A read-only owner waits on the verified original Compact trainer PIDfd, then repeats the identical frozen cases using its final checkpoint; no running recipe/runtime changed.

Shared Compact8000 completed24000questions/1167.332s (19.46min),1,146,511parameters,1,139,601changed elements. Full model/Adam finite and export exact; three-task reload exact. Fit1250 existing-ID full correct521→656 (41.68→52.48%), but first held-out DT1203 control strongly regresses: HOTA26.3157→5.5800%, IDF130.6329→6.7840%, IDSW165→488,FP579→1511,FN1962→3637. Preserve negative model; no promotion. Remaining four scopes continue unchanged. Register a separate matched MOT-only8000 update control with the exact original MOT row sequence, sameinitial/Adam/LR/weightdecay/clip and loss/3, excluding only RMOT/OV losses by explicit opt-in. This separates task-gradient interference from the changed data/sampling recipe, without claiming FLOP matching. No new features/GPUs.

Remaining first-three-scope results: RMOTsingle800 HOTA6.0247→1.0579%, IDF12.8902→0%; RMOTmulti800 HOTA19.8019→12.5237%, IDF121.0082→13.1709%. All are negative; retain completed joint8000 model and all five registered controls. The separately declaredMOT-only8000 control has actually passed100updates/100questions; its8versus4+4 full model/Adam/RNG restore is exact. Optional task-subset implementation preserves the original three-task8update model/Adam/RNG/protocol/exposure bit for bit.

MatchedMOT-only8000 completed8000questions/944.519s, full-state finite/export/reload exact; zero category/membership head changes verified. It also strongly regresses DT1203: HOTA26.3157→1.67995%, IDF130.6329→0.82034%, IDSW165→1303; singleRMO HOTA6.0247→2.49096%. Fit correct-existing733/1250=58.64% remains better than original41.68% but does not transfer. Cross-task gradient interference alone cannot explain failure. Registered targeted DT1203 causal intervention for both trained checkpoints: association unchanged, fixed auxiliary policy (write1/reliabilityNone/no learned lifecycle), retain all default negatives and entire pending five-scope studies. This diagnoses the combined auxiliary policy, not a lifecycle-only ablation or successful replacement model. JointTAO40 separately improves TETA14.517→19.252 viaLocA8.088→17.963 andAssocA5.1478→11.012, whileClsA30.314→28.781 worsens andNovelClsA stays0.

Posthoc DT1203 auxiliary-policy intervention completed5arms: joint fixed-auxHOTA5.6588% versus5.5800default;MOT-onlyfixed-aux3.2215% versus1.67995default, both far below original26.3157%. Combined auxiliary replacement alone does not repair association. Original43-sourceMOT labels measured: EXISTING22117/48051(46.03%), NEW11489,IGNORE7992,NO_MATCH6453; lifecycleACTIVE60077/93624,TERMINATE4682/93624(5.00%),LOST12117,REACTIVATE16748. Selected uniform label quotas impose25% each. Register an exact-row/initial/optimizer-matched MOT-only8000 loss-weight control using TRAIN source-frequency/selected-frequency ratios, loss/3 and all existing policies/metrics unchanged. This is an exploratory source-prior hypothesis, not guaranteed deployment calibration; no validation label ratios used.

TRAIN-source prior-weightedMOT-only control actually passed100updates/100questions, finite loss/gradient, exact8vs4+4model/Adam/RNGprefix validation passed. JointMOT20full1657 inference finished; a single resource diagnosis confirms original official evaluator actively running onCPU (491CPU seconds/~60.3GiBRSS at observation), not a stopped model job. Keep original scorer and all frames unchanged; no finalMOT20metric yet.

Next focused diagnosis: replay all1203DTvalidation original CPU actor frames exactly and evaluate original/joint/MOT-only checkpoints on identical captured pre-write input tensors. Score identity argmax and a shadow single-step global assignment using external GT/immutable old birth IDs only after inputs are fixed. Exact original outputs/states and shadow-choice parity required on every frame; no new causal score, training or feature pack claimed. This separates conditional decision failure from changed closed-loop history without GT state repair.

DT1203 frozen-prefix diagnosis completed1203frames/13664candidates; every original output/state and shadow global assignment exact. On1768correct-existing options, original argmax correct651, joint6, MOT-only55; joint predictsIGNORE on1726/1768, MOT-only1527/1768. Thus severe conditional rejection exists before any new-policy history change; exposure bias alone is insufficient. Overall argmax accuracy is dominated by IGNORE and is not a tracking metric. Audit TRAIN and validation birth-label support separately: determine whether all-proposal Hungarian matching leaves actual unique GT-supported emissions unbound, and count isolated valid IoU edges lost by assignment-before-threshold. Do not relabel or alter any running study during this diagnostic.

Birth support auditv1 finished all43train clips and DT1203 validation computations, then exited1 because final json.dumps call used an unsupported keyword. Saved computations and failure receipt retained. Fixed only final summary dictionary construction and rerun the identical read-only audit into freshv2; no labels, frames or acceptance criteria changed.

Birth supportauditv2 exits0: only2/844TRAINbirths have unbound, uniquely supported actual emissions; assignment-before-threshold loses4isolatedvalidTRAINedges. These rare verified cases cannot explain the broad collapse, so do not relabel corpus or interrupt original four-GPU training. SharedFullOneJev6000 defaultDT1203 also regresses HOTA8.7030→7.8406%, IDF15.8260→3.5341%, IDSW3631→2796; assigned-ID samecheckpoint comparison continues. TRAIN-prior-weightedCompact8000 completes8000questions/944.302s, DT HOTA2.5567%, singleRMO3.5421%, multiRMO15.7803%; prior correction alone does not repair. Measured selectedMOT source mix5934MOT20/779MOT17/1287DT, but correct-existing questions496/255/499. Selected correct-history age/100 median.06,max1.23 andconfidence median.592. Register frozen old-prefix weighted-model plus GT-agnostic zeroage/zeroconfidence/zero4scalar interventions and actual input-domain quantiles; no continuous tracking gain or changed default protocol.

Scalar interventions fail to restore conditional identity decisions: weighted model correct3/1768, noage43, noconfidence1, no4scalars26 versus original651. Real correct-alias age/100 median2.03,max8.46 exceeds selectedTRAINmedian.06,max1.23, but erasing scalars alone is insufficient. Corrected an inherited report flag that had claimed all tensors identical despite explicitly declared scalar interventions; source plans/metrics/decisions unchanged, original flag retained in scope-correction receipt. Next implementation: frozen-core multitask adapter with zero-initialized learned native ROI cosine/IoU/distance residual for matching/actions; train only small residual and category/membership heads, preserving original backbone, text-conditioning and auxiliary/association weights exactly. Use identity-onlyMOT plus nativeRMOTmembership/BaseOVcategory, same one final model and an inference residual-disabled ablation. This is a new Compact adapter study, not a native OneJev decoder or trainable shared-backbone claim.

Frozen-core native-relation adapter now concrete: total1,147,379parameters,66,790trainable (868native pair/action residual plus category/membership/scale). New output layers are exactzero; wholeDT1203/13664native candidates reproduce every original output/state/shadow assignment and every untrained model tensor output exactly. New24000-question corpus labels/options validated (8000identity/8000membership/8000Basecategory), all263unsupported query-frame contexts explicitly retained in source inventory; actualeligible images1016MOT/1795RMOT/3227OV. Real three-task8continuous versus4+4restored full model/Adam/RNG is bitexact. CPU8000update production and same-final residual-disabled inference ablation/five-scope controls launched; original backbone/text/association/auxiliary weights frozen, no newshared-backbone or OneJev decoder-gradient claim. New8MiBmetadata reservation raisesconservative temporary upper26710582238<26843545600; no newROI features or GPU.

Clarification: birth-support audit counts only provably unique GT-supported emitted cases. Remaining ambiguous original unbound births are not certified correct, so the broader binding-label mechanism is not ruled out. Keep original data and report uncertainty. Shared/MOT-only8000 full five-scope controls completed; sharedMOT20 improvesHOTA7.4316→11.4780% butIDSW2060→58632, Monly5.6407%; no model promotion.

Native-relation production has actually passed100updates/300questions (100per task); intermediate saved-state validation compares every frozen original weight exactly and checks all model/Adam values finite. Initial parity and prefix restore are execution evidence only; no new tracking gain yet.

TRAIN-prior8000 also completed allfive causal scopes; naturalMOT20 HOTA3.10833553628264%, IDF11.890913979096887%, IDSW30884. Frozen-core native-relation actual saved state observedat3000updates/9000questions: all59original fixed tensors/1080589elements exact, allmodel/Adamfinite. This verifies execution/frozen-boundary only, not gain.

Registered one blocking CPU observer for original full-source four-GPU1000step save; original ownerPID/starttick/cwd confirmed, no processes signaled/restarted. It waits for the actual full-state receipt, then checks all752393024FP32masters/1504786048Adam elements,320counters/scheduler/cursor,all4rankRNG payloads and actual sampler task exposure. It performs no optimizer updates, GPU allocation or actual1000step restart; report pending. Small report reuses1MiB of the already registered8MiB adapter/report reserve.

Native relation8000 completes24000questions/1040.460s. All59fixedweight tensors exact, final full state/export/reload pass. Residual-disabled inference restores originalDT HOTA26.3157% exactly; trained residualDT15.3059%, singleRMO0%, multiRMO24.1906% (disabled19.1828,original19.8019), multiIDF129.2044% andIDSW97 versusoriginal21.0082/165. Positive isolated multi-query component effect but overall regression remains; retain both same-final variants and finish registeredTAO/MOT20 scopes, no promotion.

2026-10-16 — Evidence milestone, not efficacy: original four-GPU459027-question joint training actual1000/16000 save passed full752393024FP32/1504786048Adam-moment finite/cursor/RNG inspection. Task exposures8817/3537/3646; subsequent1125/18000 logged. Preserve training and checkpoints; no Ground+One joint gradient/long-BPTT claim or quality promotion.

2026-10-16 — Next hypothesis/registered implementation plan: actual old Compact uses task/query embedding inside the physical association transformer; semantic-only RMOT/OV supervision can therefore leave physical decisions on a poorly supervised task-conditioned branch. Implement an opt-in canonical MOT physical stream (same original person/MOT text and weights), retaining the true query/task only at semantic readout. On identical candidates/history, physical logits/update/reliability/lifecycle must be query invariant and MOT outputs must remain exact before any training. Then run paired frozen-physical semantic-head-only8000-step studies, identical16000RMOT/OV questions/order/seed/loss/hyperparameters, query-coupled versus canonical physical branch; include untrained canonical arm to separate runtime intervention from semantic training. All5 registered complete scopes and scorers, same final checkpoint across tasks, no residual or threshold changes. No new ROI bank/GPU; reserve8MiB metadata beyond26710582238→26718970846bytes<25GiB. This is a factorization experiment alongside the original running750M joint training, not proof of shared-backbone or Ground+One end-to-end gains.

2026-10-16 — Factorization initial evidence:67real groups/61native units+6Base,335conditional task/query interventions; physical outputs exact invariant, original physical outputs change335/335, membership changes335/335. Every canonical MOT output exact original. Full1203rollout/prefix/training remain owned and pending. Native residual TAO40 all3arms officialvalidated: TETA14.517→14.320, disabled14.646; allNovelClsA0. Preserve mixed result; no promotion.

2026-10-16 — Factorization full MOT numerical regression complete:1203frames/13664candidates, every original causal state/output/shadow Hungarian assignment exact, every untrained canonical output tensor exact original on actual pre-write inputs. No metric improvement inferred. Paired semantic-only resume checks now run under original newCPU owner3908189.

2026-10-16 — Both matched frozen-physical semantic-head studies passed actual8continuous versus4+4restored model/Adam/RNG equality (16questions each), then owner3908189 launched registeredcoupled/canonical studies3913160/3913161. Physical-weight preservation and allfive officialcontrols remain required. Native750M joint training continues independently.

2026-10-16 — Native750M shared6000 final-ID DT1203 complete: official HOTA7.840637%/IDF1=3.534137%, exactly same as same-checkpoint default (original600 HOTA8.702996%). Real assigned-ID second passes1350, existing assignments4940, changed update probabilities4940, registered.5gate flips0; all1203exported outputs exact=True. Second-pass execution verified, no DT gain; RMOT/TAO controls continue under originalowner.

2026-10-16 — Actual matched head-only8000training completed botharms:16000semanticquestions each,115.290/116.032s,65922trainable/1146511totalparameters. All59fixedoriginaltensors1080589elements exact; fullmodel/Adamfinite andexport/reloadexact. Preservefinalbothmodels; fullfive-scopecontrols nowrunning, no effectivenessclaim. This pairedphysical-queryintervention is distinct from ongoingnative750Msharedtraining.

2026-10-16 — Next physical learning hypothesis: native OneJev6000 DT association remains weak and final-ID secondpass makes no gate changes; frozen physical/query factorization alone preserves MOT but does not restore single-RMOT. Train a task/query-independent native appearance projection with supervised cross-frame contrastive identities directly from immutable TRAIN candidate registry GT matches, bypassing actor birth-alias/action labels. Use all43TRAIN units/1056frames; keep original labels/sources, explicitly inventory singleton/noncontrastive identities. Same-video8identities×up-to4distinctframes,1000fixedsteps/seed0, original2560D ROI vectors by reference only. Evaluate native cosine, random initial projection, final learned projection under identical existing.5IoU+.5positive-cosine fusion/NEW.3/IGNORE0, fixed auxiliaries/gated3s, original semantic heads and all5official scopes. This representation-learning control is not native JEV probability, end-to-end Ground+One training or long-video BPTT. Reserve48MiB (16MiBmetadata+32MiBbounded streaming working buffers) beyond26718970846→26769302494<25GiB; model/Adam underdata3, no new ROI/image bank or GPU. First validate actual8vs4+4checkpoint resume, then train/fullcontrols; final1000 selected regardless heldout metrics, no threshold tuning.

2026-10-16 — Physical-factorization first4officialscopes complete:DT HOTA26.3157exactallarms; coupled/canonical singleRMOT0/3.1383vsoriginal6.0247, multi19.1420/12.0955vs19.8019. TAO40coupled14.871/canonical31.323vsoriginal14.517/untrainedcanonical26.125; canonicalLocA33.075/AssocA35.637samebeforeandaftersemantictraining, ClsA9.6637→25.259(stillbeloworiginal30.314), NovelClsA0. Keep isolatedOVassociationgain, rejectthree-taskpromotion. Newcross-frameprojection1000complete341184params/27797exposures/36.587s, all43units;887TRAINIDs/873crossframe,14singletonspreserved; actualresumeallstatebitexact and67group16572affinityformula/defaultsemanticregressionexact. All5fullcontrolsrun unchanged. Native residualMOT20HOTA19.5518/IDF1=17.4310%, IDs30874vs2060; all5enabledscopescomplete,lastdisabledablationpending.

2026-10-16 — Completed original native residual15/15officialcontrols, finaldisabledMOT20 restores original metrics exactly; preserve allcompletedproof. Next narrow causal audit: external frozenTAO40 native-versus-learned category evidence using exact official annotation+fullLVISsynsetaliases; distinguish any-box/class oracle support from class-agnostic one-to-one emitted support. An initial ad hoc print used annotation-onlyaliases and is not accepted as equivalent to official mapping; complete registered audit uses both dictionaries without changing predictions or metrics. No native-label restore is deployed before this evidence.

2026-10-16 — Native temporal projection DT1203 official4arms: originalHOTA26.3157%; fixednativecos13.3108%, randomprojection13.4881%, trainedprojection8.4607% (IDF1=3.2767%). No promotion, remaining4scopes continue. Frozen category audit with exact officialaliases: Novelboxsupport37/40 but nativecorrectlabel+boxsupport1/40; emittedGTmatchedNovel6 and0native/learnedcorrect labels. Restoring rawnative labels alone is unsupported. New foundation hypothesis/plan: many original43MOT train clips cover~0.75–2s (MOT20 up to5.08s); longer supervision may reduce appearance shortcuts, but causation is not established. Build paired short-versus-long GT teacher identity episodes from all legal frozen train_frames only, same current image/box, same randomly aliased history identities/options/targets and other24000-mixture rows; past observation interval changes.4000positive identity teacher rows +4000originalnativeMOT rows +same8000RMOT/8000OV rows perarm. Teacher snapshots remain provided across gaps; not realactor rollout, long-BPTT or a deployedTTL/archive change. Actual native request/crop recipes/CPU plans required, then independent shared750M new stages from same native6000 FP32 checkpoint onGPU7 only after original6000owner exits successfully. Matched final6000steps/default4scopecontrols perarm; no old job restart or protocol change. Reserve16MiB reference/plan metadata:26769302494→26786079710<25GiB; no new images/ROI bank, model/Adam separatelydata3.

2026-10-16 — Paired temporal teacher preparationv1 failed its final16MiBmetadata cap:22,748,803bytes, dominated by two10.14MB copies of the immutable9.96MB source mixture. No GPU/newoptimizer update started. Preserve failedinputs/code/terminalreceipt. Smallest fix: add explicit indexed-source manifest protocol and reference originalselected20000rows instead of copying bodies; newteacher4000refs remain. V2 keeps individual16MiBcap, separately reserves24MiB for failedinputpreservation; globalupper26769302494+40MiB=26811245534<25GiB. No samples/targets/seeds/scorerschanged.

2026-10-16 — Indexedv2 ordering/provenance amendment: original row IDs are preserved;20000common native/semantic rows precede4000matchedteacher refs, then native seed0indexshuffle. This differs from failed/untrainedv1 interleaving/renamedIDs and is explicitly registered; botharmssameorder/augmentationIDs. No v1gradient trajectory existed. Mandatory regression checks actual8legacyrows and3indexedrows exact before complete native24000CPU plans.

2026-10-16 — V2pairedtemporalteacher prepared/fullyCPUvalidated:43TRAINvideos/9789selectedimages,4000matchedpositivesperarm+4000oldnativeMOT+16000fixedsemanticrows. Shortgaps[.0333,.5,.96,1]s; long[3,7.95,38.65,104.2]s.86nativegroupstates/questions/172pixelsheetsexact; all24000rows/0drop/max27319,27319/6000plannedupdateseach.0newgradsteps, originaloneblockingownerwaitingnative6000GPU7success. Preservev1failedcopiedmixtures22.75MB andcode; indexedv2actual2.85MBwithin16MiB, separate24MiBfailedpreservation, global26811245534<25GiB. Native residual15/15andfactorization25/25officialcontrolscomplete; no common3taskpromotion. CategoryauditpeeleronlyNovel:37/40boxsupport,1/40box+label,0correctof6emittedGTmatches; externaloraclecountsnotClsA, no native-labelrestore. ReID12armsfirst3scopescomplete; trainedDT8.4607HOTA<13.3108nativefixed<26.3157original, no promotion.

2026-10-16 — Correct mixed projection evidence after fullRMOTcontrols: singleHOTA9.2325vsoriginal6.0247/nativefixed5.5377/random6.6298; multi31.5524vs19.8019/16.6224/14.3821, IDF1=40.0107vs21.0082%, IDSW53vs165. Keep representation as a candidate, no promotion dueDT8.4607. FixedaffinityactionNEW.3>IGNORE0forcesemissionofallrawproposals(FP9330vs579); trainedAssAalsofalls(native.06641→trained.02618). Next planned implementation: appearance-informed permutation of existing-ID probability ranks, exact original action probabilities/probability multiset and auxiliaryheads retained on identical inputs; allfive scopes, native/random/trained contrast controls, no heldout temperature tuning(use originalcontrastive.07). Not yet implemented/measured.

2026-10-16 — Implement next preregistered physical control: trained appearance helped bothRMOTscopes but fixedNEW.3/IGNORE0forcesallproposalemission andDTassociationalsoregresses. Preserve the original trained Compact action probabilities, complete probability multiset, all auxiliary/semantic outputs and policies on identical inputs. Reorder only existing-ID probability ranks by log(originalIDprob)+mean(anchor/recent/long cosine)/.07; .07 is the existing contrastive training temperature, not heldout tuned. Compare original versus native/raw, random projection and trained projection rank controls on all5unchanged scopes, same originalsharedcheckpoint and finalprojection1000. Per-row maximum-existing/action odds and entropy are preserved; global matching/closed-loop histories can still change, so no fixedFP/trajectory guarantee. No new optimizer/GPU/ROIbank; metadata suballocation within existing projection16MiB reservation, streaming buffers within existing32MiB reserve. This is an explicit Compact probability reranking control, not native OneJev decoder or calibrated contrastive posterior.

2026-10-16 — Native inference throughput hypothesis: officialMMDecisionEngine already provides DecoderGraphs/ForkGraphs but defaults are disabled; stock graph limits32rows/512suffix tokens do not cover dense tracking or1203category root requests. Probe official Torch graph implementations configured128rows/8192suffix/32768prefix/max1cached prefix, attached after identical stock eager auto-fork calibration. No vendor/kernel/model changes or FLAinference. Same native6000checkpoint and real legal sparseMOT/denseMOT20/categoryroot+fixedleaf requests; sameimages/state/questions, eager versusgraphs, warmup excluded/3timedrepeats. Preserve existing1e-3probability gate, exactdiscretechoices, finiteoutputs, realgraph-hit evidence and >=1.5component ratio; failure remains rejected, no oldjob/backend switched. PhysicalGPU7only with>=32GiBfree, bothenginevariants sequential, allprojectworkstill4/5/6/7. Reserve8MiBsmallprobe/inputmetadata beyond26811245534→26819634142<25GiB; no new persistentROI/imagebank. This is a component probe, not tracking accuracy or E2E speed.

2026-10-16 — Appearance-rank conservation passed67realgroups/566rows/8286IDoptions perarm, actionvalues/fullprobabilitymultisets/auxsemanticlife exact. First12officialcontrols: DT HOTA% original26.3157/rawrank30.2544/random19.4111/trained24.7555; singleRMO6.0247/5.7087/5.8205/6.1548; multi19.8019/18.9930/19.6370/20.1424. RawrankDT IDF1=38.2885 vs30.6329, IDs148vs165, butFP720vs579/FN2156vs1962. Preserveisolatedgain andrepresentation/policyinteraction; no per-taskselection or commonpromotion. OriginalTAO/MOT20ownerscontinue.

2026-10-16 — Graph probe terminal0/completed but scientifically rejected: sparse/root/leaf maxnativeanswerdiff .011445/.013179/.013894 exceeds1e-3; ratios1.044/1.057/1.061 below1.5. Dense16/32graph callsfall back,30.754svseager13.942s/.453ratio; no actualgraphsuccess. Smallest next diagnostic uses savedanswers only (0GPU):68literalChoices,17Noul>=.5gates and17Hungarianrowsall unchanged; no RMOTmembership/final-ID questions, no futuretrajectoryguarantee. InstalledQwen3.5forward has no deepstackinjection, so do not claim missingdeepstackrootcause. Numericalcause unresolved; no vendorpatch/protocolweakening/backendadoption. Save allrawanswers/timings/hits/terminalreceipt, retain failure and stop this unpromising speed path. Metadata remainswithin existing8MiBreserve/global26819634142<25GiB.

2026-10-16 — Previous status-only turn was a verified wait via original follower76060: originalfourGPU logged1700updates/27200questions,20688.803s/5.747h. Freeze exacthistorical1700row without polling/restarting training; log progress differsfromlastfullyinspected1000state. Update stale425doc snapshot. Original6000controls/pairedlongteacherCPUwait/retention and Full training remain original owners; no completion/three-taskqualityclaim. Publish this measuredrank/code/graphfailure evidence, excluding liveGroundownerstate.

2026-10-16 — Original64133handle confirms native6000 allDT/singleRMO/multiRMO before/default/assigned controls terminal-complete and parentlive waitingregisteredTAObaseline. New actualnative multiRMOHOTA%6.7400→15.6144(default)→15.8936(assigned), DT8.7030→7.8406 unchangedassigned. Next boundedCPUproof: revalidate each completed originalunit with originalvalidate_run(fullframes/candidates/rejections/checkpoint/nativequestions/video), freeze9unitmetrics, audit actualsecondpass call/existingassignment/gateflipcounts against sameassigned causalstate observationprobabilities, and compare fullpairedoutputs. No newGPU/model/threshold/data/cachebank; smallreceipt stayswithin original6000ownerreport reservation. Do not mistake cross-trajectory update differences for same-input counterfactuals or these registered videos for ordinary-model parity.

2026-10-16 — Posthoc freezer90288 failedexit1 beforepublication: firstDT lifecycleevent(frame15track2LOST) has no candidate/decision, normal tracker schema overlooked by newaudit. Original3DTunits alreadyvalidated, no inference/training failure. Preservefailedscript+receipt. Smallestfix: event signatures include candidate=None lifecycle records, memory-write probability audit explicitly considers candidateevents only; all lifecycleevents retained in assignmentcomparison. Rerun full9registeredunits/everyevent as targeted real-dataregression, unchangedmodel/metrics/frames, noGPU.

2026-10-16 — Correctedfreezer41575actualexit0, all9native units/fullvideos/originalcandidateinputs/GTisolation/checkpoints/rejections andeveryevent validated. HOTA% before600/default6000/assigned6000 DT8.7030/7.8406/7.8406, singleRMO2.5062/3.3287/3.4052, multi6.7400/15.6144/15.8936. MultiIDF1%6.6932/15.6747/16.7555; default→assignedFP467→457/FN1297→1303/IDs475→471(before185). Nativefoundationadaptation helpsregisteredRMOTbutnotDT. Sameassignedcausalstate updateaudit: single61calls/78assignments/5gateflips(4hold→update/1update→hold), multi620/1190/7(1/6), DT1350/4940/0. Originalalloutputframes exactdefaultvsassigned DT1203/1203,single796/800,multi388/800; RMOdoeschangefuturehistories. EarliernoassignedbenefitclaimwasDT-only, nowpositivebutsmallRMOeffect. Preservefixedcommoncheckpoint/policies, TAOpending, no ordinary-level/full-datasetpromotion. No newGPU/optimizer or featurebank; originalownerreports budget covers smallreceipts andfailedauditcode.

2026-10-16 — Previous goal turn is progress: actualnative9unit/RMOwriteaudit completed and verified GitHub2da51d6; this turn originalrank73067/projection68145/follower76060 live, Full1775/28400logged. Bothsmallappearance studies now allTAO4armscomplete (rankTETA14.517/14.104raw/14.324random/14.967trained; fixed14.517/13.964/14.194/14.395), allNovelClsA0. FinalMOT20controls stillrunning; no promotion.

2026-10-16 — Scheduling amendment plan for genuine paired temporal foundation training: prepared48krowCPUplans and immutable native6000fullFP32/Adam/RNG+nativeMMreload already validated; original60582v2owner does0GPU/0grad while waiting the unrelated slowTAOcontrol dependency. One resource snapshot showsGPU7free46653MiB with only originalfourGPUrank34488MiB, while4/5/6free18512/24756/20685MiB. Start bothpre-registered6000short/long stages earlier from exactlysame6000state/data/configs, keeping allmetrics/seed/steps/finalcheckpoint and deferredoriginalbaselinecomparisons unchanged. Only actualCPUwaitingv2owner may be handed off after exactPID/starttick/cwd/command/nochildren/0updates verification; preserveitsstateandhandoffreceipt. OriginalfourGPUepoch/native6000controls/otherGPUjobs neverstop. NewGPU7Torchallocatorcap.33of80GiB(~26.4GiB)+expandable_segments accounts for originalrankobservedup-to~43GiBreservation; not a full driver-memoryguarantee. FutureGPU7resource-wrapper launches take sharedflock, newpairedowner holds exclusivewindow (ownsequentialchildren inheritFD); oldalready-runningfourGPUrank explicitlypermitted. Releasewindow afterbothtraining/reloadBEFORE waiting originalcontrolowner, avoidingdependencycycle. First run realCPUprocess lock/bypass/regression; onlythenhandoff/launch andverify actualoptimizerupdates. This changesexecutionorder/allocator only, not scientificcurricula/acceptance. New4MiBmetadata reservation raisesglobal26819634142→26823828446<26843545600; noROI/imagebank. CapOOM must retainfailure/checkpoint, notskiprows orweakenprotocol. No claimedlongtraining/quality untilactualrecords.

2026-10-16 — ActualCPUwrapper regressionpassed: owninheritedexclusiveFDenters, GPU4normalwrapperenters, futureGPU7wrapperblocksandreleasesafterclose;0GPUupdates. Exactpreparedcourse/config/initreferencesunchanged. Verified4002071/start52168753WAITONLY/nochildren/emptyCUDA/0updates, sentSIGTERM viaexactpidfd only;60582actualexit143, oldstate/data preserved andhandoffintent/receipt saved. Replacementowner4118405/start52597639/session65852 nowholdsGPU7window, launchedshortSFT fromsameindexedv2configs. Nativeactual320FP32/752393024initialparameters maxdiff0 vsnative6000latestfullstate; 474weightsloaded, visionfrozen. ActualTorchcap28081638604bytes(.33devicecapacity) set; expandableallocator envpersists fornewownertraining/reload/eval (memoryonly, no comparablespeedclaim). Productionoptimizerupdatesstillunproven atlaunchsnapshot; waitactualrecords, notmodel-loadingclaim. OldFull/legacyinferencejobsnotstopped; onlyfutureGPU7launchesdeferreduntilwindowrelease.

2026-10-16 — Freeze bothappearance first4fullscopes/all16units each fromoriginalcompletedownerreceipts plusfreshofficial/runtime/frame/candidate/rejection/state checks, notrepeatvideo decode alreadydone byoriginalowner. Firstprojectionfreezer invocation mistakenly passedtrainingplanwithoutscopes (KeyErrorbeforeanyoutput); correctedtoownerrecordedresolved_controls_plan.json, sameoriginalcourse. RankTAOtrainedTETA14.967>14.517original butallNovelClsA0; fixedtrained14.395<14.517. Preserveallarms/no commonpromotion, finalMOT20stillrunning. No inference/metricchange or newGPU.

2026-10-16 — V3early study actualfirst5shortoptimizer verified: steps1..5/seen4..20, allfinite loss/CE/Brier/gradandpositivegrad,320FP32/752393024native6000initialmastersmaxdiff0. ActualTorchcap28081638604bytes applied; first5peak16.8548GiB, no capfit/steadyFPS/fullrunmemoryclaim beyondtheseupdates. Longbranchstill0, no trackinggainclaim. Source24000 nativeplan0drop/6000updates, rebuiltbyoriginaltrainer29s (not extra saved CPU plan speedclaim). ExistingFull4GPUepoch notstopped; old0gradientwaiterhandoff143only. Freezefirst5prooffromactual log/nativeinitial receipt andexactownerPIDtick, publishcode/plan/CPUlockproof/handoff/launch/first5 plus bothfirst4scopeappearanceproofs. Goal remainsactive, ordinary3taskqualitystillunmet.

2026-10-16 — Previous goal turn is progress: pairedtemporalshortactualoptimizerstarted, boundedwindow/CPUhandoff/first5proven and verified GitHubb09b07b; current original65852/68145/73067/76060handlesalllive, short50logged and Full1875/30000logged. Next physical hypothesis derivesfromcompletedcontrols: fixed-affinitytrainedprojection improvesmultiRMOT19.80→31.55butDTFP9330; probability-rankconservation controlsrejectproposalsbutlosetheRMOTgain. Before anotherintervention, auditall5originallearnedbeforetrajectories for existing-ID probability dilution: matchedGTwithcorrectsurvivingoption, sumP(existing)>maxP(action) butmaxP(existingID)<maxP(action). This is a same-state oracle potential/diagnostic only, not newprediction/HOTA or a metricreplacement. Streamoriginalscalarprobabilities and frozenbirthGTdiagnostics, noROI/image/model/GPU; keep no-correct-anchor/unmatched groups separate. If supported, test factorizedconditionalappearance P'(id)=oldexistingmass*softmax(logoldP(id)+cos/.07), keeping exactoriginalNEW/IGNORE/NO_MATCH and totalexistingmass plus learnedaux/semantics; probabilitymultiset/maxexistingodds intentionallychange. Same3native/random/trainedcontrols/all5scopes/fixedfinalprojection, original.07temperature, noheldouttuning. Register3MiBtotalforaudit1MiB/conditionalmetadata1MiB/boundedprojectedworking1MiB, raises26823828446→26826974174<26843545600; no newROI/imagebank, originalGPUtraining/measurementprotocolunchanged. Do not implementbasedon imagineddilution; inspectactualauditfirst.

2026-10-16 — Originalprobabilitydilutionaudit24041actualexit0/all5fullscopes287744candidates: positivecorrect-shortlist eventsDT1768/single53/multi191/TAO21/MOT2019195; diluted65/0/0/0/3040. Single-correctsubsetDT55/1170,MOT202587/15774; not a supported explanation oflostRMOTprojectiongain, so do NOT launch conditionalmasspolicy now. MOT20GTmatched219060 minuscorrectoption19195 leaves199865withoutverifiedcorrectshortlistedbirth; this is not automatically trueidentityabsence because birthGTunknown andshortlistmiss canconfound. Next scalarCPUcausal provenance audit distinguishes firstobservation/noGTboundbirthyet, allknownbirthsretired, aliveknownbirthomittedfromshortlist; replay lifecycledeathcauses (learnedterminate before3s vs timeout vsboth), bound/unbound births, and candidate-supportedGT atdeath. Auditall5originalCompactreferences plus9alreadycompletednative600controls; verify reconstructedbirth/survival againstfrozenexternaldiagnostic, keepGTstrictlyposthoc, noGTrepair/runtimechange. Repurposeplannedconditionalmetadata/working withinsame3MiBreserve, do notincreasecachebudget or modifyrunningcourses. Only thenchoose smallest justifiedmemory/birth intervention.

2026-10-16 — Provenanceaudit87987actualexit0/all14units: CompactMOT20positive219060 =first646/neverGTboundbirth147769/allboundbirthsretired45599/alivebutshortlistmiss5851/correctshortlist19195. Do not infer a physicalIDabsent fromunknownbirth binding. NativeDT600→6000knownbirths19→141 and learnedpre-timeoutdeaths0→138(131GTstillpresent); native multiRMO0→55(41GTpresent), despitebetterHOTA. CompactearlyknownGTdeaths DT0/single20/multi7/TAO0/MOT2014. Sourcecodeexport_broader_native_full_groups.py labels unknownbirth lifecycleTERMINATE, an alreadyknown exporterproxy ratherthanproofpermanentexit; this plusobservedheadbehavior motivates anexplicitcontrol, not acausalproof or alteringcurrenttraining. Registeropt-in timeout_onlydeletion: retainlearnedprobabilities/auxiliary/semantic outputs andsame3sTTL, deferonly learnedearlytermination. Defaultlearned_or_timeout andallrunningprotocolsunmodified. Validateactualnative+Compactprefix/defaultparity/realearly-deletiondeferral/unchangedexpiry, thenall5Compactfullscopecontrols againstsameoriginalcheckpoint. Native6000 counterpart willuseoriginal4scopes afterexistingGPU7window/baselinecompletion, exactsamecheckpoint/categoriesreusedonlywherehistory-independent anddeclared; noheldoutthresholdtuning or newROI/imagebank. Allmetadata stayswithinprevious3MiBreserve/global26826974174<25GiB; no newGPU now. Negativeoutcomesmustremainvisible, no policyadoptionbeforecontrols.

2026-10-16 — Actualterminationfunctional11397exit0: Compact30-frameprefix/firstearlyframe29ID1, native19-frameprefix/firstearlyframe19ID4; originalfullprefixstate/outputsexact, interventionretains exactlyearly-deletedIDs/allcommonstatesexact, original3sexpiryverified withlabelledfunctionalempty-observationcounterfactual (not scored frame). CPUall5study82298started, noGPU. Nativequeuedowner34665v1waitsonoriginal6000success and0inference. Staticreviewcaught summary key native_category_replay, actualrun_video exports category_replay. FixonefieldbeforeanyGPUwork; preservequeuedv1code/state andhand offonlythisnewCPUwaiter to v2. No experimentfailed orsamplesrerun yet; do not let a known latevalidationbug remain queued. Originaltraining/controls neverstop.

2026-10-16 — Nativev1newCPUqueue4147813/start52745273exactverified/nochildren/emptyCUDA/0inference, pidfdSIGTERM/34665actualexit143 preserved; correctedv2owner4149961/start52754749/session90700waitingoriginal6000terminalsuccess, noGPU. It takesexclusiveGPU7windowonlyafterdependencycompletion, rerunsall4nativeidentity/lifecycle/scorers atsame6000weights/3s; same-image-onlyTAOcategoriesreplayedexplicitlyfromcompletedoriginal6000source, noE2Eclaim. No originaljobterminated. Compactfirst3fullscopes/all6units frozen: HOTA%DT26.3157exact,single6.0247→6.4816,multi19.8019→19.5578; multiIDF1=21.0082→21.0638/IDs165→157/FP619→597/FN1054→1061. Mixed,noadoption; remainingTAO/MOT20continue.

2026-10-16 — Literal lifecycleinventory12901actualexit0: full-sourceMOT252258rows/lifecycle93624:ACTIVE60077,LOST12117,REACTIVATE16748,TERMINATE4682(5.00085%). Balancedcompleted24k:1000life/250each(25%TERMINATE); pairedshort/longsame500life128ACTIVE/125LOST/117REACTIVATE/130TERMINATE(26%). Thus currentnaturalFullcoursehasnot inheritedbalancedoversampling; do notstoporretargetongoingoriginaltrainingbasedon anunprovenproxycausalityclaim. All data/targets preserved. Sourcecontractunknownbirth→TERMINATEisnotindependentpermanentexitevidence. OriginalFourGPU2kstateobserver49603registeredwaitingcheckpoint, boundoriginal3457625/start49350320, fullFP32/Adam/allrankRNGgateunchanged; latestglobalfeatureupper26826974174 passedasmetadataoverride insteadofstalehistoricalupper. Lastcompletednumericmilestoneremains1000untilactual2kproof, nonewGPU.

2026-10-16 — Actualobserver49603exit0/2000fullstate validated:32000questions17590MOT/7107RMO/7303OV; all752393024FP32/1504786048Adammomentsfinite,all320Adamsteps/scheduler/cursor2000exact,fourrankTorch/CUDA/Python/NumPyRNGsaved. Atomiclatest9029193825bytes andnumberedBF16ckpt-step2000present; futurelatestreplaced, numberedexportretained. No actualmilestonerestart/newinference orcompleteepochclaim. Lateststep2000loss1.10098/grad51.8833/elapsed24227.312s isobservedconcurrenttraining, not controlledspeed. Originaltrainingunchanged. CPUtimeoutTAOcompleteTETA14.517→14.493/NovelClsA0; first4all8unitsfrozen, finalMOT20continues. Preservemixedfailure-to-improve, no promotion. Publishsourceaudits/default-prefixproof/optinpolicy/queuednativecontrol/preexecutionfixreceipts/actual2000proof; allreportswithinshared3MiBmetadatareservation, noROIbank/GPUincrease. Goalremainsactive.

2026-10-16 — Originalhandles52020/14226/46228/64133/65852allconfirmedlive. Native600TAO40beforebaselinejustcompletedandconsumer64133validated35228candidates/all40frames; native6000defaultTAO nowactuallywaitsregisteredGPU7trainingwindow. Sameoriginal600checkpointall4beforebaselines frozen:fndDT HOTA8.7030%,singleRMO2.5062,multi6.7400,rawTAOTETA27.748/LocA21.573/AssocA2.8136/ClsA58.856; BaseTETA37.437/Novel8.3686/NovelClsA0(onlyknife/baby/peelerGT). NativebaseclassificationstrongerthanCompactbutassociationweak; do notmixcomponentmetrics/predictsyntheticTETA orclaimcompleteTAOdatasets/ordinaryparity. Baselinecompletionisnotnative6000aftertraininggain. OriginalGPU5refreshedworkerhasadvancedDT(frame168latest),GPU4oldTAO(frame987started)stilllive; Groundretentionquietlive. Shortpairedtraining425logged, no newcheckpointclaim. Recheckonlycompletedbaselineownerreceipts/official/runtime metadata, preserveinputorder/GTisolation andallrawresults, no newGPU/ROIfeature or inference.

2026-10-16 — Time-description correction: sourceTAOmanifest explicitly physical_video_fps_verified=False/fps1/timestamps0..39/annotation-stepunits. Earlier lifecyclecontrolprose3seconds and provenancefield seconds_since_last_assignment were misleadingforTAO; MOT/RMOTphysicalsecondsremainvalid. Numericruntimehasalwaysusedoriginalmanifeststeps, not3physicalsecondsforTAO; do notinventFPS/changecontrols. Clarifyconfigs/docs andwritev2clock provenance: explicitphysical_seconds/annotation_steps andregistered_clock_gap_since_last_assignment. Real32945exit0, all14oldvsnewcounts/availability/retirementcauses/numericgaps exactlyequal. Retainv1, originalregistrations andalltrajectories/scores; addclockclarificationreceipt. This isdescription/schema correctiononly, notnewinference/input/timeouts/protocol/numericalchange. Shared3MiBreportreservationstillcoversallartifacts. FreezecompleteNative600before4scopebaseline withclockqualification; no qualitypromotion.

2026-10-16 — Previous goal turn is progress: lifecycleprovenance/defaultprefix/controlimplementation, actual2000fourGPUfullstate and completefour-scope native600baseline published and verified4288d02; clockclarificationretainsallnumbers. Current65852short600logged, original73067rankfinalMOT20events/68145projectionfinalMOT20training stilllive; original82298timeoutCPUowner actualexit0/all10controls complete. Validate/freeze all5 before new work, no commonpromotionfrompartialresults.

2026-10-16 — Next birth-admission hypothesis from actual API/exporter code: exporter targetsNO_MATCH only when a correct birth anchor is in the live pool but omitted from shortlist; NEW when no recoverable live correct anchor exists. V3prompt says previously observed but does not explicitly qualify live/retired, and both Compact/Full omit totalpool/omittedcount from decision inputs. Auditall14frozenunits for actualNO_MATCH with complete live-pool exposure, including GTmatched never-bound/retired/correct-option and GTunknown groups; do not declare unknownbirths physicallyabsent or change oldsemanticlabels. If supported, test explicit active-pool execution contract: leave all original model probabilities/aux/semantics untouched, forbid only the privateNO_MATCHcost when every live ID is offered. No probability renormalization or learned thresholds; global assignment/futurehistories can change. Factorial all5Compact scopes: original and completedtimeout references, feasibility-only and feasibility+timeout, samecheckpoint/candidates/scorers/no per-taskselection. Defaultpolicy and all live GPUcourses unchanged. This is a declared contract/intervention consistent with exporter labels, not proof original vagueprompt was impossible or a native calibrated posterior improvement. New2MiB audit/controlmetadata+boundedworking raises26826974174→26829071326<26843545600; no newROI/imagebank or GPU. Auditbeforeimplementation.

2026-10-16 — Exposureaudit66807actualexit0/all14: CompactDTall13664rows exposecompletepool yet10406NO_MATCH selections (GTmatched2100,GTunmatched8306); completeMOT20pool137727rows/112129NO_MATCH (GTmatched105239,GTunmatched6890). Correctknownliveanchor actuallyoffered in598DT/2986MOT20selectedNO_MATCH cases; unknownbirthgroupnotphysicalabsenceproof. CompactRMOonly1each, TAO0completepoolNO_MATCH. Native6000DTonly3completepoolNO_MATCH/RMO0each; primarynative6kregressionisnotassumedsameasCompact. No remainingprintedmasszero inauditedcompletepoolrows. This supports actualCompactactive-pool feasibilitycontrol; do notaddanotherGPUstudy to currentnativequeue. Implementmatchingconstraintonly, preserveall originalrawprobabilities/confidence/entropy and allmodelaux/semantics onidenticalinput; notconditionalrenormalization. ExtractexistingHungarianhelper withoutchangingdefault arithmetic and verifyall14originalassignments/339596candidates plusconstrainedroweligibility/probabilityimmutability beforefullfactorialrollouts. Allcontrols4arms×5scopes, original/timeoutreusedexplicitlyandonly2newarmsinfer. No GTdecisionrule or input/proposalfilter.

2026-10-16 — Realfeasibilityvalidation18145actualexit0: all14/339596originaldefaultassignments exact, originalprobabilitydicts unchanged, forbiddenprivateNO_MATCHneverchosenundernewactive-poolmode, frameswithoutmaskentireassignmentsexact. Same-inputshadowassignmentchangesDT10792/MOT20116558/single1/multi1/TAO0; these are not rollout/HOTA gains. LargeDTGTunmatched8306NO_MATCHcaseswarnofpotentialfalsebirth/duplicateproblem; do notguaranteeimprovementorhideworsening. FactorialCPUowner started afterproof, all4×5originalscopes, twooriginalcompletedreferencearms/twonewactualarms, samebasecheckpoint/noGPU/newfeaturebank. Modelrawdistributions andconfidence/entropyremainunchangedonidenticalinput, costconstraintonly; originalprompt/exporter-semanticdiscrepancydeclared. Existingnative750Mshort/long/Fulltrainingandqueuedlifecyclecontrol remain unchanged.

2026-10-16 — Previous user-question turn was a status explanation, no research-state change. Resume with concrete evidence: original48863confirmedlive through its existing blocking handle, first12factorial units actually completed; freeze their official/runtime/frame/candidate/rejection/finalstate receipts. Feasibility-only DT HOTA26.3157→12.9762,IDF130.6329→9.1995,FP579→3509,FN1962→1550,IDs165→459; bothsameDT. Singlefeasibilityunchanged6.0247; timeout/both6.4816. Multi19.8019→20.6629feasible/20.2362both, timeout19.5578. Additional admission cannot be adopted as general association repair; preserve negatives, no commonpromotion. TAO/MOT20sameoriginalowner continue, no task-specific selection/newGPU or proposalchange.

2026-10-16 — Originalrank/projection/timeoutstudies allclosed:20+20+10officialunits and all5scope receipts retained. FinalMOT20original/rawrank/randomrank/trainedrankHOTA7.4316/8.0009/7.9101/8.0870; trainedrankIDs3317>2060original. Fixedraw/random/trained31.3078/31.5077/31.7503, trainedIDs8689/FP16049 versus2060/463original. Largefixedfusiongainisnotlearnedprojectiongain because raw/randomcontrols already31.3/31.5; trainedprojectionDT8.46bad andallTAONovelClsA0. TimeoutMOT20HOTA7.6054/IDF16.3199 versus7.4316/6.1474, smallmixedacrossscopes. No modelpromotion. Exactobservedfull2250/36000 andshort875/3500logrows frozen for truthfultraininganswer; no newfullstate gate beyond2000, noGround-Onejointgradients/longBPTT. Docs updatedwithallresults/limitations. Allnewmetadataunderexisting2MiBreservation/global26829071326<25GiB; no ROI/imagebank or newtraining.

2026-10-16 — Next targeted TRAIN-only hypothesis after feasibilityDTfailure: originalbirthaudit already finds5832/7992IGNORE proposals with someGTIoU≥0.5, but doesnot showwhether the globallychosen competitor is inside the actual16-candidate request. Readexporter: one Hungarian overwholeframe determines IGNORE, then inputs aregrouped; fullimage remainsvisible. Auditall43/1056TRAINframes/48051candidates usingactualsavedgroups/rowtargets/registry, distinguishbackgroundvsGT-supportedIGNORE and competitorsinside/outsideactualgroup. ReproducealloriginalGTbindings andexactIGNOREtargets beforecounts; leaveambiguousmultitargetcasesexplicit. No claiming mathematicallyunobservable labels because modelseesfullimage, no NMS/trainingrecipe/proposal/metricchange. Small scalarreport≤128KiBfromalreadyreserved2MiB/global26829071326; no newROI/imagebank/GPU/optimizer. Do notretargetongoing459kcourse fromaudit alone.

2026-10-16 — Actual97066exit0: candidate-groupauditall43TRAIN/1056frames/48051candidates, everyoriginalregistryGTbindingandIGNORE/nonIGNOREtargetexact; matchespriorbirthaudit7992IGNORE/5832GT-supported. 4804/5832=82.37%GT-supportedIGNORE haveonlyoutsidegroupassignedwinners (4723singleGT),1024inside/4829outside(overlap)/4noassignedwinner/261multiGT. Actual3401nonemptygroups/max16, geometry/localindicesexact. Positives6754withrivals/5319outside. Fullimagepixelsstillvisible; notprooflabels mathematicallyunobservable, notcausaltrackinggain oruniquevalidproposalcertification. Readonly63350bytereportwithinexisting128KiBsuballocation/noextrareserve/features/GPU. Currentcourses/labelsuntouched; nextadmissionresearchmustexplicitlyprovidecompetition, ratherthansilentlyturningIGNOREintobackground. Original48863TAO completed, freezeall16first4units: feasibilityTAOTETA14.517exactoriginal; both14.493exacttimeout,allNovelClsA0; finalMOT20sameownerpending. Publishedprior18filesverifiedremote3f18440/tree1c9d8a3, thenrecordnewaudit/TAOreceipt separately. Goalnotcomplete.

2026-10-16 — Previousgoalturnprogress: actual43TRAINgroupcompetitionaudit/first16factorialunits and code/results verifiedGitHubafa32bd. Currentoriginal48863confirmedlive/MOT20feasibleinference, no restart. Smallest nextintervention frommeasured4804outside-onlyIGNOREcompetitors: opt-in overlap-neighbor candidate grouping with unchanged16capacity. Seedlowestremainingoriginalindex, repeatedlyaddmaximumIoU-to-groupremainingcandidate, originalindex tiebreak; no IoUthreshold/NMS/topK/GT/classrule orcandidate removal. Stream one-vs-Ngeometry(noN²featurematrix), preserve originaldecision/outputroworder aftergroup modelcalls. ExactsamefrozenCompactweights/shortlist32/learnedheads/defaultlife+NO_MATCH/registeredclocks; groupingcanchangealllearnedoutputs, shortlists/life-first-group/futurestates, explicitlynot pureassociation-only or probabilityconservation. All5scopes before/newgrouping, completedbefore references explicit, no newGPU/optimizer/bank. Validateactualall5defaultprefix states/outputs exact and alternativegroup feature/index coverage beforelaunch; quantifyTRAINcompetitor co-exposure usingactualreferencegroups, notqualitymetric. Workinggeometry≤1MiB plusreport≤256KiBwithinremaining existing2MiBaudit/controlreserve; featureupper26829071326unchanged. NativeFull/pairedcourses unchanged; no future trainingrecipepromotedwithoutheldoutrollouts.

2026-10-16 — Actual36906exit0/newall43groupaudit reproduces everypriorTRAINcount/GTbinding/IGNOREtarget; overlapneighbors only-outsidewinner4804→298, inside1024→5530, positiveswithoutsiderivals5319→320. This isnotnewlabel/qualitygain. Actual23522exit0/15realframes/3339candidates acrossall5scopes: full defaultoutputs/stateexact, alternativecausalinputfeatureownership andprobability/update/query/classrows correctlyrestoredtooriginalcandidateindices. Afterbothgates, originalnewCPUowner90021 launchedbefore/regroupedall5×2, samefrozenmodel/16capacity/shortlist32/noGPU/newoptimizer; referencesexplicit, allnewcandidaterowscheckedasfullpermutation. No NMS/threshold/GT/classfilter; alllearnedheads/shortlists/life-first-group/futurehistories maychange. Do notclaimassociation-only/constantprobabilities/futuretraininggain. OriginalFull/paired/NO_MATCHjobs untouched; Groundretention original52020confirmedlive. Newgeometry/reportworkingwithinremainingexisting2MiBreservation/global26829071326; noROI/imagebank. Goalactive.

2026-10-16 — Newgroupingoriginalowner41803/start53230371/session90021verifiedlive; firstDTfull1203 actuallycompletedandfrozen2officialunits. Samefrozenmodel groupingonly: HOTA26.3157→28.6921,DetA45.1842→46.2724,AssA15.3476→17.8072,IDF130.6329→33.2758,IDs165→159,FP579→736,FN1962→1794. ModestpositiveDT withFPcost, notuniformmetricimprovement orproven5scopebenefit. KeepallremainingRMO/TAO/MOT20controls; do notpromoteper-taskcheckpoint/policy. Registration/launchsnapshot retained; newsource/runtimegroupscheckedpercandidate byoriginalowner. NoGPU/optimizer/featuresadded.

2026-10-16 — Original90021completedbothRMO800scopes, freezeall6first3officialunits: singleHOTA6.0247%,multi19.8019%, everyreportedofficialmetric exactlyoriginal. Do notinferallprobability/stateequality fromunchangedmetrics. GroupingpositiveDTdoesnotyetconstituteall-taskbenefit; TAO/MOT20pending unchangedoriginalowner. PreserveearlyDTreceiptalonglaterfirst3receipt, publishimplementation/audit/defaultprefix+rowownershipproof/registration/actualownerlaunch/first3metrics. NoGPU/optimizer/newROIfeatures.

2026-10-16 — Previousgoalturnprogress: overlap-neighborimplementation/defaultprefix+rowalignmentproof/all43TRAINcoexposure/first6officialunits publishedandverified93fb5fa. Currentoriginal90021TAOrunning and48863MOT20feasibleofficial→events, confirmedlivebyexistinghandles; no restart. Original76060follower reportsFull2350/37600,65852short1100 andreal1000fullstate9029140889bytes saved. Reuseexisting shared-checkpointfreezer with explicittrainingpath/expectedinitial6000 and indexedsourceIO; preserveitsdefaultoldbalanced600-initializationbehavior. Validateall320masters/Adam/scheduler/cursor/RNGpresence andactualseed0index exposures once onshort1000state. Teacherrows genuinelylacksource_row_id; inventorytheiroriginalrowIDsseparately, do notinventolder-sourceIDs or inflateframecounts. No trainingchange/restart/newGPU/bank/checkpointcopy, smallscalarproof withinexistingreportsreservation/global26829071326. Trackinggainremainsunprovenfornewshortbranch.

2026-10-16 — NewcompletedGround200artifactsnow6tasks visible (COCO,dense200,VisDrone,RefCOCOgval/test,RefCOCOplus) whileoriginalfull9ownerstillwaitingremainingclients. Existingpartialfreezerhardcodesold2-taskoutput; addonlyexplicit--out(defaultoldunchanged) tofreezecurrentcompletedwholetasks withunchangedofficialsummarizer andexactoriginal/effective/providerfilteredcounts. Do notclaimfull9, infermissingtasks, launchnewclients, stopservices orattributeallweightcausality: originalqueue32/adapted-syncpost-recoveryexecutiondiffers. Newsmallscalarresultssuballocateexistingreportsreserve, nobank/GPU/checkpointcopy. This iscompletionevidence forrequiredcontrolledGroundadaptation, separatefromtrackingHOTA/ordinaryparity.

2026-10-16 — Actual79475exit0/short1000fullstate9029140889bytes:752393024FP32/1504786048Adammomentsallfinite,320steps/scheduler/epoch/cursorexact,RNGfields saved;initial6000FP32maxdiff0. Actual4000seed0questionsMOT1288/RMO1327/OV1385;673teacherMOTrowswithoutoldsourceIDcountedseparately,615providedoldMOTIDs; firstmedia1017/71/1180 includeswiderlegalteacherTRAINimages, notallhistoryframes/independenttracks.45loggedrowsfinite. No1000restart/newnativeinference orshort6000/longcompletion. Existinggenericfreezerparameterizedonlypath/initialstep/source-manifestloader/missing-sourceIDinventory; legacydefaults retained, currentnewrealcheckpointprovidestargetedregression. No broadrepeatold6000state scan.

2026-10-16 — Original90021completedTAOall40; freezefirst4/all8officialunits: groupingTETA14.517→13.997,LocA8.088→6.6283,AssocA5.1478→10.2,ClsA30.314→25.162,NovelClsA0. Associationupwhileothercomponentsdown; no syntheticcombinedscore/categoryreplay orpromotion. Original48863MOT20feasible-onlycompletedall1657/235232: HOTA7.4316→9.8980,IDF16.1474→10.1300,IDs2060→3292,FP463→564,FN286273→267117. Freshofficial/runtime/frame/candidate/isolation/finalstate checks retainoriginalfullsourcechecks; freezeonecompletedunit. Combinedfeasible-timeoutstillrunning; DTcollapse remainsfailure/nonewpolicyadoption.

2026-10-16 — Groundpartialfreezeractualexit0/original9tasks91124rows/adapted200six31986rows, alloriginal=effective/providerfiltered0. CompletedcommonF1mIoU%orig→adapted COCO47.9781→47.5713,Dense66.6435→66.0540,VisDrone29.6624→27.8704,RefCOCOgval91.7038→89.3700,test91.1193→88.5215,RefCOCOplus84.3763→81.6416. Allsixnegative; preservewholeofficialresultcounts, no Ground200promotion. PendingLVIS/RefCOCO/RefCOCOg notinferfull9; queue32originalvsadapted-syncafterCUDArecoverymeansnotexclusiveweightcausality/bitwisepairedclaim. No trackinggaininferred. Existingcurator--outonlyparameterization preservesold2tasksnapshotandallsources, no newmodelclients orservicestops. Allnewproofswithinalreadyreservedscalarreports/global26829071326; noGPU/optimizer/bank/checkpointcopy. Goalactive.

2026-10-16 — Previousgoalturnprogress: actualshort1000all752M/Adam/cursor/sourceexposurevalidation, first4groupingcontrols/MOT20feasibleunit andsixwholeGroundretentiontasks publishedandverifiedc962b4b. Current90021/48863originalhandlesconfirmedlive/finalMOT20arms. Next compute question-context reuse opportunity in actual full459027seed0/global16/fourrank index-shuffle. ReadnativeSFT/shared: existing share-prefix supports oneGPU text-packedbranch masks; groupforward doesnotuse multimodal mm_inputs, and Qwen3.5 GDN recurrentstate needs branch-aware handling. Cannotjustenableflag orreuse languageactivationsacrossoptimizerupdates. Auditall64originalsources usingnative per-row rendering toidentifyidenticalserializedstate+mediarecipes; reusealreadyvalidatedint32lengths andnativeplan_steps placement, noMMtensor/model/inference/GPU. Compute sameglobalstep andsamerank co-occurrences plusoptimisticfull-rowtokensavingupperbound fortheconservativeexact-metadata cachekey. Epoch repeatedcontexts alone arenotavailableacrosschangingweights; notgeneralboundoverallpossiblepixel-equivalences oractualthroughput. Allrows/familyfilters/order/seeds unchanged. Persist only≤128KiBscalarreportfromexistingfullsource/reportreservation, no native feature/activation bank or newoverallcacheallocation. This audit decideswhether a future exact branching implementation merits priority forcurrentrecipe; donotretargetrunningtraining.

2026-10-16 — BothoriginalCPUstudies90021/48863actualexit0/all5closedandfreshreceipts10+20units. GroupingMOT20HOTA7.4316→7.0507,IDF16.1474→5.6306,IDs2060→1992,FP463→341,FN286273→287351; NO_MATCHbothMOT20HOTA9.7962/IDF110.0231/IDs3447/FP513/FN265498 versusorig7.4316/6.1474/2060/463/286273. Keepallnegativecrossscoperesults, neitheradopted. Defaultmodel/policies unchanged.

2026-10-16 — Nextshared-core hypothesis fromcompletedcanonical controls andactualcode: frozenuntrainedcanonicalphyscore keepsMOTexact, TAOphysicalLocA33.075/AssocA35.637 vsorig8.088/5.1478, butclass9.6637vs30.314 andRMOHOTA2.005/11.753vs6.025/19.802. Existingcanonical routessemanticheadfromcanonicalcandidatevectors; actualspec onlyentersmembershiplate. Addopt-in secondpass ofthe SAMEfrozenTransformer/task/text/semanticweights forsemanticreadout, conditionedonactualtask/query; physicalmatch/update/reliability/life remaincanonical. Oneidentitymemory andonecommoncheckpoint, noGTstate repair/newparams/training. Onidenticalnativeinputs validatephys==canonical, semantic==originalquery-coupled, MOTalloutputs==original; all61actualTRAINcontextgroupsplusBaseobservations/declaredqueryvariants. Then before/canonical-untrained/newtask-readout all5fixedscopes, oldtwoarms explicitcompletedreferences, no per-taskcheckpointselection. NotpredictingcombinedTETAfromoldcomponents: newtrajectory/officialmetrics required. Existingfull/pairednative trainingunchanged. Onlysmallscalarproofs/modelvariantflag fromsameweights, no bank/GPU/featureallocation; global26829071326 retained. Categorysingleton idea deferred: original64frameOVcourse includescausalteacherhistory whereasnewBase175supplementisn1/m0, so do notclaimbaselinealreadytrainedexclusivelyon singletoncontext.

2026-10-16 — Actual41236exit0/contextauditall64sources459027rows/nativeaugmentation/seed0/native16x4dealing/tail1,1,1,0 exactlypreserved;97556serializedstate+media contexts/24560medialists. Sameglobalupdate150repeatedcontexts/150extrarows: full-rowtokensavingupper2945176/5639441074=0.0522246%; same-rank23/23,390967=0.00693273%. Theseconservativeexact-metadata-key bounds arenotactualspeed/allpixel-equivalencebounds; media-onlycooccurrences don'tmeasurevisioncost. No forward/GPU/optimizer/bank. Currentrandomshuffleofferslittle exactwithin-stepbranchreuse; do NOT switchofficialtextsharedprefixentry orreordercurrenttraining. Futuregroupedrecipe+multimodal/GDNbranchstateimplementation wouldneedseparatevalidation.

2026-10-16 — Actual79437exit0/newtaskconditionedreadout67realgroups(61withhistory)/402specinterventions/143238outputelements: phys==canonicalandqueryinvariant; category/membership==originalquerycoupled onidenticalinputs; MOTallheadsexactoriginal.335canonicalsemanticinterventionschanged. Separateactualweightcomparisonall66oldstate_tensors/1146511elements bit-exact acrossold/canonical/new; no newparamtensors, samecanonicalbuffer. Noqualityclaimfromthesetensorchecks. NewCPUowner51823launchedall5/3arms afterproofs,10completedreferences+5actualtrajectories, originalallscopes/16groups/shortlist32/TTLclocks/heads/thresholds/scorers unchanged. Singlememory andsinglecommonparamset acrossalltasks; semanticbranchactualtask/query contextonly. OriginalFull/pairednative traininguntouched, no GPU/newfeaturebank; scalarproofs/modelvariantwithinexistingreserves, global26829071326. Goalactive; defaults remainoriginalcoupled/noncanonical.

2026-10-16 — New51823actualfirst3/all9officialunitscompletedandfrozen. HOTA% original/canonical/newtaskreadout DT26.3157/26.3157/26.3157, singleRMO6.0247/2.0050/2.4619,multi19.8019/11.7532/11.9169. NewsingleFP47/FN17 vscanonical80/17/orig323/13; newmultiFP160/FN1704 vs196/1703/orig619/1054. Restoredsameweightsquery-conditionedsemanticsreducesFPslightlybutdoesnotrepaircanonicalphysicalmisses/association; no defaultpromotion or claimnewunifiedimprovement. RemainingTAO/MOT20originalsameowner, no threshold/seed/framechanges orsyntheticcomponentmix. Nextlearningworkmustaddressphysicaltransfer, notassumereadoutalonefixesalltasks. Publishcode/plan/tensor+parameterproofs/launch/first3metrics/all5previouscontrolclosures/full459kcontextaudit. AllrunningnativeGPUjobsuntouched.

2026-10-16 — Previousgoalturnprogress: full459kcontextaudit andboth5scopecontrolclosures/newtaskreadoutimplementation+exacttensorproof/first9metrics publishedverified4ad1839. Current51823TAOofficial→events confirmedlive; original65852short1550/76060Full2550logsfinite. ReadpreparedknownbirthRMO2882component andfullsourcecurriculum: new18rollout RMO component selectedmembership/presenceonly, notthese1428identity+1454assignedwrite rows. This iscurrent-curriculumcoverage, notproofnoearlierRMOTphysicaltraining orthatgapalonecausesfailure.

2026-10-16 — Prepare matched native input component (no newGPUtraining): reuseall2882certifiedknown-birth physical questions/targets andtheircorrespondingoriginalquery-membership question/target fromsamecandidate. Botharms sameoriginalactualactorstate/media; assigned-write control gets samealready-selected-identity state asphysical, not GT-repairedhistory. Keep originalroworder/pairIDs, sourcequestionprovenance separate. Native rowaugmentation shufflesdifferentnumberofoptionsbeforechoosingJSONstyle, so same seed canotherwisegive differentprefixformats. Explicitnewpairedcomponent fixesbothstates to source-equivalent prettyJSONstring; keepnative per-questionoptionpermutation, then verifyall2882native renderedstateprefixes equal. This isdeclaredderivativeinputformat, notunchangedoldtrainingrecipe. Originalinputtrees/questions/media/labels retained, unknownbirth/missing-match notinventednegative. Runbothcomplete2882-row nativeMMCPUplans zero-drop beforefuturetrainingrecipe; nooptimizer launched/queuedbycomponent. New4MiBscalar/groupmetadatareservation raises26829071326→26833265630<26843545600; noROI/image/activationbank, no currentfull/pairedcourse edits. Source18rollout1814physicalframes/977filteredmedia; anchorage isnotocclusiongap.

2026-10-16 — Actual43883exit0/newpairednativecomponent2882pairs/2074groups prepared; originalnative rowstylewoulddiffer698pairsunderthesameseed dueoption-count-dependentRNG. Fixedsource-equivalentprettystrings eliminateallrenderedstateprefixdifferences whileoriginalquestion/target/media/JSONtree/order provenanceexact. Physical1428identity+1454assignedwrite(981no/473yes), semantic2237yes/645no. Bothall2882nativeMMCPUplanszero-drop: physicalmax18096/tokens15639410, semanticmax17227/tokens15226174, noforward/optimizer/GPU. Totalcomponent1665559bytesbeforecoverageproof<4MiB; newglobal26833265630. SourcefullcurriculumRMO97112membership+3701presence only (plannedinventorynotactualconsumedcounts); noneof2882newphysicalrows silentlyinserted.

2026-10-16 — Original51823actualexit0/all15newreadoutofficialunitsfrozen. TAOTETAoriginal14.517/canonical26.125/new30.818; newLocA33.075/AssocA35.637unchangedcanonical, ClsA9.6637→23.743 stillbeloworiginal30.314; BaseTETA31.9085,NovelTETA28.637butNovelClsA0(threeGTclasses/oneNovelpeeler/40annotationframes). NotfullTAOparity orNovelrecognitionclaim. MOT20allmetrics exactlyoriginalHOTA7.4316/IDF16.1474; RMOsingle2.4619/multi11.9169 stillbeloworiginal6.0247/19.8019. No defaultpromotion orper-taskmix. Ordinary3taskgoalunmet.

2026-10-16 — Legacyoriginalhandles14226/46228confirmedlive; oldnativeTAOnowlastframe1132, refreshedcompletedallDT/RMOfirst3andstarteditsTAO. Actual750Mhistory-refreshstage additional600/2400eachfromsame600FP32initial, sameperstepfamily/tasks/400historyrowschanged, nottoken/FLOPmatched: DT old13.1075→refreshed16.4112 vsbefore8.7030; singleRMO3.0717→2.8779 vsbefore2.5062; multi6.6825→7.5062 vsbefore6.7400. Freezecompletedbefore/old/refreshedfirst3wholeunits withoriginalowner/official/runtime/sourcegeometry/video/rawofficial-input checks; no rescore/newnativeforward, no fullTAOconclusion. New1MiBnativehistoryscalar/prediction-report reserve raises26833265630→26834314206<26843545600; originalworkers/services untouched. Keepmatchednegative singleRMO outcomevisible.

2026-10-16 — Actual8215exit0 freezes all9 native history first3 units and their exact compressed raw predictions/original official scores;734102bytes within the1MiB reserve. Actual23170exit0 proves canonical versus task-readout physical trajectories across all4500frames/287744candidates/331063events plus final physical memories exact; semantic labels/counts only excluded. Restoring task semantics alone fails to recover RMOT physical coverage. Neither study changes the default model.

2026-10-16 — Previous user-question turn only explained scope and observed logs; no research-state change. Resume concrete work: original14226 now actualexit0/HISTORY_REFRESH_WORKER_ALL_REGISTERED_UNITS_VALIDATED. Old-history TAO completed TETA25.161 versus native600before27.748, NovelClsA0; refreshed46228 confirmed live at frame88. Original65852/76060 confirmed live, short1950/7800 andFull2750/44000 observed, not new numeric checkpoint gates. Freeze the newly complete old TAO unit with source/decoded-frame/GT isolation and exact official prediction transport checks, within the existing aggregate1MiB history-report reservation if it fits; no rescoring or GPU. Publish paired-input rebuild code/proofs, all5 readout closure, complete physical-rollout proof and native history results. Current courses and original workers untouched; ordinary three-task goal remains unmet.

2026-10-16 — Actual99790exit0/old TAO native40 source+geometry+decodedvideo+GT isolation verified, exact raw and official JSON tracker inputs reconstructed from every output including unknown-category -1. Original TETA27.748→25.161, LocA21.573→23.965/AssocA2.8136→2.9972/ClsA58.856→48.522, NovelClsA0. All5112official predictions preserved; no rescoring/inference/GPU. First3 plus old-TAO artifacts aggregate875796bytes<existing1MiB; global26834314206 unchanged. Refreshed TAO still original46228live, no inferred score. Observed2750/44000 and short1950/7800 exact rows frozen for prior user scope answer; numeric gates remain2000/1000, long0at observation.

2026-10-16 — Publication53ce5ed verified:53files/treefd210563 including complete readout/control proofs/native predictions and paired2882rebuild inputs, no weights/banks/live Groundowner. Next actual learning prerequisite: construct two complete24000three-task courses using all old8000MOT/8000OV plus last5118oldRMOT rows and all2882paired questions. Same21118common rows/order/IDs and same pairedphysical states/media at actual native global indices; only registered question/target families differ. Both use same future native6000FP32 initialization, seed0/nativeglobal4/indexshuffle,6000full-course updates per arm if separately launched; no GPU queue/optimizer here. Native per-step length ordering can differ, not token/FLOP matched. Run both complete native MM CPU plans, zero-drop required. New2MiB manifests/configs/log reservation raises26834314206→26836411358<26843545600; no new pixel/ROI/activationbank or edits to active Full/temporal courses. This tests physical-supervision transfer, not another semantic-head-only repair.

2026-10-16 — Actual56390exit0/both complete24000native MM CPU plans zero drops, max27480tokens; physical202072473plannedtokens versus semantic201659237, both6000nativeglobal4updates. All2882paired prefixes exact at actual21118+indices; common21118old rows unchanged. Metadata257474bytes<new2MiB reservation, global26836411358. Register actual additional6000native750M/24000 per-arm physical-versus-semantic learning study from identical verified6000FP32; physical then semantic, same seed0/newAdam, fixedfinalsharedmodel and assigned-ID write protocol across original4scopes. Both training+reload stages precede CPU-only wait for original6000assigned TAO baseline; no substituted600baseline or per-task selection. OldGPU4owner actualexit0 releases native slot; one launch observation GPU4free37341MiB/existingfullrank43804MiB, checkpointdisk350843351040bytesfree. Use GPU4only, existingfourGPUrank untouched, Torch.33cap28GB/expandable allocator; extend existing allocation wrapper with explicit--physical-gpu4 while default7 unchanged. No exclusive resource claim or process stops. New4MiBowner/report reservation raises26836411358→26840605662<26843545600; no native feature bank, originalcourses unchanged. First real optimizer updates/native initialization evidence required before claiming launched training.

2026-10-16 — Actual newowner34237 running /paired_rmot_physical_semantic6000_owner_v1 from committed d28e15b; native GPU4.33cap28081638604bytes, exact6000FP32initmaxdiff0/752393024language parameters. Original blocking handle emitted updates1..5, losses/positivegradients finite,20questions. Immutable first5validator73855actualexit0 confirms owner PID/starttick/cwd/script and exact rows4/8/12/16/20; native seed0 source exposures9MOT/6RMOT/5OV including2newRMOTidentity rows. First5Torchpeak17.87594GiB, not full-run peak/stable speed or quality gain. Semantic0at snapshot, bothfull6000andfour-scope assigned-ID controls pending. This is the real GPU4 allocation-wrapper regression; defaultGPU7 CLI retained and originalFull/temporal workers untouched. No new native feature bank, currentglobal26840605662<25GiB. Preserve failed results and full goal active.

2026-10-19 — G2 protocol repair and clean coverage launch: read-only audit confirmed the historical 199053-row G2 manifest contains 3317 MOT20-05 rows, so formal_g2_1200/G2-PL remain engineering evidence rather than independent held-out evidence. Built immutable G2_CLEAN_FULL_COVERAGE_V2 (583251 unique rows, 2249 rejects, MOT20-05 absent, TAO Novel/sparse negatives excluded, all legal Refer-KITTI frames, exact RMOT questions, BURST gap explicit; rows SHA f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71). Added exact-once source/video/image coverage accounting, global-batch gradient accumulation with DDP no_sync, token-budget checks, per-step token/gradient/peak-memory logs, and strict base/manifest/LoRA/optimizer/scheduler/per-rank-RNG resume validation. GPU3/GPU5 BS2 smoke and two-rank step-1 resume replay passed; BS8 resource probe failed with a real vision OOM and is retained. Formal clean owner is running from G0 with micro-batch2, accumulation8, global batch32 and 18227 planned optimizer updates. J1 remains active at durable step600/9571 seen rows; no quality metric is claimed before terminal receipts and fixed independent evaluation.

2026-10-19 — G0 MOT20-05 official scope evaluation completed from the preserved 1658-frame export. TrackEval receipt reports HOTA 0.433441, DetA 0.638348, AssA 0.294655, IDF1 0.441085, IDSW 6239, FP 6325, FN 98471; 183 rejected-perception frames remain in the official denominator, evaluation-only TRAIN GT was used, and official test labels were not read. This is a G0 reference scope, not a clean G2 held-out result because MOT20-05 is excluded from G2_CLEAN_FULL_COVERAGE_V2.

2026-10-19 — Post-launch runtime snapshot after the G0 scorer: the clean G2 owner remains RUNNING at optimizer step 24/18227, with 16 global samples, 17610 effective global tokens, finite loss 1.7098015, gradient norm 2.69265 and 17.3715 GiB peak device memory in its latest event. J1 remains RUNNING on GPU4/GPU6 at durable step 650/1200 with 10368 rows seen; no terminal receipt or quality claim is made for either course. The G2 and J1 owners were not restarted or modified.

2026-10-19 — J2 candidate-source audit corrected the training entrypoint: the balanced onejev_unified_matched600_v2/native manifest is preserved but rejected for candidate-driven J2 because its MOT rows use legal TRAIN GT proposals plus explicit synthetic fragments and its OVMOT rows use registered train-Base teacher crops; only its repeated 346-row RMOT component is native GroundAnything. SHA 136d029d212d831e1c72cf7c7fe538dc9b246e8edc0febb9bb14ac1abf662156, 125286 rows. A separate streaming inventory records the usable sources: 131214 native MOT rows from 7 MOT17 sequences (SHA b60d029b28a1d5bb101361ea548cf07e48b41033ccfcbe20d78f32cc7f0ba7b4), 346 native RMOT rows from 72 fit frames (SHA cdb1e38342f2bf467c0d288e532b840964328447020911043ded36968381e4a1), and 430 OVMOT teacher-control rows (SHA 48269098ac2a8ebafb5d0c50a7de9852e10bf50d37e087f71db6c6f74ecfd95b). No J2 optimizer was launched; native OVMOT candidates and broader legal coverage remain required after G1/G2 detector export.

2026-10-19 — Registered and started a GPU5 G1 continuation owner behind the active clean G2 run. Owner session 68355 is waiting on the authoritative G2 terminal receipt; it refuses to launch if G2 fails, trains a fresh 200-step projector from G0 only after GPU5 release, exports a new G1-GPU5 bundle without touching G1/G200, then runs fixed 96-frame G0/G200/G1 native controls. This is a queued executable owner, not a claim that G1 has completed.

2026-10-19 — Re-audited the exact clean G2 manifest after launch. The read-only audit found 583,251 rows and unique IDs, zero MOT20-05 rows, zero RMOT placeholder questions, and no cross-source absolute media-path overlap. Its raw MOT20 validity inventory still names the excluded MOT20-05 directory because the audit scans the source tree; the clean row filter remains MOT20-01/02/03 only. Evidence: `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/g2_clean_data_contract_audit_v2.json`. No active process or checkpoint was modified.

2026-10-19 — Clean G2 deployment export handoff registered: queue_g2_clean_bundle_export.py is the sole CPU-only downstream owner. It waits for the exact clean G2 terminal receipt and the GPU5 G1 queue receipt, then exports a fresh 569-tensor GroundAnything-DLM-G2-CLEAN-V2 bundle with the manifest SHA and immutable source check. It refuses overwrite and records explicit dependency failure states; no detector or evaluation claim is made before G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED.
- 2026-10-20 — GroundJEV dependency preflight passed without allocating a GPU or
  starting a process. The G1 plan/trainer, G2 bundle queue, independent held-out
  queue, OneJev recovery queue, registered model/checkpoint/manifest paths and
  external OneJev source all exist. All four owner states parse and report
  waiting states with `failed=null`. Receipt:
  `docs/GROUNDJEV/DEPENDENCY_PREFLIGHT_20261020.{json,md}`.
- 2026-10-20 22:59 +0800: the protected G2 Clean owner remains live on
  physical GPUs 3/5 and advanced to event step 10,710/18,227. The final
  receipt is still absent, the four dependency queues remain live and waiting,
  and the measured linear remainder is approximately 23.9–24.8 hours.

2026-10-20 23:02 +0800 — GroundJEV status refresh: protected G2 Clean torchrun remains live on physical GPUs 3/5 and reached event step 10726/18227 with finite loss 0.224377 and 12.829 GiB peak memory; final receipt absent and latest durable checkpoint remains adapter-step010500/step10500. G1, G2 bundle export, independent GPU7 held-out, and full-source OneJev resume owners remain live with failed=null and waiting states. No active MOT-only optimizer process is present. This is a runtime evidence refresh; no metric claim or protocol change.
2026-10-20 23:04 +0800 — GroundJEV TrackJEV runtime preflight completed in the registered OneJev Python 3.10 environment: compileall passed for GroundJEV code/scripts, and decision-layer, model, and native adapter self-tests all returned PASS. No GPU was allocated and no tracking metric claim was made; receipt is docs/GROUNDJEV/TRACKJEV_RUNTIME_PREFLIGHT_20261020.{json,md}. The protected G2 owner remained untouched.

2026-10-20 23:05 +0800 — Read-only G2 Clean live-event sanity passed through step 10746/18227: 10746 events strictly increasing, all loss/gradient/memory/LR values finite, and every event retained the registered 8 micro-batch / 16 global-sample contract. Final receipt remains absent; no metric claim. Receipt: docs/GROUNDJEV/G2_LIVE_EVENT_SANITY_20261020.{json,md}.

2026-10-20 23:06 +0800 — Independent G2 held-out manifest preflight passed for MOT20-05 validation-half (1657), RMOT single0007 (800), RMOT multi0007 (800), and TAO OVMOT video105 (40): task flags, frame contracts, monotonic unique frame numbers, and all local image paths verified. Queue still performs exact TRAIN overlap audit before inference; no metric claim. Receipt: docs/GROUNDJEV/G2_HELDOUT_MANIFEST_PREFLIGHT_20261020.{json,md}.
2026-10-20 — Fixed `tools/groundjev/validate_full_source_cpu_completion.py` runtime portability: registered CPU-plan receipts and source manifests live under `/data2/user/LocateMOT_GroundJEV` while the project worktree is `/data2/user/LocateMOT`. The validator now resolves project/runtime artifacts deterministically and reports the terminal receipt path. Targeted rerun passed: 459027 rows, zero drops, 28690 planned updates, world size 4, exact serial/parallel plans, and RMOT exposure audit retained; actual optimizer updates remain 0 for this CPU validation.

2026-10-20 23:10 +0800 — Read-only resource gate audit: registered OneJev resume requires exclusive GPUs 4/5/6/7; the gate is not satisfied because existing compute PIDs remain on requested devices (GPU5 also hosts protected G2). Free memory is above 32 GiB but sharing was not enabled; no foreign or GroundJEV process was stopped. Receipt: docs/GROUNDJEV/RESOURCE_GATE_AUDIT_20261020.{json,md}.
2026-10-20 — Improved GroundJEV waiting-owner observability without changing the exclusive GPU contract: the GPU7 independent held-out queue and four-GPU OneJev resume queue now persist every resource probe and an explicit `resource_gate` record (`ready`, rule, `shared_memory_mode=false`) before sleeping. Both scripts passed py_compile and git diff --check; active owner processes were not restarted.
2026-10-20 23:13 +0800 — G1 GPU5 tracking200 input/schedule preflight passed in the registered GroundJEV runtime without GPU allocation: 112 rows, 56 pairs, 72 unique images, 884 native proposals, 628 aligned regions; the GPU5 plan exactly matches the preserved GPU6 reference schedule and retains identity-loss coefficients 0.1/0.1. No optimizer update or metric claim. Receipt: docs/GROUNDJEV/G1_GPU5_INPUT_PREFLIGHT_20261020.{json,md}.
2026-10-20 23:14 +0800 — Registered existing real-input evidence: OneJev-0.8B loaded on a real MOT17 frame (GPU6, 9.25 s load, 0.287 s decision, 3.12 GB peak) and the preserved native DLM/OneJev four-frame MOT17-02 functional receipt covers frames 302–305 with strict wire-error and no-fabricated-score guards. These are plumbing checks only, not tracking or G1/G2 metric claims. Receipt: docs/GROUNDJEV/TRACKJEV_REAL_INPUT_EVIDENCE_20261020.{json,md}.

2026-10-20 23:15 +0800 — Read-only G2 durable checkpoint protection passed at step 10500: latest_training_state.pt and adapter-step010500.pt are present, manifest SHA matches f682519a…, optimizer/scheduler/sampler/per-rank RNG fields are retained, and final receipt remains absent. Receipt: docs/GROUNDJEV/G2_DURABLE_CHECKPOINT_PROTECTION_20261020.{json,md}.

2026-10-20 — Generated current GroundJEV goal completion matrix from live receipts/owner states. It explicitly leaves G2 terminal training, G1 GPU5 adaptation, bundle export, independent held-out metrics, and full-source OneJev terminal receipt incomplete; no preliminary result was promoted. Receipt: docs/GROUNDJEV/GOAL_COMPLETION_MATRIX_20261020.{json,md}.

2026-10-21 — GroundJEV single-backbone coupling Phase A started in isolated worktree `research/groundjev-coupled-single-backbone-20261021` while formal G2 Clean torchrun remained protected on GPU3/GPU5. Added `SharedGroundState`, `G2FeatureBridge`, official-grid region pooling, dynamic Identity Choice (MATCH/REACTIVATE/NEW/NO_MATCH/REJECT), typed admission/lifecycle/memory heads, candidate-local multi-birth Hungarian assignment, safe lifecycle controller, finite causal identity memory, assignment-conditioned memory writer, joint masked losses, and an opt-in memory-token projector. The bridge loaded the protected G2 step11000 projector (adapter SHA f2b7199f57dae156a05e1efcadfa04b275a7f0399187b264e0b21ddc1b58409b) into the official GroundAnything visual path; one real MOT20-03 TRAIN frame used one visual forward, feature dim 2560, 27x36 grid, typed readout backward finite with 1,980,425 trainable readout parameters and gradient norm 15.877484321594238. Receipt: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_a_g2_step11000/phase_a_receipt.json`. A two-frame MOT20-03 causal engineering replay also completed with exactly two visual forwards and serialized memory; candidate boxes were fixed annotation inputs and no metric claim was made. Five stdlib contract tests, compileall and diff checks passed. Added stable SHA-derived Yes/No corpus v2 and a 100-row internal joint-corpus smoke manifest; no formal choice/YesTrack training or MOT/RMOT/OVMOT metric claim yet. Scientific gain remains unproven; Phase B pilot and Phase C memory-conditioned Grounding remain pending.
2026-10-21 — Fixed a real Phase-B pilot failure in dynamic identity-slot construction: native causal memory exports use `id`, while tracker proposal dictionaries use `track_id`; the adapter now accepts `track_id`, `persistent_track_id`, or `id` and preserves lost-age/prior fields. Re-ran the pilot without touching formal G2. The real G2 step-11000 visual path remained frozen; six legal native TRAIN examples produced six visual forwards and 20 typed choice+Yes/No readout updates. Total loss fell 3.1731844→2.1330864, identity loss 2.2314405→1.8834045, Yes/No loss 0.9417437→0.2496820, all values and gradients finite; peak allocated memory 3,072,477,696 bytes, checkpoint SHA `4707d52812cc8ce5d3ebecd35049b161a81726686456c9c2a940330e8336b101`. This is a bounded learnability pilot only, not an HOTA/IDF1/IDSW/TETA or YesTrack gain. Receipt: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_b_yesno_pilot_v4/phase_b_receipt.json`.
2026-10-21 — The first B0/B1/B2 run was rejected as protocol-biased because its fixed 24 examples were all identity-positive. Preserved receipt `phase_b_ablations_v1`; no claim was made. A second stratified run (`phase_b_ablations_v2`) used the right labels but different seeds and was also not promoted. The final run (`phase_b_ablations_v3`) used six deterministic strata (identity +/- , lifecycle +/- , native verifier +/-), the same 24 legal TRAIN examples, the same initialization seed `20261021`, and 200 updates per arm. B0 Choice-only loss 2.6754823→2.4738863; B1 Choice+Yes/No total 3.4273107→2.5620224 and Yes/No 0.7518284→0.0829904; B2 Choice+Yes/No+lifecycle/admission-proxy total 4.6023855→2.6426239, Yes/No 0.7518284→0.0812297, lifecycle 1.8435760→0.0000751. No memory UPDATE/HOLD labels were available and were masked. Receipt: `docs/GROUNDJEV_COUPLED/PHASE_B_ABLATIONS_RECEIPT_20261021.json`; checkpoints remain external with recorded SHA values.
2026-10-21 — Loaded the final same-seed B0/B1/B2 readouts into independent causal runtimes and replayed MOT20-03 TRAIN frames 83–90 with identical fixed annotation candidate boxes. The three arms shared one G2 visual state per frame (8 visual forwards total), while each maintained its own memory. Raw assignments were B0: REJECT11/NO_MATCH6/NEW1/existing6/final1 track; B1: NO_MATCH9/NEW3/existing12/final3 tracks; B2: REJECT15/NEW2/existing7/final2 tracks. This proves shared-feature causal execution and exposes admission/decision calibration differences; it is not a tracking metric or accuracy gain. Receipt: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_b_replay_v2/phase_b_replay_receipt.json`; compact summary in `docs/GROUNDJEV_COUPLED/PHASE_B_REPLAY_SUMMARY_20261021.json`.
