Warning: truncated output (original token count: 115710)
Total output lines: 2959

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

Previous goal turn made concrete progress: complete original KITTI/BDD F0 measurements, actual temporal supervision plus gated 6k MOT-only implementation, current eight-GPU BF16/causal/GT/resume proofs, source commits/backups and verified GitHub publications. Current source30c7a56 is clean; original bounded perception owner11636 and MOT follow-on38396 are confirmed live. Read OVMOT ada…75710 tokens truncated…ral teacher preparationv1 failed its final16MiBmetadata cap:22,748,803bytes, dominated by two10.14MB copies of the immutable9.96MB source mixture. No GPU/newoptimizer update started. Preserve failedinputs/code/terminalreceipt. Smallest fix: add explicit indexed-source manifest protocol and reference originalselected20000rows instead of copying bodies; newteacher4000refs remain. V2 keeps individual16MiBcap, separately reserves24MiB for failedinputpreservation; globalupper26769302494+40MiB=26811245534<25GiB. No samples/targets/seeds/scorerschanged.

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

2026-10-21 — GroundJEV redirection checkpoint: verified project root `/data2/user/LocateMOT` and protected formal G2 Clean. The live two-rank G2 owner reached event step 11735/18227 with finite loss 0.210247, gradient norm 9.833711, peak 12.828929 GiB; latest durable checkpoint remains step11500 and the 500-step window is 11.617 s/step (linear remainder about 20.9 h). G0 MOT20-05 registered 1657-frame validation-half export remains terminal; its fresh native-source control is retained without a full TEST claim. No live independent MOT-only optimizer process exists. The historical four-GPU independent OneJev resume was waiting before launch at durable step3000, so it was safely paused and its checkpoint/optimizer/RNG/config/logs preserved; a code-level pause marker guard now blocks accidental auto-start unless `--resume-paused` is explicitly supplied. G1 GPU5, clean bundle export, and independent held-out remain waiting on their registered dependencies. No formal G2/G1/held-out performance claim was made. Evidence: `docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021.{json,md}` and the queue owner state.

## 2026-10-21 — corrected real three-task OneJev source registration

The historical full-source manifest was rejected for the independent comparison because its source inventory included MOT20-05 rows. I streamed and audited a replacement manifest at `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/real_three_task_joint_v1/`: 131,214 native MOT identity rows, 346 native RMOT membership/presence rows, and 430 registered OVMOT Base teacher-control rows (131,990 total). Duplicate IDs, train split, family filters, row counts, media paths, forbidden MOT20-05/test markers, and source SHA-256 values passed. The registered TAO native archive (40 frames/35,228 candidates) has no category supervision, so native OVMOT training rows remain pending and no relabeling or optimizer launch was performed. The old independent four-GPU OneJev queue remains paused.

## 2026-10-21 — Phase C causal memory input contract

In the isolated `research/groundjev-coupled-single-backbone-20261021` worktree, added an opt-in `inject_identity_memory` contract. It inserts projected causal identity tokens before the registered current-answer index, preserves the original prompt tensors exactly when disabled, and returns the offset needed to shift DLM labels. Eight focused CPU contract tests passed, including projector gradient flow and insertion order. This is an engineering input contract only; the GroundAnything decoder hook, real GPU C0/C1/C2 replay, generated-box recall, and HOTA/IDF1 effects remain unmeasured.

## 2026-10-21 — corrected three-task OneJev owner registered

The historical independent four-GPU resume remains paused. A separate owner `real_three_task_joint_v1_queue` was registered and started in a waiting state. It uses the audited 131,990-row manifest, a new run directory, and the dedicated OneJev environment; it will not claim GPUs 4/5/6/7 until G2 Clean, G1 GPU5 adaptation, bundle export, held-out evaluation, and an exclusive resource window are terminal. The run is explicitly a Base OVMOT teacher-control comparison while native TAO category rows remain pending; it cannot be reported as the final single-backbone GroundJEV result.

## 2026-10-21 — joint queue configuration regression fixed

The dedicated OneJev parser rejected four audit-only keys accidentally placed in the runtime training JSON. I removed those keys from `real_three_task_joint_v1_runtime.json` and kept the same provenance in the queue state and research records. The registered `.venv-onejev-py310` parser now accepts the config and resolves the corrected source manifest, one epoch, and global 16-row accumulation.

## 2026-10-21 — three-task CPU preflight found explicit overlength rows

The registered OneJev CPU plan rendered all 131,990 rows in 267 seconds. At the 32,768-token training contract, 129,891 rows are trainable and 2,099 high-candidate MOT identity rows exceed the limit (p50 12,720; p90 25,689; p99 47,023; maximum 96,734). The new queue records this exclusion explicitly and cannot report full-source exposure. A separate compacting/long-row course remains required.

## 2026-10-21 — compact zero-drop three-task source v2
The v1 native MOT/RMOT/OVMOT source plan was valid but excluded 2,099 high-candidate MOT identity rows over the 32,768-token limit. The cause was repeated dense N×N IoU and center-distance CSV matrices in every MOT row, not missing media or a changed candidate source. Added the explicit `compact_competition_v1` manifest transform: native JSONL remains immutable; exact row IDs, targets, media paths and candidate boxes are preserved; all positive-IoU edges retain native rounded IoU and center distance; zero-IoU geometry is derivable from the retained boxes. The v2 CPU plan rendered all 131,990 rows with zero drops (p50/p90/p99/max 7,939/12,173/16,050/28,156 tokens; 8,250 world-4 steps), with no GPU or optimizer updates. The old v1 waiting owner was superseded before any training; a separate v2 owner is waiting on G2/G1/bundle/held-out gates and exclusive GPUs. Native TAO category rows remain pending, so this is still a registered Base teacher-control course rather than a final OVMOT claim.

## 2026-10-21 — native YesTrack v2 queued behind an exclusive GPU7 gate

Registered the real native YesTrack corpus with 726 train rows and 390 fixed
validation rows (train SHA
`065337d2fedca2377d02b94e57c9eb3778aeca09aeb1a8b9463429fa15bbee66`, validation
SHA `d63a9fb0d54b8a1dc5d12b0960a48cb75c56ab786671f955fb317e196b3033d8`). The
CPU rendering plan passed with zero rows over 32,768 tokens (max 26,159; 91
planned optimizer steps per epoch, 273 over three epochs). Added a real GPU
owner that waits for GPU7 to have no
compute PID and at least 32 GiB free, trains native multimodal LoRA for three
epochs, then evaluates frozen base and tuned adapter on identical held-out rows
using AUROC, PR-AUC, accuracy, Brier, ECE, false rejection and false
acceptance. The owner does not stop other users and writes terminal receipts
only after training and both evaluations succeed. At registration GPU7 was
occupied, so no YesTrack metric is claimed; prior 24-row readout controls remain
engineering evidence only. Tracking HOTA/AssA/IDF1/IDSW still requires a later
fixed-candidate physical-baseline integration.

## 2026-10-21 — live G2 Clean checkpoint refresh at step 12002

A single authoritative read of the formal event log found G2 Clean at event step 12002/18227, with finite loss 0.298694, gradient norm 10.215737, peak memory 12.829 GiB, and the registered 8-micro-batch/16-sample contract. The periodic adapter checkpoint is `adapter-step012000.pt` and `checkpoints/latest_training_state.pt` is present; the terminal receipt is still absent. The recent 500-step rate is 11.615 s/step, a linear remainder of approximately 20.1 hours. G1, bundle, held-out, YesTrack, and zero-drop three-task v2 remain waiting owners; no metric claim was promoted and no protected process was modified. Evidence: `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12002.{json,md}`.

## 2026-10-21 — process-scope audit confirms no live MOT-only/J1 optimizer

A fresh process-table audit found G2 Clean as the only active GroundJEV optimizer (event step 12012/18227 on GPUs 3/5). G1 GPU5, deployment bundle, independent held-out, YesTrack and zero-drop three-task v2 are live dependency/resource waiters. No independent MOT-only or legacy J1 OneJev optimizer is live; historical checkpoints and logs remain preserved. Foreign PRISM jobs were untouched and no process was stopped. Evidence: `docs/GROUNDJEV/PROCESS_AUDIT_20261021.{json,md}`.

## 2026-10-21 — G2 step-12000 recovery checkpoint validated

Read the protected G2 `latest_training_state.pt` and `adapter-step012000.pt` with the registered local trainer environment. Both load successfully and carry format `groundanything-g2-projector-qwen3-lora-v2-strict`, step 12000, manifest SHA `f682519a…06e71`, 293 trainable tensors/signature entries, optimizer and scheduler state, RNG state, two rank RNG paths, sampler position 96000 and world size 2. The terminal receipt is still absent, so this proves recoverability only and no performance claim. PyTorch 2.6's default weights-only reader rejects the stored NumPy RNG object; the controlled audit used trusted `weights_only=False` and did not modify files. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_VALIDATION_20261021.{json,md}`.

## 2026-10-21 — G2 Clean dataset scope audited

The protected G2 Clean manifest contains 583,251 TRAIN rows from AerialMind (69,358), AnimalTrack (11,543), DanceTrack (41,796), MOT17 (15,948), MOT20 (5,616), RMOT26 (294), Refer-KITTI-V2 (379,341), SportsMOT (28,574) and TAO-Amodal (30,781). It is GroundAnything perception adaptation (projector + Qwen3 LoRA), not J1/OneJev identity training. MOT20-05 is explicitly held out; BURST is excluded for lack of a legal TRAIN image root; TAO novel annotations are excluded. Manifest SHA is `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`. Evidence: `docs/GROUNDJEV/G2_CLEAN_DATASET_SCOPE_20261021.{json,md}`.

## 2026-10-21 — GroundJEV Phase A real G2 single-backbone smoke

Implemented the isolated `locatemot/groundjev/coupled` path: frame/model-provenance checked shared state, G2 feature bridge, typed dynamic identity slots, dot-product JEV readouts, explicit Yes/No/admit/lifecycle/memory heads, causal identity memory, safe lifecycle control, assignment-conditioned memory writes and a one-to-one runtime. CPU backward and causal runtime tests passed. A real GPU smoke loaded the official GroundAnything visual module from the registered G2 base checkpoint (`3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`), encoded one MOT17 image once, pooled two real boxes into 2x2560 region tokens, and completed finite typed loss/backward/optimizer-step: total loss 4.179936, 3.07 GiB peak allocated, 14.43 s, bridge forward count 1, independent VLM instances 0. This is engineering evidence only; no tracking or scientific gain is claimed, and contextual language-state return plus identity-conditioned grounding remain pending. The first py310 attempt failed before model construction because `pybase64` is absent; the successful run used the existing GroundAnything serving environment and the same immutable checkpoint. Evidence: `docs/GROUNDJEV/PHASE_A_REAL_SMOKE_20261021.{json,md}`.

## 2026-10-21 — GroundJEV Phase B choice/Yes-No pilot completed

A bounded real pilot reused 32 MOT17-02-FRCNN TRAIN frames and 992 prior native GroundAnything proposal boxes, re-pooled once per frame with the G2 base visual checkpoint SHA `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`. The frozen G2 evidence fed typed Identity Choice, Admit, Yes/No, Lifecycle and Memory readouts; 24 frames received 192 optimizer updates and the final 8 frames were a same-video temporal development holdout. Identity accuracy changed train 0.0618->0.6720 and temporal-dev 0.0363->0.6492; admit accuracy changed train 0.1022->0.9247 and temporal-dev 0.1331->0.9153. An actual eight-frame replay without teacher memory produced three persistent predicted tracks with one cached G2 feature forward per frame. This is a teacher-forced engineering control, not an independent HOTA/generalization result; language hidden-state coupling and identity-conditioned grounding remain unmeasured. The first replay failed on the legacy/new memory field mismatch and was fixed without changing data or budget. Evidence: `docs/GROUNDJEV/PHASE_B_PILOT_20261021.{json,md}`.

Checkpoint receipt finalized for the Phase B pilot: `adapter-step000192.pt`, SHA256 `c96f27c9c8de5a01a36036c78c4e4a36c467bb96a0ca5f7d0eee27d93e32c904`; the binary remains local under the registered output directory and is not pushed to GitHub.

## 2026-10-21 — native G2 Clean hidden-state probe completed

The protected G2 Clean base plus `adapter-step012000.pt` (adapter SHA `f0dfe4f5efd6bd2f0e8ebe8f224ae060e8b0422b36571697fb4fa9efa6d85ae2`, step 12000, manifest SHA `f682519a…06e71`) completed the native GroundAnything DLM clean path on real row `mot17_MOT17-02-DPM_000001_dense_clean_v2` from MOT17 TRAIN. The real image was encoded once; the clean embedding and hybrid-language hidden state were both `[1, 1152, 2560]`, with 1,008 image-pad hidden vectors. No independent VLM instance was created. The probe used the formal BF16 autocast boundary and the existing process-local Transformers 5.17 `cache_position` compatibility shim. The first dtype and API failures were preserved and fixed at the probe boundary. This establishes native hidden-state availability for the shared-backbone work only; it is not a tracking metric, identity gain, grounding improvement, or generalization result. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_A_NATIVE_HIDDEN_PROBE_20261021.{json,md}`.

## 2026-10-21 — native G2 clean DLM coupled Phase C smoke completed

The new `NativeG2CleanDLMProvider` restores the protected G2 step-12000 adapter in memory, renders one real MOT17 TRAIN row, and uses one native clean DLM forward to produce both 1,008 image hidden tokens and a `[1, 2560]` non-image contextual token. The same-backbone bridge pooled two real candidate boxes, fused the contextual token into the typed GroundJEV readout, and completed a finite GPU backward plus optimizer step: total loss 5.533803, clipped gradient norm 73.300224, 12.059 GiB peak allocation, 16.91 s, bridge/native forward counts both 1, independent VLM instances 0. This proves the visual+clean-language coupling interface runs on a real checkpoint; it is still an engineering smoke with no HOTA/IDF1, grounding recall, identity gain or generalization claim. PyTorch emitted a BF16/FP32 RMSNorm fallback warning in the frozen language path; it did not produce a non-finite value and remains recorded in the receipt. The formal G2 run directory was untouched. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_NATIVE_DLM_COUPLED_SMOKE_20261021.{json,md}`.

## 2026-10-21 — native G2 context readout C1 pilot completed

A bounded eight-frame MOT17-02-FRCNN pilot used G2 step 12000 with the official DLM frozen. Each of 8 real frames had one native DLM forward; the downstream GroundJEV typed readout received the pooled image regions plus same-frame clean-language context. Four frames were train and four same-video temporal development, with previous-frame GT-matched region tokens as explicit teacher-forced causal slots. Twelve readout updates changed identity accuracy train 0.032258→0.725806 and temporal-dev 0.032258→0.403226; admit accuracy changed train 0.911290→0.911290 and temporal-dev 0.790323→0.911290. This is a bounded learnability control only: no HOTA/IDF1, independent-video or final tracking claim. The first attempt completed the loop but failed at receipt writing because the output directory was absent; the directory fix was the only rerun change and the failed attempt is excluded. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_NATIVE_CONTEXT_PILOT_20261021.{json,md}`.

## 2026-10-21 — active goal snapshot at G2 step 12153

The protected formal G2 Clean owner is alive on GPUs 3/5 at event step 12153/18227; recent event rate is 11.614 s/step and the linear remaining estimate is 19.6 h. The latest durable checkpoint is `adapter-step012000.pt`, with no terminal receipt. G0 MOT20-05 validation-half export is terminal for 1,657 frames (HOTA 0.073346961, IDF1 0.059598451, IDSW 1933); this remains a validation-half control rather than a full TEST claim. G1, bundle, held-out, YesTrack and compact three-task OneJev owners are alive and waiting on their registered dependencies/resource gates with no recorded failure; compact OneJev optimizer updates remain 0. The process audit found no live MOT-only optimizer. Evidence: `docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12153.{json,md}`.

## 2026-10-21 — merged Phase C causal memory input contract

Merged the isolated single-backbone memory-input implementation into the GroundJEV evidence branch. `MemoryTokenProjector` used by the typed readout remains compatible; the new `IdentityMemoryTokenProjector` and `inject_identity_memory` insert only causal identity tokens before the registered answer index, return the insertion offset for label shifting, and leave the disabled path tensor-identical to the G2 baseline. The focused contract tests passed in the serving PyTorch environment (3 tests) and the existing Phase A core functions passed (3 checks); coupled sources compile. This is an input-contract result only: the official DLM decoder hook, generated-box recall and C0/C1/C2 HOTA/IDF1 evaluation remain pending.

## 2026-10-21 — blocked accidental teacher-only OVMOT launch

The registered real three-task v2 manifest contains 131,214 native MOT rows, 346 native RMOT rows, and 430 OVMOT Base teacher-control rows; its 40-frame native TAO archive has no category labels and is validation-only. The waiting four-GPU OneJev owner had zero optimizer updates. I stopped that waiting owner before any launch, preserved its state snapshot, and marked it `PAUSED_FOR_NATIVE_OVMOT_ROWS`. The queue gate now requires an explicit native OVMOT TRAIN candidate/annotation join (`native_ovmot_training_rows=true`, `ovmot_training_mode=native`) and returns `WAITING_NATIVE_OVMOT_ROWS` otherwise. No three-task training or OVMOT gain is claimed; the historical source course remains preserved.
