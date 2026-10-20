Warning: truncated output (original token count: 115321)
Total output lines: 2951

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
- Result: prediction and evaluation completed with 94 pr…95321 tokens truncated…mpute sameglobalstep andsamerank co-occurrences plusoptimisticfull-rowtokensavingupperbound fortheconservativeexact-metadata cachekey. Epoch repeatedcontexts alone arenotavailableacrosschangingweights; notgeneralboundoverallpossiblepixel-equivalences oractualthroughput. Allrows/familyfilters/order/seeds unchanged. Persist only≤128KiBscalarreportfromexistingfullsource/reportreservation, no native feature/activation bank or newoverallcacheallocation. This audit decideswhether a future exact branching implementation merits priority forcurrentrecipe; donotretargetrunningtraining.

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
