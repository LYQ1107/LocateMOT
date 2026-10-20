Warning: truncated output (original token count: 125539)
Total output lines: 3422

# LocateMOT Research Log

Concise experimental log (latest first).

## 2026-10-21 — G2 event-integrity audit through step 12650

- Read the formal event log without altering the trainer. The completed audit
  cutoff contained 12,650 unique, contiguous steps from 1 through 12,650,
  with zero missing IDs and zero non-finite values across the recorded loss,
  gradient, memory and learning-rate fields.
- Twenty-six explicit checkpoint events were present through the durable
  step-12500 checkpoint; `final_receipt.json` remained absent. The live log
  then advanced to step 12651 with finite loss/gradient values.
- Decision: preserve the current two-rank run and its recovery state. This is
  numeric/resumability evidence, not a performance claim.
- Evidence: `docs/GROUNDJEV/G2_EVENT_INTEGRITY_AUDIT_20261021_STEP12650.md`.

## 2026-10-21 — downstream launch-gate audit

- Read the five G2-dependent entrypoints and confirmed their exact terminal
  contract: status `G2_EXACT_COVERAGE_TRAINING_COMPLETE`, step `18_227`, and
  clean manifest SHA `f682519a…06e71`; the bundle exporter also requires the
  terminal adapter file.
- Confirmed YesTrack is intentionally independent of G2 training and remains
  gated by real GPU7 exclusivity and its native split, so it cannot silently
  fall back to a CPU mock.
- No gate defect or premature launch was found. The static audit changed no
  runtime state and no benchmark claim. Evidence:
  `docs/GROUNDJEV/DOWNSTREAM_GATE_AUDIT_20261021.md`.

## 2026-10-21 — GroundJEV G2 step-12642 progress snapshot

- Evidence: the formal event log reached step `12642/18227` at
  `2026-10-21 05:17:02 +08:00`, with finite loss `0.17345873918384314` and
  finite gradient norm `10.642035484313965`.
- Gate state: `final_receipt.json` is absent. G1, deployment bundle,
  independent held-out, native TAO and YesTrack owners remain alive in their
  registered waiting states; no downstream metric or optimizer update is
  claimed.
- Decision: continue protecting the two-rank G2 run on physical GPUs 3/5 and
  preserve all checkpoints and historical G2-PL outputs.
- Boundary: this is runtime evidence, not a MOT/RMOT/OVMOT performance result.
  See `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12642.md`.

## 2026-10-21 — GroundJEV G2 step-12633 progress snapshot

- Evidence: the formal event log advanced to step `12633/18227` at
  `2026-10-21 05:15:19 +08:00`, with finite loss `0.28367656865157187`, finite
  gradient norm `10.239775657653809`, and peak memory `12.8289 GiB`.
- Gate state: `final_receipt.json` is still absent. G1, deployment bundle,
  independent held-out, native TAO and YesTrack owners remain alive but in
  their registered waiting states; no downstream optimizer update or metric is
  claimed.
- Decision: continue protecting the two-rank G2 run on physical GPUs 3/5 and
  preserve all checkpoints and historical G2-PL results.
- Boundary: this is runtime evidence, not a MOT/RMOT/OVMOT performance result.
  See `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12633.md`.

## 2026-10-21 — GroundJEV G2 step-12613 progress snapshot

- Hypothesis/contract: preserve the formal two-rank G2 Clean run and open all
  dependent G1, deployment-bundle, held-out, native-TAO and YesTrack gates only
  after the exact terminal receipt.
- Evidence: the authoritative event log reached step `12613/18227` with
  finite loss `0.2061017351`; `adapter-step012500.pt` and
  `latest_training_state.pt` are present, while `final_receipt.json` is absent.
- Decision: retain the live run and all waiting owners. No downstream optimizer,
  evaluation, label promotion, or historical G2-PL overwrite was performed.
- Boundary: this is a runtime progress record, not a MOT/RMOT/OVMOT metric or
  a claim that the final G2 detector exists. See
  `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12613.md`.

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
  No…105539 tokens truncated…nDLMProvider` restores the protected G2 step-12000 adapter in memory, renders one real MOT17 TRAIN row, and uses one native clean DLM forward to produce both 1,008 image hidden tokens and a `[1, 2560]` non-image contextual token. The same-backbone bridge pooled two real candidate boxes, fused the contextual token into the typed GroundJEV readout, and completed a finite GPU backward plus optimizer step: total loss 5.533803, clipped gradient norm 73.300224, 12.059 GiB peak allocation, 16.91 s, bridge/native forward counts both 1, independent VLM instances 0. This proves the visual+clean-language coupling interface runs on a real checkpoint; it is still an engineering smoke with no HOTA/IDF1, grounding recall, identity gain or generalization claim. PyTorch emitted a BF16/FP32 RMSNorm fallback warning in the frozen language path; it did not produce a non-finite value and remains recorded in the receipt. The formal G2 run directory was untouched. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_NATIVE_DLM_COUPLED_SMOKE_20261021.{json,md}`.

## 2026-10-21 — native G2 context readout C1 pilot completed

A bounded eight-frame MOT17-02-FRCNN pilot used G2 step 12000 with the official DLM frozen. Each of 8 real frames had one native DLM forward; the downstream GroundJEV typed readout received the pooled image regions plus same-frame clean-language context. Four frames were train and four same-video temporal development, with previous-frame GT-matched region tokens as explicit teacher-forced causal slots. Twelve readout updates changed identity accuracy train 0.032258→0.725806 and temporal-dev 0.032258→0.403226; admit accuracy changed train 0.911290→0.911290 and temporal-dev 0.790323→0.911290. This is a bounded learnability control only: no HOTA/IDF1, independent-video or final tracking claim. The first attempt completed the loop but failed at receipt writing because the output directory was absent; the directory fix was the only rerun change and the failed attempt is excluded. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_NATIVE_CONTEXT_PILOT_20261021.{json,md}`.

## 2026-10-21 — active goal snapshot at G2 step 12153

The protected formal G2 Clean owner is alive on GPUs 3/5 at event step 12153/18227; recent event rate is 11.614 s/step and the linear remaining estimate is 19.6 h. The latest durable checkpoint is `adapter-step012000.pt`, with no terminal receipt. G0 MOT20-05 validation-half export is terminal for 1,657 frames (HOTA 0.073346961, IDF1 0.059598451, IDSW 1933); this remains a validation-half control rather than a full TEST claim. G1, bundle, held-out, YesTrack and compact three-task OneJev owners are alive and waiting on their registered dependencies/resource gates with no recorded failure; compact OneJev optimizer updates remain 0. The process audit found no live MOT-only optimizer. Evidence: `docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12153.{json,md}`.

## 2026-10-21 — merged Phase C causal memory input contract

Merged the isolated single-backbone memory-input implementation into the GroundJEV evidence branch. `MemoryTokenProjector` used by the typed readout remains compatible; the new `IdentityMemoryTokenProjector` and `inject_identity_memory` insert only causal identity tokens before the registered answer index, return the insertion offset for label shifting, and leave the disabled path tensor-identical to the G2 baseline. The focused contract tests passed in the serving PyTorch environment (3 tests) and the existing Phase A core functions passed (3 checks); coupled sources compile. This is an input-contract result only: the official DLM decoder hook, generated-box recall and C0/C1/C2 HOTA/IDF1 evaluation remain pending.

## 2026-10-21 — blocked accidental teacher-only OVMOT launch

The registered real three-task v2 manifest contains 131,214 native MOT rows, 346 native RMOT rows, and 430 OVMOT Base teacher-control rows; its 40-frame native TAO archive has no category labels and is validation-only. The waiting four-GPU OneJev owner had zero optimizer updates. I stopped that waiting owner before any launch, preserved its state snapshot, and marked it `PAUSED_FOR_NATIVE_OVMOT_ROWS`. The queue gate now requires an explicit native OVMOT TRAIN candidate/annotation join (`native_ovmot_training_rows=true`, `ovmot_training_mode=native`) and returns `WAITING_NATIVE_OVMOT_ROWS` otherwise. No three-task training or OVMOT gain is claimed; the historical source course remains preserved.

## 2026-10-21 — active goal snapshot at G2 step 12195

A single authoritative read of the protected G2 event log found event step 12195/18227 with finite loss 0.225930949, gradient norm 9.566365, peak memory 12.828853 GiB, and a 500-step rate of 11.618 s/step (linear remainder about 19.5 h). The latest durable checkpoint remains adapter-step012000.pt and the terminal receipt is absent. G0 MOT20-05 remains a complete 1657-frame validation-half natural-visibility control with HOTA 0.073346961 and IDF1 0.059598451; it is not a full TEST claim. The compact three-task queue is paused before optimizer launch because native OVMOT TRAIN rows are missing; its optimizer update count remains zero and the guard requires native OVMOT provenance/mode. No final G1/held-out/OneJev/YesTrack result was promoted. Evidence: docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12195.{json,md}.

## 2026-10-21 — native OVMOT TRAIN join pipeline prepared

Added `tools/groundjev/prepare_native_ovmot_train_join.py`, a CPU-only joiner for a future complete native GroundAnything TAO TRAIN stream. It validates `/frames/train/` provenance, requires native region features and GT-free model metadata, preserves candidate boxes, assigns official Base category targets only through deterministic one-to-one IoU >=0.5 matching to TAO TRAIN annotations, and emits the existing hierarchical category questions with on-demand real-image crop recipes. The registered TAO TRAIN source contains 18274 images, 54639 annotations and 765 Base categories mapped to the 1203-class vocabulary; 11 categories are explicitly unmapped. The current 40-frame/35228-candidate native TAO archive is validation-only and is rejected by the TRAIN path guard. `py_compile`, one-frame synthetic preflight, and validation-scope rejection passed; no GPU or optimizer update occurred. The compact three-task owner remains paused pending a complete native OVMOT TRAIN archive. Evidence: docs/GROUNDJEV/NATIVE_OVMOT_TRAIN_JOIN_20261021.{json,md}.

## 2026-10-21 — TAO TRAIN native OVMOT scope registered

Registered a CPU-only scope manifest for all 18274 TAO TRAIN images across 500 videos. The runtime manifest contains only absolute image paths and the fixed 1203-category query vocabulary; it carries `training_on_this_scope=true` for an explicit export gate but no GT fields, labels or evaluation data. `stream_tao_native_candidates.py` now requires `--allow-training-scope` for this manifest and rejects validation manifests with that flag; the scope gate was tested without importing the model service. The exporter was not launched because current GPUs remain occupied, so native candidate rows and optimizer updates are still zero. Evidence: docs/GROUNDJEV/TAO_TRAIN_NATIVE_SCOPE_20261021.{json,md}.

## 2026-10-21 — corrected process audit at G2 step 12235

A corrected process-table audit (the first username-filtered attempt was invalid because the account is `lyq`) found the protected G2 parent/ranks alive at 12235/18227 on GPUs 3/5, G1/bundle/YesTrack waiters alive, and no registered MOT-only or legacy J1 OneJev optimizer command. The compact three-task owner remains paused before optimizer updates because native OVMOT TRAIN rows are absent. Foreign GPU jobs were not touched. Evidence: docs/GROUNDJEV/PROCESS_AUDIT_20261021_G2_STEP12233.{json,md}.

## 2026-10-21 — active goal snapshot at G2 step 12241

The protected G2 Clean event log reached 12241/18227 with finite loss 0.195041649; the last 500-step rate is 11.626 s/step and the linear remainder is about 19.3 h. The durable checkpoint remains step12000 and no final receipt exists. G0 remains a 1657-frame MOT20-05 validation-half control (HOTA 0.073346961, IDF1 0.059598451); the compact three-task queue remains paused with zero optimizer updates pending native OVMOT TRAIN rows. Evidence: docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12241.{json,md}.

## 2026-10-21 — GroundJEV fixed-candidate MOT20-03 replay establishes failure boundary

Added a reproducible CPU evaluator for the immutable native GroundAnything candidate/region cache. The current 192-step coupled readout replayed all 128 registered MOT20-03 TRAIN frames (14,784 candidates, 128 cached feature calls) without a second VLM and generated 152 predictions on 76 frames. TrackEval returned HOTA 0.0279006409, DetA 0.0099894107, AssA 0.0779470005, IDF1 0.0052606731, IDSW 1, FP 1, FN 14,524; 11,462 candidates were explicit REJECT. The cache predates the current G2 Clean terminal checkpoint and overlaps TRAIN scope, so this is a fixed-candidate association/runtime diagnostic, not independent G2 or end-to-end evidence. The first evaluator attempts exposed only direct-entry `sys.path` and NumPy-2/old-TrackEval compatibility issues; both were fixed locally without changing metric code or inputs. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_B_FIXED_CANDIDATE_EVAL_20261021.{json,md}` and the registered output receipt.

The same-input historical compact control (checkpoint `rmot_native72_stage_v1/broad_native_training/checkpoint.pt`) returned HOTA 0.336514881, DetA 0.309884333, AssA 0.365702041, IDF1 0.359321689, IDSW 159, FP 119 and FN 9,832. It is an offline multi-component reference, not a single-backbone result. The gap confirms that the coupled readout's 0.027900641 HOTA is a real association/admission failure and should trigger a bounded repair study rather than a long joint finetune.

## 2026-10-21 — Phase C causal memory prefill contract

Added `locatemot/groundjev/coupled/causal_memory.py`, which inserts causal identity tokens only at the terminal prompt boundary used by GroundAnything causal bbox generation, extends attention/position/cache inputs, and preserves the disabled path by tensor identity. A seven-test CPU contract suite passed; a fake language tower showed a changed next-token logit after memory insertion with exactly one prefill call and no second VLM. Real G2 GPU prefill, correct-versus-shuffled memory box generation, and HOTA/IDF1 effects remain pending because GPUs 3/5 are protected by formal G2 and the other devices have foreign compute owners. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_MEMORY_PREFILL_CONTRACT_20261021.{json,md}`.

## 2026-10-21 — authoritative G2 progress refresh at step 12294

A single non-polling audit found the protected formal G2 Clean parent/ranks alive on GPUs 3/5 at event step 12294/18227. The event is finite (loss 0.205467, gradient norm 9.551106, peak memory 12.828834 GiB); `adapter-step012000.pt` and `latest_training_state.pt` remain the latest durable checkpoints and `final_receipt.json` is absent. G1, bundle, held-out, YesTrack and compact three-task owners remain dependency/resource waiters. No protected process or experimental protocol was changed. The causal-memory prefill contract remains engineering-only pending an exclusive GPU window for real generated-box comparison.

## 2026-10-21 — fixed-candidate action-distribution repair completed

The first MOT20-03 fixed-candidate replay showed a candidate pool with 86.85% GT-box coverage at IoU >= 0.5, but the short MOT17-trained readout selected `REJECT` for 77.54% of 14,784 candidates and created only two births. A bounded repair trained only on the first 64 frames of the registered MOT20-03 TRAIN clip, weighted identity-choice classes from that training half with an inverse-frequency cap of 4x, and froze the final 64 frames before optimization. After 180 CPU updates, temporal-dev readout identity accuracy was 0.871720 and admit accuracy 0.903860; predicted-memory runtime replay returned HOTA 0.302815, DetA 0.235861, AssA 0.388886, IDF1 0.257729, IDSW 18, FP 3, FN 5,634 on 1,891 predictions across 61 frames. The 64 feature calls used the cached single-backbone state and no second VLM. This is a TRAIN-scope mechanism repair, not an independent benchmark or G2 Clean result; its scope and teacher-forced classification boundary are explicit in `docs/GROUNDJEV_COUPLED/PHASE_B_FIXED_CANDIDATE_REPAIR_20261021.{json,md}`.
## 2026-10-21 — redirect goal audit at G2 step 12326

A single authoritative audit confirmed the registered MOT20-05 G0 native export is terminal for 1,657 validation-half frames (HOTA 0.073347, IDF1 0.059598) and found no live GroundJEV MOT-only optimizer; no historical process was stopped or overwritten. Formal G2 Clean remains alive on GPUs 3/5 at step 12326/18227, with a 500-step rate of 11.6 seconds per step and a linear remainder of about 19.0 hours. The latest durable checkpoint is `checkpoints/adapter-step012000.pt`, and the final receipt is absent. G1 GPU5, bundle, held-out and YesTrack owners remain waiting; the corrected real three-task OneJev owner remains paused at zero optimizer updates because native TAO TRAIN OVMOT rows are not present. Evidence: `docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12326.{json,md}`.

## 2026-10-21 — cross-disk native OVMOT archive audit at G2 step 12345

A read-only search of `/data0/user`, `/data1/user` and `/data3/user` found no complete TAO TRAIN native candidate archive. The only existing native TAO candidate archive remains a 40-frame `frames/val` stream and is ineligible for TRAIN joining. The corrected real three-task queue remains paused with zero optimizer updates; no validation archive was relabeled and no teacher-only fallback was launched. Evidence: `docs/GROUNDJEV/NATIVE_OVMOT_ARCHIVE_AUDIT_20261021.{json,md}`.

## 2026-10-21 — GroundJEV joint identity corpus rebuild contract and Phase B ablations

Added `tools/groundjev/coupled/build_joint_identity_corpus.py` and its focused
contract tests. The builder normalizes registered MOT/RMOT/OVMOT rows without
inventing boxes or targets, records candidate provenance and causal memory,
rejects future/test evidence, and has an explicit native-only gate that rejects
teacher crops. A real 12-row sample from the registered Phase-B corpus produced
5 MOT, 3 RMOT and 4 OVMOT rows; 8 rows had native candidates and 4 remained
explicit Base teacher/empty-candidate rows, so `native_ovmot_training_rows` was
false. The existing G2 step-11000 B0/B1/B2 bounded readout receipts were
consolidated into `docs/GROUNDJEV_COUPLED/PHASE_B_ABLATIONS_20261021.{json,md}`.
The three 200-update arms and 8-frame causal replay are engineering evidence
only; no official tracking metric or OVMOT native-training claim was promoted.

The native-only corpus gate was then strengthened to reject rows carrying an
explicit `gt_in_model_input`/`GT_IDs_in_model_input` flag even when their
candidate-source string looks native. The focused contract suite now has 12
passing checks; no source rows or training queues were changed.

The Phase-B joint corpus was also audited against all 583,251 G2 Clean TRAIN
rows. There are 678 exact image-path overlaps covering 290 images (221
identity-pair, 119 occlusion-recovery, 338 RMOT); the 430 OVMOT teacher rows
have no exact overlap. The audit reads no official-test labels and marks the
joint corpus as TRAIN-scope controls only. Evidence:
`docs/GROUNDJEV_COUPLED/DATA_OVERLAP_AUDIT_20261021.{json,md}`.

## 2026-10-21 — targeted native TAO archive rescan

Re-read every discovered TAO candidate archive under `/data2/user/LocateMOT_GroundJEV/outputs/groundjev`. The two small candidate controls and the 40-frame streams all resolve to `frames/val`; the larger 40-frame stream has 35,228 candidates and a native feature cache but is explicitly a validation stream. One full Base/Novel stream is empty and failed, and the registered 18,274-image/500-video TAO TRAIN manifest has `native_candidate_export_started=false` and `native_ovmot_training_rows=false`. No complete TAO TRAIN native candidate archive exists, so the corrected three-task queue remains paused at zero optimizer updates. Validation outputs were not relabeled and no teacher-only fallback was promoted. Evidence: `docs/GROUNDJEV/NATIVE_OVMOT_ARCHIVE_RESCAN_20261021.{json,md}`.
## 2026-10-21 — single authoritative G2 progress refresh at step 12417

One non-polling status read found the protected formal G2 Clean ranks alive on GPUs 3/5 at event step 12,417/18,227 with finite loss `0.1803909168`. The latest durable adapter remains `checkpoints/adapter-step012000.pt`; `checkpoints/latest_training_state.pt` is present and `final_receipt.json` is absent. G1, the deployment bundle, held-out evaluation and YesTrack remain dependency/resource waiters. The corrected three-task queue remains `PAUSED_FOR_NATIVE_OVMOT_ROWS` with zero optimizer updates. No protected process or experimental protocol was changed. Evidence is the live G2 run directory plus the refreshed matrix.

## 2026-10-21 — TAO TRAIN native export shards registered

Because the all-GPU snapshot was occupied by protected/foreign jobs, no native TAO export was launched. Instead, added `scripts/groundjev/prepare_tao_train_native_shards.py` and registered eight contiguous, disjoint, label-free manifests covering all 18,274 TAO TRAIN images (500 videos) with parent manifest SHA `835adaadff282088599987de4c5bc906e7d8aca826deb7426cda05598e9db3a4`. CPU coverage validation reports 18,274 covered frames, zero missing and zero duplicates. Every shard remains `native_ovmot_training_rows=false`; this only makes the future native stream resumable/parallel and does not unblock training. The first preflight exposed a slice-length assertion bug, which was fixed before accepting the v2 registration; the failed partial directory was preserved separately. Evidence: `docs/GROUNDJEV/TAO_TRAIN_NATIVE_SHARDS_20261021.{json,md}` and the external shard receipts.

Added `scripts/groundjev/merge_tao_native_shards.py` as the only promotion path after streaming. It checks every shard receipt, parent SHA, exact image order, GT-free metadata, and disjoint full coverage before producing a combined candidate archive. A preflight against the registered-but-unstreamed v2 directory rejected the missing shard receipt at shard 0, as intended; no native TRAIN rows or optimizer updates were created.

The merge contract also passed a synthetic four-frame/two-shard GT-free fixture, covering the successful receipt path without touching dataset or GPU state.

## 2026-10-21 — resumable TAO TRAIN native owner started

Added `scripts/groundjev/queue_tao_native_train_shards.py` and synchronized the runtime stream copy so shard-specific GroundAnything and native-feature ports are supported while the default validation ports remain unchanged. The owner is live in `WAITING_G2_CLEAN_SUCCESS`, restricted to physical GPUs 4/6, and treats any existing compute application as unavailable; GPUs 3/5 and 7 are excluded to protect G2/G1 and held-out/YesTrack. It will run the eight native TRAIN shards and strict merge only after the exact G2 terminal receipt. A first background launch was reaped by the terminal wrapper before writing state; the persistent-session launch wrote the waiting state and made no GPU or foreign-job change. Evidence: `docs/GROUNDJEV/TAO_TRAIN_NATIVE_QUEUE_20261021.{json,md}`.

## 2026-10-21 — authoritative G2 refresh at step 12442

The protected G2 Clean ranks remain alive on GPUs 3/5 at event step 12,442/18,227 with finite loss `0.2287370411`; the latest durable adapter is still `checkpoints/adapter-step012000.pt`, the training-state checkpoint is present, and `final_receipt.json` is absent. G1, clean-bundle, held-out and YesTrack owners remain live dependency/resource waiters; the corrected three-task owner remains paused at zero updates because native TAO TRAIN rows are not available. No protected process or protocol was changed.

## 2026-10-21 — native TAO join owner and native three-task launch gate repaired

Added a CPU-only owner for the complete native TAO TRAIN stream and a separate
resumable owner for the future real MOT/RMOT/OVMOT OneJev run. The join owner
waits for the strict eight-shard merge receipt, runs the checked-out
`prepare_native_ovmot_train_join.py` and `prepare_real_three_task_joint_manifest.py`,
and never launches a model or optimizer. Its initial state is
`WAITING_COMPLETE_NATIVE_TAO_STREAM`. The native three-task owner is now live
as a dependency waiter at `WAITING_NATIVE_OVMOT_MANIFEST`; it has launched no
GPU process or optimizer and will apply the joined-manifest plus terminal
G2/G1/bundle/held-out and exclusive-GPU gates later. The historical
teacher-control queue remains paused at zero updates.

During the audit, the join owner was corrected to resolve repository joiners
from `/data2/user/LocateMOT` because the runtime output tree does not mirror
those source files. The paused compact queue was also corrected to accept the
actual bundle terminal status `G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED` while
retaining the historical spelling for old receipts. No data, checkpoint, or
running process was changed. Evidence:
`docs/GROUNDJEV/TAO_NATIVE_JOIN_QUEUE_20261021.{json,md}` and
`docs/GROUNDJEV/REAL_THREE_TASK_NATIVE_QUEUE_20261021.{json,md}`.

## 2026-10-21 — TAO native owner corrected to use a terminal G2 bundle

The TAO TRAIN shard owner was audited before its first GPU launch. Its
original model constant pointed at the published G0 bundle, which would have
made a later native candidate archive impossible to attribute to G2. The owner
now waits for the exact G2 receipt, merges `adapter_projector_final.pt` into
the private `GroundAnything-DLM-G2-CLEAN-NATIVE-TAO-V1` bundle, validates the
569-tensor/export/manifest contract, and only then starts shard services. The
private bundle is never mixed with or written over G0, G1, or the ordinary G2
deployment bundle. The owner process had not started a shard, so it was
restarted with the corrected code while remaining in its G2 wait state.
Evidence: `scripts/groundjev/queue_tao_native_train_shards.py` and
`docs/GROUNDJEV/TAO_TRAIN_NATIVE_QUEUE_20261021.{json,md}`.

## 2026-10-21 — decoupled the terminal G2 bundle from the G1 comparison arm

The clean deployment owner previously waited for both the exact G2 receipt and
G1 GPU5 completion. That ordering could block independent G2 held-out
measurement and native G2 candidate export behind an optional comparison arm.
The owner now waits only for the exact-once G2 terminal receipt, performs the
same CPU-only 569-tensor merge into the separate G2-CLEAN-V2 bundle, and leaves
G1 as an independent GPU5 adaptation/control. The existing owner was still in
its pre-launch wait state, so no model or output was changed; it will be
restarted with the corrected gate. Evidence: `tools/groundjev/queue_g2_clean_bundle_export.py`.

The owner restart also exposed a resumability defect: its old waiting directory
was treated as an existing export and refused to start. The restart guard now
accepts the preserved pre-launch WAITING_G2_AND_G1_TERMINAL state (as well as
WAITING_G2_CLEAN_SUCCESS), refuses live duplicate owners and any partial
bundle, and resumes without deleting the historical state. The corrected owner
is live again in WAITING_G2_CLEAN_SUCCESS.

## 2026-10-21 — native three-task owner no longer logically depends on G1

The new real native MOT/RMOT/OVMOT owner was narrowed to require the native
TAO manifest, terminal G2, independent G2 bundle and held-out receipts. G1
GPU5 adaptation remains an optional comparison receipt; its physical GPU5
ownership is still respected by the four-GPU resource probe. This preserves
G1/G200/G0 controls without making the final-G2 joint route wait on a separate
adaptation arm. The historical paused teacher-control queue was not changed.

## 2026-10-21 — G1 GPU5 waiting owner made resumable

The G1 GPU5 owner had the same pre-launch recovery defect: after a process
restart, its preserved WAITING_G2_CLEAN_SUCCESS directory was treated as an
existing artifact and the owner could not be resumed. The startup guard now
rejects a live duplicate owner and any partial output, but accepts the empty
preserved waiting directory and records `resumed_wait_owner`. The old waiting
owner was stopped and restarted with the registered
`GroundAnything-DLM-TrackProjG1-GPU5-V2` bundle path. It remains at
`WAITING_G2_CLEAN_SUCCESS`; no G1 optimizer or GPU process has started.

## 2026-10-21 — native three-task owner made resumable

The real native MOT/RMOT/OVMOT owner had a weaker restart boundary: a dead
waiting process could rewrite its queue directory without distinguishing a
clean wait from partial training output. Startup now accepts only the known
pre-launch waiting states, preserves the optional runtime config when it was
already created, rejects unexpected files, refuses live duplicates, and records
`resumed_wait_owner`. The existing owner is being restarted only in its
CPU-only `WAITING_NATIVE_OVMOT_MANIFEST` state; no optimizer or GPU process is
started.

## 2026-10-21 — YesTrack waiting owner made resumable

The native YesTrack v2 owner was still waiting for an exclusive GPU7 and had
no optimizer or evaluation artifacts. Its startup guard now resumes only the
known empty `WAITING_GPU7_EXCLUSIVE` queue (and the pre-created `args.json`),
rejects partial queue/training outputs, refuses live duplicates, and records
`resumed_wait_owner`. The waiting owner is restarted without allocating GPU7;
the independent YesTrack run remains pending.

## 2026-10-21 — held-out owner made resumable

The GPU7 G0/G2 independent held-out owner was still CPU-only in
`WAITING_G2_CLEAN_BUNDLE_AND_GPU7`; its directory contained only the current
wait receipt and the preserved historical pre-MOT receipt. Startup now accepts
that exact clean wait state, preserves the historical receipt, rejects any
unexpected partial evaluation outputs, refuses live duplicates, and records
`resumed_wait_owner`. The owner is restarted without allocating GPU7.
## 2026-10-21 — G2 progress evidence refreshed

The protected formal G2 Clean owner remains live on physical GPUs 3/5. The
latest observed training event is step 12,567/18,227 with finite loss
0.3145263; `adapter-step012500.pt` and `latest_training_state.pt` load with
the registered clean manifest SHA. No terminal receipt or final adapter exists
yet, so no downstream bundle, held-out metric, G1 update, or joint optimizer
update is claimed. The waiting owners remain untouched and recoverable.

## 2026-10-21 — G2 progress evidence refreshed (step 12578)

An authoritative event read at 05:04:48 +0800 records the protected formal G2
Clean run at step 12,578/18,227 with finite loss 0.24055790062993765. The two
distributed ranks are still the only active optimizer in this route on physical
GPUs 3/5. The latest durable checkpoint remains `adapter-step012500.pt` with
`latest_training_state.pt`; `final_receipt.json` and a terminal adapter are
still absent. No G1, bundle export, held-out metric, YesTrack update, or native
three-task update is claimed before that terminal receipt. This is evidence of
continued healthy progress only; the prior historical result boundaries remain.

## 2026-10-21 — GroundJEV coupled regression refreshed

The bounded coupled-core regression was rerun after the queue-owner changes: 13
unittests covering joint identity corpus, Phase A core, memory-conditioned input,
fixed-candidate contract, and causal-memory prefill all passed. The registered
G1/G2-bundle/TAO/native-joint/YesTrack queue entrypoints also passed `py_compile`,
and the coupled plus TrackJEV packages passed `compileall`. This verifies the
recoverable launch surface only; it does not promote any model or imply a
tracking metric while G2 lacks its terminal receipt.

## 2026-10-21 — waiting-owner source resolution audit

A read-only `/proc` audit of the seven registered waiting owners confirmed that
all commands run with cwd `/data2/user/LocateMOT`: the G1 and G2-bundle owners
resolve `tools/groundjev/...`, while the TAO-shard, held-out, YesTrack, native
join, and native three-task owners resolve `scripts/groundjev/...` or the
registered join entrypoint from this project root. The sibling runtime checkout
was not used to launch these owners. This closes a reproducibility ambiguity
without changing any owner, checkpoint, or GPU allocation.


## 2026-10-21 — G2 event-integrity audit (step 12594)

The formal G2 event stream contains 12,594 optimizer events from step 1 through
step 12,594 with no step gaps. Loss, gradient norm, memory, learning rates, and
all other recorded numeric fields are finite across the stream. The recent
throughput is stable at 309.8 steps/hour over 100 steps, 311.9 over 500 steps,
and 310.9 over 1,000 steps. Checkpoint events are present at each 500-step
boundary through `adapter-step012500.pt`; the terminal receipt remains absent.
This supports healthy continuation and an estimated 5,633 remaining updates at
roughly 311 steps/hour, but it is not a model-quality or generalization claim.

## 2026-10-21 — tightened G1 post-G2 terminal gate

The resumable GPU5 G1 owner previously accepted a G2 receipt with steps at or
above the registered budget and did not explicitly require the terminal adapter
file. Its post-G2 gate now requires status
`G2_EXACT_COVERAGE_TRAINING_COMPLETE`, exactly 18,227 steps, the registered clean
manifest SHA, and `adapter_projector_final.pt`. A temporary-receipt regression
confirmed that missing adapter and wrong-step receipts are rejected while the
complete contract is accepted. The live waiting owner was not restarted or
modified; this patch protects future recovery and launch paths.


## 2026-10-21 — G2 progress snapshot (step 12610)

The protected G2 Clean run reached step 12,610/18,227 at 05:10:55 +0800 with
finite loss 0.2049703631. The three expected training processes remain live and
`final_receipt.json` is still absent. This snapshot changes only progress
evidence; no downstream model, held-out metric, or generalization claim is
made.

## 2026-10-21 — coupled runtime lifecycle execution repair

Phase-A review found that the single-backbone runtime decoded lifecycle logits
but ignored them: every assigned track was forced to `ACTIVE`, every absent
track only accumulated `LOST`, and an assignment-conditioned memory write could
reactivate a track whose lifecycle decision was `LOST`.  The smallest aligned
fix now executes `ACTIVE/REACTIVATE/LOST` decisions, treats `TERMINATE` on a
matched observation as recoverable `ACTIVE`, blocks memory writes for
`LOST/TERMINATE`, and applies a registered timeout before irreversible
termination of an unmatched track.  A controlled causal replay verified one
birth, a matched `LOST` decision with a held memory write, and timeout
termination; the one-feature-forward and memory-causality regressions still
pass.  This is a runtime correctness repair, not a tracking-performance claim.

## 2026-10-21 — GPU gate for the next real pilot

The single resource audit found all eight A800 GPUs occupied.  Physical GPUs
3/5 remain the protected G2 ranks, GPU7 retains an unrelated compute owner, and
the remaining cards also have active compute workloads; no card has an
exclusive launch window for the Phase-B real-G2 pilot.  The latest durable G2
adapter remains `checkpoints/adapter-step012500.pt` with
`latest_training_state.pt`; the terminal receipt is still not present.  The
pilot was therefore not started on a shared card, and no process was killed or
reused.  This preserves the registered training contract while the existing
G2 completion waiter remains the only long-running observer.

## 2026-10-21 — corrected G0/G2 evidence boundary for early coupled pilots

An audit of the historical Phase-A smoke and Phase-B readout pilot found that
their `checkpoint` was the unadapted `GroundAnything-DLM` bundle and neither
receipt recorded a G2 adapter.  The metrics and loss values are preserved as
G0 native-visual controls only; they are not G2 Clean evidence and are not used
for a G2 performance claim.  The future Phase-A and Phase-B entrypoints now
require both an explicit adapter and the native clean-DLM loader, and record
the adapter/runtime provenance.  This fixes a provenance error without
overwriting the historical control artifacts.

## 2026-10-21 — protected G2 progress snapshot (step 12701)

The authoritative event stream now contains contiguous rows through step
12,701/18,227; the latest row at 05:28:19 +0800 has finite loss
0.2224419505.  The durable checkpoint set still ends at
`checkpoints/adapter-step012500.pt` with
`checkpoints/latest_training_state.pt`, while `final_receipt.json` is absent
and `trainer_contract.json` remains `G2_FORMAL_TRAINING_RUNNING`.  The launcher
is still live.  This is only a healthy-progress record; it does not promote a
checkpoint or claim a tracking metric.

## 2026-10-21 — native three-task training step-contract audit

The registered `queue_real_three_task_native.py` writes `epochs=1` and
`max_steps=0` into its generated config.  The pinned `train/sft.py` contract
interprets a falsy `max_steps` as the full planned epoch
(`steps_per_epoch * epochs`), rather than a zero-step plan.  The queue then
requires the terminal `done` record to report `examples == native manifest
rows` and a non-missing terminal record before declaring completion.  This
keeps the real MOT/RMOT/native-OVMOT run from being mistaken for a CPU-only or
empty optimizer launch.

## 2026-10-21 — G1 GPU5 plan provenance repair

The G1 queue is explicitly bound to physical GPU5, but the older
`native_ground_tracking200_20261014.json` retained a historical
`physical_training_GPU: 7` field.  That plan is preserved unchanged; the G1
queue now references a versioned GPU5 plan with matching physical-device
metadata and an explicit queue-owner provenance note.  The corrected plan will
be read only when its exact G2 terminal gate opens.

The already-running wait-only owner had loaded the old plan path in memory, so
it was safely replaced before any GPU work began (old PID 516321 exited; new
wait owner PID 579251 is `WAITING_G2_CLEAN_SUCCESS`).  Its output directory,
bundle, controls and historical state were preserved; the new owner records
`native_ground_tracking200_gpu5_20261021.json` and still owns no GPU.

## 2026-10-21 — protected G2 throughput snapshot (step 12731)

The formal G2 event stream is contiguous through step 12,731/18,227 with no
non-finite loss, gradient, memory or learning-rate fields.  The measured rates
are 313.73, 312.31 and 311.00 steps/hour over the last 100, 500 and 1,000
steps, respectively.  This leaves 5,496 updates and a linear estimate of
about 17.5–17.7 hours.  `final_receipt.json` and the terminal adapter remain
absent, so this is a throughput/progress record only and does not open any
downstream performance claim.

## 2026-10-21 — G2 durable checkpoint contract audit

The latest durable G2 recovery pair remains intact: `adapter-step012500.pt`
and `latest_training_state.pt` are present at 469,243,990 and 469,247,597
bytes.  The loaded state reports step 12,500, format
`groundanything-g2-projector-qwen3-lora-v2-strict`, the registered manifest
SHA, world size 2, optimizer and scheduler state, both rank RNG paths, sampler
state, and the full training contract.  This validates recoverability only;
the formal run is still not terminal and no downstream model claim is made.

## 2026-10-21 — historical G2 midterm scope audit

Indexed the Clean V2 TRAIN manifest (583,251 rows; SHA-256 `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`) against all 24 historical step-3000 G2-PL input frames. All 4 MOT17-02-FRCNN frames are TRAIN-scope; the 8-frame and 12-frame Refer-KITTI-0007 RMOT clips have zero exact image-path overlap with the Clean fit index. Every frame path and content SHA-256 is recorded. The short clips remain diagnostics and do not permit an independent generalization claim. Evidence: `docs/GROUNDJEV/G2_MIDTERM_SCOPE_AUDIT_20261021.{json,md}`.

## 2026-10-21 — J1 terminal checkpoint correction

The historical `trackjev_3task_J1_v4_birth` course is terminal at 1,200/1,200 updates, not active: 19,143 rows were consumed, the final adapter and full FP32/Adam/scheduler/RNG state are present, and no J1 optimizer is live. The earlier budget note had only observed step 650 and is corrected without restarting the run. Evidence: `docs/GROUNDJEV/J1_TERMINAL_RECEIPT_20261021.json` and the updated `J1_J2_TRAINING_BUDGET_AUDIT.md`.

## 2026-10-21 — OVMOT evaluator repair reproduced on fresh copies

The official TETA evaluator was rerun on fresh copies of the existing video-63 control and video-4 native diagnostic. The current configuration keeps `OUTPUT_SUMMARY=True`, `OUTPUT_TEM_RAW_DATA=True`, and `OUTPUT_PER_SEQ_RES=True`, which fixes the two previously recorded evaluator failures without changing GT or predictions. Video 63 completed its independent 40-frame TAO validation scope with TETA 18.640, LocA 7.0114, AssocA 28.618, and 35,228 predictions; it is a simple-backend control, not G2/G1/OneJev adaptation evidence. Video 4 remains only a four-frame native diagnostic: all four frames were safely rejected for duplicate/unsorted canonical coordinate tuples and produced zero predictions. The full 40-frame video-4 native evaluation is not claimed until that parser/output blocker is repaired. Receipt: `docs/GROUNDJEV/OVMOT_EVALUATOR_REPAIR_20261021.{json,md}`.

## 2026-10-21 — conservative OVMOT native-output parser repair

The strict native parser rejects video-4 because otherwise complete GroundAnything responses contain duplicate/unsorted output plus malformed fragments (dangling box tokens and nested object references). Added `locatemot/groundjev/native_output_repair.py` and the `official_scoring_repaired` adapter policy. The repair keeps only complete object-ref/box pairs with valid four-coordinate geometry, drops and records malformed fragments, and never fills or reorders coordinates. A targeted self-test passed; offline preflight of the four preserved raw responses recovered 104/129/133/98 complete pairs. A fresh live retry timed out at the GroundAnything service before frame 751, so no new native metric is claimed and video-4 remains pending service recovery. Evidence is in `docs/GROUNDJEV/OVMOT_EVALUATOR_REPAIR_20261021.{json,md}`.

## 2026-10-21 — G2 Clean progress snapshot at step 12873

The formal G2 Clean event stream advanced to 12873/18,227 during the audit; the latest event at 2026-10-21 06:01:24 +0800 has finite loss 0.272433674, gradient norm 11.859652, and peak memory 12.828881 GiB. The last 100/500/1,000-update windows measure 310.88/312.23/311.53 steps/hour, giving a linear ETA near 2026-10-21T23:12:34.824000+08:00. The durable adapter/state pair remains at adapter-step012500.pt plus `latest_training_state.pt`; no terminal receipt exists and all downstream queues remain correctly gated. Evidence: `docs/GROUNDJEV/G2_PROGRESS_SNAPSHOT_20261021_STEP12873.{json,md}`.

## 2026-10-21 — native joint queue config contract repair

The post-G2 native three-task owner was found to inject the audit-only
`native_manifest_rows` field into its generated OneJev config. The pinned
`train.sft` parser rejects unknown keys, so this would have failed before the
first model load when all dependency gates opened. Removed that field from the
generated config; the native manifest row count remains in the owner state and
is checked against the terminal `done.examples` receipt. Validated the
normalized config against `train.sft.build_parser`, confirmed `max_rows=1` is
only a per-microbatch cap with `accum_rows=16`, and ran compile/import smokes.
No G2 state or process was changed. Receipt:
`docs/GROUNDJEV_COUPLED/REAL_NATIVE_CONFIG_AUDIT_20261021.{json,md}`.
