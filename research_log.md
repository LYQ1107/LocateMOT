# LocateMOT Research Log

Concise experimental log (latest first).

## 2026-10-18 — J1 step300 and held-out gate reattached

- The live J1 three-task run reached a complete step300 checkpoint with
  adapter weights and full FP32/Adam/RNG state. MOT20-05 remains in its
  protected G0 export without a `result.json`; expression-conditioned RMOT has
  not started inference and remains gated on that release.
- The CPU-only held-out gate owner had exited while waiting, so it was
  reattached to the live J1 PID without restarting training.

## 2026-10-18 — GPU5 G1 functional retention control

- With explicit user authorization, GPU5 was shared after a memory check (about
  68.6 GiB free); the foreign TALON jobs were preserved. The corrected owner
  completed G0/G1 MOT4, RMOT single8, and RMOT multi12 controls in the OneJev
  checkout. HOTA changed 0.7485→0.7333, 0.2718→0.2117, and 0.2179→0.2284;
  this is short functional evidence, not a full-dataset promotion result.
- The G1 service was cleaned up and the evidence/documents were synchronized
  to the GroundJEV GitHub branch. The active J1/MOT20/RMOT pipeline was not
  interrupted.

## 2026-10-18 — DeepSeek Harness handoff re-audit and active continuation

- The requested GroundJEV-MOT redirection was checked against the actual
  DeepSeek Harness SSH-job records and live child processes. Harness launcher
  records with `exit_code=0` are not completion receipts: the detached
  children for the three-task SFT and MOT20 export are still alive. The live
  research checkouts remain `/data2/user/LocateMOT_OneJev_CVPR` and
  `/data2/user/LocateMOT_GroundJEV`; this root checkout is still the required
  project identity `/data2/user/LocateMOT`.
- Active work: `trackjev_3task_J1_v4_birth` (PID 3029659, ranks 3029872 and
  3029873) is a real MOT/RMOT/OVMOT run over 43,533 rows (21,168 / 16,677 /
  5,688), using GPUs 4 and 6. The latest observed log is step 280/1,200 and
  step 250 has a complete FP32-master/Adam/scheduler/RNG restore state; the
  old MOT-only training has no residual process.
- Protected export: MOT20-05 `mot20_MOT20-05_resumed_v2` (PID 3743898) is
  still restoring the exact 1,560-frame prefix and running with the explicit
  4 GiB native-feature budget after the earlier 2 GiB failure. It has no
  `result.json` yet. The expression-conditioned RMOT pilot (PID 3744099)
  remains correctly gated on that receipt and has performed zero model
  inference or optimizer updates.
- GPU5 is not idle: its active compute owner is another user's
  `/data2/OCD/TALON` process. The existing GroundAnything G1 evidence already
  contains a complete 200/200 projector adaptation and G0/G1 comparison; the
  measured result is a retained negative detector adaptation. No foreign job
  was stopped and no duplicate G1 run was started.
- Next recovery point: let the protected MOT20 export and J1 checkpoint reach
  their registered completion boundaries, then run the prepared
  expression-conditioned RMOT pilot and an independent held-out evaluation
  from the latest valid J1 adapter. No benchmark success or unified-goal
  completion is claimed in this audit.

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

## GroundJEV-MOT execution redirection and current handoff — 2026-10-18

The active task was re-scoped to `TRACKJEV_THREE_TASK_DETECTOR_ADAPTATION_AND_TRAINING_V1` under the continuing GroundJEV unified MOT/RMOT/OVMOT goal. The required project root for this checkout is `/data2/user/LocateMOT`; the DeepSeek Harness jobs that are still running use the separate research checkout `/data2/user/LocateMOT_OneJev_CVPR` and the model/runtime checkout `/data2/user/LocateMOT_GroundJEV`. This path split is recorded explicitly; no source from the live old checkout was copied into this U2 checkout.

Resource audit: GPU0–3 and GPU5 are occupied by another user's `/data2/OCD/TALON` jobs and were left untouched. GPU4 and GPU6 run the real three-task OneJev LoRA SFT; GPU7 hosts the protected GroundAnything decoder/feature services and the MOT20-05 export. There is no active pure MOT-only training process. The active SFT reads 43,533 rows (identity 21,168, RMOT 16,677, OVMOT 5,688; `native_G0` 21,168 and `GT_teacher` 22,365) and has a resumable FP32/Adam/RNG checkpoint at step 250 while the log has reached step 260/1,200. The protected MOT20-05 export has reached frame 1,558/1,658 in its resumed output; occasional decoder rejections are recorded as empty frames and do not terminate the run.

GroundAnything G1 was not idle: the existing evidence shows 200/200 projector steps completed and a G0/G1 comparison. The measured G1 recall is within about 1.2 points of G0 and is mostly lower, so it is retained as a negative detector-adaptation result and will not be relaunched on another user's GPU5 allocation. The next detector-related work is expression-conditioned RMOT, because the earlier RMOT native cache used a pedestrian-only request and failed its semantic quality gate.

Prepared without GPU inference: `/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/candidate_export_rmot_expression_v1/plan.json` and 24 per-query manifests cover 12 legal fit/calibration videos and 4,416 frames, exclude all held-out/evaluation videos, preserve the real expression text/files, and use `proposal_source=native_G0_expression_conditioned_pending`. The planner is `tools/groundjev/prepare_expression_rmot_pilot.py`; its manifest validation passed. Inference is intentionally deferred until the protected MOT20 export releases the shared decoder/feature service, so no existing long job is contended or restarted.

The active goal remains in progress. Next recovery points are: let the current MOT20 export and three-task SFT reach their existing checkpoints; run the expression-conditioned RMOT pilot from the prepared manifests; then evaluate the latest valid OneJev checkpoint on the independent held-out scope. No benchmark success or full-goal completion is claimed here.

The first expression-conditioned pilot attempt was stopped after the protected MOT20 owner was found to have exited at frame 1561 because its default 2 GiB native-feature budget was exceeded (the old prefix through frame 1560 remains intact and has no result receipt). The partial first RMOT query is also preserved as `*.failed_attempt`. The smallest recovery is now active: MOT20 resumes from that exact prefix in `mot20_MOT20-05_resumed_v2` with an explicit 4 GiB per-study budget (PID 3743898), while a single-worker RMOT queue (PID 3744099) waits for that new `result.json` before touching the shared decoder/feature endpoints. No result or protected source directory was overwritten.
