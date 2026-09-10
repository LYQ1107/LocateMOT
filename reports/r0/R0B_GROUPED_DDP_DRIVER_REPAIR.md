# R0B grouped DDP driver repair and formal evidence report

## 0. Status and identity

Status: STOPPED_PENDING_SUPERVISOR_REVIEW

This report records completed R0B grouped-query training, legal development
selection, and internal validation-scope TrackEval evidence. It is not a claim
of final ordinary-RMOT performance. Screening labels, official-test labels, and
production MOT/OVMOT paths were not used.

- Luna thread: 01a02014-fce8-7f51-8414-e7ed6ab44745
- Canonical asset root: /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT
- Isolated worktree: /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R0A
- Branch: codex/r0a-frame-specific-target-contract-repair-20260909
- R0B starting head actually used: 1f6a73efb3320466d1c2442b891cbfbcd983dda7
- R0A implementation base named by the prompt: b7bf1799a700ae7f85c3170c2b57629dc618a3b6
- Head before this report commit: 83c8f13
- Fixed manifest: outputs/l19/protocol/kitti_fast_eval_manifest.json
- Fixed manifest SHA256: 06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa

The canonical manifest was checked at the canonical asset root. The isolated
worktree does not contain a second copy, so a missing worktree copy is not a
manifest change.

## 1. Current Q1 root cause

The old R0 Q1 driver optimized one query at a time. Its process, PID 39571,
was stopped gracefully before it could be treated as a formal result. Its
output remains at outputs/r0/train/v1_formal_retry1/ and contains an old-v1
epoch-02 checkpoint, 8,307 loss records, and a partial epoch-03 trace. It has
no complete final status/provenance and is not selection-eligible.

The first actionable R0B problem was the training contract: Q1 scheduling could
not provide the registered complete same-frame candidate set, multi-positive
bag, or scalable grouped DDP behavior. R0B repairs this with frame-centric
grouped tiles, up to eight queries, and complete candidate sets. It does not
use the Q1 checkpoint as a scientific result.

## 2. Low-memory expectation and resource decision

The repaired driver keeps frozen visual and language inputs in their existing
legal caches, limits train-time groups to Q=8, and bounds each rank with a
deterministic video-local block and local sharding. No candidate row is deleted
to fit memory. The final training used BF16, accumulation 4, and one process
per GPU.

Formal training used the free physical GPUs 4--7:

- V1: CUDA_VISIBLE_DEVICES=4,5,6,7, world size 4.
- V2: CUDA_VISIBLE_DEVICES=4,5,6,7, world size 4.
- Physical GPUs 0--3 were not touched; they had unrelated processes.
- Formal single-GPU inference used physical GPU4 after confirming no compute
  process. TrackEval used CPU.
- No R0 job was run in parallel with another R0 training/evaluation job.

At the final resource check, /data1 had 5.6 GiB available and /data2 had
529 GiB available. Large dev scores and full-video products were written under
/data2/usr_for_deadline/locatemot_r0b_*. No detector weight was copied and no
raw/dense debug cache was created. Frozen assets were not deleted to repair
space.

## 3. Protocol-invalid Q1 preservation

The Q1 output is retained verbatim as historical evidence and is explicitly
disqualified because it is old-v1, partial, and not a grouped complete-set
run. No Q1 checkpoint is loaded by the R0B driver, scorer, shortlist,
inference, or TrackEval.

Earlier reports stating that R0-V1/V2 training was not run refer to the
pre-R0B data-contract stop and remain historical. They were not overwritten;
this report records the later, valid R0B execution.

## 4. Grouped tile implementation

The fixed grouped contract is:

- maximum queries per tile: 8;
- mean queries per tile: 8.0;
- video-local block size: 16;
- requested global frame/query tiles per epoch: 4000;
- deterministic quota when available: 3 multi-positive, 2 positive, 2 inactive,
  1 present-uncovered;
- all current-frame candidate rows retained;
- present-uncovered is not fabricated as a negative;
- inactive units receive explicit no-target supervision;
- multi-positive bags remain multi-positive;
- same-class metadata is unavailable, so the all-negative target-bag fallback
  is used;
- visual cache has no labels;
- token/span-to-region and static/motion alignment remain UNALIGNED.

Padding is at tile level only for divisibility. It is not candidate deletion or
semantic filtering. The loss implementation has a collect_info fast path that
skips expensive diagnostic synchronization while preserving the mathematical
loss; CPU equality and nonzero-gradient checks passed.

## 5. DDP implementation

The driver reads WORLD_SIZE, RANK, and LOCAL_RANK, initializes NCCL, maps one
rank to one visible GPU, and wraps only the trainable R0 sidecar in DDP.
broadcast_buffers=false and find_unused_parameters=false are used after the
grouped forward contract passed. Rank-local tiles are padded deterministically,
and gradient statistics/status are reduced.

The 4-rank wiring smoke passed with finite loss and nonzero gradients. A
harmless NCCL warning reported an unknown device mapping for a barrier. No
implicit extra GPU was used; each rank owned its assigned visible device. The
warning is retained in provenance.

## 6. Gradient accumulation

The fixed accumulation mapping is:

| visible GPU count | accumulation |
|---:|---:|
| 1 | 16 |
| 2 | 8 |
| 3 | 5 |
| 4 | 4 |

The completed formal runs used world size 4 and accumulation 4. Requested
4,000 tiles are padded to nearby multiples of the rank/accumulation contract;
actual per-epoch counts are in each sampling_trace.json.

## 7. Speed and bounded-memory optimizations

The following remained within the R0 contract:

- complete candidate sets are formed once per tile;
- rank-local padding avoids cross-rank shape divergence;
- expensive loss diagnostics are collected only when needed;
- no inner-loop empty_cache or garbage-collector calls;
- visual and language tensors are released at tile boundaries;
- score/inference products go to /data2 because /data1 was nearly full;
- no raw pixels, raw detector maps, or dense debug tensors were persisted.

The first available throughput snapshot was at the end of epoch 1 after at
least 50 optimizer updates, not exact step 50:

| dataset | updates seen | global microtiles | tiles/s | queries/s | mean Q | peak/rank | GPU sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | 763 | 12,208 | 19.9189 | 159.3513 | 8 | 1,036,282,368 B | 16 |
| V2 | 759 | 12,144 | 15.7405 | 125.9242 | 8 | 1,018,539,008 B | 78 |

These are throughput diagnostics, not model-quality results.

## 8. Old Q1 and R0B output preservation

The old Q1 output remains at outputs/r0/train/v1_formal_retry1/ and is marked
PRIMARY_SELECTION_ELIGIBLE=false in R0B_Q1_DRIVER_DIAGNOSTIC.md. All R0B
failed/partial attempts and prior R0A data-contract reports remain in place.

The first V2 full-video preview attempt also remains preserved; it stopped on a
legal GT annotation gap rather than fabricating a box. The minimal R0-only
materialize_gt repair records the missing target-box count and skips that
absent annotation row without changing source GT.

## 9. Q>1 implementation smoke

The single-GPU grouped smoke passed at Q8. It recorded candidate count 57,
membership shape [8,57], finite loss 3.0723, 108 nonzero trainable gradients,
full offsets, and peak memory about 1.433 GiB. This is implementation evidence
only.

## 10. DDP smoke

The 4-rank DDP wiring smoke passed on the free test allocation: world size 4,
two microtiles per rank, Q8, finite loss, and nonzero gradients. It did not
read screening or official-test labels and did not alter ordinary MOT/OVMOT.
The NCCL barrier-device warning is described in Section 5.

## 11. First available >=50 optimizer-step snapshot

The formal runs report their first available snapshot after the first complete
diagnostic interval. It is a >=50 snapshot, not exact step 50. Values are in
Section 7 and show finite loss/gradient norm, mean Q8, and nonzero gradients
for both datasets.

## 12. V1 epoch-2 preview

The legal V1 preview used epoch 2 and at most 32 frames per video and 32
queries per frame. It is PREVIEW_ONLY.

- Output: /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R0A/outputs/r0/preview/v1_epoch02_retry1/
- Groups/records: 96 / 3,072
- Candidate rows: 145,920; selected rows: 23,413
- Candidate precision/recall: 0.1363345 / 0.5283019
- FP/frame: 6.58236; predictions/positive: 3.87504
- top1/top5: 0.51707 / 0.83784
- hard violation: 0.790185; multi-positive recall: 0.516032
- inactive false acceptance: 0.312802; empty rate: 0.581380
- distinct target recall: 0.510353
- positive-above-negative pairwise accuracy: 0.209815

No preview metric was used to claim final R0 quality or read screening labels.

## 13. V2 epoch-2 preview

The legal V2 preview used epoch 2 and the same fixed sampling limits.

- Output: /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R0A/outputs/r0/preview/v2_epoch02_retry1/
- Groups/records: 96 / 3,072
- Candidate rows: 149,824; selected rows: 39,073
- Candidate precision/recall: 0.1529445 / 0.8137255
- FP/frame: 10.77376; predictions/positive: 5.32040
- top1/top5: 0.44560 / 0.81357
- hard violation: 0.856357; multi-positive recall: 0.739343
- inactive false acceptance: 0.576438; empty rate: 0.327474
- distinct target recall: 0.807870
- positive-above-negative pairwise accuracy: 0.143643

The zero-logit preview has no fitted threshold and is not formal selection
evidence.

## 14. Resume integrity and formal training

Both benchmark runs first completed epochs 1--2, then resumed from the
epoch-02 v2 checkpoint and completed epochs 3--12. Epoch-02 reload max
absolute difference was 0.0. Every formal checkpoint at epochs 2, 4, 6, 8,
10, and 12 has strict reload evidence.

### V1 formal run

- Output: outputs/r0/train/v1_grouped_ddp_main/
- Status: complete; epochs 1--12
- Optimizer steps: 3,058; finite: 3,058; nonzero-gradient: 3,058
- Query exposures: 391,424; mean Q: 8.0; world size: 4
- Peak memory: 1,045,657,600 B
- Wall time: 2,233.54 s
- Category exposure: inactive 101,934; multi-positive 156,893; positive
  131,615; present-uncovered 982
- Primary checkpoints: epoch02, epoch04, epoch06, epoch08, epoch10, epoch12
- Epoch mean loss: 2.47982794 at epoch 1 to 1.06025798 at epoch 12

### V2 formal run

- Output: outputs/r0/train/v2_grouped_ddp_main/
- Status: complete; epochs 1--12
- Optimizer steps: 3,046; finite: 3,046; nonzero-gradient: 3,046
- Query exposures: 389,888; mean Q: 8.0; world size: 4
- Peak memory: 1,046,859,776 B
- Wall time: 2,694.07 s
- Category exposure: inactive 97,472; multi-positive 146,372; positive
 144,828; present-uncovered 1,216
- Primary checkpoints: epoch02, epoch04, epoch06, epoch08, epoch10, epoch12
- Epoch mean loss: 2.46455407 at epoch 1 to 1.16951109 at epoch 12

All training status JSONs record candidate_deletion=false,
candidate_truncation=false, screening_gt_used=false,
official_test_labels_read=false, and ordinary_mot_ovmot_touched=false.

## 15. Legal dev scoring and shortlist

Only epochs 2, 4, 6, 8, 10, and 12 are primary-selection eligible. Each
complete V2 dev score contains 528,120 records and 24,420,425 candidate rows,
with all rows retained and no truncation.

- V1 score manifest: /data2/usr_for_deadline/locatemot_r0b_score/v1_grouped_dev_scores/score_manifest.json
- V2 score manifest: /data2/usr_for_deadline/locatemot_r0b_score/v2_grouped_dev_scores/score_manifest.json
- V1 shortlist: /data2/usr_for_deadline/locatemot_r0b_score/v1_grouped_shortlist/shortlist.json
- V2 shortlist: /data2/usr_for_deadline/locatemot_r0b_score/v2_grouped_shortlist/shortlist.json

The fixed zero-logit membership/presence rule was used for dense dev
diagnostics. No calibration, fixed validation, screening, or official-test
labels were used for this selection. Shortlists: V1 epochs 12 and 6; V2
epochs 12 and 10.

### V1 dense dev diagnostics

| epoch | candidate P | candidate R | FP/frame | pred/positive | hard | multi R | distinct R | inactive FA | empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | .151486 | .547928 | 6.91089 | 3.61701 | .833049 | .534155 | .535363 | .304172 | .572808 |
| 4 | .186305 | .541094 | 5.32150 | 2.90434 | .790459 | .528499 | .533573 | .296707 | .584581 |
| 6 | .217964 | .597439 | 4.82682 | 2.74100 | .752475 | .622755 | .590023 | .380248 | .479552 |
| 8 | .297454 | .394818 | 2.09980 | 1.32733 | .751099 | .431084 | .388735 | .185399 | .674879 |
| 10 | .292099 | .430332 | 2.34840 | 1.47324 | .747450 | .466208 | .423622 | .204855 | .645224 |
| 12 | .294441 | .420934 | 2.27130 | 1.42960 | .749544 | .458293 | .414877 | .210387 | .645140 |

### V2 dense dev diagnostics

| epoch | candidate P | candidate R | FP/frame | pred/positive | hard | multi R | distinct R | inactive FA | empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | .132101 | .526808 | 4.56028 | 3.98792 | .828414 | .415119 | .521142 | .199046 | .711848 |
| 4 | .119415 | .772433 | 7.50494 | 6.46846 | .778802 | .727378 | .762817 | .383098 | .473413 |
| 6 | .124461 | .796339 | 7.38101 | 6.39830 | .725173 | .765785 | .787986 | .483843 | .395832 |
| 8 | .143767 | .680053 | 5.33642 | 4.73024 | .720721 | .609692 | .672193 | .293567 | .577545 |
| 10 | .158873 | .687357 | 4.79478 | 4.32645 | .722784 | .628661 | .681823 | .318788 | .549527 |
| 12 | .165238 | .649591 | 4.32382 | 3.93124 | .720256 | .586563 | .644204 | .286983 | .586081 |

These are legal dense-dev diagnostics, not official benchmark scores.

## 16. Full-video dev TrackEval and selection

Full-video inference was run only for the registered shortlist. The preserved
first V2 full-video attempt stopped on a legal missing GT annotation; the
completed retry records the missing target-box rows rather than fabricating
boxes.

- V1 inference: /data2/usr_for_deadline/locatemot_r0b_infer/v1_grouped_dev_selection/summary.json
- V1 TrackEval: /data2/usr_for_deadline/locatemot_r0b_trackeval/v1_grouped_dev_selection/trackeval_matrix.json
- V2 inference: /data2/usr_for_deadline/locatemot_r0b_infer/v2_grouped_dev_selection/summary.json
- V2 TrackEval: /data2/usr_for_deadline/locatemot_r0b_trackeval/v2_grouped_dev_selection/trackeval_matrix.json

Legal dev TrackEval:

| dataset | epoch | HOTA | DetA | DetPr | DetRe | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 12 | 22.8681% | 11.9128% | 14.6912% | 38.0778% | 44.0478% | 18.8004% | 2,749 |
| V1 | 6 | 21.6129% | 9.9108% | 10.7917% | 53.6290% | 47.3033% | 15.6155% | 3,448 |
| V2 | 12 | 19.5801% | 7.6400% | 8.0435% | 59.0149% | 50.3153% | 13.0028% | 16,515 |
| V2 | 10 | 19.3482% | 7.4042% | 7.7273% | 62.3939% | 50.6968% | 12.6046% | 17,525 |

The registered dev selector froze epoch12 for both V1 and V2. Selection JSONs:

- /data2/usr_for_deadline/locatemot_r0b_selection/v1_grouped_dev_selection/selection.json
- /data2/usr_for_deadline/locatemot_r0b_selection/v2_grouped_dev_selection/selection.json

The selector did not read fixed semantic labels or screening labels.

## 17. Fixed internal validation-scope TrackEval

After dev selection, one frozen epoch12 checkpoint per benchmark was run on
registered internal scopes. These are legal internal validation-scope
TrackEval results, not official test or screening results.

### V1 internal

- Inference: /data2/usr_for_deadline/locatemot_r0b_infer/v1_grouped_internal_selection/summary.json
- TrackEval: /data2/usr_for_deadline/locatemot_r0b_trackeval/v1_grouped_internal_selection/trackeval_matrix.json
- Videos: 0004, 0018
- HOTA 26.4582%, DetA 15.0826%, DetPr 16.6992%, DetRe 59.1265%,
  AssA 46.6565%, IDF1 22.2625%, IDSW 1,784

### V2 internal

- Inference: /data2/usr_for_deadline/locatemot_r0b_infer/v2_grouped_internal_selection/summary.json
- TrackEval: /data2/usr_for_deadline/locatemot_r0b_trackeval/v2_grouped_internal_selection/trackeval_matrix.json
- Videos: 0016, 0017, 0020
- HOTA 15.6097%, DetA 6.8524%, DetPr 7.1785%, DetRe 58.4377%,
  AssA 35.7897%, IDF1 9.5610%, IDSW 15,450

No number here is a screening or official-test number.

## 18. Formal eligibility conclusion

R0B passed grouped implementation, resume, complete-set, finite-gradient, and
legal dev/internal TrackEval contracts. It produced real HOTA/TrackEval
evidence on legal development and internal validation scopes, but did not
reach the target ordinary-RMOT level and did not run the later screening gate.
The lower V2 internal HOTA and high ID-switch count show that the grouped
query-training repair alone does not establish persistent expression-grounded
tracking quality.

This branch is eligible for supervisor review, not automatic continuation.
The registered stop label is STOPPED_PENDING_SUPERVISOR_REVIEW; no additional
architecture, long run, or production integration was launched automatically.

## 19. Boundary and evidence flags

~~~text
ordinary_mot_ovmot_touched=false
tracker_source_changed=false
uidm_source_changed=false
l69_source_changed=false
production_entrypoint_changed=false
screening_gt_used=false
official_test_labels_read=false
no_screening=true
no_official_test=true
follow_up_architecture_experiment=false
~~~

R0B has no official-test result, no screening result, and no production MOT,
OVMOT, TAO, PBD, or UIDM regression. The only next action is supervisor review
of the grouped R0B evidence and authorization of any separate follow-up; this
worker does not select or start that follow-up.

