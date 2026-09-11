# R1 legal-development full-video replay report

## Status and scope

Status: `R1_NO_BREAKTHROUGH` / `STOPPED_PENDING_SUPERVISOR_REVIEW`.

This is a legal development replay only. It is not screening, official test,
or a production RMOT submission. The replay used six allowed videos:

| Domain | Videos | Groups | Queries |
|---|---|---:|---:|
| Refer-KITTI V1 | 0008, 0010, 0020 | 1,521 | 154 |
| Refer-KITTI V2 | 0000, 0008, 0009 | 864 | 1,821 |
| Total | 6 videos | 2,385 | 1,975 |

All prediction-only replays completed before legal GT was opened. The
aggregate contains six complete source directories and 8 TrackEval results
(four epochs for each domain). The local TrackEval checkout has no verifiable
git HEAD; this limitation is recorded in the aggregate provenance.

Authoritative compact aggregate:
`outputs/r1/legal_dev_aggregate_attempt11/summary.json`.

Large symlink-only prediction and evaluator views are documented under
`/data2/usr_for_deadline/locatemot_r1_legal_dev_aggregate_attempt11` and
`/data2/usr_for_deadline/locatemot_r1_legal_dev_trackeval_aggregate_attempt11`.

## Registered selection

Selection was separate by domain and used legal-development TrackEval only:

`(HOTA, DetA, AssA, distinct_target_recall,
 -inactive_false_acceptance, -epoch)`.

The selected checkpoint for both domains is epoch 4:

- V1: `checkpoint_r1_epoch004_refer_kitti_v1.pt`, SHA256
  `aec170e1c7a792c0b8ae47dab783232b17baf028beb78de3ad7d35333e9f271c`.
- V2: `checkpoint_r1_epoch004_refer_kitti_v2.pt`, SHA256
  `ec234aca4f2fa5821b12d3c5c840d80ea156e87e550aedf19580fef032ae3439`.

The choice was frozen before the fixed semantic diagnostic. No screening or
official-test labels were read.

## TrackEval matrix

Values below are percentages except IDSW. They are internal legal-development
measurements, not official benchmark results.

| Domain | Epoch | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 1 | 19.5575 | 8.9232 | 42.8984 | 11.0358 | 31.5633 | 13.6449 | 517 |
| V1 | 2 | 11.9586 | 6.0547 | 23.8846 | 7.1935 | 27.3458 | 8.5999 | 1,026 |
| V1 | 4 | 24.2345 | 15.0549 | 39.2116 | 27.6578 | 24.6233 | 20.7354 | 2,692 |
| V1 | 6 | 19.9079 | 11.2729 | 35.4382 | 14.4230 | 33.5987 | 16.7517 | 1,672 |
| V2 | 1 | 15.0494 | 5.1963 | 43.6294 | 6.5599 | 19.8694 | 8.9290 | 1,403 |
| V2 | 2 | 10.5916 | 4.1574 | 27.4104 | 4.7252 | 25.3695 | 6.6668 | 2,598 |
| V2 | 4 | 23.6542 | 13.3271 | 42.1056 | 19.3415 | 29.8059 | 20.2341 | 6,186 |
| V2 | 6 | 22.1766 | 11.8495 | 41.6549 | 16.9641 | 27.9528 | 18.3318 | 6,821 |

Compared with the frozen L89E Stage-S anchor (V1 HOTA 28.7628%, V2 HOTA
21.8385%), selected R1 epoch 4 is -4.5283 points in V1 and +1.8157 points in
V2. The V2 gain is below the preregistered three-point partial-improvement
criterion and is not accompanied by a V1 improvement. Therefore the result is
not `R1_PARTIAL`, `R1_BREAKTHROUGH`, or `R1_STRONG`.

## Integrity and boundary flags

- `screening_gt_used=false`.
- `official_test_labels_read=false`.
- `ordinary_mot_ovmot_touched=false`.
- `tracker_source_changed=false`, `uidm_source_changed=false`, and
  `l69_source_changed=false`.
- Predictions retained complete native rows; deletion/truncation were false.
- The aggregate used symlink-only large views and did not copy frozen model or
  bank weights.

The R1 legal-development TrackEval evidence is preserved for later comparison,
but it does not authorize screening, official testing, or a production merge.
