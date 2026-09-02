# L85 TrackEval protocol audit

The local evaluator checkout is `/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master`.
The package and `scripts/run_mot_challenge.py` are present, but the checkout
has no verifiable local Git HEAD. This is recorded as a provenance limitation;
the checkout is not modified. A future run must set `PYTHONPATH` explicitly
and retain the exact command and source path.

The legal L85 evaluation scope is internal validation only: V1 `0004,0018`
and V2 `0016,0017,0020`, using the existing train-pool/validation records and
the L69 query-independent track IDs. Frame indices, sequence IDs, 2D boxes,
and track IDs must be written in one explicit convention and checked against
the TrackEval adapter before running. GT is read only after checkpoint,
strategy, and calibration choices are frozen. The report must call any such
result “full-video validation HOTA”. It must not be called official test,
screening, or standard benchmark HOTA.

TrackEval has not yet been run in this stage at the time this audit is written.
No HOTA, MOTA, IDF1, or AssA number is inferred from a surrogate candidate
metric. If the adapter cannot validate the internal format, the first
traceback is preserved as `INCOMPLETE.md` and no fabricated score is emitted.

Flags: `screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`, `hota_trackeval_run=false`.

## Completed internal evaluation

The frozen full-video inference completed at
`outputs/l85/trackeval/fullvideo_validation_attempt5/`. It contains 623
query-sequences and 243,550 frame/query prediction audit records: V1 has 86
sequences over 28,029 frame/query records, and V2 has 537 sequences over
215,521 records. Every candidate row was scored before the frozen emission
rule wrote prediction rows. The audit has 243,550 unique unit keys, no
candidate deletion or truncation, 623 prediction files, 623 GT files and
matching sequence metadata. GT was materialized only after prediction
strategy, checkpoint and rule were frozen.

The first CLI attempt is retained at
`outputs/l85/trackeval/trackeval_attempt1/INCOMPLETE.md`: the local runner
passed `SEQMAP_FILE` as a list to an implementation expecting a string. The
targeted V1 retry exposed the same local type issue for `OUTPUT_FOLDER`; it is
retained at `outputs/l85/trackeval/trackeval_attempt2_targeted_v1/`. The
direct-API targeted regression passed at
`outputs/l85/trackeval/trackeval_attempt3_targeted_v1/`. The authoritative
serial full evaluation is
`outputs/l85/trackeval/trackeval_attempt4/`, using the same local checkout,
metrics (`HOTA`, `CLEAR`, `Identity`) and threshold `0.5`.

The result is explicitly **full-video validation HOTA**. The TrackEval
dataset adapter is configured as MOT17/pedestrian only because that is the
local adapter contract; the sequence names and GT are internal
Refer-KITTI query sequences, not official MOT17 or hidden KITTI test data.
The local TrackEval checkout still has no verifiable Git HEAD and was not
modified. The authoritative machine result is
`outputs/l85/trackeval/trackeval_attempt4/trackeval_summary.json`.

| domain | sequences | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW | CLR_FP | CLR_FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Refer-KITTI V1 internal validation | 86 | 25.0548 | 14.9853 | 42.4542 | 47.4128 | 17.8400 | 21.3342 | 2,330 | 64,866 | 14,690 |
| Refer-KITTI-V2 internal validation | 537 | 17.2924 | 9.7879 | 30.8389 | 45.9492 | 10.9991 | 12.9430 | 18,374 | 1,106,999 | 152,170 |

These are TrackEval outputs, not the fixed 16/24 candidate semantic gate and
not official-test scores. The inference summary intentionally keeps
`hota_trackeval_run=false`; the separate TrackEval summary sets
`hota_trackeval_run=true`.
