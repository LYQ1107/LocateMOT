# R0A safe frame-target source repair

This continuation keeps the earlier `R0A_BLOCKED_L49_QUERY_SOURCE` attempt
and its `official_test_labels_read=true` history unchanged. The valid retry
uses an R0-only allowlist-first reader. The old monolithic V2 JSON is scanned
lexically, but only explicitly allowed top-level video values are JSON
deserialized; skipped official-evaluation values are not decoded or exposed.

## Audited source contract

- Source text: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/locatemot/rmot/l49_data.py`
- Expected SHA256: `a1aae91cfbd8aadfba2e432302b06ff7b3e950fe486b48e1f5a4e80481819b3b`
- V1 source: `outputs/l13/data/refer_kitti_v1/expression/<video>/*.json`; only files for allowlisted videos are opened.
- V2 old source: `outputs/l11/data/rmot_kitti/expressions.json`.
- V2 newer source: `outputs/l16/data/kitti_missing/records/expressions.json`.
- Precedence copied from the audited source: old then newer, merge key `(video, expression)`, later source overwrites; final ordering `(video, expression, sentence)` and global query IDs.
- Normalized record: stable sentence plus frame-specific `target` mapping; no target union or temporal fallback.

The valid loader is implemented only in
`locatemot/rmot/r0_safe_target_source.py`. `R0FrameTargetSource` receives
already isolated `R0SafeQueryRecord` objects or a safe artifact and never
imports/calls the historical `load_l49_queries` function.

## CPU reader contract

`tools/r0_audit_safe_source.py` runs one temporary synthetic fixture with
allowed `trainA/trainB` and skipped `forbidden` values. It verifies that the
decoded result has only the allowlisted keys. The real annotation files are
not loaded by this audit. A later source-artifact attempt must carry
`forbidden_payload_deserialized=false` and `official_test_labels_read=false`.

## Stop rule

If the selective reader, safe artifact, or 5,314-row frame audit disagrees
with L49 unit metadata, the new retry is invalid and stops before dense
indexing or model execution. The previous invalid source attempt remains
historical evidence and is not overwritten.
