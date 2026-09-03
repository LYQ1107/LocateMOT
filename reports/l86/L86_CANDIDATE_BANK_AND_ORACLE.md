# L86 candidate bank and oracle evidence

## Frozen candidate input

L86 reads the immutable L69 budget-40 query-independent bank through native
`frame_ptr`/`frame_ids` indexing and the compact label-free L85 Z1 cache. The
L85 cache summary hash is
`e1c0d2b688f6b097528850a7f4ef4812965f2b015a435fd2cc722f57090f9f99`.
Current candidate rows are never filtered, replaced, or ranked by an old
score. The full-video inference scored 15,576,721 candidate rows across 1,844
frames and 243,550 query-frame records, then emitted 882,564 rows under the
frozen dev-selected rule. The emission audit reports no candidate deletion or
truncation.

The internal candidate ceiling from the preceding audit was approximately
unit coverage `0.8385301` and target-level micro coverage `0.8792303`. These
are oracle/coverage descriptions, not semantic model performance.

## GT-privileged semantic oracle

The source oracle at
`outputs/l86/trackeval/semantic_oracle_attempt2/` uses the same legal internal
V1/V2 validation scope and candidate rows, but assigns target-consistent
oracle track choices using GT for an upper-bound diagnostic. Boxes were not
modified and no deployable track IDs were created. Its TrackEval wrapper is
`outputs/l86/trackeval/semantic_oracle_eval_attempt2/`.

| scope | sequences | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 oracle | 86 | 66.8724 | 61.7646 | 72.8335 | 63.9167 | 90.2250 | 82.9324 | 0 |
| V2 oracle | 537 | 53.9669 | 47.8141 | 61.2163 | 49.1385 | 90.4047 | 69.0401 | 25 |

These numbers are GT-privileged internal ceilings, not L86 learned results,
not official test HOTA and not evidence that target identity is available at
deployment. They show that the candidate bank and box geometry leave a large
upper bound, while the learned semantic/emission path remains the limiting
factor for the fixed gate.

Flags: `screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`, `token_span_region_alignment=UNALIGNED`,
and `static_motion_alignment=UNALIGNED`.
