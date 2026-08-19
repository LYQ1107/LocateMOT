# Stage L10 — Supervision-Scaling Ablation

Date: 2026-08-17/18

The single new ablation of L10 compares the OVMOT training stream size
with the same architecture and init:

| stream | videos | frames | candidates | TAO TETA | AssocA | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| L9 small (7.5k) | 105 | 4,200 | 7,522 | 33.79 | 29.34 | L9-ovmot 6k steps |
| L10 expanded (322.8k) v1 | 500 | 18,274 | 322,843 | 26.39 | 7.26 | negative (over-birth) |
| L10 expanded + target fix v2 | 500 | 18,274 | 322,843 | 26.24 | 7.86 | still negative |
| PBD-zero reference | - | - | - | 34.33 | 30.44 | L8-B2 |

## v1 result (15k steps, 2026-08-18)

The first L10 run **failed**: TAO val full-PBD TETA 26.39 / AssocA 7.26
(vs L9-adapted 33.79/29.34 and PBD-zero 34.33/30.44).  LocA/ClsA were
unchanged; only association collapsed.  Diagnosis: the expanded stream is
hard-negative heavy (only 3.5% of candidates match C-TAO base GT), but
the L9 trainer gave **every unmatched candidate** a positive relevance
target and a NEW birth target, so the model learned "every detection is a
new object".  At eval it assigns a new track id to nearly every candidate
(e.g., 1,612 unique ids over 1,650 rows in one video vs 374 in L9),
collapsing AssocA.

## v2 fix (result: still negative)

Evidence-based training-target correction (one adjustment):

1. OVMOT relevance target = 1 for GT-matched candidates, 0 for unmatched
   (hard negatives), instead of all-ones;
2. NEW births gated by detection score (`--new-score-thr 0.4`): low-score
   unmatched detections are NO_MATCH, high-score unmatched detections
   remain valid NEW candidates (novel objects).

Same architecture/init/data, 15k steps.  Result: TAO val full-PBD
TETA 26.24 / AssocA 7.86 (v1 was 26.39 / 7.26) - the target correction
did **not** rescue association.  A NEW-margin sweep at evaluation
(0.0/0.5/1.0/2.0) on the same shard only lifts subset AssocA from 3.6 to
6.4, still far below the L9 regime.  Conclusion: with ~3.5% candidate-GT
match rate, Detic-detector supervision cannot teach the UIDM to associate
the ~95% unmatched detections; dense continuous GT (covering the
detector's novel detections) or a trajectory-level self-supervision
objective would be required.

The raw DLA candidate stream is 905,400; the training stream caps
unmatched candidates at 16 per frame (positives all kept), documented in
`reports/l10_tao_train_full_pbd.md`.
