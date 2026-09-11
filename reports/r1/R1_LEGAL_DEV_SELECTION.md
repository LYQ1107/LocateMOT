# R1 legal-dev selection and presence decision

Date: 2026-09-10
Thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R1`

## Frozen choice

The read-only audit at
`outputs/r1/audit/legal_dev_selection_attempt1/contract.json` verified the
L89E epoch-004 Stage-S anchor and Rule-B object.  The frozen legal-dev
TrackEval artifact reports overall HOTA `0.2965123465861875`, DetA
`0.18100053286747086`, AssA `0.4887649285897975`, and distinct-target recall
`0.9795484458843778`; the per-dataset values remain in the machine-readable
audit.  This is internal legal-dev selection evidence, not screening or
official-test evidence.

The existing full-video artifacts do not contain the candidate score arrays
needed to replay an `EXACT_INDEX_DEDUP` candidate-row strategy.  Existing
dedup references are checkpoint-shortlist path deduplication, not candidate
index suppression.  Therefore no exact-dedup advantage is claimed or
fabricated.  The registered conservative fallback freezes the raw Rule-B
candidate policy for R1.

The epoch-004 presence-only audit read exactly 498 legal-dev score records,
with 400 target-present units and 3 misses at the frozen presence threshold
0.5 (`.0075`).  Since `.0075 < .05`, `presence_residual=false` is frozen in
the R1 formal configuration.  The anchor's presence signal is retained; no
new presence head is trained.

## Boundary

No R1 training or checkpoint was created by this audit.  It did not read
screening or official-test labels and did not modify any frozen asset or
production MOT/OVMOT path.  Token/span-to-region and static/motion alignment
remain `UNALIGNED`.

The next action is the bounded R1 wiring/reload smoke with the raw candidate
policy and disabled presence residual.  Formal six-epoch fitting is
conditional on that smoke passing.

## Post-fit continuation (2026-09-12)

The conditional formal fit and legal-development replay subsequently
completed in the isolated R1 worktree. The authoritative post-fit report is
`reports/r1/R1_LEGAL_DEV_REPLAY_REPORT.md`; this appendix supersedes the
historical “conditional on smoke” next-action sentence above without deleting
that historical record.

The six-video prediction-only replay completed before legal GT was opened,
and the preregistered selection tuple selected epoch 4 separately for V1 and
V2. Selected HOTA was 24.2345% for V1 and 23.6542% for V2. Relative to the
L89E anchor values 28.7628% and 21.8385%, this is a V1 regression and only a
1.8157-point V2 increase, so the result is classified
`R1_NO_BREAKTHROUGH` and remains internal legal-development evidence.

The fixed 40-unit diagnostic is documented separately in
`reports/r1/R1_FIXED_SEMANTIC_DIAGNOSTIC.md`; it is not a final semantic gate.
