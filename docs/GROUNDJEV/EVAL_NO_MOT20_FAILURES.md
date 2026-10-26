# Validated failures and remaining work

The full registered evaluation is incomplete. All final-G2, J1-step1200 and
Coupled-step180 weights remain unchanged; no new training was started.

- The first worker-consistency protocol failed because the native random stream
  depended on a service-local request ordinal. Its artifacts are preserved under
  the original root. Formal evaluation uses the separately registered
  `request_local_seed_v2` root and passed the real four-GPU consistency gate.
- AnimalTrack species IDs were initially filtered as non-pedestrian classes by
  the MOT transport reader. The evaluation-only class transport was corrected;
  original invalid metrics were archived and unchanged predictions rescored.
- Concurrent first initialization of a per-video SQLite database caused a WAL
  lock failure. Initialization is now serialized. Four concurrent writers saved
  and recovered all 48 regression records. No completed inference was deleted.
- The original live dispatcher lacked the already-written SO_REUSEADDR port
  probe fix. Workers A and C failed on short replay-to-service rotations. Each
  was recovered independently and reproduced the original native gate output.
  A blocking supervisor recovers terminated workers on the same registered GPU;
  active services and foreign jobs are left running.
- The first complete TAO video has 15 annotated frames and 195 actual native
  requests, covering all 1,203 labels. Every frame is rejected under the frozen
  whole-frame policy; the official Simple and J1 TETA scores are zero. The raw
  chunks are preserved, including successful chunks. Novel metrics are N/A for
  this video's Base-only GT scope. This does not complete TAO validation.
- TAO outputs include malformed coordinate payloads, missing/separated wrappers,
  invalid labels and degenerate boxes. The diagnosis runs against the unchanged
  official parser without GT. It neither invents missing coordinates nor
  salvages predictions into the formal evaluation. Retokenized-output throughput
  includes full HTTP request time and is not a server decode-only TPS result.
- J1 loses most detections on the first complete MOT/RMOT scopes. Its low IDSW
  must be read with the large increase in FN and the paired HOTA/IDF1 decreases.
  All negative results are retained. Coupled is also worse than Simple on the
  first complete MOT videos and does not support RMOT/OVMOT with this checkpoint.
- A subsequent TRAIN-contract check found the J1 competition-coordinate bug:
  runtime xyxy was interpreted as xywh. A real 44-candidate TRAIN matrix has 520
  mismatches before the transport fix and none after it. The original `j1`
  outputs remain legacy diagnostics; `j1_competition_xyxy_v2` is the isolated
  corrected arm. Its gains must be measured; the earlier negative numbers cannot
  be attributed entirely to checkpoint quality.

Remaining work: finish the full immutable registered video/query scope, diagnose
native TAO generation and its execution cost, complete joint official TETA over
all available finished videos, and maintain paired Simple/J1/Coupled comparisons.
An execution optimization is adoptable only after native outputs remain equal;
changing vocabulary, decoder, precision or parser is a separate experiment.

Follow-up input audit and controls (2026-10-26):

- All43,533 frozen J1 TRAIN rows declare a second current-crop image; the
  legacy adapter omitted it. A44-candidate real TRAIN PNG is reproduced
  exactly. A separate226-frame crop-only arm has HOTA13.6705%, IDF17.8107%,
  FN2235 and one output identity. It does not resolve the failure.
- Compact memory requests used original assignment keys against locally
  renumbered candidates and xyxy boxes against xywh competition. The repair
  is a separately named control; it does not rewrite v2 or legacy results.
- All3,471 birth supervision questions lack a candidate index. Two identical
  upstream-rendered input groups conflict across39 TRAIN rows. This is a
  verified data defect; its quantitative causal contribution remains unknown,
  and no original row or checkpoint is modified in this evaluation-only goal.
- Historical per-video SportsMOT receipts inherited an incorrect full-VAL
  scope label. Their actual denominators and existing aggregate scopes were
  correct. An explicit erratum preserves those immutable receipts and fixes
  future reporting to REGISTERED_COMPLETE_VIDEO.
