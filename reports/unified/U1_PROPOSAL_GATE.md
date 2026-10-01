# U1 proposal gate — 2026-10-01

The legal development slice was run through the MMDetection Swin-B
GroundingDINO candidate path with no training. The probe uses the repository's
OpenCV/BGR input contract (PIL frames are converted to BGR before the configured
BGR-to-RGB preprocessor step), then selected 16
expression/frame rows from V1/V2 calibration and validation videos, skipped no
source target-ID anomalies, and opened each corresponding label only after the
model prediction for that frame completed.

| Metric | Result |
|---|---:|
| Rows / target boxes | 16 / 50 |
| Candidate boxes per row | 300 |
| Best-IoU >= 0.25 | 0.5400 |
| Best-IoU >= 0.50 | 0.0400 |
| Mean best IoU | 0.2756 |
| Mean score at best IoU | 0.01312 |
| Finite boxes/scores | yes |

The 0.50 coverage is still too low to use this proposal stream as a semantic
training target. This is a diagnostic gate result, not a final benchmark. The
next repair is to inspect prompt tokenization/category assignment, box
decoding and proposal selection on the same legal rows, then rerun the fixed
slice without changing the split or labels. Until that repair passes, no
semantic head or tracking core is trained from this stream.

As a selection-only diagnostic, the same rows were re-run with all 900 decoder
queries retained instead of the config's 300 candidates. Coverage moved to
`IoU@0.50=0.1000` and mean best IoU `0.3341`, which confirms that top-k ranking
contributes but does not repair the underlying localization gate. The 900-row
output is retained at `outputs/unified/u1_proposal_gate_top900.json`; it is not
used as a training bank.

A second fixed-prompt diagnostic (`car . pedestrian . cyclist .`) with all 900
queries reached `IoU@0.50=0.1800`, `IoU@0.25=0.9000`, and mean best IoU
`0.3635`. The improvement supports using a class-agnostic proposal pass for a
future repair, but it still does not meet the stage proposal gate and is also
not a training bank. Its output is
`outputs/unified/u1_proposal_gate_generic_classes_top900.json`.

The authoritative machine-readable output is
`outputs/unified/u1_proposal_gate.json`. `official_test_labels_read=false`,
`screening_gt_used=false`, and `training_launched=false` remain explicit.
