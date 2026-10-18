# G0 MOT20-05 official evaluation

The frozen G0 long export completed all 1,658 legal TRAIN frames of
`MOT20-05`. The TrackEval-backed official scope evaluator completed on
2026-10-19 and wrote:

`/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/candidate_export_v2/mot20_MOT20-05_resumed_v2/official_metrics.json`

| Metric | Value |
| --- | ---: |
| HOTA | 0.433441 |
| DetA | 0.638348 |
| AssA | 0.294655 |
| IDF1 | 0.441085 |
| IDSW | 6,239 |
| FP | 6,325 |
| FN | 98,471 |

The result covers one sequence and 1,658 frames. There are 183 rejected
perception frames (13.51%); the official evaluator keeps them as empty frames
in the denominator and records the list in the receipt. This is a G0 reference
scope, not a clean held-out claim for G2: MOT20-05 was discovered inside the
historical G2 training manifest and is excluded from `G2_CLEAN_FULL_COVERAGE_V2`.
Evaluation-only GT was read from the frozen MOT20 TRAIN scope; official test
labels were not read. The long export remains a rejected-perception-aware
reference because its original run receipt records no native detector score and
no complete end-to-end FPS.
