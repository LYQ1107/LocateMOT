# L89 internal V1/V2 TrackEval report

## Scope

This is the required final internal full-video TrackEval after the L89
checkpoint/rule selection and fixed semantic replay. It uses only Refer-KITTI
V1 validation videos `0004,0018` (86 query sequences) and Refer-KITTI-V2
validation videos `0016,0017,0020` (537 query sequences). Calibration videos
were not added to this final internal scope. No screening or official-test
labels were read. The local TrackEval checkout is
`/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master` and has no verifiable
Git HEAD in the recorded environment.

Primary frozen strategy: L89 epoch 4, Rule R, checkpoint SHA256
`5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`.
Rules B and P are retained in the matrix as diagnostics; they did not change
the pre-frozen primary selection.

## Primary TrackEval results

The table reports percentages for HOTA/DetA/AssA/DetRe/DetPr/IDF1, as emitted
by the local TrackEval summary. IDSW is a count.

| dataset | sequences | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Refer-KITTI V1 | 86 | 2.2906 | 1.0015 | 5.3123 | 1.1646 | 6.6477 | 1.9903 | 59 |
| Refer-KITTI V2 | 537 | 0.5894 | 0.2297 | 1.5550 | 0.2385 | 5.8315 | 0.4700 | 41 |
| unweighted mean | — | 1.4400 | 0.6156 | 3.4337 | 0.7016 | 6.2396 | 1.2302 | 50 |

The local raw values for the primary combined result were HOTA `0.0144000`,
DetA `0.0061560`, AssA `0.0343366`, IDF1 `0.0123015`, and IDSW `50` before
percentage formatting. The output is complete (`3` rule results, B/R/P),
all tracker files match the sequence maps, and all full-video candidate rows
were scored before legal internal GT was materialized.

## Comparison to L87-A

| dataset | L87-A HOTA | L89 epoch4/R HOTA | difference |
|---|---:|---:|---:|
| V1 | 28.5752 | 2.2906 | -26.2846 |
| V2 | 22.1300 | 0.5894 | -21.5406 |

L89 is below L87-A on both domains and is far below the registered material
improvement descriptors `31.5752/25.1300`. It therefore does not establish
ordinary-level RMOT performance. The L89 full-video runner selected 17,591
rows from 76,266 scored candidate rows under the frozen Rule R; its post-hoc
target-row recalls were only `0.0131` on V1 and `0.0026` on V2, while the
distinct-target descriptors were `0.9706` and `0.6496`. These descriptors are
diagnostics, not replacements for TrackEval detection/association metrics.

## Boundary flags

`screening_gt_used=false`; `official_test_labels_read=false`;
`ordinary_mot_ovmot_touched=false`; `groundingdino_lora_used=false`;
`groundingdino_trainable=false`; `bert_trainable=false`;
`token_span_region_alignment=UNALIGNED`; `static_motion_alignment=UNALIGNED`.

This report is internal validation TrackEval evidence only. It is not a
screening result, official-test result, or production MOT/OVMOT regression.
