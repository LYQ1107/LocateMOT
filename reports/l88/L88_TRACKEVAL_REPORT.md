# L88 Internal V1/V2 TrackEval Report

## Scope

This is a full-video internal validation TrackEval run after the checkpoint
and Rule B were frozen. It uses only the registered internal validation scope:
Refer-KITTI V1 videos `0004,0018` and Refer-KITTI V2 videos `0016,0017,0020`.
It is not the 96-query screening set, not official test, and not an ordinary
MOT/OVMOT evaluation. The local TrackEval checkout has no verifiable Git HEAD;
the exact root and this limitation are recorded in the machine provenance.

## Selected Rule B results

| dataset | sequences | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW | cleared FP | cleared FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Refer-KITTI V1 | 86 | 26.0914 | 19.2573 | 35.7070 | 56.5138 | 22.4023 | 21.4615 | 1,736 | 57,902 | 11,812 |
| Refer-KITTI V2 | 537 | 20.2386 | 12.0139 | 34.3170 | 33.1874 | 15.7444 | 16.3858 | 9,471 | 527,201 | 194,287 |

The unweighted HOTA macro is 23.1650, DetA macro 15.6356, AssA macro
35.0120, DetRe macro 44.8506, DetPr macro 19.0733, and IDF1 macro 18.9237.

For transparency, the alternative dev rules were also evaluated on the same
internal outputs, but were not selected:

| dataset | rule | HOTA | DetA | AssA | DetRe | DetPr | IDF1 |
|---|---|---:|---:|---:|---:|---:|---:|
| V1 | B | 26.0914 | 19.2573 | 35.7070 | 56.5138 | 22.4023 | 21.4615 |
| V2 | B | 20.2386 | 12.0139 | 34.3170 | 33.1874 | 15.7444 | 16.3858 |
| V1 | R | 23.6895 | 15.3520 | 36.8634 | 65.5952 | 16.5898 | 18.0432 |
| V2 | R | 20.2403 | 10.5462 | 39.0351 | 60.9709 | 11.2606 | 14.1585 |
| V1 | P | 24.5817 | 18.2608 | 33.4585 | 51.9223 | 21.7823 | 20.4606 |
| V2 | P | 20.9141 | 12.1751 | 36.1132 | 39.3494 | 14.9017 | 16.5467 |

## Comparison and interpretation

The registered L86 baselines were HOTA 29.1663 (V1) and 21.6467 (V2); the
L87-A internal values were 28.5752 and 22.1300. L88 Rule B is below both
reference values in V1 and below both V2 values. It also does not meet the
material-improvement descriptor of V1 HOTA at least 31.5752 and V2 HOTA at
least 25.1300.

The very large cleared-FP and IDSW counts, especially in V2, show that the
frame-level volume/correspondence failure is not repaired by the temporal
sidecar. These results are valid internal TrackEval evidence, but they are not
official benchmark scores.

Authoritative machine result:
`outputs/l88/internal/trackeval_matrix_attempt2/trackeval_matrix.json` with
SHA256 `4afd8ef8e251fb4c9998ed5331ecd2b73657ee509ff5d67122ae8ecc4a3e2015`.

