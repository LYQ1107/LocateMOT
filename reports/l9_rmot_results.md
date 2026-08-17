# Stage L9 — RMOT results (Refer-Dance, official RMOT TrackEval)

Status: L8 baselines below; L9 rows pending the final L9 checkpoint.

## Protocol

Refer-Dance val, 40 GT queries with non-empty GT, official RMOT
TrackEval (HOTA threshold 0.5), relevance threshold from L8 calibration
(-0.1, train F1 0.9175) unless re-calibrated for L9.

## Rows

| Method | Detector | HOTA | DetA | AssA | MOTA | IDF1 |
|---|---|---|---|---|---|---|
| TransRMOT (paper) | DETR-based | 9.58 | 4.37 | 20.99 | | |
| iKUN (paper) | ByteTrack/DLA | 29.06 | 25.33 | 33.35 | | |
| L8-B2 | LocateAnything-3B | 35.20 | 43.42 | 28.63 | | |
| L8-B1 | LocateAnything-3B | 37.88 | 46.51 | 31.02 | | |
| L9 main (10k, MOT+RMOT) | LocateAnything-3B | TBD | TBD | TBD | | |
| L9 main (+OVMOT, planned) | LocateAnything-3B | TBD | TBD | TBD | | |

Caveat: different person detectors (LocateAnything vs ByteTrack/DLA);
DetA is not directly comparable.  The 40-query evaluation set has wide
confidence intervals; numbers are indicative.

## L9 main v5 (cond-gated, official RMOT TrackEval)

| Metric | L8-B2 | L8-B1 | L9 v5 |
|---|---|---|---|
| HOTA | 35.20 | 37.88 | **37.07** |
| DetA | 43.42 | 46.51 | 45.58 |
| AssA | 28.63 | 31.02 | 30.30 |
| MOTA | | | 29.64 |
| IDF1 | | | 36.41 |

Threshold: L9-calibrated 0.45 (train F1 0.8905).  v5 is between B2 and
B1 on RMOT, and above both on ordinary Macro AssA (0.5090).

## Bootstrap 95% CI (per-query, 2000 resamples, seed 20260806)

Rows with non-empty HOTA in the official log: v5 37 queries, B1 38
queries.

| Method | HOTA mean [CI] | DetA mean [CI] | AssA mean [CI] |
|---|---|---|---|
| L8-B1 | 33.25 [26.28, 40.31] | 44.87 [35.44, 54.83] | 28.01 [21.46, 34.75] |
| L9 v5 | 32.42 [25.24, 39.66] | 43.05 [33.41, 53.20] | 28.10 [21.38, 35.17] |

The v5 vs B1 differences are within the CI overlap; with 40 queries the
RMOT comparison is indicative, not a significant ranking.

## L9 final (L9-ovmot)

| Metric | L8-B2 | L8-B1 | L9 v5 | L9-ovmot (final) |
|---|---|---|---|---|
| HOTA | 35.20 | 37.88 | 37.07 | 36.79 |
| DetA | 43.42 | 46.51 | 45.58 | 45.58 |
| AssA | 28.63 | 31.02 | 30.30 | 29.86 |
| MOTA | | | 29.64 | 29.38 |
| IDF1 | | | 36.41 | 36.56 |

Bootstrap CI (final): HOTA mean 34.32 [27.90, 40.64], AssA 29.61
[22.72, 37.01] (36 queries) — overlaps v5 and B1.
