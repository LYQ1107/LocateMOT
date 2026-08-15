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
