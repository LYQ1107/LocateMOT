# Stage L8 — RMOT Results (Refer-Dance, official protocol)

Protocol: Refer-Dance val split (25 videos, 40 queries with non-empty GT),
official RMOT TrackEval runner (`run_mot_challenge.py`, HOTA threshold
0.5, `seqmap.txt` `video+expression` layout). Candidates are the same
LocateAnything-3B person detections as the ordinary-MOT protocol; boxes are
on the 1920×1080 evaluation canvas. Relevance threshold -0.1 calibrated on
the Refer-Dance train split (train F1 0.9175).

## Table C — RMOT (Refer-Dance, 40 GT queries)

| Method | HOTA | DetA | AssA | MOTA | IDF1 | DetRe | DetPr | AssRe | AssPr | LocA |
|---|---|---|---|---|---|---|---|---|---|
| TransRMOT (paper) | 9.58 | 4.37 | 20.99 | — | — | — | — | — | — | — |
| iKUN ByteTrack+NKF (paper) | 29.06 | 25.33 | 33.35 | — | — | — | — | — | — | — |
| **L8 v2 shared UIDM** | **35.20** | **43.42** | **28.63** | 25.47 | 35.44 | 64.82 | 55.26 | 35.82 | 51.54 | 89.26 |

Protocol caveat: published baselines use ByteTrack/DLA person detections;
LocateMOT uses LocateAnything-3B detections, so DetA is not directly
comparable. Association (AssA 28.63 vs iKUN 33.35) remains below the
RMOT-specialized iKUN pipeline, while overall HOTA is above it.

## Ablation (same v2 checkpoint, no retraining)

| Relevance source | HOTA | DetA | AssA | Dets |
|---|---|---|---|---|
| unified (CLIP+spec) | 35.20 | 43.42 | 28.63 | 249332 |
| identity-only (no language) | ~0 | ~0 | ~0 | 0 |
| semantic-only (PBD zeroed) | in progress | | | |

The identity-only row confirms that without the language stream the model
has no target-selection signal (relevance logits all below threshold);
language is necessary for RMOT selection.

Full predictions/eval:
`outputs/l8/trackeval/rmot_v2_fix/`

