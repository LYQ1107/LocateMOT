# Stage L10 — OVMOT Results

Date: 2026-08-18 (final)

Official TAO val TETA with public Detic detections and full-PBD cache:

Official TAO val TETA with public Detic detections and full-PBD cache:

| model | TETA | LocA | AssocA | ClsA |
| --- | ---: | ---: | ---: | ---: |
| L8-B2 PBD-zero | 34.33 | 65.05 | 30.44 | 7.51 |
| L8-B2 naive full-PBD | 32.22 | 64.19 | 24.95 | 7.53 |
| L9-ovmot adapted full-PBD | 33.79 | 64.47 | 29.34 | 7.54 |
| L10 v1 expanded (15k) | 26.39 | 64.48 | 7.26 | 7.44 |
| L10 v2 expanded + target fix (15k) | 26.24 | 63.42 | 7.86 | 7.45 |

Reading: expanding the full-PBD OVMOT training stream to all 500 TAO
train videos with DLA detections and C-TAO base GT **collapses
association** (AssocA ~7-8).  The root cause is a training-target /
supervision-coverage problem: only ~3.5% of DLA detections match C-TAO
base GT, so the model learns to birth a new identity for nearly every
detection and fails to associate.  The L9-adapted stream (86% matched)
remains the best full-PBD configuration.  Details:
`reports/l10_supervision_scaling_ablation.md`,
`reports/l10_failure_analysis.md`.
