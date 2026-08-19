# Stage L10 — Ordinary MOT Results

Date: 2026-08-18 (final for L10 checkpoints)

Official TrackEval, one shared checkpoint, DLA/LocateAnything candidates:

Official TrackEval, one shared checkpoint per row (DLA/LocateAnything
candidates):

| checkpoint | Dance AssA | BDD AssA | MOT17 AssA | MOT20 AssA | Macro AssA |
| --- | ---: | ---: | ---: | ---: | ---: |
| L9-v5 (best ordinary) | 0.3278 | 0.5159 | 0.7037 | 0.4751 | 0.5090 |
| L9-ovmot (final shared) | 0.3278 | 0.5159 | 0.7037 | 0.4751 | 0.5056 |
| L10 v1 (expanded stream) | 0.338 | 0.5193 | 0.693 | 0.4662 | 0.5041 |
| L10 v2 (target fix) | 0.324 | 0.5102 | 0.6823 | 0.4764 | 0.4982 |

HOTA/IDF1/IDSW for the L10 rows: Dance HOTA 0.5654/0.5538, BDD
0.4872/0.4829, MOT17 0.7049/0.6992, MOT20 0.6295/0.6367 (v1/v2).
Ordinary MOT is robust across the L10 variants; the OVMOT collapse is
stream-specific (see `reports/l10_ovmot_results.md`).
