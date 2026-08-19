# Stage L10 — RMOT Results

Date: 2026-08-18 (Refer-Dance final; Refer-KITTI-V2 pending cache)

Refer-Dance (official evaluator, LocateAnything candidates) and
Refer-KITTI-V2 (official TempRMOT seqmap, 4 eval sequences, Detic-SwinB
candidates):

| dataset | checkpoint | detector | HOTA | DetA | AssA | MOTA | IDF1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Refer-Dance | L9-ovmot | LocateAnything | 36.79 | 45.58 | 29.86 | 29.38 | 36.56 |
| Refer-Dance | L10 v1 | LocateAnything | 36.32 | 46.03 | 28.79 | 28.93 | 35.70 |
| Refer-Dance | L10 v2 | LocateAnything | 36.10 | 44.88 | 29.18 | 26.90 | ~35.5 |
| Refer-KITTI-V2 | L9-ovmot | Detic-SwinB | 3.74 | 0.93 | 16.72 | -4153 | 0.97 |

Detector caveat: LocateMOT RMOT candidates come from LocateAnything /
Detic, not TempRMOT's end-to-end detector; DetA is not directly
comparable.  Identity claims focus on AssA.  The very low Refer-KITTI-V2
DetA/HOTA reflects (a) Detic emitting 50 candidates/frame (DetPr ~1%)
and (b) the shared checkpoint trained only on Refer-Dance language
grounding; this is reported as a cross-domain generalization data point,
not a fair comparison with TempRMOT (35.04 HOTA with its own detector and
in-domain training).
