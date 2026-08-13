# Stage L6 TAO Results

Status: COMPLETE (tag `uidm_final`).

TAO amodal train (105 videos) is used both as a training domain and as an
association check via the custom manifest protocol (fps=1).

| Metric | UIDM |
|---|---:|
| HOTA | 0.3446 |
| DetA | 0.2175 |
| AssA | 0.5461 |
| IDF1 | 0.2570 |
| IDSW | 392 |

DetA is low because amodal GT boxes are not fully aligned with the
candidate-box protocol; AssA 0.5461 shows reasonable association quality
once detections are given.  No same-protocol historical baseline (L5 did
not execute TAO); status PARTIAL positive evidence.
