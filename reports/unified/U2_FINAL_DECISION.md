# LocateMOT-U U2 decision — recovery in progress

Branch: `codex/u2-shared-persistent-track-core-20261001`. The immutable U1R2
base is local commit `56507f4aa9a12e76d6f9f6702f0f0b38163666ec`, and the
shared foundation SHA remains
`4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.

The implemented U2 core retains the previous causal state, task isolation,
serialization and parameter-sharing evidence (shared non-head fraction 0.937811).
This recovery adds full frozen-foundation MOT-half portability measurements:
MOT17 Top150/Top300 Recall@.50 = 0.787233/0.806253, and MOT20 =
0.731010/0.884673, over all 7,115 registered dev frames. Both registered Top150
gates fail, including the 0.80 hard stop. KITTI retains the U1 reference.

All seven required streams now have complete source-image coverage. DanceTrack's
full 25,508-frame audit passed: Top150/Top300
0.996313536/0.997685967 over 225,148 source GT units. TAO modal train/val and all
490,210 C-TAO dense training images are available. All 1,400/200 original BDD raw
tracking annotation videos and 39,973 validation images were recovered; every
validation GT object matches the original toolkit and every validation image
matches the author release ZIP. All seven original train-image archives are CRC-extracted
(1,400 videos, 278,079 frames). The full BDD validation audit is running on GPU 7;
full TAO/C-TAO audits continue on GPUs 1–6. Full TAO base/novel and BDD recall
remain unmeasured until these original blocking commands complete.

Scientific stage status:

```text
LOCATEMOT_U_BLOCKED_U2_PERCEPTION_PORTABILITY
```

The overall research goal remains active. Acquisition and bounded perception
repair preparation continue. No candidate cache, simple tracking baseline,
6000-step MOT-only run, joint run or OVMOT benchmark has started; no combined
trained checkpoint exists. U1 retention, learned tracking, negative-transfer and
TAO TETA/AssocA gates remain unproven. U2 is not ready for U3.

Track training is held until passing complete portability
measurements. The next allowed repair is at most two epochs of mixed perception
adaptation with U1 replay, followed by all original portability/retention gates.
The fixed 20,000-row, two-epoch recipe and actual frozen-BERT/native-pipeline
checks are implemented. The complete replay passes native eight-rank sampling:
no duplicate/omitted row, no removed training box, exactly 1,250 steps/rank/epoch.
The continuous controller waits for the existing full F0
audits and replay preparation, then uses eight GPUs and stops at each completed
1,250-step epoch for the full original development scopes and 500-pair language
audit. It selects only a checkpoint passing both portability and retention; a
rejected first checkpoint may continue through the second epoch within the same
2,500-step cap. Neither incomplete measurements nor rejected checkpoints can
authorize learned tracking.
No gates, frame scope, seeds, visibility filtering or source GT geometry were weakened.

`RMOT tracking launched=false`; `prompt tracking launched=false`.
`official_test_labels_read=false`; `Refer-KITTI official eval labels read=false`.
