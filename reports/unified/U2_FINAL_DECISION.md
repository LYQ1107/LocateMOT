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

MOT17/MOT20 data are complete. TAO four-volume download/extraction and
DanceTrack/BDD image acquisition remain active. C-TAO base-only annotations
are downloaded and audited, with their complete frame coverage still pending.
BDD tracking annotations still need a valid acquisition source. The full LVIS-v1
chunked vocabulary runtime passed one-image backbone-reuse and merge checks;
TAO base/novel recall has not been measured.

Scientific stage status:

```text
LOCATEMOT_U_BLOCKED_U2_PERCEPTION_PORTABILITY
```

The overall research goal remains active. Acquisition and bounded perception
repair preparation continue. No candidate cache, simple tracking baseline,
6000-step MOT-only run, joint run or OVMOT benchmark has started; no combined
trained checkpoint exists. U1 retention, learned tracking, negative-transfer and
TAO TETA/AssocA gates remain unproven. U2 is not ready for U3.

Track training is held until complete registered data and passing portability
measurements. The next allowed repair is at most two epochs of mixed perception
adaptation with U1 replay, followed by all original portability/retention gates.
No gates, frame scope, seeds, visibility filtering or source GT geometry were weakened.

`RMOT tracking launched=false`; `prompt tracking launched=false`.
`official_test_labels_read=false`; `Refer-KITTI official eval labels read=false`.
