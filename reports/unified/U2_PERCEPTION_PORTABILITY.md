# LocateMOT-U U2 perception portability

Recovery measurement: 2026-10-11. The original U1R2 foundation remains frozen at
`4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.

The MOT17/MOT20 audit measured every registered train-half validation frame, using
the official MMDetection 800×1333 preprocessing, fixed `person .` vocabulary,
score-ranked candidates, IoU ≥ 0.50, and class-1/mark-1 source GT without a visibility cutoff.
MOT17 uses one FRCNN copy per physical video. The inclusive half ranges follow the
[ByteTrack reference converter](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_mot17_to_coco.py).

| Stream | Validation images | Physical GT units | Top150 @.50 | Top300 @.50 |
|---|---:|---:|---:|---:|
| MOT17 | 2,652 | 53,890 | 0.787233 | 0.806253 |
| MOT20 | 4,463 | 615,137 | 0.731010 | 0.884673 |
| refer_kitti_v1 (existing U1 reference) | — | — | 0.986592 | 0.989960 |
| refer_kitti_v2 (existing U1 reference) | — | — | 0.958739 | 0.964753 |

Both new domains fail the registered Top150 minimum of 0.85 and hard stop of 0.80.
MOT20 improves to 0.884673 at Top300, which remains below the preferred 0.90.
No Top300 runtime override has been accepted and no track training or candidate cache has started.
MOT17-02/04 and the crowded MOT20-03/05 are the main shortfalls; complete per-sequence
numerators and denominators are retained in `outputs/unified/u2/perception/measurements/`.

BDD100K/DanceTrack and TAO base/novel measurements remain unmeasured. Acquisition and
protocol preparation continue. Once all required inputs are ready, the failed gate permits
at most two epochs of mixed perception adaptation with U1 replay, decoder LR 1e-5,
Swin stages 2/3 LR 1e-6 and frozen BERT, followed by full portability and U1 retention checks.

Decision: `LOCATEMOT_U_BLOCKED_U2_PERCEPTION_PORTABILITY`; acquisition/recovery is active.
This is a numerical gate failure plus missing/incomplete inputs, not evidence that U2 is ready.
Partial frame measurements are rejected by the gate reader. The 7,115-frame measurement
passed coverage and numerator/denominator integrity checks. Official test labels remain unopened.

The complete LVIS-v1 vocabulary runtime smoke passed on an extracted TAO validation image: 1,203 names, 38 chunks, one backbone call, 34,200 raw proposals and 300 after class-agnostic IoU-0.8 suppression. No GT was opened in this smoke, and it is not a TAO portability gate result.
