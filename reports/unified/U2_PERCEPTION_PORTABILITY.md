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
| DanceTrack | 25,508 | 225,148 | 0.996314 | 0.997686 |
| refer_kitti_v1 (existing U1 reference) | — | — | 0.986592 | 0.989960 |
| refer_kitti_v2 (existing U1 reference) | — | — | 0.958739 | 0.964753 |

Both new domains fail the registered Top150 minimum of 0.85 and hard stop of 0.80.
MOT20 improves to 0.884673 at Top300, which remains below the mandatory 0.90
acceptance threshold for the preregistered crowded-scene K=300 override.
No Top300 runtime override has been accepted and no track training or candidate cache has started.
MOT17-02/04 and the crowded MOT20-03/05 are the main shortfalls; complete per-sequence
numerators and denominators are retained in `outputs/unified/u2/perception/measurements/`.

BDD100K, original KITTI tracking and TAO base/novel full measurements remain
unmeasured. The U1 Refer-KITTI references above remain immutable retention gates.
Ordinary KITTI additionally requires every original benchmark GT unit on the
frozen legal V2 development scope: 3,472 images / 15,875 car/pedestrian units,
using unchanged official visibility preprocessing and original xyxy coordinates.
Its real-image smoke passed; the full F0 audit runs on GPU 0. The same Top150
minimum .85 / hard stop .80 applies, without a KITTI Top300 runtime override.
Each U2-P candidate now audits all 76,068 ordinary development images in addition
to every original U1 and TAO/C-TAO scope. This supplies a missing ordinary-GT
measurement; no existing scope, gate, seed or two-epoch budget is weakened.
On 2026-10-12, the complete DanceTrack audit passed: 224,318/225,148 Top150
hits and 224,627/225,148 Top300 hits across all 25,508 validation frames.
Exact registered frame coverage and GT totals match the frozen protocol. The
DanceTrack parent exited successfully and released GPU 7 for the queued BDD audit.
TAO/C-TAO now have complete required image coverage; their registered audit covers
36,375 validation images and 17,131 original training-audit images, respectively,
and runs on GPUs 1–6. Real smoke frames from both domains passed the one-backbone-call
contract with the specified foundation. Smoke frames do not establish aggregate recall.

The TAO localization audit retains all 113,112 source GT units, including 314 unmapped
LVIS-v0.5 synsets: 110,055 base and 3,057 novel GT units. C-TAO's original-frame audit
contains 53,263 base GT units and zero novel GT. Both streams must meet base ≥0.70;
TAO must also meet novel ≥0.60. C-TAO novel recall is explicitly inapplicable for this
base-only release. All 490,210 C-TAO frames remain registered for later identity training.

BDD raw train/val tracking annotations and all validation images have been recovered
from a pinned mirror. All 39,973 validation frames and 442,814 positive GT boxes are
registered; positive/ignore handling and box coordinates match the official toolkit
on every validation object. Validation image sizes/CRCs match the original release.
The full BDD candidate audit is running on GPU 7 after a successful real-image check.
All original training images now match the complete raw annotation scope; all seven
source streams are available. Filtered and converted SOT labels are not used.
Once all required inputs are ready, the failed gate permits
at most two epochs of mixed perception adaptation with U1 replay, decoder LR 1e-5,
Swin stages 2/3 LR 1e-6 and frozen BERT, followed by full portability and U1 retention checks.

The bounded U2-P recipe is now preregistered: seed 20261012, 20,000 replay rows
per epoch, at most two epochs. U1 generic/V1/V2 proportions are 10%/15%/25%;
BDD, Dance, MOT17, MOT20 and TAO-base each contribute 10%. Only registered train
frames and unchanged U1 legal-fit replay are eligible. Mapped TAO f/c labels
supervise detection; rare/unmapped labels supply neither semantic nor identity
supervision. The existing native ODVG grounding path retains multi-box regions
and empty-positive images. BERT stays frozen and in eval mode; Swin stages 0/1
stay frozen. FP32 batch 2 on eight ranks gives exactly 1,250 steps per epoch
and a total cap of 2,500. Aspect-ratio buckets are disabled to avoid singleton
tail batches changing that budget.

CPU checkpoint loading has no missing/unexpected keys and the freeze contract
passes. Native pipeline checks pass on one real training image per stream plus
an empty-positive image; these checks are not portability measurements. The
complete replay preparator finished all source checks and generated its 20,000 rows.
The actual native eight-rank sampler retains every row and all 300,070 positive
training boxes, with no duplicates/omissions and exactly 1,250 steps per epoch.
No optimization has launched. The guarded trainer pauses after epoch 1 for full
portability, unchanged U1 generic/expression audits, and the original 500-pair
language audit. If epoch 1 fails either gate, its checkpoint is rejected for
tracking and the second bounded epoch may continue from those evaluated weights.
Epoch 2 must pass both unchanged gates to be selected. Candidate evaluators require
an explicit checkpoint SHA and separate outputs; F0 references remain intact.

Decision: `LOCATEMOT_U_BLOCKED_U2_PERCEPTION_PORTABILITY`; full audits and bounded
repair preparation continue. Source availability has passed, while numerical gate
failures and incomplete full measurements still prevent U2 readiness.
Partial frame measurements are rejected by the gate reader. The 7,115-frame measurement
passed coverage and numerator/denominator integrity checks. Official test labels remain unopened.

The complete LVIS-v1 vocabulary runtime smoke passed on an extracted TAO validation image: 1,203 names, 38 chunks, one backbone call, 34,200 raw proposals and 300 after class-agnostic IoU-0.8 suppression. No GT was opened in this smoke, and it is not a TAO portability gate result.
