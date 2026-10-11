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

The eight-GPU temporal core check now passes on both tasks with two four-frame
clips per rank, finite recurrent gradients and bit-exact optimizer parameters
across ranks. Actual F0 candidate extraction also preserves native predictions
and final-decoder provenance on MOT and the full 1,203-class OVMOT vocabulary;
fixed-box identity features are invariant to text. These checks support the
runtime contract, not learned tracking performance or full portability.

All seven required streams now have complete source-image coverage. DanceTrack's
full 25,508-frame audit passed: Top150/Top300
0.996313536/0.997685967 over 225,148 source GT units. TAO modal train/val and all
490,210 C-TAO dense training images are available. All 1,400/200 original BDD raw
tracking annotation videos and 39,973 validation images were recovered; every
validation GT object matches the original toolkit and every validation image
matches the author release ZIP. All seven original train-image archives are CRC-extracted
(1,400 videos, 278,079 frames). The full BDD validation audit completed: 39,973 frames / 442,814 GT,
Top150/Top300 .821895875/.852545312. BDD fails the .85 Top150 minimum without
a BDD Top300 exception. All five ordinary F0 audits are complete; full TAO/C-TAO
audits continue on GPUs 1–6 and remain unmeasured until their blocking command ends.

Original KITTI tracking IDs are now frozen separately from the Refer label
namespace. The complete 3,472-frame / 15,875-GT ordinary KITTI portability audit
completed with Top150/Top300 recall .928503937/.940535433, passing the preferred
.90 Top150 threshold. Its original-GT result supplements the unchanged U1 references.
The label-free candidate inventory covers 924,593 unique physical images with
no train/dev overlap. Video-sharded cache construction and the fixed simple
baseline are implemented; a separate blocking queue waits for perception
selection before launching them. All five official file readers, full HOTA
thresholds, ignore preprocessing and ID-switch metrics passed synthetic checks.
No full feature cache or benchmark tracking result has been produced yet.

The actual persistent state now participates in past-negative query augmentation.
Registered temporal losses, source-frame gap handling and partial model rollout
pass synthetic gradient/causal checks. One complete actual train video per ordinary
dataset matches the upstream GT reader on every positive box. The eight-GPU BF16
loss contract passes. A fixed 6,000-step training/evaluation/resume/selection entry
is implemented and preregistered; its own blocking queue waits for the complete
selected cache and simple metrics. These are implementation/contract results, not
trained tracking metrics.

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
