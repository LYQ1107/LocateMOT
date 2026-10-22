# G2 Clean dataset coverage and legal held-out scope — 2026-10-22

The terminal G2 Clean training manifest contains 583,251 TRAIN rows from:

- AerialMind dense/referring
- AnimalTrack dense
- DanceTrack dense
- MOT17 dense
- MOT20 dense
- RMOT26 referring
- Refer-KITTI-V2 referring
- SportsMOT dense
- TAO-Amodal Base categories

The manifest explicitly excludes MOT20-05 as the frozen held-out sequence,
uses Refer-KITTI-V2 only for fitting, and keeps TAO Novel annotations out of
training. The current independent held-out queue therefore evaluates the legal
unseen scopes that are registered and overlap-audited: MOT20-05 validation
frames, two Refer-KITTI-V2 RMOT expressions on validation sequence 0007, and
TAO-Amodal validation video 105 with official Base/Novel policy.

Several other G2 source datasets currently have only TRAIN media/labels in
the registered runtime. Reusing those frames would be train-scope leakage, so
they are not silently reported as independent generalization results. New
held-out manifests will be added only when an independent split and evaluation
labels are registered.
