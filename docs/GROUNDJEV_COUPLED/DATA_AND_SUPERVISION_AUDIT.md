# Data and supervision contract

The protected G2 Clean run remains the source of the shared backbone:

- manifest: `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/g2_clean_train_manifest.json`
- rows: 583,251 TRAIN rows, 2,249 explicit rejects
- families: AerialMind, AnimalTrack, DanceTrack, MOT17, MOT20 (01/02/03),
  RMOT26, Refer-KITTI-V2, SportsMOT and TAO-Amodal Base
- excluded/held-out: BURST and MOT20-05; TAO Novel is not a training label
- durable adapter used by the Phase-A gate: `adapter-step011000.pt`, SHA256
  `f2b7199f57dae156a05e1efcadfa04b275a7f0399187b264e0b21ddc1b58409b`

`tools/groundjev/yestrack/build_native_yesno_corpus.py` is copied into this
branch as the project-owned data adapter.  Its new v2 records use a stable
SHA-256-derived clip seed; the old v1 corpus is never modified.  The coupled
builder emits an internal schema with explicit validity masks for identity
choice, Yes/No, birth, lifecycle and memory.  It rejects future/GT fields in
causal state and excludes MOT20-05 when reading legacy choice exports.

The Phase-A real image receipt is deliberately a TRAIN engineering gate.  It
uses fixed annotation boxes to exercise one visual forward, typed decisions and
backward; it is not a detector or tracking metric.  Formal held-out videos and
native GroundAnything candidate generation remain separate queues and cannot be
used to tune the Phase-B pilot.

The Phase-B corpus is a separate TRAIN-only development artifact: 1,116
native Yes/No rows (726 TRAIN, 390 video-disjoint DEV rows retained for audit)
were converted to the 1,116-row coupled manifest. The fixed ablation used 24
TRAIN rows: 7 identity-candidate, 9 lifecycle-candidate, and 8 native RMOT
verifier rows. It contains 99 valid identity-choice labels and 119 valid
lifecycle labels overall; memory UPDATE/HOLD labels are absent and therefore
masked. The native and coupled manifests are recorded in
`PHASE_B_NATIVE_YESNO_MANIFEST_20261021.json` and
`PHASE_B_CORPUS_MANIFEST_20261021.json`. No official TEST labels, future
frames, or GT boxes were copied into model state or question text.
