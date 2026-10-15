# GroundJEV-MOT

User-directed research pivot, 2026-10-13. Goal:
`GOAL_GROUNDJEV_UNIFIED_ONLINE_TRACKING`; first milestone:
`GOAL_4GPU_REAL_VIDEO_PILOT`.

Build and train one causal online MOT/RMOT/OVMOT system using real
GroundAnything DLM 4B localization, real pretrained OneJev-0.8B structured
identity/lifecycle/update decisions, and shared persistent identity memory.
Preserve the old LocateMOT results and datasets. Limit simultaneous computation
to four physical GPUs and temporary features to 25 GiB. Do not restart U2's
whole-dataset cache route.

The Pilot requires a complete legal MOT video, legal RMOT and OVMOT loops,
tracking-specific pretrained OneJev fine-tuning with before/after measurements,
a fixed-candidate association baseline, official tracking metrics and actual
end-to-end speed/memory records. Publish code/configuration/results and retain
real checkpoint paths/sizes/digests locally. Negative results guide controlled
follow-up experiments; they do not terminate the entire research project.

After the Pilot, continue Full/Compact distillation, controlled GroundAnything
tracking adaptation, memory and multitask studies. Interactive prompting is
secondary to the first three-task Pilot. Never claim synthetic tensors or
perception recall as completed video tracking results.

User clarification,2026-10-15: the final purpose is ordinary-model-level
performance on MOT,RMOT andOVMOT. Completing the Pilot, measurements or negative
diagnoses is insufficient. The goal stays active until a shared candidate
system/checkpoint reaches the registered conventional references under matched
evaluation protocols on all three tasks. Report identity metrics, Base/Novel
recognition and actual end-to-end cost alongside the main accuracy metrics.
Short clips and one-video scores cannot establish dataset-level parity.

Initial references are ByteTrack forMOT,TempRMOT forRefer-KITTI-V2 andOVTrack
forTAO open-vocabulary tracking, registered in
`configs/groundjev/ordinary_model_targets_v1.json`. Published figures are
directional targets until split,scope,annotations and evaluation settings
match. Released RMOT training includes our local held-out videos, so a released
checkpoint on those videos is a training-overlap reference, not an unbiased
held-out baseline. Preserve existing forbidden evaluation-label boundaries;
use compatible reference training/splits for formal comparisons. No current
model has established the final performance target.
