# LocateMOT-U U2 plan

U2 starts from local commit `56507f4aa9a12e76d6f9f6702f0f0b38163666ec` on
`codex/u2-shared-persistent-track-core-20261001`. The U1R2 shared foundation is
immutable at SHA-256
`4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.

The work is ordered as follows:

1. Freeze the machine-readable U1 reference and audit all ordinary-MOT and
   TAO/C-TAO assets without opening Refer-KITTI official evaluation labels.
2. Replace the API-only unified tracking skeleton with a real U2 core: visual
   identity/semantic separation, persistent track state, GRU memory and anchor,
   identity-only set interaction and association, lifecycle/motion, Hungarian
   online update, and a bounded OVMOT semantic residual.
3. Run causal, save/reload, task-isolation, identity-invariance, and parameter
   sharing tests before any long run.
4. Run U1 candidate portability. If any registered stream is unavailable or
   below the preregistered minimum, stop before learned track training rather
   than inventing benchmark metrics.
5. Only after portability passes, freeze candidate caches, train the simple
   baseline, MOT-only core, and then the joint MOT+OVMOT core with the stated
   negative-transfer and TETA/AssocA gates.

No RMOT, point, box, or mask tracking is part of U2. Every major stage writes a
machine-readable artifact and a concise entry in `research_log.md`.

## Recovery plan — 2026-10-11

The U1R2 foundation hash was reverified against the actual 2,780,444,970-byte
checkpoint. The working descendant remains `666ab44`. No training or download
worker was live at recovery. MOT17/MOT20 archives are extracted under their
nested release roots; the four TAO shard directories contain no files.

1. Replace the recursive download launcher with four bounded workers, preserve
   the registered volume assignment, prioritize modal train/validation
   annotations and frames, and record completed file sizes and source hashes.
2. Refresh the data audit using actual image/annotation availability and correct
   nested MOT release roots. A present directory is not a completed dataset;
   available data without measured portability cannot permit track training.
3. Freeze the exact MOT train-half protocols and measure the immutable U1
   foundation on the acquired domains while acquiring BDD/Dance/C-TAO. Retain
   all original gates and the U1 reference; start track training only when the
   required portability measurements pass.
