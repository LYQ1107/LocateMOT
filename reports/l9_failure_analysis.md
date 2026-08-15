# Stage L9 — Failure Analysis

Date: 2026-08-15 (interim; OVMOT rows pending full PBD cache)

## 1. Implementation failures (root-caused, fixed)

### 1.1 Randomly initialized UIDM core (v1-v4)

The L9 training script's `--init-ckpt` loader matched only bare state-dict
keys, while L8 checkpoints store `uidm.`/`adapter.`-prefixed keys.  The
core was silently random while the adapter loaded, producing:

- step-10 training loss ~105-110 (vs ~2-8 with correct init);
- ordinary Macro AssA ~0.42 and RMOT AssA ~10-20 (vs 0.5090 / 30.30 after
  the fix);
- a false "self-training drift" hypothesis that was disproved by a
  controlled diagnostic.

Evidence: `outputs/l9/checkpoints/uidm_l9_main_v{1,2,3}_*/`,
`uidm_l9_control_nogate_randomcore/`, research log entries.

### 1.2 Degenerate semantic-transform init

`sem_transform` was initialized as an all-ones matrix instead of the
identity matrix; fixed to `eye_` (commit d3d51bc).

### 1.3 Cache throughput

Per-frame `checkpoint_hash` recomputation read ~6 GB of model files per
frame (28 s overhead); fixed by computing the hash once per worker.

## 2. Task-level failure patterns (ordinary + RMOT, L9 v5)

To be finalised with per-sequence analysis; preliminary observations from
the v5 results:

- DanceTrack IDSW 6282 (B1 6503) and AssA 0.3509 (B1 0.3405) — crowded
  same-appearance dancers remain the hardest identity regime; the gate
  slightly helps.
- RMOT AssA 30.30 (B1 31.02) — language-driven identity in dance crowds
  is still below the RMOT-specialised iKUN (33.35); detector protocol
  differs (LocateAnything vs ByteTrack/DLA).
- MOT17/MOT20 improve with the gate (AssA 0.7017 / 0.4727), suggesting
  the gate can close semantic modulation when identity evidence is
  sufficient.

## 3. Planned OVMOT failure analysis

- TAO Novel association (Base vs Novel) under full PBD observation;
- worst TAO sequences (low AssocA) and their causes (crowd, occlusion,
  detector miss, semantic FP);
- NEW/NO-MATCH and reactivation error counts from the official evaluator.

