# Stage L9 — TAO PBD Identity-Feature Cache (full observation)

Date: 2026-08-15 (status: in progress, auto-scaled)

## Purpose

Stage L7/L8 OVMOT was evaluated in the PBD-zero regime (CLIP + spec only)
because no PBD identity tokens existed for TAO val candidates.  Stage L9
caches LocateAnything-3B PBD box-end tokens (`pbd_box_end_last`, 2048-d)
for every public Detic candidate of every TAO val frame, enabling
full-observation OVMOT: PBD identity + CLIP semantics + specification
through the same UIDM core.

## Why crop-based (evidence)

The L1B full-image generation cache for TAO train (`cache_dla/tao_amodal/
train`, 4200 frames) returns zero candidates for 46% of frames and only
~3.3 candidates/frame when non-empty, while TAO val has ~44 Detic
candidates/frame.  Full-image generation cannot cover the candidate set.
Crop-based extraction (one crop per candidate) succeeds reliably:
smoke tests produced exactly one accepted PBD box token per crop
(50/50, 48/50, 150/150 with 2-3 tiny-box failures per frame).

## Implementation

- `tools/cache_l9_tao_pbd.py`: per-frame, per-candidate crop inference;
  BF16, no-grad, `generation_mode="hybrid"`, `max_new_tokens=64`,
  `in_token_limit=2048`, query "Locate the main object in the image.";
  write-through via `locatemot.data.token_cache` (safetensors + meta +
  `.complete`); resume-safe; sharded by video (md5).
- Stored per frame: `pbd_box_end_last/penultimate`,
  `pbd_coord_mean_last`, `pbd_full_block_mean_last` (all [N,2048]),
  `boxes` [N,4], `clip` [N,512], `gen_score` [N]; meta carries
  `candidate_count`, `failed_candidates`, query, model commit/hash.
- Failed/degenerate candidates get zero vectors and are recorded in meta
  (observed 0-3 per 50-candidate frame).
- `tools/check_l9_pbd_cache.py`: one-time regression (key/dim/finite/
  candidate alignment; box precision within float16 tolerance ≤1 px).
- `tools/monitor_l9.py`: detached watcher that restarts crashed workers,
  pauses them under host-RAM pressure (<6 GB, hysteresis) and scales up
  to 8 workers (2 per GPU on free GPUs) as RAM frees.

## Protocol / speed

- TAO val: 988 videos, 36,375 frames, 1,614,849 candidates (~44/frame).
- Measured: ~0.23 s/crop, ~11-13 s/frame (50 crops) per worker on 40 GB
  A100-class GPU; 2 workers ~50-60 h for the full val set; 8 workers
  ~14-16 h.  Cache is write-through, so any interruption only redoes the
  current frame.

## PBD distribution consistency check

Crop-based PBD vs L1B full-frame PBD for the same MOT17 objects:
mean cosine 0.69 (range 0.58-0.78).  The features are correlated but not
identical — expected because context and prompt differ.  This motivates
training the unified core on crop-based PBD for OVMOT (Stage L9 joint
training) rather than assuming zero-shot transfer from full-frame PBD.

## Current status

See `outputs/l9/cache/cache_status.json` (monitor writes every 5 min).

