# Stage L6 Training Design

Date: 2026-08-11

## 1. Data

Per-video frame-sequence files built by `tools/build_l6_sequences.py`:

| Split | Videos | Frames/video | Notes |
|---|---:|---:|---|
| BDD100K train | 200 | ~40 | multi-class, dense |
| DanceTrack calibration | 8 | ~1000 | dense same-class |
| DanceTrack train | 32 | ~1000 | dense same-class |
| MOT17 train | 3 | ~80 | sparse pedestrian |
| MOT20 train | 2 | ~80 | crowded pedestrian |
| TAO amodal train | 105 | ~40 | open-vocabulary, amodal |

Manifests: `outputs/l1_c/fixed_candidate_manifest/*.jsonl` and
`outputs/l4/manifests/tao_amodal_train_l4.jsonl`.

No LocateAnything re-run; PBD features are read from the existing token
cache (only `pbd_box_end_last` + `gen_score` are used as object evidence).

## 2. Sampler (domain-balanced, no dataset ID)

1. Pick a domain uniformly from the 5 training pools (BDD, Dance calib,
   Dance train, MOT17, MOT20, TAO).
2. Pick a video uniformly inside that domain.
3. Pick a random contiguous H=16-frame window (frames are consecutive
   manifest frames; gaps preserved via frame ids).
4. Train on the ALL-candidate view.  Category/instance views are reserved
   for the cross-spec diagnostic, not for training.

This prevents BDD/Dance frame count from swamping MOT17/MOT20/TAO without
any dataset-conditioned parameter.

## 3. Clip structure passed to the model

For each frame in the window:

- candidate boxes (pixel) + image size,
- candidate PBD (fp16, loaded as fp32) + gen scores,
- candidate GT id (from manifest `matched`; `None` = unlabeled/FP),
- GT boxes per id (for NO_MATCH/alive supervision),
- frame id (for gaps).

The rollout engine keeps a padded slot table `[B, T_max, d]`; slots are
born on demand (first free slot) and die by learned alive threshold or
MAX_AGE=30.  Every slot knows its current GT id (the id of its last
matched/created observation) so labels can be computed even for student
rollouts.

## 4. Optimisation protocol

- Seed 20260806; AdamW lr 3e-4 (OneCycle, 5% warmup, cosine);
  grad clip 5.0; weight decay 1e-4.
- Batch: 8 clips/GPU × 3 GPUs (DDP) = 24 clips/step; H=16.
- Teacher forcing → scheduled sampling: teacher prob 1.0 for the first
  ~1k steps, then cosine decay to 0.4 over the run.
- Mixed precision (bf16) if the GPUs support it; checkpoints every epoch
  (small checkpoints at fixed steps for the pilot).
- Pilot: ~30–40 epochs on the small subset to verify learning trends,
  then full multi-domain training until convergence (judged by loss/AssA
  curves, not fixed epoch count).

## 5. Validation / evaluation protocol (fresh-tracker rule)

- Every key evaluation uses a **fresh tracker tag** and a **fresh
  TrackEval directory**; checkpoint path is verified before running.
- Domains: DanceTrack val (25 videos), BDD train subset (200 videos),
  MOT17 train (3), MOT20 train (2), TAO train (105; association metrics
  where GT protocol permits).
- Metrics: HOTA, DetA, AssA, IDF1, IDSW, Macro (domain-equal average).
- Cross-spec drift: ALL vs restricted-view rollouts (same metric as L5).

## 6. Ablations (≤5, after the main model)

1. no persistent memory (per-frame re-encode, L5-style) — tests memory
2. no inter-track interaction — tests set competition
3. fixed L1DK decision vs learned transition — tests learned transition
4. no tracking-level loss (row-CE only) — tests UniTrack-style objective
5. no learned lifecycle (fixed NO_MATCH/termination) — tests lifecycle

## 7. Efficiency

Report params, trainable params, peak VRAM, GPU count, training wall time,
FPS at inference, and a FLOPs proxy (forward MACs on a canonical batch).
