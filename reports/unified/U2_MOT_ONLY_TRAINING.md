# LocateMOT-U U2 MOT-only training

The exact shared-core 6,000-step MOT-only run is implemented and preregistered
in `outputs/unified/u2/protocol/mot_only_training_recipe.json`. Formal optimization
has not started: full perception selection, the complete ordinary candidate
cache and all-five simple-baseline metrics are prerequisites. There is no trained
MOT-only checkpoint or benchmark metric yet.

`tools/unified/train_u2_mot_only.py` consumes frozen real detector caches. It uses
the same 256-dimensional SharedTrackCore, eight GPU ranks, two clips per rank,
AdamW 1e-4 / weight decay 1e-2, gradient clip 1 and BF16 neural autocast. Losses and
parameters stay FP32. Dataset probabilities are BDD .25, Dance .25, MOT17 .20,
MOT20 .10 and original legal KITTI .20. All ranks use MOT and the same dataset
at each step, with independently seeded video/start/stride draws. Clips grow
2/4/6/8 frames at steps 1,000/2,500/4,000, ending at 6,000 without extension.

Causal strides are uniform 1..min(4,ceil(source_fps/7.5)): BDD 5Hz, KITTI 10Hz,
Dance 20Hz and original MOT seqinfo rates. State motion is per source frame;
gap propagation, age, lost age, decay and the absolute 30-frame expiry use
physical frame indices. Expired IDs are retired before matching. No motion-only
box is emitted. Batched GRU/birth updates preserve the architecture while avoiding
per-track CUDA synchronization.

Teacher forcing is 1 through step 500, linear to .4 at 2,500 and .4 thereafter.
Other frames use the core's own assignment/state. Query drop .10 changes actual
active rows. Injection .20 uses past-frame unmatched non-ignored detector
proposals, with at most max(1,round(.10*active_tracks)) injected rows. No GT identity
is stored in runtime state. External supervision preserves each runtime track's
first known identity after a switch and prevents duplicate births from adopting
the same live canonical GT.

Current-frame IoU Hungarian (.5) constructs row/column targets. Loss weights are
row + col + .25 switch + .25 alive + .50 motion + .25 identity contrast. Column
loss includes positive T+NEW CE and NEW BCE for true births versus existing/real
negative proposals; this trains the fixed zero reject boundary even when T=0.
Ignored candidates receive no negative loss. Switch margin is .5; contrast uses
memory/anchor with same-frame negatives and temperature .07, without class text.
Alive labels use eligible current/future observations within 30 registered TRAIN
source frames; that future GT is a loss label only. Original MOT distractors,
BDD raw toolkit ignore rules, and KITTI visibility/distractor/DontCare/min-height
handling remain separate from identity inputs.

At 1,000/2,500/4,000/6,000, all five complete dev sets are tracked causally and
evaluated by unchanged upstream TrackEval. Candidate cache, source GT scope,
HOTA thresholds, metric scale and simple reference match exactly. Resume snapshots
every 500 updates save optimizer, each rank's CPU/CUDA RNG and the next step;
clip sampling is stateless by step/rank. Architecture/source and experiment
provenance must match on resume. Selection requires macro AssA >= simple+.03 or
>=.50 and every HOTA >= simple-.02, then ranks eligible checkpoints by macro
AssA, HOTA, lower IDSW/GT tracks and earlier step.

Validation evidence (contract fixtures, not benchmark results):

- `eval/temporal_supervision_contract.json`: crossing/occlusion, real state
  augmentation, false births, ignored proposals, mixed teacher/model rollout,
  connected recurrent gradients, physical gap four and expiry.
- `eval/original_train_supervision_contract.json`: every positive GT box on one
  complete actual train video per dataset matches upstream TrackEval; the scope
  is explicitly five videos, not the entire training collection.
- `eval/ddp_eight_rank_supervision.json`: actual eight GPUs, global 16 clips,
  BF16, all seven necessary gradient branches nonzero and rank parameters equal.
- `eval/mot_run_contract.json`: train-only deterministic causal sampling, optimizer/
  RNG resume reproduces the next update bit-for-bit, eligibility and tie breakers.

The initial BF16 proof failed because NumPy cannot consume BF16 logits. Casting
only the Hungarian CPU boundary to FP32 fixed the root cause; a single-GPU
forward/backward and the full eight-GPU proof then passed. An initial GT oracle
comparison mixed TrackEval XYWH with core XYXY; correcting the oracle's coordinate
conversion made all five actual-video comparisons pass. The batched update additionally exposed BF16 vector-index writes into FP32
persistent memory; explicit casts to the destination memory dtype fix that
transition without changing parameters or losses. The fresh single-GPU and eight-GPU regressions
passed before publication. No protocol was weakened.

`queue_u2_mot_only.py` is one blocking follow-on owner (PID 870157 / session 38396).
It waits for ordinary cache/simple owner 864319, validates the full prerequisites,
then launches exactly the registered eight-GPU run. It holds if the earlier stage
ends without a selected foundation/cache/simple result. Joint OVMOT is not launched.
