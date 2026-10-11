# LocateMOT-U U2 OVMOT training

Formal joint MOT+OVMOT training has not launched. All TAO/C-TAO training and
validation assets are present. Joint optimization requires the selected eligible
MOT-only checkpoint and complete selected-foundation candidate caches; neither
exists yet. There is no benchmark TETA/AssocA/ClsA/LocA measurement.

The completed base-only supervision compiler covers all 500 TRAIN videos and
491,353 unique dense+modal frames. It preserves 1,489,990 unique base GT units,
including the nine original zero-area C-TAO annotations. Of 53,264 modal base
units, 52,911 also occur in C-TAO. Modal novel/unmapped units (1,156/219) receive
no training identity or class labels. Original annotation UUIDs align source
identities; all overlapping UUIDs actually have equal numeric source IDs.
Validation/test identities are never read by the training compiler.

Physical time uses numbered frame filenames for ordinary camera/LaSOT sources.
For ArgoVerse, original TAO indices anchor integer-rounded timestamp interpolation
between original images, preserving missing-image gaps. Dense frame_id compresses
some gaps and reset interpolation frame_index is unsuitable: 196,400 C-TAO image
indices differ from the physical mapping. Every registered video remains strictly
chronological, with source geometry and all images retained.

Actual frozen F0 BERT plus its native 256-dimensional projection produced all
1,203 class features in 38 native phrase chunks (maximum 104 tokens), with no
image-backbone invocation or second text encoder. This is reference-only evidence;
formal joint use must regenerate/verify features from the selected foundation.
The same existing core/adapter parameters classify persistent semantic memory by
normalized textual similarity and learned temperature. No learned per-class table
or replacement identity core was added.

OVMOT retains the six identity loss weights and adds .50 base classification,
.25 semantic consistency and .10 gate regularization. Association losses train the
final identity-plus-gated-semantic logits. Classification uses only base labels
and base softmax columns; inference evaluates all 1,203 textual classes. Unknown
proposals remain inputs and model births, with no automatic negative/NEW/alive
labels. Certified absent base-category proposals and duplicate base detections
provide hard negatives; original ignored regions retain their exclusion.

Actual eight-GPU BF16 proof passed with real F0 class features and synthetic
tracking candidates: both identity and semantic branches receive finite/nonzero
gradients, all post-update parameters agree, unknown births/rows remain unlabeled,
MOT text isolation and exact zero semantic gate hold. This one scratch update is
not a formal joint run or a benchmark score. The ordinary temporal-loss regression
also passed after the shared optional-branch changes.

Official unmodified TETA is pinned to upstream
`b498aa87b252bfb75d7cc0d8d6435c3e1e74e4de`. Its TAO non-exhaustive local-cluster
evaluation and official base/novel class summaries are used at cluster threshold
50 on the 0..100 scale. The official LVIS-v1 conversion preserves all 36,375
validation images and 988 videos: 112,798 of 113,112 original units and 5,473 of
5,485 original tracks map to v1. The 314 unmapped units/12 tracks are explicitly
recorded; raw v0.5 localization audit scope remains unchanged. No test GT is read.

Official TETA fixture execution passed: perfect predictions score 100 in each
component; wrong classes preserve localization/association and reduce ClsA;
identity switches reduce AssocA; distant unlabeled predictions retain perfect
local-cluster results. Every input frame, including empty emissions, is exported.
Compact video-local runtime IDs avoid large upstream dense ID-map allocations;
official TAO resolves cross-video namespaces. Formal joint requires TETA >=34
and AssocA >=37 plus every ordinary HOTA/AssA within .01 of selected MOT-only.

The exact U2-B execution is now implemented and preregistered in
`outputs/unified/u2/protocol/joint_training_recipe.json`: 4,000 optimizer steps,
eight ranks/two eight-frame clips per rank, global MOT/OVMOT probabilities
.65/.35, unchanged ordinary sub-sampling, teacher probability .4, shared
core/candidate/semantic projection LR5e-5 and adapter LR1e-4. Every selected
MOT-only core tensor is loaded strictly; optimizer moments start fresh for the
joint stage. The shared actual foundation/BERT remain outside the optimizer.

At steps1,000/2,000/3,000/4,000, every original ordinary dev sequence and all
TAO validation images are evaluated through the existing official readers.
Each checkpoint is ineligible if any dataset loses more than .01 HOTA or AssA.
Passing selection prioritizes overall TAO AssocA, then TETA, ordinary macro AssA
and earlier step. A failed main OVMOT gate permits only the separately bounded
refinement condition (ordinary retained, TETA>=31 and AssocA>=34); it does not
authorize a larger joint budget or declare readiness.

The eight-GPU BF16 joint proof initializes every tensor from an explicit
synthetic MOT-shaped checkpoint, alternates MOT/OVMOT through the same core,
and verifies both optimizer groups and identical rank weights. Actual joint
checkpoint/optimizer and per-rank CPU/CUDA RNG restore reproduces the next
stochastic update bit-exact, including scheduled teacher Bernoulli draws.
MOT gives unused semantic-adapter parameters no gradient/weight decay.

The complete eight-rank joint evaluation entry also passes on synthetic shards:
all five official ordinary readers (three frames each), two base/novel TAO
videos (six frames) and official parallel TETA. Resuming after complete video
units invokes no core forward calls and returns identical metrics. These are
integration/contract fixtures, not selected-foundation caches or benchmark
tracking scores. The actual full-source evaluations remain pending.

One blocking joint owner (PID883611/session32672) waits for the original
MOT-only owner870157 to genuinely finish. It recomputes all four MOT-only
eligibility boundaries before full OVMOT caching/class-text preparation and
joint launch. Current state is `WAITING_ELIGIBLE_MOT_ONLY`; no OVMOT cache or
formal joint optimizer has started. Full rank-zero metric collection has a
24-hour collective wait limit so peer ranks can wait through complete evaluation;
no metric threshold, sequence, seed, optimizer step or candidate budget changed.
