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
