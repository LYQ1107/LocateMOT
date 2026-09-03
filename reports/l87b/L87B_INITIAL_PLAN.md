# L87-B initial plan

L87-B is the isolated zero-training corrected-reselection line. It starts
from frozen L86 commit `97bff208929474d4c4b0d659c80e7eba2f3f5d0a` in its own
worktree and never reads L87-A outputs. It reads only the immutable L86
checkpoint directory and L86 cheap-dev score records; no optimizer, backward,
parameter update, new checkpoint, LoRA or fine-tuning is permitted.

The sole change is evaluation attribution: reselect the existing checkpoint
with unique `candidate_gt` target bags and the exact six-field L87 tuple, and
deploy each candidate only when its energy beats NULL by the frozen margin in
addition to presence/candidate thresholds. Background rows remain singleton
negative bags; all candidate rows are scored and retained.

The corrected selection precedes fixed 16-calibration/24-validation scoring.
After the strategy is frozen, the line runs legal internal full-video V1/V2
inference and the unchanged local TrackEval wrapper. It does not read
screening or official-test labels and does not touch ordinary MOT/OVMOT/TAO.

Inputs are read from `LOCATEMOT_ASSET_ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`.
New outputs stay under `outputs/l87b/`; code and reports stay in this
worktree. B uses physical GPU `[3]` (logical `cuda:0`) for fixed/full-video
inference, with one process and `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
