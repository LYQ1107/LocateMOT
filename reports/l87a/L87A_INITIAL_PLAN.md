# L87-A initial plan

L87-A is the isolated corrected-temporal-negative retraining line. It starts
from frozen L86 commit `97bff208929474d4c4b0d659c80e7eba2f3f5d0a` in its own
worktree and never reads L87-B outputs. The L86 model, Z1 representation,
L69 budget-40 bank, L85 label-free cache, optimizer, seed, S/T/J curriculum
and 40-epoch budget remain unchanged.

The sole science change is temporal-negative construction: for a real shared
query/target pair, negatives are all non-referred `candidate_gt` target bags
available in current or previous frames. No synthetic objectness negatives,
IDs, old scores, top-k, NMS or candidate deletion are used. Candidate rows
remain complete and causal history is bounded by eight observations.

The fresh contract regression precedes the 40-epoch fit. Dev scoring and the
correct unique-target-bag Rule B/R/P selection occur before fixed semantic
labels. The fixed semantic output and internal full-video V1/V2 TrackEval
are run only after selection is frozen. No screening or official-test labels
are read, and ordinary MOT/OVMOT/TAO remain untouched.

Inputs are read from `LOCATEMOT_ASSET_ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`.
New outputs stay under `outputs/l87a/`; code and reports stay in this
worktree. The A mapping is physical GPUs `[0, 2, 8]` with world size 3,
effective clip batch 9, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, BF16 after
the FP32 contract smoke, and `MASTER_PORT=29687`. The final status is based
on complete 40-epoch training, corrected selection/deployment, and legal
internal V1/V2 HOTA, not on a smoke or one metric.
