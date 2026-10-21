# Phase B fixed-candidate repair — 2026-10-21

The first 192-step readout trained on a short MOT17 pilot rejected 77.54% of the MOT20-03 candidate cache and produced only two births. The cache itself covers 86.85% of GT boxes at IoU ≥ 0.5, so the immediate failure was the identity-action distribution rather than an empty proposal stream.

The repair uses the immutable MOT20-03 native GroundAnything region cache. The first 64 contiguous frames are the training half; the last 64 are frozen before optimization and used as a temporal development suffix. Identity cross-entropy is weighted by inverse training-half class frequency with a pre-registered 4× cap. The formal G2 process and all independent evaluation owners remain untouched.

After 180 bounded CPU readout updates, temporal-dev identity accuracy was 87.17196% and admit accuracy was 90.38596%. The actual coupled runtime, with predicted memory rather than teacher memory, returned:

| HOTA | DetA | AssA | IDF1 | IDSW | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 0.302815 | 0.235861 | 0.388886 | 0.257729 | 18 | 3 | 5,634 |

It produced 1,891 predictions on 61 of 64 development frames, made 64 cached feature calls, and loaded no second VLM. The result is materially above the previous 0.027901 fixed-candidate replay, but the two numbers use different temporal scopes and the repair is still entirely within MOT20 TRAIN. It is a mechanism repair diagnostic, not an independent benchmark or G2 Clean terminal result; the next valid step is to freeze this procedure and test an unregistered independent development video.

The exact receipt is stored outside GitHub at the run output and summarized in `PHASE_B_FIXED_CANDIDATE_REPAIR_20261021.json`. The repair checkpoint is local only; binaries and caches are not committed.
