# L89C corrected developer selection

Date: 2026-09-08
Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89C`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`

## Contract and inputs

This is zero-training evidence. The 9,960 immutable raw development score
records were read from the L89 dev file (20 even checkpoints, 498 records per
checkpoint). The corrected implementation uses the single source of truth in
`tools/l88c_eval_metrics.py`:

```text
candidate_energy >= candidate_threshold
candidate_energy - null_logit >= null_margin
presence_logit >= presence_threshold
```

The three registered B/R/P rule fits were recomputed with
`l88c_eval_metrics.fit_rule_set`; each reports rule
`grid_candidate_energy_null`. No scorer, optimizer, bank builder, detector,
CLIP, or checkpoint-writing operation ran.

## Selection protocol

Five checkpoints were shortlisted and evaluated on the complete developer
groups with rules B, R, and P. The developer TrackEval matrix contains 15
results. Selection was frozen before the fixed 16/24 replay by the registered
lexicographic order: TrackEval HOTA, then DetA, AssA, distinct target recall,
lower inactive false acceptance, earlier epoch, and rule tie order. The
selection output is
`outputs/l89c/dev/final_selection_attempt1/checkpoint_selection.json` (SHA256
`c227a604cfc4cbbab18d6087d91d0aab298ebccc47fccab9aa3709dfd3ef2a38`).

| epoch | rule | HOTA % | DetA % | AssA % | distinct target recall |
|---:|:---:|---:|---:|---:|---:|
| 8 | B | 0.872450 | 0.226444 | 3.415761 | 0.352402 |
| 8 | P | 1.008499 | 0.284016 | 3.667592 | 0.409664 |
| 8 | R | 1.185524 | 0.369027 | 3.905860 | 0.454320 |
| 20 | B | 0.796280 | 0.209837 | 3.096565 | 0.343999 |
| 20 | P | 0.551138 | 0.100915 | 3.112634 | 0.240304 |
| 20 | R | 0.921816 | 0.270805 | 3.224042 | 0.397059 |
| 40 | B | 0.801906 | 0.198506 | 3.309522 | 0.301336 |
| 40 | P | 0.665476 | 0.134930 | 3.416564 | 0.258888 |
| 40 | R | 0.888024 | 0.237802 | 3.390879 | 0.356820 |
| 32 | B | 0.868773 | 0.221725 | 3.463829 | 0.322560 |
| 32 | P | 0.731990 | 0.153219 | 3.613622 | 0.284314 |
| 32 | R | 0.914801 | 0.250120 | 3.423334 | 0.365223 |
| 2 | B | 1.030837 | 0.300252 | 3.627040 | 0.401261 |
| 2 | P | 1.016163 | 0.281390 | 3.741766 | 0.394850 |
| **2** | **R** | **1.273246** | **0.406842** | **4.097888** | **0.462724** |

The selected checkpoint/rule is not the historical buggy L89 selection
(epoch 4/R). The corrected result is consequently an end-to-end corrected
selection, not a claim that only one scalar formula changed while every
downstream choice stayed identical.

## Boundary

`zero_training=true`; `screening_gt_used=false`;
`official_test_labels_read=false`; `ordinary_mot_ovmot_touched=false`.
The developer TrackEval result is legal dev evidence only. It is not a fixed
validation gate, screening result, official test, or ordinary RMOT benchmark.
