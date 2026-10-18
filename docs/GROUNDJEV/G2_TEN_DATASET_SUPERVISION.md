# G2 multidomain supervision and coverage

The G2 corpus is TRAIN-only. Every emitted row has the official four-key schema `id/messages/images/source`; messages carry canonical GroundAnything prompts and answers, while the collator decodes media online from absolute paths. The row file and source counts are registered in `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_PROJECTOR_QWEN3_LORA_FULLDATA/g2_train_manifest.json`.

| Dataset | Supervision in current corpus | Rows | Coverage decision |
| --- | --- | ---: | --- |
| MOT17 | dense person boxes | 15,948 | TRAIN sequences only |
| MOT20 | dense person boxes | 8,931 | TRAIN sequences only |
| DanceTrack | dense person boxes | 41,796 | TRAIN sequences only, symlink resolved |
| SportsMOT | dense object/person tracking boxes | 28,574 | TRAIN sequences only |
| AnimalTrack | dense species boxes from decoded TRAIN frames | 11,543 | 32 TRAIN videos, test split unopened |
| Refer-KITTI-V2 | referring expressions and target boxes | 4,333 | fit sequences only; official evaluation videos excluded |
| AerialMind | dense object boxes plus referring expressions | 45,625 + 23,733 | image/label intersection audited |
| RMOT26 | referring target boxes from complete annotation rows | 296 | 20 unrelated failed-download entries remain recorded; unusable rows are rejected |
| TAO-Amodal | dense open-vocabulary category boxes | 18,274 | all 18,274 TRAIN images resolved across `/data0`–`/data3` |
| BURST | none yet | 0 | annotations exist, but no legal TRAIN image root is present; excluded to prevent TAO test leakage |

The resulting corpus contains nine datasets and a transparent BURST gap. No test/validation media is substituted for BURST. The excluded rows and causes are written to `g2_train_rejects.jsonl`; source-balanced scheduling can therefore be reproduced without hiding missing data.
