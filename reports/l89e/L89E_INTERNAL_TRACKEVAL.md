# L89E internal TrackEval

The frozen selection was replayed over the complete native internal timeline:
V1 videos 0004/0018 and V2 videos 0016/0017/0020.  This is internal
validation-scope TrackEval only; screening and official-test labels were not
read.

| dataset | HOTA | DetA | AssA | DetRe | DetPr | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 28.7628 | 18.7680 | 44.3927 | 80.3045 | 19.5312 | 1997.0 |
| V2 | 21.8385 | 13.3358 | 35.9990 | 45.1160 | 15.8179 | 9915.0 |

The phase/history contract passed for the selected checkpoint: `{"epoch": 4, "history_length": 0, "history_mode": "zero_history", "phase": "S", "temporal_enabled": false}`.
