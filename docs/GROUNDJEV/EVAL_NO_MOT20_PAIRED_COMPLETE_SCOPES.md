# Same-candidate complete-video comparisons

Status: REGISTERED_COMPLETE_VIDEO, INCOMPLETE full goal. Snapshot 2026-10-26T14:46:52.154153+00:00; G2 final step18227 7c4a81349e333a304e32f42cf1999db289802d8c31eba415b85827038d14801f.

Each dataset uses exactly the common complete video/query set for all listed methods. HOTA/DetA/AssA/IDF1 are percentages; IDSW is a count. Official TrackEval combine_sequences is used.

| Dataset | Videos / jobs | Query-frames | Method | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| animaltrack | 8 / 8 | 2308 | coupled | 8.6239 | 19.5759 | 3.9082 | 6.6749 | 6610 |
| animaltrack | 8 / 8 | 2308 | j1 | 8.3665 | 3.2092 | 22.3459 | 3.1673 | 172 |
| animaltrack | 8 / 8 | 2308 | simple | 26.8390 | 35.3995 | 21.6631 | 27.5690 | 3021 |
| refer_kitti_v2 | 1 / 8 | 1672 | j1 | 9.6432 | 6.9053 | 13.5341 | 7.4733 | 0 |
| refer_kitti_v2 | 1 / 8 | 1672 | simple | 15.0411 | 15.8631 | 14.4762 | 13.6796 | 385 |
| sportsmot | 8 / 8 | 2050 | coupled | 22.1193 | 30.7100 | 15.9653 | 20.9695 | 511 |
| sportsmot | 8 / 8 | 2050 | j1 | 14.8062 | 6.3992 | 34.2634 | 7.7878 | 0 |
| sportsmot | 8 / 8 | 2050 | simple | 60.9771 | 76.1057 | 49.1228 | 59.1141 | 606 |

TAO: 7 complete videos/124 annotated frames. Official joint TETA is0 for Simple and J1. Novel category metrics areN/A: no Novel GT in this completed scope. Whole-frame chunk rejection explains the empty candidate stream; see EVAL_NO_MOT20_TAO_FAILURE_ANALYSIS.md.

Main J1 uses current-crop/local-memory v4 and the original step1200 checkpoint. Coupled is the separate step180 ordinary-MOT readout trained on G0 feature inputs; applying final-G2 projector features is a recorded distribution shift. RMOT/OVMOT Coupled: NOT_SUPPORTED_BY_CURRENT_CHECKPOINT.

These fixed checkpoints show no overall tracking advantage over Simple in the completed scope. Lower IDSW with severe FN and trajectory collapse does not establish better identity maintenance. This does not establish that a future trained unified architecture cannot work. No new training is started.

Exact job IDs and deltas: EVAL_NO_MOT20_PAIRED_COMPLETE_SCOPES.json. Timings, models and prediction checksums: EVAL_NO_MOT20_RESULTS_SNAPSHOT.json. Attribution: EVAL_NO_MOT20_NET_GAIN.md.
