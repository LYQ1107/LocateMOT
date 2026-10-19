# Live experiment metrics and verified boundaries (2026-10-19)

This is an evidence index. A missing receipt means unfinished work; a zero with a receipt is reported only with its exact scope and rejection accounting.

| Work item | State | Evidence / current result |
|---|---|---|
| G2 Clean formal training | RUNNING on GPU3/GPU5 | Event 4889/18227; recent 500-step mean is 14.51 s/step, linear remaining estimate 53.76 h; latest peak memory 17.37 GiB. Training was not interrupted. |
| J1 step-1200 training | COMPLETE | Durable three-task checkpoint and optimizer/RNG state exist. |
| J1 original held-out gate | COMPLETE, immutable | All registered scopes were zero; root cause is strict parser rejection plus all-NO_MATCH association. |
| J1 repaired DT1203 | COMPLETE | HOTA 0.1585619177, DetA 0.1976921614, AssA 0.1272432780, IDF1 0.1282768263, IDSW 0, FP 110, FN 3560. |
| J1 repaired RMOT smoke | COMPLETE | Single8 HOTA 0.3792207792 / IDF1 0.6666666667; multi12 HOTA 0.1598656815 / IDF1 0.1463414634. |
| J1 repaired OVMOT video4 | COMPLETE WITH REJECTED PERCEPTION | 40/40 frames rejected; 0 predictions, 146 GT boxes, TETA/LocA/AssocA/ClsA all 0, official receipt present, 0.0037955 FPS. |
| J1 repaired OVMOT video63 | NOT STARTED AFTER VERIFIED BOUNDARY | Video4 exposed malformed wire output and 239.58--288.17 s/frame; stopped before indefinite invalid-output GPU consumption. |
| YesTrack real verification | NOT STARTED | No current real YesTrack process is active. |
| G1 | WAITING | No current G1 process is active; queued owner remains dependency-gated. |

## Current errors

The OVMOT decoder emits malformed coordinate/box tokens, duplicate coordinate or phrase entries, non-degenerate-box violations, and missing separators. The pipeline records the raw rejection and emits no guessed box. Video4’s official receipt includes all rejected frames; video63 has no result receipt.

## Pushed evidence scope

The JSON and Markdown records contain receipts, scopes, metrics, hashes and failure boundaries. Model weights, videos and caches remain at their recorded absolute paths and are not copied into Git.
