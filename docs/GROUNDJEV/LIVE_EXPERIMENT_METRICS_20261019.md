# Live experiment metrics and unresolved problems (2026-10-19)

This is a point-in-time evidence index. A missing receipt means the run is
unfinished; it is not interpreted as zero.

| Work item | State | Evidence / current result |
|---|---|---|
| G2 clean formal training | RUNNING on GPU3/GPU5 | Last confirmed event was 3955/18227; linear remaining estimate was about 60 h. Main training was not interrupted. |
| J1 step-1200 training | COMPLETE | Durable three-task checkpoint and optimizer/RNG state exist. |
| J1 original held-out gate | COMPLETE, immutable | All registered scopes were zero; root cause audit attributes this to strict parser rejection plus all-NO_MATCH association. |
| J1 repaired DT1203 | COMPLETE | HOTA 0.1585619177, DetA 0.1976921614, AssA 0.1272432780, IDF1 0.1282768263, IDSW 0, FP 110, FN 3560. |
| J1 repaired OVMOT video4 | RUNNING / partial | Malformed GroundAnything decoder wire output causes empty candidates and 250--281 s slow-path frames; no official receipt yet. |
| J1 repaired OVMOT video63 | NOT STARTED in this suite | Waiting for video4. |
| YesTrack real verification | NOT STARTED | Historical controls/checkpoints exist, but no current real YesTrack process is active. |
| G1 | WAITING | No current G1 process is active. |

## Current errors and empty outputs

The active OVMOT replay records malformed decoder payloads, including missing
coordinate/box tokens, duplicate coordinate or phrase entries, and a
non-degenerate-box violation. The pipeline emits an empty candidate list for
those frames and keeps the raw rejection reason. It is unsafe to synthesize a
box or silently drop the frame. The next repair will be a separately named
replay with an explicit protocol receipt.

## Pushed evidence scope

The accompanying JSON and the J1 repair report are intended to be committed
without model weights, videos, or caches. Large runtime artifacts remain at
their recorded absolute paths and are not copied into Git.

