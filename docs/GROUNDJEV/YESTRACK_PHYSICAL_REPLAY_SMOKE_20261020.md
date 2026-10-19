# YesTrack physical replay smoke — 2026-10-20

The opt-in verifier was connected to the actual `OnlineTracker` assignment
path and run with the fine-tuned OneJev model on the preserved MOT17-02
candidate trace `mot4_after`.  The source SHA-256 is
`9dfd8ee5f3158c1251f8b33123cce2c526d3b57cd894cc5901936d6382e6e31c`; it
contains the exact native candidate features, causal decision rows and four
physical frames (302–305).

The verifier-disabled replay reproduced all four source output rows exactly.
The enabled arm used the same candidates and the same single global assignment
per frame, called the real fine-tuned model six times (two calls per frame on
frames 303–305), and retained 31/28/27/29 output tracks on frames 302–305.
The merged verifier SHA is
`eb2e7e252600911fd4695c5193129389c3851db1afe7399cb5b9be1aee7b8c1f`; the
enabled replay took 2.900 seconds (1.379 frames/s including model calls).

This is a wiring and latency receipt only.  Four frames are not a legal
complete-video scope, so no HOTA, AssA, IDF1 or IDSW value is reported.  The
complete-video comparison remains pending until one full legal candidate archive
can be paired with the same causal verifier request key for the no-verifier,
base-verifier and fine-tuned-verifier arms.

Runtime artifacts:

- `/data3/user/LocateMOT-runs/YesTrack/physical_replay_mot4_finetuned_v1/report.json`
- `/data3/user/LocateMOT-runs/YesTrack/physical_replay_mot4_finetuned_v1/baseline_outputs.json`
- `/data3/user/LocateMOT-runs/YesTrack/physical_replay_mot4_finetuned_v1/verified_outputs.json`
