# TAO TRAIN native export owner — 2026-10-21

The resumable owner `scripts/groundjev/queue_tao_native_train_shards.py` is now running in `WAITING_G2_CLEAN_SUCCESS`. It will not start until the exact G2 Clean terminal receipt exists and it sees an idle physical GPU. It is restricted to GPU4 and GPU6; GPU3/5 remain reserved for G2/G1 and GPU7 remains reserved for held-out/YesTrack. A GPU with any compute application is treated as unavailable, so existing foreign jobs are preserved.

For each of the eight registered shards, the owner starts an isolated GroundAnything service pair with unique local ports, streams all 1,203 vocabulary classes with the explicit TRAIN gate, records a terminal receipt, and stops the owned services. After all eight receipts pass, it invokes the strict merge gate. The owner never relabels validation data and never sets `native_ovmot_training_rows=true` before the complete native stream and offline TAO TRAIN join exist.

The stream entrypoint now accepts `--grounding-url` and `--native-feature-url`, while retaining the old 8104/8112 defaults for existing validation controls. The corresponding runtime copy under `/data2/user/LocateMOT_GroundJEV` was synchronized so the queued services and stream use the same TRAIN contract.

At registration, no TAO candidate row or optimizer update had been produced. The owner is a waiting process only; no GPU or foreign job was changed.
