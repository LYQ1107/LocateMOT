# TAO TRAIN native export owner — 2026-10-21

The resumable owner `scripts/groundjev/queue_tao_native_train_shards.py` is now
running in `WAITING_G2_CLEAN_SUCCESS` after a resource-gate repair. The first
owner was restricted to GPU4/6, but both remained occupied by unrelated
processes, so it could never launch. The resumed owner uses the registered
fallback pair GPU1/3 and launches only when each selected GPU reports at least
48 GiB free. Existing foreign processes may remain resident under that memory
floor; they are recorded and never terminated or modified. GPU5 remains
reserved for G1 and GPU7 for held-out evaluation.

After the exact G2 terminal receipt, the owner first merges the terminal G2
adapter into the private bundle
`models/GroundAnything-DLM-G2-CLEAN-NATIVE-TAO-V1`. This bundle is separate
from G0, the G1 bundle and the ordinary G2 deployment bundle; the published
G0 weights are never used to label this stream. For each of the eight
registered shards, the owner then starts an isolated GroundAnything service
pair with unique local ports, streams all 1,203 vocabulary classes with the
explicit TRAIN gate, records a terminal receipt, and stops the owned services.
After all eight receipts pass, it invokes the strict merge gate. The owner
never relabels validation data and never sets `native_ovmot_training_rows=true`
before the complete native stream and offline TAO TRAIN join exist.

The stream entrypoint now accepts `--grounding-url` and `--native-feature-url`, while retaining the old 8104/8112 defaults for existing validation controls. The corresponding runtime copy under `/data2/user/LocateMOT_GroundJEV` was synchronized so the queued services and stream use the same TRAIN contract.

At re-registration, all eight input shard manifests and the private G2 native
bundle were already complete, but no native candidate row or optimizer update
had been produced. The owner was resumed from this point; no G2 checkpoint was
retrained and no foreign job was changed.
