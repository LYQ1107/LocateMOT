# Reproduce this server's frozen evaluation

The research branch is `codex/g2-four-gpu-no-mot20-eval-20261026` in
`LYQ1107/LocateMOT`. It contains the inference/replay/scoring code, manifests
builder, exact model identities, completed-result snapshots and negative results.
Large pretrained weights, raw images and candidate databases stay on the server.

From a clean checkout on this server:

```bash
python3 tools/groundjev/eval_no_mot20/link_local_assets.py --asset-root /data2/user/LocateMOT_GroundJEV
```

Use the two separate Python executables in
`configs/groundjev/eval_no_mot20_4gpu_v1.json`. Package versions are recorded in
`EVAL_NO_MOT20_GROUND_ENV.json` and `EVAL_NO_MOT20_ONEJEV_ENV.json`.
`EVAL_NO_MOT20_DEPENDENCY_SOURCES.json` pins the clean GroundAnything, OneJev,
TrackEval and TETA source revisions. The decoder seed overlay belongs to this
research checkout, not the shared official source tree.

`prepare_eval_manifests.py --out ...` rebuilds the registered video/query manifests
from the original source corpus and legal annotations. `audit_models.py --out ...`
checks the immutable final G2, original J1 and existing Coupled readout. Paths in
`common.py`, the runtime config and the manifests bind this run to its original
server assets. A new output root requires consistent paths and fresh manifests;
do not combine cached requests from different model/query/decoder contracts.

The live detection command is:

```bash
/data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python tools/groundjev/eval_no_mot20/dispatch_4gpu.py --config configs/groundjev/eval_no_mot20_4gpu_v1.json --stage detection
```

Do not start a second owner over the active queue. After a worker has actually
terminated, `--recover-worker A` (or B/C/D) recovers only that slot and its
unfinished units. `supervise_workers.py` performs this internally. A complete
cache is scored with Simple before the slot rotates to immutable J1 and compatible
Coupled replay. No video's association history is split into independent parts.

For reports and an immutable small publication snapshot:

```bash
/data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python tools/groundjev/eval_no_mot20/snapshot_results.py
```

The source tests do not establish tracking quality. Actual results must have
complete video/query receipts, unchanged candidate input, official metrics and
their frame counts. TETA uses joint official scoring with Base/Novel boundaries;
cache replay FPS excludes G2 detection. The full registered goal remains
incomplete until all requested evaluable inputs have been processed.
