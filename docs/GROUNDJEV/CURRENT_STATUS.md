# Current status — 2026-10-13

Old U2 route stopped by the user research pivot; feature cleanup complete.
Independent branch `codex/groundjev-4gpu-20261013` and worktree
`/data2/user/LocateMOT_GroundJEV` inherit the freshly fetched remote U2
commit `51614cb149f7b4d11606b976589f9476af99d72c`.

Official sources cloned directly without proxy:
GroundAnything `8a3b43a1b42952e117960e85d63ff173bb9fba0a`,
OneJev `2470c7595688c6a402560cf18b4753feb1fe290a`.
Independent Python 3.12.15 installed. GroundAnything custom SGLang setup is
active in its separate Python 3.12 environment. OneJev's isolated Python3.10
environment is installed: Torch2.10.0+cu128, torchvision0.25.0, Transformers5.5.4,
69 packages compatible; CUDA availability checked with physical GPU6 visible.
Matching CPython3.10 framework files were copied from the installed local
framework distribution into the new environment; no shared package symlinks
or source-environment writes. This supersedes the slow Python3.12 OneJev
network installation, stopped deliberately after the copied runtime passed.
PyYAML setup hit a shared uv cache lock; retry with a dedicated setup cache
succeeded. Official model revisions are fixed to GroundAnything
`e44e8f1bd518323db31c551aeaa07e370e12d264` and OneJev
`c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8`.

Downloads encountered aria2 TLS failure and curl HTTP/2 cancellation. Partial
bytes were preserved. The current owner uses HTTP/1.1 and 16 parallel official
8 MiB ranges per large model. A bounded 8 x 1 MiB test completed in 7.397s
(1,134,050 bytes/s aggregate), versus single-connection 91,809 bytes/s.
This measures transport only, not model FPS or a promised whole-file ETA.
No proxy is used. Official weight receipt completion and real model loading
remain pending. One blocking queue automatically launches the real MOT-frame
OneJev check once verified weights are complete; it does not load partial files.

Official PyPI Torch/cuDNN/cuBLAS/Triton/sgl-kernel/nvshmem wheels were prefetched
in verified byte ranges. GroundAnything reused only exact-version `py3-none`
NVIDIA dependencies, keeping its Torch2.9.1/Transformers5.5.4/custom SGLang
recipe unchanged. The remaining official serve installation stages resume with
the local wheelhouse and must pass the official cuDNN compatibility check.
Transport-control interruptions preserve completed package caches and are not
training failures.

Implemented initial DLM candidate adapter, real pretrained OneJev decision
adapter, shared persistent memory and serializable causal tracker, a streaming
video runner and fixed-candidate replay control. Native GroundAnything hidden
features are not yet exported; Full's historical visual evidence currently uses
real anchor/long-term/recent crops. These interfaces await real model integration.
CPU memory save/restore and past-time rejection passed on a clearly labeled
two-frame annotated fixture; it is not a detector/OneJev benchmark.

Verified legal images: MOT17-02 dev 299 frames, DanceTrack0004 full validation
video 1,203 frames, MOT20-01 dev 214 frames. No GroundJEV GPU job, tracking
metric or training run exists yet. Device allowlist is 4,5,6,7; individual jobs
can use fewer, and other projects remain untouched.

Official ordinary-MOT export/evaluation entrypoint is implemented. A complete
299-frame GT-oracle fixture passed unchanged TrackEval HOTA/CLEAR/Identity
evaluation and frame reindexing. Its perfect fixture result is explicitly not
a model benchmark. Replay controls require identical physical image paths.

B0 has no formally trained U2 tracking checkpoint. Selected epoch-1 results
are perception measurements only. Correction to inherited U2_STATUS.md:
generic 0.9787925789 is Top150 recall at IoU .5, not AP50; expression Top100
V1/V2 recall is 1.0 / .9785374828. The .7237566558 / .6556441180 values are
mean query AP50 diagnostics, not Top100 recall. Raw result JSON is unchanged.

Next: load DLM and OneJev on at most four registered physical GPUs, construct
real visual-evidence decisions and causal memory, then run a short legal clip
followed by complete-video baseline/Full tracking.
