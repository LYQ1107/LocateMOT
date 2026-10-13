# Current status — 2026-10-14

Old U2 route stopped by the user research pivot; feature cleanup complete.
Independent branch `codex/groundjev-4gpu-20261013` and worktree
`/data2/user/LocateMOT_GroundJEV` inherit the freshly fetched remote U2
commit `51614cb149f7b4d11606b976589f9476af99d72c`.

Official sources cloned directly without proxy:
GroundAnything `8a3b43a1b42952e117960e85d63ff173bb9fba0a`,
OneJev `2470c7595688c6a402560cf18b4753feb1fe290a`.
Independent Python 3.12.15 installed. GroundAnything custom SGLang setup is
complete in its separate Python 3.12 environment, with the official serving
dependency/ABI check passed (documented cuDNN9.16 override only).
OneJev's isolated Python3.10 environment is installed: Torch2.10.0+cu128, torchvision0.25.0, Transformers5.17.0,
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
No proxy is used. OneJev official weights are complete: 2,214,590,296-byte
model.safetensors SHA256
`cb4b7456703baa26a0cfacf177067200b50021df5a6ba2a2cd1e6f6c868f2244`
matches the official LFS digest. GroundAnything DLM is also complete: its
9,687,419,736-byte model.safetensors SHA256
`3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`
matches the official LFS digest. The original downloader has finished.

Actual OneJev GPU loading exposed the wrong initial Transformers5.5.4 pin:
official shared-prefix self-check error .0857 exceeded its unchanged .05
threshold. The checkpoint itself declares5.17.0. Correcting only the isolated
OneJev runtime to5.17.0 made the original auto check pass and selected batched
mode. All474 weight entries loaded; the real MOT17 frame000302 produces native
Choice/Noul/Score results. Request time .286737s, peak allocator3,119,928,832
bytes on physical GPU6. This is a model-load/real-frame check, not tracking FPS.
Its generic identity question selected IGNORE (.63311) over NEW (.36689);
these results are preserved and no tracking accuracy claim follows from them.

Official PyPI Torch/cuDNN/cuBLAS/Triton/sgl-kernel/nvshmem wheels were prefetched
in verified byte ranges. GroundAnything reused only exact-version `py3-none`
NVIDIA dependencies, keeping its Torch2.9.1/Transformers5.5.4/custom SGLang
recipe unchanged. Remaining official serve installation stages completed with
the local wheelhouse and passed the official cuDNN compatibility check.
Transport-control interruptions preserve completed package caches and are not
training failures.

Implemented initial DLM candidate adapter, real pretrained OneJev decision
adapter, shared persistent memory and serializable causal tracker, a streaming
video runner and fixed-candidate replay control. Native GroundAnything hidden
features are not yet exported; Full's historical visual evidence currently uses
real anchor/long-term/recent crops. The official DLM service is running and its model endpoint responds; the
first real four-frame MOT17 run is in progress with the native DLM and OneJev.
CPU memory save/restore and past-time rejection passed on a clearly labeled
two-frame annotated fixture; it is not a detector/OneJev benchmark.

Verified legal images: MOT17-02 dev 299 frames, DanceTrack0004 full validation
video 1,203 frames, MOT20-01 dev 214 frames. The OneJev real-frame GPU check
completed; no tracking metric or training run exists yet. Device allowlist is 4,5,6,7; individual jobs
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

Current: the single service owner launched official DLM denoising on physical
GPU6 with .4 static memory fraction. The first clip has one blocking owner and
reads real images plus its own prior identity memory; its completion and
tracking outputs remain unverified. The user-requested GPU audit confirmed
UDMT jobs on physical4/5/7; GroundJEV currently uses only physical6, alongside
one other user's process. Other projects are untouched.

Next: validate the real online clip, run full-video baseline/Full tracking,
then genuine tracking-SFT and RMOT/OVMOT experiments.
