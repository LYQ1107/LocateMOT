# Current status — 2026-10-13

Old U2 route stopped by the user research pivot; feature cleanup complete.
Independent branch `codex/groundjev-4gpu-20261013` and worktree
`/data2/user/LocateMOT_GroundJEV` inherit the freshly fetched remote U2
commit `51614cb149f7b4d11606b976589f9476af99d72c`.

Official sources cloned directly without proxy:
GroundAnything `8a3b43a1b42952e117960e85d63ff173bb9fba0a`,
OneJev `2470c7595688c6a402560cf18b4753feb1fe290a`.
Independent Python 3.12.15 installed. OneJev runtime installation and the
official GroundAnything custom SGLang setup are active in separate environments.
PyYAML setup hit a shared uv cache lock; retry with a dedicated setup cache
succeeded. Official model revisions are fixed to GroundAnything
`e44e8f1bd518323db31c551aeaa07e370e12d264` and OneJev
`c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8`.

Downloads encountered aria2 TLS failure and curl HTTP/2 cancellation. Partial
bytes were preserved. The current owner uses HTTP/1.1 and 16 parallel official
8 MiB ranges per large model. A bounded 8 x 1 MiB test completed in 7.397s
(1,134,050 bytes/s aggregate), versus single-connection 91,809 bytes/s.
This measures transport only, not model FPS or a promised whole-file ETA.
No proxy is used. Installation and weight receipt completion are pending.

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

B0 has no formally trained U2 tracking checkpoint. Selected epoch-1 results
are perception measurements only. Correction to inherited U2_STATUS.md:
generic 0.9787925789 is Top150 recall at IoU .5, not AP50; expression Top100
V1/V2 recall is 1.0 / .9785374828. The .7237566558 / .6556441180 values are
mean query AP50 diagnostics, not Top100 recall. Raw result JSON is unchanged.

Next: load DLM and OneJev on at most four registered physical GPUs, construct
real visual-evidence decisions and causal memory, then run a short legal clip
followed by complete-video baseline/Full tracking.
