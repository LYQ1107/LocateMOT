# Current status — 2026-10-13

Old U2 route stopped by the user research pivot; feature cleanup complete.
Independent branch `codex/groundjev-4gpu-20261013` and worktree
`/data2/user/LocateMOT_GroundJEV` inherit the freshly fetched remote U2
commit `51614cb149f7b4d11606b976589f9476af99d72c`.

Official sources cloned directly without proxy:
GroundAnything `8a3b43a1b42952e117960e85d63ff173bb9fba0a`,
OneJev `2470c7595688c6a402560cf18b4753feb1fe290a`.
Installing isolated runtimes and downloading real weights next.
No GroundJEV GPU job or tracking result exists yet.

B0 has no formally trained U2 tracking checkpoint. Selected epoch-1 results
are perception measurements only. Correction to inherited U2_STATUS.md:
generic 0.9787925789 is Top150 recall at IoU .5, not AP50; expression Top100
V1/V2 recall is 1.0 / .9785374828. The .7237566558 / .6556441180 values are
mean query AP50 diagnostics, not Top100 recall. Raw result JSON is unchanged.

Next: load DLM and OneJev on at most four registered physical GPUs, construct
real visual-evidence decisions and causal memory, then run a short legal clip
followed by complete-video baseline/Full tracking.
