# R0 source/interface audit

Status: `PASS` for the static frozen-source interface audit; this is not a model
or RMOT result.

The corrected audit is
`outputs/r0/audit/source_interfaces_retry1/audit.json`. It verified the
registered symbols in the read-only L82 GroundingDINO runtime/reference, the
read-only L89D common helper, the L86 full-video helper, and the L89 TrackEval
matrix. The L89D/L89 TrackEval files were referenced from the existing frozen
`LocateMOT_L89E` worktree because they are not present in the R0 base worktree;
no file was copied or modified.

The manifest was read-only verified at
`outputs/l19/protocol/kitti_fast_eval_manifest.json` with SHA256
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
The local MMDetection checkout had no verifiable git HEAD, which is recorded in
the machine audit rather than presented as a verified upstream commit.

The first attempt is retained at
`outputs/r0/audit/source_interfaces/INCOMPLETE.md`; it failed only because the
wrapper used the wrong absolute location for the frozen L89D helper. The
minimal path correction and retry passed. No detector forward was run.
