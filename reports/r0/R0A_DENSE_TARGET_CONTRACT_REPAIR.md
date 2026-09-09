# R0A dense target contract repair

Status: `NOT_RUN — blocked by authoritative source isolation`.

R0 correctly identified that target IDs are frame-specific, but R0A cannot
claim the required 5,314-row source consistency audit because the local V2
loader parses a monolithic file containing forbidden official-eval records.
The source blocker precedes the dense audit. No target union, pseudo-label,
nearest-frame fallback, or frame propagation was used.

Required counts are therefore `sparse_rows_checked=0 (blocked)`,
`sentence_mismatch=not_run`, `target_mismatch=not_run`, `missing_query=not_run`,
and `invalid_frame=not_run`. No V1 or V2 dense train index was created.

See `R0A_L49_TARGET_SOURCE_AUDIT.md` and the machine audit for the exact source
paths, hash, and failure. The next action is only to obtain a safe train-scope
source/isolation reader and rerun the source audit.
