# R0A research-log addendum

This entry remains under the R0 allow-list; the pre-R0/R0 `research_log.md`
was not modified.

## 2026-09-09 — R0A source-isolation stop

- Hypothesis: repair R0's false canonical target invariant by treating target
  IDs as frame-specific while keeping `(dataset,video,query_id)` sentence
  identity stable.
- Action: created the R0A worktree from failed R0 commit
  `19fcb5fd3dcf1d26adb60a5fcceb3121c6f58c5b`; inspected the actual local L49
  helper and began the frame-specific implementation.
- First actionable blocker: V2 `load_l49_queries` calls `json.loads` on the
  full L11 monolithic expressions JSON before filtering video keys. Its keys
  include forbidden official-eval videos `0005`, `0011`, and `0013`.
- Contract consequence: the preliminary V2 loader inspection parsed that
  source, so this attempt is invalid and records
  `official_test_labels_read=true`. No 5,314-row audit, dense index, cache,
  training, dev, semantic, screening, or TrackEval result is valid.
- Unique next action: provide a supervisor-approved source-isolation reader or
  train-scope target artifact that never opens forbidden official-eval records,
  then rerun only the R0A source audit.
