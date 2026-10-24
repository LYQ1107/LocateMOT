# TrackJEV box-contract audit — 2026-10-24

The immutable MOT identity source used by the active native three-task run was
audited on a deterministic 512-row prefix. Every scanned row has candidate
state boxes in native pixel `xyxy` and stored competition rows that match the
correct `xyxy` calculation. None has an explicit `competition_box_format` tag.

The same stored rows do **not** match the historical runtime failure mode that
passes the raw `xyxy` values to an `xywh` implementation. On the first row, the
stored first row is `1000,44,...`, the correct `xyxy` value is `44`, while the
raw-as-`xywh` interpretation produces `571` (and additional false overlaps).
The stored values also match the equivalent GT-matching calculation after
converting candidates to `xywh`; this is why the old exporter appeared correct
inside its own converted computation while the runtime state remained
ambiguous.

The active manifest and source rows were not changed. The runtime/exporter fix
adds an explicit `box_format` boundary, uses `xyxy` for native candidates and
keeps MOT GT matching in `xywh`. The audit is a bounded contract diagnosis,
not a full-row performance result.

Machine-readable evidence: `TRACKJEV_BOX_CONTRACT_AUDIT_20261024.json`.
