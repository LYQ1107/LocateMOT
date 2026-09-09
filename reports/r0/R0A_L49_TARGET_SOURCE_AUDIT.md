# R0A L49 target-source audit

## Decision

`R0A_BLOCKED_L49_QUERY_SOURCE`. The local source path and API are identifiable,
but they are not safe to use under the R0A label boundary.

The authoritative local file is
`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/locatemot/rmot/l49_data.py`,
SHA256 `a1aae91cfbd8aadfba2e432302b06ff7b3e950fe486b48e1f5a4e80481819b3b`.
Its API is `load_l49_queries(dataset: str) -> list[dict[str, Any]]`. V1
records come from the per-expression L13 directory. V2 records come from the
old L11 monolithic JSON and the newer L16 JSON. Each record contains a stable
sentence and a frame-keyed target map normalized to `dict[int, set[str]]`.

The critical implementation detail is that the old V2 path executes
`json.loads(path.read_text())` on the complete L11 JSON, then loops over video
keys and filters to the allowed set. The file's top-level keys include official
evaluation videos `0005`, `0011`, and `0013`. Thus filtering after parse is not
the R0A-required “restrict before loading” behavior.

This attempt is invalid under the explicit R0A rule. Its machine evidence is
`outputs/r0/audit/r0a_l49_target_source/audit.json`; the preserved stop marker
is `INCOMPLETE.md`. Because the preliminary V2 loader inspection parsed that
monolithic source, the attempt records
`official_test_labels_read=true` and no result from it may be used as a valid
training or evaluation result. This is reported rather than hidden.

## Stop boundary

The 5,314 sparse-row consistency audit was not safely run, and no claim of
sentence/target mismatch counts is made for R0A. No dense index, visual cache,
training, dev selection, semantic diagnostic, screening, official-test metric,
HOTA, or TrackEval was run. Ordinary MOT, OVMOT, UIDM, tracker, L69 bank and
production code were untouched.

## One next action

Obtain a supervisor-approved train-scope target source or an isolation reader
that never opens official-eval records, then rerun only this source audit. Do
not resume R0A dense indexing or model work before it passes.
