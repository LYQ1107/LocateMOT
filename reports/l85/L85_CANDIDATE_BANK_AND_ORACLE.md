# L85 candidate bank and oracle ceiling

The L85 input is a validated, query-independent reuse of the L69 budget-40
bank. The full internal V1/V2 candidate bank audit and manifest are at
`outputs/l85/audit/protocol/` and `outputs/l85/bank/fullvideo_manifest.json`.
The fixed manifest SHA is
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.

The post-hoc candidate oracle at
`outputs/l85/audit/candidate_oracle/oracle.json` covers 1,218 internal
validation units. It reports unit coverage `0.8385300668151447` and target
micro coverage `0.8792302587923025` (the per-video/category denominators are
in the JSON). These are GT-privileged candidate upper bounds only. They are
not model precision, correspondence success, HOTA, or TrackEval.

The oracle uses sidecar candidate-to-GT IDs only after candidate rows have
been constructed. Present-uncovered is not treated as inactive, and duplicate
candidate indices remain legal row metadata. No screening or official-test
labels were read. The audit does not authorize changing the bank or selecting
a semantic threshold.
