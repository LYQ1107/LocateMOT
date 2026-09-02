# L84 failure decomposition and handoff

L84 implementation contracts passed: the four-group regression reached the
real newer GroundingDINO path, all selected states were finite, rows were not
deleted/truncated, canonical initialization differences were zero, and all
three seeds completed the registered ten-epoch paired probe. The only failed
attempts were implementation/protocol issues: the first contract summary
treated false deletion flags as failed checks, and the first selection
correction used the wrong schedule metadata. Both were fixed minimally in new
attempts; their evidence is retained.

The scientific stable gate passed for Z1 and R1. It did not pass for R4/R6
after all conditions and paired bootstrap. Z1/R1 were tied in the saved model
state and dev records; the corrected earliest-stage tie-break selects Z1. The
single corrected Z1 no-refPE test failed, so no-refPE is not selected.

This does not establish deployable expression correspondence, full-video
RMOT, HOTA or TrackEval. The exact next action authorized by the L84→L85 plan
is `L85_FULL_RMOT`, using Z1 and the query-independent L69 contract.
