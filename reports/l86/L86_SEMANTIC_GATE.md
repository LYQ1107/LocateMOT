# L86 fixed semantic gate

This is the fixed 16-calibration/24-validation candidate semantic diagnostic,
not HOTA and not a screening or official-test result. The authoritative
machine output is
`outputs/l86/eval/fixed_semantic_attempt2/semantic.json`; attempt1 is retained
with its first implementation traceback. The checkpoint and Rule-B emission
rule were selected on internal dev before this fixed slice was read.

| metric | immutable L29 validation | L86 candidate-only | L86 final frozen rule | requirement |
|---|---:|---:|---:|---:|
| candidate recall | 0.7333333 | 0.3548387 | 0.3225806 | >= 0.7233333 |
| precision | 0.0830189 | 0.1145833 | 0.1176471 | >= 0.0830189 |
| FP/frame | 10.1250 | 3.5417 | 3.1250 | <= 11.125 |
| predictions/positive | 8.8333 | 3.0968 | 2.7419 | <= 4.069 |
| hard violation | 0.9166667 | 0.8461538 | 0.8461538 | <= 0.8666667 |
| multi-positive recall | 0.8194444 | 0.2638889 | 0.2638889 | >= 0.7894444 |
| inactive false acceptance | 1.0000 | 1.0000 | 0.8333333 | < 1.0 |
| empty rate | 0.0000 | 0.1250 | 0.2083 | no collapse |

The frozen-rule validation slice contains 24 units, 1,468 candidate rows and
31 positive rows. All candidate rows remained in score records; no top-k,
NMS, candidate deletion or truncation was used. The preselection audit proves
that 40 fixed records were scored before labels were attached, and validation
labels were attached only after the dev checkpoint/rule was frozen.

The simultaneous gate is `semantic_gate_fail`: hard violation, precision,
volume and inactive false acceptance satisfy their individual floors, but
recall drops by about `0.41075` and multi-positive recall drops by about
`0.55556`. The low-volume result is therefore not a deployable correspondence
fix. The immutable L29 control remains the accepted control; L53/L54 controls
are included in the machine semantic JSON for comparison.

No HOTA, TrackEval, screening or official-test conclusion is inferred from
this table.
