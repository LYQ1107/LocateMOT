# L84 source of truth

The authoritative machine snapshot is
`outputs/l84/preregister/source_of_truth.json`; the frozen asset manifest is
`outputs/l84/preregister/frozen_assets.json`.

* project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
* Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
* L83 base / L84 starting HEAD: `5fd92b87927d60c831e4ac75774929f07f371d7e`
* L84 branch: `codex/l84-paired-middecoder-20260902`
* fixed manifest SHA256: `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`
* UIDM step-11000 SHA256: `f6529743630e335b947118bf860cc2215a95315a4c551351e6866ea9e1076343`
* L69 feature manifest SHA256: recorded in the machine snapshot; all 17
  per-video files and sidecars are hashed there.

The snapshot completed with no source-of-truth mismatch.  It records the
local MMDetection checkout as having no verifiable Git HEAD, while the public
v3.3.0 reference is commit
`44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`.  GroundingDINO, BERT, L48 text,
L81/L82 controls, L83 metric/loss code, and the shared UIDM checkpoint are
read-only inputs.  The first snapshot attempt had only a script dictionary
classification error (`l83_source_files` was incorrectly treated as a file);
the corrected rerun completed without changing any input.

The L84 flags are:

`screening_gt_used=false`, `official_test_labels_read=false`,
`hota_trackeval_run=false`, `ordinary_mot_ovmot_touched=false`,
`candidate_deletion=false`, `candidate_truncation=false`,
`token_span_region_alignment=UNALIGNED`, and
`static_motion_alignment=UNALIGNED`.
