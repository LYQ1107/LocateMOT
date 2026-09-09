# R0A real forward/backward contract smoke

Status: `PASS` for the R0A implementation gate; this is not a semantic or
TrackEval result.

The authoritative run is `outputs/r0/audit/r0_contract_smoke_retry2/`.  It
used the isolated R0A worktree and the registered `/home/lwr/anaconda3/envs/masaenv/bin/python`
runtime on one GPU.  The selected legal fit example was Refer-KITTI-V1,
video `0001`, frame `0`, query `1`, with 57 native L69 candidate rows.  The
18 duplicate `candidate_index` values were retained as distinct row offsets.

The query-independent GroundingDINO visual item was constructed and removed
from a temporary `/data2/usr_for_deadline` directory after the check.  Its
inner/context shapes were `[57,36,256]`; the pure L89 language input was
`[1,256,256]` with mask `[1,256]` and 8 valid tokens; geometry was `[57,5,10]`.
The frame-specific label lookup ran only after the complete visual item had
been constructed.  Loss was finite (`3.555824041366577`), all 108 trainable
parameter gradients were finite and nonzero, detector/tracker parameters were
not in the optimizer, and detector gradients were absent.  Strict reload
passed with maximum membership-logit difference `0.0`.

The known harmless checkpoint warning about
`language_model.language_backbone.body.model.embeddings.position_ids` remains
an expected non-strict-load warning in this local runtime; it is recorded in
the machine-readable provenance and is not treated as a data or weight
failure.

This smoke does not authorize semantic selection by itself.  The next R0A
action is the registered query-independent visual cache, followed by the
separate benchmark-specific training/evaluation chain.  The fixed manifest
and all old MOT/OVMOT assets remain unchanged.

Flags: `screening_gt_used=false`, `official_test_labels_read=false`,
`ordinary_mot_ovmot_touched=false`, `hota_trackeval_run=false`.
