# Phase C — official G2 DLM dense decoder check

This is a decoder-contract diagnostic for the single-backbone GroundJEV
coupling work.  The protected G2 Clean two-rank run was not modified.  The
G2 adapter at step 17,000 was loaded once on GPU4, and the real
GroundAnything `DLMInferenceEngine` with `hierarchy_dynamic` decoding was
used.  Structured termination was enabled with unlimited coordinate groups;
the evaluator did not force a single-box limit and did not synthesize boxes.

The run encoded one real frame from each MOT20 sequence:

| sequence/frame | GT person boxes | predicted boxes | recall @ IoU 0.5 | generated answer |
| --- | ---: | ---: | ---: | --- |
| MOT20-03/82 | 96 | 1 | 0.0000 | `[1,1,5,49]` |
| MOT20-05/13 | 247 | 1 | 0.0000 | `[1,1,998,999]` |

Both outputs were legal `object_ref -> box` pairs and the official parser
recovered them without repair.  The decoder stopped at the model's first
`<|box_end|>` even though unlimited coordinate groups were available.  This
establishes that the earlier one-box causal observation is also present in the
official hybrid DLM path; it does not establish a tracking or coupled-model
gain.  The next coupling implementation must inject persistent identity state
before this same official decoder and train/evaluate it under the complete
GroundAnything answer contract.

Receipt: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_g2_official_dlm_dense_step17000_v1/receipt.json`.
