# L88 Initial Plan

The authoritative preregistration is
[`L88_PREREGISTERED_PLAN.md`](l88/L88_PREREGISTERED_PLAN.md). L88 tests one
variable: zero-initialized rank-16/alpha-32 local LoRA on the final two
GroundingDINO fusion layers and decoder layer 0, with the L86/L87-A sidecar and
loss unchanged. L87-A/B evidence, frozen inputs, cache placement, target
manifest, parity smoke, 40-epoch S/T/J training, video-disjoint dev-HOTA
selection, fixed semantic evaluation, internal V1/V2 TrackEval, stopping rules,
and forbidden paths are recorded there before implementation.

The branch starts at L87-A commit
`0f5d8e9cf5b7d31966104cf06302630011580601`; the fixed manifest must remain
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
