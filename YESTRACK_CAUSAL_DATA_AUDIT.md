# Yes/No corpus causal and split audit

The CPU preparation tools under `tools/groundjev/yestrack/` consume native
candidate exports and retain source provenance. They write the OneJev-style
fields `id`, `split`, `source`, `task`, `state`, `question`, `target`, `media`,
and `supervision`. The target is stored outside the model question; the
question only asks for Yes or No.

RMOT rows use native candidate boxes and the supplied expression. Identity
pairs use a current region and a strictly earlier region. OVMOT rows use the
provided category/vocabulary and reject novel training labels by default.
MOT20-05 is rejected in all three preparers. The native source path and
proposal type are retained in `supervision.proposal_source`.

`audit_yesno_corpus.py` checks required schema fields, duplicate IDs, per-video
split overlap, GT markers in prompts, future evidence flags, non-causal
identity pairs, forbidden MOT20-05, and rejected-frame retention. A split
collision is a failed audit even when the images happen to be different. A
rejected perception frame remains a row/receipt item; it is not dropped to
improve apparent coverage.

The intended train/held-out boundary follows existing LocateMOT legal scopes:
Refer-KITTI fit/native RMOT sources for preparation, base-category OVMOT only,
and no official evaluation labels or reserved videos. Any future J2 corpus
must publish a source manifest and a split-leakage report before training.

This audit does not assert that native candidate coverage is sufficient. It
only makes the proposed verifier's data boundary inspectable and causal.
