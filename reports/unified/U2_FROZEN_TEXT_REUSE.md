# U2 fixed native text reuse

Upcoming selected-foundation cache workers reuse frozen native BERT outputs
and token/prompt maps for each exact caption. Returned tensors/maps are cloned
before native projection/fusion or prediction consumers can modify them.
The context is local to one worker and restores all native methods on exit.
It caches neither image features nor multimodal encoder/decoder outputs.
Every original image still runs its backbone once and its full registered
vocabulary chunks, native score selection and cross-chunk NMS.

Real FP32 F0 validation covers two distinct MOT17 frames with Top150 and two
TAO validation frames with all 1,203 classes/38 chunks and Top300. Boxes,
scores, classes, query/chunk provenance and every identity/semantic/decoder
feature remain bit-exact. Native two-frame BERT calls are 2/76 for MOT/OVMOT;
cached calls are 1/38, with no BERT computation after the first fixed caption.
Backbone and multimodal decoder call counts remain 2 and 2/76 respectively.
Mutating returned language tensors or token-positive maps does not corrupt
cached values; original native methods are restored afterwards.

The observed warm OVMOT frame took 5.256 seconds natively and 4.638 seconds
with reuse (1.133x throughput, about 11.8% lower elapsed time). MOT took .1890
and .1808 seconds. These are two-frame observations on a shared GPU, not a
measurement of full-cache throughput or final experiment wall time.

The actual cache worker also passes on explicitly separate reference-only
two-frame inventories for both phases. Serialized FP32 box/score/integer fields
and BF16 feature tensors are bit-exact, semantic/decoder storage remains shared,
and GT labels are absent. Re-entering the worker after a complete shard performs
no candidate inference and preserves the completed unit manifest. Evidence:
`outputs/unified/u2/eval/frozen_text_cache_contract.json`.

Cache specifications record the implementation and validation provenance;
completed worker/full manifests retain actual BERT and prompt hit/call counts.
All full inventories, eight-worker assignment, four-volume storage, candidate
caps, frozen-foundation eligibility and original simple/learned tracking gates
stay unchanged. The running F0 audit and perception owner were not modified or
restarted. Formal selected caches and tracking benchmarks remain pending.
