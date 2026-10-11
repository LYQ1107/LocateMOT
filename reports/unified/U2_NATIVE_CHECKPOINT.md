# U2 actual native foundation/core checkpoint

The one-file format `locatemot-u-u2-shared-track-core-v1` embeds every actual
MM-GroundingDINO foundation/shared BERT tensor, the resolved native architecture,
BERT attention implementation and tokenizer assets, every shared persistent
core/candidate/memory/motion/lifecycle/OVMOT tensor, task embeddings and full
1,203-class shared text features. Its loader builds native MMDetection modules
from this file and loads both state dicts strictly. It needs installed model code;
it does not load an external U1 checkpoint, pretrained BERT weights or model
configuration file. Both tasks call the same actual foundation/core objects.

Task embeddings are two zero-initialized 256-D learned vectors in the bounded
semantic gate. Supplied zeros consume no initialization RNG, preserving all
subsequent shared-core initialization at the original seed. They never enter
raw identity logits, anchors or identity GRU. Ordinary MOT gate remains exactly
zero and its adapter receives no gradients/AdamW decay. Their 512 parameters
belong to the existing OVMOT adapter optimizer group, without LR/loss changes.
The joint recipe records this required component before formal training begins.

`tools/unified/pack_u2_checkpoint.py` recomputes the original full perception,
simple/MOT-only eligibility and all four complete joint selection boundaries.
It accepts only the actual selected frozen-foundation U2-B checkpoint, never a
contract fixture. Its metadata includes U1R2 base/source hashes, selected
foundation/cache/manifests, dense base-only supervision, original MOT reference,
joint step/provenance, task sampler, frozen state, and actual sharing counts.
Packing alone leaves readiness false. Run
`tools/unified/audit_u2_checkpoint_retention.py` on the actual packed artifact
before final readiness assessment; the reference-only audit does not replace it.

Actual F0 reference-pack validation passed. External tensor/config/HF pretrained
loads were explicitly prohibited during reconstruction. All foundation/core
tensors and full class features are bit-exact; MOT Top150 and full-vocabulary
OVMOT Top300 candidate features, scores/classes/boxes and nonempty causal state
resume are bit-exact. The same small deterministic V1/V2 legal images give
identical native/packed generic, expression and true/shuffled predictions and
metrics. All original full U1 references remain unchanged. Evidence:
`outputs/unified/u2/eval/native_checkpoint_contract.json`.

Count actual unique parameters, excluding shared native detection and core
association/motion/lifecycle heads and conservatively retaining the entire
OVMOT adapter in the task-specific non-head denominator. The reconstructed
reference architecture has 236,751,992 shared non-head parameters and 329,475
task-specific parameters, giving .9986102878 >=.90. Recompute on the actual
selected artifact too. This count uses the real foundation, not the fallback.

This local reference file is about 960 MB and contains actual F0 weights with
an explicitly random contract core. It is ignored by Git and is not the final
selected U2 model. No formal track benchmark or U3 readiness is claimed. Updated
eight-GPU BF16 joint/RNG and whole official evaluation proofs, causal isolation,
temporal supervision and gate checks pass with task embeddings included.
