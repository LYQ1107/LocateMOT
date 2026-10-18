# YesTrack code and paper audit

The official repository was read at commit
[`ed42b8315b28`](https://github.com/ggbondrighthere24/YesTrack/tree/ed42b8315b28)
and the primary paper is
[`arXiv:2609.02318`](https://arxiv.org/abs/2609.02318). The paper describes a
two-stage referring tracker in which an MLLM directly verifies a candidate
with Yes/No, then uses Temporal Confidence Prior (TCP) and Temporal Reference
Propagation (TRP). It also describes YesTrack-MOT as a generic MOT
instantiation. This audit records implementation facts; it does not reproduce
the paper's scores.

## What the official code actually does

* `refer_llm/modeling/build.py` builds Qwen3-VL, attaches PEFT LoRA to the
  attention/MLP projection modules, and obtains tokenizer IDs for `Yes` and
  `No`. The released implementation is therefore a Qwen3-VL/PEFT path, not a
  OneJev checkpoint adapter.
* `refer_llm/data/datasets.py` expands sequence/frame/annotation/expression
  tuples. It crops an annotated box, constructs positive/negative labels from
  expression annotations, and its video dataset uses preceding frames ending
  at the current frame. That is a useful causal data pattern, but the source
  labels are not allowed into a LocateMOT model request.
* `refer_llm/data/collate.py` renders user messages with one or more images,
  appends the literal answer `Yes` or `No`, and masks prompt tokens from the
  supervised loss. It also checks that multimodal image counts reach the
  processor.
* `refer_llm/infer_utils.py::build_and_forward_yes_probs` reads the final
  token logits, selects the Yes/No token pair, and applies a two-way softmax.
  The first probability is explicitly `p_yes`; this direction is preserved in
  `locatemot/groundjev/yes_verification.py`.
* `refer_llm/llm_eval_from_mot.py` supports single-frame verification, a
  historical multi-frame refinement interval, TCP over a previous confidence
  window, and TRP-style inference every `N` frames with immediate scoring for
  new IDs. These mechanisms reuse the latest causal confidence and do not
  solve OneJev's global identity assignment by themselves.
* `eval_refer_kitti_mot.py` collects nested per-expression MOT rows and calls
  TrackEval. The official usage notes warn that V1 and reduced seqmaps are not
  comparable and that frame indexing must be checked first.

## What is reusable here

LocateMOT reuses the discriminative probability contract, explicit Yes/No
direction, causal history window, and selective temporal refinement idea. The
new implementation keeps semantic verification after physical candidate
generation and global assignment, records latency/checkpoint/evidence, and
keeps the baseline disabled by default.

## What is not copied as a hidden assumption

OneJev uses its own multimodal request schema, persistent track memory,
candidate competition, lifecycle actions, and MOT/RMOT/OVMOT task interface.
Qwen3-VL token IDs, crop dimensions, PEFT target modules, TCP thresholds, and
the official Refer-KITTI split cannot be silently transplanted. In particular,
`No` is not converted into `NEW`, semantic verification does not replace
Hungarian/global assignment, and future frames or GT boxes never enter a
request. Any future Qwen3/OneJev fine-tune must be a declared ablation with a
new checkpoint and fixed scope.

Status: `IMPLEMENTED` for the audit and adapter boundary; `WAITING_DEPENDENCY`
for model training and scientific comparison.
