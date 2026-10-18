# YesTrack official code and paper audit

**Status:** `IMPLEMENTED` / `READ_ONLY_AUDIT` / `SCIENTIFIC_RESULT: NONE`

Audit inputs:

- Official repository: <https://github.com/ggbondrighthere24/YesTrack>
- Audited commit: `ed42b8315b28c94dc595d83b0d49bf57002bb281`
- Paper: <https://arxiv.org/abs/2609.02318>
- Local read-only checkout: `/tmp/YesTrack-20261019-v2`

The repository was read file by file for the model builder, datasets,
collator, trainer, probability extraction, temporal evaluator, protocol
documents, evaluator, and `TrackEval` entry points. No official training
framework or weights were copied into LocateMOT. LocateMOT keeps an adapter
boundary around its existing OneJev `NoulQuestion` engine.

## Model and token contract

| Paper/code item | Official implementation | LocateMOT decision |
| --- | --- | --- |
| Backbone | `refer_llm/modeling/build.py::build_model_and_processor` loads `Qwen3VLForConditionalGeneration`; the default is Qwen3-VL-2B, with optional 4-bit NF4 and bf16/fp16 | Do not load Qwen3-VL into OneJev. Use OneJev's native multimodal engine first and record a separate Qwen baseline if one is ever run. |
| LoRA | PEFT targets `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj`; default `r=16`, `alpha=32` | The verifier is an inference adapter now. A future Yes/No fine-tune must start from a registered J1 checkpoint and use a separate run directory. |
| Decision IDs | The builder obtains `convert_tokens_to_ids("Yes")` and `convert_tokens_to_ids("No")`; its fallback for multi-token or unknown encodings is commented out | OneJev must expose an explicit `noul` yes/no mapping. Preflight both IDs and reject unknown or multi-token mappings rather than silently swapping direction. |
| Probability | `refer_llm/infer_utils.py::build_and_forward_yes_probs` takes final logits and applies `softmax([yes_id,no_id])`; index zero is `P(Yes)` | `locatemot/groundjev/yes_verification.py::parse_yes_no_output` follows the same direction and records the raw source. |
| Training objective | The paper defines two-logit BCE (`p_yes` from the two-token softmax). The public `refer_llm/llm_train.py` mainly trains native answer-token language-model loss through `CollateOut.labels`; it is not an explicit BCE trainer | This distinction is recorded as a research factor. Native OneJev Noul SFT and explicit Yes/No probability training are separate ablations. |

## Data and rendering

`refer_llm/data/datasets.py` expands every sequence/frame/expression/annotated
box, maps expression target IDs to positive labels, and samples other boxes as
negatives. `QwenReferVideoYesNoDataset` constructs a causal window ending at
the current frame and skips frames where that object is absent. Crops use
`crop_with_margin`; coordinates are normalized into the prompt. The official
code therefore uses annotation boxes for supervision/crops, whereas the
LocateMOT corpus tools require real GroundAnything candidates and retain GT
only as a label/evaluation source.

`refer_llm/data/collate.py::make_collate_fn` applies the processor chat
template, verifies the number of image tensors, appends the answer token and
EOS, masks the prompt portion, and supervises the answer region. This is
native answer-token SFT, not the paper's explicit BCE head.

LocateMOT rows use the native contract
`id/split/source/task/state/question/target/media/supervision`. They are not
passed directly to GroundAnything-DLM `messages/images` training. The
CPU-only validator `tools/groundjev/yestrack/validate_yesno_inputs.py` checks
that boundary, and a later render/tokenize preflight must use the actual
OneJev collator before any GPU run.

## Temporal mechanisms

The paper defines two inference-time mechanisms:

1. **TCP:** if the minimum of the previous `K` identity scores is at least
   `alpha`, add `lambda` to the current score and clip at one.
2. **TRP:** verify key frames at interval `Delta`, immediately verify new
   identities, and propagate the last score on active tracks between keys.

The official evaluator implements these in
`refer_llm/llm_eval_from_mot.py`: `stability_cache` is keyed by tracker ID,
new IDs are re-evaluated, and the video refinement path samples a causal
history window. Official experiment defaults are `p_l=0.2`, `p_h=0.8`, final
`gamma=0.4`, `K=3`, `alpha=0.4`, `lambda=0.3`, memory size 4, and
`Delta=5` (V1) / `10` (V2). These are paper settings, not validated
LocateMOT optima; our stage-0 config leaves them disabled.

LocateMOT's `semantic_temporal_cache.py` includes persistent track ID, task,
normalized expression, verifier checkpoint hash, and protocol version in the
key. It invalidates on query change, reappearance, conflict, and track
termination. It caches semantic/category probabilities only; it cannot skip
physical association.

## YesTrack-MOT mapping

The paper's MOT extension uses distance gating, image-image pair verification,
cost `1-P(Yes)`, Hungarian one-to-one assignment, and a short lost-track
window. LocateMOT retains its existing Choice/IoU/appearance shortlist,
NEW/IGNORE/NO_MATCH actions, lifecycle, and assigned-ID memory update. The
optional hook in `locatemot/groundjev/assignment.py` fuses verifier evidence
inside the same global solve and returns an event audit; a `No` never creates
a new identity, and memory writes remain after final assignment in the caller.

## Protocol and reproducibility risks

- `docs/USAGE.md` and `docs/POSSIBLE_ISSUES.md` document V1/V2 GT, the
  released 158-entry V1 seqmap, comma-safe expression names, and frame
  conversion. `eval_refer_kitti_mot.py` rewrites prediction frame IDs before
  TrackEval; a mismatched offset can silently pair an adjacent image.
- Official V2 uses negative downsampling and has different score calibration;
  thresholds must not be copied across V1/V2.
- Official referential Accuracy/Precision/Recall can use GT tracklets to
  isolate language selection. Those diagnostics are separate from full
  RMOT HOTA/DetA/AssA and cannot be reported as tracking gains.
- Public code covers RMOT and a simple YesTrack-MOT description; it does not
  provide a complete OneJev MOT/RMOT/OVMOT trainer. We must not claim a
  direct public YesTrack-MOT training port.
- The official implementation uses MIT licensing. LocateMOT's independent
  adapter contains no copied code; if a future substantive excerpt is added,
  preserve the upstream license and attribution.

## Reusable versus non-reusable pieces

Reusable ideas are two-token logit extraction, causal image-history sampling,
key-frame verification, TCP/TRP as explicitly named ablations, and separate
referential versus full tracking metrics. Non-reusable assumptions are the
Qwen processor/tokenizer, GT annotation crops, official V1/V2 sequence
splits, and the simple tracker replacing LocateMOT's persistent physical
identity core. Any future experiment must record the model/checkpoint hash,
candidate provenance, frame convention, verifier call budget, and exact
scope before it can produce a scientific result.
