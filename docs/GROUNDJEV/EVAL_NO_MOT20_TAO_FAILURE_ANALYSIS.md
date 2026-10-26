# TAO native output failure analysis

Status: registered complete-video diagnostic; full993-video scope remains INCOMPLETE.

The final G2 step18227 uses the frozen official_scoring parser, tracking_dense decoder, request-local seed42 and all1,203 LVIS-v1 labels in13 fixed chunks. This analysis only reads persisted raw responses; it does not read GT, call G2, alter candidates or rescore a partial vocabulary.

The completed scope is7 videos and124 annotated frames. Of1612 native requests, 375 chunks pass the official parser (23.26%) and1237 fail. Every completed frame has at least one failed chunk, so all124 formal frame outputs are empty under the registered protocol.

Accepted chunks contain33,911 native proposal records that are suppressed by the whole-frame rejection rule. These are not unique objects, verified detections, GT coverage or a legal partial-vocabulary score. Their existence explains why zero formal output does not imply every raw request returned no coordinates.

| First detected structural failure | Rejected requests |
|---|---:|
| malformed_coordinate_payload | 659 |
| invalid_native_label | 244 |
| malformed_wrappers_or_entry_separators | 223 |
| invalid_native_box_geometry | 65 |
| missing_complete_native_wrappers | 37 |
| strict_failure_then_reference_fallback_rejects_None_entry | 9 |

A request may contain several defects; this table assigns its first located structural defect. It is not an independent causal decomposition. Duplicate tuples alone may be recovered by the pinned official deduplication parser; the rejected categories describe the remaining failed responses.

The full per-frame invalid-chunk histogram, completed-cache cross-checks and partial-progress separation are in `EVAL_NO_MOT20_TAO_REJECTION_FUNNEL.json`. All seven completed-cache frame/rejection/request counts agree with their frozen receipts. Incomplete videos remain explicit diagnostic entries and are excluded from the completed-video totals above.

Decision: retain the failure evidence and continue the entire frozen validation queue. The formal parser, prompts, vocabulary, decoder, thresholds and model checkpoint are unchanged. No partial-chunk rescue is used in official Simple/J1 TETA.

Use the GroundAnything environment for this diagnostic: the OneJev environment lacks the grounding_anything client required by the pinned official parser. The initial offline import failed before producing a report; rerunning in the separate GroundAnything environment passed the real-cache cross-checks. This was an offline environment error and did not affect running inference.

```bash
CUDA_VISIBLE_DEVICES='' /data2/user/LocateMOT_GroundJEV/external/GroundAnything/.venv-serve/bin/python \
  tools/groundjev/eval_no_mot20/diagnose_native_outputs.py \
  --tokenizer /data2/user/LocateMOT_GroundJEV/models/GroundAnything-DLM-G2-CLEAN-V2
```
