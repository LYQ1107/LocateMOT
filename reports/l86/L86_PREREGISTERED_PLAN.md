# L86 Preregistered Plan

This file freezes the L86 architecture, objective, schedule, selection and
evaluation protocol before any L86 validation labels are interpreted.

```json
{
  "stage": "L86",
  "hypothesis": "faithful target-bag semantics plus independent presence/NULL energy and causal target-identity supervision can convert L85 target ordering into active recall and full-video HOTA gains",
  "representation": "L84 original-content Z1 fixed-reference decoder layer 1",
  "candidate_source": "frozen L69 budget-40 query-independent bank",
  "obs_dim": 1432,
  "max_history": 8,
  "clip_length": 4,
  "hidden": 256,
  "seed": 20260829,
  "epochs": 40,
  "curriculum": {"S": [1, 8], "T": [9, 20], "J": [21, 40]},
  "loss_weights": {
    "semantic_total": 1.0,
    "semantic_static": 0.3,
    "membership": 1.0,
    "presence": 0.5,
    "null": 0.5,
    "temporal_identity": 0.1,
    "delta_regularization": 0.01
  },
  "temporal_margin": 0.2,
  "null_margin": 0.5,
  "candidate_threshold_grid": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
  "presence_threshold_grid": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
  "null_margin_grid": [0.0, 0.25, 0.5, 0.75],
  "effective_clip_batch": 8,
  "microbatch_per_gpu": 1,
  "checkpoint_epochs": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
  "dev_hota_epochs": [8, 20, 40],
  "dev_checkpoint_selection": [
    "lower target-bag hard violation",
    "higher target-bag hit@1",
    "higher multi-target exact",
    "lower inactive false acceptance",
    "higher candidate recall",
    "earlier epoch"
  ],
  "emission": "presence_logit >= presence_threshold AND candidate_energy >= candidate_threshold AND candidate_energy - null_logit >= null_margin",
  "semantic_gate": {
    "recall_min": 0.7233333,
    "precision_min": 0.0830188679,
    "fp_per_frame_max": 11.125,
    "predictions_per_positive_max": 4.069,
    "hard_violation_max": 0.8666667,
    "multi_positive_recall_min": 0.7894444,
    "inactive_false_acceptance_max_exclusive": 1.0
  },
  "scope": {
    "fit": "L49 V1/V2 split=fit only",
    "dev": "L82 video-disjoint internal fit/dev split",
    "fixed_validation": "L85 internal 16 calibration + 24 validation, in fixed order",
    "full_video_validation": {"v1_videos": ["0004", "0018"], "v2_videos": ["0016", "0017", "0020"]},
    "screening_gt_used": false,
    "official_test_labels_read": false,
    "ordinary_mot_ovmot_touched": false,
    "candidate_deletion": false,
    "candidate_truncation": false,
    "z1_representation_changed": false,
    "groundingdino_lora_used": false,
    "token_span_region_alignment": "UNALIGNED",
    "static_motion_alignment": "UNALIGNED"
  }
}
```

The GT-privileged track-consistent oracle is diagnostic only. It will not alter
model inputs, checkpoint selection, thresholds, candidate boxes, or tracker
IDs. Semantic gate failure does not cancel the authorized full-video HOTA
evaluation; once HOTA output is complete, the branch stops pending supervisor
review.
