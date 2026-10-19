# YesTrack real GPU verification — 2026-10-20

## Scope

This receipt records the first real native OneJev Yes/No training and the
independent held-out probability comparison.  It is a verifier result, not a
claim that the physical MOT/RMOT tracker improved.  The baseline tracker was
not silently changed and no J1 training was restarted.

The corpus is `/data3/user/LocateMOT-runs/YesTrack/native_yesno_v4/train.jsonl`
(SHA-256 `ce0c67625c41c653b366d49b0d03cd35d0c37e2c0896f796374c6acf05b4191e`,
4,516 rows).  It contains 20 train videos and 14 video-disjoint validation
videos, with identity-pair, occlusion-recovery, RMOT and TAO Base OVMOT rows.
The audit receipt is
`/data3/user/LocateMOT-runs/YesTrack/native_yesno_v4/audit.json` and reports
`NATIVE_YESTRACK_CORPUS_AUDIT_PASS`: no split leakage, no MOT20-05, no future
evidence, no GT in the model prompt, and no official TEST labels read.

The training config is
`configs/groundjev/yestrack_native_gpu7_v1.json`.  It used the real
`/data2/user/LocateMOT_GroundJEV/models/OneJev-0.8B` model, native multimodal
inputs, LoRA (`r=16`, `alpha=32`), bfloat16, 100 optimizer steps, and CUDA
device 7.  The run completed 100/100 steps in 1,954.844 seconds; peak allocated
GPU memory was 7.315 GiB.  The final adapter is
`/data3/user/LocateMOT-runs/YesTrack/training/yestrack_native_gpu7_v1/adapter-step100`.
The merged model SHA-256 is
`eb2e7e252600911fd4695c5193129389c3851db1afe7399cb5b9be1aee7b8c1f`.

## Held-out comparison

The same selected validation rows, same native renderer, same media and same
137 valid examples were evaluated for the base model, the completed J1
step-1200 merged checkpoint, and the fine-tuned verifier.  Twenty-three of the
160 requested rows exceeded the declared 16,384-token limit and were dropped
before inference; this rule was identical for all three models.  The reports
and prediction logs are under
`/data3/user/LocateMOT-runs/YesTrack/eval/`.

| verifier | accuracy | balanced accuracy | ROC-AUC | PR-AUC | Brier | ECE | uncertain reject rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OneJev base | 0.6277 | 0.6234 | 0.8031 | 0.8224 | 0.1777 | 0.2842 | 0.3066 |
| J1 step 1200 | 0.5401 | 0.5550 | 0.6694 | 0.7466 | 0.2763 | 0.4648 | 0.1752 |
| YesTrack fine-tuned | **0.7810** | **0.7707** | **0.8880** | **0.8963** | **0.1761** | 0.3374 | **0.0949** |

The fine-tuned verifier improves discrimination and reduces uncertain rejects
relative to both controls.  Its ECE is worse than the base model, so this run
does not support a calibration-improvement claim.  The J1 checkpoint is weaker
than the base model on this verifier task.

## Physical tracking integration status

The repository now contains the opt-in `YesNoVerifier`, selective policy and
the assignment hook.  The default tracker remains verifier-disabled and is
bitwise-compatible with the existing baseline path.  A real four-frame
candidate-aligned replay smoke passed: the disabled arm reproduced all source
outputs exactly, while the enabled arm made six real fine-tuned-model calls
through the same global assignment path.  The receipt is
`YESTRACK_PHYSICAL_REPLAY_SMOKE_20261020.{md,json}`.  A full HOTA/AssA/IDF1/
IDSW comparison was **not** declared from this receipt: the held-out verifier
rows are video-disjoint development rows, while the preserved physical tracker
runs use separate candidate archives and do not contain the full causal
history/media state required to reconstruct a verifier request for every
association.  Reusing those probabilities would be a candidate mismatch.
The next valid integration run must use one complete legal video, the same
GroundAnything candidate archive, causal history, and the same evaluator for
no-verifier, base verifier and fine-tuned verifier arms.

## Reproducibility

The native corpus builder, audit tool, evaluator and GPU-7 config are included
in this branch.  Model weights, media and corpus data remain outside GitHub;
the paths, SHA-256 values, protocol and runtime receipt above are the immutable
references.  This report deliberately separates measured verifier metrics from
the still-pending physical tracking comparison.
