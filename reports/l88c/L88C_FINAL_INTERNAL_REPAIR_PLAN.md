# L88C final-internal repair plan

Date: 2026-09-06  
Thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`  
Base: `364435610f2c42c3a3ae6bdc404a7419ff1f6cf0`  
Branch: `codex/l88c-final-internal-repair-20260906`

This is a zero-training repair of two evaluation-contract gaps. The historical
L88/L88C fixed controls will be split and checked as 16 calibration plus 24
validation units, with the original L88 24-validation JSON as the authoritative
six-metric control. Separately, the already frozen epoch 30 / Rule B selection
(`candidate_threshold=1.0`, `presence_threshold=-1.0`, `null_margin=0.0`) will
be replayed on the complete internal scope: Refer-KITTI V1 videos 0004/0018
(86 sequences) and V2 videos 0016/0017/0020 (537 sequences), followed by the
existing TrackEval wrapper.

No optimizer, backward pass, checkpoint update, LoRA/sidecar update, bank
change, threshold sweep, model selection, screening/official-test label read,
or ordinary MOT/OVMOT operation is authorized. Existing artifacts remain
immutable; each new output uses a fresh attempt directory. The only required
code changes are the fixed-control partition, frozen-selection inference mode,
and scope-aware TrackEval wrapper. The stage stops after the corrected fixed
semantic result and same-scope internal V1/V2 metrics are reported.

Required immutable inputs include the fixed manifest
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`, the
L88 cache, the L88 epoch-30 checkpoint selected by the existing JSON, the
L88C selection JSON, and the historical L88 fixed records. All outputs must
carry `zero_training=true`, `screening_gt_used=false`,
`official_test_labels_read=false`, and `ordinary_mot_ovmot_touched=false`.
