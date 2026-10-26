Warning: truncated output (original token count: 138444)
Total output lines: 4053

## 2026-10-21 — 双序列耦合训练 step48 审计：JEV 学会但 C2 坐标未学会

MOT20-03 与 MOT20-05 各 5 帧、6 个 transition，共恢复到 step48；冻结 G2 step17000，10 帧各一次视觉编码。teacher-forced Grounding loss 从 4.6592 降至 1.1808，两个序列的 typed JEV MATCH/ADMIT/Yes-No/ACTIVE 达到 1.0，但实际 greedy C2 框在两个序列平均 IoU 仍为 0.0000，C0 也为 0.0。部分 C2 输出合法但坐标接近冻结基线，部分出现前缀/解析失败。结论：离散 JEV 监督已可学习，单 token memory 到 G2 坐标生成的优化尚未打通；停止无界加步，转向 coordinate-token 级诊断/目标、wrong-memory 与多身份竞争。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_COUPLED_MULTISEQUENCE_STEP17000_20261021.{json,md}`。

## 2026-10-21 — G2 耦合模块跨序列诊断失败（MOT20-05）

将 MOT20-03 帧 82–85 上训练的耦合 checkpoint 原样用于 MOT20-05 帧 13–17，5 帧各一次 G2 视觉编码，评测期间无参数更新。C0/C2 平均 IoU 都是 0.0000；但 JEV 仍给出近乎满置信的 MATCH/Yes（P(ID) 约 1.0，P(Yes) 约 0.9995）。这定位为单视频记忆/几何过拟合和概率过度自信，否定跨序列泛化主张。保留为负证据，下一步扩大多视频训练并加入错误 memory、多身份竞争和校准。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_COUPLED_CROSS_SEQUENCE_MOT20_05_STEP17000_20261021.{json,md}`。

## 2026-10-21 — 真实 G2 单主干耦合 memory/JEV pilot（step 17000）

在冻结 G2 step17000 主干的条件下完成 24 步真实 GPU 更新，只训练 G2 输入空间记忆 token 投影、JEV 身份槽投影和结构化 typed readout（4,075,271 参数）；5 个 MOT20-03 TRAIN 帧各只做一次 G2 视觉编码。causal Grounding teacher-forcing loss 从 4.6700318654 降至 0.1691639870，JEV Yes/No 从 0/4 到 4/4。训练后 C0 无记忆平均框 IoU 为 0，C2 上一帧记忆平均 IoU 为 0.99636，同视频时序留出帧 86 的 C2 IoU 为 0.99270。该结果证明记忆 token 已影响真实 G2 生成前路径，但只覆盖单视频、单身份、24 步，不能作为独立 MOT/RMOT/OVMOT 泛化或 HOTA 提升；后续必须做多视频、错误 memory 和正式视频级评测。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_COUPLED_MEMORY_JEV_TRAINING_STEP17000_20261021.{json,md}`。

## 2026-10-21 — G2 真实四帧 causal memory replay（step 17000）

假设：单主干 GroundJEV 可以在不加载第二个视觉模型的情况下，将上一帧状态插入下一帧 G2 causal prefill。使用 MOT20-03 TRAIN 帧 82–85 和目标 ID 27，每帧只做一次 G2 视觉编码；第 83–85 帧对比 C0 无记忆与 C2 上一帧记忆。结果：4 次视觉编码、3 个跨帧记忆 prefill 均完成，官方单框结构终止和 parser 均成功；未训练的 `IdentityMemoryTokenProjector` 没有改变任何恢复框，C0/C2 的平均标注 IoU 均为 0.0000。该结果保留为真实耦合边界的负证据，不能表述为 tracking/grounding 提升；下一步需要训练 memory/projector 与结构化 JEV 决策。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_MULTIFRAME_REPLAY_STEP17000_20261021.{json,md}`。

## 2026-10-21 — real G2 causal memory prefill and structured-output repair

A real GPU7 Phase-C comparison loaded the protected G2 Clean adapter at step
17000 and encoded MOT20-03 TRAIN frames 82 and 83 once each. The current-frame
visual embedding was reused for memory-disabled, correct-previous-memory and
shuffled-memory language prefills; each condition used one causal prefill and no
independent VLM. The first attempt exposed a real dtype boundary failure: BF16
embeddings reached a restored FP32 LoRA projection outside the formal G2
autocast context. The smallest fix wrapped only the real G2 prefill in the same
CUDA BF16 autocast boundary. The next attempt showed unconstrained greedy
decoding could run past four coordinates without `<|box_end|>`; the runner now
uses the official single-box structural termination rule, constraining only
coordinate slots and emitting the boundary without synthesizing or reordering
coordinates. The repaired C0/C2 outputs all yielded one complete legal pair;
C0 and both C2 boxes were identical in this untrained-memory control. This is
real prefill/parser evidence and a neutral result, not identity-conditioned
grounding or tracking improvement. Evidence:
`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_MEMORY_PREFILL_STEP17000_20261021.{json,md}`.

## 2026-10-21 — single-backbone native G2 context pilot at adapter step 17000

Following the new GroundJEV single-backbone direction, I ran a bounded native
G2 context readout pilot on physical GPU7 without touching the protected G2
Clean two-rank run. The official GroundAnything DLM was restored with the
step-17000 G2 adapter and encoded 16 MOT17-02-FRCNN TRAIN frames exactly once
per frame (16 native DLM/bridge forwards). Eight frames were used for training
and eight for a same-video temporal development split; the frozen G2 outputs
and teacher-forced previous-frame GT-matched region tokens fed the typed
GroundJEV readout. Eighty optimizer updates reduced identity accuracy from
0.052419 to 0.750000 on the train prefix and from 0.032258 to 0.536290 on the
temporal development split; admit accuracy changed from 0.653225 to 0.927419
and 0.596774 to 0.915322. These are bounded learnability controls only: no
HOTA/IDF1/IDSW, independent-video, or learned identity-conditioned box claim
is made, and teacher-forced causal memory remains a limitation. Evidence:
`docs/GROUNDJEV_COUPLED/PHASE_C_NATIVE_CONTEXT_PILOT_STEP17000_20261021.{json,md}`.

## 2026-10-21 — G2 Clean checkpoint step 15000 audited on CPU

The protected exact-once two-rank G2 run wrote and successfully reloaded `adapter-step015000.pt` and `latest_training_state.pt` on CPU with explicit `weights_only=False`. Both report step 15,000, world size 2, the registered Clean TRAIN manifest SHA, optimizer/scheduler/RNG state and the strict G2 format. The latest event is step 15002 with finite loss, gradient norm and memory; `final_receipt.json` is absent, so TAO/G1/held-out/three-task OneJev remain gated. This is checkpoint/recovery evidence only, not a metric claim. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP15000.{json,md}`.

## 2026-10-21 — G2 Clean checkpoint step 14500 audited on CPU

The protected exact-once two-rank G2 run wrote and successfully reloaded `adapter-step014500.pt` and `latest_training_state.pt` on CPU with explicit `weights_only=False`. Both report step 14,500, world size 2, the registered Clean TRAIN manifest SHA, optimizer/scheduler/RNG state and the strict G2 format. The latest event is step 14512 with finite loss, gradient norm and memory; `final_receipt.json` is absent, so TAO/G1/held-out/three-task OneJev remain gated. This is checkpoint/recovery evidence only, not a metric claim. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP14500.{json,md}`.

## 2026-10-21 — post-G2 downstream preflight

A read-only launch audit verified the registered TAO TRAIN eight-shard scope, runtime GroundAnything exporter/streamer, native OVMOT joiner, native three-task OneJev owner, G2 bundle owner, and G1 GPU5 owner. All required paths were present, the registered entrypoints compiled, and the wait owners were live. The formal G2 terminal receipt is still absent, so no TAO shard, optimizer, model service, or GPU was started. Evidence: `docs/GROUNDJEV/POST_G2_DOWNSTREAM_PREFLIGHT_20261021.{json,md}`.

## 2026-10-21 — G2 Clean live progress at step 14331

The protected two-rank G2 Clean run is still active on physical GPUs 3/5 and has a contiguous event stream through `14331/18,227`. The latest event at `2026-10-21 10:39:32 +0800` has finite loss `0.165262759198`, gradient norm `9.687554359436`, and peak event memory `12.828872 GiB`. The durable recovery pair remains `adapter-step014000.pt` and `latest_training_state.pt`; `final_receipt.json` and `adapter_projector_final.pt` are absent. Native TAO/OVMOT joining, three-task OneJev optimization, G1, and YesTrack still report zero optimizer updates under their dependency/resource gates. This is training-health evidence only and makes no coupled or tracking metric claim. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP14331.{json,md}`.

## 2026-10-21 — YesTrack permission failure repaired; coupled optimizer still gated

The first real YesTrack GPU7 owner reached its resource gate but failed before
creating the training process with `PermissionError(13, 'Permission denied')`.
The actionable cause was direct execution of the mode-0666 source wrapper
`run_with_resources.py`; the queue now invokes that wrapper through the pinned
OneJev Python environment. Compilation and the queue entrypoint check pass. The
terminal failed queue is preserved, and a new retry owner is waiting under a
separate queue root for GPU7 exclusivity because an unrelated `szgx` process
currently occupies GPU7. No process was stopped and no YesTrack metric is
claimed. The real G2/OneJev three-task coupled optimizer remains at zero updates
until the protected G2 Clean terminal receipt exists. Evidence:
`docs/GROUNDJEV/YESTRACK_PERMISSION_ROOT_CAUSE_20261021.{md,json}`.

## 2026-10-21 — G2 step 16977 snapshot and YesTrack owner recovery

The protected G2 Clean trainer remains live at step 16,977/18,227 on GPUs 3/5
with finite loss and gradient norm; its terminal receipt and terminal adapter are
still absent. The native TAO eight-shard owner, CPU OVMOT joiner, and native
three-task OneJev owner remain live under their explicit gates, with zero joint
optimizer updates. The GPU5 G1 owner is also waiting for the G2 terminal receipt.
The repaired YesTrack retry owner had exited without a failure receipt, so it was
resumed in a detached session and is now live at the GPU7 exclusivity gate. The
unrelated GPU7 compute process remains untouched. Evidence:
`docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_STEP16977.{md,json}`.

## 2026-10-21 — YesTrack entered real GPU training after permission repair

The detached retry owner passed the GPU7 exclusivity gate and entered real
native YesTrack LoRA training using the OneJev step-600 checkpoint. The 726-row
train and 390-row validation manifests match their registered SHA-256 values;
the run is scheduled for 273 optimizer steps and had reached step 10 with finite
loss, slot accuracy and memory. This is training-start evidence only. The
terminal adapter, frozen-base/fine-tuned validation and YesTrack metrics are
still pending, and the formal three-task G2/OneJev optimizer remains at zero
updates. Evidence:
`docs/GROUNDJEV/YESTRACK_TRAINING_STARTED_20261021_STEP10.{md,json}`.

## 2026-10-21 — YesTrack paused to restore coupling priority

The latest direction is the GroundJEV single-backbone coupling experiment, so
the auxiliary YesTrack verifier was stopped cleanly after real GPU training
step 20/273 (160/2178 examples). Its partial log and queue state are preserved;
no adapter, validation receipt or metric was promoted. GPU7 is kept available
for the independent held-out gate after G2 terminal completion. G2, G0/G200/G1
artifacts and the native three-task queue were not modified. Evidence:
`docs/GROUNDJEV/YESTRACK_INTERRUPTED_FOR_COUPLING_PRIORITY_20261021.{md,json}`.

## 2026-10-21 — G2 Clean step 17000 checkpoint audited on CPU

The protected exact-once G2 run wrote adapter-step017000 and the full latest
recovery state. Both load on CPU with explicit `weights_only=False`, report
world size 2, the registered Clean TRAIN manifest SHA, strict training contract,
optimizer/scheduler/sampler state and all Python/NumPy/Torch/CUDA RNG state.
The live event stream is at step 17003/18227; the terminal receipt and terminal
adapter remain absent, so G1, TAO, native OVMOT, joint OneJev and held-out gates
remain closed. This is recovery evidence only. Evidence:
`docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP17000.{md,json}`.

# LocateMOT Research Log

Concise experimental log (latest first).

## 2026-10-21 — dense-answer G2/JEV coupling completed and official decoder checked

在冻结 G2 step17000 上新增合法 dense-answer 耦合训练：DanceTrack-0001/0016 各 3 帧，每个 row 保留该帧全部 person boxes（7/6 个），训练 2 个 transition、2 optimizer steps；只训练 `IdentityBoxStateTokenProjector`、JEV memory projector 和 typed readout，共 4,077,327 参数，G2 视觉每帧一次。Grounding teacher-forcing loss 3.83649→3.83393，JEV structured loss 4.11511→1.14514。随后接入官方 `DLMInferenceEngine`：0001 的训练转移 recall@0.5 从 C0=0.1429 到 C2=0.2857（1→2 对比），但 0001 development 2→3 及 0016 两个转移 C2 输出均 malformed/0；因此这是共享主干接口和小样本可学习性证据，不是跟踪增益或泛化结论。MOT20 高密度 dense teacher forcing 曾在 GPU7 约 63.8 GiB 达到 OOM，已保留失败边界并改用低密度合法 clip 验证，未改 target-only 合同。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_DENSE_COUPLING_DANCE_STEP17000_20261021.md`，两个 receipt 位于 `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_g2_dense_coupling_dance_step17000_v1/receipt.json` 与 `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_g2_official_dlm_dense_coupled_dance_step17000_v1/receipt.json`。

## 2026-10-21 — official G2 DLM dense contract confirms one-box baseline before coupling

用冻结 G2 step17000 在 GPU4 通过 GroundAnything 官方 `DLMInferenceEngine` 的 `hierarchy_dynamic` 路径重跑真实 dense 解码；结构化终止开启且 `structured_max_coordinate_groups=0`，没有强制 single-box。MOT20-03/82（96 个 TRAIN person boxes）和 MOT20-05/13（247 个）各编码一次，官方 parser 均恢复 1 个合法框，但 IoU0.5 召回均为 0。模型在第一个 `<|box_end|>` 结束，说明先前 causal 单框现象也存在于官方混合 DLM 路径；这不是耦合收益，也不是 tracking 指标。下一步把持久身份状态插入同一官方 DLM prompt/clean stream，并在完整 GroundAnything answer 合同下训练和评测。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_OFFICIAL_DLM_DENSE_STEP17000_20261021.md`，receipt 位于 `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_g2_official_dlm_dense_step17000_v1/receipt.json`。




## 2026-10-21 — G2 Clean checkpoint step 14000 audited on CPU

The protected exact-once two-rank G2 run wrote `adapter-step014000.pt` (SHA-256 `0cbf160b57a4feccb11a77f602cd8ebd39e79d93d79453cb78316a1d3ef6371e`) and `latest_training_state.pt` (SHA-256 `d7bbdfdecfeba789ff022d31800e1cd7d1fedb465e34e325ed738f33f66bae78`). Both load on CPU with explicit `weights_only=False`, report step 14,000, world size 2, exact-once contract, registered manifest SHA, optimizer/scheduler/RNG state and seed 20261019. The event stream is at step 14,021 with finite loss/gradient/memory; final receipt and terminal adapter remain absent, so dependent queues stay gated. This is checkpoint/recovery and training-health evidence only. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP14000.{json,md}`.

## 2026-10-21 — G2 Clean checkpoint step 13500 audited on CPU

The protected exact-once two-rank G2 run reached step `13,500/18,227` and wrote `adapter-step013500.pt` (SHA-256 `84a7e84a391d375cfeb3c74dea59bdb36f9a608fdf523132f0c6c9f6e1273c70`) plus `latest_training_state.pt` (SHA-256 `3a3a9a51d6f1e6b8f3e4a7d1ad399bd4864805c50dd0eab2eeb19c6def85626c`). Both were loaded successfully on CPU with explicit trusted-checkpoint `weights_only=False`; the initial PyTorch 2.6 default `weights_only=True` refusal was a validation-tool issue, not a checkpoint failure. Step, world size, exact-once contract and Clean TRAIN manifest SHA match. The final receipt and terminal adapter remain absent, so downstream gates stay closed. This is checkpoint/recovery evidence only, not a metric claim. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP13500.{json,md}`.

## 2026-10-21 — G2 Clean live progress at step 13361

The protected two-rank G2 Clean event stream advanced to `13,361/18,227` at `2026-10-21 07:35:07 +0800`. The latest loss (`0.2484929720`), gradient norm (`11.54479408`) and peak event memory (`12.828843 GiB`) are finite; the known parent PID remains live on GPUs 3/5. `final_receipt.json` and `adapter_projector_final.pt` are still absent, so G1, native TAO/OVMOT joining, three-task OneJev optimization and independent held-out evaluation remain gated. This is training-health evidence only and makes no G2 metric claim. Evidence: `docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP13361.{json,md}`.

## 2026-10-21 — G2 event-integrity audit through step 12650

- Read the formal event log without altering the trainer. The completed audit
  cutoff contained 12,650 unique, contiguous steps from 1 through 12,650,
  with zero missing IDs and zero non-finite values across the recorded loss,
  gradient, memory and learning-rate fields.
- Twenty-six explicit checkpoint events were present through the durable
  step-12500 checkpoint; `final_receipt.json` remained absent. The live log
  then advanced to step 12651 with finite loss/gradient values.
- Decision: preserve the current two-rank run and its recovery state. This is
  numeric/resumability evidence, not a performance claim.
- Evidence: `docs/GROUNDJEV/G2_EVENT_INTEGRITY_AUDIT_20261021_STEP12650.md`.

## 2026-10-21 — downstream launch-gate audit

- Read the five G2-dependent entrypoints and confirmed their exact terminal
  contract: status `G2_EXACT_COVERAGE_TRAINING_COMPLETE`, step `18_227`, and
  clean manifest SHA `f682519a…06e71`; the bundle exporter also requires the
  terminal adapter file.
- Confirmed YesTrack is intentionally independent of G2 training and remains
  gated by real GPU7 exclusivity and its native split, so it cannot silently
  fall back to a CPU mock.
- No gate defect or premature launch was found. The static audit changed no
  runtime state and no benchmark claim. Evidence:
  `docs/GROUNDJEV/DOWNSTREAM_GATE_AUDIT_20261021.md`.

## 2026-10-21 — GroundJEV G2 step-12642 progress snapshot

- Evidence: the formal event log reached step `12642/18227` at
  `2026-10-21 05:17:02 +08:00`, with finite loss `0.17345873918384314` and
  finite gradient norm `10.642035484313965`.
- Gate state: `final_receipt.json` is absent. G1, deployment bundle,
  independent held-out, native TAO and YesTrack owners remain alive in their
  registered waiting states; no downstream metric or optimizer update is
  claimed.
- Decision: continue protecting the two-rank G2 run on physical GPUs 3/5 and
  preserve all checkpoints and historical G2-PL outputs.
- Boundary: this is runtime evidence, not a MOT/RMOT/OVMOT performance result.
  See `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12642.md`.

## 2026-10-21 — GroundJEV G2 step-12633 progress snapshot

- Evidence: the formal event log advanced to step `12633/18227` at
  `2026-10-21 05:15:19 +08:00`, with finite loss `0.28367656865157187`, finite
  gradient norm `10.239775657653809`, and peak memory `12.8289 GiB`.
- Gate state: `final_receipt.json` is still absent. G1, deployment bundle,
  independent held-out, native TAO and YesTrack owners remain alive but in
  their registered waiting states; no downstream optimizer update or metric is
  claimed.
- Decision: continue protecting the two-rank G2 run on physical GPUs 3/5 and
  preserve all checkpoints and historical G2-PL results.
- Boundary: this is runtime evidence, not a MOT/RMOT/OVMOT performance result.
  See `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12633.md`.

## 2026-10-21 — GroundJEV G2 step-12613 progress snapshot

- Hypothesis/contract: preserve the formal two-rank G2 Clean run and open all
  dependent G1, deployment-bundle, held-out, native-TAO and YesTrack gates only
  after the exact terminal receipt.
- Evidence: the authoritative event log reached step `12613/18227` with
  finite loss `0.2061017351`; `adapter-step012500.pt` and
  `latest_training_state.pt` are present, while `final_receipt.json` is absent.
- Decision: retain the live run and all waiting owners. No downstream optimizer,
  evaluation, label promotion, or historical G2-PL overwrite was performed.
- Boundary: this is a runtime progress record, not a MOT/RMOT/OVMOT metric or
  a claim that the final G2 detector exists. See
  `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12613.md`.

## 2026-10-01 — latest R1 restoration completed

- Hypothesis: restore only the newest R1 execution boundary and its legal
  public-data/runtime inputs on `/data2/user/LocateMOT`; do not pretend that
  unavailable L69/L89E/R1 payloads were reproduced.
- Restored: official KITTI training images, Refer-KITTI V1, Refer-KITTI V2,
  MMDetection 3.3.0/GroundingDINO, BERT, and the direct R1 source dependency
  modules. V1 audit is complete. V2 has 157 legal expression-frame target
  references (11 repeated physical video/frame/ID pairs) pointing to IDs
  absent from corrected labels; missing frames are zero, and the anomaly is
  retained for the next adapter instead of silently filtered. The new public
  query manifest records every affected row.
- Runtime evidence: GroundingDINO loads the SHA-verified official checkpoint
  with 172,977,693 frozen parameters; one real KITTI frame/expression GPU
  inference returned 300 finite candidate boxes and finite scores. R1 imports
  and compile checks pass after path migration. The safe source loader also
  reads all 8,397 legal V1/V2 expression records from the restored
  per-expression layout without opening official-evaluation videos.
- Failure/limit: historical R1 checkpoints, L69 bank, dense indexes,
  language/visual/aligned caches and fixed manifest are unavailable. No old
  HOTA or score is claimed as reproduced; the old R1 sidecar is not extended
  with threshold/layer/learning-rate sweeps.
- Next action: construct a new legal public-data manifest/index and then a
  freshly generated candidate bank before any new correspondence training.
  The manifest is now complete. A 40-row midpoint-frame GroundingDINO probe
  found mean best IoU `.3188`, IoU@.25 `.6585`, IoU@.50 `.1463`, and mean best
  score `.0107` over 123 target boxes; the frame-zero attempt is preserved as
  a separate artifact. This supports a new bank audit but is not an RMOT
  score. Official-evaluation videos remain unread.

## 2026-08-19 Stage L11 — OVMOT temporal pseudo-track + KITTI front-end

- Hypothesis: unmatched DLA detections on expanded TAO-train OVMOT
  stream must receive high-precision temporal pseudo-track supervision
  instead of "every detection is NEW"; KITTI RMOT DetPr ~1% is a
  candidate front-end problem, fixable with category whitelist + NMS +
  query-conditioned CLIP top-k.
- Implementation:
  - Pseudo-track generator (forward + backward cycle consistency,
    appearance/motion/category filters, confidence gating) for all 500
    TAO train videos; sidecars in `outputs/l11/data/pseudo_tracks`.
  - Class-A GT upgraded from C-TAO base @0.5 IoU to base_and_novel
    @0.3 IoU (~10% -> ~28% coverage).
  - Trainer: no_unmatched_new, confidence-weighted identity/relevance
    losses, pseudo ids in the same slot/lifecycle machinery.
  - KITTI: whitelist + cross-NMS + top-30 -> ~15.3 cands/frame (was 50);
    CLIP top-12 calibration: query precision 10.2%, target recall 65%.
- Quality: pseudo same-ID precision 99.26% (8-video audit, target 90%);
  cycle pass 0.997; NEW rate 0.110 vs L10 collapse ~1.0.
- Running: 4-GPU repair training (from L9-ovmot, ~6-8 s/step),
  repaired-KITTI eval (L9-ovmot ckpt), 10-video OVMOT over-birth
  baseline: unique/rows 0.211 (L9), reused IDs 467.
- Pending: step-7000 OVMOT quick eval, full TAO TETA, KITTI TrackEval,
  L12 prompt seeding.

## 2026-08-20 continued

- Low-LR continuation (10000->11000, LR~6e-6): RMOT-Dance 32.49 ->
  34.05 HOTA; ordinary Macro 0.4855 -> 0.4924.  OVMOT full TETA on
  s11k running.
- Fresh-optimizer balance run (p_rmot 0.45/p_ovmot 0.20) REJECTED at
  step 1000 (RMOT 32.06 / ordinary 0.4663).
- L12 frozen prompt-seeded (DAVIS 2017, 10 multi-object videos):
  mask/point > box for identity robustness; joint fine-tune not
  launched (mixed signal).  Infrastructure complete.

## 2026-09-04 Stage L88 — RMOT-aware GroundingDINO LoRA adaptation

- Hypothesis: a zero-initialized rank-16/alpha-32 LoRA on the final two
  GroundingDINO fusion layers and decoder layer 0, with the frozen L86/L87
  sidecar and causal history, would improve held-out target-bag and
  expression-to-candidate correspondence without changing the L69 bank.
- Contract: label-free, distributed, and temporal regressions passed; full
  V1/V2 fit used seed `20260829`, 40 epochs, 2,640 optimizer steps, world 4,
  effective group batch 8, and 20 even checkpoints. Local MMCV required FP32
  rather than the registered BF16 autocast. Candidate deletion/truncation was
  false and all frozen-base/LoRA gradient and reload checks passed.
- Dev: all 20 checkpoints were scored on 138 video-disjoint groups; the fixed
  selection chose epoch20/Rule B before fixed validation. Checkpoint SHA is
  `7012706140cfa94278ce15bb8da3e2318eb3efcfb8d26d3b1dbc5206f1145538`.
- Fixed semantic validation: immutable L29 was
  `recall=.7333333, precision=.0830189, FP/frame=10.125,
  pred/positive=8.8333, hard=.9166667, multi=.8194444`. L88 candidate-only
  was `recall=.4193548, precision=.1830986, FP/frame=2.4167,
  pred/positive=2.2903, hard=.8461538, multi=.3333333`; final Rule B was
  `recall=.1935484, precision=.1363636, FP/frame=1.5833,
  pred/positive=1.4194, hard=.8461538, multi=.1666667`, with empty rate
  `.5416667` and inactive false acceptance `.5`. The semantic gate failed on
  recall and multi-positive preservation; the lower volume is not a fix.
- Internal TrackEval: selected Rule B produced HOTA `26.0914` (V1) and
  `20.2386` (V2), below L86 `29.1663/21.6467` and L87-A
  `28.5752/22.1300`. No screening or official-test labels were read.
- Preserved failures/fixes: first formal causal-key drift, sparse seqinfo,
  merge, device, and score-shape retries remain in new directories with their
  provenance; no old bank/checkpoint/production entrypoint was overwritten.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`; no L88 long continuation,
  screening, official test, TrackEval beyond internal validation, or
  ordinary MOT/OVMOT change is authorized. Unique next action is supervisor
  review of one new RMOT correspondence/proposal design.

## 2026-09-05 Stage L88C — corrected candidate-vs-NULL replay

- Hypothesis: the prior L88 semantic failure may include an implementation
  error in the candidate-vs-NULL gate. Corrected emission is
  `candidate_energy >= candidate_threshold` AND
  `candidate_energy-null_logit >= null_margin` AND
  `presence_logit >= presence_threshold`; candidate-only uses only the first
  condition. No training, backward, optimizer, checkpoint update, bank
  change, top-k/NMS or NULL suppression was performed.
- Implementation: refit all 20 even checkpoints on the registered B/R/P
  grids, replayed the corrected shortlist (epochs 8/20/40/30/4) over complete
  V1/V2 dev video, ran the corrected internal TrackEval matrix, froze epoch
  30 Rule B before fixed validation, and ran the fixed 16-calibration/
  24-validation replay. A first TrackEval provenance alias failure was
  preserved; the minimal `scope_key/full_video` alias fix was pushed in
  `c5a6685` and the corrected matrix is attempt2.
- Result: L88C corrected final on validation was
  `recall=.354839, precision=.207547, FP/frame=1.75,
  pred/positive=1.7097, hard=.846154, multi=.305556,
  empty=.375, inactive-FA=.666667`. Immutable L29 remains
  `.733333/.083019/10.125/8.8333/.916667/.819444` for
  recall/precision/FP/pred-positive/hard/multi. Candidate-only recall was
  `.419355`; the corrected NULL comparison removed additional positives.
- Internal dev TrackEval for the frozen epoch30/Rule B was V1
  HOTA/DetA/AssA `.296615/.216472/.408302` and V2
  `.265682/.156622/.451938`; these are dev-only, not screening or official
  results. The semantic gate failed on recall and multi-positive recall.
- Root cause: existing candidate scores have useful average pairwise ranking,
  but the lowest-scoring positives in multi-target bags do not survive the
  fixed emission boundary; V2 is worse. This is a correspondence and
  multi-positive recall failure, not a universal NULL-acceptance,
  coverage, finite/reload, or implementation-smoke failure.
- Preserved outputs: corrected replay, two TrackEval attempts, frozen
  selection, fixed semantic JSON/gate, and offline A–I diagnosis under
  `outputs/l88c/`. Code was pushed to branch
  `codex/l88c-candidate-null-corrected-replay-20260905` at `e8f5f62`.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`. No screening/official-test
  labels, HOTA claim beyond internal dev TrackEval, ordinary MOT/OVMOT/TAO
  change, or new L88C training was run. The only next action is the single
  supervisor approval request in `reports/l88c/NEXT_TEST_APPROVAL_REQUEST.md`.

## 2026-09-06 — L88C recoverable storage cleanup

- While the authorized zero-training L88C final-internal replay was running,
  its open paths were checked. Three superseded L1 intermediate LoRA
  checkpoint directories (`outputs/l1_c/checkpoints/lora/checkpoint-100/200/300`,
  about 8.0G each) were not open by any process and were moved, not deleted,
  to `/data2/usr_for_deadline/locatemot_cleanup_20260906/`.
- Frozen banks, formal reports, current L88 epoch30 checkpoint/cache, and all
  ordinary MOT/OVMOT assets were left in place. `/data1` free space increased
  from about 76G to 114G. The recoverable inventory is documented in
  `reports/l88c/L88C_RESOURCE_CLEANUP_20260906.md`.

## 2026-09-07 Stage L89 — query-conditioned set correspondence decoder

- Hypothesis: a fresh QSC-D head over the frozen L84/L87-A Z1 candidate states,
  with candidate self-attention followed by cross-attention to pure frozen
  GroundingDINO/BERT language tokens, could improve same-frame correspondence
  without changing the bank, detector, tracker, or ordinary MOT/OVMOT.
- Implementation: isolated worktree at
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89`, base
  `ece9fc5549a9210424eb127fb10430c3fed7b7ba`; two set layers, eight heads,
  FFN 1024, hidden 256, history 8, 4,169,829 trainable parameters. Pure
  language cache has 3,418 label-free entries. Contract smoke passed after
  preserving two package/field failures. No token/span or static/motion
  supervision is verified (`UNALIGNED`).
- Fit: seed `20260829`, one GPU, BF16, 40 epochs/2,640 optimizer steps,
  finite loss and nonzero gradients, all 20 even checkpoints plus final.
  Only L49 fit supervision was used; no screening or official-test labels.
- Dev/selection: 20 checkpoints scored over 138 legal dev groups; five-item
  shortlist and full-video B/R/P matrix selected epoch 4, Rule R before fixed
  validation. Checkpoint SHA is
  `5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`.
- Fixed semantic: authoritative retry4 failed. L29 was
  `recall=.7333333, precision=.0830189, FP/frame=10.125,
  pred/positive=8.8333, hard=.9166667, multi=.8194444`; L89 was
  `recall=.8709677, precision=.0794118, FP/frame=13.0417,
  pred/positive=10.9677, hard=.6923077, multi=.9027778`, with inactive
  false acceptance `1.0`. Hard-negative ranking moved, but deployable volume,
  precision, and no-match behavior failed.
- Internal TrackEval: frozen epoch4/Rule R on V1 validation (86 sequences)
  gave HOTA `2.2906`, V2 validation (537 sequences) HOTA `0.5894`, versus
  L87-A `28.5752/22.1300`. Full-video post-hoc target-row recall was only
  `.0131/.0026` and inactive acceptance was `1.0`. This is internal
  validation TrackEval evidence, not screening or official-test evidence.
- Root cause judgment: query-conditioned set representation gave limited
  frame-level ranking signal but failed calibrated persistent emission and
  absence handling. The branch is stopped pending supervisor review; no L89
  continuation, screening, official test, or ordinary MOT/OVMOT change is
  authorized. All L89 outputs are preserved under `outputs/l89/` and the new
  source/reports are on branch `codex/l89-query-conditioned-set-correspondence-20260907`.

## 2026-09-08 Stage L89C — corrected candidate-vs-NULL replay

- Hypothesis/repair: L89 training and the registered loss use `candidate_energy`
  against `null_logit`, but the old L89 evaluation/deployment path emitted
  using `presence_logit - null_logit`. L89C changed only the evaluation,
  selection, inference, and TrackEval contract to the shared
  `corrected_emission_mask`: candidate threshold, candidate-minus-NULL margin,
  and presence threshold. No training, scorer forward, bank/cache rebuild, or
  model/ordinary-MOT change was made.
- Inputs/selection: immutable L89 raw dev records (9,960; SHA recorded in
  `outputs/l89c/dev/corrected_shortlist_attempt1/`), 20 existing checkpoints,
  and the fixed manifest SHA
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
  Corrected dev TrackEval evaluated 5 checkpoints × B/R/P (15 results), then
  froze epoch 2 / Rule R, checkpoint SHA
  `f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b`, with
  candidate/presence/NULL thresholds `-0.75/-1.0/0.0` before fixed validation.
- Fixed semantic evidence: authoritative retry2 has 40 ordered records (16
  calibration, 24 validation), with preselection forbidden-label fields
  absent. Validation is recall `.8387097`, precision `.0869565`, FP/frame
  `11.375`, predictions/positive `9.6452`, hard violation `.6923077`,
  multi-positive recall `.8333333`, inactive false acceptance `1.0`.
  The gate is `semantic_gate_fail`: hard/recall/precision/multi floors pass,
  but volume and inactive/no-match conditions fail.
- Internal evidence: frozen epoch-2/Rule-R full-video inference covered V1
  `0004,0018` (86 sequences) and V2 `0016,0017,0020` (537 sequences).
  Corrected TrackEval HOTA is V1 `2.244718%`, V2 `0.583059%`; no screening or
  official-test labels were read. These are internal validation-scope metrics,
  not a final RMOT result. Historical buggy L89 was `2.2906%/0.5894%` and
  L87-A was `28.5752%/22.1300%` on V1/V2.
- First actionable root cause: corrected candidate-vs-NULL accounting still
  leaves no-match acceptance universal and output volume too high. All rows,
  keys, finite checks, and frozen boundaries passed; the initial GPU OOM is
  preserved as an implementation/resource attempt and the exact CPU retry
  completed. Unique next action: supervisor review and authorization of one
  new RMOT branch explicitly addressing absence/volume calibration with
  candidate correspondence. Do not extend L89/L89C or tune thresholds.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`; `zero_training=true`,
  `screening_gt_used=false`, `official_test_labels_read=false`,
  `ordinary_mot_ovmot_touched=false`.

## 2026-09-08 Stage L89D — true full-video timeline repair

- Hypothesis/scope: repair only the L89C sparse-prediction versus dense-GT
  timeline mismatch. No training, model, loss, bank, tracker, threshold, or
  ordinary MOT/OVMOT change was made. New code is confined to `tools/l89d_*`
  and new evidence is under `reports/l89d/` and `outputs/l89d/`.
- Contract repair: native L69 `frame_ids` and frame pointers now drive every
  video timeline; all query/frame pairs and all candidate rows are scored.
  The multi-video query-map overwrite was fixed and covered by a targeted
  regression. Dev inference attempt3 covers 1,152,690/1,152,690 candidate
  pairs across 2,385 frame groups; internal attempt1 covers
  243,550/243,550 pairs across 1,844 frame groups. Both passed full-video and
  pair-completeness checks. The dev and internal dense Z1 supplements were
  built from frozen query-independent caches and finalized with no labels.
- Dev evidence: the valid 5-checkpoint × B/R/P TrackEval matrix selected
  epoch 8 / Rule P on the preregistered dev tuple. The frozen checkpoint is
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch008.pt`,
  SHA256 `28d4c35a189474d446a26fc096bbafbad77e1298429674383ce7ce8b14677d99`.
  The fixed 16-calibration/24-validation semantic replay then failed: recall
  `.5483871`, precision `.1517857`, FP/frame `3.9583333`,
  predictions/positive `3.6129032`, hard violation `.6923077`,
  multi-positive recall `.4861111`, inactive false acceptance `.6666667`.
  Recall and multi-positive floors failed; no screening/test labels were read.
- Internal evidence: fixed epoch 8 / Rule P true full-video TrackEval gives
  combined HOTA `23.8586078862%`, DetA `15.1294290119%`, AssA
  `37.9461647913%`, DetRe `57.2559333985%`, DetPr `16.9954368707%`, IDF1
  `18.6931722536%`, and IDSW `6337.5`; V1 HOTA `25.0798607896%`, V2 HOTA
  `22.6373549828%`. These are internal validation-scope TrackEval results,
  not screening, official-test, or final ordinary-RMOT evidence.
- Attempts preserved: dev ENOSPC, internal exit-137/OOM retries, the expected
  smoke full-video rejection, the selector directory-argument error, and the
  earlier multi-video GT-map mismatch. The final boundary guard passed after
  the path contract correction. The fixed manifest SHA remained
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
- Status: `L89D_COMPLETE_ZERO_TRAINING_TRUE_FULLVIDEO_REPAIR /
  semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`.
  Flags: `zero_training=true`, `screening_gt_used=false`,
  `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
  Unique next action: supervisor review and authorization of one separate
  absence/volume-calibration study; do not extend L89/L89C or alter the
  production MOT/OVMOT paths.

## 2026-09-09 Stage L89E — phase-consistent temporal/history replay

- Hypothesis/scope: L89D evaluated every checkpoint with temporal history
  enabled even though L89 training used Stage-S epochs 1–8 with temporal off
  and zero history. L89E changed only this replay contract: S epochs 2/4/6/8
  use `temporal_enabled=false` and zero history; T epochs 10–20 and J epochs
  22–40 use causal last-four history. No training, model, loss, bank, tracker,
  threshold science, UIDM, ordinary MOT or OVMOT source was changed.
- Implementation: isolated branch/worktree based on L89D commit
  `d0960e1d1679414765f79203bef444f0f9968928`; phase policy and wrappers are
  in `tools/l89e_*`. The required CPU equality check passed for S and T
  `L86ClipStore` history construction. The first replay failure was a
  `FrameExample`/`_clip_history` interface mismatch; the minimal adapter fix
  passed a targeted S/T causality regression. A subsequent historical-score
  merge check found that the immutable sparse file contains S+T/J rows; the
  wrapper now filters only the historical T/J epochs and preserves their
  values. Launch-path and shortlist argument mistakes were preserved as
  `INCOMPLETE`/operator evidence; no affected model run was treated as data.
- Stage-S evidence: fresh phase-correct replay of epochs 2/4/6/8 completed
  `4 x 498 = 1,992` records, all finite, complete and zero-history. Merged
  phase-consistent fit/dev pool contains 20 epochs and 9,960 records. The
  registered maximum-five fit/dev shortlist was frozen at epochs
  `8(S),20(T),40(J),32(J),4(S)`.
- Dev selection: full native dev replay retained `11,003,527/11,003,527`
  candidate rows for all five shortlist candidates. The 15-result TrackEval
  matrix selected epoch 4 / phase S / Rule B by the registered tuple; dev
  HOTA was `0.296512` (internal fit/dev selection evidence only).
- Fixed semantic evidence: the new 16-calibration/24-validation replay used
  the frozen epoch-4/S zero-history policy and Rule B. Immutable L29 remains
  recall/precision/FP-frame/pred-positive/hard/multi
  `.7333333/.0830189/10.125/8.8333/.9166667/.8194444`. L89E validation is
  recall/precision/FP-frame/pred-positive/hard/multi
  `.4516129/.1458333/3.4166667/3.0967742/.7692308/.3611111`, with inactive
  false acceptance `.8333333` and empty rate `.2083333`. The semantic gate
  fails recall and multi-positive floors despite lower volume and higher
  precision; no threshold rescue was performed.
- Internal TrackEval: after the frozen selection, exact internal timeline
  replay covered `243,550/243,550` pairs for V1 0004/0018 and V2 0016/0017/0020.
  Phase-consistent HOTA/DetA/AssA/DetRe/DetPr were V1
  `28.7628/18.7680/44.3927/80.3045/19.5312%` and V2
  `21.8385/13.3358/35.9990/45.1160/15.8179%`; IDSW was 1,997 and 9,915.
  These are internal validation-scope TrackEval results, not screening or
  official-test results.
- Root cause: correcting S-phase temporal/history evaluation does not
  restore expression correspondence; the fixed emission remains recall- and
  multi-positive-limited, with substantial inactive acceptance. Status is
  `L89E_COMPLETE_ZERO_TRAINING_PHASE_CONSISTENT_REPLAY /
  semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`.
- Preserved outputs: Stage-S attempts 1–4, merge attempts 1–2, shortlist
  attempts 1–4, dev/internal full-video and TrackEval outputs, fixed semantic
  output, and all reports under `reports/l89e/` and `outputs/l89e/`. The fixed
  manifest SHA remains
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
  Flags remain `zero_training=true`, `screening_gt_used=false`,
  `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
- The only next action is supervisor review of a separately authorized
  absence/volume-calibration study. Do not extend L89/L89C/L89E, alter the
  bank, or touch production MOT/OVMOT.

## R0A — safe target contract, dense native indexes, and implementation smoke — 2026-09-09

- Hypothesis: isolate only allowlisted V1/V2 target records, repair the
  frame-specific target contract, and construct native-frame R0 data without
  parsing forbidden official-eval payloads.
- Safe-source retry3 passed: lexical monolithic-JSON isolation deserialized
  only allowlisted video values and recorded
  `forbidden_payload_deserialized=false` and
  `official_test_labels_read=false`.  The exact 5,314 sparse fit rows matched
  the authoritative frame map with zero sentence/target/missing-query/invalid-
  frame mismatches.
- Dense native-frame index retry2 and its contract audit passed for V1/V2;
  all L69 rows and duplicate candidate indices were retained.  The geometry
  `image_hw` and visual-cache dataset/group-key contracts were corrected in
  R0-only files.  Retry1 dense/smoke failures remain preserved with their
  first trace; retry2 is authoritative.
- Real R0 contract smoke retry2 passed on one V1 fit frame: loss
  `3.555824041366577`, 108/108 finite nonzero trainable gradients, frozen
  GroundingDINO, no tracker optimizer parameters, strict reload max diff `0`,
  and temporary visual item removed.  This is implementation evidence only;
  no semantic, screening, official-test, HOTA, or production MOT/OVMOT result.
- Next action: commit/freeze the R0A safe code and construct the registered
  query-independent visual cache on `/data2`; no old asset or production
  entrypoint is modified.

## R0B — grouped DDP driver repair and formal legal evidence — 2026-09-09

- Hypothesis: the old Q1 query-at-a-time driver was the actionable R0 training
  contract bottleneck. A frame-centric grouped driver with complete candidate
  sets, Q=8, deterministic strata, and DDP accumulation should make the
  registered R0 experiment executable without changing the frozen bank,
  tracker, detector, or ordinary MOT/OVMOT.
- Preserved/disqualified Q1: PID 39571 was stopped gracefully; the old partial
  `outputs/r0/train/v1_formal_retry1/` remains historical and is not
  selection-eligible.
- Repairs: grouped Q8 schedule, video-local tiles, world4 DDP, accumulation 4,
  frozen detector/language inputs, complete candidate rows, present-uncovered
  masking, and resumable `locatemot-r0-tcgh-checkpoint-v2`. The legal GT
  materialization gap was repaired only in the R0 wrapper by recording and
  skipping missing annotation rows; frozen GT was not edited.
- Formal training: V1 completed 12 epochs/3,058 optimizer steps and V2
  completed 12 epochs/3,046 steps. Both were finite with nonzero gradients,
  strict reload, all four strata and both domains. Checkpoints 2/4/6/8/10/12
  are retained. Formal jobs used the free physical GPUs 4–7; physical GPUs
  0–3, which had unrelated processes, were not touched.
- Legal dev selection: V1 shortlist epochs 12/6 and V2 12/10; full dev
  TrackEval selected epoch12 for both. Dev HOTA was V1 22.8681% and V2
  19.5801% for the selected checkpoints.
- Internal validation-scope TrackEval: V1 0004/0018 HOTA 26.4582%, V2
  0016/0017/0020 HOTA 15.6097%. These are not screening or official-test
  results. No screening/official labels were read, and ordinary MOT/OVMOT,
  UIDM, L69 and production entrypoints were untouched.
- Outputs: training remains under `outputs/r0/`; large dev/inference/TrackEval
  products are under `/data2/usr_for_deadline/locatemot_r0b_*` because `/data1`
  was nearly full. No raw/dense debug cache or detector copy was created.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`. This is valid grouped R0
  implementation plus legal dev/internal TrackEval evidence, not final
  ordinary-RMOT completion. The only next action is supervisor review and a
  separately authorized follow-up.

## R1 — aligned track conditioning legal-dev contract — 2026-09-10

- Hypothesis/scope: test a zero-initialized RMOT-only residual conditioned on
  aligned frozen L89E Stage-S Z0/Z1/Z4 states and causal observation history;
  the anchor, L69 bank, ordinary MOT/OVMOT, UIDM and all production paths stay
  frozen. This is not final RMOT completion.
- Read-only legal-dev audit: frozen epoch-004 Rule-B anchor selection was
  rechecked. The epoch-004 presence-only miss is `3/400=.0075`, below the
  preregistered `.05` activation threshold, so the formal R1 config freezes
  `presence_residual=false`. Existing full-video artifacts have no candidate
  score arrays for a valid EXACT_INDEX_DEDUP replay; the conservative RAW
  policy is therefore frozen without claiming a dedup comparison.
- Preserved evidence: `outputs/r1/audit/legal_dev_selection_attempt1/` and
  `reports/r1/R1_LEGAL_DEV_SELECTION.md`; no training or new checkpoint was
  created by the audit. The next action is the bounded wiring/reload smoke;
  formal fit remains conditional on that implementation gate.

## R1 — formal fit and legal-development replay — 2026-09-12

- Hypothesis/scope: a zero-initialized aligned track-conditioning residual
  over frozen L89E stage states, causal observation history, language-directed
  R0 detail and complete L69 candidate sets could improve expression-to-track
  correspondence without changing the anchor, bank, tracker or ordinary
  MOT/OVMOT paths.
- Formal fit completed in isolated worktree
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R1`: six epochs, 2,250
  optimizer steps, all finite with nonzero gradients, strict reload for eight
  V1/V2 checkpoints, frozen anchor, and no persistent raw/dense debug cache.
- Prediction-only replays for legal-dev V1 videos `0008/0010/0020` and V2
  videos `0000/0008/0009` completed before legal GT was opened. The aggregate
  contains 2,385 groups and eight internal TrackEval results. Registered
  selection chose epoch 4 separately for V1 and V2. Selected HOTA was
  `24.2345%` (V1) and `23.6542%` (V2), versus the L89E anchor's `28.7628%`
  and `21.8385%`; this is not a stable breakthrough and remains legal-dev
  evidence only.
- Fixed 16-calibration/24-validation diagnostic retained all 2,435 L69 rows
  and 40 keys. R1 candidate-only recall/precision/FP-frame/pred-positive/
  hard/multi were `0.1000/0.1842/0.7750/0.5429/0.7727/0.1818`; the frozen
  Rule-B replay emitted zero rows for both anchor and R1 on this slice. The
  lower volume is a recall/multi-positive collapse, not a semantic repair;
  no threshold or NULL rescue was attempted.
- Status: `R1_NO_BREAKTHROUGH / STOPPED_PENDING_SUPERVISOR_REVIEW`.
  Outputs and failed technical attempts are preserved under `outputs/r1/`
  and `reports/r1/`. Flags remain
  `screening_gt_used=false`, `official_test_labels_read=false`, and
  `ordinary_mot_ovmot_touched=false`. The only next action is supervisor
  review and explicit authorization of one separate follow-up; no automatic
  continuation, screening, official test, or production merge is allowed.

## R1 — fixed semantic diagnostic closeout — 2026-09-12

- The fixed-slice evaluator used the native L62 order and attached labels only
  after row/feature construction and scoring. It used the frozen Rule-B values
  `candidate_threshold=1.0`, `presence_threshold=0.5`, `null_margin=0.0` and
  did not fit or tune a new threshold.
- This is a diagnostic, not a preregistered semantic gate and not HOTA. The
  authoritative files are in `outputs/r1/eval/fixed_semantic_attempt1/`; the
  corresponding report is `reports/r1/R1_FIXED_SEMANTIC_DIAGNOSTIC.md`.
# 2026-10-01 — new-server restoration started

- Hypothesis: restoring public V1/V2 data and exact base inference dependencies enables a new, attributable correspondence experiment; extending the failed R1 sidecar is unjustified.
- Evidence: fetched all remote branches; latest remains R1 `99090c8`. Read chronological historical endings and R1 reports: V1 legal-dev regression, limited V2 gain, fixed-slice score mismatch/output collapse. Old trained checkpoints are unavailable.
- Changes: started resumable official KITTI S3 **direct** download (15,813,146,295 bytes); downloaded and CRC-validated V1 expression/labels archives; isolated local data-tools environment. Full inventory/short plan: `reports/RESTORE_STATUS_20261001.md`.
- Failure/repair: shell sandbox lacks bubblewrap; execution requires tool escalation. Standard venv lacks ensurepip; local virtualenv bootstrap succeeded. Neither is a scientific result.
- Results: public V1 annotations recovered; KITTI/V2/runtime/project-generated assets pending. No historical checkpoint/bank reproduction claimed. Keep recovery scripts and new provenance; preserve migration AGENTS.md edit.

## LocateMOT-U U0 — unified specification and reference audit — 2026-10-01

- Hypothesis: a single specification-conditioned foundation can absorb the
  reusable persistent-query, temporal-language, open-vocabulary and reliability
  ideas from the ten requested repositories, provided the legal video boundary
  and one-checkpoint/shared-state contract are frozen before training.
- Code: cloned and shallow-audited MOTR, TransRMOT, TempRMOT, DKGTrack,
  FlexHook, iKUN, OVTR, COVTrack, ReferDINO and Open-GroundingDino; created
  `UnifiedSpecTrack` U0 shape skeleton, shared legal-scope guards, task/sample
  contracts, loss/runtime interfaces, parameter-sharing audit and reproducible
  audit/sanity tools. Exact source recipes and commit IDs are in
  `reports/unified/OPEN_SOURCE_TRAINING_AUDIT.md` and
  `outputs/unified/open_source_audit.json`.
- Data/protocol: froze V1/V2 fit/calibration/validation splits and reserved
  V1 `0005,0011,0013` plus V2 `0005,0011,0013,0019`; retained all 157 V2
  source target-ID-missing rows as unlabeled records. No official-test labels
  or screening ground truth were opened.
- Checkpoint/GPU: no LocateMOT-U checkpoint and no training run yet; GPU
  allocation was zero. External sanity was compile/help only (about 23 s wall
  time): 10/10 selected source files compiled, 1/10 entrypoints reached help,
  and the nine remaining failures are recorded dependency/version findings.
- Results: skeleton CPU forward finite with score shape `[1,100]`; legal guard
  smoke passed; skeleton non-head parameter fraction `0.9998259786` (not a
  final-model claim). No benchmark metric was produced. U0 scope/audit/sanity
  passed; U1 MM-GroundingDINO-B acquisition and proposal/expression gates are
  next.

## LocateMOT-U U1 — Swin-B foundation acquisition and adapter smoke — 2026-10-01

- Hypothesis: the audited GroundingDINO Swin-B CogCoOR weights can serve as the
  shared visual/specification foundation after selecting the checkpoint format
  compatible with the local MMDetection GroundingDINO implementation.
- Assets: acquired and hashed both the original release
  (`46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7`,
  938,057,991 bytes) and the OpenMMLab conversion
  (`55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`,
  935,971,702 bytes). The MMDetection asset is the runtime candidate because
  its key layout matches the adapter; provenance is in
  `outputs/unified/protocol/foundation_asset.json`.
- Code/smoke: built the local Swin-B model (232,996,763 parameters), loaded the
  converted checkpoint with 8 expected initialization keys and 3 BERT metadata
  keys reported as non-fatal mismatch, then ran one CUDA label-free forward on
  legal V1 video `0004` frame `000063.png` using `car . pedestrian . cyclist .`.
- Results: 300 finite predictions, finite boxes and finite scores;
  `outputs/unified/u1_foundation_smoke.json` records the exact config,
  checkpoint hash, device and legal frame. This is a compatibility smoke only;
  no proposal recall, expression grounding metric or tracking result is
  claimed. Official-test labels and screening ground truth remain unread.
- Checkpoint/GPU/walltime: no LocateMOT-U checkpoint; one A800-equivalent GPU;
  model build/forward about 9 s. Next gate is label-free proposal and
  expression grounding on legal development frames.

## LocateMOT-U U1 — proposal gate diagnostic — 2026-10-01

- Hypothesis: the loaded Swin-B foundation would provide sufficiently localized
  open-vocabulary candidates for the shared semantic/tracking path on a fixed
  legal development slice.
- Protocol: 16 V1/V2 calibration/validation expression-frame rows, 50 target
  boxes, 300 candidates per row; model inference completed before each
  corresponding label file was opened. No threshold fitting, training,
  official-test labels or screening ground truth.
- Result after correcting the MMDetection BGR input contract: best-IoU
  `>=0.25` was `0.5400`, best-IoU `>=0.50` was `0.0400`, mean best IoU
  `0.2756`, and mean score at best IoU `0.01312`. Boxes and scores were finite.
  This remains below a usable proposal gate, so semantic-head/tracking training
  from this stream is disallowed. The initial RGB/BGR-swapped diagnostic was
  discarded and is not used as evidence.
- Diagnosis/next change: inspect prompt tokenization/category assignment, box
  decoding and proposal selection on this same legal slice. Preserve the split,
  rows, checkpoint and labels; do not tune a threshold or delete hard cases.
  Authoritative output: `outputs/unified/u1_proposal_gate.json` and
  `reports/unified/U1_PROPOSAL_GATE.md`.
- Selection-only check: retaining all 900 decoder queries raised IoU@0.50 to
  `0.1000` and mean best IoU to `0.3341`, so top-k selection contributes but
  does not repair localization. The top-900 output is retained separately and
  is not used as a training bank.
- A fixed class prompt (`car . pedestrian . cyclist .`) with all 900 queries
  reached IoU@0.25 `0.9000`, IoU@0.50 `0.1800`, mean best IoU `0.3635`; this is
  a useful direction for the repair but still below the stage gate and remains
  excluded from training. Output:
  `outputs/unified/u1_proposal_gate_generic_classes_top900.json`.

## LocateMOT-U U1.3 — proposal-selection repair diagnostics finalized — 2026-10-01

- Change: made the canonical 300-query output reproducible with explicit
  `max_per_img` and prompt metadata, and synchronized the report/status values
  with the corrected BGR run. Re-ran the 300-query baseline, the 900-query
  selection diagnostic, and the fixed class prompt (`car . pedestrian .
  cyclist .`) diagnostic on the identical legal 16-row slice.
- Results: the baseline stayed at IoU@0.25 `0.5400`, IoU@0.50 `0.0400`, and
  mean best IoU `0.2756`; retaining 900 queries reached `0.1000` at IoU@0.50;
  the fixed class prompt with 900 queries reached `0.1800` at IoU@0.50 and
  `0.9000` at IoU@0.25. All boxes/scores were finite, no rows were skipped,
  and no official-test labels or screening ground truth were read.
- Interpretation: prompt/category assignment and top-k selection are concrete
  repair directions, but the proposal gate remains below the stage criterion.
  No semantic-head/tracking training bank was created and no training was
  launched. The next experiment must keep this split/checkpoint fixed while
  inspecting prompt tokenization, category assignment and box decoding.
- Verification: unified modules/tools compile; CPU skeleton smoke is finite;
  parameter-sharing audit remains skeleton-only with non-head fraction
  `0.9998259786`; legal guard accepts V1 `0004` and rejects reserved `0013`.

## LocateMOT-U U0R/U1R measurement-contract repair — 2026-10-01

- Scope: resumed from `014b8a5ec81bb83c5993b56da171089f8a440a11` on
  `codex/u0r-u1r-foundation-repair-20261001`; no U2 work and no training was
  launched. The project root was verified as `/data2/user/LocateMOT`.
- Reproducibility hypothesis: a fresh checkout should import the unified data
  package using tracked source only. The first clone failed because the broad
  `data/` ignore rule also hid `locatemot/unified/data/`. Added a narrow ignore
  exception, committed the package source, pushed it, and re-cloned the remote
  branch at `26f9c989c8b13773d681bbe6962b77e73a0026bd`. Imports, skeleton smoke,
  and foundation smoke passed in the clone. Decision: keep the repair and mark
  U0 legal-scope reproducibility PASS.
- Checkpoint hypothesis: the MMDetection Swin-B conversion must have only
  documented non-core load mismatches. `tools/unified/audit_foundation_checkpoint.py`
  now requires the exact one training-only missing key
  `dn_query_generator.label_embedding.weight`, the exact three BERT metadata
  unexpected keys, and zero backbone/encoder/decoder/bbox-head/language-fusion
  missing keys. Audit result: `PASS_CONDITIONAL_TRAINING_ONLY_MISSING`.
- Preprocessing hypothesis: the formal runtime must be equivalent to the
  official MMDetection path. Ten deterministic legal images were compared with
  the same fixed generic vocabulary; all produced 300 finite predictions,
  maximum box difference `0.0`, and maximum score difference `0.0`. Decision:
  preprocessing contract PASS; the prior hand-built tensor smoke is not formal
  evidence.
- Split hypothesis: a deterministic video-disjoint split balancing expression
  volume, physical-target volume, and motion proxy is adequate before formal
  evaluation. `tools/unified/build_legal_split.py` froze v2 with seed `20261001`
  and split SHA
  `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`.
  V1 fit/calibration/validation are `9/3/3` videos and V2 are `9/3/3`; V2
  source-target-ID-missing references remain `157` (fit only in this split),
  with no replacement or silent drop. Decision: freeze this split before any
  U1 adaptation or formal gate.
- Hardware/wall time: one CUDA device (A800-equivalent) was used for the
  checkpoint audit, smoke, and equivalence check; these short checks completed
  in under a minute each. Official-test labels and screening ground truth were
  not opened. External baseline execution remains incomplete and is the next
  blocking U0 task.

## LocateMOT-U U0R — TempRMOT legal-development execution — 2026-10-01

- Hypothesis: at least one published external RMOT method can be executed on a
  frozen legal-development subset without reading official-test labels or
  mixing its environment into LocateMOT-U.
- Protocol: copied TempRMOT at commit
  `6a65640d849fdee4a32bb055945ee34c3b0edeb1` into the isolated environment
  `/data2/user/reference_envs/temprmot`, loaded the released checkpoint
  `checkpoint_rk.pth` (SHA-256
  `55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8`), and
  ran the adapter on V1 validation videos `0012` and `0014`, one legal
  expression per video. TrackEval then evaluated the two generated sequences
  with HOTA, CLEAR, and Identity.
- Result: prediction and evaluation completed with 94 prediction rows and 93
  ground-truth rows. Combined HOTA was `94.679`, MOTA `98.925`, and IDF1
  `99.465`. Official-test labels, screening ground truth, and training were not
  used. The raw outputs and environment remain outside the repository;
  `outputs/unified/external_baselines/temprmot_legal_v1.json` records the
  reproducible metadata and `reports/unified/EXTERNAL_BASELINE_LEGAL_DEV.md`
  records the bounded scope.
- Compatibility work was confined to the isolated copy: a torchtext shim,
  current PyTorch `torch.load` behavior, disabling a redundant pretrained
  download, a CUDA scalar-type API update, and NumPy aliases in TrackEval.
  These changes do not alter the LocateMOT-U model or the frozen split.
- Decision: keep the TempRMOT run as real external-baseline evidence, while
  retaining `U0_EXTERNAL_BASELINE=INCOMPLETE` until a second independent
  method also produces predictions and evaluation. DKGTrack and FlexHook
  remain pending because their published weight links are not yet available
  through the isolated download path.

## LocateMOT-U U1R — formal generic and expression measurements — 2026-10-01

- Measurement contract: the frozen split SHA is
  `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`; the
  foundation checkpoint SHA is
  `55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`; every
  forward used `LoadImageFromFile`, `FixScaleResize(800,1333)`,
  `PackDetInputs`, and `DetDataPreprocessor` with BGR-to-RGB conversion.
- Generic protocol: one fixed query-independent vocabulary, all legal
  calibration/validation frames with materialized labels, physical units
  `(dataset, video, frame, track_id)`, Top100/150/300/900 diagnostics, and
  small/medium/large height strata. The run covered 4,201 unique images,
  33,526 physical GT units, and 5,992 unique physical objects. Top150 recall
  was V1 `0.0803` and V2 `0.0669` at IoU 0.50 (V1/V2 IoU .25: `.5309/.4940`,
  IoU .75: `.0045/.0062`). Decision: formal generic gate FAIL; this is below
  the `.80` adaptation early-pass, so a detection adaptation probe is required
  before any grounding adaptation.
- Expression protocol: every legal calibration/validation query contributed
  one deterministic middle labeled frame and, when available, one first empty
  target frame. Units were `(dataset, video, query_id, frame, target_id)`;
  source-missing references stayed in the source denominator. The run covered
  7,160 query-frame rows and 7,199 valid target units with zero source-missing
  units in this calibration/validation sample. Top1 IoU .50 recall was V1
  `0.0030` and V2 `0.0028`; AP50 and empty-frame FP/query are recorded in
  `reports/unified/U1_EXPRESSION_GROUNDING_GATE.md`. Decision: formal
  expression gate FAIL; no expression adaptation has started.
- No official-test labels or screening ground truth were read. No U2 or
  universal track core work was started. The external-baseline state remains
  incomplete because only TempRMOT has real predictions/evaluation; DKGTrack
  and FlexHook public Baidu links require an authenticated download path.

## LocateMOT-U U1-A — protocol correction, valid adaptation, and stop — 2026-10-01

- The first eight-process launch was invalid because MMEngine's default
  `--launcher none` left the processes outside DDP; it was stopped before any
  formal evidence. A later exploratory FP32 run also used `frozen_stages=1`.
  MMDetection's Swin implementation shows that value freezes patch embedding
  and stage0 only, while the registered contract requires stages 0 and 1.
  Those exploratory metrics are excluded from the final record.
- Corrected hypothesis/protocol: a fixed legal-fit generic stream can repair
  the foundation proposal localization while freezing Swin stages 0/1 and
  BERT. The legal-fit stream has 3,606 rows and fixed `label 0 -> car`; the
  split, vocabulary, official `FixScaleResize(800,1333)` pipeline, and all
  denominators stayed unchanged. The valid run used
  `frozen_stages=2`, BF16 AMP, eight-GPU PyTorch DDP, batch 2/GPU, and no
  accumulation (global batch 16). The logged peak was 5,945 MB/GPU. A two-step
  smoke passed before the full run.
- A batch-2/two-step-accumulation attempt passed a one-step smoke but failed in
  the full run with PyTorch's `expect_autograd_hooks_` DDP reducer assertion.
  The smallest working fix was accumulation 1; this kept the recommended
  microbatch 2 and did not alter the official input resolution. This failure
  and the corrected smoke are retained in the adaptation report.
- The valid schedule completed epochs 1, 2, 4, and 6, each formally evaluated
  on the unchanged 4,201-image, 33,526-physical-unit scope. Top150
  Recall@IoU0.50 (V1/V2) was foundation `0.0803/0.0669`, then
  `0.8729/0.8515` (epoch1), `0.8873/0.8766` (epoch2),
  `0.8945/0.8843` (epoch4), and `0.8938/0.8815` (epoch6). IoU0.25 reached
  about `.99` in both domains, while epoch-6 IoU0.75 was `0.7215/0.6119`.
  Size-stratum IoU0.50 was small `0.9136`, medium `0.9445`, and large
  `0.7085`.
- Training logs contain occasional `grad_norm: nan` values, but loss values
  remained finite, all six checkpoints were written, and every formal shard
  produced finite predictions. This telemetry is reported rather than
  filtered. The final epoch-6 checkpoint SHA-256 is
  `a1ff39f9c0bd59a5fd2b14968a528fc11e882b8fa6a3e85a9b64253a1cdc5ccc`.
- Decision: the valid V1/V2 generic gate did not reach `.90/.90` after the
  registered six-epoch maximum. Expression adaptation was not started because
  it is downstream of the generic gate. Added the corrected adaptation report,
  failure decomposition, and four valid formal JSONs. Final status is
  `LOCATEMOT_U_BLOCKED_GENERIC_GATE_AFTER_MAX_ADAPTATION` with
  `STOPPED_PENDING_SUPERVISOR_REVIEW`; U2 was not launched.

## LocateMOT-U U1R2 — measurement repair, class-aware refinement, and foundation completion — 2026-10-01

- Contract hypothesis: the remaining U1 failures were caused by a measurement/data contract mismatch rather than a need to replace the MM-GroundingDINO-B foundation. The source `labels_with_ids` fields are normalized top-left `xywh`; the repaired formal gates preserve source extents and use Top150 IoU.50 for generic proposals and Top100 IoU.50 target coverage for expressions. The previous U1-A-v1 files remain preserved as historical artifacts and were not overwritten.
- Coordinate/source audit: legal calibration/validation contains 33,526 physical GT boxes. 195 V1 boxes extend below the image at `y1>height` (166 large, 27 medium, 2 small); no x-side overflow exists. Same-frame geometry matching to official KITTI labels gives mean IoU `0.7414883`; Refer and official track IDs are separate namespaces. Decision: retain source extents and recover classes by geometry; do not clip evaluation GT.
- Data hypothesis: a class-aware physical-frame union can repair localization while removing V1/V2 duplicate training. The fit stream contains 2,545 images and 10,502 boxes; 3,424 V1/V2 overlaps were removed, 580 conflicting boxes and 21 missing class matches were recorded, and observed classes are car 6,787, pedestrian 3,700, cyclist 13, van 2. Decision: retain the stream with fixed class policy and explicit low-confidence audit.
- Numerical stability: the prior BF16 log has three finite-loss `grad_norm:nan` records (epoch/iter 1/50, 3/100, 6/100). A fresh FP32 one-step audit from epoch4 found 981 trainable parameters, total loss `40.27210998535156`, zero nonfinite gradients, and zero no-gradient parameters. Decision: R2 uses FP32/TF32, no AMP/BF16.
- Generic refinement: one bounded FP32 epoch from the frozen epoch4 checkpoint passed the corrected gate: V1/V2 Top150 IoU.50 `0.982716/0.971837`; small/medium/large `0.937467/0.989984/0.997825`. Decision: freeze the R2 epoch1 generic checkpoint and skip epoch2/mixed fallback.
- Expression hypothesis: whole-expression positive spans with all referred boxes, legal empty frames, and preserved V2 missing-target references can add language-conditioned candidate coverage without erasing generic proposals. One FP32 epoch completed from the selected generic checkpoint. Expression Top100 IoU.50 is V1/V2 `1.000000/0.961367`; the same checkpoint's generic regression is V1/V2 Top150 IoU.50 `0.986592/0.958739`.
- Language audit: on exactly 500 legal validation same-image pairs with a same-video sentence from a different target set, true versus shuffled Top10 recall is `0.959333/0.871133` and AP50 is `0.736901/0.491159`. Both strict comparisons pass, so language conditioning is measurable rather than generic-box-only behavior.
- External evaluator: synthetic perfect/empty/ID-switch and cross-evaluator arrays pass with maximum difference `0.0`. TempRMOT evaluated videos 0012/0014 occur in its declared training pool; its HOTA 94.679 is reclassified as `PIPELINE_SANITY_ONLY_TRAINING_OVERLAP`.
- Decision: the final shared expression checkpoint satisfies all U1 hard gates. `official_test_labels_read=false`, `screening_gt_used=false`, V2 source-target-ID-missing references remain explicit (157, silently dropped=false), and no U2/U3/U4 work started. Set `LOCATEMOT_U_READY_FOR_U2` and stop pending supervisor review.

## LocateMOT-U U2 — shared persistent track core and portability stop — 2026-10-02

- Provenance: working branch `codex/u2-shared-persistent-track-core-20261001`,
  authoritative local base `56507f4aa9a12e76d6f9f6702f0f0b38163666ec`, and
  immutable U1 foundation SHA
  `4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.
- Implementation: replaced the U0 tracking placeholders with one shared U2
  core (`PerceptionBridge`, independent identity/semantic candidate encoders,
  `UnifiedTrackState`, four-layer identity set interaction, GRU identity
  memory/anchor/EMA, association/new/no-match, lifecycle, motion, Hungarian
  rollout, and bounded OVMOT residual). The runtime is
  `locatemot/unified/runtime/u2_online_tracker.py`; no RMOT or prompt tracking
  was launched.
- Contract evidence: `tools/unified/u2_causal_contract.py` passed birth,
  occlusion, no-future, MOT specification invariance, OVMOT isolation, and
  online-state save/reload. Actual-model sharing audit passed with
  `shared_non_head_fraction=0.9378111093950003`.
- U2-P data manifest SHA-256 is
  `575bd349e00bab23634c6929c04513149afec838de419bb3f1a53469fafc7fda`.
  KITTI legal development portability passes (V1/V2 Top150 @.50
  `0.986592/0.958739`), while BDD100K, DanceTrack, MOT17, MOT20, TAO, and
  C-TAO are missing locally. Candidate-cache SHA is `not built`.
- Training command: not launched; GPU count `0`, optimizer steps `0`, wall
  clock training runtime `0`. Simple MOT, MOT-only, and joint OVMOT metrics
  are intentionally `null` because the registered portability protocol is
  incomplete.
- Decision: `LOCATEMOT_U_BLOCKED_U2_PERCEPTION_PORTABILITY`; stop pending local
  acquisition/audit of the missing streams. Official test labels and
  Refer-KITTI official evaluation labels remain unread.
- Validation note: the 2-process CUDA forward/backward smoke passed with the
  same MOT task on both ranks. An 8-process smoke could not initialize because
  all eight A800 devices were occupied by an unrelated process; it was
  terminated before any training step and is not counted as a pass.

## LocateMOT-U U2 — acquisition recovery — 2026-10-11

- Revalidated local descendant `666ab44` and the actual U1R2 checkpoint SHA
  `4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.
  U1 references and the checkpoint remain unchanged.
- Observed MOT17/MOT20 extraction counts of 33,810/13,430 files, with nested
  release roots. TAO's four volume roots exist but contain zero files; no
  training or download worker was live. The old download module spawned
  workers while being imported by each child, recursively launching itself.
- Replaced that launcher with one guarded entry point and four thread workers
  running bounded download clients. Keep the registered four-volume assignment,
  prioritize modal train/validation inputs, validate downloaded sizes, and
  record actual per-file SHA-256 for source provenance. A fresh direct-network
  README transfer passed: 9,189/9,189 bytes, so existing Baidu authentication is
  usable. This does not imply that the dataset or portability gate is complete.
- Identified a separate eligibility bug: the old portability artifact could
  return PASS solely from directory availability while all new-domain metrics
  were null. Correct the audit and eligibility calculation before any training;
  metric definitions, split rules, thresholds, and seeds remain unchanged.

## U2 frozen-foundation MOT portability — 2026-10-11

Hypothesis: the selected U1R2 foundation may need bounded adaptation on crowded/non-road pedestrian domains. Registered ByteTrack inclusive train-half ranges, FRCNN physical-video deduplication, `person .`, positive mark-1/class-1 GT and unmodified source box geometry before measurement. The loaded checkpoint SHA matched the immutable U1R2 reference, with no missing/unexpected weights.

Ran all 7,115 dev frames on GPUs 1–7 in one parent launch/wait command, leaving an externally occupied GPU 0 alone. MOT17: 2,652 images / 53,890 GT; Top150 0.787233, Top300 0.806253. MOT20: 4,463 images / 615,137 GT; Top150 0.731010, Top300 0.884673. Exact coverage and monotonic count checks passed; partial measurements are rejected. MOT17-02/04 and crowded MOT20-03/05 dominate the shortfall. Retain the measured failure; do not alter visibility filtering, vocabulary, frame scope or gates to improve the reported values.

Track training is held. At most two epochs of mixed U2-P perception adaptation are justified after the missing domains/protocols are ready, with U1 replay and full retention gates. MOT20 Top300 has not met the preferred 0.90 or been accepted as a runtime override. TAO four-volume acquisition/extraction continues. Cloud BDD resources are static detection rather than MOT 2020. Started official DanceTrack train/val and C-TAO base-only acquisition, pinning release revisions; aria2 TLS failed, curl direct HTTP-206 probe succeeded and curl replaces it. No tracker or RMOT/prompt training was launched.

U2-P follow-up: C-TAO base-only source downloaded and inspected: 490,210 train images, 1,489,637 base-category boxes, 500 videos; supervision frequencies f/c only and no validation paths. LVIS v1 full vocabulary frozen (1,203 names; 866 f/c base, 337 rare novel). Native MMDetection chunking passed a real TAO-image smoke with 38 chunks and exactly one backbone call, followed by class-agnostic IoU-0.8 merge and global Top300. This is runtime evidence, not a substitute for full TAO coverage. Original modal TAO validation has 36,375 images / 113,112 boxes; 314 boxes lack a LVIS-v1 synset mapping, so the benchmark conversion/audit must explicitly account for them. Original BDD MOT 2020 image host dl.yf.io is live; started eight train/val archives (38,886,924,336 bytes) with original published MD5 checks, four bounded workers and eight connections/file. Official 2021 tracking-label S3 URL returns 404; keep BDD labels missing while locating a valid original release.

## U2 TAO/C-TAO portability registration — 2026-10-12

Continue from the published full MOT-half failure measurements, preserving the immutable U1R2 checkpoint. Plan: freeze all official TAO validation images plus every original annotated C-TAO training-audit frame; preserve the full dense C-TAO set for later identity training. Use all 1,203 LVIS-v1 names, the already verified 32-name chunks, one image backbone extraction, class-agnostic IoU-0.8 merge and global Top300. Count all original modal TAO GT including the 314 annotations without a v1 synset mapping; report those separately with source-frequency grouping rather than dropping them. Wait for CRC-completed source archives internally in the single evaluation launcher while downloads/extraction continue. In parallel, finish Dance extraction and locate original BDD tracking labels. No track training is eligible yet.

## U2 data completion and full portability launch — 2026-10-12

Hypothesis: recovering the original data and measuring complete dev scopes will distinguish foundation domain failure from incomplete measurement. DanceTrack download/CRC extraction completed; preserved 32/8/25 project videos and exact 33,772/8,024/25,508 frames. TAO modal train/val and all 490,210 C-TAO training images now pass availability coverage (six available streams, BDD pending). Frozen full TAO validation (36,375 images, 110,055 base / 3,057 novel GT) and original C-TAO train audit (17,131 images, 53,263 base GT); retain all 314 unmapped original TAO GT units. Immutable-foundation real smokes passed in both chunked domains and DanceTrack; full audits run on GPUs 1–6 and 7 respectively. No partial recall promoted; MOT gates still fail and track training remains held.

BDD root cause: old official annotation URLs are dead and the user cloud BDD folder is static detection. Found a pinned HF archive whose directory contains all 1,400/200 raw tracking JSON videos and validation images. Download/hash/selected-member CRC extraction is active; source labels have not yet established availability. Added original toolkit category/ignored/crowd and inclusive Box2D handling; raw-label full-scope validation will precede BDD measurement. Original seven train-image archives continue downloading, with an internal blocking extractor. Keep these recovery changes; no learned model, seeds or gates changed.

## U2 BDD raw tracking recovery validated — 2026-10-12

Recovered and SHA/CRC-validated the pinned mirror: all 1,400 train and 200 val raw JSON videos; every 39,973 validation image matches the original release ZIP member size/CRC. Frozen 278,079 train / 39,973 val frames and the fixed eight-category vocabulary before inference. Full official-reference comparison covers all validation objects: 442,814 positive boxes, 18,625 crowd and 4,073 official ignored-category objects. First strict comparison exposed a 2.84e-14 rounding difference from direct x2+1; changed arithmetic to the official xywh reconstruction order, then every positive box/category/ID matched exactly (no tolerance relaxation). Keep the correction. Full BDD portability is queued after the DanceTrack worker on GPU 7 with a real-image smoke first. The first original train-image archive is CRC-extracted (200 videos, 39,418 frames); remaining train-image downloads/extraction continue. Numerical MOT gate failures and track-training hold remain unchanged.

## U2-P bounded perception adaptation preparation — 2026-10-12

Plan before any adaptation result: use a fixed 20,000-anchor replay epoch, maximum two epochs, seed 20261012. U1 generic/V1-expression/V2-expression proportions 0.10/0.15/0.25; BDD/Dance/MOT17/MOT20/TAO base detection each 0.10. Stratified video sampling covers every training video where the allocated budget permits; cycle deterministically when replay pools are smaller than their allocations. Preserve original U1 replay boxes/phrases; use only official train or registered MOT-half train frames. TAO class supervision uses mapped LVIS-v1 f/c categories only, with novel/unmapped annotations excluded from class and identity supervision. Encode detection classes as positive phrase spans through the existing native ODVG grounding path. Empty positives retain a fixed absent base-class caption. Keep FP32, decoder/multimodal LR 1e-5, Swin stages 2/3 LR 1e-6, BERT and Swin stages 0/1 frozen. Do not start optimization until all source images and full F0 domain audits are complete. Any adapted checkpoint must pass the unchanged U1 retention and complete portability gates; no tracker/cache before that.

Implemented the fixed replay preparator, checkpoint-compatible frozen-BERT model, guarded eight-GPU trainer, explicit candidate-SHA evaluation, crowded-K registration, and full-audit checkpoint selection. Config parsing exposed an inherited-variable NameError; use MMEngine's `_base_.train_pipeline`. Disabled aspect-ratio buckets because odd bucket tails can add singleton steps; ordinary batching preserves exactly 1,250 steps/epoch at global batch 16. Actual immutable checkpoint CPU loading matches every state key, BERT stays eval/frozen, Swin stages 0/1 stay frozen and stages 2/3 train. Native ODVG packing passed on eight real training-stream images and one empty-positive image; first check assumed a box wrapper but the native pipeline returns tensors, so corrected that check and reran. Reaggregated all 7,115 original MOT frames: every result remains identical. Mixed-checkpoint measurement/worker aggregation is rejected; F0 MOT20 Top300 0.8846728452 correctly fails the mandatory 0.90 override threshold. Original generic/expression/500-pair full U1 reference scopes and shard provenance validate. Keep these changes. Replay preparation session 81929 waits internally for actual BDD extractor PID 686873; training preflight reports incomplete BDD images, incomplete full F0 audits and pending full replay. No perception optimizer, track training, or candidate cache launched.

Scientific source backups use `/data2/user/locatemot_git_backups/u2_latest.bundle`; the adjacent `u2_latest.commit` records the exact original local HEAD preserved in each updated bundle. Original U1R2 history and the BDD recovery commits remain in this history.

Scope correction before any optimization: an epoch-1 retention failure rejects that checkpoint for tracking, but the specification still permits the second bounded perception epoch. Allow that optimizer continuation after all epoch-1 audits complete; both retention and portability acceptance thresholds remain unchanged, and no rejected candidate becomes track-eligible. The two-epoch/2,500-step cap remains fixed.

## U2 DanceTrack full measurement and bounded continuation — 2026-10-12

DanceTrack's original blocking audit completed: all 25,508 dev images, 225,148 positive source GT; Top150 224,318/225,148 = 0.996313536, Top300 224,627/225,148 = 0.997685967. Full frame coverage, registered GT count and immutable foundation provenance pass. Keep the complete measurement; it shows that the current foundation's failure is not universal across ordinary MOT domains. BDD's queued real-image check passed and its complete 39,973-frame audit began on GPU 7; the second original training-image archive completed CRC extraction (200 videos, 39,990 frames). No partial BDD/TAO values are promoted.

Plan: connect the already registered two-epoch trainer to complete ordinary/TAO portability, original eight-shard U1 generic/expression and fixed 500-pair audits, conditional MOT20 K300 registration, and checkpoint selection. One blocking controller will wait for the existing F0 audit/data-preparation parents, train only after all prerequisites hold, pause at each saved epoch for full audits, and stop after selection or the absolute two-epoch cap. Preserve exact worker partitions and resume only unfinished units; no metric, seed, frame scope or acceptance threshold changes.

All original BDD train-image extraction and replay preparation completed successfully during this stage. Required data availability is now 7/7; 1,400 train videos and every 278,079 training annotation frame have an image. The fixed replay is complete (SHA `1be4a6e2d03fd3f59aeeca8b931c3f638a2ca1db015ea7d54afa2ca725970947`), all 1,400 BDD and 500 TAO train videos represented; actual prompts use at most 21 tokens. Native eight-rank Gloo verification used the real 20,000-row ODVG dataset and actual configured sampler/batching: 2,500 rows and 1,250 batch-2 steps per rank, globally no duplicate/omitted row, all 300,070 positive boxes retained, 1,385 empty-positive rows retained, zero native box-filter removals. No GPU model or optimizer was created by that check. The first launch exposed a missing `.venv/bin/torchrun` entry point; use the same virtual-environment Python's `torch.distributed.run` module, rerun the targeted eight-rank check, and apply the identical launcher fix to the continuous trainer. Keep this fix; no package, seed, precision, LR or budget changes. FP32 training keeps the already registered U1 TF32 flags. All original eight-shard U1 scopes validate, and the original U1 checkpoint is correctly rejected as a completed 1,250-step U2-P candidate. Only incomplete full F0 BDD/TAO audits now hold optimization. The controller validates each existing parent's full artifact as it exits, then owns training and all audits; it never terminates unrelated jobs or advances through a failed/incomplete measurement.

Scientific source commit `cf41d7312e7046c10ebc802aa624d1be81e1a021` is backed up in the latest local bundle. Launched the continuous command with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/run_u2_perception_adaptation.py --tao-audit-pid 670232 --bdd-audit-pid 781721 --replay-pid 816267 --extraction-pid 625610`. Controller PID 854970 / exec session 11636 is confirmed live and has already validated the completed replay parent; it still waits for the two original full F0 audit parents. GPU count 8; optimizer steps 0; candidate cache not built. Clarified the existing native resume semantics after reading installed MMEngine: optimizer, scheduler and sampler epoch resume, while separately launched epochs use the same fixed base seed 20261012; worker RNG streams are not restored. This documents the configured epoch-boundary behavior, without changing seeds or training implementation. End-to-end training wall time is pending the blocking controller's completion.

## U2 real feature bridge and eight-GPU temporal validation — 2026-10-12

Hypothesis: attaching final-decoder query provenance before native chunk concatenation/NMS and sampling pre-text neck maps can supply correctly aligned identity/semantic features without changing the frozen detector's candidate distribution. Implemented `runtime/u2_grounding_candidates.py`: original boxes remain original-coordinate xyxy; identity sampling scales them into resized/padded image coordinates, preserving preprocessor padding rather than normalizing against only original image extent. The extractor reads no annotations and stores no GT identities. Actual checkpoint-backed GPU-0 validation (`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/validate_u2_candidate_features.py --device cuda:0`, terminal session 84086, exit 0) passed native box/score/class bit-exact parity for MOT17-02 frame 000302 Top150 and one TAO validation frame under the complete 1,203-category/38-chunk Top300 protocol. Selected decoder features remain bit-exact after cross-chunk NMS; backbone executes once; fixed-box identity features remain bit-exact after changing vocabulary. Scope is an alignment contract, not a replacement for full-dataset recall. Keep the bridge; full cache awaits passing U2-P selection.

The eight-GPU temporal contract (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 tools/unified/u2_ddp_temporal_contract.py`, terminal session 18873, exit 0) passed both MOT and OVMOT. Two four-frame clips/rank (global 16) retain temporal gradients; identity GRU, set interaction and association gradients are nonzero, with semantic GRU/text gradients also nonzero on OVMOT. Parameters remain bit-exact across all eight distinct GPU ranks after scratch updates. This synthetic probe does not modify the foundation or constitute a formal association run; no tracking metrics are claimed. Keep the full-clip-in-one-DDP-forward contract for the eventual trainer. Evidence: `eval/ddp_eight_rank_core.json` and `eval/real_candidate_feature_contract.json`.

Confirmed GitHub publication of original source `0b6f1d4e20faed17bc105b67aef4505bcdc87ea2` at remote `5d79bfcc0b0b3f58594817dc727d5707bb1d6331` (verified branch and commit tree through the connected app). Corrected stale final-status fields that still called BDD missing and Dance unmeasured: all seven sources are available, Dance complete, BDD/TAO full audits remain pending. The existing blocking continuous controller remains live; no gates, seeds, candidates or active audit jobs were changed.

## U2 candidate-cache and ordinary-baseline execution plan — 2026-10-12

Read the real candidate bridge, frozen ordinary/TAO protocols, KITTI legal scope and TrackEval readers. The TempRMOT fork changes MOT file loading into expression-specific prediction paths; do not use that reader for ordinary MOT evaluation. Original KITTI tracking labels already exist under `/data2/user/reference_envs/kitti_tracking_labels_20261001/training/label_02`; no extra dataset download is required. Plan: pin an unmodified upstream TrackEval, freeze KITTI V2 legal fit/calibration/validation frame lists with original tracking IDs, implement a label-free video-sharded cache that requires the existing complete perception selection, and implement the IoU/motion/visual-cosine simple baseline with fixed parameters before any benchmark results. Keep supervision in separate source adapters, preserve all existing gates and running full F0 jobs, and test only the contracts needed for valid tracking/evaluation.

Pinned direct upstream TrackEval at `12c8791b303e0a0b50f753af204249e622d0281a`, verified unmodified, and recorded its six relevant readers/metrics. Frozen original KITTI V2 legal fit/calibration/validation covers 15 videos, 1,814/2,087/1,385 frames; never opened the four reserved video labels. Original KITTI and Refer tracking namespaces differ, so the earlier U1 generic result alone was incomplete evidence for ordinary tracking GT. Added the missing full original-GT portability audit over 3,472 dev images / 15,875 eligible car/pedestrian units with unchanged pinned TrackEval occlusion/truncation handling. Existing U1 references, all old domain scopes, .85/.90/.80 ordinary thresholds, .70/.60 OVMOT thresholds and the two-epoch cap remain intact. Actual F0 KITTI image smoke passed (session 37939 exit 0, native state keys all matched). Launched the full original-KITTI F0 audit on GPU 0: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/u2_mot_portability.py --protocol outputs/unified/u2/protocol/kitti_ordinary_portability.json --out outputs/unified/u2/perception/kitti_original_gt_f0_full --gpus 0`; parent 864092, worker 864093, blocking session 95238. The old full BDD/TAO jobs remain untouched. U2-P candidate evaluation now includes 76,068 ordinary images, plus all original 53,506 TAO/C-TAO and U1 scopes.

Implemented full label-free cache inventory and selected-checkpoint-gated 64-frame video shards. C-TAO dense and modal TAO train union contains 491,353 unique train images (including 1,143 modal images outside the C-TAO set), plus 36,375 TAO dev images. OVMOT storage assignment balances four volumes at 131,934/131,931/131,931/131,932 frames. Ordinary train/dev cache covers 396,865 images. Actual inventory verification (session 32148 exit 0) proves all 3,182 groups and all 924,593 physical images occur exactly once, with no train/dev image overlap and exact eight-worker group coverage. Cache generation remains conditional on recomputing the original complete selection; features are BF16, original boxes/scores FP32, semantic/decoder storage is shared, and supervision stays separate with no automatic novel labels.

Implemented the frozen simple baseline and official full-sequence metric/export path. Registered parameters before benchmark results: IoU/motion-IoU/id-cosine .25/.35/.40, margin .45, distance 2, birth .10, appearance EMA .90, max_age 30. All input candidates participate in matching, low scores can recover IDs, and no motion-only box is emitted. Synthetic causal/export/metric validation (session 40233 exit 0) passed all five upstream readers: perfect HOTA/DetA/AssA/IDF1 1 and IDSW 0, while swapped MOT17 identities give DetA 1, AssA .4, IDF1 2/3, IDSW 2. BDD inclusive extents/aliases/ignored attributes and KITTI DontCare/MOT distractors are honored. No benchmark metric is claimed. The optional unrelated BURST import warns about missing tabulate; all required ordinary readers and metrics execute successfully, so no unrelated package repair is needed. Keep the cache and simple pipeline.

Launched one blocking follow-on queue: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_ordinary_baseline.py --wait-perception-pid 854970 --kitti-audit-pid 864092 --extraction-pid 625610`, controller 864319 / session 35256. It waits for genuine completion of the original-KITTI audit and the existing bounded perception owner, then builds the selected ordinary cache and measures all five simple-baseline dev sets. If the existing owner genuinely terminates at preflight before any optimizer because the newly supplied KITTI audit was still incomplete, the queue resumes only after that audit completes and the full unchanged preflight is READY. No live job is restarted on an observation timeout. Formal cache and formal track training are still unlaunched; the research goal remains active.

## U2-A temporal supervision implementation plan — 2026-10-12

The complete original-KITTI F0 audit terminated successfully (session 95238 exit 0): 3,472 images, 15,875 source GT units, Top150/Top300 recall .928503937/.940535433. The original perception owner and ordinary follow-on queue are confirmed live. Read the U2 temporal curriculum/teacher forcing/loss specification and actual core: query drop/injection is currently diagnostic-only, and motion/lifecycle assume consecutive processed frames despite required stride 1–4. Implement real training-state augmentation from past unmatched candidates, current-frame Hungarian GT targets and scheduled model-in-loop state assignment, the six registered identity/motion/lifecycle losses with explicit false-birth suppression, and physical frame-gap-aware state propagation. Use the same shared core, keep GT bookkeeping outside its runtime state, retain original gates and do not launch formal association optimization before selected caches and simple metrics exist. Validate crossing/occlusion, negative proposals, mixed teacher/model rollouts, connected temporal gradients, causality and task isolation before the 6,000-step run.

Implemented actual query-state drop and previous-frame hard-negative injection; external first-identity bookkeeping does not hide model switches or duplicate births. Registered six losses include positive column CE plus NEW reject BCE so false proposals (including the first frame) are not blindly taught births. Current-frame positive/distractor Hungarian and separate crowd IoA/min-height rules preserve official ignore semantics. Physical source-frame gaps drive motion, lifecycle and memory decay; expired IDs are retired before current matching. Batched association data transfers, GRU/birth updates and switch margins avoid per-track GPU/CPU synchronization without changing core parameters. Crossing/occlusion, false-birth/ignore gradients, mixed teacher/model rollout and connected temporal gradients pass; all existing causal/isolation/save-load checks pass.

Actual eight-GPU BF16 registered-loss proof initially failed at the Hungarian NumPy conversion (`TypeError: unsupported ScalarType BFloat16`, session 51784 exit 1; reproduced single GPU session 68632). Smallest fix: cast detached logits to FP32 only at the CPU assignment boundary. Single-GPU regression session 30093 and eight-GPU session 2469 passed: global16 clips, finite/nonzero candidate projection/identity GRU/pair/set/motion/alive/birth gradients, identical post-update rank parameters. Subsequent batched-transition/expiry edits trigger fresh targeted and dual-task/eight-GPU checks before publication. The first original-GT oracle check (53170 exit1) compared MOT reader XYWH to core XYXY; verified native source/reader format and fixed the oracle only. Session 16135 passed every positive box on five complete actual train videos: BDD203/1126 units, Dance703/4912, MOT17 301/8701, MOT20 215/9060, KITTI154/234. Scope is these five videos, not all train labels or benchmark results.

Added frozen-cache causal clip sampling, 6,000-step eight-GPU MOT-only entry, registered curriculum/teacher probabilities/seed20261012/dataset probabilities, full all-five dev evaluations at1000/2500/4000/6000, 500-step resume snapshots with optimizer and per-rank CPU/CUDA RNG, unchanged identity-benefit/HOTA gates and four-level checkpoint selection. Synthetic sampler/resume/selection proof passes: next stochastic update bit-exact after restore, train-only rank/step sampling, >2-point individual HOTA regressions rejected. Preflight (55952 exit2) correctly holds without the complete selected cache and all-five simple metrics; zero formal track updates. Launched one blocking follow-on queue: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_mot_only.py --wait-ordinary-pid 864319`; actual PID870157/session38396 waits for original ordinary owner, then launches exactly 6,000 MOT-only steps when prerequisites pass, without OVMOT. Original full F0/perception owners remain live and untouched. Keep implementation; U2/U3 and all trained metrics remain unproven.

The fresh batched-transition eight-GPU command (76774 exit1) passed both MOT/OVMOT temporal tasks first, then exposed a BF16 vector-index assignment mismatch: GRU output BF16 into FP32 persistent memory. Added explicit differentiable casts to destination identity/semantic memory dtype; no loss or parameter change. Run the actual single-GPU BF16 backward/active-state inference regression, then the full eight-GPU registered-loss proof again before publishing. The failure is explicitly recorded; old PASS artifacts do not certify the new code until the fresh proof succeeds.

Final current-source targeted regressions completed: session21978 exit0 (six-loss crossing/expiry, five actual-video GT comparison, sampler/RNG/selection), session1677 exit0 (post-dtype-fix temporal/causal/save-load and resume/selection), session87550 exit0 (fresh actual eight-GPU BF16 registered losses, all necessary gradients nonzero, both updates identical across ranks). The preceding dual-task eight-GPU FP32 phase of76774 passed for MOT and OVMOT; only its BF16 phase failed and was repaired. Single-GPU session57241 exit0 proves batched BF16 backward and nonempty-state causal inference. Keep these final implementations and proofs. No formal feature cache, learned MOT baseline or joint benchmark is claimed; the research goal remains active.

During verified publication/follow-on waits, original perception owner11636 reported BDD audit completion. The original BDD blocking session36974 terminated exit0; strict full measurement validation confirms all39,973 images,442,814 GT, coverage/protocol/F0 SHA. Hits363,947 Top150 (.8218958750175017),377,519 Top300 (.8525453124788286). BDD fails .85 minimum but exceeds .80 hard stop; no BDD K300 exception is allowed. All five ordinary F0 audits are now measured; bounded U2-P still waits for original full TAO/C-TAO audit. Updated canonical portability/status/reports and retain the same registered 2-epoch repair and six-thousand-step downstream gates. No formal track optimizer launched; keep result and publish this new full measurement.

## U2-B supervision/semantic/evaluation implementation plan — 2026-10-12

Previous goal turn made concrete progress: complete original KITTI/BDD F0 measurements, actual temporal supervision plus gated 6k MOT-only implementation, current eight-GPU BF16/causal/GT/resume proofs, source commits/backups and verified GitHub publications. Current source30c7a56 is clean; original bounded perception owner11636 and MOT follow-on38396 are confirmed live. Read OVMOT adapter, shared language bridge and U2 sections66–82. The OVMOT report incorrectly says TAO/C-TAO absent; all assets are present. Raw C-TAO interpolation resets frame_index (e.g. original frame0176=175, interpolated frame0177=1), and C-TAO/TAO numeric IDs occupy different source namespaces; physical filename chronology and original annotation UUID alignment are required. Plan: audit all dense/base supervision with no automatic rare/unmapped class or identity labels; implement shared-U1 BERT/projected full-vocabulary class features, semantic classification/consistency/gate losses and same-core OVMOT temporal rollout; pin official upstream TETA and implement complete original validation export/overall/base/novel reporting with explicit category mapping counts. Keep all ordinary loss weights and gates. Formal joint optimization requires selected MOT-only eligibility and is not authorized by contract fixtures or incomplete measurements; do not launch it now.

Full compiler failures were diagnosed sequentially without dropping units: camera timestamps require original index anchors (66320 exit1), LaSOT uses numeric filenames (second command terminal exit1), dense ArgoVerse frame_id compresses missing-image gaps (22205 exit1; 17/37 camera videos have nonconstant anchor offsets), and official dense annotations include nine zero-area boxes (97023 exit1). Retain source geometry/counts, use numbered filenames or timestamp interpolation between original TAO anchors, and leave zero-area GT at zero IoU. Final compiler29355 exit0 covers all500 train videos/491,353 images/1,489,990 unique base units; 52,911 base units deduplicate across sources, 353 modal-only base units remain. Novel/unmapped modal1,156/219 units never provide class/identity labels. The initial assumption of differing overlapping numeric IDs was not observed: all shared UUIDs have equal numeric IDs. Keep UUID alignment with original IDs retained separately. No validation/test labels enter training.

Actual F0 shared text extraction63671 exit0 yields1203x256 features,866 base/337 novel,38 native BERT chunks/max104 tokens/backbone0; F0 remains reference-only, not selected for formal training. Implemented existing-parameter semantic classification, final pair supervision, persistent-memory classification, consistency and gate regularization, with unknown candidate/row/contrast labels masked and certified negatives retained. Single80792 exit0 and actual eight-GPU7759 exit0 prove BF16 connected identity/semantic gradients, unknown birth-label exclusion, ordinary text isolation, full novel inference vocabulary and identical post-update ranks. This is one scratch update on synthetic track candidates with actual text features, not joint training. Ordinary temporal regression20603 and32602 exit0 passes unchanged six-loss values. Actual AVA/ArgoVerse/LaSOT full-video loaders67140 exit0 cover976/451/1169 frames and4074/1372/1169 units, all base-only/strictly increasing time. Deterministic joint task draws give6518 MOT/3482 OVMOT in10,000 steps; registered probabilities remain.65/.35.

Pinned unchanged official TETA b498aa87 and prepared complete LVIS-v1 TAO validation10545 exit0: all36,375 images/988 videos;112,798/113,112 original GT units and5473/5485 tracks map to v1;314 units/12 tracks explicitly unmapped by the official OVTrack synset conversion. Original v0.5 full-localization scope remains unchanged. TETA fixture22487 failed because OUTPUT_SUMMARY=False leaves upstream returned threshold uninitialized; use the official default summary=True with no source/metric modification. Fixture67046 exposed our bit-packed export IDs allocating huge dense upstream ID maps; stopped that owned synthetic command (exit143), removed the packing, and let official TAO resolve video-local IDs. Final61184 exit0 passes perfect100, wrong-class localization/association separation, identity-switch sensitivity and non-exhaustive unlabeled-object neutrality, with empty frames retained. No benchmark TETA/ordinary tracking score claimed, no formal joint optimizer or U3 launch. Keep all changes; original perception and downstream queues continue under unchanged gates.

Fresh ordinary eight-GPU BF16 regression71729 exit0 after the OVMOT changes again confirms all six registered loss gradients and identical rank parameters. Existing gate recipes, candidate caps, seeds, metrics and training scopes stay unchanged.

Current-source causal/state save-reload regression58539 exit0 passes on nonempty states, including ordinary text invariance and OVMOT raw identity/GRU isolation. No full candidate cache, simple tracking metric, learned MOT checkpoint or joint benchmark yet; continue the active U2 goal after publishing this completed source/supervision/semantic/TETA stage.

## U2-B gated joint execution plan — 2026-10-12

Previous goal turn made progress: full base-only dense supervision, actual shared F0 class features, same-core semantic/partial-label losses, official TETA validation conversion/export and actual eight-GPU proofs were committed3d2a9ec and published/verified as aff3b00. Current project identity/worktree are correct/clean; perception854970, ordinary864319 and MOT-only870157 are confirmed live. Read current baseline trainer/evaluator, both cache phases and U2 sections64–85. Implement exact4,000-step joint entry initialized with every selected MOT-only core tensor; global task.65/.35, all8 ranks/two8-frame clips, frozen foundation/BERT and separate5e-5/1e-4 core/adapter groups. At each1,000-step boundary evaluate every ordinary dev sequence plus all TAO validation through official TETA; reject any individual HOTA or AssA loss>.01 and enforce TETA34/AssocA37. Add exact RNG/optimizer resume and one blocking downstream owner that waits for real MOT-only completion/eligibility before OV feature-cache/class-text construction and joint launch. Actual semantic fixtures do not authorize a formal run. The full rank-zero metrics can exceed default NCCL collective timeout; extend only the collective wait timeout, with no change to data/metrics/optimizer/evaluation boundaries. Validate gate rejection, checkpoint initialization/resume and alternating-task eight-GPU execution before queuing.

Implemented the complete4,000-step joint trainer/evaluator and gated downstream owner. Registration/preflight55888 and29449 correctly return exit2 HELD with zero formal updates while selected caches/MOT-only are absent. Joint boundary85960 exit1 exposed .59-.60 floating comparison falsely rejecting the inclusive1pt threshold; MOT boundary4594 exit1 also exposed .4+.03 rejecting exact3pt gain. Compare the recorded decimal scores at the same literal.01/.02/.03 thresholds, with no epsilon/extra rounding tolerance or gate weakening. Final joint62479 and ordinary22178 exit0 pass inclusive boundaries, individual HOTA/AssA rejection, both OVMOT gates, refinement conditions and four-checkpoint selection. Original ordinary14478 regression also passed before this targeted numeric fix.

Actual eight-GPU BF16 alternating-task/init/optimizer proof14038 exit0 passed; strengthened1629 exit0 additionally restores the real scheduled teacher Bernoulli RNG for a bit-exact next update after actual joint checkpoint save/load. Both use all256-D core tensors,16 global clips/eight frames, actual reference-only F0 class text and synthetic tracking candidates; no formal optimizer exists. First whole eight-rank joint-evaluation fixture84831 exit1 failed at BDD predicted category person: the fixture incorrectly used the raw GT alias as detector vocabulary. Correct only the fixture to the existing registered pedestrian vocabulary; keep the actual BDD normalization, official reader, metrics and original scope unchanged. Run the whole entry again before queuing/publication.

Final whole joint-evaluation proof99228 exit0 executes the actual eight-rank entry, all five official ordinary readers, both base/novel TAO fixture videos and official TETA with8 CPU workers. Simulated interruption after completed video units resumes with zero core forward calls and identical full metrics. These scores are explicitly synthetic and are not benchmark evidence; no real tracking/foundation optimization or U3 readiness is claimed. Keep exact whole-entry implementation and all gates. The current source stage now has a single blocking joint downstream owner waiting for genuine completion of MOT-only870157; only a recomputed eligible selected reference permits full balanced OVMOT feature caching, selected actual class text, and the4k joint run. No live source jobs were restarted.

Launched that single blocking owner: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_joint.py --wait-mot-only-pid 870157`, actualPID883611/session32672. State confirmedWAITING_ELIGIBLE_MOT_ONLY; fullOVcache_started=False and formal_joint_training_launched=False. Keep all new implementations/proofs and publish the completed joint execution stage; the full U2 goal remains active with actual perception/cache/simple/learned/joint measurements unproven.

## U2 self-contained native checkpoint implementation plan — 2026-10-12

Previous goal turn made progress: scientific355c152 and exact eight-GPU joint execution/resume/evaluation proofs were committed, backed up and verified at GitHub c16cd98. Project root is correct and original perception/cache/baseline/joint owners are confirmed live. Read actual native runtime, U2 core and specification sections83–86. Final format must embed actual MM-GroundingDINO/shared BERT, tokenizer/config, every persistent core/head and task embeddings; the shape-only fallback is insufficient. Current core has no task embedding tensor. Add zero-initialized task embeddings only to the bounded semantic gate input; ordinary MOT gate remains exactly zero and raw identity/GRU remains task isolated. No formal track optimization has begun, so register this missing required component before selected caches/checkpoints. Implement native model reconstruction from embedded BERT architecture/tokenizer and strict embedded weights, with no external checkpoint/HF pretrained load. Validate actual F0 candidate/class-text bit-exact reload, causal state resume and same-core parameter sharing with a clearly reference-only random core pack; never treat it as a selected final checkpoint or U3 readiness. Actual final packing must recompute joint and perception eligibility and retain all original metadata/scopes.

Native reference-pack32496 exit1 exposed bbox-head constructor-derived fields saved after native configuration mutation;64857 exit1 then exposed the analogous denoising-query fields. Remove only those six constructor-derived entries before serialization; native construction recreates them.41966 exit1 passed strict complete foundation/core tensor reload but failed shared-BERT feature equality. Targeted31560 exit0 proves native from_pretrained uses SDPA while bare embedded BERT construction uses eager: max feature discrepancy1.144409e-5; current native output remains bit-exact with the immutable F0 text cache. Store and restore the actual native attention implementation rather than weaken comparison tolerance. Task-embedding joint eight-GPU60299 exit0 retains bit-exact next stochastic optimizer update, nonzero embedding gradient and ordinary adapter gradNone.35642 exit0 passes causal/temporal and CPU semantic regressions; strengthened38946 exit0 explicitly perturbs task embeddings and proves ordinary MOT unchanged/raw OVMOT identity isolation plus unchanged eligibility/initialization checks. No formal optimizer, reference, live owner, seed or gate was changed.

Attention attempt51064 exit1 still used eager because HF BertConfig.from_dict mutates the supplied dictionary and overwrites attn_implementation from kwargs;15925 verifies the actual packed class remained BertSelfAttention vs native BertSdpaSelfAttention. Pass a copied config plus explicit attention kwarg. Final native19361 exit0 proves all actual F0 foundation/core tensors,1203 class features, native MOT150/OVMOT300 candidate features and nonempty online state reload bit-exact, with external tensor/config/pretrained-HF calls prohibited. V1/V2 small deterministic generic/expression/true/shuffled integration outputs and metrics are bit-exact; this is a reference-only random core pack, not a selected final artifact or tracking benchmark. Task embeddings now use supplied zero weights without consuming RNG, preserving all subsequent shared-core initialization draws at the original seed. Final joint52310, causal/temporal/gate8930 and whole eight-rank official evaluation33800 exit0 pass. Targeted32182 recomputes actual parameter sharing after excluding shared detection/association/motion/lifecycle heads:236,751,992 shared non-head vs329,475 conservatively task-specific, fraction.9986102878>=.90. Only count metadata changed in the reference file; actual restored foundation/core tensors remain bit-exact. Formal packer recomputes original perception/MOT/joint eligibility, embeds original source/protocol/sampler/step metadata and remains pending actual selected-runtime retention. Keep all changes. No selected cache, real tracking benchmark, formal optimizer or U3 readiness yet.

Current CPU semantic18032 exit0 confirms task-embedding gradients and unknown-row/candidate masks after preserving initialization RNG. Store CPU evidence separately as ovmot_training_contract_cpu.json, retaining the historical GPU single-frame report. Updated eight-GPU joint evidence covers BF16 gradients and exact restore with current embeddings.

## U2 frozen text reuse for upcoming candidate caches — 2026-10-12

Previous goal turn made progress: actual native one-file F0/core reconstruction, task-embedding isolation and real parameter-sharing/retention proofs were committed2b775b7, backed up and verified at GitHub9c12586. Project root/clean worktree verified; original F0/perception and downstream owners are live and ordinary cache is still unlaunched. Read native GroundingDINO.predict: every full LVIS frame re-encodes the same38 BERT prompts although the language backbone is frozen; native prompt/token-positive-map preprocessing is also repeated. Implement worker-local fixed-text memoization by wrapping native language forward and prompt construction, preserving all native image/backbone/encoder/decoder/NMS operations and cloning returned tensors/maps before mutable native consumers. Apply only to upcoming full selected-foundation cache workers, without changing any running full audit or perception-training process. Measure actual FP32 F0 two-frame MOT and full1203-class OVMOT candidate/selected-feature equality, mutation isolation and actual BERT-call counts/timing; no approximate precision, cap, chunk, vocabulary, seed or scope change. Keep only if bit-exact outputs hold.

Initial frozen-text38589 exit0 proves actual FP32 F0 native/cached MOT150 and complete1203-class OVMOT300 two-frame candidate/feature parity, tensor/map mutation isolation and exact method restoration. Warm OVMOT elapsed5.208873→4.619842s (1.1275x observed throughput), with two-frame BERT76→38 calls and unchanged2 backbones/76 decoder calls. Implement only in upcoming selected-foundation cache workers; record source/proof provenance in spec and actual counters in all worker/full reports. Integrated80238 exit0 reconfirms full parity and executes the actual worker/serializer on explicitly separate two-frame reference inventories for both phases: serialized FP32/BF16 fields bit-exact, semantic/decoder storage shared, zero GT labels, completed-unit resume with no inference and identical unit manifests. Final warm OVMOT5.255846→4.638013s (1.1332x), MOT.188991→.180778s (1.0454x); shared-GPU two-frame observations, not full-cache speed estimates. Keep optimization; all actual source frames, chunk/candidate limits, precision, eight-rank assignment, four-volume distribution, original gates and running full jobs are unchanged. No formal selected cache, track optimizer, benchmark tracking score or U3 readiness exists yet.

## U2 selected-artifact finalization execution plan — 2026-10-12

Previous goal turn made progress: bit-exact real F0 fixed-text reuse and actual cache-worker serialization/resume were committed1ad7563, backed up and verified at GitHub4306013. Current project/worktree are correct/clean. Original joint blocking handle32672 is confirmed live and waits eligible MOT-only; no formal OV cache/joint has begun. Read U2 sections83–89/98, actual packer, small retention audit and joint owner. Implement a separate single blocking finalization owner waiting original883611 completion, then recompute full perception/MOT/joint eligibility, package the actual selected U2-B foundation/core, audit actual packed retention and unmodified selected-core causal/isolation/save-load behavior, and generate final readiness solely from current full artifacts/provenance. Reference-only F0 packs and scratch proofs cannot fulfill selected-artifact gates. Reject or retain exact held/blocked status when earlier gates fail; conditional U2-C remains a separate bounded task if eligible, never substitute a U2-B pack. Preserve all running owners and budgets. Use deterministic real candidate probes for actual learned heads; never modify selected weights to make runtime contract checks pass.

Existing historical two-GPU smoke had only one backward, while the requested two-step contract was missing. Updated the existing joint proof entry to accept expected world size2 or8 without changing its default eight-rank behavior; actual CUDA0/7 two-rank26053 exit0 now covers MOT/OVMOT updates and exact stochastic next-update checkpoint/RNG/optimizer restore. Fresh single-GPU8844 exit0 includes nonzero current task-embedding gradient and unchanged BF16 identity/semantic loss values; existing current-eight-GPU evidence stays intact. Reference selected-core-audit entry99248 exit1 exposed native candidate metadata strings being cloned as tensors; clone only tensor fields and retain metadata unchanged, then rerun targeted entry. This is an audit-interface repair, not a selected-weight, seed, gate or training-protocol change.

Selected-core entry48760 exit0 now passes four actual native F0 MOT frames and unchanged reference-core weights:150 births/repeat matches, one-frame occlusion/recovery, future-state independence, arbitrary MOT text invariance, raw OVMOT/fixed-pair GRU isolation and bit-exact state resume for both tasks on one core object. Readiness69971 exit0 tests the independent complete requirement set/material failure cases and the actual CLI returns2 HELD with missing full cache/simple measurements. Actual selected-audit CLI75746 exit0 proof rejects a reference-only random pack before cache read. Review found old small generic retention used only car rather than U1 original10-entity vocabulary; use original VOCABULARY in final audit and require that exact vocabulary in readiness. Full-vocabulary native/reference-pack96632 exit0 is bit-exact on both V1/V2; all original full U1 references are unchanged, and historical reference metadata now names the restricted car-only scope. Missing actual selected audits never become readiness evidence. Keep all repairs/entries.

Launched one blocking finalization owner with OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_finalize.py --wait-joint-pid 883611. ActualPID896659/session18355 confirms WAITING_ORIGINAL_JOINT_OWNER, pack_started=False, ready_for_u3=False. It waits real original completion, recomputes current full selection, packs actual frozen U2-B tensors, runs original-vocabulary retention and unmodified selected-core audits, then verifies every final gate against bound artifacts. Completed units resume by actual checkpoint receipt; reference fixtures and conditional U2-C cannot substitute for selected U2-B. No original owner was restarted, no additional formal optimizer launched and no U3 readiness claimed.

## U2 native image-batch admissibility probe — 2026-10-12

Previous goal turn made progress: selected-artifact finalization and bound real-audit/readiness entries were committed4102740, backed up and verified at GitHub592d097. Current worktree/project are correct/clean; original full TAO parent63615 is confirmed live through its existing blocking handle, and finalization waits original883611. No actual formal baseline/cache is ready. Read native inference_detector and fusion attention before considering future throughput: list-image API still loops singleton test_step, whereas the multimodal fusion uses a scalar max over the full batch. Probe actual FP32 native batch2 versus singleton on the two fixed original MOT17 frames, plus a second-image perturbation while keeping the first fixed. This is a small admissibility measurement, not a new formal audit or source/model change. Reject batching if candidate parity or first-frame independence fails; keep running jobs, candidate policy and original native singleton protocol unchanged.

Native image-batch probe10904 exit0 yields a genuine negative optimization result: same native preprocessing shape on two fixed MOT17 frames, but batch2 boxes/scores differ from singleton on both. Max same-rank score differences.000565775/.002606869; large coordinate differences1033.97/1865.55 reflect changed sorted proposal alignment and are not a matched-box localization error measurement. All labels remain equal. Actual six fusion stable_softmax_2d flags are False; the suspected scalar max is inactive, and changing only second-image pixels leaves the first batch output bit-exact in this probe. Reject native batch2 on candidate parity; no full OVMOT batch or semantic-feature integration is attempted, no TF32/precision tolerance or candidate policy is changed. Keep native singleton plus already proven fixed-text reuse. Original full TAO/perception/baseline/finalization jobs remain under their original blocking owners and gates; no tracking benchmark or U3 readiness is claimed.

## U2 upcoming perception-audit fixed-text reuse — 2026-10-12

Previous goal turn made progress: actual FP32 native image-batch parity failure was measured and committed24be26a/published64cb0ff; retain native singleton. Current project/worktree are correct/clean, and existing full TAO63615 blocking handle is confirmed live. Read current full TAO worker and bounded adapted-checkpoint evaluation caller. The already bit-exact fixed native text reuse can reduce repeated38-caption BERT work in upcoming adapted TAO/C-TAO audits too. Add an explicit optional CLI flag with unchanged default F0 behavior; only future adapted-checkpoint evaluator requests it. Keep identical singleton/native NMS, protocols, GT denominators, checkpoint/source fingerprints, resume rows and gates. Validate actual unchanged two-source smoke rows/recall counts with and without the flag before retaining this integration. Do not modify or restart the current loaded F0 workers.

Actual paired portability-worker37402 exit0 confirms optional text reuse: original two-source smoke metadata/rows identical,300 proposals each, TAO2base/1novel GT and C-TAO10base GT hit counts unchanged, one backbone/image, BERT38calls/38hits across two complete1203-class passes. Native/reuse elapsed11.545474/10.907659s are limited two-image cold/warm observations, not full throughput. Flag defaultFalse preserves F0 metadata/commands and resumable exact frame prefixes; only future adapted-checkpoint evaluation explicitly passes it to new workers. Keep this verified integration; all original full processes, protocols, GT denominators, weights, budgets, seeds and acceptance criteria remain unchanged. Actual adapted full audit and all tracking results remain unmeasured.

## U2 full F0 completion and native checkpoint load repair — 2026-10-12

Original blocking TAO audit63615 returned exit0/COMPLETE: all36,375 TAO validation and17,131 C-TAO original training frames measured. TAO base105,408/110,055=.957775658, novel2,777/3,057=.908406935; C-TAO base50,975/53,263=.957043351, novel inapplicable. Both OV portability streams pass; BDD/MOT17/MOT20 ordinary F0 gates still fail, so bounded U2-P is required. Its original owner11636 genuinely terminated at native MMEngine checkpoint loading before any optimization, because PyTorch's default weights_only=True rejects the original checkpoint's HistoryBuffer. Plan: retain original full measurements and exact eight-GPU replay/protocol; apply the existing U1 process-local TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD convention, verify the actual native loader on immutable U1R2, then resume the unfinished epoch and terminal downstream queues with updated owner PIDs. No metric, seed, budget or acceptance change.

Targeted regression94105 exit0: all six terminal shards exactly cover53,506 unique registered frames, every frame has one backbone call and300 native proposals, both source GT denominators and measurement totals match. The native MMEngine loader now loads the immutable SHA-verified actual U1R2 state_dict1,069 tensors plus HistoryBuffer/optimizer; full unchanged preflight is READY. No epoch checkpoint or first optimization update existed at failure. All four original downstream handles returned terminal HELD states; recover their owners only after this evidence. Keep the localized compatibility fix.

Recovery blocking supervisor23776 launched genuine new owners after all old handles terminated: perception1054069, ordinary1054070, MOT-only1054071, joint1054072, finalization1054074. Original failed states preserved in outputs/unified/u2/recovery_20261012/*.original; original training log retained. The single supervisor waits these exact processes; downstream unchanged gates prevent cache or tracking launch before a fully selected passing foundation. Full budget remains1,250 steps/epoch, at most2epochs, followed by6,000 MOT-only and eligible4,000 joint steps. Actual selected pack and U3 readiness remain unproven.

## U2-P epoch1 complete; adapted audit worker argument repair —2026-10-12

Recovery23776 returned genuine terminal exit1 after epoch1 completed all1,250 updates and saved epoch_1.pth (SHA1d9077ae603f40de0e955a11a88083f914f98faa967bc46cdd06bcdfb5129688). Actual checkpoint contract PASS compares265 immutable BERT/Swin0/1 tensors bit-exactly with original U1R2. First actionable evaluation failure: all ordinary workers exit2 before inference because the parent omits --measurements-dir, so their adapted-checkpoint parser sees the default F0 directory. The TAO parent has the same omission and would fail at its next stage. Plan: propagate the already explicitly supplied independent directory in both existing worker commands; verify actual adapted native smoke workers through both parent launchers, preserving guards and scopes; resume unfinished full evaluation from the completed epoch1 checkpoint, never rerun epoch1 optimization. All recovery downstream owners are terminal HELD, and will be restored behind the new perception owner.

Storage rebalance requested by user —2026-10-12: move only11 completed/CRC-extracted TAO archives from /data0/user (87.981GiB,49.54% of prior usage) and /data1/user (85.328GiB,48.72%) to matching /data3/user/TAO-Amodal/frames paths. Destination has789GiB free; extracted train/val/test image paths and scientific protocols remain unchanged. Two parallel volume transfers verify each destination against its existing download SHA before atomically replacing the original archive with a symlink; no source data is removed before verified copying. Plan recorded in outputs/unified/u2/storage_rebalance_20261012/plan.json.

User superseded archive relocation: original transfer69601 stopped with exit143; deleted all21 CRC-extracted completed TAO archives across allfour volumes (345.674GiB) plus cancelled relocation temporary copies (68.760GiB). New plan migrates approximately half of cleaned /data0/user and /data1/user image storage, using only test/validation directories; train directories remain in place. rsync validates transferred data, then a dry-run checks complete matching trees; Linux renameat2 exchange atomically replaces each original directory with a target symlink before removing its verified redundant original copy. Scientific image paths, original source annotations and all protocols stay valid. New explicit plan: outputs/unified/u2/storage_rebalance_20261012/image_migration_plan.json.

Post-cleanup image migration completed:95 directory units, /data0/user44.814GiB and /data1/user44.907GiB relocated to /data3/user, each49.999% of cleaned initial occupancy. rsync full-tree parity passed before atomic symlink exchange; all original image paths remain resolvable, only test/validation storage moved, train paths and experimental contracts unchanged. Final actual directory sizes and df recorded in outputs/unified/u2/storage_rebalance_20261012/image_migration_result.json.

Adapted launcher regression96599 exit0: through the actual ordinary and TAO parent launchers, each actual child receives the separate measurements directory and real epoch1 checkpoint SHA; one original legal KITTI frame and original two-source TAO/C-TAO smoke frames execute native inference successfully. Test-only smoke injection and collector do not produce formal measurements or weaken aggregate completeness; report is explicitly reference_only. Full recovery supervisor95952 resumes epoch1 evaluation from existing epoch_1.pth, new perception1089957/ordinary1089958/MOT-only1089959/joint1089960/finalization1089961 owners retain unchanged budgets and gates. All previous owners returned terminal, their failure states retained. Full U1 retention and adapted portability remain pending; no tracking or U3 claim.

User-requested inference sharding throughput plan —2026-10-12: full audits already use8 physical GPU shards, while native full1203-class TAO performs38 multimodal transformer passes per image. User asks to shorten runtime. Temporarily SIGSTOP exact evaluator1090197 and its existing TAO parent1238269/workers1238270..1238277, retaining all completed ordinary measurements and existing TAO exact frame prefixes; no original source inference re-started. Measure1/2/4 independent singleton processes on GPU1, real adapted epoch1 SHA1d9077a, identical12 reference frames spanning original TAO source domains, full vocabulary/native FP32/static text reuse and exact boxes/scores/classes parity. This is a bounded throughput probe, not a metric or scope change. Adopt higher concurrency only if actual throughput improves with unchanged outputs; otherwise retain the fastest measured concurrency and resume the existing owner. Probe shell60967; hold receipt outputs/unified/u2/perception_adaptation/throughput_probe_hold.json.

Same-GPU probe60967 completes: identical12 actual native full-vocabulary reference frames take158.996s with1 process,251.286s with2 and220.054s with4; all proposal boxes/scores/classes are bit-exact. Reject extra processes: single worker per GPU is fastest on this sample. Initial default-device memory counters were on GPU0 instead of model GPU1; omit those zero counters from published aggregate, retaining actual wall-clock/parity evidence.

Native metadata optimization85924 completes: explicit target CUDA binding plus one bulk integer-label copy for native label_names construction reduces the same12-frame sample to103.870s (observed1.531x), boxes/scores/classes bit-exact for all12, native full label_names identical on the first full-vocabulary frame; native methods restored. Actual model GPU1 peak allocation1,664,202,752bytes. This comparison includes CUDA binding and changed background load (seven formal audit workers resumed), so it is not an isolated causal label-copy benchmark or a full-run speed guarantee. Keep integer-only optimization provisionally; verify actual two-source native worker rows before resuming formal prefixes with it. Vocabulary1203, chunks32/38, NMS.8, Top300, precision/weights/protocol/GT denominators remain unchanged.

Actual bulk native worker63758 exit0: unchanged two-source smoke rows/GT hit counts and resume metadata are exact,300 candidates per frame, one backbone per frame,76 head calls/68,400 integer class labels; native/bulk CLI wall68.842/51.966s includes initialization and two frames, not formal throughput. After verified controlled interruption,95952 returned terminal exit1 (expected interrupted TAO command) and all four gated followers returned HELD without tracking launch. Every existing exact TAO prefix parsed successfully:440 completed frames preserved across initialized shards; shard1 had not yet written metadata or rows. All76,068 ordinary frames and epoch1 original1,250 updates retained. New supervisor57530 launches perception1313649/ordinary1313650/MOT-only1313651/joint1313652/finalization1313653. Full evaluation skips complete ordinary measurements and resumes unchanged eight TAO partitions with explicit GPU binding plus bulk integer metadata; labels alone temporarily use CPU for native name construction and return to the original device before NMS. Both flags preserve full1,203 vocabulary,38 native chunks, FP32, backbone counts, GT scope and all gates.

Actual epoch1 ordinary summary: BDD Top150.936574, Dance.999036, MOT17.878066, MOT20 Top150.819136/Top300.973281, original legal KITTI.946394. All five full ordinary streams are measured; MOT20 satisfies the registered Top300 crowded threshold but formal registration/selection still waits complete TAO and U1 retention. No selected foundation, candidate cache, tracking benchmark or U3 readiness is claimed.

### 2026-10-13 — U2 perception epoch 1 selected

- Completed the unchanged full TAO/CTAO portability audit: 36,375 / 17,131 images, Top300 base recall 0.9742855845 / 0.9729080225; TAO novel recall 0.9460255152 (2,892 / 3,057). Unmapped TAO GT retained separately (305 / 314 hits).
- Epoch 1 foundation `1d9077ae603f40de0e955a11a88083f914f98faa967bc46cdd06bcdfb5129688` passed all seven portability streams and U1 retention: generic V1/V2 0.9918663023 / 0.9672269380; expression Top100 V1/V2 1.0 / 0.9785374828. Fixed 500-pair language audit true/shuffled AP50 0.7301285093 / 0.4067468580. Generic scope 4,201 images; expression scope 7,160 representative query frames, 7,199 target references.
- Selection source receipts and their registered digests agree; retain epoch 1, no second adaptation epoch required. Existing recovery owner exited successfully and advanced the eight-GPU ordinary candidate cache automatically. Formal tracking scores and U3 readiness remain pending.
- Evidence: `outputs/unified/u2/perception_adaptation/eval_epoch_1/selection.json`, seven `measurements/*.json`, and generic/expression/counterfactual `result.json`.

### 2026-10-13 — user-directed GroundJEV research pivot

- User terminated full U2 candidate caching and authorized GroundAnything DLM + pretrained OneJev tracking research on at most four GPUs. Goal changed; this is not algorithm failure/NO_GO.
- Verified /proc ancestry/start ticks and stopped 13 old U2 processes with SIGTERM; no old queue/cache worker remains. Other project GPU jobs preserved. Removed 4,748 verified feature shards (39.89 GiB allocated), retaining manifests/data/results and loadable epoch-1 checkpoint SHA 1d9077ae… (1,909,880,505 bytes).
- New worktree /data2/user/LocateMOT_GroundJEV and branch codex/groundjev-4gpu-20261013 inherit freshly fetched remote 51614cb. Official GroundAnything/OneJev code cloned directly without proxy; isolated environments/weights setup started. No new training or tracking metric claimed.
- Correct inherited status terminology: generic .9787925789 is Top150 recall IoU .5, not AP; expression Top100 V1/V2 recall 1/.9785374828, while .7237566558/.6556441180 are query AP diagnostics. Raw receipts unchanged.
- Next: real DLM/OneJev loading, own-state short clip → full legal MOT video, fixed-candidate Simple/Full comparison, tracking-SFT and shared RMOT/OVMOT evaluation. Keep evidence-based failure repair and finite training budgets.

### 2026-10-13 — GroundJEV legal inputs and initial online interfaces

- Verified real legal MOT17-02 dev299, DanceTrack0004 full-val1203, MOT20-01 dev214 frames. No new data download or test-GT access needed for first video work.
- Implemented official-grid DLM HTTP candidates with missing native confidence explicitly preserved, pretrained OneJev batched Choice/Noul/Score identity and lifecycle adapter, Hungarian global competition, reliable long-memory updates and causal save/restore. Real crop evidence is explicitly distinct from pending native GroundAnything features. Replay control preserves identical candidates and reports no end-to-end FPS for cached perception.
- A two-frame real annotated CPU fixture passes identity persistence, exact resumed state and rejected past timestamps; GT is used only in this explicit fixture, not as a claimed runtime/benchmark result. Real model integration remains pending dependencies and weights.
- Installed independent Python3.12.15. Separate OneJev/SGLang installs active. PyYAML shared uv lock failure fixed by dedicated setup cache. Direct model transport hit aria2 TLS and curl HTTP2 cancellation; retained byte prefixes. Eight concurrent official1MiB ranges complete in7.397s (1.134MB/s aggregate); adopt sixteen8MiB ranges/large model on HTTP1.1 with official final weight digests. No proxy, no GPU usage yet.

### 2026-10-13 — official video evaluation and runtime setup progress

- Added complete-scope ordinary MOT export using unchanged TrackEval12c8791; all299 legal MOT17-02 dev frames in a labeled GT oracle fixture yield expected perfect metrics/zero IDSW/FP/FN. This validates format/scope handling, not GroundJEV performance. Runtime replay now checks actual physical image paths.
- To avoid repeated multi-GB dependency transfers, copied matching CPython3.10 installed framework files into an independent OneJev environment, excluding old Transformers/MMDetection and source console scripts. First copy attempt used the assumed venv package directory and copied zero files; corrected to actual distribution roots, then recorded nonzero distributions. Source environment untouched, no shared symlinks. Torch2.10+cu128/vision0.25 imports and CUDA availability on GPU6 pass; OneJev/Transformers5.5.4 installed and all69 package requirements compatible.
- Prefetched official pinned large PyPI wheels via verified byte ranges. Exact-version py3-none NVIDIA dependencies reused for GroundAnything's separate Python3.12/Torch2.9.1 setup. Controlled package transport interruptions preserve caches/bootstrap/custom vendor source; remaining official installation/ABI check continues. Universal flashinfer-cubin wheel was initially excluded by the platform selector; corrected from actual PyPI filename and completed the missing wheel.
- One blocking readiness queue owns the actual pretrained MOT-frame check, gated on full official OneJev weight receipt. Models still downloading; no real-model tracking or training result claimed. Optimizer steps zero, at most four physical GPUs enforced by common launcher; other projects preserved.

### 2026-10-14 — first actual OneJev frame and checkpoint-version repair

- Official OneJev weights complete and LFS digest matches: model.safetensors2,214,590,296bytes SHA cb4b7456703baa26a0cfacf177067200b50021df5a6ba2a2cd1e6f6c868f2244. No weight/model substitution.
- FirstGPU load reads474 weight entries but default official fork self-check fails (.0857 rounded probability difference > .05). Initial OneJev5.5.4 pin incorrectly reused GroundAnything's requirement; actual checkpoint config declaresTransformers5.17.0. Update only isolated OneJev to5.17.0/tokenizers.23.3, keep all official checks and .05 tolerance; same pretrained weights then pass auto validation and use batched mode.
- Real MOT17-02 frame000302 processed on GPU6 with Choice/Noul/Score: NEW.36689/IGNORE.63311, visible people.978877, quality expected level.919935. Preserve the negative generic choice; this is a load/interface check, not association accuracy. Actual three-question request.286737s, allocator peak3,119,928,832bytes, steps0.
- GroundAnything's independent Python3.12/Torch2.9.1/Transformers5.5.4 custom serve environment completes and official ABI check reports only its expected cuDNN9.16 metadata override, no unexpected conflicts. Single DLM owner waits for verified bundle and available GPU6 memory before starting official denoising with static memory fraction.4. GroundAnything download remains active. No full online video or fine-tuning claimed yet.

## 2026-10-14 — official DLM service and GPU ownership

GroundAnything DLM download completed at the pinned official revision with
its 9,687,419,736-byte weight matching the official LFS SHA256. The single
service owner launched official custom SGLang denoising on physical GPU6
(static memory fraction .4); the model endpoint responds. A single blocking
clip owner starts four real MOT17 dev frames using native DLM candidates,
pretrained OneJev, global assignment and its own causal identity memory.
Completion, output validity and tracking quality are still pending.

The user asked why physical4/5/6/7 still showed processes. A one-time GPU
process/cwd/argv audit identified UDMT training/inference on4/5/7, the two
GroundJEV model processes on6, and an unrelated user process on6. This is
not four-GPU GroundJEV usage and not old U2 caching. Other project processes
were preserved. Keep the explicit project device allowlist and ownership
attribution; no metric, seed or protocol changed.

## 2026-10-14 — tracking SFT teacher curriculum hypothesis

Hypothesis: the pretrained generic decision model needs explicit tracking
identity, lifecycle and memory-update supervision before own-state rollout.
Prepare real 16-observation legal training clips from MOT17-02, DanceTrack0001
and MOT20-01, mined for overlap, low visibility and missing observations.
Use a single shared runtime question/state builder and randomized persistent
identity aliases per clip. Teacher GT proposals/history are allowed only in
this initial curriculum; they are not GroundAnything output or benchmark
predictions. Missing shortlist identity and withheld correct-option rows are
explicit NO_MATCH supervision, never mislabeled as NEW. Full completion also
requires verified false-candidate supervision and actual model-state rollout.

Start with official multimodal language-model full SFT, frozen vision,
100 optimizer steps, one-row microbatches and four-row accumulation. Retain
the original seed0 and5e-6 recipe, verify all rows fit the token plan, save
every25steps, inspect real gradients/loss and reload the checkpoint. The
subsequent finite first training budget remains500–1000steps including
model rollout; this100-step check does not satisfy formal training completion.
The first native online clip is unexpectedly slow, so profile that exact live
process once before proposing a speed change.

## 2026-10-14 — native video failure, official parsing and resume

The native DLM + OneJev loop completed real frames302–304:31 candidates
each,30/31/31 output tracks,18.194/19.647/19.763s per frame. OneJev
created51 persistent IDs by the third frame, evidence of unadapted identity
fragmentation, not tracking success. The fourth frame failed the strict
client parser with non-canonical coordinate ordering/repeated phrase blocks.
The runner then deadlocked during cleanup: its parent waited for ffmpeg while
ffmpeg waited for an unclosed raw-frame pipe. A one-time targeted process
diagnosis identified this, the exact owned encoder received SIGTERM and
then SIGKILL after its verified10s timeout; no model/other-project process
was signalled. The original traceback returned and is preserved.

Repeating the exact frame/request three times yielded the same strict parse
error (17.009s total); resampling is not a fix and default inference still
uses one request per frame with unchanged model/decoder/sampling parameters.
The pinned official evaluation scoring parser preserves35 geometrically
valid boxes in this response. Add an explicit official_scoring adapter policy
using that unmodified function, retaining strict_wire_valid=false, raw output
and error. A regression on the observed failure matches every label/coordinate
with the official scorer and still rejects truncated generations. Client
strict mode remains selectable; no GT, coordinate repair or metric change.

Fix encoder cleanup by closing stdin before waiting, persist the first
failure, and reconstruct completed causal identity states from saved native
Choice/Noul/Score decisions. Real prefix reconstruction reproduced all three
frames' outputs/events/state exactly without GT or further model calls.
A refactoring mistake (len(questions) after moving request construction)
caused the first resumed attempt to fail at frame305; use len(request.questions)
and retain that failed attempt. The second resume starts from the same three
verified completed frames. Resumed legacy timing cannot account for the
original failed request/interruption, so no complete end-to-end FPS is claimed.

Teacher curriculum:18 legal16-observation clips,8964rows,59,665,520 history
PNG bytes,3 real reappearance observations in DanceTrack. All8964 official
question/target schemas verified;1728 Noul targets corrected from true/false
to the official yes/no labels before any training. No development/test GT
used. Official plan-only: p50/p90/p99/max5553/7201/7637/7858tokens,0 rows
dropped at16384, maximum18options (actual API limit255). Full pretrained
language-model100-step check launched on physicalGPU4 with frozen vision;
GroundJEV now uses two physical cards4and6, other jobs preserved. The training
owner is unique and blocking. This teacher phase still lacks complete false
candidate/termination/long-gap/model-rollout coverage, explicitly pending
for the next curriculum and formal500–1000step budget.

## 2026-10-14 — first native online output and negative association control

Four native MOT17 frames completed and passed prediction/one-to-one/memory
and ffprobe integrity checks:31/31/31/35 candidates,30/31/31/35 tracks,62
persistent IDs,next_id63. Official same-candidate4-frame TrackEval: Full
HOTA.421629,DetA.650042,AssA.281686,IDF1.427481,60IDSW,16FP,24FN; Simple
HOTA.748813,DetA.642304,AssA.895161,IDF1.836502,1IDSW,17FP,24FN. Retain
this negative generic-pretraining result; it supports tracking adaptation,
not a full-video/paper claim or global NO_GO. The resumed Full invocation
allocator peak8,350,338,048bytes covers that invocation only. Complete
end-to-end FPS remains null because the legacy interrupted run lacks failed
attempt timing; Simple10.8387FPS is association replay only.

The actual100-step SFT check is running: all474weights loaded,752.39M/852.99M
parameters trainable, first5 losses finite and nonzero gnorm, peak15.1GiB.
A separate single blocking owner starts all1,203DanceTrack0004 validation
frames with native DLM + pretrained Full, then same-candidate Simple and
unchanged TrackEval. Internal GPU monitoring excludes foreign jobs.
GroundJEV uses physical4and6 only; no queue/task was restarted implicitly.

## 2026-10-14 — complete-video generic decoder failure and official dense repair

DanceTrack native attempt completed20frames then failed frame21. The raw
output repeats the same box and ends with a trailing separator; even the
unmodified official scorer returns no valid predictions. Preserve all20
real frames, the valid latest memory, failure JSON and internally recorded
resource peaks. Encoder cleanup now returns the real failure promptly.
No frame is skipped or accepted as a valid prediction.

Root cause hypothesis: generic client uses greedy inner/outer sampling rather
than the officially documented dense request profile. The running server
YAML fixes sampling_seed42; retain42. Probe the same frame with official
DecodeV4 shared fields plus dense temperature.1/top_p.95/max8192 in both
outer sampling and nested gam_dlm_decode. Actual probe3.345s produces5
official-scoring-valid boxes; strict duplicate-coordinate error still logged.
Do not relax geometry/truncation/metric validation or switch DLM to VLM/AR.
Run a distinct, explicitly registered dense_v2 full video from its first
frame because request numerics changed; the old20frames remain a separate
negative attempt. This repair is retained pending full-video evidence.

SFT owner remains live and has reached observed step40 with finite losses,
nonzero gradients and15.2GiB allocator peak. Full ckpt-step25 is saved; no
completed100-step or reload result yet. Disk check:365GiB free on/data2,
enough for the finite model saves; feature-cache budget unchanged.

## 2026-10-14 — checkpoint validation and same-candidate SFT check plan

Previous goal turn made concrete progress: native model tracking, official
four-frame baseline metrics, actual SFT and verified GitHub publication.
Current single training handle is live at observed step80; dense full-video
owner is live at observed frame14. Continue those exact owners.

After the100-step owner exits, validate its done record and finite losses/
gradients; compare saved language-weight probes against the original at the
same BF16 precision, then load the checkpoint twice and check the same real
tracking Choice/Noul/Score probability distributions. Follow with the exact
four-frame candidate replay before/after control and unchanged TrackEval.
No evaluation GT enters inference and no full-video result is inferred from
this short scope. Then create real legal-train detector/model-history rollout
supervision for the finite formal adaptation budget; teacher-only training
does not complete the target.

## 2026-10-14 —100-step negative result and complete-frame scoring policy

Official SFT completed100steps/400examples in1509.262s. All474weight names
retained; three BF16-aligned language MLP probes changed1.882M/1.623M/1.083M
elements. Two reloads yield identical real tracking probabilities (maxdiff0),
unchanged official fork auto check passes. Four same-candidate official
metrics worsen: HOTA.328338,AssA.172383,IDF1.311787,75IDSW,17FP,24FN vs
pretrained HOTA.421629,60IDSW. Keep the100-step negative checkpoint; do not
claim completed formal adaptation or improvement from a falling training loss.

Dense request full-video attempt also stops at frame21 with repeated/malformed
wrappers, despite the isolated frame21 probe passing earlier. The request
contract repair does not guarantee valid output and must not be described
as a complete fix. Preserve both failures and20 valid native states.

For complete-video measurement, register explicit empty_and_record policy:
the same unchanged official scorer rejects malformed/truncated output; use
its empty prediction set, record raw error and latency, and run the actual
causal tracker on zero candidates. Every image remains in the official
metric denominator. Mark completion WITH_REJECTED_PERCEPTION, never all-valid
model success. Only GroundingProtocolError is handled; file/service/runtime
failures still stop. Reuse the verified20-frame prefix under unchanged
model/request/core settings; no skipped sample or reconstructed fake box.
This is an explicit measurement policy, not a GroundAnything quality fix.
Analyze correct-option coverage, selected identity and memory contamination
to determine why GT-teacher SFT is not sufficient before formal rollout SFT.

## 2026-10-14 — identity error diagnosis and evidence-format controlled repair

Post-hoc fixed-anchor diagnostics on the same four frames show correct-option
coverage100% for both pretrain and SFT100. Existing identity accuracy falls
from14/78(17.95%) to6/74(8.11%); native correct-option mass.09925 to.04026.
Candidate GT recall is the same111/135(.82222) for all three controls. The
shortlist is not the immediate cause on this scope. SFT100 holds all93
existing memory observations, while pretrain has7 wrong/unmatched memory
updates and only1 identity-consistent update. Preserve diagnostics as
post-hoc analysis, not official metric replacements or inference GT input.

Two concrete exposure mismatches need repair: inference candidates have no
explicit crop identifiers, relying on pixel-coordinate arithmetic in the
full image; teacher rows use6candidates/16historical options vs16/32 at
runtime, and miss wrong detections/model histories. Preregister optional
v2_crop_motion evidence: current candidate crop sheet plus immutable
anchor/long/recent history sheet, explicit causal IoU/center-distance
matrices as features, and a genuine NO_MATCH option. Hungarian still uses
native OneJev probabilities. Keep v1 unchanged for the ongoing full-video
baseline. Run pretrain and SFT100 with identical v2 evidence/candidate lists
as a controlled diagnosis before forming the new own-history curriculum.

Evidence v2 pretrain produced no tracks: all128 candidate actions were
NO_MATCH, including the first frame with no past identity. The new criterion
was semantically overlapping with NEW. Retain that version/results and finish
the matched SFT100 control. Register v3 as a minimum semantic repair: NO_MATCH
requires a previously observed object and is not offered with empty history.
Current crop/motion evidence otherwise identical. Do not silently overwrite
v2 or alter its model scores.

Evidence v2 pretrain and SFT100 both collapse to NO_MATCH on all128
candidates (HOTA0,FN135); retain this actual semantic-error result.
V3 removes NO_MATCH from empty history and distinguishes previously observed
objects from NEW. Pretrain produces10/29/24/35tracks,HOTA.343944,47IDSW,48FN.
SFT100 produces31/0/0/0tracks,HOTA.197395,0IDSW,110FN: zeroIDSW here reflects
missing outputs, not good association. No improvement or full-video claim.
The initial16-option teacher curriculum never exposes production35-option
questions; together with teacher-only states/absent wrong detections this
requires a matched-density own-state curriculum.

Start actual checkpoint100 V3 rollouts on three legal16-observation train
clips,32GT proposals plus two explicitly synthetic box-fragment negatives
per dense frame, actual model states/decisions and no GT state correction.
Reuse the observed real DanceTrack reappearance clip. Export supervised
requests with correctly permuted IDs and crop-sheet labels,35-option
exposure where actual memory permits, false-candidate IGNORE and real
wrong-history/lifecycle/memory labels. This isolates decision learning;
GT proposals are declared, not mislabeled as GroundAnything detections.
Actual DLM-proposal training rollout remains a further required stage.

The actual SFT100 V3 student rollout completed48legal observations and
6792supervised rows,38,165,373bytes crop evidence,max35options. It has96
IGNORE negatives,139NEW labels,87withheld-option NO_MATCH cases,206
TERMINATE and2620REACTIVATE supervision. Actor after the first frame chooses
NO_MATCH and keeps lost memory, so this branch does not supply enough wrong
identity updates. Preserve that collapsed real rollout, then generate the
same registered clips with the real pretrained V3 actor to obtain diverse
wrong association/memory states. Both branches use GT proposals plus explicitly
synthetic fragments, with GT labels only in the post-hoc supervisor.
Formal adaptation will mix both branches and increase identity row weight
4x (explicit repeat sampling) because the initial curriculum overweights
lifecycle relative to association. Keep seed0,optimizer/lr and all registered
frames; do not claim a completed formal500-step run until it actually ends.

Pretrained V3 rollout completed6709rows/48observations/44,023,768bytes
and526 contaminated history observations. Combined with the6792-row SFT100
rollout gives13,501unique rows,20,881 after explicit4x identity repeats.
Official complete plan: max22817/p50 15520/p90 22158/p99 22544,0 dropped
at32768 context; actual35-option exposure. Finite500-step official full
language-model SFT began from checkpoint100 with a deliberate new curriculum
optimizer, seed0/lr5e-6 unchanged. First two losses3.2479/8.0658 and nonzero
gradients299.979/513.188; peak18.1GiB,75.8/76.2s per step. Formal completion
and before/after remain pending. Source proposals remain declared GT plus
synthetic fragments; no claim of completed GroundAnything-proposal rollout.

## 2026-10-14 — proposal-aware lifecycle label regression and resume

A targeted audit found203 wrong ACTIVE/REACTIVATE lifecycle targets:164/3015
student100 and39/2936pretrain rows label visibility despite no matching
current proposal. Lifecycle requires association, so these targets are LOST.
Signal only the verified exact500-step trainer (PID3993066, own cwd/config,
SIGTERM), preserve its first5logged steps and alloriginal artifacts. No
checkpoint was saved in this aborted stage; resume from validated100-step
weights, seed0 unchanged. Old owner state STOPPED_FOR_VERIFIED_LABEL_REPAIR.

The repair initially parsed life_357 as _357; fix the delimiter, preserve
the partial derived repair, and regenerate from original real actor traces.
The prematurely started repaired preflight correctly refused incomplete
corpora and did not train. Source corpora, allstates/media, identity aliases
and non-lifecycle targets stay identical. Assert all5951 lifecycle targets
consistent with supplied current proposals and exactly164+39labelchanges.
A new corrected500-step owner starts with unchanged weights/lr/seed/context/
identity-repeat budget. The stopped run is not declared formal completion
or silently combined into an improved result.

## 2026-10-14 — native DLM visual feature adapter plan

Previous goal turn made verified progress: real SFT/reload/negative controls,
real actor histories, lifecycle label repair/regression, resumed finite500
training and exact GitHub publication. Continue existing live owners, not
new copies. Add a native visual adapter using DLM checkpoint visual/projector
weights and the unmodified official eager MoonViT/K3 preprocessing path.
Compare the official reference visual module with the SGLang visual adapter
on the same real image, preserve feature-grid geometry through resize/padding,
then ROI pool candidates. Only this small validation uses physicalGPU5,
within the project allowlist/4card ceiling; live training4 and video6 unchanged.
This provides genuine native features for temporal adaptation/Compact, not
a substituted crop encoder or a full-dataset cache.


## 2026-10-14 — Native feature adaptation and legal task loops

Plan after source inspection: verify the actual DLM visual/projector output,
train a finite frozen-backbone temporal adapter, wire these features into
causal memory, then run legal Refer-KITTI/TAO loops with the existing OneJev
checkpoint and exact-candidate Simple controls. Keep formal SFT500 and the
full DanceTrack owner unchanged; use physical5 for new checks within4/5/6/7.

Native DLM:428,482,048 visual parameters/170weights; real MOT regions31x2560
and local tokens31x2560x3x3. Official serving visual function and immutable
checkpoint reference both match exactly; independent ROI mapping differs
9.54e-7. Initial config.hidden_size lookup failed after real forward; fixed
only the adapter to use vision_config.out_hidden_size and retained failure.

Temporal adapter:200steps, seed0,693,632 changed parameters, loss4.2086 to
.09104, nonzero finite gradients; exact reload difference0. Frozen native
backbone and decoder unchanged. Independent TRAIN calibration retrieval
131/134=.977612, identical to unadapted features (random adapter130/134).
No tracking-improvement or decoder-fine-tuning claim. Features are transient;
training-device/CPU-training regions11,612,160bytes each, calibration1,464,320.
Checkpoint receipt records absolute path,8,331,829bytes and SHA256.

RMOT:legal current v2 validation0007, single+empty8frames and multi+empty
12frames, GT-free inference/shared V1 core. Single Full HOTA0/IDF1=0/FN5;
Simple HOTA.084894/FP11/FN5/IDF1=0. Multi Full HOTA.072117/FP16/FN30; Simple
HOTA.081860/FP16/FN30; both IDF1=0. Proposal target coverage at IoU.5 is0/5
and0/30. First bottleneck is referring proposals, not evidence of an identity
head improvement. All rejected/empty frames remain in denominators. Multi
full video-processing FPS.579804 including serialization/encoding; preloaded
remote DLM initialization excluded. Single legacy frame-compute FPS.627761
is explicitly not complete end-to-end speed. Exact fixed-candidate controls
never claim detector FPS.

Evaluation first failed on GT_LOC_FORMAT, then on omitted YOLO files in empty
query frames. Set the correct path format, and look up boxes only when the
release expression has targets; nonempty targets still require real labels.
Original failures and predictions remain; completed multi Full/Simple units
were reused by a new evaluation owner. No GT boxes, thresholds or scopes altered.

TAO:registered first video40annotation frames and4frame functional scope,
all1203fixed LVIS-v1 names in96-class native requests. Failure in any chunk
rejects the whole frame explicitly; no class or sample silently dropped.
Source physical FPS unavailable, so this functional scope declares annotation
step time and does not claim a verified physical video clock. Full TETA with
all original modal GT/category/frequency metadata is required; currently pending.

Memory:real native proposals/features on2frames pass exact serialization,
anchor immutability, held-long-feature consistency and past-time rejection,
with10,682,927byte memory. V1 crops-only state remains backward compatible.
New memory adds actual region/temporal anchors, recent/reliable-long features,
16-observation feature history, native lifecycle survival probability, identity
entropy and semantic observation counts. These are used only with native input.
V4 JEV native-matrix request failed at56,426tokens on frame303; preserve it.
V5 explicitly changes numeric evidence to CSV milli-units, retaining every
candidate/identity pair and the original32768limit. Its real4frame test is
running; no performance gain claimed. Compact/full-task and retention studies
remain required. Previous goal work is progress, not goal completion.


### Completed native-feature control and measured limitations

V5 still exceeded context at34313tokens. V6 compresses native-history rows
in addition to all-pair similarity matrices; frame304 then reached34419tokens.
The smallest further fix is context-adaptive candidate grouping only when the
unchanged official native preflight rejects a group before inference. Every
proposal and up-to32 shortlisted identity options remains. The two valid
V6 prefix frames restore exact native outputs/state; resume completes304/305.
Original V4/V5/V6 failed artifacts remain, no sample skipped, no token-limit
relaxation and no retraining/checkpoint change.

MOT17 same128native candidates,4frames, unchanged TrackEval:

| Decision | HOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|
| Simple IoU | .748813 | .836502 | 1 | 17 | 24 |
| Fixed IoU + native2560cosine | .738898 | .813688 | 4 | 17 | 24 |
| Fixed IoU + trained temporal128cosine | .746248 | .828897 | 2 | 17 | 24 |
| Pretrained OneJev V6 native/crop/motion | .359328 | .330709 | 61 | 17 | 33 |

Native/temporal fixed controls use exactly the exported features, coefficients
.5/.5, NEWscore.3,32-identity geometry shortlist and16-candidate groups. They
are untrained association controls, not JEV/Compact. Temporal-vs-native is
also a dimensionality change; random128 and larger video controls remain
required before attributing the difference solely to training. Native-feature
OneJev still underperforms Simple; no isolated feature or foundation gain
claim. Existing-ID diagnostic accuracy9/75=.12, option coverage72/75=.96,
mean correct native mass.062495; candidate GT coverage remains111/135.
Its8,504,115,712byte allocator peak and.042028association replay FPS exclude
live detection; resumed timing does not support an end-to-end speed claim.

TAO first4frames have zero predictions and TETA/LocA/AssocA/ClsA0 under the
unmodified official source b498aa87b252bfb75d7cc0d8d6435c3e1e74e4de. There
are18GT boxes, source canonical class baby/person.n.01, Base. Novel is null
because this scope has no Novel GT; never report it as a Novel benchmark.
Two evaluator formatting configuration errors were repaired by restoring
its official defaults OUTPUT_PER_SEQ_RES=True and OUTPUT_SUMMARY=True;
metrics/source/GT/predictions unchanged, failures retained.

The first vocabulary run registered1203classes but stopped on a rejected
96-class chunk:192/192/192/288classes actually requested across the4frames.
This is recorded in vocabulary_coverage.json, not claimed as exhaustive
1203-class inference. A new finite owner requests ALL13chunks on every frame,
records every native outcome, and only then applies the same explicitly
registered whole-frame-rejection policy. It is active at publication.
First-prefix run processing FPS.020682 is not full-vocabulary inference FPS.
Original raw failures show many class-conditioned hallucinated boxes. The
next grounding research must address proposal quality and negative classes
through the official spatial/DLM training route, with dense/referring retention.

Visualization integrity verified8/12/4frames; KITTI display edge padded from
1242x375 to1242x376, predictions/GT coordinates unchanged. Formal500-step
SFT and fullDanceTrack original blocking owners continue. Pilot and final
unified research goal remain incomplete.


## Pre-publication RMOT GT-contract repair (authoritative results)

The new RMOT evaluator incorrectly read release top-left x/y as centers.
This was found before updating GitHub's branch. Existing established
read_gt in tools/unified/u1_formal_expression_gate.py defines normalized
top-left xywh. Corrected transport agrees for all70target box comparisons
to5.68e-14pixels. Predictions, target sets, frames, source extents, seeds,
metrics and thresholds unchanged. Original metric receipts are preserved
as invalid_center_format_metrics.json and explicitly marked invalid;
local invalid GT views remain but raw GT is excluded from GitHub.

Authoritative corrected results:

| RMOT scope / backend | HOTA | IDF1 | IDSW | FP | FN | Proposal target recall IoU.5 |
|---|---:|---:|---:|---:|---:|---:|
| Single/empty8frames Full | 0 | 0 | 0 | 0 | 5 | 4/5 |
| Exact candidates Simple | .242284 | .375000 | 2 | 7 | 1 | 4/5 |
| Multi/empty12frames Full | .194792 | .217391 | 0 | 11 | 25 | 5/30 |
| Exact candidates Simple | .243151 | .217391 | 0 | 11 | 25 | 5/30 |

The earlier zero-proposal-coverage diagnosis is invalid. On the single
query, four correct observations are present but V1 OneJev suppresses them
as IGNORE; association/birth/input evidence is the first failure. On the
multiple-target query, proposal coverage5/30 still limits every tracker.
Next controlled check should compare V1 against current-crop V3 on the
same RMOT proposals, while preserving the DLM multi-target recall deficit.
Dense/referring adaptation and original retention remain required, but
not every RMOT failure can be attributed to the perception front end.

The new all-class TAO owner remains active. A800-SXM4-80GB hardware was
verified for4/5/6; foreign UDMT processes are unchanged. Original OneJev
500step and fullDanceTrack owners remain responsible for their jobs.


## Exhaustive TAO functional control completed

All52native requests completed:13chunks and1203fixed categories for each
of4registered annotation frames. Unchanged TETA yields0TETA/LocA/AssocA/
ClsA and0predictions under the recorded whole-frame-rejection policy; Base0,
Novelnull because this video's GT contains no Novel class. No full-video
or complete-dataset result is claimed. Actual complete processing
FPS.0037577955 includes all calls, association, file/video writes and writer
completion; preloaded remote DLM startup and9s-class OneJev loading excluded.

Rejected/scored chunks per frame:11/2,10/3,7/6,10/3. Independently scored
chunks contain201/441/777/354boxes. Evaluation-only class-agnostic target
coverage is10/18 atIoU.5 before whole-frame rejection. This is a localization
diagnostic, not TETA or classification correctness. Whole-frame rejection
discards useful native outputs; retain this negative control and preregister
a query-level empty-prediction control that retains independently scored
chunks, all class requests, all GT and frames. Never label invalid chunks
valid or change the official parser/metrics. Raw native responses remain local.

At this completion check, the original blocking SFT500 owner is live, last
reported50steps/loss2.8577/20.6GiB allocator; original fullDanceTrack owner
is live, completed769of1203frames. Dataset physical clocks independently
agree with seqinfo:MOT17=30Hz,DanceTrack=20Hz,MOT20=25Hz. TAO sourceFPS
remains unavailable and annotation-step units explicitly declared.
The69-file native-feature/GT-contract stage was verified on GitHub at
1724df9de3465fce5fd1de96ccf81510029cd7f1 with exact local tree equality.
No Pilot or final-goal completion claim.


## 2026-10-14 — Next controlled task decisions

Previous goal turn:progress, with real native adaptation/task controls and
verified GitHub publication. Current project is the authorized independent
GroundJEV worktree. Plan after code inspection:fix proposals/checkpoints/data
scope, compare RMOT V1 vs V3 current-crop evidence for pretrained/SFT100;
then retain independent scored TAO chunks, keeping failed chunks empty and
all1203classes/every GT/frame. Preserve whole-frame rejection as a separate
negative control. No official parser/metric or seed changes. Reconstruct
only already-recorded native responses, with no GT access or new detector
requests; all resulting timings are association replay, not end-to-end.
A single blocking owner executes registered finite controls on physical5;
original SFT4/DanceTrack6 owners continue. Peak monitoring stays internal.


## 2026-10-14 — Registered task and inference controls

All six controls completed on A800 GPU5 with original OneJev weights or
actual SFT100. Predictions use exactly the same saved native candidates;
GT uses verified top-left Refer-KITTI release coordinates and unchanged TETA.
Official scopes remain8/12RMOT frames and4TAO annotation frames.

| RMOT / V3 | HOTA (fraction) | IDF1 | FP | FN |
|---|---:|---:|---:|---:|
| Single pretrained | 0 | 0 | 0 | 5 |
| Single SFT100 | 0 | 0 | 3 | 5 |
| Multi pretrained | .222720 | .177778 | 10 | 25 |
| Multi SFT100 | 0 | 0 | 1 | 30 |

Current crops alone do not fix cold IGNORE. MOT-only SFT100 has negative
transfer; it is not evidence of useful RMOT fine-tuning. Original finite
MOT SFT500 continues, and mixed-task/actual-proposal training remains needed.

| TAO scored-query retention | TETA (%) | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|
| Fixed native candidates Simple | 24.171 | 29.452 | 43.061 | 0 |
| Same native candidates OneJev Full | 16.815 | 27.459 | 22.987 | 0 |

Failed query chunks remain empty; independently scored chunks retained,
without re-running the detector, filtering classes/GT, repairing invalid
syntax, altering metrics or dropping frames. This is a registered combiner
control, not an output-format pass. Full remains slower and less accurate.
Both classification terms0 expose missing/wrong class decisions; the native
shared JEV class head is still required. Novel is null in this Base-only scope.
No detector-inclusive or full-dataset speed claim:replay FPS.838284Simple
and.00352281Full, with all1773label/box rows kept.

Exact-coordinate object-hypothesis grouping preserves every distinct native
box and all label alternatives:201/441/777/354rows become54/58/109/85.
No overlap threshold/NMS/topK/GT filtering. Independent Simple/Full controls
are active; first native label remains an explicitly unlearned class baseline.
A future genuine typed classification head must use the full dynamic vocabulary.

Isolated FLA0.5.2 + einops0.8.2 addon (original environments untouched):
GDN kernels2.29/3.24/8.43x faster at257/1024/4096tokens after both sides'
warmup, maxoutput difference.0004883, finite gradients. First import lacked
einops and was fixed only in the addon. First unbalanced warmup is preserved
and does not support63x speed. Actual3multimodal OneJev requests differ up
to.021211 in typed answers, exceeding the preregistered.001. Choices unchanged
but probabilities/material memory gates can change. Accelerator rejected;
original automatic fork check/.05 tolerance remains unchanged, training
and original video owners stay on the reference backend. No end-to-end gain.

Native DLM training preparation reconstructs the actual official
GAMQwen3DLM wrapper, MDM.5+causal.5/block32, with all569tensor keys/shapes
matched. The released JSON has152670AR rows; real checkpoint/tokenizer have
152671including MASK152670. Native reconstruction uses real header size; no
weight is added or randomized, no new AR download required. Initial mismatch
retained. This is a key audit, not gradient proof.

A finite3step native projector-only gradient check on legalMOT17TRAIN209/210
combines original DLM spatial loss and same-ID contrastive ROI loss. Language/
ViT frozen, projectorFP32masters, seed42, all32targets each frame with no
clipping or dropping. It is a custom native-adapter policy, not the stock full
language SFT recipe or completed adaptation. Meta nonpersistent rotary buffers
initially caused allocation failure; native Accelerate initialization fixes
that without creating learned values. The strict native cached-image schema
then rejected a path-only dict; real original bytes/path are now supplied.
Both failed units preserved, new real-gradient check active onGPU7. Four
physical GPUs4/5/6/7max; foreign UDMT jobs untouched. Dense/referring retention,
exported bundle inference and formal adaptation remain required.


Exact-geometry controls completed:Simple TETA33.042/LocA48.539/AssocA50.587,
Full23.320/49.095/20.866, both ClsA0. Simple replay9.8226FPS, Full.0197772,
about5.6x the ungrouped Full. No detector-inclusive speed claim, no learned
category assignment and no Full advantage. Native geometry set/all label
alternatives retained; every original GT and frame evaluated.

Native3-step adaptation then exposed projector output as a per-image tensor
list, which was corrected using native concatenation. The first native
backward hit CUDA-graph lifetime replay; first failed units remain unchanged.
The next unit compiles the identical native FlexAttention operator without
CUDA graphs and probes native spatial-loss and temporal-loss gradients
independently before optimizer updates. Seed, block masks/loss weights/GT
and all target instances unchanged. PreserveFP32nonpersistent RoPE buffers.
No native decoder/localization adaptation success is claimed until actual
finite loss/gradients, changed BF16weights and checkpoint reload are verified.


### Actual native DLM spatial and temporal gradient proof

Final v6 native training unit completed3optimizer steps on legalMOT17TRAIN
209/210, all32targets each frame and32same-ID positives. Official native
GAMQwen3DLM MDM.5+causal.5/block32 retained, with.1identity ROI contrastive
loss. Language and native ViT frozen;27,267,584native projector parameters
useFP32masters. Independent dense-gradient norm8.75947 and temporal-gradient
norm2.13338, finite combined norms8.77141/9.25557/9.02194. Native losses
1.61708/1.66780/1.65477 include different sampled noise; do not claim training
improvement from this warmup.26,766,531FP32elements and1,968,676BF16elements
changed. Delta reload weight difference0,327,218,423bytes, SHA256
bf8537a97675e465d80dfb08cb5b21454ee37711f622510fe8181b86370e2b3e,
peak18,455,674,368GPUbytes. This proves actual localization and identity
gradients, not formal adaptation or tracking gain.

Checkpoint:/data2/user/LocateMOT_GroundJEV/outputs/groundjev/ground_adaptation/
native_tracking_grad3_v6_native_runtime_flags/projector_delta.pt. Frozen
nonpersistent rotary buffers remainFP32. Native official CUDA-graph flag0
and task-local donated_buffer=False accommodate checkpointed backward and
independent gradient probes; math/masks/loss weights/seed42unchanged.
Failed meta-buffer/image-schema/list-output/CUDAgraph/donation units retained.

Exported complete DLM variant models/GroundAnything-DLM-TrackProj3 replaces
only5native projector tensors, retaining all569keys and every other native
weight/tokenizer/code/config unchanged.9,687,419,704bytes, SHA256
a626aad29c5bbc17e3a465a519e93681aaedacd81cbb21805cce40b7122777e9.
Original model/server untouched. A distinct GPU7replica/8102 and launch bundle
are used for48same-profile native frames:4dense+8single-ref+12multi-ref,
original/updated, fixed Simple association. Requesteddecoderseed42, same
unchanged official parsers and evaluator. These functional retention probes
do not stand in for the full original GroundAnything benchmark suite.
The one blocking owner stops only its owned adapted replica on completion.


Adapted replica first launch passed real weight loading butKV profiling at
mem-fraction-static.25produced a negative cache budget. Owned replica
was stopped and failed launch preserved. New isolated replica uses.4(the
original service's working fraction), same checkpoint/decoder/seed/scopes.
No other process stopped and no GPU beyond4/5/6/7.


### Verified native projector retention controls (2026-10-14)

All six fresh inference and official evaluation units passed scope and encoded
video frame checks:48frames total, no runtime GT or future access, every
rejected/empty frame retained. Original versus projector3 HOTA/IDF1 fractions:

| Scope | Original HOTA | Projector3 HOTA | Original IDF1 | Projector3 IDF1 | IDSW before/after |
|---|---:|---:|---:|---:|---:|
| MOT17 4frames | .745083 | .755029 | .824427 | .847328 | 3/1 |
| Single RMOT 8frames | .299355 | .303853 | .375000 | .400000 | 1/1 |
| Multi RMOT 12frames | .229517 | .227592 | .222222 | .212766 | 0/1 |

Both single variants reject frame152; both multi variants reject frame34.
These frames remain in official denominators. Decoder tracking_dense,
requested seed42, official_scoring parser, unchanged geometry Simple and
legal validation scopes are identical. Short dense and single controls
improve slightly, multi-referring declines. This is not formal adaptation,
full original grounding/referring benchmark retention or a broad gain claim.
Actual detector-inclusive FPS are in aggregate_results.json; concurrent
workloads and original GPU6 versus replica GPU7 preclude an isolated speed
comparison. The first replica failed KV allocation at mem-fraction.25;
its failure is retained. The completed replica used.4 and stopped its own
process group. Original server/model and foreign UDMT are unchanged.

Authoritative receipt:
outputs/groundjev/ground_adaptation/projector3_retention_v2_memory_budget/aggregate_results.json.
OneJev formal500step owner remains live, latest returned report100/500;
full1203frame DanceTrack owner remains live, latest returned frame1183.
No completion claim for either. Current persistent feature service on5
uses about4GiB; no GroundJEV process remains on7 after replica cleanup.
Next research hypothesis: exact object geometry grouping improves localization,
but copying a native query label…48444 tokens truncated…；完整latest4500已验证但未来5000会原子替换，编号BF16权重保留。

全量CPU completion核验计划：原94290已先通过serial full plan并进入FULL_JOINT_FOUR_GPU_TRAINING，原17189实际返回同阶段等待更新；并行49885后完成actual exit0，coordinator2858 actual exit0/status ORIGINAL_ALREADY_ADVANCED_NO_HANDOFF，无任何signal/重启。比较两份completed native plan全部summary（除不同计时定义），核验int32 459027长度及int64总和/原生global16同seed0计划全行守恒和tail3；读取原RMOT source refs/group first-media，追溯raw1814与semantic1796实际曝光差18，保留全部raw/原行，不更改正在训练数据。报告与已完成lengtharray沿原4MiB报告预留，不新增GPU/feature档案；实测CPU计划tokens5639441074/28690updates是计划，不是已消费tokens/训练结果。

原35297 actual exit0：serial与4CPU全部完整数值summary相同，459027长度array int32/全int64总5639441074核验，原生index-shuffle/global16恰28690updates、末step rank行数1/1/1/0，全部行一次不漏不重。native allreduce_grads原实现为无本地梯度参数补零再SUM，末rank空不丢样本。实际输入首媒体MOT1056/RMO1796/OV17023；RMO raw1814的18张差额在两query原保留监督中都仅有lifecycle，登记semantic-only不选这些proxy，原始帧/row完整保留，不造新presence标签、不改当前数据。原serial CPU计时7943.0s仅capture后全plan；4CPU2415.9s含加载及192probe，不作严格受控速度基准。原四卡owner94290仍保留，原17189已实证阶段切换但首5实际optimizer仍待观测；无handoff/重启/信号，三任务普通水平未证明。

原17189 actual exit0：first5四卡真实训练已验证，actual step1..5/seen16,32,48,64,80，MOT49/RMO12/OV19；5条loss/CE/Brier/gradient全finite且gradient>0，原生600初始化320FP32/752393024参数maxdiff0，exact4rank分别在物理4/5/6/7，原owner94290未重启。新phase28690步/459027问题/vision frozen/seed0/确定性FA，非running6000续训、非Ground-One联合梯度、非长视频BPTT；首5仅启动证明不是完整epoch/跟踪收益。前5实际global输入吞吐约13.1k..15.0k tokens/s、每步9.7..19.3秒，仅极短早期/同GPU共用记录，不作稳定速度或结束时间预测。原6000另已至少5300updates，5000完整9029140633byte恢复state/save receipt与编号BF16存在已核验；不重复4500已完成的全部数值扫描。更新TRAINING_LOG/CURRENT_STATUS顶部，旧active状态标历史，保存first5实际args和5000save小receipt；沿原observer1MiB及既有future-curriculum metadata预留，0新feature/model/optimizer。所有tracking终评与ordinary-level目标仍未完成。

连续观测计划：原四卡训练实际first5已完成并发布，但94290原shell句柄不转发逐步stdout，实际日志在training/log.jsonl。新增单一blocking只读跟随器，核对原owner PID/starttick/cwd后用pidfd内部等待，持续转发新完整JSON日志/phase变化并检查loss/gradient finite；从已证明step5之后接续，不重启训练、不写新feature或模型、不进行agent-level ps/tail轮询。优化器done仅表示训练步骤结束，最终reload/跟踪指标仍交原owner校验。

原76060单一blocking follower实际接上原94290的同一训练，立即转发实际step25/400rows，loss2.044512/grad208.539均finite，最近窗口12.097s/update、14576.845global planned-input tokens/s；原样记录共用GPU的极早期样本，不当稳定速度/结束时间/推理E2E或质量收益。原日志原样读不改，first400实际seed0 sampler task counts={"mot": 203, "rmot": 98, "ovmot": 99}；注册follower句柄供后续直接write_stdin，直到optimizer done交原parent完成reload/scorers，0新增模型/feature/optimizer。

6000完成核验准备：复用已实际验证4500的CPU完整状态/曝光freezer，仅将结果字段拆为complete6000_training与new_tracking_metrics_available，避免6000实际结束时仍把二者合并标false；不改验证门槛、训练、采样或metric。原owner负责保存全部6000 FP32/Adam/RNG、final BF16与native-MM reload后固定四scope对照。待原阻塞训练输出完成/阶段事件后才运行CPU验证，不重复扫描仍在进行的5500状态。

Compact扩展最小计划：现有共享Compact只有48MOT/72RMO/64OV小pack，而已有43MOT1056帧+18RMO3628query观察的真实Compact actor原生Full-v3导出，输入内含确切pre-write geometry与anchor/recent/long像素来源。做只读FP32引用loader：按source archive(frame,candidate)重建当前与三份历史2560向量，保存小geometry和reference元数据，不重复保存每group约1MB历史向量；原监督行独立保留，assignment/opaque alias只作真实输入条件不引入GT。先CPU检验与原Compact.forward实际输入逐tensor相等、source outputs/states一致；保留全部空/拒绝帧，空n0/m0明确原Compact不调用forward，不能伪称新增presence head。此为未来三任务Compact扩展输入组件，175类ROI与新train recipe尚未准备/启动；0GPU/optimizer、仅32MiB metadata预留，feature upper26618307550<25GiB。当前两轮OneJev训练及测评不变。

Compact input组件原35764 actual exit0：全部61unit/4684query-frame contexts、7071ordinary输入group与5870真实assigned条件，45空n0m0group保留；MOT1056/RMO1814unique物理图。仅21265696byte metadata，FP32 arrays全部引用原归档不复制；8920次重复box引用在真实数据中逐值确认原F32向量相同，未猜测匹配。原85147 actual exit0：3真实source各前3帧、36实际forward groups/22带历史、6784728输入tensor元素torch.equal，所有native模型forward/actual source outputs-state完全相同，0GT修正/0GPU/optimizer。全61数据准备与九帧数值regression范围分开记录，仍不是完成三任务Compact训练pack，OV175native features未采集、新model训练未启动；后续共享训练需补齐该组件及独立登记recipe。完整21MB可再生metadata cache server-only，发布loader/builder/validator/config/完成证据，不把缓存当tracking结果。原两个OneJev训练和所有评测继续，goal未完成。

Compact OV175补齐计划：仅复用原正跑24k课程中已登记4000个root对象draw及其leaf对应，不另选GT难度/类别或读eval/Novel标签。CPU逐一核对train-Base annoID/vocab index/image/GT bbox与原crop recipe；去重观察向量按annotationID索引，保留rare-class原draw重复。新私有GPU4 ROI-only视觉服务与现有任务共用4/5/6/7总4卡，不停止任何既有进程，原生MoonViT/projector无语言解码/新DLM请求，RPC只给原图+框。单一FP32memmap bank约36MiB及逐图完成记录支持未完成单位恢复，无新PNG/整集缓存。64MiB预留使upper26685416414<25GiB；此组件不启动Compactoptimizer，头/采样recipe另登记。

## 2026-10-16 — shared6000 completion and joint-training scope

Balanced shared OneJev MOT/RMOT/OVMOT training completed6000updates/24000questions (8000per task),29432.9s. Final full FP32/Adam/RNG state validates all752393024parameters and1504786048moment elements finite, all320Adam counters/scheduler/cursor6000; native multimodal repeated reload probability difference0. Retain final and numbered checkpoint; downstream controls have started and tracking gains remain pending. Actual first-media paths:1056MOT/72RMOT/3227OVMOT, distinct from question counts. Independent four-GPU full-source epoch latest observed425/28690updates,6800questions,finite loss/gradient. Ground and OneJev are separately optimized; no end-to-end joint foundation gradient or long-video BPTT. Evidence: outputs/groundjev/training/shared6000_actual6000_snapshot_v1/validation.json.

## 2026-10-16 — broader native Compact supervised adaptation plan

Hypothesis: the previous Compact corpus (48MOT/72RMOT/64OV images) limits transfer; first isolate broader data with the unchanged semantic Compact architecture and existing runtime. Build a fresh explicitly registered question-level curriculum from43MOT native units,18RMOT native query units and the selected175Base native teacher ROI bank. Route only supported original heads (identity/observation-update/reliability/lifecycle/Membership/flat category); inventory unsupported presence and final-ID questions rather than relabeling them. No RMOT unknown-birth auxiliary labels or OV teacher identity/update targets are invented. Train one common final checkpoint on CPU, retain original actor as before control, and run unchanged causal evaluation before judging gains. This is supervised Compact adaptation, not native OneJev decoder reasoning or a completed large-scale epoch.

Compact curriculum first target validation failed onMOT20-05_766_batch0_match_13: no-history native questions have onlyNEW/IGNORE, while existing Compact keeps a third hard-maskedNO_MATCH logit. Encoded target already correctly padded zero; fixed only the validator legal-option expectation, preserving all source labels/questions and the existing model. Failure receipt retained. Base175 native-input validation meanwhile passed all175class representatives/216205500tensor elements with exact inputs and forward outputs.

Broader Compact curriculum now concretely prepared and all24000selected labels checked against original native questions/option order. Balanced MOT8000 (5000identity/2000life/500observation-update/500reliability), native RMOT8000membership (4000yes/4000no), OV8000flat-class exposures preserve the4000originalBase draws twice. Actual eligible first-media coverage1053MOT/1795RMOT/3227OVimages; all226query-frame contexts without a selected supported head explicitly inventoried, original1056/1814source frames preserved. This new question-level balanced recipe also changes sampling/loss aggregation, so resulting comparison is not an isolated data-only ablation. Real three-task8-update continuous versus4+4restored CPU trajectories match every model/Adam/counter/RNG value exactly. Registered8000updates/24000questions unchanged-architecture training and subsequent five unchanged causal controls launched onCPU, no additional GPU. Native OneJev large epoch and6000model controls continue separately.

Broader Compact actual first100updates/300questions verified from its complete log and live PID/starttick, all first logged losses/gradients finite; this is real training, not only registration. Single Base bank component42,597,901bytes<64MiB and curriculum metadata547,051bytes<16MiB; registered caps retained conservatively. CPU before/after scope protocol uses nonoverlapping MOT20train766–893 versus validation1659–3315. Final8000 and five-scope scores pending; no training-gain claim.

Frozen fit affinity diagnosis (1250selected unique correct-existing questions) completed before training: native2560cosine retrieves correct listed identity1050/1250=84.00%, projected128cosine1026/1250=82.08%, IoU986/1250=78.88%, learned existing-only logits893/1250=71.44%; adding existing action competition leaves521/1250=41.68% full correct decisions, with589NO_MATCH choices despite a correct existing option. Thus this fit evidence points more strongly to association/action learning than a wholesale loss of appearance in compression. It is neither held-out accuracy nor causal tracking gain. A read-only owner waits on the verified original Compact trainer PIDfd, then repeats the identical frozen cases using its final checkpoint; no running recipe/runtime changed.

Shared Compact8000 completed24000questions/1167.332s (19.46min),1,146,511parameters,1,139,601changed elements. Full model/Adam finite and export exact; three-task reload exact. Fit1250 existing-ID full correct521→656 (41.68→52.48%), but first held-out DT1203 control strongly regresses: HOTA26.3157→5.5800%, IDF130.6329→6.7840%, IDSW165→488,FP579→1511,FN1962→3637. Preserve negative model; no promotion. Remaining four scopes continue unchanged. Register a separate matched MOT-only8000 update control with the exact original MOT row sequence, sameinitial/Adam/LR/weightdecay/clip and loss/3, excluding only RMOT/OV losses by explicit opt-in. This separates task-gradient interference from the changed data/sampling recipe, without claiming FLOP matching. No new features/GPUs.

Remaining first-three-scope results: RMOTsingle800 HOTA6.0247→1.0579%, IDF12.8902→0%; RMOTmulti800 HOTA19.8019→12.5237%, IDF121.0082→13.1709%. All are negative; retain completed joint8000 model and all five registered controls. The separately declaredMOT-only8000 control has actually passed100updates/100questions; its8versus4+4 full model/Adam/RNG restore is exact. Optional task-subset implementation preserves the original three-task8update model/Adam/RNG/protocol/exposure bit for bit.

MatchedMOT-only8000 completed8000questions/944.519s, full-state finite/export/reload exact; zero category/membership head changes verified. It also strongly regresses DT1203: HOTA26.3157→1.67995%, IDF130.6329→0.82034%, IDSW165→1303; singleRMO HOTA6.0247→2.49096%. Fit correct-existing733/1250=58.64% remains better than original41.68% but does not transfer. Cross-task gradient interference alone cannot explain failure. Registered targeted DT1203 causal intervention for both trained checkpoints: association unchanged, fixed auxiliary policy (write1/reliabilityNone/no learned lifecycle), retain all default negatives and entire pending five-scope studies. This diagnoses the combined auxiliary policy, not a lifecycle-only ablation or successful replacement model. JointTAO40 separately improves TETA14.517→19.252 viaLocA8.088→17.963 andAssocA5.1478→11.012, whileClsA30.314→28.781 worsens andNovelClsA stays0.

Posthoc DT1203 auxiliary-policy intervention completed5arms: joint fixed-auxHOTA5.6588% versus5.5800default;MOT-onlyfixed-aux3.2215% versus1.67995default, both far below original26.3157%. Combined auxiliary replacement alone does not repair association. Original43-sourceMOT labels measured: EXISTING22117/48051(46.03%), NEW11489,IGNORE7992,NO_MATCH6453; lifecycleACTIVE60077/93624,TERMINATE4682/93624(5.00%),LOST12117,REACTIVATE16748. Selected uniform label quotas impose25% each. Register an exact-row/initial/optimizer-matched MOT-only8000 loss-weight control using TRAIN source-frequency/selected-frequency ratios, loss/3 and all existing policies/metrics unchanged. This is an exploratory source-prior hypothesis, not guaranteed deployment calibration; no validation label ratios used.

TRAIN-source prior-weightedMOT-only control actually passed100updates/100questions, finite loss/gradient, exact8vs4+4model/Adam/RNGprefix validation passed. JointMOT20full1657 inference finished; a single resource diagnosis confirms original official evaluator actively running onCPU (491CPU seconds/~60.3GiBRSS at observation), not a stopped model job. Keep original scorer and all frames unchanged; no finalMOT20metric yet.

Next focused diagnosis: replay all1203DTvalidation original CPU actor frames exactly and evaluate original/joint/MOT-only checkpoints on identical captured pre-write input tensors. Score identity argmax and a shadow single-step global assignment using external GT/immutable old birth IDs only after inputs are fixed. Exact original outputs/states and shadow-choice parity required on every frame; no new causal score, training or feature pack claimed. This separates conditional decision failure from changed closed-loop history without GT state repair.

DT1203 frozen-prefix diagnosis completed1203frames/13664candidates; every original output/state and shadow global assignment exact. On1768correct-existing options, original argmax correct651, joint6, MOT-only55; joint predictsIGNORE on1726/1768, MOT-only1527/1768. Thus severe conditional rejection exists before any new-policy history change; exposure bias alone is insufficient. Overall argmax accuracy is dominated by IGNORE and is not a tracking metric. Audit TRAIN and validation birth-label support separately: determine whether all-proposal Hungarian matching leaves actual unique GT-supported emissions unbound, and count isolated valid IoU edges lost by assignment-before-threshold. Do not relabel or alter any running study during this diagnostic.

Birth support auditv1 finished all43train clips and DT1203 validation computations, then exited1 because final json.dumps call used an unsupported keyword. Saved computations and failure receipt retained. Fixed only final summary dictionary construction and rerun the identical read-only audit into freshv2; no labels, frames or acceptance criteria changed.

Birth supportauditv2 exits0: only2/844TRAINbirths have unbound, uniquely supported actual emissions; assignment-before-threshold loses4isolatedvalidTRAINedges. These rare verified cases cannot explain the broad collapse, so do not relabel corpus or interrupt original four-GPU training. SharedFullOneJev6000 defaultDT1203 also regresses HOTA8.7030→7.8406%, IDF15.8260→3.5341%, IDSW3631→2796; assigned-ID samecheckpoint comparison continues. TRAIN-prior-weightedCompact8000 completes8000questions/944.302s, DT HOTA2.5567%, singleRMO3.5421%, multiRMO15.7803%; prior correction alone does not repair. Measured selectedMOT source mix5934MOT20/779MOT17/1287DT, but correct-existing questions496/255/499. Selected correct-history age/100 median.06,max1.23 andconfidence median.592. Register frozen old-prefix weighted-model plus GT-agnostic zeroage/zeroconfidence/zero4scalar interventions and actual input-domain quantiles; no continuous tracking gain or changed default protocol.

Scalar interventions fail to restore conditional identity decisions: weighted model correct3/1768, noage43, noconfidence1, no4scalars26 versus original651. Real correct-alias age/100 median2.03,max8.46 exceeds selectedTRAINmedian.06,max1.23, but erasing scalars alone is insufficient. Corrected an inherited report flag that had claimed all tensors identical despite explicitly declared scalar interventions; source plans/metrics/decisions unchanged, original flag retained in scope-correction receipt. Next implementation: frozen-core multitask adapter with zero-initialized learned native ROI cosine/IoU/distance residual for matching/actions; train only small residual and category/membership heads, preserving original backbone, text-conditioning and auxiliary/association weights exactly. Use identity-onlyMOT plus nativeRMOTmembership/BaseOVcategory, same one final model and an inference residual-disabled ablation. This is a new Compact adapter study, not a native OneJev decoder or trainable shared-backbone claim.

Frozen-core native-relation adapter now concrete: total1,147,379parameters,66,790trainable (868native pair/action residual plus category/membership/scale). New output layers are exactzero; wholeDT1203/13664native candidates reproduce every original output/state/shadow assignment and every untrained model tensor output exactly. New24000-question corpus labels/options validated (8000identity/8000membership/8000Basecategory), all263unsupported query-frame contexts explicitly retained in source inventory; actualeligible images1016MOT/1795RMOT/3227OV. Real three-task8continuous versus4+4restored full model/Adam/RNG is bitexact. CPU8000update production and same-final residual-disabled inference ablation/five-scope controls launched; original backbone/text/association/auxiliary weights frozen, no newshared-backbone or OneJev decoder-gradient claim. New8MiBmetadata reservation raisesconservative temporary upper26710582238<26843545600; no newROI features or GPU.

Clarification: birth-support audit counts only provably unique GT-supported emitted cases. Remaining ambiguous original unbound births are not certified correct, so the broader binding-label mechanism is not ruled out. Keep original data and report uncertainty. Shared/MOT-only8000 full five-scope controls completed; sharedMOT20 improvesHOTA7.4316→11.4780% butIDSW2060→58632, Monly5.6407%; no model promotion.

Native-relation production has actually passed100updates/300questions (100per task); intermediate saved-state validation compares every frozen original weight exactly and checks all model/Adam values finite. Initial parity and prefix restore are execution evidence only; no new tracking gain yet.

TRAIN-prior8000 also completed allfive causal scopes; naturalMOT20 HOTA3.10833553628264%, IDF11.890913979096887%, IDSW30884. Frozen-core native-relation actual saved state observedat3000updates/9000questions: all59original fixed tensors/1080589elements exact, allmodel/Adamfinite. This verifies execution/frozen-boundary only, not gain.

Registered one blocking CPU observer for original full-source four-GPU1000step save; original ownerPID/starttick/cwd confirmed, no processes signaled/restarted. It waits for the actual full-state receipt, then checks all752393024FP32masters/1504786048Adam elements,320counters/scheduler/cursor,all4rankRNG payloads and actual sampler task exposure. It performs no optimizer updates, GPU allocation or actual1000step restart; report pending. Small report reuses1MiB of the already registered8MiB adapter/report reserve.

Native relation8000 completes24000questions/1040.460s. All59fixedweight tensors exact, final full state/export/reload pass. Residual-disabled inference restores originalDT HOTA26.3157% exactly; trained residualDT15.3059%, singleRMO0%, multiRMO24.1906% (disabled19.1828,original19.8019), multiIDF129.2044% andIDSW97 versusoriginal21.0082/165. Positive isolated multi-query component effect but overall regression remains; retain both same-final variants and finish registeredTAO/MOT20 scopes, no promotion.

2026-10-16 — Evidence milestone, not efficacy: original four-GPU459027-question joint training actual1000/16000 save passed full752393024FP32/1504786048Adam-moment finite/cursor/RNG inspection. Task exposures8817/3537/3646; subsequent1125/18000 logged. Preserve training and checkpoints; no Ground+One joint gradient/long-BPTT claim or quality promotion.

2026-10-16 — Next hypothesis/registered implementation plan: actual old Compact uses task/query embedding inside the physical association transformer; semantic-only RMOT/OV supervision can therefore leave physical decisions on a poorly supervised task-conditioned branch. Implement an opt-in canonical MOT physical stream (same original person/MOT text and weights), retaining the true query/task only at semantic readout. On identical candidates/history, physical logits/update/reliability/lifecycle must be query invariant and MOT outputs must remain exact before any training. Then run paired frozen-physical semantic-head-only8000-step studies, identical16000RMOT/OV questions/order/seed/loss/hyperparameters, query-coupled versus canonical physical branch; include untrained canonical arm to separate runtime intervention from semantic training. All5 registered complete scopes and scorers, same final checkpoint across tasks, no residual or threshold changes. No new ROI bank/GPU; reserve8MiB metadata beyond26710582238→26718970846bytes<25GiB. This is a factorization experiment alongside the original running750M joint training, not proof of shared-backbone or Ground+One end-to-end gains.

2026-10-16 — Factorization initial evidence:67real groups/61native units+6Base,335conditional task/query interventions; physical outputs exact invariant, original physical outputs change335/335, membership changes335/335. Every canonical MOT output exact original. Full1203rollout/prefix/training remain owned and pending. Native residual TAO40 all3arms officialvalidated: TETA14.517→14.320, disabled14.646; allNovelClsA0. Preserve mixed result; no promotion.

2026-10-16 — Factorization full MOT numerical regression complete:1203frames/13664candidates, every original causal state/output/shadow Hungarian assignment exact, every untrained canonical output tensor exact original on actual pre-write inputs. No metric improvement inferred. Paired semantic-only resume checks now run under original newCPU owner3908189.

2026-10-16 — Both matched frozen-physical semantic-head studies passed actual8continuous versus4+4restored model/Adam/RNG equality (16questions each), then owner3908189 launched registeredcoupled/canonical studies3913160/3913161. Physical-weight preservation and allfive officialcontrols remain required. Native750M joint training continues independently.

2026-10-16 — Native750M shared6000 final-ID DT1203 complete: official HOTA7.840637%/IDF1=3.534137%, exactly same as same-checkpoint default (original600 HOTA8.702996%). Real assigned-ID second passes1350, existing assignments4940, changed update probabilities4940, registered.5gate flips0; all1203exported outputs exact=True. Second-pass execution verified, no DT gain; RMOT/TAO controls continue under originalowner.

2026-10-16 — Actual matched head-only8000training completed botharms:16000semanticquestions each,115.290/116.032s,65922trainable/1146511totalparameters. All59fixedoriginaltensors1080589elements exact; fullmodel/Adamfinite andexport/reloadexact. Preservefinalbothmodels; fullfive-scopecontrols nowrunning, no effectivenessclaim. This pairedphysical-queryintervention is distinct from ongoingnative750Msharedtraining.

2026-10-16 — Next physical learning hypothesis: native OneJev6000 DT association remains weak and final-ID secondpass makes no gate changes; frozen physical/query factorization alone preserves MOT but does not restore single-RMOT. Train a task/query-independent native appearance projection with supervised cross-frame contrastive identities directly from immutable TRAIN candidate registry GT matches, bypassing actor birth-alias/action labels. Use all43TRAIN units/1056frames; keep original labels/sources, explicitly inventory singleton/noncontrastive identities. Same-video8identities×up-to4distinctframes,1000fixedsteps/seed0, original2560D ROI vectors by reference only. Evaluate native cosine, random initial projection, final learned projection under identical existing.5IoU+.5positive-cosine fusion/NEW.3/IGNORE0, fixed auxiliaries/gated3s, original semantic heads and all5official scopes. This representation-learning control is not native JEV probability, end-to-end Ground+One training or long-video BPTT. Reserve48MiB (16MiBmetadata+32MiBbounded streaming working buffers) beyond26718970846→26769302494<25GiB; model/Adam underdata3, no new ROI/image bank or GPU. First validate actual8vs4+4checkpoint resume, then train/fullcontrols; final1000 selected regardless heldout metrics, no threshold tuning.

2026-10-16 — Physical-factorization first4officialscopes complete:DT HOTA26.3157exactallarms; coupled/canonical singleRMOT0/3.1383vsoriginal6.0247, multi19.1420/12.0955vs19.8019. TAO40coupled14.871/canonical31.323vsoriginal14.517/untrainedcanonical26.125; canonicalLocA33.075/AssocA35.637samebeforeandaftersemantictraining, ClsA9.6637→25.259(stillbeloworiginal30.314), NovelClsA0. Keep isolatedOVassociationgain, rejectthree-taskpromotion. Newcross-frameprojection1000complete341184params/27797exposures/36.587s, all43units;887TRAINIDs/873crossframe,14singletonspreserved; actualresumeallstatebitexact and67group16572affinityformula/defaultsemanticregressionexact. All5fullcontrolsrun unchanged. Native residualMOT20HOTA19.5518/IDF1=17.4310%, IDs30874vs2060; all5enabledscopescomplete,lastdisabledablationpending.

2026-10-16 — Completed original native residual15/15officialcontrols, finaldisabledMOT20 restores original metrics exactly; preserve allcompletedproof. Next narrow causal audit: external frozenTAO40 native-versus-learned category evidence using exact official annotation+fullLVISsynsetaliases; distinguish any-box/class oracle support from class-agnostic one-to-one emitted support. An initial ad hoc print used annotation-onlyaliases and is not accepted as equivalent to official mapping; complete registered audit uses both dictionaries without changing predictions or metrics. No native-label restore is deployed before this evidence.

2026-10-16 — Native temporal projection DT1203 official4arms: originalHOTA26.3157%; fixednativecos13.3108%, randomprojection13.4881%, trainedprojection8.4607% (IDF1=3.2767%). No promotion, remaining4scopes continue. Frozen category audit with exact officialaliases: Novelboxsupport37/40 but nativecorrectlabel+boxsupport1/40; emittedGTmatchedNovel6 and0native/learnedcorrect labels. Restoring rawnative labels alone is unsupported. New foundation hypothesis/plan: many original43MOT train clips cover~0.75–2s (MOT20 up to5.08s); longer supervision may reduce appearance shortcuts, but causation is not established. Build paired short-versus-long GT teacher identity episodes from all legal frozen train_frames only, same current image/box, same randomly aliased history identities/options/targets and other24000-mixture rows; past observation interval changes.4000positive identity teacher rows +4000originalnativeMOT rows +same8000RMOT/8000OV rows perarm. Teacher snapshots remain provided across gaps; not realactor rollout, long-BPTT or a deployedTTL/archive change. Actual native request/crop recipes/CPU plans required, then independent shared750M new stages from same native6000 FP32 checkpoint onGPU7 only after original6000owner exits successfully. Matched final6000steps/default4scopecontrols perarm; no old job restart or protocol change. Reserve16MiB reference/plan metadata:26769302494→26786079710<25GiB; no new images/ROI bank, model/Adam separatelydata3.

2026-10-16 — Paired temporal teacher preparationv1 failed its final16MiBmetadata cap:22,748,803bytes, dominated by two10.14MB copies of the immutable9.96MB source mixture. No GPU/newoptimizer update started. Preserve failedinputs/code/terminalreceipt. Smallest fix: add explicit indexed-source manifest protocol and reference originalselected20000rows instead of copying bodies; newteacher4000refs remain. V2 keeps individual16MiBcap, separately reserves24MiB for failedinputpreservation; globalupper26769302494+40MiB=26811245534<25GiB. No samples/targets/seeds/scorerschanged.

2026-10-16 — Indexedv2 ordering/provenance amendment: original row IDs are preserved;20000common native/semantic rows precede4000matchedteacher refs, then native seed0indexshuffle. This differs from failed/untrainedv1 interleaving/renamedIDs and is explicitly registered; botharmssameorder/augmentationIDs. No v1gradient trajectory existed. Mandatory regression checks actual8legacyrows and3indexedrows exact before complete native24000CPU plans.

2026-10-16 — V2pairedtemporalteacher prepared/fullyCPUvalidated:43TRAINvideos/9789selectedimages,4000matchedpositivesperarm+4000oldnativeMOT+16000fixedsemanticrows. Shortgaps[.0333,.5,.96,1]s; long[3,7.95,38.65,104.2]s.86nativegroupstates/questions/172pixelsheetsexact; all24000rows/0drop/max27319,27319/6000plannedupdateseach.0newgradsteps, originaloneblockingownerwaitingnative6000GPU7success. Preservev1failedcopiedmixtures22.75MB andcode; indexedv2actual2.85MBwithin16MiB, separate24MiBfailedpreservation, global26811245534<25GiB. Native residual15/15andfactorization25/25officialcontrolscomplete; no common3taskpromotion. CategoryauditpeeleronlyNovel:37/40boxsupport,1/40box+label,0correctof6emittedGTmatches; externaloraclecountsnotClsA, no native-labelrestore. ReID12armsfirst3scopescomplete; trainedDT8.4607HOTA<13.3108nativefixed<26.3157original, no promotion.

2026-10-16 — Correct mixed projection evidence after fullRMOTcontrols: singleHOTA9.2325vsoriginal6.0247/nativefixed5.5377/random6.6298; multi31.5524vs19.8019/16.6224/14.3821, IDF1=40.0107vs21.0082%, IDSW53vs165. Keep representation as a candidate, no promotion dueDT8.4607. FixedaffinityactionNEW.3>IGNORE0forcesemissionofallrawproposals(FP9330vs579); trainedAssAalsofalls(native.06641→trained.02618). Next planned implementation: appearance-informed permutation of existing-ID probability ranks, exact original action probabilities/probability multiset and auxiliaryheads retained on identical inputs; allfive scopes, native/random/trained contrast controls, no heldout temperature tuning(use originalcontrastive.07). Not yet implemented/measured.

2026-10-16 — Implement next preregistered physical control: trained appearance helped bothRMOTscopes but fixedNEW.3/IGNORE0forcesallproposalemission andDTassociationalsoregresses. Preserve the original trained Compact action probabilities, complete probability multiset, all auxiliary/semantic outputs and policies on identical inputs. Reorder only existing-ID probability ranks by log(originalIDprob)+mean(anchor/recent/long cosine)/.07; .07 is the existing contrastive training temperature, not heldout tuned. Compare original versus native/raw, random projection and trained projection rank controls on all5unchanged scopes, same originalsharedcheckpoint and finalprojection1000. Per-row maximum-existing/action odds and entropy are preserved; global matching/closed-loop histories can still change, so no fixedFP/trajectory guarantee. No new optimizer/GPU/ROIbank; metadata suballocation within existing projection16MiB reservation, streaming buffers within existing32MiB reserve. This is an explicit Compact probability reranking control, not native OneJev decoder or calibrated contrastive posterior.

2026-10-16 — Native inference throughput hypothesis: officialMMDecisionEngine already provides DecoderGraphs/ForkGraphs but defaults are disabled; stock graph limits32rows/512suffix tokens do not cover dense tracking or1203category root requests. Probe official Torch graph implementations configured128rows/8192suffix/32768prefix/max1cached prefix, attached after identical stock eager auto-fork calibration. No vendor/kernel/model changes or FLAinference. Same native6000checkpoint and real legal sparseMOT/denseMOT20/categoryroot+fixedleaf requests; sameimages/state/questions, eager versusgraphs, warmup excluded/3timedrepeats. Preserve existing1e-3probability gate, exactdiscretechoices, finiteoutputs, realgraph-hit evidence and >=1.5component ratio; failure remains rejected, no oldjob/backend switched. PhysicalGPU7only with>=32GiBfree, bothenginevariants sequential, allprojectworkstill4/5/6/7. Reserve8MiBsmallprobe/inputmetadata beyond26811245534→26819634142<25GiB; no new persistentROI/imagebank. This is a component probe, not tracking accuracy or E2E speed.

2026-10-16 — Appearance-rank conservation passed67realgroups/566rows/8286IDoptions perarm, actionvalues/fullprobabilitymultisets/auxsemanticlife exact. First12officialcontrols: DT HOTA% original26.3157/rawrank30.2544/random19.4111/trained24.7555; singleRMO6.0247/5.7087/5.8205/6.1548; multi19.8019/18.9930/19.6370/20.1424. RawrankDT IDF1=38.2885 vs30.6329, IDs148vs165, butFP720vs579/FN2156vs1962. Preserveisolatedgain andrepresentation/policyinteraction; no per-taskselection or commonpromotion. OriginalTAO/MOT20ownerscontinue.

2026-10-16 — Graph probe terminal0/completed but scientifically rejected: sparse/root/leaf maxnativeanswerdiff .011445/.013179/.013894 exceeds1e-3; ratios1.044/1.057/1.061 below1.5. Dense16/32graph callsfall back,30.754svseager13.942s/.453ratio; no actualgraphsuccess. Smallest next diagnostic uses savedanswers only (0GPU):68literalChoices,17Noul>=.5gates and17Hungarianrowsall unchanged; no RMOTmembership/final-ID questions, no futuretrajectoryguarantee. InstalledQwen3.5forward has no deepstackinjection, so do not claim missingdeepstackrootcause. Numericalcause unresolved; no vendorpatch/protocolweakening/backendadoption. Save allrawanswers/timings/hits/terminalreceipt, retain failure and stop this unpromising speed path. Metadata remainswithin existing8MiBreserve/global26819634142<25GiB.

2026-10-16 — Previous status-only turn was a verified wait via original follower76060: originalfourGPU logged1700updates/27200questions,20688.803s/5.747h. Freeze exacthistorical1700row without polling/restarting training; log progress differsfromlastfullyinspected1000state. Update stale425doc snapshot. Original6000controls/pairedlongteacherCPUwait/retention and Full training remain original owners; no completion/three-taskqualityclaim. Publish this measuredrank/code/graphfailure evidence, excluding liveGroundownerstate.

2026-10-16 — Original64133handle confirms native6000 allDT/singleRMO/multiRMO before/default/assigned controls terminal-complete and parentlive waitingregisteredTAObaseline. New actualnative multiRMOHOTA%6.7400→15.6144(default)→15.8936(assigned), DT8.7030→7.8406 unchangedassigned. Next boundedCPUproof: revalidate each completed originalunit with originalvalidate_run(fullframes/candidates/rejections/checkpoint/nativequestions/video), freeze9unitmetrics, audit actualsecondpass call/existingassignment/gateflipcounts against sameassigned causalstate observationprobabilities, and compare fullpairedoutputs. No newGPU/model/threshold/data/cachebank; smallreceipt stayswithin original6000ownerreport reservation. Do not mistake cross-trajectory update differences for same-input counterfactuals or these registered videos for ordinary-model parity.

2026-10-16 — Posthoc freezer90288 failedexit1 beforepublication: firstDT lifecycleevent(frame15track2LOST) has no candidate/decision, normal tracker schema overlooked by newaudit. Original3DTunits alreadyvalidated, no inference/training failure. Preservefailedscript+receipt. Smallestfix: event signatures include candidate=None lifecycle records, memory-write probability audit explicitly considers candidateevents only; all lifecycleevents retained in assignmentcomparison. Rerun full9registeredunits/everyevent as targeted real-dataregression, unchangedmodel/metrics/frames, noGPU.

2026-10-16 — Correctedfreezer41575actualexit0, all9native units/fullvideos/originalcandidateinputs/GTisolation/checkpoints/rejections andeveryevent validated. HOTA% before600/default6000/assigned6000 DT8.7030/7.8406/7.8406, singleRMO2.5062/3.3287/3.4052, multi6.7400/15.6144/15.8936. MultiIDF1%6.6932/15.6747/16.7555; default→assignedFP467→457/FN1297→1303/IDs475→471(before185). Nativefoundationadaptation helpsregisteredRMOTbutnotDT. Sameassignedcausalstate updateaudit: single61calls/78assignments/5gateflips(4hold→update/1update→hold), multi620/1190/7(1/6), DT1350/4940/0. Originalalloutputframes exactdefaultvsassigned DT1203/1203,single796/800,multi388/800; RMOdoeschangefuturehistories. EarliernoassignedbenefitclaimwasDT-only, nowpositivebutsmallRMOeffect. Preservefixedcommoncheckpoint/policies, TAOpending, no ordinary-level/full-datasetpromotion. No newGPU/optimizer or featurebank; originalownerreports budget covers smallreceipts andfailedauditcode.

2026-10-16 — Previous goal turn is progress: actualnative9unit/RMOwriteaudit completed and verified GitHub2da51d6; this turn originalrank73067/projection68145/follower76060 live, Full1775/28400logged. Bothsmallappearance studies now allTAO4armscomplete (rankTETA14.517/14.104raw/14.324random/14.967trained; fixed14.517/13.964/14.194/14.395), allNovelClsA0. FinalMOT20controls stillrunning; no promotion.

2026-10-16 — Scheduling amendment plan for genuine paired temporal foundation training: prepared48krowCPUplans and immutable native6000fullFP32/Adam/RNG+nativeMMreload already validated; original60582v2owner does0GPU/0grad while waiting the unrelated slowTAOcontrol dependency. One resource snapshot showsGPU7free46653MiB with only originalfourGPUrank34488MiB, while4/5/6free18512/24756/20685MiB. Start bothpre-registered6000short/long stages earlier from exactlysame6000state/data/configs, keeping allmetrics/seed/steps/finalcheckpoint and deferredoriginalbaselinecomparisons unchanged. Only actualCPUwaitingv2owner may be handed off after exactPID/starttick/cwd/command/nochildren/0updates verification; preserveitsstateandhandoffreceipt. OriginalfourGPUepoch/native6000controls/otherGPUjobs neverstop. NewGPU7Torchallocatorcap.33of80GiB(~26.4GiB)+expandable_segments accounts for originalrankobservedup-to~43GiBreservation; not a full driver-memoryguarantee. FutureGPU7resource-wrapper launches take sharedflock, newpairedowner holds exclusivewindow (ownsequentialchildren inheritFD); oldalready-runningfourGPUrank explicitlypermitted. Releasewindow afterbothtraining/reloadBEFORE waiting originalcontrolowner, avoidingdependencycycle. First run realCPUprocess lock/bypass/regression; onlythenhandoff/launch andverify actualoptimizerupdates. This changesexecutionorder/allocator only, not scientificcurricula/acceptance. New4MiBmetadata reservation raisesglobal26819634142→26823828446<26843545600; noROI/imagebank. CapOOM must retainfailure/checkpoint, notskiprows orweakenprotocol. No claimedlongtraining/quality untilactualrecords.

2026-10-16 — ActualCPUwrapper regressionpassed: owninheritedexclusiveFDenters, GPU4normalwrapperenters, futureGPU7wrapperblocksandreleasesafterclose;0GPUupdates. Exactpreparedcourse/config/initreferencesunchanged. Verified4002071/start52168753WAITONLY/nochildren/emptyCUDA/0updates, sentSIGTERM viaexactpidfd only;60582actualexit143, oldstate/data preserved andhandoffintent/receipt saved. Replacementowner4118405/start52597639/session65852 nowholdsGPU7window, launchedshortSFT fromsameindexedv2configs. Nativeactual320FP32/752393024initialparameters maxdiff0 vsnative6000latestfullstate; 474weightsloaded, visionfrozen. ActualTorchcap28081638604bytes(.33devicecapacity) set; expandableallocator envpersists fornewownertraining/reload/eval (memoryonly, no comparablespeedclaim). Productionoptimizerupdatesstillunproven atlaunchsnapshot; waitactualrecords, notmodel-loadingclaim. OldFull/legacyinferencejobsnotstopped; onlyfutureGPU7launchesdeferreduntilwindowrelease.

2026-10-16 — Freeze bothappearance first4fullscopes/all16units each fromoriginalcompletedownerreceipts plusfreshofficial/runtime/frame/candidate/rejection/state checks, notrepeatvideo decode alreadydone byoriginalowner. Firstprojectionfreezer invocation mistakenly passedtrainingplanwithoutscopes (KeyErrorbeforeanyoutput); correctedtoownerrecordedresolved_controls_plan.json, sameoriginalcourse. RankTAOtrainedTETA14.967>14.517original butallNovelClsA0; fixedtrained14.395<14.517. Preserveallarms/no commonpromotion, finalMOT20stillrunning. No inference/metricchange or newGPU.

2026-10-16 — V3early study actualfirst5shortoptimizer verified: steps1..5/seen4..20, allfinite loss/CE/Brier/gradandpositivegrad,320FP32/752393024native6000initialmastersmaxdiff0. ActualTorchcap28081638604bytes applied; first5peak16.8548GiB, no capfit/steadyFPS/fullrunmemoryclaim beyondtheseupdates. Longbranchstill0, no trackinggainclaim. Source24000 nativeplan0drop/6000updates, rebuiltbyoriginaltrainer29s (not extra saved CPU plan speedclaim). ExistingFull4GPUepoch notstopped; old0gradientwaiterhandoff143only. Freezefirst5prooffromactual log/nativeinitial receipt andexactownerPIDtick, publishcode/plan/CPUlockproof/handoff/launch/first5 plus bothfirst4scopeappearanceproofs. Goal remainsactive, ordinary3taskqualitystillunmet.

2026-10-16 — Previous goal turn is progress: pairedtemporalshortactualoptimizerstarted, boundedwindow/CPUhandoff/first5proven and verified GitHubb09b07b; current original65852/68145/73067/76060handlesalllive, short50logged and Full1875/30000logged. Next physical hypothesis derivesfromcompletedcontrols: fixed-affinitytrainedprojection improvesmultiRMOT19.80→31.55butDTFP9330; probability-rankconservation controlsrejectproposalsbutlosetheRMOTgain. Before anotherintervention, auditall5originallearnedbeforetrajectories for existing-ID probability dilution: matchedGTwithcorrectsurvivingoption, sumP(existing)>maxP(action) butmaxP(existingID)<maxP(action). This is a same-state oracle potential/diagnostic only, not newprediction/HOTA or a metricreplacement. Streamoriginalscalarprobabilities and frozenbirthGTdiagnostics, noROI/image/model/GPU; keep no-correct-anchor/unmatched groups separate. If supported, test factorizedconditionalappearance P'(id)=oldexistingmass*softmax(logoldP(id)+cos/.07), keeping exactoriginalNEW/IGNORE/NO_MATCH and totalexistingmass plus learnedaux/semantics; probabilitymultiset/maxexistingodds intentionallychange. Same3native/random/trainedcontrols/all5scopes/fixedfinalprojection, original.07temperature, noheldouttuning. Register3MiBtotalforaudit1MiB/conditionalmetadata1MiB/boundedprojectedworking1MiB, raises26823828446→26826974174<26843545600; no newROI/imagebank, originalGPUtraining/measurementprotocolunchanged. Do not implementbasedon imagineddilution; inspectactualauditfirst.

2026-10-16 — Originalprobabilitydilutionaudit24041actualexit0/all5fullscopes287744candidates: positivecorrect-shortlist eventsDT1768/single53/multi191/TAO21/MOT2019195; diluted65/0/0/0/3040. Single-correctsubsetDT55/1170,MOT202587/15774; not a supported explanation oflostRMOTprojectiongain, so do NOT launch conditionalmasspolicy now. MOT20GTmatched219060 minuscorrectoption19195 leaves199865withoutverifiedcorrectshortlistedbirth; this is not automatically trueidentityabsence because birthGTunknown andshortlistmiss canconfound. Next scalarCPUcausal provenance audit distinguishes firstobservation/noGTboundbirthyet, allknownbirthsretired, aliveknownbirthomittedfromshortlist; replay lifecycledeathcauses (learnedterminate before3s vs timeout vsboth), bound/unbound births, and candidate-supportedGT atdeath. Auditall5originalCompactreferences plus9alreadycompletednative600controls; verify reconstructedbirth/survival againstfrozenexternaldiagnostic, keepGTstrictlyposthoc, noGTrepair/runtimechange. Repurposeplannedconditionalmetadata/working withinsame3MiBreserve, do notincreasecachebudget or modifyrunningcourses. Only thenchoose smallest justifiedmemory/birth intervention.

2026-10-16 — Provenanceaudit87987actualexit0/all14units: CompactMOT20positive219060 =first646/neverGTboundbirth147769/allboundbirthsretired45599/alivebutshortlistmiss5851/correctshortlist19195. Do not infer a physicalIDabsent fromunknownbirth binding. NativeDT600→6000knownbirths19→141 and learnedpre-timeoutdeaths0→138(131GTstillpresent); native multiRMO0→55(41GTpresent), despitebetterHOTA. CompactearlyknownGTdeaths DT0/single20/multi7/TAO0/MOT2014. Sourcecodeexport_broader_native_full_groups.py labels unknownbirth lifecycleTERMINATE, an alreadyknown exporterproxy ratherthanproofpermanentexit; this plusobservedheadbehavior motivates anexplicitcontrol, not acausalproof or alteringcurrenttraining. Registeropt-in timeout_onlydeletion: retainlearnedprobabilities/auxiliary/semantic outputs andsame3sTTL, deferonly learnedearlytermination. Defaultlearned_or_timeout andallrunningprotocolsunmodified. Validateactualnative+Compactprefix/defaultparity/realearly-deletiondeferral/unchangedexpiry, thenall5Compactfullscopecontrols againstsameoriginalcheckpoint. Native6000 counterpart willuseoriginal4scopes afterexistingGPU7window/baselinecompletion, exactsamecheckpoint/categoriesreusedonlywherehistory-independent anddeclared; noheldoutthresholdtuning or newROI/imagebank. Allmetadata stayswithinprevious3MiBreserve/global26826974174<25GiB; no newGPU now. Negativeoutcomesmustremainvisible, no policyadoptionbeforecontrols.

2026-10-16 — Actualterminationfunctional11397exit0: Compact30-frameprefix/firstearlyframe29ID1, native19-frameprefix/firstearlyframe19ID4; originalfullprefixstate/outputsexact, interventionretains exactlyearly-deletedIDs/allcommonstatesexact, original3sexpiryverified withlabelledfunctionalempty-observationcounterfactual (not scored frame). CPUall5study82298started, noGPU. Nativequeuedowner34665v1waitsonoriginal6000success and0inference. Staticreviewcaught summary key native_category_replay, actualrun_video exports category_replay. FixonefieldbeforeanyGPUwork; preservequeuedv1code/state andhand offonlythisnewCPUwaiter to v2. No experimentfailed orsamplesrerun yet; do not let a known latevalidationbug remain queued. Originaltraining/controls neverstop.

2026-10-16 — Nativev1newCPUqueue4147813/start52745273exactverified/nochildren/emptyCUDA/0inference, pidfdSIGTERM/34665actualexit143 preserved; correctedv2owner4149961/start52754749/session90700waitingoriginal6000terminalsuccess, noGPU. It takesexclusiveGPU7windowonlyafterdependencycompletion, rerunsall4nativeidentity/lifecycle/scorers atsame6000weights/3s; same-image-onlyTAOcategoriesreplayedexplicitlyfromcompletedoriginal6000source, noE2Eclaim. No originaljobterminated. Compactfirst3fullscopes/all6units frozen: HOTA%DT26.3157exact,single6.0247→6.4816,multi19.8019→19.5578; multiIDF1=21.0082→21.0638/IDs165→157/FP619→597/FN1054→1061. Mixed,noadoption; remainingTAO/MOT20continue.

2026-10-16 — Literal lifecycleinventory12901actualexit0: full-sourceMOT252258rows/lifecycle93624:ACTIVE60077,LOST12117,REACTIVATE16748,TERMINATE4682(5.00085%). Balancedcompleted24k:1000life/250each(25%TERMINATE); pairedshort/longsame500life128ACTIVE/125LOST/117REACTIVATE/130TERMINATE(26%). Thus currentnaturalFullcoursehasnot inheritedbalancedoversampling; do notstoporretargetongoingoriginaltrainingbasedon anunprovenproxycausalityclaim. All data/targets preserved. Sourcecontractunknownbirth→TERMINATEisnotindependentpermanentexitevidence. OriginalFourGPU2kstateobserver49603registeredwaitingcheckpoint, boundoriginal3457625/start49350320, fullFP32/Adam/allrankRNGgateunchanged; latestglobalfeatureupper26826974174 passedasmetadataoverride insteadofstalehistoricalupper. Lastcompletednumericmilestoneremains1000untilactual2kproof, nonewGPU.

2026-10-16 — Actualobserver49603exit0/2000fullstate validated:32000questions17590MOT/7107RMO/7303OV; all752393024FP32/1504786048Adammomentsfinite,all320Adamsteps/scheduler/cursor2000exact,fourrankTorch/CUDA/Python/NumPyRNGsaved. Atomiclatest9029193825bytes andnumberedBF16ckpt-step2000present; futurelatestreplaced, numberedexportretained. No actualmilestonerestart/newinference orcompleteepochclaim. Lateststep2000loss1.10098/grad51.8833/elapsed24227.312s isobservedconcurrenttraining, not controlledspeed. Originaltrainingunchanged. CPUtimeoutTAOcompleteTETA14.517→14.493/NovelClsA0; first4all8unitsfrozen, finalMOT20continues. Preservemixedfailure-to-improve, no promotion. Publishsourceaudits/default-prefixproof/optinpolicy/queuednativecontrol/preexecutionfixreceipts/actual2000proof; allreportswithinshared3MiBmetadatareservation, noROIbank/GPUincrease. Goalremainsactive.

2026-10-16 — Originalhandles52020/14226/46228/64133/65852allconfirmedlive. Native600TAO40beforebaselinejustcompletedandconsumer64133validated35228candidates/all40frames; native6000defaultTAO nowactuallywaitsregisteredGPU7trainingwindow. Sameoriginal600checkpointall4beforebaselines frozen:fndDT HOTA8.7030%,singleRMO2.5062,multi6.7400,rawTAOTETA27.748/LocA21.573/AssocA2.8136/ClsA58.856; BaseTETA37.437/Novel8.3686/NovelClsA0(onlyknife/baby/peelerGT). NativebaseclassificationstrongerthanCompactbutassociationweak; do notmixcomponentmetrics/predictsyntheticTETA orclaimcompleteTAOdatasets/ordinaryparity. Baselinecompletionisnotnative6000aftertraininggain. OriginalGPU5refreshedworkerhasadvancedDT(frame168latest),GPU4oldTAO(frame987started)stilllive; Groundretentionquietlive. Shortpairedtraining425logged, no newcheckpointclaim. Recheckonlycompletedbaselineownerreceipts/official/runtime metadata, preserveinputorder/GTisolation andallrawresults, no newGPU/ROIfeature or inference.

2026-10-16 — Time-description correction: sourceTAOmanifest explicitly physical_video_fps_verified=False/fps1/timestamps0..39/annotation-stepunits. Earlier lifecyclecontrolprose3seconds and provenancefield seconds_since_last_assignment were misleadingforTAO; MOT/RMOTphysicalsecondsremainvalid. Numericruntimehasalwaysusedoriginalmanifeststeps, not3physicalsecondsforTAO; do notinventFPS/changecontrols. Clarifyconfigs/docs andwritev2clock provenance: explicitphysical_seconds/annotation_steps andregistered_clock_gap_since_last_assignment. Real32945exit0, all14oldvsnewcounts/availability/retirementcauses/numericgaps exactlyequal. Retainv1, originalregistrations andalltrajectories/scores; addclockclarificationreceipt. This isdescription/schema correctiononly, notnewinference/input/timeouts/protocol/numericalchange. Shared3MiBreportreservationstillcoversallartifacts. FreezecompleteNative600before4scopebaseline withclockqualification; no qualitypromotion.

2026-10-16 — Previous goal turn is progress: lifecycleprovenance/defaultprefix/controlimplementation, actual2000fourGPUfullstate and completefour-scope native600baseline published and verified4288d02; clockclarificationretainsallnumbers. Current65852short600logged, original73067rankfinalMOT20events/68145projectionfinalMOT20training stilllive; original82298timeoutCPUowner actualexit0/all10controls complete. Validate/freeze all5 before new work, no commonpromotionfrompartialresults.

2026-10-16 — Next birth-admission hypothesis from actual API/exporter code: exporter targetsNO_MATCH only when a correct birth anchor is in the live pool but omitted from shortlist; NEW when no recoverable live correct anchor exists. V3prompt says previously observed but does not explicitly qualify live/retired, and both Compact/Full omit totalpool/omittedcount from decision inputs. Auditall14frozenunits for actualNO_MATCH with complete live-pool exposure, including GTmatched never-bound/retired/correct-option and GTunknown groups; do not declare unknownbirths physicallyabsent or change oldsemanticlabels. If supported, test explicit active-pool execution contract: leave all original model probabilities/aux/semantics untouched, forbid only the privateNO_MATCHcost when every live ID is offered. No probability renormalization or learned thresholds; global assignment/futurehistories can change. Factorial all5Compact scopes: original and completedtimeout references, feasibility-only and feasibility+timeout, samecheckpoint/candidates/scorers/no per-taskselection. Defaultpolicy and all live GPUcourses unchanged. This is a declared contract/intervention consistent with exporter labels, not proof original vagueprompt was impossible or a native calibrated posterior improvement. New2MiB audit/controlmetadata+boundedworking raises26826974174→26829071326<26843545600; no newROI/imagebank or GPU. Auditbeforeimplementation.

2026-10-16 — Exposureaudit66807actualexit0/all14: CompactDTall13664rows exposecompletepool yet10406NO_MATCH selections (GTmatched2100,GTunmatched8306); completeMOT20pool137727rows/112129NO_MATCH (GTmatched105239,GTunmatched6890). Correctknownliveanchor actuallyoffered in598DT/2986MOT20selectedNO_MATCH cases; unknownbirthgroupnotphysicalabsenceproof. CompactRMOonly1each, TAO0completepoolNO_MATCH. Native6000DTonly3completepoolNO_MATCH/RMO0each; primarynative6kregressionisnotassumedsameasCompact. No remainingprintedmasszero inauditedcompletepoolrows. This supports actualCompactactive-pool feasibilitycontrol; do notaddanotherGPUstudy to currentnativequeue. Implementmatchingconstraintonly, preserveall originalrawprobabilities/confidence/entropy and allmodelaux/semantics onidenticalinput; notconditionalrenormalization. ExtractexistingHungarianhelper withoutchangingdefault arithmetic and verifyall14originalassignments/339596candidates plusconstrainedroweligibility/probabilityimmutability beforefullfactorialrollouts. Allcontrols4arms×5scopes, original/timeoutreusedexplicitlyandonly2newarmsinfer. No GTdecisionrule or input/proposalfilter.

2026-10-16 — Realfeasibilityvalidation18145actualexit0: all14/339596originaldefaultassignments exact, originalprobabilitydicts unchanged, forbiddenprivateNO_MATCHneverchosenundernewactive-poolmode, frameswithoutmaskentireassignmentsexact. Same-inputshadowassignmentchangesDT10792/MOT20116558/single1/multi1/TAO0; these are not rollout/HOTA gains. LargeDTGTunmatched8306NO_MATCHcaseswarnofpotentialfalsebirth/duplicateproblem; do notguaranteeimprovementorhideworsening. FactorialCPUowner started afterproof, all4×5originalscopes, twooriginalcompletedreferencearms/twonewactualarms, samebasecheckpoint/noGPU/newfeaturebank. Modelrawdistributions andconfidence/entropyremainunchangedonidenticalinput, costconstraintonly; originalprompt/exporter-semanticdiscrepancydeclared. Existingnative750Mshort/long/Fulltrainingandqueuedlifecyclecontrol remain unchanged.

2026-10-16 — Previous user-question turn was a status explanation, no research-state change. Resume with concrete evidence: original48863confirmedlive through its existing blocking handle, first12factorial units actually completed; freeze their official/runtime/frame/candidate/rejection/finalstate receipts. Feasibility-only DT HOTA26.3157→12.9762,IDF130.6329→9.1995,FP579→3509,FN1962→1550,IDs165→459; bothsameDT. Singlefeasibilityunchanged6.0247; timeout/both6.4816. Multi19.8019→20.6629feasible/20.2362both, timeout19.5578. Additional admission cannot be adopted as general association repair; preserve negatives, no commonpromotion. TAO/MOT20sameoriginalowner continue, no task-specific selection/newGPU or proposalchange.

2026-10-16 — Originalrank/projection/timeoutstudies allclosed:20+20+10officialunits and all5scope receipts retained. FinalMOT20original/rawrank/randomrank/trainedrankHOTA7.4316/8.0009/7.9101/8.0870; trainedrankIDs3317>2060original. Fixedraw/random/trained31.3078/31.5077/31.7503, trainedIDs8689/FP16049 versus2060/463original. Largefixedfusiongainisnotlearnedprojectiongain because raw/randomcontrols already31.3/31.5; trainedprojectionDT8.46bad andallTAONovelClsA0. TimeoutMOT20HOTA7.6054/IDF16.3199 versus7.4316/6.1474, smallmixedacrossscopes. No modelpromotion. Exactobservedfull2250/36000 andshort875/3500logrows frozen for truthfultraininganswer; no newfullstate gate beyond2000, noGround-Onejointgradients/longBPTT. Docs updatedwithallresults/limitations. Allnewmetadataunderexisting2MiBreservation/global26829071326<25GiB; no ROI/imagebank or newtraining.

2026-10-16 — Next targeted TRAIN-only hypothesis after feasibilityDTfailure: originalbirthaudit already finds5832/7992IGNORE proposals with someGTIoU≥0.5, but doesnot showwhether the globallychosen competitor is inside the actual16-candidate request. Readexporter: one Hungarian overwholeframe determines IGNORE, then inputs aregrouped; fullimage remainsvisible. Auditall43/1056TRAINframes/48051candidates usingactualsavedgroups/rowtargets/registry, distinguishbackgroundvsGT-supportedIGNORE and competitorsinside/outsideactualgroup. ReproducealloriginalGTbindings andexactIGNOREtargets beforecounts; leaveambiguousmultitargetcasesexplicit. No claiming mathematicallyunobservable labels because modelseesfullimage, no NMS/trainingrecipe/proposal/metricchange. Small scalarreport≤128KiBfromalreadyreserved2MiB/global26829071326; no newROI/imagebank/GPU/optimizer. Do notretargetongoing459kcourse fromaudit alone.

2026-10-16 — Actual97066exit0: candidate-groupauditall43TRAIN/1056frames/48051candidates, everyoriginalregistryGTbindingandIGNORE/nonIGNOREtargetexact; matchespriorbirthaudit7992IGNORE/5832GT-supported. 4804/5832=82.37%GT-supportedIGNORE haveonlyoutsidegroupassignedwinners (4723singleGT),1024inside/4829outside(overlap)/4noassignedwinner/261multiGT. Actual3401nonemptygroups/max16, geometry/localindicesexact. Positives6754withrivals/5319outside. Fullimagepixelsstillvisible; notprooflabels mathematicallyunobservable, notcausaltrackinggain oruniquevalidproposalcertification. Readonly63350bytereportwithinexisting128KiBsuballocation/noextrareserve/features/GPU. Currentcourses/labelsuntouched; nextadmissionresearchmustexplicitlyprovidecompetition, ratherthansilentlyturningIGNOREintobackground. Original48863TAO completed, freezeall16first4units: feasibilityTAOTETA14.517exactoriginal; both14.493exacttimeout,allNovelClsA0; finalMOT20sameownerpending. Publishedprior18filesverifiedremote3f18440/tree1c9d8a3, thenrecordnewaudit/TAOreceipt separately. Goalnotcomplete.

2026-10-16 — Previousgoalturnprogress: actual43TRAINgroupcompetitionaudit/first16factorialunits and code/results verifiedGitHubafa32bd. Currentoriginal48863confirmedlive/MOT20feasibleinference, no restart. Smallest nextintervention frommeasured4804outside-onlyIGNOREcompetitors: opt-in overlap-neighbor candidate grouping with unchanged16capacity. Seedlowestremainingoriginalindex, repeatedlyaddmaximumIoU-to-groupremainingcandidate, originalindex tiebreak; no IoUthreshold/NMS/topK/GT/classrule orcandidate removal. Stream one-vs-Ngeometry(noN²featurematrix), preserve originaldecision/outputroworder aftergroup modelcalls. ExactsamefrozenCompactweights/shortlist32/learnedheads/defaultlife+NO_MATCH/registeredclocks; groupingcanchangealllearnedoutputs, shortlists/life-first-group/futurestates, explicitlynot pureassociation-only or probabilityconservation. All5scopes before/newgrouping, completedbefore references explicit, no newGPU/optimizer/bank. Validateactualall5defaultprefix states/outputs exact and alternativegroup feature/index coverage beforelaunch; quantifyTRAINcompetitor co-exposure usingactualreferencegroups, notqualitymetric. Workinggeometry≤1MiB plusreport≤256KiBwithinremaining existing2MiBaudit/controlreserve; featureupper26829071326unchanged. NativeFull/pairedcourses unchanged; no future trainingrecipepromotedwithoutheldoutrollouts.

2026-10-16 — Actual36906exit0/newall43groupaudit reproduces everypriorTRAINcount/GTbinding/IGNOREtarget; overlapneighbors only-outsidewinner4804→298, inside1024→5530, positiveswithoutsiderivals5319→320. This isnotnewlabel/qualitygain. Actual23522exit0/15realframes/3339candidates acrossall5scopes: full defaultoutputs/stateexact, alternativecausalinputfeatureownership andprobability/update/query/classrows correctlyrestoredtooriginalcandidateindices. Afterbothgates, originalnewCPUowner90021 launchedbefore/regroupedall5×2, samefrozenmodel/16capacity/shortlist32/noGPU/newoptimizer; referencesexplicit, allnewcandidaterowscheckedasfullpermutation. No NMS/threshold/GT/classfilter; alllearnedheads/shortlists/life-first-group/futurehistories maychange. Do notclaimassociation-only/constantprobabilities/futuretraininggain. OriginalFull/paired/NO_MATCHjobs untouched; Groundretention original52020confirmedlive. Newgeometry/reportworkingwithinremainingexisting2MiBreservation/global26829071326; noROI/imagebank. Goalactive.

2026-10-16 — Newgroupingoriginalowner41803/start53230371/session90021verifiedlive; firstDTfull1203 actuallycompletedandfrozen2officialunits. Samefrozenmodel groupingonly: HOTA26.3157→28.6921,DetA45.1842→46.2724,AssA15.3476→17.8072,IDF130.6329→33.2758,IDs165→159,FP579→736,FN1962→1794. ModestpositiveDT withFPcost, notuniformmetricimprovement orproven5scopebenefit. KeepallremainingRMO/TAO/MOT20controls; do notpromoteper-taskcheckpoint/policy. Registration/launchsnapshot retained; newsource/runtimegroupscheckedpercandidate byoriginalowner. NoGPU/optimizer/featuresadded.

2026-10-16 — Original90021completedbothRMO800scopes, freezeall6first3officialunits: singleHOTA6.0247%,multi19.8019%, everyreportedofficialmetric exactlyoriginal. Do notinferallprobability/stateequality fromunchangedmetrics. GroupingpositiveDTdoesnotyetconstituteall-taskbenefit; TAO/MOT20pending unchangedoriginalowner. PreserveearlyDTreceiptalonglaterfirst3receipt, publishimplementation/audit/defaultprefix+rowownershipproof/registration/actualownerlaunch/first3metrics. NoGPU/optimizer/newROIfeatures.

2026-10-16 — Previousgoalturnprogress: overlap-neighborimplementation/defaultprefix+rowalignmentproof/all43TRAINcoexposure/first6officialunits publishedandverified93fb5fa. Currentoriginal90021TAOrunning and48863MOT20feasibleofficial→events, confirmedlivebyexistinghandles; no restart. Original76060follower reportsFull2350/37600,65852short1100 andreal1000fullstate9029140889bytes saved. Reuseexisting shared-checkpointfreezer with explicittrainingpath/expectedinitial6000 and indexedsourceIO; preserveitsdefaultoldbalanced600-initializationbehavior. Validateall320masters/Adam/scheduler/cursor/RNGpresence andactualseed0index exposures once onshort1000state. Teacherrows genuinelylacksource_row_id; inventorytheiroriginalrowIDsseparately, do notinventolder-sourceIDs or inflateframecounts. No trainingchange/restart/newGPU/bank/checkpointcopy, smallscalarproof withinexistingreportsreservation/global26829071326. Trackinggainremainsunprovenfornewshortbranch.

2026-10-16 — NewcompletedGround200artifactsnow6tasks visible (COCO,dense200,VisDrone,RefCOCOgval/test,RefCOCOplus) whileoriginalfull9ownerstillwaitingremainingclients. Existingpartialfreezerhardcodesold2-taskoutput; addonlyexplicit--out(defaultoldunchanged) tofreezecurrentcompletedwholetasks withunchangedofficialsummarizer andexactoriginal/effective/providerfilteredcounts. Do notclaimfull9, infermissingtasks, launchnewclients, stopservices orattributeallweightcausality: originalqueue32/adapted-syncpost-recoveryexecutiondiffers. Newsmallscalarresultssuballocateexistingreportsreserve, nobank/GPU/checkpointcopy. This iscompletionevidence forrequiredcontrolledGroundadaptation, separatefromtrackingHOTA/ordinaryparity.

2026-10-16 — Actual79475exit0/short1000fullstate9029140889bytes:752393024FP32/1504786048Adammomentsallfinite,320steps/scheduler/epoch/cursorexact,RNGfields saved;initial6000FP32maxdiff0. Actual4000seed0questionsMOT1288/RMO1327/OV1385;673teacherMOTrowswithoutoldsourceIDcountedseparately,615providedoldMOTIDs; firstmedia1017/71/1180 includeswiderlegalteacherTRAINimages, notallhistoryframes/independenttracks.45loggedrowsfinite. No1000restart/newnativeinference orshort6000/longcompletion. Existinggenericfreezerparameterizedonlypath/initialstep/source-manifestloader/missing-sourceIDinventory; legacydefaults retained, currentnewrealcheckpointprovidestargetedregression. No broadrepeatold6000state scan.

2026-10-16 — Original90021completedTAOall40; freezefirst4/all8officialunits: groupingTETA14.517→13.997,LocA8.088→6.6283,AssocA5.1478→10.2,ClsA30.314→25.162,NovelClsA0. Associationupwhileothercomponentsdown; no syntheticcombinedscore/categoryreplay orpromotion. Original48863MOT20feasible-onlycompletedall1657/235232: HOTA7.4316→9.8980,IDF16.1474→10.1300,IDs2060→3292,FP463→564,FN286273→267117. Freshofficial/runtime/frame/candidate/isolation/finalstate checks retainoriginalfullsourcechecks; freezeonecompletedunit. Combinedfeasible-timeoutstillrunning; DTcollapse remainsfailure/nonewpolicyadoption.

2026-10-16 — Groundpartialfreezeractualexit0/original9tasks91124rows/adapted200six31986rows, alloriginal=effective/providerfiltered0. CompletedcommonF1mIoU%orig→adapted COCO47.9781→47.5713,Dense66.6435→66.0540,VisDrone29.6624→27.8704,RefCOCOgval91.7038→89.3700,test91.1193→88.5215,RefCOCOplus84.3763→81.6416. Allsixnegative; preservewholeofficialresultcounts, no Ground200promotion. PendingLVIS/RefCOCO/RefCOCOg notinferfull9; queue32originalvsadapted-syncafterCUDArecoverymeansnotexclusiveweightcausality/bitwisepairedclaim. No trackinggaininferred. Existingcurator--outonlyparameterization preservesold2tasksnapshotandallsources, no newmodelclients orservicestops. Allnewproofswithinalreadyreservedscalarreports/global26829071326; noGPU/optimizer/bank/checkpointcopy. Goalactive.

2026-10-16 — Previousgoalturnprogress: actualshort1000all752M/Adam/cursor/sourceexposurevalidation, first4groupingcontrols/MOT20feasibleunit andsixwholeGroundretentiontasks publishedandverifiedc962b4b. Current90021/48863originalhandlesconfirmedlive/finalMOT20arms. Next compute question-context reuse opportunity in actual full459027seed0/global16/fourrank index-shuffle. ReadnativeSFT/shared: existing share-prefix supports oneGPU text-packedbranch masks; groupforward doesnotuse multimodal mm_inputs, and Qwen3.5 GDN recurrentstate needs branch-aware handling. Cannotjustenableflag orreuse languageactivationsacrossoptimizerupdates. Auditall64originalsources usingnative per-row rendering toidentifyidenticalserializedstate+mediarecipes; reusealreadyvalidatedint32lengths andnativeplan_steps placement, noMMtensor/model/inference/GPU. Compute sameglobalstep andsamerank co-occurrences plusoptimisticfull-rowtokensavingupperbound fortheconservativeexact-metadata cachekey. Epoch repeatedcontexts alone arenotavailableacrosschangingweights; notgeneralboundoverallpossiblepixel-equivalences oractualthroughput. Allrows/familyfilters/order/seeds unchanged. Persist only≤128KiBscalarreportfromexistingfullsource/reportreservation, no native feature/activation bank or newoverallcacheallocation. This audit decideswhether a future exact branching implementation merits priority forcurrentrecipe; donotretargetrunningtraining.

2026-10-16 — BothoriginalCPUstudies90021/48863actualexit0/all5closedandfreshreceipts10+20units. GroupingMOT20HOTA7.4316→7.0507,IDF16.1474→5.6306,IDs2060→1992,FP463→341,FN286273→287351; NO_MATCHbothMOT20HOTA9.7962/IDF110.0231/IDs3447/FP513/FN265498 versusorig7.4316/6.1474/2060/463/286273. Keepallnegativecrossscoperesults, neitheradopted. Defaultmodel/policies unchanged.

2026-10-16 — Nextshared-core hypothesis fromcompletedcanonical controls andactualcode: frozenuntrainedcanonicalphyscore keepsMOTexact, TAOphysicalLocA33.075/AssocA35.637 vsorig8.088/5.1478, butclass9.6637vs30.314 andRMOHOTA2.005/11.753vs6.025/19.802. Existingcanonical routessemanticheadfromcanonicalcandidatevectors; actualspec onlyentersmembershiplate. Addopt-in secondpass ofthe SAMEfrozenTransformer/task/text/semanticweights forsemanticreadout, conditionedonactualtask/query; physicalmatch/update/reliability/life remaincanonical. Oneidentitymemory andonecommoncheckpoint, noGTstate repair/newparams/training. Onidenticalnativeinputs validatephys==canonical, semantic==originalquery-coupled, MOTalloutputs==original; all61actualTRAINcontextgroupsplusBaseobservations/declaredqueryvariants. Then before/canonical-untrained/newtask-readout all5fixedscopes, oldtwoarms explicitcompletedreferences, no per-taskcheckpointselection. NotpredictingcombinedTETAfromoldcomponents: newtrajectory/officialmetrics required. Existingfull/pairednative trainingunchanged. Onlysmallscalarproofs/modelvariantflag fromsameweights, no bank/GPU/featureallocation; global26829071326 retained. Categorysingleton idea deferred: original64frameOVcourse includescausalteacherhistory whereasnewBase175supplementisn1/m0, so do notclaimbaselinealreadytrainedexclusivelyon singletoncontext.

2026-10-16 — Actual41236exit0/contextauditall64sources459027rows/nativeaugmentation/seed0/native16x4dealing/tail1,1,1,0 exactlypreserved;97556serializedstate+media contexts/24560medialists. Sameglobalupdate150repeatedcontexts/150extrarows: full-rowtokensavingupper2945176/5639441074=0.0522246%; same-rank23/23,390967=0.00693273%. Theseconservativeexact-metadata-key bounds arenotactualspeed/allpixel-equivalencebounds; media-onlycooccurrences don'tmeasurevisioncost. No forward/GPU/optimizer/bank. Currentrandomshuffleofferslittle exactwithin-stepbranchreuse; do NOT switchofficialtextsharedprefixentry orreordercurrenttraining. Futuregroupedrecipe+multimodal/GDNbranchstateimplementation wouldneedseparatevalidation.

2026-10-16 — Actual79437exit0/newtaskconditionedreadout67realgroups(61withhistory)/402specinterventions/143238outputelements: phys==canonicalandqueryinvariant; category/membership==originalquerycoupled onidenticalinputs; MOTallheadsexactoriginal.335canonicalsemanticinterventionschanged. Separateactualweightcomparisonall66oldstate_tensors/1146511elements bit-exact acrossold/canonical/new; no newparamtensors, samecanonicalbuffer. Noqualityclaimfromthesetensorchecks. NewCPUowner51823launchedall5/3arms afterproofs,10completedreferences+5actualtrajectories, originalallscopes/16groups/shortlist32/TTLclocks/heads/thresholds/scorers unchanged. Singlememory andsinglecommonparamset acrossalltasks; semanticbranchactualtask/query contextonly. OriginalFull/pairednative traininguntouched, no GPU/newfeaturebank; scalarproofs/modelvariantwithinexistingreserves, global26829071326. Goalactive; defaults remainoriginalcoupled/noncanonical.

2026-10-16 — New51823actualfirst3/all9officialunitscompletedandfrozen. HOTA% original/canonical/newtaskreadout DT26.3157/26.3157/26.3157, singleRMO6.0247/2.0050/2.4619,multi19.8019/11.7532/11.9169. NewsingleFP47/FN17 vscanonical80/17/orig323/13; newmultiFP160/FN1704 vs196/1703/orig619/1054. Restoredsameweightsquery-conditionedsemanticsreducesFPslightlybutdoesnotrepaircanonicalphysicalmisses/association; no defaultpromotion or claimnewunifiedimprovement. RemainingTAO/MOT20originalsameowner, no threshold/seed/framechanges orsyntheticcomponentmix. Nextlearningworkmustaddressphysicaltransfer, notassumereadoutalonefixesalltasks. Publishcode/plan/tensor+parameterproofs/launch/first3metrics/all5previouscontrolclosures/full459kcontextaudit. AllrunningnativeGPUjobsuntouched.

2026-10-16 — Previousgoalturnprogress: full459kcontextaudit andboth5scopecontrolclosures/newtaskreadoutimplementation+exacttensorproof/first9metrics publishedverified4ad1839. Current51823TAOofficial→events confirmedlive; original65852short1550/76060Full2550logsfinite. ReadpreparedknownbirthRMO2882component andfullsourcecurriculum: new18rollout RMO component selectedmembership/presenceonly, notthese1428identity+1454assignedwrite rows. This iscurrent-curriculumcoverage, notproofnoearlierRMOTphysicaltraining orthatgapalonecausesfailure.

2026-10-16 — Prepare matched native input component (no newGPUtraining): reuseall2882certifiedknown-birth physical questions/targets andtheircorrespondingoriginalquery-membership question/target fromsamecandidate. Botharms sameoriginalactualactorstate/media; assigned-write control gets samealready-selected-identity state asphysical, not GT-repairedhistory. Keep originalroworder/pairIDs, sourcequestionprovenance separate. Native rowaugmentation shufflesdifferentnumberofoptionsbeforechoosingJSONstyle, so same seed canotherwisegive differentprefixformats. Explicitnewpairedcomponent fixesbothstates to source-equivalent prettyJSONstring; keepnative per-questionoptionpermutation, then verifyall2882native renderedstateprefixes equal. This isdeclaredderivativeinputformat, notunchangedoldtrainingrecipe. Originalinputtrees/questions/media/labels retained, unknownbirth/missing-match notinventednegative. Runbothcomplete2882-row nativeMMCPUplans zero-drop beforefuturetrainingrecipe; nooptimizer launched/queuedbycomponent. New4MiBscalar/groupmetadatareservation raises26829071326→26833265630<26843545600; noROI/image/activationbank, no currentfull/pairedcourse edits. Source18rollout1814physicalframes/977filteredmedia; anchorage isnotocclusiongap.

2026-10-16 — Actual43883exit0/newpairednativecomponent2882pairs/2074groups prepared; originalnative rowstylewoulddiffer698pairsunderthesameseed dueoption-count-dependentRNG. Fixedsource-equivalentprettystrings eliminateallrenderedstateprefixdifferences whileoriginalquestion/target/media/JSONtree/order provenanceexact. Physical1428identity+1454assignedwrite(981no/473yes), semantic2237yes/645no. Bothall2882nativeMMCPUplanszero-drop: physicalmax18096/tokens15639410, semanticmax17227/tokens15226174, noforward/optimizer/GPU. Totalcomponent1665559bytesbeforecoverageproof<4MiB; newglobal26833265630. SourcefullcurriculumRMO97112membership+3701presence only (plannedinventorynotactualconsumedcounts); noneof2882newphysicalrows silentlyinserted.

2026-10-16 — Original51823actualexit0/all15newreadoutofficialunitsfrozen. TAOTETAoriginal14.517/canonical26.125/new30.818; newLocA33.075/AssocA35.637unchangedcanonical, ClsA9.6637→23.743 stillbeloworiginal30.314; BaseTETA31.9085,NovelTETA28.637butNovelClsA0(threeGTclasses/oneNovelpeeler/40annotationframes). NotfullTAOparity orNovelrecognitionclaim. MOT20allmetrics exactlyoriginalHOTA7.4316/IDF16.1474; RMOsingle2.4619/multi11.9169 stillbeloworiginal6.0247/19.8019. No defaultpromotion orper-taskmix. Ordinary3taskgoalunmet.

2026-10-16 — Legacyoriginalhandles14226/46228confirmedlive; oldnativeTAOnowlastframe1132, refreshedcompletedallDT/RMOfirst3andstarteditsTAO. Actual750Mhistory-refreshstage additional600/2400eachfromsame600FP32initial, sameperstepfamily/tasks/400historyrowschanged, nottoken/FLOPmatched: DT old13.1075→refreshed16.4112 vsbefore8.7030; singleRMO3.0717→2.8779 vsbefore2.5062; multi6.6825→7.5062 vsbefore6.7400. Freezecompletedbefore/old/refreshedfirst3wholeunits withoriginalowner/official/runtime/sourcegeometry/video/rawofficial-input checks; no rescore/newnativeforward, no fullTAOconclusion. New1MiBnativehistoryscalar/prediction-report reserve raises26833265630→26834314206<26843545600; originalworkers/services untouched. Keepmatchednegative singleRMO outcomevisible.

2026-10-16 — Actual8215exit0 freezes all9 native history first3 units and their exact compressed raw predictions/original official scores;734102bytes within the1MiB reserve. Actual23170exit0 proves canonical versus task-readout physical trajectories across all4500frames/287744candidates/331063events plus final physical memories exact; semantic labels/counts only excluded. Restoring task semantics alone fails to recover RMOT physical coverage. Neither study changes the default model.

2026-10-16 — Previous user-question turn only explained scope and observed logs; no research-state change. Resume concrete work: original14226 now actualexit0/HISTORY_REFRESH_WORKER_ALL_REGISTERED_UNITS_VALIDATED. Old-history TAO completed TETA25.161 versus native600before27.748, NovelClsA0; refreshed46228 confirmed live at frame88. Original65852/76060 confirmed live, short1950/7800 andFull2750/44000 observed, not new numeric checkpoint gates. Freeze the newly complete old TAO unit with source/decoded-frame/GT isolation and exact official prediction transport checks, within the existing aggregate1MiB history-report reservation if it fits; no rescoring or GPU. Publish paired-input rebuild code/proofs, all5 readout closure, complete physical-rollout proof and native history results. Current courses and original workers untouched; ordinary three-task goal remains unmet.

2026-10-16 — Actual99790exit0/old TAO native40 source+geometry+decodedvideo+GT isolation verified, exact raw and official JSON tracker inputs reconstructed from every output including unknown-category -1. Original TETA27.748→25.161, LocA21.573→23.965/AssocA2.8136→2.9972/ClsA58.856→48.522, NovelClsA0. All5112official predictions preserved; no rescoring/inference/GPU. First3 plus old-TAO artifacts aggregate875796bytes<existing1MiB; global26834314206 unchanged. Refreshed TAO still original46228live, no inferred score. Observed2750/44000 and short1950/7800 exact rows frozen for prior user scope answer; numeric gates remain2000/1000, long0at observation.

2026-10-16 — Publication53ce5ed verified:53files/treefd210563 including complete readout/control proofs/native predictions and paired2882rebuild inputs, no weights/banks/live Groundowner. Next actual learning prerequisite: construct two complete24000three-task courses using all old8000MOT/8000OV plus last5118oldRMOT rows and all2882paired questions. Same21118common rows/order/IDs and same pairedphysical states/media at actual native global indices; only registered question/target families differ. Both use same future native6000FP32 initialization, seed0/nativeglobal4/indexshuffle,6000full-course updates per arm if separately launched; no GPU queue/optimizer here. Native per-step length ordering can differ, not token/FLOP matched. Run both complete native MM CPU plans, zero-drop required. New2MiB manifests/configs/log reservation raises26834314206→26836411358<26843545600; no new pixel/ROI/activationbank or edits to active Full/temporal courses. This tests physical-supervision transfer, not another semantic-head-only repair.

2026-10-16 — Actual56390exit0/both complete24000native MM CPU plans zero drops, max27480tokens; physical202072473plannedtokens versus semantic201659237, both6000nativeglobal4updates. All2882paired prefixes exact at actual21118+indices; common21118old rows unchanged. Metadata257474bytes<new2MiB reservation, global26836411358. Register actual additional6000native750M/24000 per-arm physical-versus-semantic learning study from identical verified6000FP32; physical then semantic, same seed0/newAdam, fixedfinalsharedmodel and assigned-ID write protocol across original4scopes. Both training+reload stages precede CPU-only wait for original6000assigned TAO baseline; no substituted600baseline or per-task selection. OldGPU4owner actualexit0 releases native slot; one launch observation GPU4free37341MiB/existingfullrank43804MiB, checkpointdisk350843351040bytesfree. Use GPU4only, existingfourGPUrank untouched, Torch.33cap28GB/expandable allocator; extend existing allocation wrapper with explicit--physical-gpu4 while default7 unchanged. No exclusive resource claim or process stops. New4MiBowner/report reservation raises26836411358→26840605662<26843545600; no native feature bank, originalcourses unchanged. First real optimizer updates/native initialization evidence required before claiming launched training.

2026-10-16 — Actual newowner34237 running /paired_rmot_physical_semantic6000_owner_v1 from committed d28e15b; native GPU4.33cap28081638604bytes, exact6000FP32initmaxdiff0/752393024language parameters. Original blocking handle emitted updates1..5, losses/positivegradients finite,20questions. Immutable first5validator73855actualexit0 confirms owner PID/starttick/cwd/script and exact rows4/8/12/16/20; native seed0 source exposures9MOT/6RMOT/5OV including2newRMOTidentity rows. First5Torchpeak17.87594GiB, not full-run peak/stable speed or quality gain. Semantic0at snapshot, bothfull6000andfour-scope assigned-ID controls pending. This is the real GPU4 allocation-wrapper regression; defaultGPU7 CLI retained and originalFull/temporal workers untouched. No new native feature bank, currentglobal26840605662<25GiB. Preserve failed results and full goal active.

2026-10-19 — G2 protocol repair and clean coverage launch: read-only audit confirmed the historical 199053-row G2 manifest contains 3317 MOT20-05 rows, so formal_g2_1200/G2-PL remain engineering evidence rather than independent held-out evidence. Built immutable G2_CLEAN_FULL_COVERAGE_V2 (583251 unique rows, 2249 rejects, MOT20-05 absent, TAO Novel/sparse negatives excluded, all legal Refer-KITTI frames, exact RMOT questions, BURST gap explicit; rows SHA f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71). Added exact-once source/video/image coverage accounting, global-batch gradient accumulation with DDP no_sync, token-budget checks, per-step token/gradient/peak-memory logs, and strict base/manifest/LoRA/optimizer/scheduler/per-rank-RNG resume validation. GPU3/GPU5 BS2 smoke and two-rank step-1 resume replay passed; BS8 resource probe failed with a real vision OOM and is retained. Formal clean owner is running from G0 with micro-batch2, accumulation8, global batch32 and 18227 planned optimizer updates. J1 remains active at durable step600/9571 seen rows; no quality metric is claimed before terminal receipts and fixed independent evaluation.

2026-10-19 — G0 MOT20-05 official scope evaluation completed from the preserved 1658-frame export. TrackEval receipt reports HOTA 0.433441, DetA 0.638348, AssA 0.294655, IDF1 0.441085, IDSW 6239, FP 6325, FN 98471; 183 rejected-perception frames remain in the official denominator, evaluation-only TRAIN GT was used, and official test labels were not read. This is a G0 reference scope, not a clean G2 held-out result because MOT20-05 is excluded from G2_CLEAN_FULL_COVERAGE_V2.

2026-10-19 — Post-launch runtime snapshot after the G0 scorer: the clean G2 owner remains RUNNING at optimizer step 24/18227, with 16 global samples, 17610 effective global tokens, finite loss 1.7098015, gradient norm 2.69265 and 17.3715 GiB peak device memory in its latest event. J1 remains RUNNING on GPU4/GPU6 at durable step 650/1200 with 10368 rows seen; no terminal receipt or quality claim is made for either course. The G2 and J1 owners were not restarted or modified.

2026-10-19 — J2 candidate-source audit corrected the training entrypoint: the balanced onejev_unified_matched600_v2/native manifest is preserved but rejected for candidate-driven J2 because its MOT rows use legal TRAIN GT proposals plus explicit synthetic fragments and its OVMOT rows use registered train-Base teacher crops; only its repeated 346-row RMOT component is native GroundAnything. SHA 136d029d212d831e1c72cf7c7fe538dc9b246e8edc0febb9bb14ac1abf662156, 125286 rows. A separate streaming inventory records the usable sources: 131214 native MOT rows from 7 MOT17 sequences (SHA b60d029b28a1d5bb101361ea548cf07e48b41033ccfcbe20d78f32cc7f0ba7b4), 346 native RMOT rows from 72 fit frames (SHA cdb1e38342f2bf467c0d288e532b840964328447020911043ded36968381e4a1), and 430 OVMOT teacher-control rows (SHA 48269098ac2a8ebafb5d0c50a7de9852e10bf50d37e087f71db6c6f74ecfd95b). No J2 optimizer was launched; native OVMOT candidates and broader legal coverage remain required after G1/G2 detector export.

2026-10-19 — Registered and started a GPU5 G1 continuation owner behind the active clean G2 run. Owner session 68355 is waiting on the authoritative G2 terminal receipt; it refuses to launch if G2 fails, trains a fresh 200-step projector from G0 only after GPU5 release, exports a new G1-GPU5 bundle without touching G1/G200, then runs fixed 96-frame G0/G200/G1 native controls. This is a queued executable owner, not a claim that G1 has completed.

2026-10-19 — Re-audited the exact clean G2 manifest after launch. The read-only audit found 583,251 rows and unique IDs, zero MOT20-05 rows, zero RMOT placeholder questions, and no cross-source absolute media-path overlap. Its raw MOT20 validity inventory still names the excluded MOT20-05 directory because the audit scans the source tree; the clean row filter remains MOT20-01/02/03 only. Evidence: `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/g2_clean_data_contract_audit_v2.json`. No active process or checkpoint was modified.

2026-10-19 — Clean G2 deployment export handoff registered: queue_g2_clean_bundle_export.py is the sole CPU-only downstream owner. It waits for the exact clean G2 terminal receipt and the GPU5 G1 queue receipt, then exports a fresh 569-tensor GroundAnything-DLM-G2-CLEAN-V2 bundle with the manifest SHA and immutable source check. It refuses overwrite and records explicit dependency failure states; no detector or evaluation claim is made before G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED.
- 2026-10-20 — GroundJEV dependency preflight passed without allocating a GPU or
  starting a process. The G1 plan/trainer, G2 bundle queue, independent held-out
  queue, OneJev recovery queue, registered model/checkpoint/manifest paths and
  external OneJev source all exist. All four owner states parse and report
  waiting states with `failed=null`. Receipt:
  `docs/GROUNDJEV/DEPENDENCY_PREFLIGHT_20261020.{json,md}`.
- 2026-10-20 22:59 +0800: the protected G2 Clean owner remains live on
  physical GPUs 3/5 and advanced to event step 10,710/18,227. The final
  receipt is still absent, the four dependency queues remain live and waiting,
  and the measured linear remainder is approximately 23.9–24.8 hours.

2026-10-20 23:02 +0800 — GroundJEV status refresh: protected G2 Clean torchrun remains live on physical GPUs 3/5 and reached event step 10726/18227 with finite loss 0.224377 and 12.829 GiB peak memory; final receipt absent and latest durable checkpoint remains adapter-step010500/step10500. G1, G2 bundle export, independent GPU7 held-out, and full-source OneJev resume owners remain live with failed=null and waiting states. No active MOT-only optimizer process is present. This is a runtime evidence refresh; no metric claim or protocol change.
2026-10-20 23:04 +0800 — GroundJEV TrackJEV runtime preflight completed in the registered OneJev Python 3.10 environment: compileall passed for GroundJEV code/scripts, and decision-layer, model, and native adapter self-tests all returned PASS. No GPU was allocated and no tracking metric claim was made; receipt is docs/GROUNDJEV/TRACKJEV_RUNTIME_PREFLIGHT_20261020.{json,md}. The protected G2 owner remained untouched.

2026-10-20 23:05 +0800 — Read-only G2 Clean live-event sanity passed through step 10746/18227: 10746 events strictly increasing, all loss/gradient/memory/LR values finite, and every event retained the registered 8 micro-batch / 16 global-sample contract. Final receipt remains absent; no metric claim. Receipt: docs/GROUNDJEV/G2_LIVE_EVENT_SANITY_20261020.{json,md}.

2026-10-20 23:06 +0800 — Independent G2 held-out manifest preflight passed for MOT20-05 validation-half (1657), RMOT single0007 (800), RMOT multi0007 (800), and TAO OVMOT video105 (40): task flags, frame contracts, monotonic unique frame numbers, and all local image paths verified. Queue still performs exact TRAIN overlap audit before inference; no metric claim. Receipt: docs/GROUNDJEV/G2_HELDOUT_MANIFEST_PREFLIGHT_20261020.{json,md}.
2026-10-20 — Fixed `tools/groundjev/validate_full_source_cpu_completion.py` runtime portability: registered CPU-plan receipts and source manifests live under `/data2/user/LocateMOT_GroundJEV` while the project worktree is `/data2/user/LocateMOT`. The validator now resolves project/runtime artifacts deterministically and reports the terminal receipt path. Targeted rerun passed: 459027 rows, zero drops, 28690 planned updates, world size 4, exact serial/parallel plans, and RMOT exposure audit retained; actual optimizer updates remain 0 for this CPU validation.

2026-10-20 23:10 +0800 — Read-only resource gate audit: registered OneJev resume requires exclusive GPUs 4/5/6/7; the gate is not satisfied because existing compute PIDs remain on requested devices (GPU5 also hosts protected G2). Free memory is above 32 GiB but sharing was not enabled; no foreign or GroundJEV process was stopped. Receipt: docs/GROUNDJEV/RESOURCE_GATE_AUDIT_20261020.{json,md}.
2026-10-20 — Improved GroundJEV waiting-owner observability without changing the exclusive GPU contract: the GPU7 independent held-out queue and four-GPU OneJev resume queue now persist every resource probe and an explicit `resource_gate` record (`ready`, rule, `shared_memory_mode=false`) before sleeping. Both scripts passed py_compile and git diff --check; active owner processes were not restarted.
2026-10-20 23:13 +0800 — G1 GPU5 tracking200 input/schedule preflight passed in the registered GroundJEV runtime without GPU allocation: 112 rows, 56 pairs, 72 unique images, 884 native proposals, 628 aligned regions; the GPU5 plan exactly matches the preserved GPU6 reference schedule and retains identity-loss coefficients 0.1/0.1. No optimizer update or metric claim. Receipt: docs/GROUNDJEV/G1_GPU5_INPUT_PREFLIGHT_20261020.{json,md}.
2026-10-20 23:14 +0800 — Registered existing real-input evidence: OneJev-0.8B loaded on a real MOT17 frame (GPU6, 9.25 s load, 0.287 s decision, 3.12 GB peak) and the preserved native DLM/OneJev four-frame MOT17-02 functional receipt covers frames 302–305 with strict wire-error and no-fabricated-score guards. These are plumbing checks only, not tracking or G1/G2 metric claims. Receipt: docs/GROUNDJEV/TRACKJEV_REAL_INPUT_EVIDENCE_20261020.{json,md}.

2026-10-20 23:15 +0800 — Read-only G2 durable checkpoint protection passed at step 10500: latest_training_state.pt and adapter-step010500.pt are present, manifest SHA matches f682519a…, optimizer/scheduler/sampler/per-rank RNG fields are retained, and final receipt remains absent. Receipt: docs/GROUNDJEV/G2_DURABLE_CHECKPOINT_PROTECTION_20261020.{json,md}.

2026-10-20 — Generated current GroundJEV goal completion matrix from live receipts/owner states. It explicitly leaves G2 terminal training, G1 GPU5 adaptation, bundle export, independent held-out metrics, and full-source OneJev terminal receipt incomplete; no preliminary result was promoted. Receipt: docs/GROUNDJEV/GOAL_COMPLETION_MATRIX_20261020.{json,md}.

2026-10-21 — GroundJEV redirection checkpoint: verified project root `/data2/user/LocateMOT` and protected formal G2 Clean. The live two-rank G2 owner reached event step 11735/18227 with finite loss 0.210247, gradient norm 9.833711, peak 12.828929 GiB; latest durable checkpoint remains step11500 and the 500-step window is 11.617 s/step (linear remainder about 20.9 h). G0 MOT20-05 registered 1657-frame validation-half export remains terminal; its fresh native-source control is retained without a full TEST claim. No live independent MOT-only optimizer process exists. The historical four-GPU independent OneJev resume was waiting before launch at durable step3000, so it was safely paused and its checkpoint/optimizer/RNG/config/logs preserved; a code-level pause marker guard now blocks accidental auto-start unless `--resume-paused` is explicitly supplied. G1 GPU5, clean bundle export, and independent held-out remain waiting on their registered dependencies. No formal G2/G1/held-out performance claim was made. Evidence: `docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021.{json,md}` and the queue owner state.

## 2026-10-21 — corrected real three-task OneJev source registration

The historical full-source manifest was rejected for the independent comparison because its source inventory included MOT20-05 rows. I streamed and audited a replacement manifest at `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/real_three_task_joint_v1/`: 131,214 native MOT identity rows, 346 native RMOT membership/presence rows, and 430 registered OVMOT Base teacher-control rows (131,990 total). Duplicate IDs, train split, family filters, row counts, media paths, forbidden MOT20-05/test markers, and source SHA-256 values passed. The registered TAO native archive (40 frames/35,228 candidates) has no category supervision, so native OVMOT training rows remain pending and no relabeling or optimizer launch was performed. The old independent four-GPU OneJev queue remains paused.

## 2026-10-21 — Phase C causal memory input contract

In the isolated `research/groundjev-coupled-single-backbone-20261021` worktree, added an opt-in `inject_identity_memory` contract. It inserts projected causal identity tokens before the registered current-answer index, preserves the original prompt tensors exactly when disabled, and returns the offset needed to shift DLM labels. Eight focused CPU contract tests passed, including projector gradient flow and insertion order. This is an engineering input contract only; the GroundAnything decoder hook, real GPU C0/C1/C2 replay, generated-box recall, and HOTA/IDF1 effects remain unmeasured.

## 2026-10-21 — corrected three-task OneJev owner registered

The historical independent four-GPU resume remains paused. A separate owner `real_three_task_joint_v1_queue` was registered and started in a waiting state. It uses the audited 131,990-row manifest, a new run directory, and the dedicated OneJev environment; it will not claim GPUs 4/5/6/7 until G2 Clean, G1 GPU5 adaptation, bundle export, held-out evaluation, and an exclusive resource window are terminal. The run is explicitly a Base OVMOT teacher-control comparison while native TAO category rows remain pending; it cannot be reported as the final single-backbone GroundJEV result.

## 2026-10-21 — joint queue configuration regression fixed

The dedicated OneJev parser rejected four audit-only keys accidentally placed in the runtime training JSON. I removed those keys from `real_three_task_joint_v1_runtime.json` and kept the same provenance in the queue state and research records. The registered `.venv-onejev-py310` parser now accepts the config and resolves the corrected source manifest, one epoch, and global 16-row accumulation.

## 2026-10-21 — three-task CPU preflight found explicit overlength rows

The registered OneJev CPU plan rendered all 131,990 rows in 267 seconds. At the 32,768-token training contract, 129,891 rows are trainable and 2,099 high-candidate MOT identity rows exceed the limit (p50 12,720; p90 25,689; p99 47,023; maximum 96,734). The new queue records this exclusion explicitly and cannot report full-source exposure. A separate compacting/long-row course remains required.

## 2026-10-21 — compact zero-drop three-task source v2
The v1 native MOT/RMOT/OVMOT source plan was valid but excluded 2,099 high-candidate MOT identity rows over the 32,768-token limit. The cause was repeated dense N×N IoU and center-distance CSV matrices in every MOT row, not missing media or a changed candidate source. Added the explicit `compact_competition_v1` manifest transform: native JSONL remains immutable; exact row IDs, targets, media paths and candidate boxes are preserved; all positive-IoU edges retain native rounded IoU and center distance; zero-IoU geometry is derivable from the retained boxes. The v2 CPU plan rendered all 131,990 rows with zero drops (p50/p90/p99/max 7,939/12,173/16,050/28,156 tokens; 8,250 world-4 steps), with no GPU or optimizer updates. The old v1 waiting owner was superseded before any training; a separate v2 owner is waiting on G2/G1/bundle/held-out gates and exclusive GPUs. Native TAO category rows remain pending, so this is still a registered Base teacher-control course rather than a final OVMOT claim.

## 2026-10-21 — native YesTrack v2 queued behind an exclusive GPU7 gate

Registered the real native YesTrack corpus with 726 train rows and 390 fixed
validation rows (train SHA
`065337d2fedca2377d02b94e57c9eb3778aeca09aeb1a8b9463429fa15bbee66`, validation
SHA `d63a9fb0d54b8a1dc5d12b0960a48cb75c56ab786671f955fb317e196b3033d8`). The
CPU rendering plan passed with zero rows over 32,768 tokens (max 26,159; 91
planned optimizer steps per epoch, 273 over three epochs). Added a real GPU
owner that waits for GPU7 to have no
compute PID and at least 32 GiB free, trains native multimodal LoRA for three
epochs, then evaluates frozen base and tuned adapter on identical held-out rows
using AUROC, PR-AUC, accuracy, Brier, ECE, false rejection and false
acceptance. The owner does not stop other users and writes terminal receipts
only after training and both evaluations succeed. At registration GPU7 was
occupied, so no YesTrack metric is claimed; prior 24-row readout controls remain
engineering evidence only. Tracking HOTA/AssA/IDF1/IDSW still requires a later
fixed-candidate physical-baseline integration.

## 2026-10-21 — live G2 Clean checkpoint refresh at step 12002

A single authoritative read of the formal event log found G2 Clean at event step 12002/18227, with finite loss 0.298694, gradient norm 10.215737, peak memory 12.829 GiB, and the registered 8-micro-batch/16-sample contract. The periodic adapter checkpoint is `adapter-step012000.pt` and `checkpoints/latest_training_state.pt` is present; the terminal receipt is still absent. The recent 500-step rate is 11.615 s/step, a linear remainder of approximately 20.1 hours. G1, bundle, held-out, YesTrack, and zero-drop three-task v2 remain waiting owners; no metric claim was promoted and no protected process was modified. Evidence: `docs/GROUNDJEV/STATUS_SNAPSHOT_20261021_G2_STEP12002.{json,md}`.

## 2026-10-21 — process-scope audit confirms no live MOT-only/J1 optimizer

A fresh process-table audit found G2 Clean as the only active GroundJEV optimizer (event step 12012/18227 on GPUs 3/5). G1 GPU5, deployment bundle, independent held-out, YesTrack and zero-drop three-task v2 are live dependency/resource waiters. No independent MOT-only or legacy J1 OneJev optimizer is live; historical checkpoints and logs remain preserved. Foreign PRISM jobs were untouched and no process was stopped. Evidence: `docs/GROUNDJEV/PROCESS_AUDIT_20261021.{json,md}`.

## 2026-10-21 — G2 step-12000 recovery checkpoint validated

Read the protected G2 `latest_training_state.pt` and `adapter-step012000.pt` with the registered local trainer environment. Both load successfully and carry format `groundanything-g2-projector-qwen3-lora-v2-strict`, step 12000, manifest SHA `f682519a…06e71`, 293 trainable tensors/signature entries, optimizer and scheduler state, RNG state, two rank RNG paths, sampler position 96000 and world size 2. The terminal receipt is still absent, so this proves recoverability only and no performance claim. PyTorch 2.6's default weights-only reader rejects the stored NumPy RNG object; the controlled audit used trusted `weights_only=False` and did not modify files. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_VALIDATION_20261021.{json,md}`.

## 2026-10-21 — G2 Clean dataset scope audited

The protected G2 Clean manifest contains 583,251 TRAIN rows from AerialMind (69,358), AnimalTrack (11,543), DanceTrack (41,796), MOT17 (15,948), MOT20 (5,616), RMOT26 (294), Refer-KITTI-V2 (379,341), SportsMOT (28,574) and TAO-Amodal (30,781). It is GroundAnything perception adaptation (projector + Qwen3 LoRA), not J1/OneJev identity training. MOT20-05 is explicitly held out; BURST is excluded for lack of a legal TRAIN image root; TAO novel annotations are excluded. Manifest SHA is `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`. Evidence: `docs/GROUNDJEV/G2_CLEAN_DATASET_SCOPE_20261021.{json,md}`.

## 2026-10-21 — GroundJEV Phase A real G2 single-backbone smoke

Implemented the isolated `locatemot/groundjev/coupled` path: frame/model-provenance checked shared state, G2 feature bridge, typed dynamic identity slots, dot-product JEV readouts, explicit Yes/No/admit/lifecycle/memory heads, causal identity memory, safe lifecycle control, assignment-conditioned memory writes and a one-to-one runtime. CPU backward and causal runtime tests passed. A real GPU smoke loaded the official GroundAnything visual module from the registered G2 base checkpoint (`3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`), encoded one MOT17 image once, pooled two real boxes into 2x2560 region tokens, and completed finite typed loss/backward/optimizer-step: total loss 4.179936, 3.07 GiB peak allocated, 14.43 s, bridge forward count 1, independent VLM instances 0. This is engineering evidence only; no tracking or scientific gain is claimed, and contextual language-state return plus identity-conditioned grounding remain pending. The first py310 attempt failed before model construction because `pybase64` is absent; the successful run used the existing GroundAnything serving environment and the same immutable checkpoint. Evidence: `docs/GROUNDJEV/PHASE_A_REAL_SMOKE_20261021.{json,md}`.

## 2026-10-21 — GroundJEV Phase B choice/Yes-No pilot completed

A bounded real pilot reused 32 MOT17-02-FRCNN TRAIN frames and 992 prior native GroundAnything proposal boxes, re-pooled once per frame with the G2 base visual checkpoint SHA `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`. The frozen G2 evidence fed typed Identity Choice, Admit, Yes/No, Lifecycle and Memory readouts; 24 frames received 192 optimizer updates and the final 8 frames were a same-video temporal development holdout. Identity accuracy changed train 0.0618->0.6720 and temporal-dev 0.0363->0.6492; admit accuracy changed train 0.1022->0.9247 and temporal-dev 0.1331->0.9153. An actual eight-frame replay without teacher memory produced three persistent predicted tracks with one cached G2 feature forward per frame. This is a teacher-forced engineering control, not an independent HOTA/generalization result; language hidden-state coupling and identity-conditioned grounding remain unmeasured. The first replay failed on the legacy/new memory field mismatch and was fixed without changing data or budget. Evidence: `docs/GROUNDJEV/PHASE_B_PILOT_20261021.{json,md}`.

Checkpoint receipt finalized for the Phase B pilot: `adapter-step000192.pt`, SHA256 `c96f27c9c8de5a01a36036c78c4e4a36c467bb96a0ca5f7d0eee27d93e32c904`; the binary remains local under the registered output directory and is not pushed to GitHub.

## 2026-10-21 — native G2 Clean hidden-state probe completed

The protected G2 Clean base plus `adapter-step012000.pt` (adapter SHA `f0dfe4f5efd6bd2f0e8ebe8f224ae060e8b0422b36571697fb4fa9efa6d85ae2`, step 12000, manifest SHA `f682519a…06e71`) completed the native GroundAnything DLM clean path on real row `mot17_MOT17-02-DPM_000001_dense_clean_v2` from MOT17 TRAIN. The real image was encoded once; the clean embedding and hybrid-language hidden state were both `[1, 1152, 2560]`, with 1,008 image-pad hidden vectors. No independent VLM instance was created. The probe used the formal BF16 autocast boundary and the existing process-local Transformers 5.17 `cache_position` compatibility shim. The first dtype and API failures were preserved and fixed at the probe boundary. This establishes native hidden-state availability for the shared-backbone work only; it is not a tracking metric, identity gain, grounding improvement, or generalization result. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_A_NATIVE_HIDDEN_PROBE_20261021.{json,md}`.

## 2026-10-21 — native G2 clean DLM coupled Phase C smoke completed

The new `NativeG2CleanDLMProvider` restores the protected G2 step-12000 adapter in memory, renders one real MOT17 TRAIN row, and uses one native clean DLM forward to produce both 1,008 image hidden tokens and a `[1, 2560]` non-image contextual token. The same-backbone bridge pooled two real candidate boxes, fused the contextual token into the typed GroundJEV readout, and completed a finite GPU backward plus optimizer step: total loss 5.533803, clipped gradient norm 73.300224, 12.059 GiB peak allocation, 16.91 s, bridge/native forward counts both 1, independent VLM instances 0. This proves the visual+clean-language coupling interface runs on a real checkpoint; it is still an engineering smoke with no HOTA/IDF1, grounding recall, identity gain or generalization claim. PyTorch emitted a BF16/FP32 RMSNorm fallback warning in the frozen language path; it did not produce a non-finite value and remains recorded in the receipt. The formal G2 run directory was untouched. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_NATIVE_DLM_COUPLED_SMOKE_20261021.{json,md}`.

## 2026-10-21 — native G2 context readout C1 pilot completed

A bounded eight-frame MOT17-02-FRCNN pilot used G2 step 12000 with the official DLM frozen. Each of 8 real frames had one native DLM forward; the downstream GroundJEV typed readout received the pooled image regions plus same-frame clean-language context. Four frames were train and four same-video temporal development, with previous-frame GT-matched region tokens as explicit teacher-forced causal slots. Twelve readout updates changed identity accuracy train 0.032258→0.725806 and temporal-dev 0.032258→0.403226; admit accuracy changed train 0.911290→0.911290 and temporal-dev 0.790323→0.911290. This is a bounded learnability control only: no HOTA/IDF1, independent-video or final tracking claim. The first attempt completed the loop but failed at receipt writing because the output directory was absent; the directory fix was the only rerun change and the failed attempt is excluded. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_NATIVE_CONTEXT_PILOT_20261021.{json,md}`.

## 2026-10-21 — active goal snapshot at G2 step 12153

The protected formal G2 Clean owner is alive on GPUs 3/5 at event step 12153/18227; recent event rate is 11.614 s/step and the linear remaining estimate is 19.6 h. The latest durable checkpoint is `adapter-step012000.pt`, with no terminal receipt. G0 MOT20-05 validation-half export is terminal for 1,657 frames (HOTA 0.073346961, IDF1 0.059598451, IDSW 1933); this remains a validation-half control rather than a full TEST claim. G1, bundle, held-out, YesTrack and compact three-task OneJev owners are alive and waiting on their registered dependencies/resource gates with no recorded failure; compact OneJev optimizer updates remain 0. The process audit found no live MOT-only optimizer. Evidence: `docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12153.{json,md}`.

## 2026-10-21 — merged Phase C causal memory input contract

Merged the isolated single-backbone memory-input implementation into the GroundJEV evidence branch. `MemoryTokenProjector` used by the typed readout remains compatible; the new `IdentityMemoryTokenProjector` and `inject_identity_memory` insert only causal identity tokens before the registered answer index, return the insertion offset for label shifting, and leave the disabled path tensor-identical to the G2 baseline. The focused contract tests passed in the serving PyTorch environment (3 tests) and the existing Phase A core functions passed (3 checks); coupled sources compile. This is an input-contract result only: the official DLM decoder hook, generated-box recall and C0/C1/C2 HOTA/IDF1 evaluation remain pending.

## 2026-10-21 — blocked accidental teacher-only OVMOT launch

The registered real three-task v2 manifest contains 131,214 native MOT rows, 346 native RMOT rows, and 430 OVMOT Base teacher-control rows; its 40-frame native TAO archive has no category labels and is validation-only. The waiting four-GPU OneJev owner had zero optimizer updates. I stopped that waiting owner before any launch, preserved its state snapshot, and marked it `PAUSED_FOR_NATIVE_OVMOT_ROWS`. The queue gate now requires an explicit native OVMOT TRAIN candidate/annotation join (`native_ovmot_training_rows=true`, `ovmot_training_mode=native`) and returns `WAITING_NATIVE_OVMOT_ROWS` otherwise. No three-task training or OVMOT gain is claimed; the historical source course remains preserved.

## 2026-10-21 — active goal snapshot at G2 step 12195

A single authoritative read of the protected G2 event log found event step 12195/18227 with finite loss 0.225930949, gradient norm 9.566365, peak memory 12.828853 GiB, and a 500-step rate of 11.618 s/step (linear remainder about 19.5 h). The latest durable checkpoint remains adapter-step012000.pt and the terminal receipt is absent. G0 MOT20-05 remains a complete 1657-frame validation-half natural-visibility control with HOTA 0.073346961 and IDF1 0.059598451; it is not a full TEST claim. The compact three-task queue is paused before optimizer launch because native OVMOT TRAIN rows are missing; its optimizer update count remains zero and the guard requires native OVMOT provenance/mode. No final G1/held-out/OneJev/YesTrack result was promoted. Evidence: docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12195.{json,md}.

## 2026-10-21 — native OVMOT TRAIN join pipeline prepared

Added `tools/groundjev/prepare_native_ovmot_train_join.py`, a CPU-only joiner for a future complete native GroundAnything TAO TRAIN stream. It validates `/frames/train/` provenance, requires native region features and GT-free model metadata, preserves candidate boxes, assigns official Base category targets only through deterministic one-to-one IoU >=0.5 matching to TAO TRAIN annotations, and emits the existing hierarchical category questions with on-demand real-image crop recipes. The registered TAO TRAIN source contains 18274 images, 54639 annotations and 765 Base categories mapped to the 1203-class vocabulary; 11 categories are explicitly unmapped. The current 40-frame/35228-candidate native TAO archive is validation-only and is rejected by the TRAIN path guard. `py_compile`, one-frame synthetic preflight, and validation-scope rejection passed; no GPU or optimizer update occurred. The compact three-task owner remains paused pending a complete native OVMOT TRAIN archive. Evidence: docs/GROUNDJEV/NATIVE_OVMOT_TRAIN_JOIN_20261021.{json,md}.

## 2026-10-21 — TAO TRAIN native OVMOT scope registered

Registered a CPU-only scope manifest for all 18274 TAO TRAIN images across 500 videos. The runtime manifest contains only absolute image paths and the fixed 1203-category query vocabulary; it carries `training_on_this_scope=true` for an explicit export gate but no GT fields, labels or evaluation data. `stream_tao_native_candidates.py` now requires `--allow-training-scope` for this manifest and rejects validation manifests with that flag; the scope gate was tested without importing the model service. The exporter was not launched because current GPUs remain occupied, so native candidate rows and optimizer updates are still zero. Evidence: docs/GROUNDJEV/TAO_TRAIN_NATIVE_SCOPE_20261021.{json,md}.

## 2026-10-21 — corrected process audit at G2 step 12235

A corrected process-table audit (the first username-filtered attempt was invalid because the account is `lyq`) found the protected G2 parent/ranks alive at 12235/18227 on GPUs 3/5, G1/bundle/YesTrack waiters alive, and no registered MOT-only or legacy J1 OneJev optimizer command. The compact three-task owner remains paused before optimizer updates because native OVMOT TRAIN rows are absent. Foreign GPU jobs were not touched. Evidence: docs/GROUNDJEV/PROCESS_AUDIT_20261021_G2_STEP12233.{json,md}.

## 2026-10-21 — active goal snapshot at G2 step 12241

The protected G2 Clean event log reached 12241/18227 with finite loss 0.195041649; the last 500-step rate is 11.626 s/step and the linear remainder is about 19.3 h. The durable checkpoint remains step12000 and no final receipt exists. G0 remains a 1657-frame MOT20-05 validation-half control (HOTA 0.073346961, IDF1 0.059598451); the compact three-task queue remains paused with zero optimizer updates pending native OVMOT TRAIN rows. Evidence: docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12241.{json,md}.

## 2026-10-21 — GroundJEV fixed-candidate MOT20-03 replay establishes failure boundary

Added a reproducible CPU evaluator for the immutable native GroundAnything candidate/region cache. The current 192-step coupled readout replayed all 128 registered MOT20-03 TRAIN frames (14,784 candidates, 128 cached feature calls) without a second VLM and generated 152 predictions on 76 frames. TrackEval returned HOTA 0.0279006409, DetA 0.0099894107, AssA 0.0779470005, IDF1 0.0052606731, IDSW 1, FP 1, FN 14,524; 11,462 candidates were explicit REJECT. The cache predates the current G2 Clean terminal checkpoint and overlaps TRAIN scope, so this is a fixed-candidate association/runtime diagnostic, not independent G2 or end-to-end evidence. The first evaluator attempts exposed only direct-entry `sys.path` and NumPy-2/old-TrackEval compatibility issues; both were fixed locally without changing metric code or inputs. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_B_FIXED_CANDIDATE_EVAL_20261021.{json,md}` and the registered output receipt.

The same-input historical compact control (checkpoint `rmot_native72_stage_v1/broad_native_training/checkpoint.pt`) returned HOTA 0.336514881, DetA 0.309884333, AssA 0.365702041, IDF1 0.359321689, IDSW 159, FP 119 and FN 9,832. It is an offline multi-component reference, not a single-backbone result. The gap confirms that the coupled readout's 0.027900641 HOTA is a real association/admission failure and should trigger a bounded repair study rather than a long joint finetune.

## 2026-10-21 — Phase C causal memory prefill contract

Added `locatemot/groundjev/coupled/causal_memory.py`, which inserts causal identity tokens only at the terminal prompt boundary used by GroundAnything causal bbox generation, extends attention/position/cache inputs, and preserves the disabled path by tensor identity. A seven-test CPU contract suite passed; a fake language tower showed a changed next-token logit after memory insertion with exactly one prefill call and no second VLM. Real G2 GPU prefill, correct-versus-shuffled memory box generation, and HOTA/IDF1 effects remain pending because GPUs 3/5 are protected by formal G2 and the other devices have foreign compute owners. Evidence: `docs/GROUNDJEV_COUPLED/PHASE_C_MEMORY_PREFILL_CONTRACT_20261021.{json,md}`.

## 2026-10-21 — authoritative G2 progress refresh at step 12294

A single non-polling audit found the protected formal G2 Clean parent/ranks alive on GPUs 3/5 at event step 12294/18227. The event is finite (loss 0.205467, gradient norm 9.551106, peak memory 12.828834 GiB); `adapter-step012000.pt` and `latest_training_state.pt` remain the latest durable checkpoints and `final_receipt.json` is absent. G1, bundle, held-out, YesTrack and compact three-task owners remain dependency/resource waiters. No protected process or experimental protocol was changed. The causal-memory prefill contract remains engineering-only pending an exclusive GPU window for real generated-box comparison.

## 2026-10-21 — fixed-candidate action-distribution repair completed

The first MOT20-03 fixed-candidate replay showed a candidate pool with 86.85% GT-box coverage at IoU >= 0.5, but the short MOT17-trained readout selected `REJECT` for 77.54% of 14,784 candidates and created only two births. A bounded repair trained only on the first 64 frames of the registered MOT20-03 TRAIN clip, weighted identity-choice classes from that training half with an inverse-frequency cap of 4x, and froze the final 64 frames before optimization. After 180 CPU updates, temporal-dev readout identity accuracy was 0.871720 and admit accuracy 0.903860; predicted-memory runtime replay returned HOTA 0.302815, DetA 0.235861, AssA 0.388886, IDF1 0.257729, IDSW 18, FP 3, FN 5,634 on 1,891 predictions across 61 frames. The 64 feature calls used the cached single-backbone state and no second VLM. This is a TRAIN-scope mechanism repair, not an independent benchmark or G2 Clean result; its scope and teacher-forced classification boundary are explicit in `docs/GROUNDJEV_COUPLED/PHASE_B_FIXED_CANDIDATE_REPAIR_20261021.{json,md}`.
## 2026-10-21 — redirect goal audit at G2 step 12326

A single authoritative audit confirmed the registered MOT20-05 G0 native export is terminal for 1,657 validation-half frames (HOTA 0.073347, IDF1 0.059598) and found no live GroundJEV MOT-only optimizer; no historical process was stopped or overwritten. Formal G2 Clean remains alive on GPUs 3/5 at step 12326/18227, with a 500-step rate of 11.6 seconds per step and a linear remainder of about 19.0 hours. The latest durable checkpoint is `checkpoints/adapter-step012000.pt`, and the final receipt is absent. G1 GPU5, bundle, held-out and YesTrack owners remain waiting; the corrected real three-task OneJev owner remains paused at zero optimizer updates because native TAO TRAIN OVMOT rows are not present. Evidence: `docs/GROUNDJEV/ACTIVE_GOAL_STATUS_20261021_G2_STEP12326.{json,md}`.

## 2026-10-21 — cross-disk native OVMOT archive audit at G2 step 12345

A read-only search of `/data0/user`, `/data1/user` and `/data3/user` found no complete TAO TRAIN native candidate archive. The only existing native TAO candidate archive remains a 40-frame `frames/val` stream and is ineligible for TRAIN joining. The corrected real three-task queue remains paused with zero optimizer updates; no validation archive was relabeled and no teacher-only fallback was launched. Evidence: `docs/GROUNDJEV/NATIVE_OVMOT_ARCHIVE_AUDIT_20261021.{json,md}`.

## 2026-10-21 — GroundJEV joint identity corpus rebuild contract and Phase B ablations

Added `tools/groundjev/coupled/build_joint_identity_corpus.py` and its focused
contract tests. The builder normalizes registered MOT/RMOT/OVMOT rows without
inventing boxes or targets, records candidate provenance and causal memory,
rejects future/test evidence, and has an explicit native-only gate that rejects
teacher crops. A real 12-row sample from the registered Phase-B corpus produced
5 MOT, 3 RMOT and 4 OVMOT rows; 8 rows had native candidates and 4 remained
explicit Base teacher/empty-candidate rows, so `native_ovmot_training_rows` was
false. The existing G2 step-11000 B0/B1/B2 bounded readout receipts were
consolidated into `docs/GROUNDJEV_COUPLED/PHASE_B_ABLATIONS_20261021.{json,md}`.
The three 200-update arms and 8-frame causal replay are engineering evidence
only; no official tracking metric or OVMOT native-training claim was promoted.

The native-only corpus gate was then strengthened to reject rows carrying an
explicit `gt_in_model_input`/`GT_IDs_in_model_input` flag even when their
candidate-source string looks native. The focused contract suite now has 12
passing checks; no source rows or training queues were changed.

The Phase-B joint corpus was also audited against all 583,251 G2 Clean TRAIN
rows. There are 678 exact image-path overlaps covering 290 images (221
identity-pair, 119 occlusion-recovery, 338 RMOT); the 430 OVMOT teacher rows
have no exact overlap. The audit reads no official-test labels and marks the
joint corpus as TRAIN-scope controls only. Evidence:
`docs/GROUNDJEV_COUPLED/DATA_OVERLAP_AUDIT_20261021.{json,md}`.

## 2026-10-21 — targeted native TAO archive rescan

Re-read every discovered TAO candidate archive under `/data2/user/LocateMOT_GroundJEV/outputs/groundjev`. The two small candidate controls and the 40-frame streams all resolve to `frames/val`; the larger 40-frame stream has 35,228 candidates and a native feature cache but is explicitly a validation stream. One full Base/Novel stream is empty and failed, and the registered 18,274-image/500-video TAO TRAIN manifest has `native_candidate_export_started=false` and `native_ovmot_training_rows=false`. No complete TAO TRAIN native candidate archive exists, so the corrected three-task queue remains paused at zero optimizer updates. Validation outputs were not relabeled and no teacher-only fallback was promoted. Evidence: `docs/GROUNDJEV/NATIVE_OVMOT_ARCHIVE_RESCAN_20261021.{json,md}`.
## 2026-10-21 — single authoritative G2 progress refresh at step 12417

One non-polling status read found the protected formal G2 Clean ranks alive on GPUs 3/5 at event step 12,417/18,227 with finite loss `0.1803909168`. The latest durable adapter remains `checkpoints/adapter-step012000.pt`; `checkpoints/latest_training_state.pt` is present and `final_receipt.json` is absent. G1, the deployment bundle, held-out evaluation and YesTrack remain dependency/resource waiters. The corrected three-task queue remains `PAUSED_FOR_NATIVE_OVMOT_ROWS` with zero optimizer updates. No protected process or experimental protocol was changed. Evidence is the live G2 run directory plus the refreshed matrix.

## 2026-10-21 — TAO TRAIN native export shards registered

Because the all-GPU snapshot was occupied by protected/foreign jobs, no native TAO export was launched. Instead, added `scripts/groundjev/prepare_tao_train_native_shards.py` and registered eight contiguous, disjoint, label-free manifests covering all 18,274 TAO TRAIN images (500 videos) with parent manifest SHA `835adaadff282088599987de4c5bc906e7d8aca826deb7426cda05598e9db3a4`. CPU coverage validation reports 18,274 covered frames, zero missing and zero duplicates. Every shard remains `native_ovmot_training_rows=false`; this only makes the future native stream resumable/parallel and does not unblock training. The first preflight exposed a slice-length assertion bug, which was fixed before accepting the v2 registration; the failed partial directory was preserved separately. Evidence: `docs/GROUNDJEV/TAO_TRAIN_NATIVE_SHARDS_20261021.{json,md}` and the external shard receipts.

Added `scripts/groundjev/merge_tao_native_shards.py` as the only promotion path after streaming. It checks every shard receipt, parent SHA, exact image order, GT-free metadata, and disjoint full coverage before producing a combined candidate archive. A preflight against the registered-but-unstreamed v2 directory rejected the missing shard receipt at shard 0, as intended; no native TRAIN rows or optimizer updates were created.

The merge contract also passed a synthetic four-frame/two-shard GT-free fixture, covering the successful receipt path without touching dataset or GPU state.

## 2026-10-21 — resumable TAO TRAIN native owner started

Added `scripts/groundjev/queue_tao_native_train_shards.py` and synchronized the runtime stream copy so shard-specific GroundAnything and native-feature ports are supported while the default validation ports remain unchanged. The owner is live in `WAITING_G2_CLEAN_SUCCESS`, restricted to physical GPUs 4/6, and treats any existing compute application as unavailable; GPUs 3/5 and 7 are excluded to protect G2/G1 and held-out/YesTrack. It will run the eight native TRAIN shards and strict merge only after the exact G2 terminal receipt. A first background launch was reaped by the terminal wrapper before writing state; the persistent-session launch wrote the waiting state and made no GPU or foreign-job change. Evidence: `docs/GROUNDJEV/TAO_TRAIN_NATIVE_QUEUE_20261021.{json,md}`.

## 2026-10-21 — authoritative G2 refresh at step 12442

The protected G2 Clean ranks remain alive on GPUs 3/5 at event step 12,442/18,227 with finite loss `0.2287370411`; the latest durable adapter is still `checkpoints/adapter-step012000.pt`, the training-state checkpoint is present, and `final_receipt.json` is absent. G1, clean-bundle, held-out and YesTrack owners remain live dependency/resource waiters; the corrected three-task owner remains paused at zero updates because native TAO TRAIN rows are not available. No protected process or protocol was changed.

## 2026-10-21 — native TAO join owner and native three-task launch gate repaired

Added a CPU-only owner for the complete native TAO TRAIN stream and a separate
resumable owner for the future real MOT/RMOT/OVMOT OneJev run. The join owner
waits for the strict eight-shard merge receipt, runs the checked-out
`prepare_native_ovmot_train_join.py` and `prepare_real_three_task_joint_manifest.py`,
and never launches a model or optimizer. Its initial state is
`WAITING_COMPLETE_NATIVE_TAO_STREAM`. The native three-task owner is now live
as a dependency waiter at `WAITING_NATIVE_OVMOT_MANIFEST`; it has launched no
GPU process or optimizer and will apply the joined-manifest plus terminal
G2/G1/bundle/held-out and exclusive-GPU gates later. The historical
teacher-control queue remains paused at zero updates.

During the audit, the join owner was corrected to resolve repository joiners
from `/data2/user/LocateMOT` because the runtime output tree does not mirror
those source files. The paused compact queue was also corrected to accept the
actual bundle terminal status `G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED` while
retaining the historical spelling for old receipts. No data, checkpoint, or
running process was changed. Evidence:
`docs/GROUNDJEV/TAO_NATIVE_JOIN_QUEUE_20261021.{json,md}` and
`docs/GROUNDJEV/REAL_THREE_TASK_NATIVE_QUEUE_20261021.{json,md}`.

## 2026-10-21 — TAO native owner corrected to use a terminal G2 bundle

The TAO TRAIN shard owner was audited before its first GPU launch. Its
original model constant pointed at the published G0 bundle, which would have
made a later native candidate archive impossible to attribute to G2. The owner
now waits for the exact G2 receipt, merges `adapter_projector_final.pt` into
the private `GroundAnything-DLM-G2-CLEAN-NATIVE-TAO-V1` bundle, validates the
569-tensor/export/manifest contract, and only then starts shard services. The
private bundle is never mixed with or written over G0, G1, or the ordinary G2
deployment bundle. The owner process had not started a shard, so it was
restarted with the corrected code while remaining in its G2 wait state.
Evidence: `scripts/groundjev/queue_tao_native_train_shards.py` and
`docs/GROUNDJEV/TAO_TRAIN_NATIVE_QUEUE_20261021.{json,md}`.

## 2026-10-21 — decoupled the terminal G2 bundle from the G1 comparison arm

The clean deployment owner previously waited for both the exact G2 receipt and
G1 GPU5 completion. That ordering could block independent G2 held-out
measurement and native G2 candidate export behind an optional comparison arm.
The owner now waits only for the exact-once G2 terminal receipt, performs the
same CPU-only 569-tensor merge into the separate G2-CLEAN-V2 bundle, and leaves
G1 as an independent GPU5 adaptation/control. The existing owner was still in
its pre-launch wait state, so no model or output was changed; it will be
restarted with the corrected gate. Evidence: `tools/groundjev/queue_g2_clean_bundle_export.py`.

The owner restart also exposed a resumability defect: its old waiting directory
was treated as an existing export and refused to start. The restart guard now
accepts the preserved pre-launch WAITING_G2_AND_G1_TERMINAL state (as well as
WAITING_G2_CLEAN_SUCCESS), refuses live duplicate owners and any partial
bundle, and resumes without deleting the historical state. The corrected owner
is live again in WAITING_G2_CLEAN_SUCCESS.

## 2026-10-21 — native three-task owner no longer logically depends on G1

The new real native MOT/RMOT/OVMOT owner was narrowed to require the native
TAO manifest, terminal G2, independent G2 bundle and held-out receipts. G1
GPU5 adaptation remains an optional comparison receipt; its physical GPU5
ownership is still respected by the four-GPU resource probe. This preserves
G1/G200/G0 controls without making the final-G2 joint route wait on a separate
adaptation arm. The historical paused teacher-control queue was not changed.

## 2026-10-21 — G1 GPU5 waiting owner made resumable

The G1 GPU5 owner had the same pre-launch recovery defect: after a process
restart, its preserved WAITING_G2_CLEAN_SUCCESS directory was treated as an
existing artifact and the owner could not be resumed. The startup guard now
rejects a live duplicate owner and any partial output, but accepts the empty
preserved waiting directory and records `resumed_wait_owner`. The old waiting
owner was stopped and restarted with the registered
`GroundAnything-DLM-TrackProjG1-GPU5-V2` bundle path. It remains at
`WAITING_G2_CLEAN_SUCCESS`; no G1 optimizer or GPU process has started.

## 2026-10-21 — native three-task owner made resumable

The real native MOT/RMOT/OVMOT owner had a weaker restart boundary: a dead
waiting process could rewrite its queue directory without distinguishing a
clean wait from partial training output. Startup now accepts only the known
pre-launch waiting states, preserves the optional runtime config when it was
already created, rejects unexpected files, refuses live duplicates, and records
`resumed_wait_owner`. The existing owner is being restarted only in its
CPU-only `WAITING_NATIVE_OVMOT_MANIFEST` state; no optimizer or GPU process is
started.

## 2026-10-21 — YesTrack waiting owner made resumable

The native YesTrack v2 owner was still waiting for an exclusive GPU7 and had
no optimizer or evaluation artifacts. Its startup guard now resumes only the
known empty `WAITING_GPU7_EXCLUSIVE` queue (and the pre-created `args.json`),
rejects partial queue/training outputs, refuses live duplicates, and records
`resumed_wait_owner`. The waiting owner is restarted without allocating GPU7;
the independent YesTrack run remains pending.

## 2026-10-21 — held-out owner made resumable

The GPU7 G0/G2 independent held-out owner was still CPU-only in
`WAITING_G2_CLEAN_BUNDLE_AND_GPU7`; its directory contained only the current
wait receipt and the preserved historical pre-MOT receipt. Startup now accepts
that exact clean wait state, preserves the historical receipt, rejects any
unexpected partial evaluation outputs, refuses live duplicates, and records
`resumed_wait_owner`. The owner is restarted without allocating GPU7.
## 2026-10-21 — G2 progress evidence refreshed

The protected formal G2 Clean owner remains live on physical GPUs 3/5. The
latest observed training event is step 12,567/18,227 with finite loss
0.3145263; `adapter-step012500.pt` and `latest_training_state.pt` load with
the registered clean manifest SHA. No terminal receipt or final adapter exists
yet, so no downstream bundle, held-out metric, G1 update, or joint optimizer
update is claimed. The waiting owners remain untouched and recoverable.

## 2026-10-21 — G2 progress evidence refreshed (step 12578)

An authoritative event read at 05:04:48 +0800 records the protected formal G2
Clean run at step 12,578/18,227 with finite loss 0.24055790062993765. The two
distributed ranks are still the only active optimizer in this route on physical
GPUs 3/5. The latest durable checkpoint remains `adapter-step012500.pt` with
`latest_training_state.pt`; `final_receipt.json` and a terminal adapter are
still absent. No G1, bundle export, held-out metric, YesTrack update, or native
three-task update is claimed before that terminal receipt. This is evidence of
continued healthy progress only; the prior historical result boundaries remain.

## 2026-10-21 — GroundJEV coupled regression refreshed

The bounded coupled-core regression was rerun after the queue-owner changes: 13
unittests covering joint identity corpus, Phase A core, memory-conditioned input,
fixed-candidate contract, and causal-memory prefill all passed. The registered
G1/G2-bundle/TAO/native-joint/YesTrack queue entrypoints also passed `py_compile`,
and the coupled plus TrackJEV packages passed `compileall`. This verifies the
recoverable launch surface only; it does not promote any model or imply a
tracking metric while G2 lacks its terminal receipt.

## 2026-10-21 — waiting-owner source resolution audit

A read-only `/proc` audit of the seven registered waiting owners confirmed that
all commands run with cwd `/data2/user/LocateMOT`: the G1 and G2-bundle owners
resolve `tools/groundjev/...`, while the TAO-shard, held-out, YesTrack, native
join, and native three-task owners resolve `scripts/groundjev/...` or the
registered join entrypoint from this project root. The sibling runtime checkout
was not used to launch these owners. This closes a reproducibility ambiguity
without changing any owner, checkpoint, or GPU allocation.


## 2026-10-21 — G2 event-integrity audit (step 12594)

The formal G2 event stream contains 12,594 optimizer events from step 1 through
step 12,594 with no step gaps. Loss, gradient norm, memory, learning rates, and
all other recorded numeric fields are finite across the stream. The recent
throughput is stable at 309.8 steps/hour over 100 steps, 311.9 over 500 steps,
and 310.9 over 1,000 steps. Checkpoint events are present at each 500-step
boundary through `adapter-step012500.pt`; the terminal receipt remains absent.
This supports healthy continuation and an estimated 5,633 remaining updates at
roughly 311 steps/hour, but it is not a model-quality or generalization claim.

## 2026-10-21 — tightened G1 post-G2 terminal gate

The resumable GPU5 G1 owner previously accepted a G2 receipt with steps at or
above the registered budget and did not explicitly require the terminal adapter
file. Its post-G2 gate now requires status
`G2_EXACT_COVERAGE_TRAINING_COMPLETE`, exactly 18,227 steps, the registered clean
manifest SHA, and `adapter_projector_final.pt`. A temporary-receipt regression
confirmed that missing adapter and wrong-step receipts are rejected while the
complete contract is accepted. The live waiting owner was not restarted or
modified; this patch protects future recovery and launch paths.


## 2026-10-21 — G2 progress snapshot (step 12610)

The protected G2 Clean run reached step 12,610/18,227 at 05:10:55 +0800 with
finite loss 0.2049703631. The three expected training processes remain live and
`final_receipt.json` is still absent. This snapshot changes only progress
evidence; no downstream model, held-out metric, or generalization claim is
made.

## 2026-10-21 — coupled runtime lifecycle execution repair

Phase-A review found that the single-backbone runtime decoded lifecycle logits
but ignored them: every assigned track was forced to `ACTIVE`, every absent
track only accumulated `LOST`, and an assignment-conditioned memory write could
reactivate a track whose lifecycle decision was `LOST`.  The smallest aligned
fix now executes `ACTIVE/REACTIVATE/LOST` decisions, treats `TERMINATE` on a
matched observation as recoverable `ACTIVE`, blocks memory writes for
`LOST/TERMINATE`, and applies a registered timeout before irreversible
termination of an unmatched track.  A controlled causal replay verified one
birth, a matched `LOST` decision with a held memory write, and timeout
termination; the one-feature-forward and memory-causality regressions still
pass.  This is a runtime correctness repair, not a tracking-performance claim.

## 2026-10-21 — GPU gate for the next real pilot

The single resource audit found all eight A800 GPUs occupied.  Physical GPUs
3/5 remain the protected G2 ranks, GPU7 retains an unrelated compute owner, and
the remaining cards also have active compute workloads; no card has an
exclusive launch window for the Phase-B real-G2 pilot.  The latest durable G2
adapter remains `checkpoints/adapter-step012500.pt` with
`latest_training_state.pt`; the terminal receipt is still not present.  The
pilot was therefore not started on a shared card, and no process was killed or
reused.  This preserves the registered training contract while the existing
G2 completion waiter remains the only long-running observer.

## 2026-10-21 — corrected G0/G2 evidence boundary for early coupled pilots

An audit of the historical Phase-A smoke and Phase-B readout pilot found that
their `checkpoint` was the unadapted `GroundAnything-DLM` bundle and neither
receipt recorded a G2 adapter.  The metrics and loss values are preserved as
G0 native-visual controls only; they are not G2 Clean evidence and are not used
for a G2 performance claim.  The future Phase-A and Phase-B entrypoints now
require both an explicit adapter and the native clean-DLM loader, and record
the adapter/runtime provenance.  This fixes a provenance error without
overwriting the historical control artifacts.

## 2026-10-21 — protected G2 progress snapshot (step 12701)

The authoritative event stream now contains contiguous rows through step
12,701/18,227; the latest row at 05:28:19 +0800 has finite loss
0.2224419505.  The durable checkpoint set still ends at
`checkpoints/adapter-step012500.pt` with
`checkpoints/latest_training_state.pt`, while `final_receipt.json` is absent
and `trainer_contract.json` remains `G2_FORMAL_TRAINING_RUNNING`.  The launcher
is still live.  This is only a healthy-progress record; it does not promote a
checkpoint or claim a tracking metric.

## 2026-10-21 — native three-task training step-contract audit

The registered `queue_real_three_task_native.py` writes `epochs=1` and
`max_steps=0` into its generated config.  The pinned `train/sft.py` contract
interprets a falsy `max_steps` as the full planned epoch
(`steps_per_epoch * epochs`), rather than a zero-step plan.  The queue then
requires the terminal `done` record to report `examples == native manifest
rows` and a non-missing terminal record before declaring completion.  This
keeps the real MOT/RMOT/native-OVMOT run from being mistaken for a CPU-only or
empty optimizer launch.

## 2026-10-21 — G1 GPU5 plan provenance repair

The G1 queue is explicitly bound to physical GPU5, but the older
`native_ground_tracking200_20261014.json` retained a historical
`physical_training_GPU: 7` field.  That plan is preserved unchanged; the G1
queue now references a versioned GPU5 plan with matching physical-device
metadata and an explicit queue-owner provenance note.  The corrected plan will
be read only when its exact G2 terminal gate opens.

The already-running wait-only owner had loaded the old plan path in memory, so
it was safely replaced before any GPU work began (old PID 516321 exited; new
wait owner PID 579251 is `WAITING_G2_CLEAN_SUCCESS`).  Its output directory,
bundle, controls and historical state were preserved; the new owner records
`native_ground_tracking200_gpu5_20261021.json` and still owns no GPU.

## 2026-10-21 — protected G2 throughput snapshot (step 12731)

The formal G2 event stream is contiguous through step 12,731/18,227 with no
non-finite loss, gradient, memory or learning-rate fields.  The measured rates
are 313.73, 312.31 and 311.00 steps/hour over the last 100, 500 and 1,000
steps, respectively.  This leaves 5,496 updates and a linear estimate of
about 17.5–17.7 hours.  `final_receipt.json` and the terminal adapter remain
absent, so this is a throughput/progress record only and does not open any
downstream performance claim.

## 2026-10-21 — G2 durable checkpoint contract audit

The latest durable G2 recovery pair remains intact: `adapter-step012500.pt`
and `latest_training_state.pt` are present at 469,243,990 and 469,247,597
bytes.  The loaded state reports step 12,500, format
`groundanything-g2-projector-qwen3-lora-v2-strict`, the registered manifest
SHA, world size 2, optimizer and scheduler state, both rank RNG paths, sampler
state, and the full training contract.  This validates recoverability only;
the formal run is still not terminal and no downstream model claim is made.

## 2026-10-21 — historical G2 midterm scope audit

Indexed the Clean V2 TRAIN manifest (583,251 rows; SHA-256 `f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`) against all 24 historical step-3000 G2-PL input frames. All 4 MOT17-02-FRCNN frames are TRAIN-scope; the 8-frame and 12-frame Refer-KITTI-0007 RMOT clips have zero exact image-path overlap with the Clean fit index. Every frame path and content SHA-256 is recorded. The short clips remain diagnostics and do not permit an independent generalization claim. Evidence: `docs/GROUNDJEV/G2_MIDTERM_SCOPE_AUDIT_20261021.{json,md}`.

## 2026-10-21 — J1 terminal checkpoint correction

The historical `trackjev_3task_J1_v4_birth` course is terminal at 1,200/1,200 updates, not active: 19,143 rows were consumed, the final adapter and full FP32/Adam/scheduler/RNG state are present, and no J1 optimizer is live. The earlier budget note had only observed step 650 and is corrected without restarting the run. Evidence: `docs/GROUNDJEV/J1_TERMINAL_RECEIPT_20261021.json` and the updated `J1_J2_TRAINING_BUDGET_AUDIT.md`.

## 2026-10-21 — OVMOT evaluator repair reproduced on fresh copies

The official TETA evaluator was rerun on fresh copies of the existing video-63 control and video-4 native diagnostic. The current configuration keeps `OUTPUT_SUMMARY=True`, `OUTPUT_TEM_RAW_DATA=True`, and `OUTPUT_PER_SEQ_RES=True`, which fixes the two previously recorded evaluator failures without changing GT or predictions. Video 63 completed its independent 40-frame TAO validation scope with TETA 18.640, LocA 7.0114, AssocA 28.618, and 35,228 predictions; it is a simple-backend control, not G2/G1/OneJev adaptation evidence. Video 4 remains only a four-frame native diagnostic: all four frames were safely rejected for duplicate/unsorted canonical coordinate tuples and produced zero predictions. The full 40-frame video-4 native evaluation is not claimed until that parser/output blocker is repaired. Receipt: `docs/GROUNDJEV/OVMOT_EVALUATOR_REPAIR_20261021.{json,md}`.

## 2026-10-21 — conservative OVMOT native-output parser repair

The strict native parser rejects video-4 because otherwise complete GroundAnything responses contain duplicate/unsorted output plus malformed fragments (dangling box tokens and nested object references). Added `locatemot/groundjev/native_output_repair.py` and the `official_scoring_repaired` adapter policy. The repair keeps only complete object-ref/box pairs with valid four-coordinate geometry, drops and records malformed fragments, and never fills or reorders coordinates. A targeted self-test passed; offline preflight of the four preserved raw responses recovered 104/129/133/98 complete pairs. A fresh live retry timed out at the GroundAnything service before frame 751, so no new native metric is claimed and video-4 remains pending service recovery. Evidence is in `docs/GROUNDJEV/OVMOT_EVALUATOR_REPAIR_20261021.{json,md}`.

## 2026-10-21 — G2 Clean progress snapshot at step 12873

The formal G2 Clean event stream advanced to 12873/18,227 during the audit; the latest event at 2026-10-21 06:01:24 +0800 has finite loss 0.272433674, gradient norm 11.859652, and peak memory 12.828881 GiB. The last 100/500/1,000-update windows measure 310.88/312.23/311.53 steps/hour, giving a linear ETA near 2026-10-21T23:12:34.824000+08:00. The durable adapter/state pair remains at adapter-step012500.pt plus `latest_training_state.pt`; no terminal receipt exists and all downstream queues remain correctly gated. Evidence: `docs/GROUNDJEV/G2_PROGRESS_SNAPSHOT_20261021_STEP12873.{json,md}`.

## 2026-10-21 — native joint queue config contract repair

The post-G2 native three-task owner was found to inject the audit-only
`native_manifest_rows` field into its generated OneJev config. The pinned
`train.sft` parser rejects unknown keys, so this would have failed before the
first model load when all dependency gates opened. Removed that field from the
generated config; the native manifest row count remains in the owner state and
is checked against the terminal `done.examples` receipt. Validated the
normalized config against `train.sft.build_parser`, confirmed `max_rows=1` is
only a per-microbatch cap with `accum_rows=16`, and ran compile/import smokes.
No G2 state or process was changed. Receipt:
`docs/GROUNDJEV_COUPLED/REAL_NATIVE_CONFIG_AUDIT_20261021.{json,md}`.

## 2026-10-21 — corrected coupled architecture boundary

The architecture/source audit had stale wording that described the bridge as
visual-only. The current code includes `NativeG2CleanDLMProvider`, and the
Phase-C receipt records one real G2 clean visual-plus-language forward with
same-frame context consumed by the typed readout. Updated the audit to state
that native context is complete while SGLang integration and trained
pre-grounding memory feedback remain open gates. No metric or model claim was
changed.

## 2026-10-21 — opened C2 prefill training gradient path

The causal memory prefill adapter previously wrapped every call in
`torch.inference_mode`, which was correct for deployment but would silently
block gradients for a memory-conditioned training run. Added an explicit
`inference_mode` switch that preserves default inference behavior and allows
training callers to pass `False`. A new backward test confirms the inserted
memory token receives finite gradient through the prefill and readout. This
is an interface correction, not a tracking or grounding gain.

## 2026-10-21 — G2 Clean progress snapshot at step 12907

The protected G2 event stream is contiguous through 12,907/18,227. The latest
event at 06:07:55 +0800 has finite loss 0.1906903072, gradient norm 9.2506351,
and peak memory 12.8288 GiB. Last-100/500/1,000-step throughput is
312.084/312.309/311.484 steps per hour, leaving 5,320 updates and a linear
ETA around 23:10–23:13 Asia/Shanghai. The durable recovery pair remains
`checkpoints/adapter-step012500.pt` and `checkpoints/latest_training_state.pt`;
there is no final receipt. All downstream owners remain gated. Evidence:
`docs/GROUNDJEV/G2_PROGRESS_SNAPSHOT_20261021_STEP12907.{json,md}`.

## 2026-10-21 — repaired G1 GPU5 waiter runtime paths

The G1 waiter was correctly idle but its future command resolved the model,
GroundAnything source and ground-train Python environment under the code
checkout, where they do not exist. Updated the queue to run training, export
and controls from `/data2/user/LocateMOT_GroundJEV` while preserving the code
checkout for the registered plan and data/protocol files. Verified the runtime
paths, terminated only the old wait-only owner PID 579251, and started PID
631075 with status `WAITING_G2_CLEAN_SUCCESS`, zero optimizer updates and no
GPU launch. G2 and historical artifacts were untouched. Evidence:
`docs/GROUNDJEV/G1_GPU5_RUNTIME_PATH_REPAIR_20261021.{json,md}`.

## 2026-10-21 — repaired G2 bundle and TAO exporter runtime paths

The G2 deployment-bundle owner and the native TAO shard owner were waiting on
the formal G2 receipt, but both future export calls resolved the base model and
exporter under the source checkout. That checkout intentionally has no
`models/GroundAnything-DLM`; the dedicated GroundJEV runtime does. Both queues
now invoke the runtime GroundAnything Python and runtime exporter from
`/data2/user/LocateMOT_GroundJEV`, while the source checkout remains the code
and manifest authority. The two wait-only owners were replaced without
touching G2: bundle PID 634486 and TAO PID 634487 are both still gated at
`WAITING_G2_CLEAN_SUCCESS`, with no optimizer updates or GPU allocation.
`py_compile` and a runtime Torch import passed. Evidence:
`docs/GROUNDJEV/G2_BUNDLE_RUNTIME_PATH_REPAIR_20261021.{json,md}`.

## 2026-10-21 — G2 Clean progress snapshot at step 12938

The protected event stream advanced to 12938/18227. The latest finite event at
06:13:50 +0800 has loss 0.1918672598, gradient norm 10.0799599, and peak
allocated memory 12.8289 GiB. Recent 100/500/1000-step rates are
313.316/312.283/311.850 steps per hour, giving a linear ETA around
23:06–23:11 Asia/Shanghai. The final receipt is absent; adapter-step012500 and
latest_training_state.pt remain the durable recovery pair. This is a training
health snapshot only, not a tracking result. Evidence:
`docs/GROUNDJEV/G2_PROGRESS_SNAPSHOT_20261021_STEP12938.{json,md}`.

## 2026-10-21 — corrected G2 native evaluation identity and runtime paths

The short-control entrypoint still hard-coded its adapted arm as `G2-PL` and
resolved future inference/scoring scripts from the historical OneJev checkout.
It now derives the Clean arm as `G2-CLEAN-V2`, validates the terminal adapter
step 18227, Clean TRAIN manifest SHA, merged model SHA and export receipt before
launch, and executes from `/data2/user/LocateMOT_GroundJEV`. The historical
`GroundAnything-DLM-G2-PL` path deliberately keeps the `G2-PL` label and is not
reclassified. The code was compiled and the historical identity path was
smoke-tested; no service or GPU was started. Evidence:
`docs/GROUNDJEV/G2_NATIVE_EVAL_IDENTITY_REPAIR_20261021.{json,md}`.

## 2026-10-21 — relaunched exited G1 GPU5 wait owner

The G1 dependency owner had exited with only its initial
`WAITING_G2_CLEAN_SUCCESS` state; no training output, optimizer update or GPU5
launch existed. After verifying that the registered output directory still
contained only the owner state, relaunched it with the same G2 root, runtime
GroundAnything base, output, bundle and control paths. New PID 642505 is live
and waiting; protected G2 was untouched. Evidence:
`docs/GROUNDJEV/G1_GPU5_WAIT_OWNER_RELAUNCH_20261021.{json,md}`.

## 2026-10-21 — G2 Clean live health snapshot at step 12964

The protected two-rank G2 process remains running on physical GPUs 3/5; both
ranks are in the compute loop with about 19.3 GiB allocated each. The event
stream is contiguous through 12964/18227, with finite loss 0.3086177218,
gradient norm 13.7076988 and peak memory 12.8289 GiB. Recent throughput is
312.500/312.067/311.850 steps per hour over 100/500/1000 updates, giving a
linear ETA around 23:09–23:11. No terminal receipt exists and no metric claim
is made. Evidence:
`docs/GROUNDJEV/G2_PROGRESS_SNAPSHOT_20261021_STEP12964.{json,md}`.

## 2026-10-21 — downstream pre-launch contract preflight

While G2 remained non-terminal at step 12970, imported every registered G1,
G2 bundle, held-out, TAO, native join, three-task and YesTrack queue module;
all targeted Python compilation checks passed. The dedicated runtime Python
environments, GroundAnything base, service interpreter, exporter, video runner
and Clean TRAIN manifest were present. No GPU, optimizer or model service was
started. Evidence:
`docs/GROUNDJEV/DOWNSTREAM_PRELAUNCH_PREFLIGHT_20261021.{json,md}`.

## 2026-10-21 — G2 Clean live progress snapshot at step 12980

The protected two-rank G2 Clean process remains alive and its event stream is
finite and contiguous through 12980/18227. The latest event at
2026-10-21 06:21:56 +0800 has loss 0.2029841521, gradient norm
11.7594414, and peak event memory 12.8288527 GiB.
The measured 100/500/1000-step rates are
312.5/312.012/311.877
steps per hour, giving a linear estimate of about 16.817 hours from the
500-step window. The final receipt is still absent; adapter-step012500 and
latest_training_state.pt remain the durable recovery pair. Downstream owners are
still dependency-gated and have no optimizer updates. This is a training-health
snapshot only and makes no tracking or benchmark claim. Evidence:
`docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP12980.{json,md}`.

## 2026-10-21 — YesTrack GPU7 resource gate confirmed

The live YesTrack owner remains `WAITING_GPU7_EXCLUSIVE` because physical GPU7
has an unrelated process (PID 4008192, user `ljh`, PRISM/CUB training) using about
16.2 GiB at 100% utilization. The queue's rule is no compute PIDs and at least
32768 MiB free; the external process was left untouched. This is an observed
resource gate, not a training or metric result. Evidence:
`docs/GROUNDJEV/YESTRACK_GPU7_RESOURCE_GATE_20261021.{json,md}`.

## 2026-10-21 — MOT-only long-training safe-close audit

A process-name audit over all `lyq` Python/torch/train/eval/service processes
found no residual MOT-only or old OneJev long optimizer. The only project
optimizer is the protected G2 Clean two-rank run at step 12991/18227
on physical GPUs 3/5. Dependency queues and the GroundAnything service remain
registered; no legacy MOT-only process was restarted. Evidence:
`docs/GROUNDJEV/MOT_ONLY_SAFE_CLOSE_AUDIT_20261021.{json,md}`.

## 2026-10-21 — G0 MOT20 long export terminal receipt audited

The frozen G0 long export has an authoritative official-scope receipt for
MOT20-05: 1658 legal TRAIN frames, HOTA 0.433441,
DetA 0.638348, AssA 0.294655, IDF1
0.441085, IDSW 6239. The receipt is a
G0 TRAIN/validation-scope reference only; it is not MOT20 TEST and is not an
independent G2 held-out claim. The older 1,657-frame diagnostic artifact is
kept separate. Evidence:
`docs/GROUNDJEV/G0_MOT20_LONG_EXPORT_AUDIT_20261021.{json,md}`.

## 2026-10-21 — live completion matrix refreshed at G2 step 12997

The previous matrix was stale at step12002. A new execution-state matrix records
G0 MOT20-05 long export complete in its legal TRAIN scope, MOT-only safely closed,
G2 running at 12997/18227 on GPUs3/5, and all downstream G1/TAO/held-out/
three-task/YesTrack gates with their current statuses. Scientific gain remains
unproven and independent metrics remain pending. Evidence:
`docs/GROUNDJEV/GOAL_COMPLETION_MATRIX_LIVE_20261021_STEP12997.{json,md}`.

## 2026-10-21 — G2 recovery checkpoint validated on CPU

While the protected G2 run continued, the latest durable training checkpoint
was loaded with the CPU-only runtime. It is step 13000 at world size
2, and contains model trainable state, optimizer, scheduler,
sampler position, Python/NumPy/Torch/CUDA RNG state, the training contract and
trainable audit. The adapter-step012500 file is also present. No GPU was touched
and no metric claim is made. Evidence:
`docs/GROUNDJEV/G2_CHECKPOINT_RECOVERY_AUDIT_20261021.{json,md}`.

## 2026-10-21 — G2 Clean live progress at step 13065

The protected two-rank G2 Clean run advanced to step 13065/18227 while both ranks remained live on GPUs 3/5. The event stream remains finite with finite loss and gradient norm; no final receipt exists yet, so downstream G1, native TAO/OVMOT joining, three-task OneJev optimization and independent held-out evaluation remain gated. This receipt records training health only and makes no scientific metric claim. Evidence: `docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP13065.{json,md}`.

## 2026-10-21 — G2 throughput estimate at step 13072

Using actual event timestamps from the protected two-rank G2 Clean run, the recent 100/500/1000-event windows give approximately 5.20 steps/min and 16.52 hours to step 18227. This is an operational ETA only; it is not a metric or quality claim. Evidence: `docs/GROUNDJEV/G2_THROUGHPUT_ESTIMATE_20261021_STEP13072.{json,md}`.

## 2026-10-21 — G2 adapter checkpoint step 13000 audited at event step 13076

The protected G2 Clean run advanced to event step 13076/18227 and produced the durable `adapter-step013000.pt` checkpoint. Its SHA-256 is `08cf28930eced61b2086785494728221e113ac75cd7dc6aefe02ab1bc2ebfe59`; the latest full recovery state remains step 13000 with SHA-256 `4a79cbf3e64e69ffc56aaf964754d0209382b87c2124d46f13506325ca52fd22`. The final receipt is absent, so downstream gates remain closed. This is checkpoint and training-health evidence only. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP13076.{json,md}`.

## 2026-10-21 — live goal-completion matrix at G2 step 13116

The execution matrix was refreshed from the current G2 event stream and downstream owner states. G0 MOT20 long export remains complete in legal TRAIN scope and MOT-only training remains safely closed; G2 is still running at step 13116/18227, while G1, native TAO/OVMOT joining, three-task OneJev, YesTrack and held-out evaluation remain gated. This is an execution audit and makes no scientific metric claim. Evidence: `docs/GROUNDJEV/GOAL_COMPLETION_MATRIX_LIVE_20261021_STEP13116.{json,md}`.

## 2026-10-21 — G2 adapter checkpoint step 15500 audited at event step 15737

The protected two-rank G2 Clean run advanced to event step 15737/18227 on physical GPUs 3 and 5 and produced the durable adapter-step015500 checkpoint. The adapter and latest full recovery state both loaded on CPU with `weights_only=False`; optimizer, scheduler, sampler and RNG state are present. The terminal receipt is still absent, so G2-dependent bundle export, G1, native TAO/OVMOT joining, three-task OneJev optimization and held-out/YesTrack evaluation remain gated. This is checkpoint and training-health evidence only; it does not establish a coupled JEV metric improvement. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP15500.{json,md}`.

## 2026-10-21 — G2 adapter checkpoint step 16000 audited at event step 16014

The protected two-rank G2 Clean run advanced to event step 16014/18227 on physical GPUs 3 and 5 and produced the durable adapter-step016000 checkpoint. The adapter and latest full recovery state both loaded on CPU with `weights_only=False`; optimizer, scheduler, sampler and RNG state are present and the registered Clean manifest SHA matches. The terminal receipt is still absent, so G2-dependent bundle export, G1, native TAO/OVMOT joining, three-task OneJev optimization and held-out/YesTrack evaluation remain gated. This is checkpoint and training-health evidence only; it does not establish a coupled JEV metric improvement. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP16000.{json,md}`.

## 2026-10-21 — G2 adapter checkpoint step 16500 audited at event step 16501

The protected two-rank G2 Clean run advanced to event step 16501/18227 on physical GPUs 3 and 5 and produced the durable adapter-step016500 checkpoint. The adapter and latest full recovery state both loaded on CPU with `weights_only=False`; optimizer, scheduler, sampler and RNG state are present and the registered Clean manifest SHA matches. The terminal receipt is still absent, so G2-dependent bundle export, G1, native TAO/OVMOT joining, three-task OneJev optimization and held-out/YesTrack evaluation remain gated. This is checkpoint and training-health evidence only; it does not establish a coupled JEV metric improvement. Evidence: `docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP16500.{json,md}`.
## 2026-10-21 — 坐标加权耦合 step60 仍未恢复真实框

在冻结 G2 step17000 的前提下，从双序列耦合 step48 恢复到 step60，使用坐标 token 权重 5.0，MOT20-03/05 共 6 个 transition、10 次视觉编码，正式 G2 未改动。Grounding teacher-forcing loss 末步为 2.09848，JEV structured loss 为 0.000779；两个序列的 MATCH/ADMIT/Yes-No/ACTIVE 仍为 1.0，但 C2 greedy 框平均 IoU 仍为 0。对 step60 做无更新 token trace，C2 坐标均值损失降到 2.5554/2.5421，却仍无法贪心生成首个坐标 `<214>`/`<339>`（分别输出 `<1>`/`<|box_start|>`）。结论：耦合链路真实训练有效影响 token 概率，但输出坐标接口/课程仍未打通，停止盲目加步，下一步改坐标输出目标或显式候选框条件并重新做独立视频验证；不宣称 MOT/RMOT/OVMOT 提升。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_COUPLED_COORDINATE_WEIGHT_STEP17000_20261021.{json,md}`。
## 2026-10-21 — native visual-prompt memory also leaves G2 geometry unchanged

用 GroundAnything 官方 `build_visual_prompt` 把上一帧合法框作为当前帧 visual reference，MOT20-03/05 各 5 帧共 10 次视觉编码，G2 step17000 冻结、无参数更新、无第二个 VLM。所有 C2 输出都能被官方 parser 恢复为合法框，但两序列 C2 平均 IoU 均为 0；首个坐标仍接近冻结 G2 错误基线。结论：原生 reference-box 接口本身不能修复当前坐标头，不能把该诊断称为跟踪收益。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_VISUAL_PROMPT_MEMORY_STEP17000_20261021.{json,md}`。

## 2026-10-21 — identity + previous-box state projector 双序列耦合失败边界

新增小型 `IdentityBoxStateTokenProjector`，将上一帧共享视觉摘要和持久框坐标一起插入 G2 causal prompt；冻结 G2 step17000，MOT20-03/05 共 6 个 transition，真实 GPU 更新 12 步。Grounding loss 降至 1.70039，typed JEV MATCH/ADMIT/Yes-No/ACTIVE 在两序列均为 1.0，但 C2 greedy 框 IoU 仍为 0，首坐标仍输出 `<1>`/`<0>`。结论：只加入数值框状态仍不足以控制 G2 坐标头，下一步需要合法的 coordinate-output adaptation/候选条件训练，而不是盲目增加 JEV 步数。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_STATE_CONDITIONED_MEMORY_STEP17000_20261021.{json,md}`。
## 2026-10-21 — corrected dense G2 contract exposes one-box baseline

审计发现此前 target-only 耦合 pilot 将 `Locate all persons` 与单目标 answer 混用，不能据此解释 G2 坐标头。新增多框 causal decoder：每个框的四个坐标受官方 token 范围约束，只在 `<|im_end|>` 停止，并在 MOT20-03/05 各 3 个真实帧保留全部 GT person boxes。冻结 G2 step17000、6 次视觉编码、无参数更新时，模型仍在每帧只输出一个合法框；IoU0.5 召回均值 MOT20-03=`0.010345`、MOT20-05=`0.004038`。结论：这是 G2 dense 前端的真实基线；后续耦合训练必须使用完整多框合同或合法 referring/visual-prompt 合同，之前 target-only IoU 结果降级为协议不匹配的接口诊断。证据：`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_DENSE_MULTIBOX_STEP17000_20261021.{json,md}`。

## 2026-10-21 — G2 Clean live progress at step 17488

The protected two-rank G2 Clean run remains live at `17488/18227` on GPUs 3/5.
The latest event is finite (loss `0.1523213747`, gradient norm `8.3949747`, peak
`12.828824 GiB`); recent 500-step throughput is `307.919 steps/hour`, giving a
linear remainder estimate of about `2.40 h`. The durable recovery state and
`adapter-step017000.pt` are present, while the terminal receipt is absent.
G1, bundle export, native OVMOT joining, three-task optimization and held-out
evaluation remain gated; no independent OneJev/YesTrack optimizer was started.
This is a training-health/recoverability record only. Evidence:
`docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP17488.{json,md}`.

## 2026-10-21 — GroundJEV fixed-candidate cross-sequence diagnostic on MOT20-05

冻结在 MOT20-03 TRAIN repair split 上训练的 180-step readout，在不同的
MOT20-05 TRAIN 视频帧 766–893（128 帧）上零更新运行；仍使用同一注册的
native candidate/region cache 合同和单主干缓存回放。TrackEval 得到
HOTA=`0.115666`、DetA=`0.147839`、AssA=`0.090593`、IDF1=`0.071458`、
IDSW=`341`、FP=`19`、FN=`21407`，126 帧有输出，关联回放约 `9.06 FPS`。
相比 MOT20-03 同视频 temporal repair 的 HOTA=`0.302815`、IDF1=`0.257729`、
IDSW=`18`，跨视频明显退化，说明当前 fixed-candidate readout 尚未泛化。
两个范围均为 TRAIN，缓存早于 G2 Clean 终点，不能当作 G2 Clean、TEST 或
最终 GroundJEV 增益。证据：
`docs/GROUNDJEV_COUPLED/PHASE_B_FIXED_CANDIDATE_CROSS_SEQUENCE_MOT20_05_20261021.{json,md}`。

## 2026-10-21 — registered GroundJEV coupled video entrypoint

Added `tools/groundjev/coupled/run_coupled_video.py` as the clean-checkout
entrypoint for the already-tested fixed-candidate GroundJEV video evaluator.
It forwards the real clip/checkpoint/output/device arguments to the existing
single-backbone runtime, keeps the TrackEval scope explicit, and does not start
a second VLM or relabel cached features as terminal G2 Clean output. The new
entrypoint passes `py_compile` and `--help`; no GPU job was launched by this
change.

## 2026-10-21 — G2 Clean step 17500 checkpoint recovery audit

在受保护的两卡 G2 训练继续运行期间，用注册 CPU 环境只读加载
`adapter-step017500.pt` 和 `latest_training_state.pt`。两者均报告 step
17500、world size 2、严格 G2 格式和一致的 Clean manifest SHA；adapter 与
完整 state 各有 293 个 trainable tensors，optimizer/scheduler/sampler/RNG
状态均存在，adapter 浮点参数全部有限。终态 receipt 仍未生成，训练进程
未修改。证据：`docs/GROUNDJEV/G2_CHECKPOINT_PROGRESS_20261021_STEP17500.{json,md}`。

## 2026-10-21 — post-G2 downstream queue contract audit

在正式 G2 仍运行期间，对终态后自动执行的 G2 bundle、GPU5 G1、native
OVMOT join、native MOT/RMOT/OVMOT 三任务和独立 held-out 五个入口执行只读
合同审计。五个脚本均通过注册 Python 环境的 `py_compile`；终态 G2
status/step/manifest 校验、bundle receipt、native OVMOT manifest gate、
held-out scope gate 和资源等待条件均存在，未启动任何 GPU 或 optimizer。
这是终态前置审计，不代表下游已完成。 

## 2026-10-21 — step17500 coupled decoder regression and DanceTrack dimension repair

G2 step17500 保持冻结，在两个真实 DanceTrack 序列各训练一个 transition、共 2
个 optimizer steps；官方 `hierarchy_dynamic` 解码在 4 个 transition 上均返回
合法框。C0/C2 recall@IoU0.5 分别为 1/7→2/7（0001）和 1/6→2/6（0016），
只作为受限耦合开发证据。回归发现官方 evaluator 把 DanceTrack-0016 写成
1280×720，实际 `seqinfo.ini` 是 1920×1080；已修正并用同一 checkpoint 重跑，
没有触碰正式 G2。未报告 HOTA/IDF1/TETA 或独立泛化提升。证据：
`docs/GROUNDJEV_COUPLED/PHASE_C_REAL_G2_DENSE_COUPLING_STEP17500_20261021.{json,md}`。

## 2026-10-21 — formal G2 remains live; no downstream restart

单次权威状态检查确认正式 G2 launcher PID `1799359` 仍存在；终态 receipt 与
`adapter_projector_final.pt` 尚未生成，最新可用周期 adapter 仍为
`adapter-step017500.pt`，完整 recovery state 保留。G1 GPU5、G2 bundle、GPU7
独立 held-out、TAO native shards 和 native 三任务 owner 分别仍处于注册的等待
状态，三任务 owner optimizer updates 为 0。没有重启、覆盖或提前晋升非终态
checkpoint。证据：`docs/GROUNDJEV/G2_LIVE_WAIT_20261021.{json,md}`。

## 2026-10-21 — formal G2 event step 17593 live progress

重新读取同一个正式 event stream，确认 G2 已从 step17500 继续推进到
`17593/18227`；最后事件 loss=`0.171383`、gradient norm=`10.1324`、peak
memory=`12.829 GiB`，最近 500 步吞吐约 `307.1 steps/hour`，线性剩余约
`2.1 h`。launcher PID `1799359` 仍存活，终态 receipt/final adapter 仍缺失，
所以没有启动或改写任何下游 owner。证据：
`docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP17593.{json,md}`。

## 2026-10-21 — downstream coupling regression remains green during G2 wait

在正式 G2 继续占用 GPU3/5、终态 receipt 尚未生成期间，对
`locatemot/groundjev/coupled`、`tools/groundjev/coupled` 和
`scripts/groundjev` 执行 registered Python 环境的 `compileall`；随后运行
11 个核心 coupling 单元测试（official DLM memory、causal prefill、memory
conditioned input、Phase-A core），结果全部通过。没有启动 GPU、修改 G2 或
改变下游等待条件；该验证只证明代码入口可重建，不构成性能结果。

## 2026-10-21 — G2 latest recovery state CPU integrity recheck

在正式 G2 仍运行到 event step17615、终态 receipt 尚缺失期间，用 registered
ground-train Python 环境只读加载 `checkpoints/latest_training_state.pt`。
state 报告 durable step17500、world size2、293 个 trainable tensors，
optimizer/scheduler、sampler position、四类 RNG 和训练合同字段均存在；所有
浮点 trainable tensors 有限，文件 SHA=`9aad3f6bd6b4bfcab386d1ec1e39ea03420fc0a202df0d67a447e158f4e66139`。
未改写 checkpoint、未启动 GPU，确认正式 G2 仍可安全恢复。

## 2026-10-21 — independent held-out image-overlap gate cleared

在 G2 终态前用 CPU-only 直接调用注册的 held-out `overlap_audit()`，对
MOT20-05 validation-half 1,657 帧、RMOT single 800 帧、RMOT multi 800 帧和
TAO OVMOT 40 帧逐项比对 G2 Clean TRAIN rows；四个范围 exact image overlap
均为 0，TRAIN manifest SHA 为
`4a8e39d29a30fe1ef7df10484be57a10ab5f0baff0638130ff93fa9c1181ab6a`。没有模型
推理或 GPU 分配；该结果只清除独立性门槛，不代表任何性能指标。证据：
`docs/GROUNDJEV/HELDOUT_OVERLAP_AUDIT_20261021.{json,md}`。

## 2026-10-21 — independent held-out input preflight passed

对注册的四个独立 held-out manifest 做 CPU-only 文件检查：MOT20-05 validation
1,657 帧、RMOT single 800 帧、RMOT multi 800 帧和 TAO OVMOT 40 帧图片路径全部
存在（missing=0），且均声明 `training_on_this_scope=false`。没有 GPU 推理或
指标计算；这只保证终态 G2 bundle 生成后 held-out owner 不会因输入路径缺失
失败。证据：`docs/GROUNDJEV/HELDOUT_INPUT_PREFLIGHT_20261021.{json,md}`。

## 2026-10-21 — G0/G200/G1 comparison matrix aligned with the coupled mainline

按最新 GroundJEV 单主干耦合方向重新核对并登记三组对照：G0 的 MOT20-05
1,658 帧合法 TRAIN 长导出已完成；G200 是物理 GPU5 上完成的 200 步隔离定位
控制，保留 MOT17-02 四帧、RMOT single8/multi12 的短功能 receipt；新 G1
GPU5 owner 仍为 `WAITING_G2_CLEAN_SUCCESS`，optimizer updates 为 0，未启动
训练。耦合实现（官方 dense 多框解码、typed JEV、身份/框状态 memory、固定候选
视频回放）已有受限真实数据证据，但没有把短片或跨序列诊断升级为独立性能提升。
证据：`docs/GROUNDJEV/G0_G200_G1_COMPARISON_MATRIX_20261021.{json,md}`。

## 2026-10-21 — YesTrack route corrected to auxiliary offline reference

按新的 GroundJEV 单主干耦合任务重新校正执行路线：独立 YesTrack/OneJev
不再作为在线主模型，也不应阻塞 G2、JEV 结构化决策或持续身份记忆的耦合训练。
YesTrack 的 Yes/No 数据组织和概率读取只作为共享 GroundAnything 主干上的辅助监督
与离线参考对照；保留既有历史结果，不启动新的独立 YesTrack 长训练。正式主线仍是
GroundAnything hidden states + typed JEV choice/NEW/NO_MATCH + persistent identity
memory + one-to-one assignment。

## 2026-10-21 — independent native OneJev owner paused and coupled replacement registered

审计发现 `real_three_task_joint_native_v1_queue` 虽然尚未启动 GPU，但终态后会调用
独立 `run_source_manifest_sft.py`，与新的单主干 GroundJEV 路线冲突。该 owner 已安全
停在 `PAUSED_FOR_GROUNDJEV_SINGLE_BACKBONE`，原 manifest/config 和 owner 状态保留，
没有任何 optimizer update。新增 `queue_groundjev_native_three_task.py` 与
`train_native_three_task.py`：依赖精确 G2 Clean 终态和 native TAO 三任务 manifest，
在 GPU4 上只加载一个 G2 主干，按 MOT/RMOT/OVMOT 均衡样本提取一次视觉 hidden state，
训练共享 typed JEV、因果 memory projector 和小型 OVMOT category readout；不启动
独立 OneJev/YesTrack。CPU loader、loss/backward synthetic regression、compileall 和
11 个既有 coupling tests 均通过，尚未启动 GPU 训练，待 native manifest 与 G2 终态门
打开后自动恢复。

## 2026-10-21 — coupled three-task owner confirmed at dependency gate

新的 `queue_groundjev_native_three_task.py` 已在独立 owner 会话中运行，当前状态为
`WAITING_G2_NATIVE_MANIFEST`，`optimizer_updates=0`、`gpu_launched=false`。它没有占用
GPU，也没有重复启动旧 OneJev；G2 Clean 终态和 native TAO 三任务 manifest 仍是唯一
启动条件。该等待是可恢复的真实 owner 状态，不是训练结果或指标。

## 2026-10-21 — G2 Clean live progress at step 17728

同一个受保护 G2 Clean 训练句柄仍在推进，event stream 已到
`17728/18227`（`2026-10-21 21:33:28 +0800`），loss=`0.1114838467`、
gradient norm=`10.3513269`。最近 100/500/1000 事件吞吐约为
`286.73/303.04/307.59 steps/hour`，线性剩余约 `1.74/1.647/1.622 h`。
终态 receipt 和终态 adapter 尚未出现，持久恢复对仍为
`adapter-step017500.pt` 与 `checkpoints/latest_training_state.pt`；没有启动或
改写任何下游 owner。这是训练健康/ETA 证据，不是 MOT/RMOT/OVMOT 指标结果。
证据：`docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP17728.{json,md}`。

## 2026-10-21 — G2 Clean live progress at step 17750

event stream 已到 `17750/18227`（`2026-10-21 21:40:20 +0800`），最新
loss=`0.0942097213`、gradient norm=`7.7979364`，均有限。终态 receipt 和
终态 adapter 尚未生成，持久恢复对仍为 `adapter-step017500.pt` 与
`checkpoints/latest_training_state.pt`；G1、TAO native、held-out 和共享三任务
owner 继续受依赖门保护。这是训练健康证据，不是 MOT/RMOT/OVMOT 指标结果。
证据：`docs/GROUNDJEV/G2_PROGRESS_LIVE_20261021_STEP17750.{json,md}`。

## 2026-10-21 — G0 MOT20-05 长时导出终态复核

重新读取 G0 权威 owner 与完成 receipt：`mot20_05_full_validation_gpu5_v1` 为
`NATURAL_MOT20_SCOPE_AND_FROZEN_MEMORY_CONTROLS_COMPLETE`，fresh source 输出覆盖
1657 帧，官方指标 receipt 完整且归档校验通过（HOTA=0.07334696、DetA=0.09026983、
AssA=0.05967427、IDF1=0.05959845、IDSW=1933、FP=417、FN=288297）。该 G0 结果
保留作为对照，不被新的 GroundJEV 单主干耦合训练覆盖。

## 2026-10-21 — G2 Clean live progress at step 17854

同一受保护 G2 Clean 句柄推进至 `17854/18227`（`2026-10-21 22:04:25 +0800`），
近 500/1000 事件吞吐约 `0.0792/0.0825 steps/s`，线性剩余约 `1.31/1.26 h`。
`final_receipt.json` 与 `adapter_projector_final.pt` 尚未出现；下游 G1、TAO native、
held-out 和共享三任务 owner 仍处于依赖等待，未启动重复训练或独立 YesTrack。

## 2026-10-21 — G2 Clean exact-coverage training terminal

受保护的 G2 Clean exact-coverage 训练已到达终态 `18227/18227`，正式 receipt
为 `G2_EXACT_COVERAGE_TRAINING_COMPLETE`。训练合同为 world size 2、effective
global batch 32、exact-once coverage；manifest SHA256 为
`f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71`。终态
adapter SHA256 为
`cf1b5a6f626810ead94eb4e16adaf3feb6c6da3f99ca5023b6b5c234726f5742`，恢复
state SHA256 为
`cab791def46c264393636c5697f452ec81dd0a242cfd9408644490b92cf643dd`。最后
训练事件 loss=`0.6241224888`、gradient norm=`6.61599`、peak memory
`17.374252 GiB` 均有限；这只是训练终态与可恢复性证据，不宣称任何
MOT/RMOT/OVMOT 性能提升。G2 bundle 已完成，TAO native 导出和 GPU5 G1
启动由持久 owner 自动推进；共享三任务训练仍等待原生 manifest，未手动重复启动。

## 2026-10-21 — TAO native export resource-gate repair

TAO native OVMOT 的八个输入 shard manifest 和 G2 native merged bundle 已经
存在，但原注册 owner 固定等待 GPU4/6；这两张卡持续被外部 PRISM/ShiftRMOT
作业占用，因此没有产生 native candidate。未终止或修改这些外部作业。按当前
显存实测，将 owner 安全停止并从未完成的 shard 边界恢复到 GPU1/3；新的资源
合同要求每张选定卡至少保留 49152 MiB free，允许记录共存进程但不结束它们。
G2 checkpoint 未重训，三任务 owner 仍只等待合并后的 native manifest；这是
资源门修复，不是模型指标结果。

同一恢复过程中，第一次从交互 shell 后台启动的 owner 在 shell 结束时被
SIGHUP 收尾，留下了非终态的旧状态文件但没有改动任何 shard。已识别该首个
可操作原因，并用 `nohup setsid` 重新挂起同一个 owner；当前恢复仍从已有
manifest/shard 边界开始，不重训 G2、不覆盖部分输出。

恢复后的 TAO owner 已通过 G2 native bundle gate，进入 `STREAMING_TAO_NATIVE_TRAIN_SHARDS`；
GPU1/3 的资源探针分别记录 52804/74757 MiB free，并行处理 shard 0/1，未终止
共存的外部进程。G1 GPU5 owner 已进入 `RUNNING_G1_GPU5_ISOLATED_CONTROLS`。
三任务耦合 owner 仍保持 `optimizer_updates=0`，等待八个 shard 合并后的 native
manifest；当前没有提前宣称三任务指标。

随后验证到 GPU1/3 不属于 GroundJEV 的合法物理 GPU allowlist `[4,5,6,7]`；
两份服务在 readiness 前由 `serve_groundanything.py` 拒绝，首个可操作错误为
`DLM TP=1 launch requires exactly one registered physical GPU`。失败状态已保存为
`queue_owner_state_failed_gpu1_gpu3_allowlist.json`，没有产生候选或覆盖输出。已
恢复到合法 GPU4/7，并将资源门设为每卡至少 32768 MiB free；G1 继续保留 GPU5，
外部进程不被终止。

合法恢复后的 `services_gpu4` 与 `services_gpu7` 已通过启动检查，日志出现真实
`[PREFILL]` 请求，说明 TAO shard 0/1 已进入模型推理，而非仅创建等待进程。
两份 `native_stream.log` 尚未写入终态 receipt，故当前仍不合并 native manifest，
也不启动三任务 optimizer update。

## 2026-10-22 — Replace infeasible full TAO stream with registered native subset

权威 parent manifest 审计显示全量 TAO TRAIN 为 18274 帧、500 个视频、1203
类别；全量 stream 在产生终态候选前已安全中止，owned 服务和进程快照保留，
source manifest 未修改。为符合“不默认导出全部缓存”的执行约束，新增明确注册的
64 帧/64 视频 TRAIN 子集：每个均匀抽样视频选择一个有 eligible TRAIN annotation
的帧，仍对每帧请求全部 1203 类别，分为四个 16 帧 shard。join contract 允许且
明确标记 `registered_train_subset=true`、`full_train_image_scope=false`；这只能
作为三任务联合训练 source，不作全量 TAO 性能结论。新的 owner 使用合法 GPU4/5/6/7
并记录显存与共存进程，G2/G1/checkpoint 不变。

## 2026-10-22 — G1 GPU5 adaptation and controls terminal

G1 GPU5 owner 已完成 `NATIVE_GROUND_TRACKING200_TRAINING_COMPLETE`，共 200
steps，全部 registered training pairs visited，checkpoint reload 最大差异为
0.0。projector delta 改变 16211750 个 BF16 元素，导出 569 个 native projector
tensors 到 `GroundAnything-DLM-TrackProjG1-GPU5-V2`；source bundle 未覆盖。
随后完成 12 个 isolated control units、共 96 个 registered native frames，覆盖
MOT dense4、RMOT refer-single8 和 RMOT refer-multi12，三者重复运行均保持同一
raw wire candidates 和 outputs。该终态证明 G1 适配和控制一致性，不宣称全数据集
benchmark 提升；TAO native shard 推理和三任务耦合训练仍在继续。

注册子集 owner 已进入实际推理阶段：4 个 stream 进程分别使用合法 GPU4/5/6/7，
对应 GroundAnything 服务日志已出现数百至上千次 `[PREFILL]` 请求。四个 shard
仍未产生终态 receipt，故三任务 owner 继续等待 native manifest；没有把服务启动
或部分请求计作训练指标。
# 2026-10-26 — final G2 evaluation without MOT20

Hypothesis: frozen native final-G2 detections allow a fair Simple/J1/Coupled
association comparison. Work started from fetched `1cf48ce9` in an isolated
evaluation worktree. No new training; all existing checkpoints preserved.
Initial failure in prior scope assumptions: RMOT26 training includes all 25
local DanceTrack val videos; G2 also saw labeled MOT17 videos. Exclude these
from independent claims. SportsMOT val (45 clips) is source-video disjoint.
Implement detector-only persistence and complete-video cached replay before
measuring model gains. Results remain pending until genuine inference/scoring.

The first real four-worker gate FAILED: native GAMDecodeV2Block adds the
service-local request ordinal to sampling seed42. The same image therefore
changed candidates across workers/order. Preserve the original cache root as
`protocol_v1_failure.json` diagnostic evidence. The new `request_local_seed_v2`
root uses an explicit local decoder overlay that removes only the ordinal
term; seed42, weights, sampling temperature, native parser and block policy
remain unchanged. No shared GroundAnything source is modified.

The first AnimalTrack public-test video exposed a scorer adapter bug:
duck species class4 was discarded by the pedestrian-only MOT reader, yielding
zero GT. Add a documented evaluation-only species-to-class1 transport,
preserving original geometry, identity and mark; visibility=-1 is unknown.
Archive the invalid metrics and re-score unchanged predictions. This is an
evaluation fix, not a detector/threshold change or new training.

## 2026-10-26 — complete-video negative results and independent worker recovery

The formal request-local seed42 protocol passed the real four-GPU native-output
gate. Completed scopes now include SportsMOT, AnimalTrack, Refer-KITTI and the
first complete TAO video. The immutable snapshot records exact jobs, frame counts,
candidate/prediction digests, evaluator metrics, stage timing and checkpoint IDs.
The first two complete SportsMOT videos give official combined HOTA
Simple 0.619010, J1 0.170348 and Coupled 0.254628. This is a negative paired
association result; zero J1 IDSW accompanies FN 4647 versus Simple 143.
AnimalTrack and RMOT paired results are also negative in HOTA/IDF1 on the first
completed scopes. Keep these checkpoints and failures; do not tune on VAL.

TAO's first complete 15-frame video performed 195 full-vocabulary block requests
in 3917.508 detector seconds. All 15 frames were rejected under the registered
whole-frame policy, yielding zero Simple/J1 TETA. Its GT has Base classes only;
Novel is unavailable in this scope. Raw successful and failed chunks are retained.
CPU analysis of persisted outputs identifies malformed coordinate payloads,
wrappers, labels and geometry, rather than missing GT or a missing scorer.
Retokenized responses average roughly 1.3k tokens per request in the first clips;
this plus 13 serial requests explains much of the per-frame cost. Preserve the
native parser and weights while investigating execution overhead independently.

Operational failures were a concurrent initial WAL transition and the old live
dispatcher's port probe during short replay rotations. Serialize initialization
(four-writer/48-record regression passed); recover stopped A/C workers separately
with the fixed port probe and verify native gate output equality. Their completed
cache rows are preserved. Add an internal blocking recovery supervisor so worker
failures need not wait for agent-level process polling. No fifth GPU is allocated.
Official TETA is combined jointly across videos, never averaged per-video.
The full 45 SportsMOT, 26 AnimalTrack, 1694 RMOT queries and 993 TAO video scope
remains registered and incomplete. No training or checkpoint rewrite occurred.

## 2026-10-26 — actual image audit and J1 competition geometry root cause

A separate pass over all 583251 actual G2 TRAIN rows compares their image keys
with all 78084 distinct registered evaluation images, normalizing TAO disk
aliases. There is no exact-image overlap for any of the 26 AnimalTrack videos,
45 SportsMOT videos, 1694 RMOT queries or 993 TAO videos. This supplements the
source-video exclusion audit and does not assert pretraining disjointness.

Further J1 attribution found a real interface bug: OnlineTracker candidates are
xyxy, but the TrackJEV competition module consumes xywh. Full-frame and split-
subgroup paths both passed the wrong convention. Actual TRAIN row
MOT17-04-FRCNN_000001_c7_identity_birth has 44 candidates: legacy competition
differs in 520 saved IoU entries, whereas geometry-only conversion matches all
entries. The first SportsMOT frame also changes 18 pair entries and its groups.
Keep historical `j1` predictions untouched and evaluate the correction separately
as `j1_competition_xyxy_v2`, with the same checkpoint/candidates/options/policies.
The disjoint-object regression and two relevant existing tests pass (3 total).
Old negative results describe the legacy adapter, not an isolated measure of
J1 checkpoint quality. Improved tracking from the repair is still unproven.

Native execution profiling is registered separately: same 16 persisted native
inputs over all four task families, eager / diagnostic-sync suppression / eager.
No precision, parser, decoding, seed or weight change is allowed in this probe;
raw text, candidates and rejection status must agree before any adoption.

The geometry-only full-video control finished on the exact same SportsMOT
226-frame cache. HOTA Simple/legacy-J1/corrected-J1 is
0.647157/0.136083/0.135220; both J1 arms have exactly NEW1/ACTIVE225/IGNORE2434,
FN2230 and IDSW0. This is not a performance improvement. Keep the mathematically
correct transport and investigate the evidence/training contract; do not tune
birth thresholds or force NEW. The rerun shared existing physical GPU3 (57,235
MiB free at launch), adding zero physical GPUs; no speed-ratio claim.

The 16-input execution probe passed raw-text/candidate/rejection equivalence.
Native eager / diagnostic-sync suppression / eager repeat took
130.785/95.054/95.253 request seconds. The first-phase timing is confounded;
the warm-repeat difference is only 0.209%. Do not adopt or advertise an 18.9%
steady speedup. The separate native BF16 Triton B32 Graph probe is pending;
no graph or FP8 change has been applied to formal detection.

The native BF16 Triton B32 Graph probe completed on existing GPU6. All 16
ordinary eager baseline/repeat requests are exact, but zero of the 16 Graph
responses match: coordinates and candidate counts change, including one TAO
chunk switching from rejected/empty to 114 parsed candidates. Thus Graph's
103.420 request seconds versus warm eager 118.094 is not a matched-output speed
comparison. Reject Graph for formal inference and preserve all probe artifacts.
The formal workers continue the original native BF16 eager profile; no FP8,
decoder, parser, seed, precision or checkpoint change was adopted.


### 2026-10-26 — J1 current-crop and memory-input controls (registered before inference)

Canonical project remains GroundJEV; all work uses its authorized no-MOT20 evaluation worktree. Main native G2 final step18227 queue remains intact on physical GPUs 3/4/6/1. No new training or checkpoint writes. The previous geometry control completed 226 SportsMOT frames and did not fix the one-identity collapse (HOTA 13.5220% vs legacy 13.6083%).

Hypothesis: runtime misses TRAIN's current-candidate crop sheet. Separately, memory requests renumber candidates but retain original assignment keys, and still feed xyxy boxes to xywh competition. Add independently selectable current-crop evidence and local-memory transport; defaults reproduce historical inputs. Preserve all prior directories and compare new full-video arms against the already frozen v2 and Simple results on exactly the same native candidates.

Controls: `j1_current_crops_v3` adds only current crops to v2; `j1_current_crops_memory_v4` adds aligned memory indices/geometry to that crop arm. Register the complete first SportsMOT video (226 frames, candidate SHA 46fce75613893b8642a0b0be118b0462e2e89996e38fba2feec7b711159e34b6) before measuring either. No probability threshold, NEW/NO_MATCH policy, seed or GT-dependent change. Verify crop rendering against a real saved TRAIN crop sheet, and verify noncontiguous memory assignment remapping. Any GPU control must share an already registered physical GPU with adequate free memory, never a fifth GPU. Unit request capture is an engineering check, not research performance.

J1 frozen TRAIN input audit completed all 43,533 rows from the actual step1200 args.json recipe. Every row declares and carries current crop image2. All 3,471 birth questions omit a candidate number; two identical rendered input groups contain contradictory targets across 39 rows (MOT17-09-FRCNN frames15/16). Verified with upstream train.common.render_example and identical media, not row-ID heuristics. This confirms a training-data defect without proving its performance contribution; original rows and checkpoint were not changed. Crop rendering also exactly matches the 44-candidate saved TRAIN PNG. Targeted tests: 3 passed. The two full226-frame GPU controls are owned by session51676 and may continue while native queue detection runs on the same four physical GPUs.

Reporting repair: SportsMOT per-video evaluation.json carried OFFICIAL_FULL_VAL even though a job scores only its complete video. Main aggregate tables already gate the full45-video scope correctly. Future per-job scoring now labels REGISTERED_COMPLETE_VIDEO and preserves the original manifest label in a separate field. Published erratum lists the affected immutable historical receipts; prediction bytes and metric values were not altered, and active frozen manifests were not rewritten.

Both real226-frame controls completed successfully: current-crop v3 and current-crop+memory-transport v4 have byte-identical predictions, HOTA13.67049%, IDF17.81074%, IDSW0, FN2235, FP16,222 boxes and one distinct identity. V2 remains13.52196% HOTA/226 boxes; Simple64.71572%. No effective J1 benefit is established. The single-ID trace does not test multi-ID memory accuracy; the unit request check is only engineering evidence. Full TRAIN also has39,985 nonempty history rows but no media beyond the current frame/current crop sheet, so no trained past-visual-crop pathway is restored by this fix.

Main J1 is explicitly versioned as j1_current_crops_memory_v4 after real inference validation, to fix known input omissions independent of accuracy. Historical legacy/v2/v3 metrics remain separate and are not merged. A finite already-Simple-scored backlog is being replayed by backfill_j1_ready.py on the existing GPU3 (session55559); native G2 detection continues on3/4/6/1. New queue invocations use the same repaired protocol. No new training, checkpoint writes, metric/threshold changes or additional physical GPUs.

Published-source regression: clean Git archive248e78ac imports every TrackJEV Python module and the evaluation/audit/backfill entry modules, with compileall passing using the declared immutable local dependency/model links. This is an engineering check, not a performance result. Both real full-video controls already completed and preserved frozen receipts. Snapshot on that clean runtime code records22 complete G2 caches,21 frozen Simple arms,1 current-v4 J1 arm and10 Coupled arms. Older J1 variants remain separately reported. Backfill handle55559 and native owners61523/21877/1728 were confirmed still running through their live command handles. Full2,758-job goal remains incomplete; no new training or checkpoint mutation.


### 2026-10-26 — preregistered empty-history TRAIN-question control

The original add_birth_rows.py explicitly introduced empty-history birth supervision to repair missing births, but those rows use a dedicated question absent from the current runtime. Register `j1_train_birth_prompt_v5` against the complete226-frame v4 baseline, same J1 step1200/G2 candidates/parser/allocation thresholds. Only when the actual shortlist/history is empty, use the exact frozen TRAIN birth question: “No persistent identity exists yet in this sequence. Which option owns current candidate?” Keep all NEW/NO_MATCH/REJECT options, native probabilities and existing NO_MATCH feasibility rule. Once any live history exists, retain the normal candidate-index identity question. This is a question-contract ablation, not forced NEW and not retraining. The TRAIN question’s missing candidate index is an already audited defect; do not quietly repair it in this single-factor control. No result exists at registration time. Use existing physical GPU4 only if free space covers the J1 requirement and the detector’s restart margin; leave the main v4 protocol unchanged pending a full result.

Current-v4 backfill completed actual duck_2(196frames), Refer0016 query00cd(209frames), and firstTAO video(15 annotatedframes), identical candidate receipts across arms. V4 HOTA7.1423% vs Simple29.0059% on duck_2; V4 HOTA2.1092% vs Simple4.9299% on Refer; TAO TETA0 with all G2 candidates rejected. These do not establish a useful J1 identity advantage. Full birth-question v5 control remains a separate live experiment on existing GPU4 (handle20441); mainv4 and native4GPU queue continue. Source/test protocol for that control is published at767086ad; targeted4tests and compileall passed. No new weights or threshold changes.


### 2026-10-26 — birth-question result and causal-history input control

V5 completed all226 registered SportsMOT frames with the unchanged candidate SHA and J1 step1200. It creates7 identities/1356 output boxes (v4:1/222), HOTA18.8133%, DetA44.4246%, AssA7.9778%, IDF116.8554%, IDSW844, FN1177, FP92. The higher HOTA comes with much worse identity association; mainv4 is retained.

Post-hoc probability evidence on1237 adjacent GT-matched candidate/previous-output pairs finds all previous IDs in the options, but only344 raw best-existing choices preserve that ID; actual preserved219, switched748, omitted270. Mean native probability of the previous ID is.2313 versus.5406 for the best other ID. Every one of226 identity requests contains only the current image and current crops. These are offline diagnostics, not official IDSW replacements or runtime GT use.

Next single-factor experiment `j1_causal_history_v6`: retain v5 birth question, weights, boxes, parser, options, memory thresholds and assignment, and add only a visual sheet of the actual shortlisted tracks’ causal anchor/long/recent images. No motion-score prior, probability reweighting or additional training. TRAIN did not contain past visual crops, so this is an explicitly untrained inference-evidence intervention, not restoration of a validated training recipe. Freeze the same complete226-frame development video before inference. Keep mainv4 and all earlier results. Use one already registered physical GPU with restart memory margin, no additional SGLang service or fifth GPU.

An actual missing-output root cause was found: TAO AVA_BCiuXAuCKAU_scene_16_109527-110290 has exactly32 registered annotated frames, so D’s32-frame preflight completed its entire cache and marked the queue unit COMPLETE, bypassing the Simple/JEV path called only inside the ordinary-unit loop. The exact cache was scored on CPU without new G2 calls (TETA0); its frozen receipt remains byte-equivalent on repeated CPU pipeline entry. Source now routes preflight and ordinary-unit completion through the same slot handler. Busy services were not restarted. The already-scored video is eligible for normal J1 backlog replay.

### 2026-10-26: causal-history v6 complete-video control

Hypothesis: past-track visual evidence may improve J1 identity continuity. Prespecified v6 adds causal anchor/recent crop media to the unchanged v5 empty-history birth prompt; fixed J1 step1200 and identical G2 step18227 candidates, no training or threshold changes. Full SportsMOT v_9MHDmAMxO5I_c003: 226/226 frames; HOTA 17.5892%, DetA 38.6148%, AssA 8.0255%, IDF1 14.2299%, IDSW 729, FN 1374, FP 76. Association replay 0.3843 FPS; control wall 600.95 s. Compared with v5 HOTA 18.8133%, AssA 7.9778%, IDF1 16.8554%, IDSW 844, the intervention does not establish an identity benefit; both remain far below Simple HOTA 64.7157%. TRAIN had no historical-image sheet, so v6 is explicitly an untrained inference intervention. Keep as a diagnostic; main v4 unchanged. Candidate SHA 46fce75613893b8642a0b0be118b0462e2e89996e38fba2feec7b711159e34b6; prediction SHA fae8cef521552ead5d8a7ff32325b8053cc022fefdeb2e88ca795c3da5f1f6d6.

### 2026-10-26: final-G2 detector-only report and current cache equality

The frozen 29-job snapshot uses final step18227 and retains full registered scopes. {"sportsmot": {"complete_jobs": 7, "frames": 1779, "Recall50_percent": 98.64927806241266, "Recall75_percent": 87.988407597164, "rejected_frames": 0, "detector_only_fps": 0.285840442990364}, "animaltrack": {"complete_jobs": 8, "frames": 2308, "Recall50_percent": 51.687382077765456, "Recall75_percent": 31.434906926533234, "rejected_frames": 229, "detector_only_fps": 0.22178898579390788}, "refer_kitti_v2": {"complete_jobs": 7, "frames": 1463, "Recall50_percent": 35.273972602739725, "Recall75_percent": 26.902587519025875, "rejected_frames": 293, "detector_only_fps": 0.519908979327768}, "tao_amodal": {"complete_jobs": 7, "frames": 124, "Recall50_percent": 0.0, "Recall75_percent": 0.0, "rejected_frames": 124, "detector_only_fps": 0.003245858671354856}}. Recall is unranked GT coverage; AP unavailable because native per-box confidence is absent. No association model participates in these detection numbers. All completed snapshot JSONL checksums and actual frozen Simple/current-v4 J1/Coupled candidate inputs match immutable cache receipts, without rewriting historical receipts or regenerating boxes. TAO zero is the fixed whole-frame rejection policy when any of13 vocabulary chunks is invalid, not proof that all individual chunks have no detections. No G0 matched comparison is complete, so no fine-tuning gain claim is made. Full scope remains incomplete; retained native queue continues on physical3/4/6/1.

### 2026-10-26: native TAO chunk failure attribution and production resource measurement

The finite v4 backfill owner exited; all19 registered jobs (3810 frames, including jobs completed by their main owners) now have frozen same-input receipts. The new horse_2 Coupled full310-frame result is HOTA10.2628%, DetA16.9013%, AssA6.2905%, IDF111.2429%, IDSW249; candidates match Simple SHA3fa724245d400dabc5623d275898e01db1ac67e8ec7c638ebae887ecec6eaece. Post-hoc TAO analysis finds375/1612 chunks accepted on7 full videos/124 annotated frames;1237 fail and all124 frames trigger the frozen whole-frame rejection rule. The33911 accepted-chunk native proposal records are unverified and do not constitute rescued predictions or a valid score. Root causes are recorded by first structural defect; no parser/weight/protocol change. Initial offline use of the OneJev environment failed due missing grounding_anything; corrected to the separate GroundAnything environment and verified all completed cache counts. Added an observer for existing production frame/request counter increments and sampled owned residency GPU-hours; it neither launches nor stops an inference process and makes no extra G2 calls. Compile and real60-second telemetry/counter cross-checks passed; source published419a2ce3. A registered64-new-frame production observation is running under a single blocking command. Full goal remains active.

Production observer completed:64 new persisted query-frames and83 native requests in197.7563 s (0.323631 query-frames/s,0.419708 requests/s), including62 AnimalTrack frames (1 rejected) and2 TAO frames (2 rejected). Registered physical-GPU peak concurrency4; sampled task residency0.213189 GPU-hours for this window, maximum sample gap7.7007 s. This is window-only cost under shared load, not full-run GPU-hours or a controlled fourfold speedup. Existing native workers continue. Latest snapshot contains31 complete detector caches with matched main arms; paired official scope metrics are archived separately.
