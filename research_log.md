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
but copying a native query label leaves ClsA0. A real typed OneJev category
head using every supplied vocabulary class, separate from identity and absent
detection confidence, is required. It must preserve all candidates and GT.


### Native shared JEV category hypothesis (registered before execution)

Exact-geometry Simple/Full ClsA0 exposes first-label copying. Add an opt-in
native typed category head on the same OneJev parameters, with all1203input
labels represented in deterministic128class root groups, selected conditional
leaf and explicitUNKNOWN. Hierarchical routing is approximate, not a flat
1203class probability distribution. Preserve all306geometries and unknown
predictions; no GT/class/box filtering or metric changes. First run fixes
Simple identity to isolate semantics, then same-policy Full uses the same
engine for identity/category. Keep class confidence separate from identity
and missing detector confidence. Plan:configs/groundjev/native_category_control_20261014.json.


### Actual GroundAnything proposal own-history rollout

The current500step training corpus uses GT proposals and declared synthetic
fragments. To close that gap, register48legal training frames from the same
three MOT sources, original DLM candidates with no topK/NMS/GT substitution,
and the valid100step OneJev actor. Ground failures remain empty frames.
Record only successful context groups, retain every native candidate, keep
GT matching/aliases outside uncorrected actor state, and supervise lifecycle
from targets matched to current actual proposals. This creates training
supervision, not validation predictions or a new500step result.
Plan:configs/groundjev/native_candidate_rollout_20261014.json.


### Real-video memory export performance repair

Native Full1203predictions completed, but Simple control spends seconds per
frame re-encoding unchanged historical images. Actual155track profile:3.283s
state serialization,3.256s inside465PNG saves. Cache at most one exact PNG
encoding per live track/image field, keyed by image object identity. Native
tracking replaces images rather than mutating saved crops. Encodings remain
byte-equivalent and are released with each track; no persistent feature cache.
Verify actual32frame decision replay, cold/warm/full-state reload equality,
then preserve complete Full and the interrupted valid Simple prefix in a
fresh owner. No detector rerun, new seed, metric/filter/state change or E2E
speed claim. Source failure/controlled interruption remains recorded.


### Native class decisions, actual detector rollout, and serialization repair

Opt-in native_hierarchical category inference now shares the actual OneJev
engine/parameters with Full identity decisions. All1203input categories are
represented in128class root groups; selected-group leaf and explicit UNKNOWN.
This is approximate hierarchical routing, not a flat global MAP classifier.
Native path probability remains separate from identity probability and absent
native detector confidence. Every306exact native geometry is retained.

Completed Simple/Full TAO controls:33.042/23.320TETA, bothClsA0, unchanged
localization/association compared to native-label controls.299/306objects
choose rootUNKNOWN; seven receive vehicle labels. Full and Simple category
decisions are exactly equal; all identity probability differences are0.
Actual four-frame videos, six-decimal probability rounding, all candidates,
GT/empty/rejected frames and unmodified TETA verified. Native category
inference is implemented but does not solve classification; it remains
opt-in. No OVMOT gain or Novel result claim. Next step is legal category
supervision and targeted visual/group-routing diagnosis, not GT relabeling.
Receipt:outputs/groundjev/task_controls/native_category_v1/aggregate_results.json.

Actual GroundAnything proposals now feed a finite48frame legal own-history
training exporter on physical7, original DLM service6. Actor is the valid
OneJev100step checkpoint, V3 crops/motion. No GT proposal substitution,
synthetic fragments, topK/NMS or corrected actor state. Independent GT
matching supplies training targets only; lifecycle labels use targets matched
to actual current proposals. Record only accepted context groups and retain
all candidates including empty/rejected native frames. This closes a genuine
training-data gap; corpus owner remains active, not completed training.
Plan:configs/groundjev/native_candidate_rollout_20261014.json.

Full native DanceTrack0004 predictions completed1203frames,27rejected frames
(2.244%) retained. Original20frame resume means end_to_end_fps remains null.
Official full-video metrics remain pending the exact-candidate Simple control.
Simple recompressed all historical PNG crops every frame:actual155track
profile3.283s,3.256s in465PNG saves. Exact per-field cached encoding retains
at most3encodings per live track and releases them with the track. Actual
223track snapshot:4.487s cold/.003874s unchanged warm call;20,884,556cached
encoded bytes.32frame recorded causal replay, cold/warm payload and fresh
reload are exact. These are serializer timings, not whole-video speed gains.

Only verified owned Simple child4127910 was interrupted after its valid
200frame prefix; original failed owner and interruption receipt preserved.
Fresh owner reuses all complete native Full outputs and restores the entire
unchanged Simple prefix. First reuse preflight wrongly expected the newer
multitask specification field in the older ordinary-MOT receipt; retained
failure and fixed using actual registered paths/frame numbers/native prompts,
not synthesized metadata. v5 owner is active with cached PNG export, no new
native detector calls or changed experimental protocol.
Full/Compact/matched trained controls, mixed task supervision, full original
GroundAnything retention and formal500step checkpoint remain required.


### Fresh full-video V3 evidence measurement

Full V1 actual1203frames:HOTA.0214546,IDF1.00482694,IDSW3792; identical
Simple:.0852976,.0395616,746. Native candidate recall4341/4586=.946577,
Full existing-identity accuracy234/4330=.054042 and option coverage.847113.
Retain this negative baseline. Register a fresh complete1203frame V3 run:
real current/history crops and geometry, native NO_MATCH, all proposals,
unchanged native decoder/official metrics, adaptive context groups without
option/candidate removal. Use verified cached PNG persistence; no resumption
or candidate replay in Full, so a complete E2E record can finally be measured.
Still pretraining baseline, not trained500checkpoint or a completed Pilot.
Plan:configs/groundjev/dancetrack_fresh_v3_20261014.json.


### Full DanceTrack0004 baseline is now official and validated

Both1203frame video artifacts and exact candidate equality passed. Native
Full V1 HOTA.0214546/IDF1.00482694/IDSW3792/FP9300/FN241; same-candidate
Simple HOTA.0852976/IDF1.0395616/IDSW746/FP9334/FN256.27native rejected
frames (2.244%) stay in every official denominator. GT4586observations,
native13664candidates, one-to-one IoU.5 coverage4341/4586=.946577.
Full correct surviving identity associations234/4330=.054042; correct
option coverage3668/4330=.847113.9323unmatched candidate diagnostics
include official ignore regions and are not themselves official FP labels.
These negative results show both excessive candidate output and weak
pretrained identity decisions; detection recall alone is insufficient.

The complete200frame Simple prefix matches its preserved source row for row.
Original Full20frame resumption and Simple200frame resumption preclude a
continuous E2E claim. Simple32.62198replayFPS excludes native detection and
full persistence/rendering costs. Scalar diagnostics never modify output.
Receipt:outputs/groundjev/pilot_runs/dancetrack0004_pretrained_dense_v5_cached_memory_control/aggregate_results.json.

A fresh complete1203frame V3 native run is now active on physical6. Same
pretrained OneJev/original DLM, decoder and official evaluator, real current
candidate/history crops and motion evidence, explicitNO_MATCH, no resumed
prefix or Full candidate replay. Native context splitting retains all boxes
and up-to32identity options. Exact cached PNG export preserves checkpoint
payloads. Completion will measure actual complete pipeline E2E speed; no
performance or full-goal completion claim while active. Formal500step
training remains the original live owner; no restart or shortened protocol.


### Actual native detector own-history corpus completed and validated

48declared legal training frames,1184actual native proposals,5661supervised
rows:1184identity,1184update,1184reliability,69withheldNO_MATCH,2040lifecycle.
All typed targets/options/media and training split membership validated,
no GT IDs in model state, no GT state correction or synthetic fragments.
Native GT matches:474/526MOT17,110/110DanceTrack,496/806MOT20. Uncovered
GT remains explicitly counted; never invented as native proposals. This is
decision supervision, not localization supervision for missing boxes.
Actor outputs only74across48frames (31/12/31first-frame births, subsequent
collapse), exposing the valid100step actor's negative transfer rather than
hiding it. All raw own histories are retained.31,045,329bytes of crop evidence,
no full source-dataset feature cache. Corpus completion is not weight update
or formal training completion. Receipt:
outputs/groundjev/training_data/native_ground_own_history_v1/corpus_validation.json.


### Compact native evidence/distillation study (registered)

Full baseline is weak: use actual recorded OneJev100 distributions as a
.2auxiliary term with GT supervision dominant. Pack all48legal train frames
using native frozen DLM ROI features and the actual actor anchor/recent/long
events, without correcting its state. Compare structured set Compact, the
identical architecture with GT-only supervision, and a parameter-matched
feedforward control with the same heads/evidence/distillation. Keep dynamic
IDs, NEW/IGNORE/NO_MATCH, memory update, reliability and lifecycle. Declared
option-withholding moves removed native teacher mass toNO_MATCH; it is an
augmentation, not a new native teacher request. InitialMOT-only stage, not
full unified-task or long-memory proof. Plan:configs/groundjev/compact_native_20261014.json.


### Actual trained Compact and matched baselines (2026-10-14)

All48legal native detector/OneJev100own-history train frames are packed as
80base groups plus69declared option-withholding augmentations,149examples.
Native DLM ROI2560and uncorrected anchor/recent/long feature history; noGT
state correction. The removed teacher identity mass is projected toNO_MATCH
only in the declared training augmentation. Native pack131,123,485bytes,
not a whole-source-dataset feature cache.

Three actual300step/seed0 models completed with finite nonzero gradients,
changed weights, exact checkpoint reloads. Compact distilled and GT-only
set control each930,829parameters; feedforward distilled control929,805,
ratio1.001101. Two set blocks provide interactions; feedforward control has
no set attention but shares mean context, evidence and structured heads.
All support dynamic identities/NEW/IGNORE/NO_MATCH, update, reliability and
lifecycle. They are trained structured decision heads, not direct pretrained
OneJev decoding. GT supervision plus.2native recorded Full teacher loss;
identical set GT-only control omits only that loss. Auxiliary visual backbone
remains original frozen DLM. Parameters for all models changed, reload0,
candidate/history permutation error<=7.63e-6. Allocation peaks about40.5MB
are core-only, excluding CUDA runtime and shared native visual/DLM services.
Initial Compact training is MOT-only: task-index conditioning exists, but
joint expression/class training and full unified-task/long-memory proof remain
required. Never present natural set capabilities as JEV-exclusive novelty.

Same real128native proposals/ROIs,4MOT17development frames and unchanged
TrackEval, actual4frame videos validated:

| Core | HOTA fraction | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|
| Compact distilled set | .669111 | .738197 | 1 | 9 | 46 |
| Same set, GT-only | .576415 | .638889 | 8 | 3 | 57 |
| Matched distilled feedforward | .634360 | .703196 | 3 | 3 | 54 |
| Existing plain Simple reference | .748813 | .836502 | 1 | 17 | 24 |

Distillation helps the same architecture in this tiny single-seed control;
Compact still trails Simple and has more missed targets. ReplayFPS4.358/
3.959/4.512exclude detector/native feature extraction and full export costs;
no E2E or full-video accuracy claim. Receipts in
outputs/groundjev/compact/native300_controls_v1/aggregate_results.json;
trained checkpoints and SHA receipts in outputs/groundjev/compact/native300_v1.

Registered full1203frame DanceTrack0004 Compact and two controls reuse every
original native detector candidate, unchanged official GT and scorer. Stream
ROI once through physical5native feature service; controls reuse identical
ROI values on physical7. Source frames are read progressively rather than
loading the complete replay into memory; actual4/1203row equivalence passed.
These are candidate/feature replay controls, not detector-inclusive E2E.
First run failed at actual empty-proposal frame21: empty ROI pool returned
BF16 whereas nonempty interpolation and optional temporal layer useFP32.
Preserved20complete frames and failure; fixed only empty output dtype.
Only verified owned native feature service4043767 was restarted on5. A
real frame21 empty RPC/lifecycle probe passes and real frame20nonempty ROI
max difference after restart is0. A probe initially used the wrong timestamp
constant; it was corrected to the registered20Hz clock, with no data/clock
protocol changes. Fresh v2 owner restores all20native prediction/state rows
and resumes frame21, without dropping it. Original OneJev500/Ground/freshV3
and foreign UDMT remain untouched. Full metrics are pending its terminal
owner. First model frame latency includes ROI RPC; later controls read fixed
ROIs, so do not claim a direct comparable speed ranking across those latencies.


### Shared semantic Compact adaptation (registered)

MOT-only Compact cannot establish task transfer or category accuracy. Keep
all native MOT149groups, add48frozen-fitKITTI teacher-GT-proposal frames
with expression membership targets and64TAOtrain Base-only GT-region frames
with category targets. Clearly separate teacher proposals/history from native
MOT rollout; never present GT boxes as evaluation predictions. No NovelGT
supervision and no non-exhaustive TAO unmatched-as-negative labeling.
Use original real OneJev frozen token embedding means for text; add shared
learned expression/category heads to the same visual/identity/set core.
All1203classes remain available at inference. Balanced1200step seeded joint
stage; unchanged evaluation scopes/metrics and before-after native proposals.
Plan:configs/groundjev/compact_joint_semantics_20261014.json.


### Lost original trainer and exact-protocol recovery

Authoritative/proc audit finds no original trainer or model services. FullV3
and Compact full-video owners have terminal success receipts. Original
OneJev500has only275last log,250BF16weight checkpoint, no final/done or
optimizer/FP32master/RNG state. External termination cause is not established.
Do not call this complete or silently reset Adam from250. Preserve that
intermediate; replay the unfinished original500step unit from its original
100step initialization with unchanged seed/corpus/lr/schedule/steps. A small
wrapper extends official checkpoint handling only: atomic FP32masters,
Adam moments, scheduler, RNG and exact planned-step cursor. Single latest
training state is a checkpoint, not a feature cache. Original loss/rendering/
row plans/native fork checks remain unchanged. New config:
configs/groundjev/onejev_own_history_v3_exact500_replay_stateful.json.

### Completed joint semantic controls and full-video evidence

Joint1200 truly trained1,146,511 shared parameters with exact3task reload.
Pack149nativeMOT+45RMOT+62TAO teacher groups,112declaredteacherframes,
5emptyframeswithoutregionexamples. TeacherGT/idealupdate proxies remain
explicit. Same candidates/nativeROI and decodedframes validated for all four
short scopes. MOT HOTA66.911->71.556%, butIDSW1->3. SingleRMOT8.932->0%,
multi0->0%; TAOBaseTETA14.013->39.272%,ClsA0->58.286%,AssocA21.350->17.971%.
Reject a unified-success claim. Posthoc evaluation finds all6single/all8multi
GT-covered candidate observations have membership below.008 and no emitted
coveredboxes. Keep threshold.5, allGT/frames/negative results unchanged.
Hypothesis: limited-fit teacher region supervision and frozen token means do
not transfer expression membership to native validation proposals; covered
regions show semantic suppression, not an absence of proposals. Preserve
stage as an ablation, not production promotion. Next training must use broader
legal fit/native proposal supervision and verified query sensitivity.

Full1203DanceTrack Compact300/setGT/MLP and freshFullV3 controls are terminal
and validated. Original same-candidate HOTA20.157/5.977/11.197% vsSimple8.530%
andFullV1 2.145%; single-video/seed only. FreshV3 HOTA2.389% vs its exactSimple
8.257%, detector-inclusive uninterrupted E2EFPS.118560 excluding model/server
initialization; do not compare fresh responses toV1 as identical candidates.
Register/execute joint1200 retention on all1203sameCompact300candidate+ROI
rows, unchangedGT/TrackEval and rejectiondenominator, onefiniteowner GPU7.

Stateful original500protocol replay latest returned50/500 onGPU4, not old250
exactAdam resumption. Resume-state functional probe nextloss/params difference0.
Old250/last275/interruption retained; completion and modelgain remain unproved.

Training-fit membership96/96correct(45positive/51negative). Swapping other
fitexpression vectors changes probability mean.10595/max.99159; this is
input sensitivity only, not a semantic correctness score. Native validation
covered positives remain suppressed. Hypothesis strengthened: limited-fit
teacher/domain/query generalization rather than failure to update head weights.

Joint1200full1203retention terminal and validated: HOTA.22103185,IDF1.21413276,
IDSW346,FP1653,FN1485 vsCompact300.20156623/.17140097/588/2129/951.
All1203candidate+ROIrows match, video1203decoded, all27rejectedframes kept.
No GT/parser/metric/candidate changes. Keep as positive complete-video MOT
retention plus negative RMOT/OVassociation ablation; not unified-success.
GPU7owner exited. ReplayFPS22.747excludes detector/ROI costs.

### RMOT fit breadth and native own-history intervention (registered)

Current membership fits96teacher regions but suppresses every covered native
validation target. Register72frames from all9fit-v2 videos, exact18primary/
counter expressions. Compare broader teacher curriculum with same broadened
teacher+actual nativeGroundproposals/uncorrectedCompact ownhistory; original
MOT/OVgroups, initialization,seed,optimizer,1200steps and allvalidation protocols
remain fixed. Alternate-expression supervised snapshots add query contrast;
no alternate detector request or actor update is claimed. AllGT mappings and
quality proxies remain outside model inputs. Never replace a rejected native
proposal frame byGT. Plan:configs/groundjev/rmot_native72_20261014.json.

Native72owner has confirmed owned original Ground API readiness onGPU6
and is building72fit frames through actualCompact own histories onGPU7.
Existing conservative feature/crop/readable-result bytes6,288,115,739(5.856GiB),
models/optimizer/videos excluded; newtensorserialization checkedbeforewrite.
Empty-candidate lifecycle check fromactualnative train history31slots passes
finite loss3.8286855,lifegradient12.4010; no droppedemptyframe rule needed.

### Original Ground dense/referring retention data preparation

Original nine dense/referring benchmark inputs are absent locally. Official
GroundAnything README links Skywalker0410/Grounding-EvalData. Direct no-proxy
HF API succeeds; pin772e34bb1505053ed4857bba4405ee7b308bbfb6. Fetch only
complete archives/annotations needed by4dense+5referring task entries, not
GUI/OCR/etc. Rawdata/model checkpoints are outside25GiBfeaturebudget.
Parallel3downloads, noGPU. Original full30helper requires allotherdata, so
reuse its verified unmodified archive extraction/path functions and explicitly
label this a nine-task data subset. No evaluation/retention score is claimed
until original unmodified scorer runs on all declared examples.

### Broad fit/native own-history outcomes and CUDA recovery

Both1200additional-step models genuinely trained fromjoint1200, all3task
reload differences0. Same params/initialization/seed/lr/taskcycle; not equal
training-budget/parameter comparison againstoldCompact300. Broadteacher:
MOT4HOTA71.359%,singleRMO20.921%,multi17.740%,TAOBaseTETA46.279/ClsA67.0%.
Matched teacher+native:63.077/0/14.846/56.733/100.0%. Allpairedcandidates/ROIs
and videos validated. Native data165candidates/49GTaligned,8rejectedframes,
all72retained;28contaminatedactorhistoryobservations; noGTstate repair.
Native mixture restores6/6covered singlequeryprobabilities below.5(.362-.498),
versusteacher2/6. Keep threshold.5. Hypothesis: weak native proposal support
and negative-heavy own-history/counter samples shift query decisions; mixing
not universally beneficial. Preserve bothmethods/negative transfer, no promotion.

Full1203GT/candidate/ROI/video controls: teacherHOTA.2040093/IDF.2383171,
IDSW151/FP615/FN2276;nativeHOTA.2631573/IDF.3063294,165/579/1962.
Native run failedCUDAunspecifiedlaunch at799, preserved798rows/memory39.9.
NVMLECC0/recoveryNone; privilegedkernel unavailable, externalcauseunknown.
FreshsameGPU7 actual799probe finite; CPU/GPUprob delta.000115(referenceonly).
No reset ormodelmath/data change. Strictsource798prefixreplayexact, resume799,
all1203candidate/ROIrows anddecodedvideoframes validated,all27rejections kept.
Originalfailureowner retained; separateterminalrecoveryreceipt. E2Enull.

Native Ground original retentioninputs direct3parallel downloads complete:
6.929GB archives, all4dense+5ref canonicaltasks, 91,124rows. Officialhelper
hash/extraction unchanged. Originalevaluator full/limitnull dry-run reports
no missinginputs. Initialrecipe-path resolution and Pythonvenv-symlink mistakes
fixed locally; no metric/source modifications. Benchmark scores still pending.

OneJevstatefuloriginal500replay reached125/500 with500examples, actualcomplete
FP32/Adam/scheduler/RNG/cursor checkpoint9,029,140,249bytes. Genuineintermediate
125BF16MMcheckpoint reload is executing onGPU7; notformal500completion.

ActualOne125MMreload terminal:474weights, BF16changed2,306,395/2,035,222/
1,627,638elements in threeMLPprobes vsinit100; independent nativeMMreload
probabilitydifference0(autoFork unchanged). Onlyintermediatecorrectness, no
formal500or trackinggain/exactFullFP32nextstep claim. GPU7probe exited.

### Native Ground formal200 tracking adaptation (registered)

Advance3step projector proof to200seed42steps on32dense train images plus
40fit-v2KITTIimages with2expressions each (112native SFT rows/56pairs).
Alternate dense/ref; allpairs visited. Preflight proves entireselected target
sets grid-representable, not clipped: MOT20window has69out-of-grid targets
and4otherKITTIwindows haveviolations, so theyareoutside thisnewregistered
native-token corpus, not droppedfromoldprotocols/evaluation. Frozen native
vision/language, trainoriginal27Mprojector FP32; officialMDM.5/causal.5 plus
.1GTidentity/.1actualnative-proposalidentity contrastive. Native boxes come
fromcompletedoriginal decoderrollouts, GTonlylabelalignment; adaptedlive
decoderrollout notclaimed. CompleteAdam/RNG/cursorcheckpoint every25steps.
Fulloriginaldense/ref9 inputs are ready; adaptation benefit remains unproved
until original unmodified before/after scoring. Newplan native_ground_tracking200_20261014.json.

Native200v1all112rows/56pairs fit native collator; actual0negative ref rows
(so absent-target supervision remains a later required stage). Firststep1
completed nativeMLM+teacher/native temporal gradients,27,267,584parameters;
step2cleanBF16attention cuBLAS internal failure onGPU7 before next update.
No validoptimizer/RNG checkpoint yet. Preserve v1failure/step1log. New GPU6
executionvariant holds actualdata/seed42/200steps/loss/optimizer unchanged,
adds checkpoint atstep1 toexisting25interval. SameA800architecture, noGPU
reset orforeignjob mutation. Externalrootcause notestablished; verify actual
secondbatch in freshGPU6context while replayingunfinished initialunit.

GPU6same protocol progressed3steps then failedcuBLAS;1fullstate saved.
Sync nativeFlexAttention beforecleanattention diagnostic resumedexact1state,
reached20steps then exposedillegalmemoryaccess immediatelyafterNoisyFlex
(step21), so clean cuBLAS was downstream/asynchronous symptom. NoGPUreset,
no samples/masks/seeds/loss changes. Native trainingprofile pinsTorch2.10,
whereassmoke usedServeTorch2.9.1. PrepareindependentPy312Torch2.10cu128
(A800/driver570-compatible CUDA family) /nativeTF5.7/Triton3.6; no serving/
One env edits. Resume native200 fromvalid1withall optimizer/RNG/cursor.
Any runtime numerical differences are to be recorded, not silently ignored.

NativeCPUevalenvironment officialsetup completed (Torch2.5.1CPU, shipped
lmms snapshot; serving/training environments unchanged). Startoneowned
originalGroundservice GPU7 + originalfull9canonicalbenchmark evaluator
CPU-only, limitnull,91,124declaredrows. No8sample-smoke orretentiongain claim.
Nativeevalresponsecache is originalscorer output, not U2candidatefeatures.

Originalfull9preflight/service ready; all9clients failedbeforefirstinference
while constructingHTTPproxy mappings in SDK/httpx0.23. Task-inherited proxy
configuration conflictswithlocalAPIclient; cleartasklocalHTTP/HTTPS/ALLproxy
vars only, preserveglobalenv/source/metric. Fresh v2runid retainsold9failure
logs/caches; all9unfinishedcanonicaltasks re-executed, no sample/seed changes.

IsolatedTorch2.10cu128/TF5.7/Triton3.6 installed; missingSciPy preflight
fixed withoutchangingdata. Actualstate1 resumed, validthrough57 with55full
state. Earlierbad-runtimegradientnorm at20=387471 vsnew=13.49, numerical
unsafe-runtimebehavior retained, notclaimed equivalent. StillillegalnoisyFlex
access at58, so runtimeupgrade alone is insufficient. Concrete-shape compile
specializes SAMEoriginalnative_flex_attention_kernel/masks/GQA; no score/
loss/model changes, no dense-reference fallback. Resume55in freshv6out.
Recompileguards raisedonlyforthefinite declaredvariable-shape corpus.

### Native Ground200 concrete-shape failure and targeted diagnosis

Concrete-shape v6 completed185updates and saved full FP32/Adam/allRNG/cursor185, then failed before186update with CUDA illegal access. Static shapes alone are insufficient; no metrics or samples are changed. Previous output handle is absent and owner terminal. Next action: replay from185 with CUDA_LAUNCH_BLOCKING and a persisted Q/K/V boundary plus full traceback, same operator/masks/data/seed/loss. This localizes the actual first failing kernel before choosing a repair; failing attempts remain preserved. Full original dense/ref9 baseline handle73408 was confirmed live, so it is not restarted.

### Native Ground200 resumed completion and full original-protocol retention

v7 synchronous replay from185 completed186–200, keeping exact native operator,
masks, seed42, data/loss/optimizer/schedule. Asynchronous kernel root cause is
still unestablished; all v1–v6 failures remain. Complete checkpoint chain:
v2step1, v5steps2–55, v6steps56–185, v7steps186–200. All200loss/gradient rows
finite;100dense+100ref, all56pairs visited. Native200reload maximum delta0;
27,267,353FP32 and16,204,590BF16 elements changed. Exported complete569tensor
9,687,419,704byte model; only five projector tensors replaced. These are
training/execution results, not benchmark or tracking accuracy improvements.

Original full9baseline handle73408 remains live. New adapted full9owner uses
GPU6/API8102 and separate execution rendezvous ports22345+, preserving all
canonical tasks, full rows, original scoring and native decode profiles. No
benchmark GT training or limit reductions. Four project GPUs maximum; other
projects are untouched. Real absent-target40rows remain a separate next stage.

Fresh nativeGround200 MOT4/RMOTsingle8/RMOTmulti12 before/after controls also launched through the two existing owned retention services, same Simple associator/parser/tracking_dense/seed42, no new GPU service or benchmark-limit changes. This48frame functional comparison is separate from original full91,124-row dense/ref9 scoring. All registered empty/rejected frames remain.

### Ground200 short-control negative transfer and registered absence120

All48frames and six rendered videos validated. Same fresh Simple associator:
MOT4 HOTA75.2498->84.5532%, IDSW2->0,FP16->2,FN24->13.
RMO single8 HOTA26.8665->25.2769%, empty-frame outputs3->8; multi12
HOTA24.7800->23.9589%, IDSW0->2. All empty/rejected frames retained, official
GT/scorers unchanged. These are single-seed short scopes, not full datasets.
Hypothesis: missing absent-target supervision contributes to referring false
positives; not established as the sole cause. Keep original200 and bothfull9
retention owners unchanged. Register a separate additional120step curriculum:
initialize exactFP32 projector200, explicitlyresetAdamW/lr1e-6, seed42;
all112existing rows+40previouslyregistered actualfit absences=152rows/76pairs,
all76visited. Same native loss/teacher and native identity losses. No synthetic
emptyimages, no evaluation labels in training. Verify first real native noisy
attention forward/QKV gradients for dense, positive referring and absence
against PyTorch eager mathematical reference; diagnostics consume no RNG.
Idle ownedROIservice5 stopped after no established clients; all consumers were
terminal. GPU5 is reused for this declared training, no other project stopped.

Absence120v1 preflight verifies152rows/76pairs/40realabsences, but stops beforefirstupdate: detached reference-probe backward inherits native activation-checkpoint saved-tensor hooks and recursively re-enters itself (_StopRecomputationError). This is a diagnostic integration failure, not a measured attention discrepancy. Small repair: isolate only detached diagnostic saved tensors with identity pack/unpack hooks; production model checkpointing and forward remain unchanged. Preserve failedv1. Freshv2 replays unfinished stage from exactTrackProj200, identical seed/data/loss/criterion.

Actual v2 native attention reference checks pass for first real dense/positive-ref/absent-ref layer: maxBF16 output delta.005859/.003906/.005859; Q-gradient relativeL2.043767/.033754/.036040, K/V below.007. Criteriaunchanged(output rtol/atol.02; QKVrelativeL2<.05), no diagnosticRNG consumption. This targeted check covers three real input types, not every layer/step. Training remains owned by the same v2blocking command.

### Real absence120 completion and unchanged native200/320 tracking comparison

Additional120 completed: all76pairs/152rows visited, including20actualabsent
pairs/40fit-negative rows; finite losses/gradients, FP32changed27,266,797,
BF16changed8,256,449 versus initialization200, native reload delta0, peak
allocated18,414,616,064bytes. Complete200model/full9owners retained.
Export full569tensor model320 (explicit200+120; five projector tensors only).
Next test: fresh200-versus320 native MOT4/RMO8/RMO12, same Simple associator,
parser/tracking_dense/seed42/GT/empty+rejected frames. New ownedGPU5/API8103
service only, auto-clean at terminal; existing200/API8102 onGPU6 untouched.
Short-scope result is not fulloriginal9retention or a completeRMOTclaim.

### Absence320 short negative result and native sampling-counter confound

Fresh shared200vs320 controls finish48frames/sixvideos. MOT HOTA83.737->83.402,
RMO single18.406->20.046, multi23.496->24.817%. Empty outputs unchanged7and3;
IDSWsingle1->2/multi2->4. No successful absence correction is established.
Originalnativecode inspection finds sampling seed depends on monotonically
incremented request_index, not just suppliedsampling_seed42. Shared full9
service consumes many intervening requests, so preliminary checkpoint
comparisons do not pair effective stochastic seeds. Preserve their real scores,
but do not attribute small changes solely to adaptation. New explicitly
registered control: all original/200/200repeat/320 on fresh isolatedGPU5
services, identical ordered24requests each, native source/math/seed42/scorers
unchanged. Require repeated200raw wire/candidates/outputs identical. Keep
both originalfull9owners unaffected. This is a small controlled protocol
verification, not a repetition of full91k benchmarks or threshold tuning.

Registered one finite blocking OneJev500completion owner: wait exactlive originaltrainer and verify terminal500receipt, actual474weight nativeMM2reload, then same candidate/GT/core4way Simple/pretrained/init100/500 controls acrossMOT/RMOT/TAOBase. GPU4 reused only after originaltraining exits. No original500restart or math/seed/corpus change. No detector E2E or Novelclaim from replay/Base-only clip. Fullvideo500before/after remains subsequent required work.

Isolated96v1 original24frame unit completed, then next immediate replica bind failedEADDRINUSE after ownedserverparentexit. Later targetedlistener/bind check shows portfree; sourceparentwait does not guarantee immediate TCPport reuse. Small execution fix: SO_REUSEADDR probe plus internal30second port-release wait, no model/math/seed changes or other process signals. Preserve originalfailedowner; newv2reuses complete original24rows/metrics and resumes only200/200repeat/320fresh replicas.

### Isolated native96 completed: confirmed positiveMOT/negativeRMOT transfer

All96frames/12videos validated;200repeat rawwire/candidates/causaloutputs
identical on24frames. Original/200/320 HOTA: MOT75.0687/83.4673/83.4016,
single30.3853/24.8762/20.0457, multi22.6643/23.6754/24.8173%. Singleempty
outputs3/7/7; multiple2/3/3. MultiIDSW0/2/4 and AssocA41.678/33.457/30.669.
Keep adaptedcheckpoints as research controls, do not promote320as unified
foundation. No successful absencecorrection. FixedSimple means observedMOT
benefit is front-end adaptation, not a claimedJEVassociation gain. Shared
service metrics remain preserved but not causal checkpoint evidence. Both
full9owners unaffected; isolatedservices all stopped. Next major stage is
actual500nativeMMreload and exact-proposal Fullbefore/after/multitask controls;
completion owner already waits verifiedcurrenttrainer, no restart/sample skip.

Originalfull9 v2terminalpartial: COCO allcanonical rows completed; other8clients HTTP503 queuefull. Native service defaultsmaxrunning1/maxqueued2 versus9clients*concurrency2. Small execution repair: task-local boundedqueue32, unchangedsource/scorers/data/limits/concurrency/genprofiles/seed42. Actual18parallel HTTPregression in a separate ownedservice, then freshoriginalresume. NativeCLIrejectsreusedoutput; wrapper invokes same nativefunctions in a freshresultroot, copies all-sample completedCOCO result and SQLitebackup partialcaches, spawns onlyunfinished8tasks. Preserve originalfailure/results; no exact stochasticcounter-resume or causalretentiongain claim.

Confirmed sameHTTP503 failure in8of9adapted200tasklogs. Verified nativeevaluator1053747 command/cwd and requested its supported cooperativeSIGTERM; originalowner97074 then failedterminal and stopped only its ownedservice. Preserve allpartialcaches/failedlogs/nativecancelledsummary; resumeall9unfinished after queue32regression, same200checkpoint/parser/scorer/data/seed42. No unrelatedprocess signals. Avoid waitinghours for a doomedpartialnine-task unit.

Queue32regressionv1timeout: newserverlog actuallyEADDRINUSE, while HTTPmodelsprobe accepted oldorphanAPI775339/1051641 after their launchers exited. Verified commands/cwd/model and originalisolatedPGIDs775337/1051638; priorTERM/grace hadfailed, targetedSIGKILLonlytheseownedgroups. Rootcleanupbug: waitlauncher returns before orphanAPI exits. Newhelper verifies/waits allownedPGIDmembers, forcekillsafter20secondfailedTERM, checksnonealive; add portbindguard to regression and recordgroupPID. Preserve oldfailedregression and recoverygate; rerun same18client queuecriterion in freshv2, no nativebenchmark protocol changes.

Freshclean-port queue32v2 actual18parallel callbacks allHTTP200; fullownedgroup1339199 verifiedemptyaftercleanup, old775337/1051638 verifiedempty. Recoveryv2 hadlocalargparseNamespace overwrittenbyprocargvlist beforeanychildrenlaunch; repairloopvariablename, preservefailure, v3runs samevalidregression gate/no retest or changedcriterion.

Queue32v2passed all18HTTP200; verifiedfullgroupcleanup and currentoldAPIports released. Full9recoveryv3startsoriginal8remaining plusadapted200all9 with validatedCOCO4952/4952reuse andSQLitecache backups through unchangednativefunctions/scorers. Keep allfailedstates/cache prefixes; no samples skipped/metric changes/base seed changes or exactstochasticcontinuation claim.

### Fixed-checkpoint full-video memory intervention (registered)

Read actualCompact/tracker: neuralhistory merges immutableanchor/recent/long
native2560features, learnedupdate>=.5, 3secondlifespan with learnedtermination.
Hypotheses: persistentanchors/gating improve naturalgap recovery, while always
updates can contaminate;30secondretention may recover longer gaps but increase
competition. Keep exactbroadnative1200checkpoint/1203originalV1candidates/
realnativeROIs/GT/scorer/thresholds. Six frozenmodel inferenceinterventions:
referenceanchor+gated3s, immediatepreviousframe noLostmemory/recentinput,
latest-observationinput withnormal3sretention, unconditionalEMA(.1)input,
anchor+alwaysreplace3s, anchor+gated30s. EMA is a convex mixture of realfeatures,
not a synthetic image encoder; lastreliabletimestamp still uses learnedreliability.
These are input/policy interventions, not independently retrained competitors.
No nativeDLMquery/newcache; reuse790,667,425byte sharedfullvideo featurelog.
Only featurearrays omitted from new logs, retain source reference/allboxes/
labels/scores/decisions; actualNNinputs unchanged. Evaluate allframes/rejected
frames with originalTrackEval and posthocfixed-birth-anchor recovery/NoMatch
counts (never inputGT/replaceofficialmetrics). Actualnativefeature2frame EMA
memorysave/load regression precedes fullruns; defaultprediction retention also
checked against preserved existing baseline, recording any numerical variance.

Actualnative two-frame EMAprobe passed:2560wide features, one matchedpriortrack expectedconvexupdate, originalanchor unchanged, forcedwrites leave lastreliabletime honest, restorednextoutputs/state exact. Sixfull1203frame memoryowner42123launchedGPU5; featurearrays omitted only in duplicatedlogs, exactsharedsource retained for inference/rehydration. Originalfull9owner39013 and One500/500completion owners remain active without restart.

Prior fullvideo posthoc: GTobservations4586/nativeone-to-one matches4341,
existingbirth-anchor associations575/1768, modelcorrectbeforematching651,
globalcompetition changes94correct->wrong and18wrong->correct. MatchedGT
NO_MATCH2100predictions with0option-missinggoldpositives under declared
birth-anchor rule; this diagnostic excludesunmatched/ignoredregions and is
notofficialNoMatchtruth beyond its rule. Candidate-gap returns194, ≤1s recovery
9/177,1–3s0/4, >3ssupport0/null. This video cannot establish long>3srecovery;
keep30sintervention asretention/competition sensitivity, not invented support.

Novel evaluation breadth: select lowestexistingcompletevalidationvideo with
bothofficialBase andNovel: TAOvideo63,40annotationframes/160GTboxes (Base120,
Novel40peeler), allpresentwithcorrectimagegeometry. Fullfixed1203vocabulary
unchanged, GTonlyscopeinventory/evaluation, noNoveltraininglabels. Register
40fullannotation-frame online nativeDLM520classqueries plus realnativeROIs,
then fixedproposalSimple/sharedCompact controls with unmodifiedTETA. Explicit
per-queryrejection contributesempty for thatquery only, concatenates remaining
original-parserboxes, noNMS/topK/GTselection; matches existing scored-query
control policy, not old whole-frame-rejection scores. One blocking owner waits
currentGPU5memoryjobexit beforeowning temporaryDLM/feature services; stops
bothcompletegroups beforecorecontrols. Timestampannotationsteps, notphysical
seconds/FPS. No rawdatasets/modelweights uploaded or all-datasetfeaturecache.

Prepare explicit nativeOne200curriculum for proposal-domain mismatch: retainall20881existingteacher-weighted rows; native5661unique realDLM/uncorrectedown-history rows identityweight4=>9213, deterministiccyclicexpand to20881, total41762equal-source rows. No eval/Novellabels, no state correction or omittednativeunit. Register matchedteacher-only200 andnative-mixture200 fromsame eventualOne500FP32 initialization, same lr2e-6/seed0/budget/officialloss. These are additional200 stages, not changes to current500. Run officialCPUplan-only with existing100tokenizer/config (same released tokenizer/visiongeometry) before anytraining; reject anydroppedrow. Stage launch requires actual500completion/evaluation and correctFP32 initialization.

Beforetraining, strengthen domaincomparison: preparedv1eligible familyratios differed. Preserveunusedv1; v2pairs eachteacherrow withnative row ofidenticalquestionfamily (9840identity/2460update/2460reliability/5951lifecycle/170withheld per source), equal50/50sources41762rows andall5661unique native rows. No label/dataframe or metricchange. Preflightv1stilldiagnoses allunique originals, newv2willget its own officialplan preflight; current500untouched.

Realfull375stage initialization probe passed: all320trainableparameter tensors/752393024parameters copiedfromactualFP32trainingstate withmaxdelta0;498430914elements differfromBF16inference-export-derived initialization. CPUonly, no old375weights/state/live500modification. Newstageinitializerexplicitreset is distinctfrom exactresume. v1officialplan-only41762rows max26028/cap32768, zero drops; exactfamilymatchedv2officialpreflight nowrunning separately. Stage200launch remains pending real500completion/controlreview, no blind extra training started.

### Full7218frame memory interventions complete

Allsix1203frame traces/videos/GT/scorers/samecandidate+ROI sources validated;
all27native rejectionframes eachretain denominator. Freshdefault matches
preservedGPU7/resumed baseline exactly, maxprobdelta0. OfficialHOTA/IDF1:
referencegated3=26.3157/30.6329%; frameonly11.4408/8.4615;
recentobs3=16.3943/18.2023; EMAalways3=14.6426/16.8764;
anchoralways3=24.2449/29.8179; anchorgated30=18.5523/19.8490.
ReferenceIDSW165 vsalways154, but referenceFN1962 vs2143 andFP579vs604;
no claim that gating improves everymetric. Keep default3s; longerretention
worsens thisvideo and haszero >3snaturalrecovery support, so do not claim
long-term recovery. Anchor/update evidence is frozenmodel interventions only,
not optimalEMA/newarchitecture superiority. MatchedGTNoMatch excessive;
nextnativeOne data-domain intervention remains pendingreal500results.

Selectedv2family-matched native mixture officialCPUpreflight passed41762rows, max26028/cap32768, zero drops andPLAN_OK,119seconds,723724879tokens. Alloriginalunique native rows retained. RealFP32stage initialization proved on375fullmodel; no additional200training started before500result review.

TAO63v1 failedbeforefirstnativequery: stream call omitted requiredexpression parameter (OVMOTNone). Bothownedservicegroups stoppedverified. SmallfixexplicitNone; preservev1empty prefix/failure; v2replays entireunfinished40frame unit withidenticalvocab/parser/scorer/seed/scope, actualfirstquery is targeted regression beforeremaining519requests.

Adapted200full9 child terminal separately: schedulerGPUlaunchfailure at00:04:41 inrawforward end sync, all9clientconnection failures downstream. Queue32isnotcurrentroot. No completedtask metrics; caches/logspreserved. Firstactionablediagnostic: task-localCUDA_LAUNCH_BLOCKING1 in freshGPU6service, resume9fromlatestv2responsecaches withsameoperator/model/data/seed/scorers/concurrency/queue. RootGPUkernel stillunknown, no reset/driver/globalenv/foreignprocess changes. Originalnative8taskowner and parent39013 remain live; newadaptedrecovery has its ownowner without duplicateoldservice.

2026-10-15 publication: all39files for completed six full1203 memory interventions, official metrics, predictions, SVG and matched real-native One continuation preparation remotely verified: GitHub085125569af96d47b89f2fd269c6501f02b0b28f tree3ffa25f12188b6190992de0baad740735fc9c43c matcheslocal034bc056e812d45f66eea34d54b1ca70cdb817e8. Existingblockinghandles confirmedOne500450/500, TAOBaseNovel63v2live, adapted200syncv3live. Retain failedadaptedv2 and originalfull9owner; no additional200launch or globalGPUreset.

Hypothesis before CPU RMOT diagnostic: validation membership failures may reflect weak expression response or proposal/history domain mismatch despite fitted training examples. Keep three existing semantic checkpoints, exact20registerednative validationframes/realROI vectors and unchanged0.5threshold; causal primary-query actor, compare None/counter expression only on captured same core input, allGT posthoc. Quantify fit membership and real validation query response plus inverse-aligned candidate/history permutations. No new training, synthetic features, altered official metrics or GPU consumption.

CPU real-query diagnostic complete on3existing semantic Compact checkpoints,20registerednativevalidationframes/27actualcandidateobservations. AllCPU replay boxes/IDs identical to recordedGPU outputs afterexcluding floatingprobabilities; maxmembershipGPUCPUdelta7.152557373046875e-5. Within captured candidategroups/shortlists inverse-aligned6headpermutationerror<=6.556510925292969e-7, not fullpipelinecrossgroupproof. Broadnative redcoatwoman6primary-onlyGTcovered observations havecorrectquerymean.432221 vs arrangingautomobilescounter.772220,0/6correctabove.5 vs6/6wrongcounterabove.5. Broadteacher same6wrongcounteralso6/6; language response is present butwronglydirected, not just orderbias. Fitoriginaljoint45positive/51negative perfect; broadnative282/316positive and460/588negative correct atunchangedhalf. No thresholdtuning/GTfeedback/officialmetricreplacement. FirstCPUv1diagnostic compared floatingqueryprobability in outputgeometry; v2 excludes both association/queryprobabilities and separately reports numericaldelta, preservesv1.

ActualOne SFTscope audit: original50020881rows andpreparednative41762rows ALLstate.task=mot,personvocabulary,noexpressions. Familymix is not tracking-taskmix. Preserveoriginal500 and matchedMOT-domaincontinuationcontrol, but these cannot prove unifiedFullsemantic adaptation. Nextimplementation: opt-in genuine nativeOneJev Noul query membership, defaultbehavior unchanged; export allactualDLM candidates for144primary/counterquery contexts across72legalfitRMOTframes withrealcropmedia, GTonlyoutsideinput and instantaneousquerysupervisionexplicit. Run officialCPU plan-only before anynewtrain; actualnativekernelruntime/semantictraining remainsrequired.

Fullsemantic preparation retained: genuineOneJev Noul membership opt-in implemented; all346nativefitRMOTrows officialCPUplanOK,max2293,0drops,457637tokens. v1data-rootfailure beforefirstrow preserved; v2usesregisteredreleasepath, all72frames/144contexts/all165nativecandidates twice/58positivemember targets retained plus16presencequestions on8empty/rejectedframes. Basecategory430questions from215GTteacher crops/64registeredtrainframes/9Baseclasses, all1203vocabchoices retained, noNovelGT orunknownnegativesupervision, explicitconditionalrootteacherforcing; officialCPUplanOK,max4233,0drops,1283259tokens. Sharedcategorybuilders actualsix-object root/leafrequest+mediapixels exactlyequalto8b880source undercapture-onlyrouting, notnativeinference. One blockingnativeOne500Noul4frameprobe owner waitsverifiedexistingcontrollerPID1209639/startticks thenrequiresitscomplete500controlsreceipt beforeGPU4reuse; session50119. No additionaltraining launched; actualOnekernelprobe/multitaskSFT and completevalidation remainrequired.

Plan for genuine sharedFull semantic adaptation: preserve original500MOTstage and existingMOT-only200study as separate protocols, neither is unifiedSFT. Prepare explicitadditional600matched arms from actual500FP32masters: teacher-onlyMOT vs50/50teacher/nativeMOT, identical common346nativefitRMOTmembership/presence and430trainBasecategoryroot/leafrows. EqualeligibleMOT/RMOT/OVMOTbucket counts bydeclaredcycling, everyexistingeligibleMOTrow andsemanticrow retained; exactMOTquestionfamilycounts matched. Sameofficialsampler/loss/seed0/lr2e-6/4rowsperstep/frozenvision/752393024trainablelanguage params/32768context; budgetmatchedoptimizersteps, notidenticaltokenFLOPsexposures. NativeRMOinstantaneous(noOne500historyclaim) andBaseGTcropteacherstage explicit; noNovel/validationGTtraining. Computeconservative25GiBfeaturebudget beforegeneration, officialCPUplan-only all125286rowsperarm; actualtraining launch awaits500matchedcontrolsreview and nativeNoulprobe. New600stage is registeredresearch, not silent alteration of existing500/200protocol.

Unified600v1 bothofficialplans completed125286each,zeroDropped,max22817teacher/26028native. Actual600prefixes2400rows haveMOT842/847,RMO799/801,OV759/752; only686/1558teachersemanticrepetitionIDs shared. Native length sort changes actualsemantic exposure despiteequaleligiblebuckets. Beforeanytraining registerv2: adjacentteacherMOTduplication aligns everyMOTquestionfamily withnative mixture; explicitpaired-index-shuffle-v1 underseed0 makes botharms selectidenticalrowindices/stepsemanticquestions/families, retainsallrows/nativebatchmax1/native4rowstep/loss/AMP. Optionalnew-stageflag only; defaultold500/200sampler and checkpointprotocol unchanged. V2checkpointprotocol recordssampler onresume. Preservev1data/plans; count bothversions conservativelyunder25GiB.

ActualOne500complete at2026-10-15 01:30:56:500updates/2000examples/33469.233s, valid9,029,140,249byte fullFP32AdamRNGcursor state. RealnativeMMtworeloads maxprobabilitydelta0, all474exportentries retained. CompleteCPU500masterinit320/752393024 restored0,498431025F32elements retainedvsBF16init; originaldefaults checkpointprotocol exactlycompatible. Fullfrozenvisual153tensors unchanged;BF16uniquechanged333632768afterdeduplicatingtiedembedding+lmhead (254279680duplicatedexportelements). Actualunique0.852985920Bparams incl0.100592896Bfrozenvision,0.752393024Blanguage. Matched16controller stillfinishingTAOunits; completeMOT/RMO review mixed:pretrained/100/500MOT HOTA.343944/.197395/.308680 vsSimple.748813,IDSW47/0/57 vs1;singleRMOp500.248862 vsoriginal0/Simple.242284;multi.214246 vsoriginal.222720/Simple.243151. No promotedmodel ormultiTaskSuccessclaim.

Unified600v2all125286rowsperarm passnativeCPUplan,0dropped,max22817teacher/26028native. Bothactual600prefixes2400rows exactly811MOT/749RMO/840OV; allper-stepquestionfamilies and1589semanticrepetitionIDs exact. DirectactualJSONpayloadcomparison all83524eligiblecommonsemanticrows (1589selected) byte-identical inclstate/question/media/target/provenance. Prefixtrainingtokens15774003teacher vs17216498native, notFLOPmatched. Conservativeexisting+newv2text/features16,289,535,076bytes<25GiB, v1data/failedcomparisonproofretained. Registeredfinitepaired600owner waits verifiednativeNoulprobe/500controls then startsGP4teacher;GP5native onlyafterexactownedTAOparentexit+emptyservicegroups, independentTAOfailuredoesnotglobalNoGo. Bothreadimmutable500masters/newstatepaths; streamactualchildlogs; nativeofficialcontrolsvs500andcompletevideoE2E remainrequiredaftertraining.

NativeOne500Noulprobe actualfourframescompleted (150..153, candidatecounts3/1/0/2) andsixactualprobabilities.172542..176372; ownerfailedonlyKeyErrorGT_read becausevalidatorlookedinsideOneJevmetadata ratherthanexistingrun-levelgt_read_by_runtime=False/future_frames_read_by_runtime=False. Preservev1nativeoutputs/failure and prelaunchpairedownerv2whichcorrectlyfailedbeforeanytraining. Fixvalidatorfieldlocation, add futuremetadataGT_read consistency, freshv2validation reusesentirecompleted4frame run withoutnewinference/skippedframes orweakenedcriteria; no accuracyclaim.

Original500matchedcontroller nowterminal0, all16officialcontrols/112decodedframes validated. TAOBase4pretrained13.243/10011.732/50011.981TETA, allClsA0 underlegacylabelpolicy; no consistentfoundationgain. NativeNoulv2validated all4actualcompletedframes/6probabilities via correctrun-levelGT/futureflags, no inference rerun or criterionweakening. Resource/GTvalidator firstGPU-memoryfield lookup assumedSimple hadlearnedGPUallocation; fixedSimpleunmeasured value=null, not0, targetedall16videos rerunpasses. Pairedownerv3actualteacherprocess launchedGPU4 (72945MiBfree atlaunch), actual500FP32copy320/752393024 difference0 andpaired_index_shuffle=True inrealargs.json. Native5arm waits exactliveTAOowner1444656/start37238782 andemptyownedgroups. Blockingpairedowner session33936; no newtrainingpoll oroldU2queue. Cumulativebudget andper-taskofficialbefore/afterevaluation remainpending.

Research plan: current3legalReferKITTIvalidation label scopes0007/0015/0016 have0naturalGTidentitygap>3s; do not infer longrecovery support from this. Audit onlyregisteredMOTvalidation frames for naturalannotationgaps/lowvisibility intervals, nevermaskpredictions/createocclusion. Extendposthoc physicalidentity/memory/NoMatchdiagnostics toRMO(allreleasephysicalGT, querylabels separate) andTAO(nonexhaustiveoriginalGT, unknownFPSannotationunits explicit). RuncompletedrealOne500all16scopes onCPU withoutGTfeedback/officialmetricreplacement; preserveoldMOTdefinitions. Thenregistersame-native-candidate native-semantic500/600 evaluation owner afterpairedtraining ratherthanmixlegacydefault andsemanticflags.

2026-10-15 13:05audit: oldpairedowner1557613 andchildren1557618/1584409 all/procabsent; unifiedhandlesmissing. No capturedPython/math/CUDAexception, logsabruptlystopteacher475(native250). ValidfullAdam/FP32/RNGcursors teacher450/native250 retained. Terminationsignal/origin notrecorded; uptime disproves assumingnewboot. OldchildPopeninheritedownerPGID/stdoutpipe; modifyexecutiononlydetachedsessions/directfilelogs tosurviveownercleanup. Freshdetachedresumeowner preserves originalunsaved475log beforetrimmingmainlogto450, restoresfull450/250 state andsampler unchanged, nooptimizerreset orBF16restart.

NativeTAO63streamcompletedall40frames/520requests:317scored/203originalparserrejectedqueries,35228boxes,1885903934bytes,10850.079sperception+ROI(noE2Eclaim). SimplecontrolfailedstatePNGserializationframe755 after28completecausaldecisionlogrows/27encodedframes. Actualnativebox[92.18018,522.51652,151.07307,523.23624] yields59x0PILcrop fromsubpixelheight. PreserveboxinclFP/noNMS/noGTclipping. Newmemoryv3sentinelexactemptydimensions/mode andloadroundtrip, noinventedpixels; nativecategorysheetmarksno-pixelsforactualemptycrop. Actualframe755probe passes unchangedbox/output count. FreshCPUcontrolowner resumes28validdecisions, all40frames required; sharedCompact CPU sameweights/ROI/source withdevice explicit, noGPU/E2Espeedclaim.

Registered same-semantic-policy before500/teacher600/native600 evaluation on GPU4 after verifiedteacherchildexit. Exact existing MOT4/SingleRMO8/MultiRMO12/TAOBase4 native candidate and frame sequences preflightpassed; unchangedMOT500 reused, newnativeNoul.5 andhierarchical1203category policy applied tobothbefore/after. Actual600MMreload comparesBF16exports against500 and repeatsnativeMMtwice withunchanged1e-6criterion. This12unitshortcomparison doesnotclaimNovel orfullvideos/E2E. Naturalvalidationaudit completed36sequences/647annotation-definedchallenges; selectedcompleteMOT20-05registeredhalfvalidation1659..3315 includes29.76s lowvisibilityinterval, notactualcandidateabsence/recoveryproof. No artificialocclusion, candidatefilter, testGT orofficialmetricchange.

B2 research plan: run originalU2SharedTrackCore unchanged with newlearned2560to256GroundROIbridge on same48legalnativeMOTtrainingframes, originalsixlosses/GTforcing/queryaugmentation andoriginalGTignoreloader. Newseed0training300updates; no existingtrainedU2trackingcheckpoint. CPUonly/noextraGPU/nofeaturecache/newperception/oldU2queue. ExactsameV1DTnativecandidates for4framefunctional memory roundtrip thenfull1203officialTrackEval. Same300updates asCompact butdifferentparams/losses/optimizer/frameexposure/maxage, thereforearchitecturebaseline notisolatedJEVeffect orparameter/FLOPmatch. MOTonly; nofabricatedRMOT/OVMOTcapability.

TAOCPUrecovery Simpleterminalvalidated:40decodedframes/35228nativepredictions/160GT; officialTETA18.640 Base23.3725 Novel9.1737 NovelClsA0. Alloriginal520queryoutcomes preserved. SharedCompactsamecandidateCPUcontrol active; noveldetection/localizationdoesnotmeanNovelrecognition. B2preflightreconstructedexactnativeboxes/features onall48frames, three16frame sequences,45adjacentobservationpairs (includingactualsourcegaps); noallocatedfeaturecache. DetachedB2ownerlaunched withall300updates/fourframefunctional/full1203officialcontrol inoneshellwait.

B2actual300trainingcomplete17.325s CPU,5942795params/4361991changed elements/nativecheckpointreload0. Fourframevideo/eval/memoryroundtrippassed. Fullvideo failed199 after198validframes: actualfinite retainedtrackcoordinates~7.6e19 andvelocity145.9 cause FP32norms inf/inf incenter-distance; allmodelweights/currentfeatures/statevaluesfinite. NotGPUbackend/GT/NaNfeature. Firstcandidatefix alloverflowingnorms altered previouslydefinedframe198pairlogits by.12063676 andwasREJECTED beforevideoresume. Smallacceptedfix recomputesonlyundefinedcenter-distance ratios inFP64, leaves everydefinedFP32distanceexact. Full198frame replay reproducesallpair/new/noMatch/alive logits andIDs exactly(maxdelta0), actual199passesfinite. Savevalidatedstate198; freshv2owner reusesreal300checkpoint/completedfunctional4 andstrictall198sourceprefix, resumes199..1203, no retraining/weight/seed/candidate/threshold/memory-lifetimechange. Originalfailure/rejectedregression retained; no shortened metric scope.

B2v2full1203terminalvalidated: all1203decodedframes/same13664nativecandidates/all27rejectedframes; originalTrackEvalHOTA.0124341105732 IDF1.00625459896983 IDSW250 FP455 FN4191. Completegeometry/order/imageframeagreement andperframeIDuniqueness checked. Strongnegativebaseline(majorGTmisses), noIDSWorJEVcausalitysuccessclaim. Actual17.325secondtraining/5942795params/4361991updates/reload0 retained. Resumed198prefix bypassescorecompute; removedinvalid standaloneassociationFPS metadata (raw26.915 retainedasresumedscopeencodingtimingonly), preservedrawreceipts andreranunmodifiedofficialscorer: allaccuracymetricsexact. NoE2Espeedclaim. Naturalchallenge manifest nowlinksitsalreadyregisteredMOT20protocol forunchangedofficialevaluation; images/GT/scopeunchanged.

Researchplan fullRMOT: twoexistinglegal0007full800frame expression scopes, firstgenuinefreshsharedCompactsinglepass throughoriginalDLM+actualROI+memory+video; include sidearchivewrites inactualE2Etime, remoteinitializationexcludedasinpriorFull. Newoptional run_video currentnativearchive preservesfeaturesonce andallowsgeometryonlylogs; olddefault/replayprotocolunchanged. Exactsame archived candidates thenSimple/pretrainedOne/500/teacher600/native600 nativeNoul.5/v3controls, officialqueryTrackEval andphysicalIDposthoc. SingleGPU5owner waits bothverified600childrenexit+valid600master/done beforeallocation, independentGPU4shortcontrols/nativefull9 unaffected. Budgetnew2GiB acrossbothsources, conservativeprior16.289GB+TAO1.886GB+new2.147GB=20.323GB<25GiB; no duplicatedfeaturesincontrol logs/datasetcache/oldU2queue. Actualnative archive/replay/cropmemory roundtrip targetedregression onrealDT4frameROI source beforequeue. No fullRMOTdataset/CompactFullreasoning/standalonereplayE2Eclaims.

Currentnativearchive regressionpassedactualDT4frames/13nativeROI2560candidates/742680bytes: botharchival and replay geometry/output/fullmemory states exactlyequal, GT/futurereadfalse, replayE2EFPS correctlynull. BothregisteredfullRMOT0007single/multi manifests validateall800images0..799/currentlegalv2validation withoutopeningGT. NewfullRMOTowner preconditionbindsthisactualreport; featurebudgetwithprobe20323665338bytes<25GiB. QueuewaitsGPU5traininghandoff, nochangescurrenttraining/nativefull9/TAOCPU owners.

RawTAO40bothcontrolscomplete: SimpleTETA18.640/AssocA28.618/ClsA20.290 vs sharedCompact14.517/5.1478/30.314. BothNovelClsA0; CompactBaseTETA18.93545/Novel5.679 <Simple23.3725/9.1737. CandidateGTrecall.91875 both, butcorrectphysicalanchor32optioncoverage.156716 (Simple1). No promotion. GTfreeexactcoordinateaudit:35228rawnativecandidates but7343uniquegeometry,27885aliasobservations(79.1558%),max188samebox; actualduplication isactionableroot. Preregister separateproposalpolicycontrol: exactxyxyequality, preserveall35228aliasindices/labels/scores andall520queries/40frames, representativefirstnativeboxwithidenticalROIsasserted; originalrawsource/resultsunchanged. SameSimple/CompactCPUweights/officialTETA, noIoUthreshold/topK/NMS/GTselection andnoassociation-only/JEVcausalclaim. Newrawsourcecap1GiB alongside2GiBRMO sources; cumulativeconservative21397407162bytes<25GiB. NoextraGPU/training/perception.

Researchplan tocompleteexplicit14.5task-sharingmatrix: foursemanticCompactarms MOT/MOT+RMO/MOT+OV/allthree, sameMOT-onlynative300initialstem andseed0newsemanticheads (notjointpretrained), sameexistingbroadnative536pack, same1200CPUFP32updates/loss/optimizer/params. Optionaltrainerdevice/tasks/per-task-stream flags leaveolddefaults unchanged. Independenttaskseed0+index samplers makecommonselectedexample-IDprefixes identical; exposure1200MOTvs600pairs/400triple andtokensdiffer, sofixedbudget comparisonnotisolatednegativeinterference. Existingolderactorhistoriesexplicitlyoffpolicy; no newfeaturecache/GPU/GTvaltraining/NovelGT. PlannedofficialcompleteDT1203/rawTAO40+shortRMO8/12, thencomplete800RMOafterexistingfreshsourcesready; preserveoriginalrawcandidateprotocol, notaliasconfound. CPUfullstate/Adam/RNG/sampler100step snapshots forunfinished-unitresume.

Actualsubsetpackpreflight536groups (MOT149/RMO325/OV62),142actualnativeRMOgroups,1203classes. Fourarms readoneexactsemanticinitialization checkpoint constructedfromMOT300+seed0heads, eliminatingpriorjointtraining. Separateper-task drawprefixes verifiedfromactualsampleIDs aftertraining. Evaluateall1203DT/all40rawTAO +RMO8/12 beforeblockingforfresh800sources; latterwaitscompleteactualsource, notcompletionofslowerGPUFullcomparators. No sourcefeaturefilescopied; onlysmallmodel/Adam/RNGcheckpoints added.

Tasksubsetallfouractual1200CPUtrainingscompleted/1146511params each/three-taskcheckpointreload0; actualsampleprefixesMOT400/RMO400/OV400exact, same sharedMOT300+seed0semantic initialization. FirstMOTonlyfull1203officialHOTA.1963864227 IDF1.2078625235 IDSW348 FP1089 FN1765; othercohortarms stillpending, notcomparedtolegacy2400budget26.316. Configurationerror discoveredbeforeunfinishedshortRMO scoring: v1pointedoldFullrawlogswithoutnativeROI. Existingjoint1200beforelogs haveexactsame8/12frames11/16boxes/labels/scores andreal2560ROIs; targetedall-rowmappingpasses. Preservev1badplan/oldmanagerstate, SIGTERMonlyverifiedCPUmanager1915707, leaveactualMOTinference1925936+ffmpeg1926238alivewithsameoutputs; newv2manager waitsoldgroupfinish andusesallfourcompletedcheckpoints/completedunits, no trainingorvalidinference restart. Sourcecontractproof saved.

ExactaliasTAO40bothcontrolscomplete: SimpleTETA22.556/LocA22.871/AssocA26.472/ClsA18.325; sharedCompact21.773/23.273/9.6266/32.419 vsraw14.517/8.088/5.1478/30.314. Compactcorrectbirthanchoroptioncoverage.156716->.530973, existingaccuracy.052239->.088496; stillunderSimpleTETA22.556/AssocA26.472. BothNovelClsA0; no promotion. Rawvs7343exactgroups physicalproposalIoU.5 one-to-one recallboth147/160=.91875 (GTposthocaftergrouping), allboxsetsequal; source402723754bytes withall35228aliases. Separateproposal/representativelabelpolicy, notpureJEVassociationclaim. Nextscientificgaps retain: conventionalmatchedvisual/semanticinformation baseline, Compact32optionretrieval/contextscaling underactualmissing-option failure, samevideo/hardware genuineFull-vs-CompactE2E, Groundvisual-only/jointtracking-aware factorial andnativefull9retention, furtherown-historymultiTaskresearch. Do notdeclarefullgoalcomplete.

SharedFullteacher600trainingterminalverified byexistingGPU4controlowner:600retainedupdates/2400examples, latest9,029,141,401byteFP32AdamRNGstate. Actualrestored450mastersdifference0; newtail150updates/600examples took3595.081s. Nativeforkdonecounterdividescumulative2400bytailtime(.66758) so notvalidtail/whole-stageFPS; curateactualtail.166895examples/s, nofull-stagewallclaim. Original475logloss1.404405526 vsreplayed4751.402670712 delta.001734814,slotaccsame.78/gradnormdelta.226344; restorestateexact doesnotprovebitwiseCUDAtrajectory. Causeunestablished; source deterministicrowseededrender/sampler andkernels/tolerances unchanged, no masking/reset/retraining. Original25unretainedloggedupdates replayed,totalobservedwork>=625updates; rawlogs preserved. Newcompletionvalidator recordslimitations. Native600arm andthree-taskofficial600scores pending. Same-policybefore500RMO8/12bothHOTA0 duetoexplicitnativequerygate, notcomparabletolegacyungated24.886/21.425.

E2Eresearchplan addressesactualmissingcomparison: run Fullnative600 andcurrent sharedCompact onexact samelegalRMO0007single800frames/GPU5, freshisolatedoriginalDLMserviceforeach andsingleclientrequestcounterreset (priorisolated96repeatvalidated). FullactualV3crops vsCompactactualnativeROI path, notcached/syntheticlatencies; timeallIO/nativeDLM/ROI/memory/sourcearchive/render/encode excludespreloadedmodel/serviceinitializationboth. Strictactualdetectorbox/label/score/parserstatuspairingguard, every800frame included; ifdifferent preservefailure/nopairclaim. FixedFullthenCompactorder/singlerepetition/concurrentotherwork limitations explicit. Wait completeoldfullRMOowner exit+ownedservicegroups empty, no overlapGPU5/newtraining. Newsourcecap1GiB, cumulativeconservative22471148986<25GiB; sampledownedtotalVRAM/coreallocatorpeak/processRSS andfrozenOneCPUtable footprint included.

TasksubsetDTfull1203fourarms nowcomplete HOTA/IDF1/IDSW/FN: MOT19.6386/20.7863/348/1765; M+R6.7597/9.8843/80/3865; M+OV21.2285/20.0809/414/1006; triple19.3119/17.8267/422/1372. Mixedoutcome, M+Rmisscollapse butMOTexposure600vs1200confoundsinterference. PreregistercomplementaryMOT400/600CPUcontrols sameactualsharedinitialization/source/per-taskrng/constantoptimizer/loss, assertactualsampleIDs+loggedloss exactprefixofMOT1200; thenfull1203fixednativeTrackEval. MatchMOTexposureofpairs/triple whiletotalupdatesdifferent, keeporiginalfixedtotal-budgetmatrixunchanged. No newGPU/queries/featurefiles; reportallfixedbudgets notselectbest.

PairedE2E canonicalcontract onactual4nativeframes/13ROIobjects passes acrossfeatures+geometrylogs; resourceprobeactualCompactCPU4frames records0GPUalloc andcorrect1146511core plus248320x1024BF16frozenCPUtable508559360nominalbytes (notresidentRSS). Newruntime peakRSS/UTC/gitcommit fields tested; no freshnativeSpeed claim fromreplayprobe. Queuedpairedowner1953808/session87824 waits priorGPU5fullRMOcomplete, ports8106/8114, onefreshserviceperarm. Optionaladditionalmetadata enrichments afterownerlaunch do notchangeinference/pairingprotocol; rawchildresultreports areauthoritativeforcomponentmemory. Legal0007v2validation800manifest checkedagainwithoutopeninglabels/images.

2026-10-15：四组完整MOT评测均完成，固定总1200步下 HOTA 为19.639/6.760/21.228/19.312。M+RMOT漏检3865，不能把IDSW减少当成改善；M+OV提升HOTA但IDF1/IDSW变差。因MOT实际更新量1200/600/600/400不同，补充MOT400/600同前缀训练量对照，保持原矩阵不变。已启动单一阻塞CPU owner1957205/session71241；真实端到端同视频/同GPU对照 owner1953808/session87824 仅排队等待GPU5，不与现有任务争用。新增RSS/UTC/git记录已用真实4帧Compact重放校验；未将该重放当成E2E。

研究计划：按任务书14.2，在同一已训练共享Compact、同一候选/三份native外观/几何/文本语义输入下，比较几何固定关联、几何+三记忆cos固定关联、固定关联+学习辅助头、学习关联+固定辅助头及原学习参考。所有语义分类/成员判断函数权重相同；状态由各策略自身逐帧生成，不做GT回填。固定分数明确非概率，常数沿用已有.5/.5与NEW=.3；这是一组冻结策略干预，不冒充重新训练最优传统模型或JEV独有证明。计划完整DT1203、原始/精确合并TAO40、RMO8/12，CPU、不新增feature源。已观察完成MOT耗时408.52s中350.69s、原始TAO1152.83s中773.39s在已测关联块之外，不能全部归因于状态导出。新CPU对照可选逐帧决策trace+最终无损二进制状态，保留所有状态字段，先验证序列化/下一帧一致；原默认JSON/每帧导出和已注册E2E保持不变，不比较这组存储不同的速度。

MOT400/600补充对照已验证完整1203帧/13664候选/27拒绝帧：HOTA13.3173/13.9788，IDF1 17.7572/13.4107，IDSW174/312，FN3483/2566。实际样本ID及已记录loss与MOT1200前缀完全一致(delta0)，各1146511参数/三任务reload0。同MOT更新量下，M+R600为6.7597、M+OV600为21.2285、三任务400为19.3119；总更新量仍不同，不能直接归因纯任务梯度干扰。无模型提升/全数据集/E2E声明。

固定关联及存储回归完成：真实4帧全预测/状态、完整序列化字段和恢复后真实第5帧一致；三个任务在同状态下语义/保留辅助输出一致，固定外观分数与现有FeatureDecision差1.11e-16。新注册五种冻结策略×完整DT1203/短RMO8与12/TAO原始及alias40的CPU序列对照；使用同一checkpoint/原始native2560三记忆/同一文本语义头，逐帧自身历史，无GT修正。所有新组都用binary-final存储且不报告跨存储速度。run_video最终完整状态导出移入pipeline计时，修正旧版重复最终导出未计时的问题；既有速度原样保留，未来配对双方均按修正口径。

冻结关联对照owner1986817/session48561已启动，单一阻塞shell等待所有25个单位；所有CPU比较均新跑同设备参考。逐观察posthoc诊断各约5MB保留本地，Git发布精简统计/官方全指标/实际运行结果和代码，不复制大逐观察列表；完整逐帧trace、视频、状态仍在原输出目录。

研究计划：针对Compact32身份短名单遗漏，先在原始/alias TAO40已完成轨迹上真实因果重放，核对全部原始输出/状态和32-ID排序一致，再做同一轨迹的32/64/128排序覆盖诊断；GT只读取已完成posthoc birth-anchor归属，绝不输入tracker修正历史。覆盖上限不冒充更大context实际精度。后续固定同checkpoint/16候选组，预注册32/64/128实际完整DT与aliasTAO对照；上下文增大超出原训练32选项分布，会改变softmax/语义/状态，必须如实报告。

已验证aliasTAO40因果重放全部7343候选/原始输出/状态，113个已有正确birth-anchor事件下32/64/128覆盖53.097/60.177/74.336%，最大420活跃身份、正确最小rank最大395。说明128也非完整覆盖，不能用覆盖率当精度；原始TAO排名审计owner2000016/session86629仍按原单一阻塞命令运行。真实2帧高密度检查：第2帧前251live IDs，64/128各候选选项数量正确，默认32与显式32全预测/状态一致。后续完整DT1203与aliasTAO40、学习/固定cos两策略×64/128共8新单位，CPU owner2003663/session5404等待当前affinity全部完成并退出，复用并重新核验32参考，不新增feature/GPU。

Full teacher600四个同策略短测均完成：MOT HOTA30.868->30.601，单RMO0->0，多RMO0->9.945，BaseTAO TETA11.981->52.181/AssocA18.399->28.308/ClsA0->100。四个before与四个after都保留原候选/帧并完整decode，RMO双方native Noul.5一致；TAO都输入1203类但4帧只有Base person/baby，不作Novel/广泛分类结论。实际native600训练/后续控制仍待完成，教师600恢复非bitwise限制未更改，无整体成功/模型提升/E2E声明。

原paired600阻塞训练句柄68568已exit0，teacher/native均600更新/2400例，两个最新FP32/Adam/scheduler/RNG文件均9029141401bytes。两个600真实nativeMM重复加载均delta0/474entries；native从250完整state精确恢复，后350更新1400例耗8767.184s，尾段.159686例/s，不能使用累计2400除尾段耗时的.273748当全程速度。native无250之后旧日志重叠，无法证明bitwise重放；teacher475已有差异仍保留。GPU4原owner继续native600评测；GPU5原fullRMOTowner已完成训练交接，实际状态single800_FRESH_COMPACT_NATIVE800，禁止重复启动。后续pairedE2E仍等待其全部结束。

补齐关键完整MOT证据：此前Full微调前后只有短MOT4，不能据此替代完整视频。新注册原V1 DT1203固定候选上Full native600/before500/teacher600三组、固定顺序，各用原生Choice/Noul/Score+V3真实crop/motion、动态分组不增token上限、全部27拒绝帧保留。GPU4新owner只在原短控制owner1870814成功并退出后执行，不与其重叠；GPU5完整RMOT继续。复用已验证600真实MM加载结果，三组统一binary-final存储/共享输入feature只读/逐帧决策trace，无新features或E2E速度声明。通用控制runner流式核对源候选，避免整源重复常驻内存；默认短测行为不变。

原始TAO40排名审计session86629已exit0：全35228候选/原轨迹逐帧输出和状态精确恢复，最大2670live IDs，134已有正确anchor事件，32/64/128覆盖15.672/31.343/44.776%。不把覆盖反事实当精度，也不排除已终止/从未正确初始化身份这一诊断局限。Full DT1203新owner2024544/session88330已验证等待原GPU4短测owner1870814/start41851012，未提前占卡。

研究计划：原始TAO32选项覆盖15.67%，扩大128也仅44.78%（冻结轨迹诊断），已有正确anchor常被纯几何排序挤出。新增独立候选协议：保持32总选项、16候选分组/同权重/同输入，先取16几何，再按实际anchor/recent/long原生2560特征平均cos（对组内候选取最大）补足至32，稳定以lost_age和track_id解同分；不读GT，不更改身份状态或训练。旧geometry默认保持。完整DT1203与raw/aliasTAO40，学习关联及固定cos两策略，四个nativeROI输入仍共享，不新建featurefiles；CPU新owner等既有contextowner完成后串行跑。与原geometry32分别报告，非纯JEV唯一收益或最优检索声明。先检查真实251-ID片段的候选合法性与完整前缀可执行性。

完成并核验任务组合16现有范围：DT1203 HOTA19.639/6.760/21.229/19.312；rawTAO40 TETA12.421/19.284/12.850/10.081；单RMO8 HOTA0/0/26.494/37.265；多RMO12 4.096/3.513/9.622/3.513。所有实际结果/官方指标/候选完整性一致、空失败表，full800仍待源。无普遍最优，三任务单query受益但多query IDF1=0/FN30且TAO退化；四组NovelClsA0。共享总1200步但每任务暴露不同，保留MOT400/600对照，不宣称纯梯度干扰。

5个同输入完整MOT策略完成：learned/learned26.316，IoU/fixed7.488，cos/fixed13.311，cos/learnedaux13.009，learned/fixedaux24.283。学习辅助只在学习关联组合有增益；learned FP579/FN1962 vsfixedcos9328/250，不能将总体收益全归因身份排序。各轨迹内生不同，不是唯一JEV证明。Hybrid32真实251身份片段检查通过：几何默认与旧版全输出状态精确一致，每个新候选32个独特live IDs、前16几何一致，230候选的身份集合实际变化。新CPU owner2034367/session35730只等待已有contextowner2003663完成+退出，再跑3视频范围×2关联策略；保留原默认、不改训练、GT阈值和指标。

研究计划（补齐任务书14.4必需控制）：原Ground200含native MDM/AR+.1GT身份+.1decoded-proposal身份，缺少只视觉定位对照。新localization200从原569张量初始化，同72图/112native行/56pair、seed42/200步/AdamWlr2e-6、相同采样/参数27M，只将两个身份loss系数置0，仍记录非加权分量，不改变native算子/掩码/GT。原track200有中断恢复，声明相同注册配方/采样，不声称bitwise轨迹配对。先CPU比较全部样本payload与schedule，GPU6 owner仅等已有adapted200 full9客户全部结束、该服务组已被原owner释放后才训练，不要求GPU7也结束。随后导出新bundle，并用同GPU6/私有新service/requestcounter同序列比较原始/只定位/跟踪视觉适配；同各自原生候选再比较原始One与native600，组成受控视觉×JEV组合，不称端到端联合梯度训练。无新featurecache，保留全部原测评。

localization CPU预检v1失败：比较最终Ground200 checkpoint与最初GPU7登记的整个protocol，实际差异仅GPU6、step1保存和execution_variant。全部200采样索引已匹配，loss/data/seed/lr未改变。改以实际完成的native_ground_tracking200_gpu6_20261014.json为参考，仍要求实际protocol精确相同及全部112行/200索引相同，fresh v2重跑；保留v1诊断，不放宽数值标准。

发现并修复Full1203排队driver句柄变量遮蔽：upstream registered list/None覆盖teacher/native dict。原等待owner2024544在短测owner1870814正常exit0交接时已实际报TypeError退出（88330 exit1），completed={}，未启动GPU模型；试图替换前的状态断言阻止了任何误杀。修复为独立upstream_registered。真实已完成owner+两份实际600记录的CPU调度回归通过，不启动推理、不冒充评测。原failure保留，fresh v2同1203协议恢复，无已有训练/有效推理重跑。

Full三分支12同策略短测已终端验证：before500/teacher600/native600 MOT HOTA30.868/30.601/32.236、单RMO全0、多RMO0/9.945/11.214、Base4 TETA11.981/52.181/47.682。native多RMO IDF10.256/IDSW1/FP6/FN27，Base4ClsA100只涵盖person/baby单一Basecanonical类，不泛化Novel。native历史效果混合、不晋升模型。Full1203 freshv2 owner2049900/session70182实际进入native_mot_full1203_INFERENCE，原v1错误/无模型执行事实保留。

localization200实际CPU预检v2通过112行图像bytes/prompt/targets/nativealign及与完成checkpoint全部200draw完全相同，72图/56pairs、884nativeproposals/628alignedregions/24支持身份pairs；只定位两个身份coeff0，原图像标签和native核不变。新owner2052936/session76692已等待GPU6 native200 full9真实clients+服务组释放，不中止其原测评；随后200步训练+导出+原始/只定位/跟踪视觉三前端及localization repeat共96帧，再与pretrainedOne/native600组合18短测，明确非联合梯度训练，不新增featurefiles。

真实完整RMO初次完成1600帧：single800 Compact HOTA6.0247/IDF2.8902/FP323/FN13，samecandidateSimple4.9658/1.2587/FP1402/FN8；multi800Compact19.8019/21.0082/IDSW165/FP619/FN1054。fresh E2E .678005/.882942含真实DLM+ROI+state/archive/video，不能与cachedSimple或旧不同视频Full测速算加速比。single仅18GT/782空目标帧，不能只报宏指标；官方empty_target_frame_predictions是检测条数，新增posthoc另算空帧告警率。

训练监督审计假设：现prepare_rmot_native72 update_target仅quality>=.7且目标不是NO_MATCH，没有检查实际Hungarian所选身份是否与该候选一致。完整RMO后验中大量memory_updates_wrong_identity_or_unmatched不等同普遍错误真值，但提示此代理目标可能教会错误写入。先只在原72合法fit帧审计，按原GT对齐和不可修改birth-anchor归属复算原142native组标签；不改原包/模型/评测，候选无实际写入时不臆造条件标签。之后自身历史训练应使用当前权重逐帧状态并以真实所选身份监督更新，保留待验证假设。

训练更新标签审计结论修正：原72fit/165候选、128非空primary/counter组的原proxy更新标签全部精确复现；28个正标签中16个已知一致、12个出生GT归属未知，0个正标签同时存在两个已知冲突身份。不能把未知锚点判成错误；v1把未知proposed_target置0的诊断建议无效、从未用于训练，v2/v3改为None/mask，保留解释限制。已知跨两个标注身份写入仅1例，原标签已是0，说明不能据此宣称正标签导致全视频失败。优先补当前权重逐帧actor状态训练/硬错误覆盖，不改未知样本为负、不改原包/权重/评测。

完整multi800的Simple基线在原GPU5队列中排在慢Full对照之后；为及时完成必要固定候选比较，独立CPU目录按原800/2208候选/19拒绝帧/Simple/JSON每帧原参数执行，不中断原GPU5 owner或已有推理、不新建features。原owner若日后重复生成其计划基线，保留两份并明确是同输入重复、不是新seed或额外独立样本，不做FPS加速比。

任务范围澄清：任务书16要求teacher-forcing后增加自身预测历史且逐帧状态真实影响未来；现有One100实际rollout→500SFT，以及Compactjoint1200实际72帧rollout→native1200，已提供离线迭代的自身状态训练证据，不能称为每个梯度更新都重新采样的on-policy训练。进一步逐梯度当前权重rollout是改进方向，不擅自把它增设为用户原任务的额外完成门槛；保留离线历史、模型版本和无GT修正限制的明确表述。

完整800输出核验通过：native档案与全部1600输出geometry/labels/scores/order/ROI2560/时钟一致，1412/2208候选、36/19解析拒绝帧、197218901总档案bytes<2GiB。真实完整wall1179.932/906.062s、FPS.678005/.882942，init5.232/5.362s及服务预加载除外，含最终状态存盘。single空帧200/782告警=25.575%（318检测），multi143/247=57.895%（185检测）；别混淆检测数量与帧告警率。核心GPU allocator14970368/14855168bytes、RSS1.467/1.478GB不含外部DLM/ROI或整pipelineVRAM；不与cachedSimple或其他视频Full算速比。新CPU multiSimple owner2079614/session84279不改变老GPU5队列。

multi800早期Simple CPU控制exit0并验证全部800帧/2208候选/19拒绝：HOTA31.8769/IDF37.6852/IDSW99/FP1076/FN980，vs共享Compact19.8019/21.0082/165/619/1054。身份关联退化不能被较少FP掩盖，不晋升。为回答语言筛选与学习关联的贡献，将现有5个冻结策略原样扩展到两段full800，另跑CPU学习参考、同语义权重/输入/32选项，全部binary-final存储、不测E2E，不读GT改状态、不新建features、不增加GPU。独立CPU4线程并行，不修改原GPU5Full队列或原25个范围协议。

文档整理：CURRENT_STATUS改为当前证据快照，集中完整MOT/RMO/TAO表、真实FPS/资源口径、训练/监督限制和GPU/CPU待完成清单；原逐次追加全文原样保存STATUS_HISTORY_20261015.md及Git历史，未删除失败/旧结果。

fullRMO现有5策略×2完整800已全部exit0/验证，CPUlearned参考的全部官方accuracy与freshGPU5精确一致。singleHOTA6.025/7.766/5.538/6.009/6.636；multiHOTA19.802/32.813/16.622/22.035/19.558，IDF21.008/41.432/14.009/23.803/21.064，IDSW165/79/105/119/157（注册顺序）。固定IoU+同冻结语义函数优于当前学习关联，nativeappearancecos反而恶化纯IoU。此对照否定本场景学习关联优势，不以MOT正结果掩盖；非最优重训baseline/唯一JEV证明，保留语义状态内生差异与actionprior差异。全部800/36或19拒绝/相同1412或2208候选，binary-final同存储、不测E2E，不改模型默认。

研究计划：补齐已登记自然MOT20-05完整validation半段1657帧（1659..3315,25Hz），比较原Simple/共享Compact的3s/30s及已有memory干预；固定同一真实原DLM候选/ROI，无人工遮挡/丢框/GT修正/可见度过滤。GT身份854的29.76s只是低可见度间隔且743中间帧仍有标注，不预称真实候选缺失或成功重识别。GPU7队列只在原nativefull9客户结束且原owner释放服务后启动，GPU6适配队列互不干扰。新archive最多3GiB，既有保守资源上界22471148986+3221225472=25692374458<26843545600（25GiB）。为避免denseROI JSON膨胀，保存原始FP32ROI为逐帧无损npz、JSON索引留全部框/标签/分数/metadata，读回float数值必须精确，非FP16/量化，不新增独立候选。默认老JSON格式不变，先用真实帧验证序列及下一帧恢复一致；所有新存储/IO均计入真实pipeline时钟，重放不声明E2E。

2026-10-15：完成后统一整理25冻结策略、24任务组合和12context单位，逐项将owner指标/解码计数与实际official_metrics/result对齐，核对完整帧、无GT/未来读取及最终状态存在。整理脚本初次遇到旧JSON result无final_identity_memory_path元数据；实际final_identity_memory.json存在，按旧格式真实路径读取修复，未重跑或改指标。三组摘要均通过。TAOalias学习32/64/128 TETA21.773/23.100/24.749、固定cos23.913/26.099/29.062；DT学习均26.316，固定cos13.311/13.289/12.491。完整single800任务三组0、三任务7.860；multi800为7.673/18.255/15.794/18.674，均低于Simple31.877。保留负结果，不调验证集阈值。

真实21帧packed原生FP32存储/空拒绝帧/下一帧恢复精确通过，15.0MB到3.4MB；两臂实际scheduler及实际single800自然gap后验功能检查通过。自然MOT20单owner2126039等待GPU7原native retention释放，未提前占卡；登记低可见度29.76s不等于真实候选缺失。缓存上界含3GiB新源及32MiB回归余量25725928890<25GiB；原默认JSON不变。

Full single800 pretrained/500/teacher600/native600实际完成HOTA16.119/0/6.847/2.506%，FN16/18/14/17（GT18），训练不改善该scope；pretrained只有2TP，不能只看较高HOTA宣称成功。Full multi800/DT1203、原生retention、localization和pairedE2E仍按原owner继续。

研究计划（任务14.2输入证据控制）：冻结策略干预不等价输入消融。新增geometry_motion（屏蔽rawROI投影与spec条件）、geometry_appearance（保留rawROI，屏蔽spec条件）、geometry_appearance_language（原完整输入）三组同Set结构/1146511resident参数。所有组几何/运动保持，类别词汇只在输出分类头保留，原DLM感知自带语义不消除，不能称全系统无语言。复用已完成all_three1200完整组，仅新增两组相同MOT300+seed0语义初始化/1200步/逐项相同样本ID/原GT与native softteacher/同536组既有actor历史；teacher视觉知识本身保持，非从零纯单模态监督。共享原生候选，完整DT1203/rawTAO40/单多RMO800各3组，binary-final同存储，无GT回填/新增feature/GPU/E2E速度声明。Full真实image原生JEV已有独立完整控制继续，但架构/预算不同，不能将其与Compact差异解释为JEV唯一收益。

输入数值回归通过：三个任务真实pack的同初始权重默认全输出与原e9111b5精确一致；屏蔽spec后更换实际spec不影响任何输出；屏蔽外观后置换真实外观不影响输出、视觉分支无梯度；仍用外观组match真实改变。两模式各2步实际训练保存/三任务reload差0，随后真实21帧scheduler/packed原生源/266候选/真实拒绝空帧和官方scorer均通过，仅功能回归不当精度结果。正式owner等hybrid CPU owner退出再串行，不挤占现有Full/native/GPU队列。

Full原生性能诊断用已完成四组single800实际日志，不新增模型调用：全部3200请求fork_mode=batched。pretrained/500/teacher600/native600平均prefix1347/3987/8235/5474tokens、branches7.93/14.20/24.38/17.72，native forward总142.531/344.670/787.309/481.178s，render28.226/89.389/247.319/135.818s。较长内生历史/更多分支与更高耗时并存，不归因未控制的并发差异；日志没有全程sequentialfallback证据。原引擎已有prefix共享与末token logits，不盲改原fork阈值/算子/上下文或删候选。

Input-evidence正式单一阻塞owner2137548/session36336已启动，实际等待hybrid owner2034367/start42762109；日志/tmp/groundjev_input_evidence1200_v1.log。未启动额外GPU/model服务、不重复既有Full/native实验；预计正式CPU工作仅在原hybrid成功退出后开始。

2026-10-15：Full预训练multi800已完成并验证800解码/2208相同候选/19拒绝帧/官方指标：HOTA8.6210、IDF7.6347、IDSW215、FP214、FN1766（2112GT）。低于同源Compact19.8019/21.0082和Simple31.8769/37.6852，不能将低FP当成功。按任务18要求只分析实际已完成失败，不增加实验范围：直接交叉原GT后验对齐与真实保存的每候选action/member/output，未重跑模型/修改阈值/预测或状态。1132个对齐query-positive candidates，Full=105无物理assignment+687有assignment但member拒绝+340实际输出；Compact=4+71+1057。全部2208候选的输出ID/box逐帧能由真实action与原member>=.5精确重建，head混淆计数与原association_events完全相同。Full成员召回31.625% vsCompact93.728%，已有正确birth-anchor选项覆盖99.149%但对应关联准确率10.207%，说明本scope语义拒绝与决策排序均有问题，不优先盲增短名单。两模型内生已有身份事件集合不同，不作严格相同事件因果对照；上述funnel不是HOTA各阈值下官方FN的可加分解、未对未知candidate贴负标签。先等待已注册teacher/native600完整对照，再决定训练改进，不新增训练量或调验证阈值。

2026-10-15：原hybrid阻塞句柄35730返回exit0，六新轨迹+六已验证geometry参考共12单位全部官方artifact核验通过。原始TAO学习TETA14.517→15.178、fixedcos13.964→15.652；alias学习21.773→21.466、fixedcos23.913→24.435；DT学习26.316不变、fixedcos13.311→12.811。混合检索非普遍收益，不更改默认geometry，不宣称Novel/唯一JEV或速度提升。后续输入对照owner2137548原依赖成功解除，沿既有CPU队列自动衔接，不重启。

Full MOT500 multi800完整退化核验：HOTA0/IDF0/FP11/FN2112，原2208候选/19拒绝/800decoded全部保留。加入同源后验funnel：1132query-positive候选全被member<.5拒绝，其中263无物理assignment、869已有assignment但被过滤。原始真实输出逐帧精确重构及混淆计数一致，不修改阈值/状态；不要错误归因869候选均未获物理身份。仅MOT训练对RMOT不迁移，等原注册共享teacher/native600结果，不加验证阈值搜索或额外训练。

输入对照队列已实际自动衔接：两新模式均完成1200步，每个实际sample ID与完整输入all_three1200相同，初始tensor由owner精确核验；三任务reload全0、GPU allocator0。原owner已进入ALL_THREE_INPUT_EVIDENCE_FULL_VIDEO_CONTROLS，无重启。报告澄清同1146511为resident参数；geometry_motion屏蔽visual/history_merge输出，382208参数无梯度，不宣称相同有效容量或唯一JEV证明。输出类别prototype文本/原DLM语义/native软教师知识均保留，完整视频结果仍待完成。

研究计划：已完成rawTAO真实final state/同一实际图像前16候选的CPU profile（2365live IDs）定位geometry shortlist耗时，sorted累计.2116s、38352overlap累计.1311s、40432predicted_box累计.0518s，NN module累计.0095s；hybrid native array转换另.5039s、初次1203类别文本冷缓存.2639s，不能把单次cold profile当整视频FPS。只优化Compact几何IoU排序为FP64批量计算并每次decide缓存同timestamp预测框；候选/轨迹/几何公式/tie order/分组/权重/metadata不变。先核对真实高密度全部pair IoU和各16候选组shortlist精确一致，再比较真实dense state的完整模型输出；不改运行中任务、不新训练或重跑官方评测。

几何批量优化真实回归通过：591实际候选×2365live tracks=1397715个FP64IoU逐值与原scalar完全相同；37实际16候选组的32/64/128顺序及空组/空history相同。普通geometry和hybrid16_16各自dense32候选/空观测/无history的全部决策概率、生命周期及metadata与ba94b94原实现精确一致。完整scalar IoU矩阵2.1921s→.03212s；16候选暖缓存3次中位geometry .14708→.05111s、hybrid .78482→.67581s（native数组转换仍占用），不是整视频或E2E速度结论。保留优化，原权重/候选/tie order/状态算法不变；已有进程继续旧import，之后新进程按actualgit记录新实现，历史FPS不改写。不重复已有全部官方测评。

2026-10-15：input owner2137548/session36336及FullRMOT owner1896249/session16294均实际exit0。输入三模式×4完整scope共12正式控制全部官方/帧数/最终状态核验通过，完整输入四scope的官方metrics与原all_three1200精确一致。geometry-motion/appearance/appearance+spec：DT HOTA9.144/26.061/19.312，rawTAO TETA13.568/11.444/10.081，single800 HOTA0/2.491/7.860，multi80014.462/12.900/18.674。外观改善MOT，spec改善RMOT但损害MOT/TAO，NovelClsA均0，multi仍低于Simple31.877；不作普遍多模态或JEV唯一收益结论。保留三个已训练模型和全部负结果，不按验证阈值调参。

完整FullRMOT全部12arm正式artifact核验通过：pretrained/MOT500/teacher600追加/native600追加，multi HOTA8.621/0/4.578/6.740，IDF7.635/0/4.228/6.693，FN1766/2112/1963/1859。teacher/native部分救回MOT500崩塌但不及pretrained；不将短12帧改善外推到完整800。新curator初次错误假设result包含native_rmot_membership布尔字段；实际语义存在逐帧decision_metadata及原posthoc报告，改读真实原policy和head标志/原.5阈值，不改原输出/标准。候选funnel扩到全部四Full加Compact：teacher544无assignment+445assigned/member拒绝+143输出；native330+555+247。原owner已释放服务，原配对E2E owner1953808实际进入full_native600_ACTUAL_FRESH800，未重复启动。

执行计划：两个后续owner经实际PID/starttick/空children/空completed核验仍仅CPU等待（2052936、2126039），GPU6/7的原生保留评测与它们科学上独立。将未开始的localization200和自然MOT20改为GPU5配对E2E成功结束并实际释放全部owned groups后串行执行，仍只使用物理4/5/6/7。克隆新计划，明确训练参考GPU6与执行GPU5（型号均NVIDIA A800-SXM4-80GB），保留112行/200采样/seed/loss/算子、96原生帧/18组合、1657自然帧/8控制和25GiB缓存界。先CPU核对新旧样本/训练计划与仅硬件元数据差异，再对两个尚未启动实验的等待owner做记录化交接；不停止任何正在训练、推理或原生评测的进程，不改指标/门槛，不声称跨GPU bitwise轨迹。

GPU5后续调度预检实际通过：新旧定位计划除明确硬件/说明元数据外所有字段相同；再次与实际完成track200 checkpoint核验112真实行bytes/prompt/answers/nativealign、全部200pairdraw、seed42/AdamW/lr/原native损失。GPU5/6/7均NVIDIA A800-SXM4-80GB；96前端帧/18组合/1657自然帧/8memory控制和25GiB预算均原样保留。真实已完成FullRMOT GPU5 owner/group release检查通过，无新增GPU模型或feature文件。

两个旧等待owner2052936/start42864217、2126039/start43333839经冻结PID身份、空children、空completed和无currentchild再次确认后仅SIGTERM这两个CPU等待进程，manager_handoff.json及旧日志保留，state明确UNSTARTED_CPU_WAITER_SUPERSEDED_WITHOUT_GPU_JOB。旧loc shell76692返回241（有意SIGTERM的parent退出码），不是实验失败或完成；所有native retention/Full/E2E模型进程未动。新单一阻塞owner2207841/start43963616/session36847实际WAITING_ORIGINAL_PAIRED_E2E_GPU5_RELEASE，精确登记1953808/start42350971；成功结束并验证ownedgroups全退出后串行localization200→自然MOT20，参考训练GPU6与执行5差异明确、不声称bitwise跨卡轨迹。此改动只缩短独立任务的排队依赖，不减少原生保留评测或改变指标/种子。

2026-10-15：原配对E2E阻塞87824实际exit0。fresh Fullnative600与Compact各完整800/1412完全一致native候选/36解析拒绝，canonical所有frame/image/box/label/native_score/score_kind精确一致，两个官方accuracy与旧同候选reference精确一致。实际pipeline FPS.4959138201/.6870989019、wall1613.1835/1164.3156s，观察速度比1.3855208、wall减少27.825%。五秒采样ownedGPU峰37052/19626MiB，lowerbound非瞬时精确max，减少47.031%。Full/Compact HOTA2.506/6.025、FN17/13（GT18），空帧告警67/782 vs200/782，较少漏检以更多误报为代价；两者非实时且召回差，不能推广多目标或全数据集。服务startup26.052/26.046和coreinit8.010/5.004s单列除外，实际pipeline包含image/native/ROIwhereused/association/state/档案/绘图/编码等候退出/最终存盘。固定Full先Compact后、单seed、并发背景负载不控制；不称统计或反平衡速度收益。

已逐项核验真实run/official/owner/严格候选stream、完整800frame/video/finalstate和原GT口径；per-frame inference延迟Full p50/p951.810/2.930s、Compact1.202/2.136s不含后续逐帧video/state写入，fullwall包含，明确分开。Full core allocator8.426GB/RSS3.305GB，Compact.01497GB/1.466GB不含外部服务，不能用1.146M核心冒充全部模型成本。新增curator和rawresults/official/resources保存，不重跑推理或评测。

GPU5新串行队列已真正接棒：原owner1953808结束且五组2173897/2174768/2197749/2197750/2198799全部退出，release receipt真实free81151MiB；owner2207841进入LOCALIZATION200、child2213633/start44011696。实际trainer state physical5/seed42/native_dense_referring_localization_only/target200/identity coeff0，训练正在进行，未宣称200完成或新指标。旧保留6/7及Full4未动，仍四卡；下一阶段96native/18factorial之后原自然1657/8controls。

Pilot文档核对：原FOUR_GPU_PILOT仍写2026-10-14未完成，现用真实stop/cache receipts、originalOneFull1203run、实际500updates/2000examples/state/checkpoint、RMOT/TAO正式结果、paired800实际FPS/资源及已验证GitHub树逐项核对用户§17十二项执行交付。新增evidence_audit，限定注册开发范围，不等于模型晋升/Novel成功/全数据集或完整goal完成。资源口径澄清：持久化before/after全设备表可重算净减少14042MiB，非全归因U2；17584MiB为历史compute-app观测，不能从这两个device表独立复算，不伪造更精确归因。清理4748分片/42831618048allocatedbytes及1.909GB保留checkpoint实际存在。同步将ARCHITECTURE中的已过时“nativefeature/Compact待实现”改为实际实现及范围限制。剩余Full1203/native9/localization/natural继续原队列，无新增模型运行或评测。

执行改进计划：现有阻塞wrapper将研究owner的stdout写文件，所以工具句柄只在最终退出时返回、真实阶段变化未向终端转发。增加一个只读Linux inotify+pidfd事件转发器：单一阻塞命令监听已注册owner状态及定位训练/前端/自然子阶段，状态改变/进程退出时输出真实摘要，不做周期ps/tail/wc/nvidia-smi/模型级poll，不启动或修改任何实验。保留原owner命令和退出码作为最终完成/失败权威，事件仅用于及时取得已完成子单元和训练阶段。

2026-10-15：定位对照200实际完成，27,267,584参数、identity系数0、200真实pair_indices与完成tracking200相同，全部total_loss=native_loss且有限；16,164,180BF16元素改变，reload0。导出569张量其余不变。96native帧/18Full组合完成并核验真实官方artifact、完整帧、GT/future不读、相同前端候选在两个One中逐项精确保留；定位重复raw wire/候选/输出一致。native MOT4 HOTA原始75.069→定位84.099/跟踪83.467，single8原始30.385→18.776/24.876，multi12仅22.664→23.355/23.675；Full Single8六组合均0且FN5，multi12非普遍提升。保留负结果，不将4/8/12短片外推完整视频，不称联合梯度训练或模型晋升，reference训练GPU6/controlGPU5区别明确。parent正常接入自然MOT20，无重新启动推理。

只读事件转发器实际注册Full/GPU5queue/native-retention三个PID/starttick并输出九个状态路径已有快照，原阻塞session70182/36847/52020保持权威；新observer session58829只inotify/pidfd阻塞等候，不周期检查实验进度、不开新模型或改实验。注册证据单独固定保存，持续events.jsonl本地增长不作为冻结完成结果发布。

研究计划：先固化Full1203三臂中已完成的两个不可变单元，逐帧核对注册V1候选（仅按原计划剥离feature向量）、官方metrics、完整1203帧视频、checkpoint和最终memory；第三teacher臂未完成即明确pending。补齐same-source Full对Simple/Compact的正式表，不新增模型推理或更改科学协议。然后沿原三个阻塞句柄等待剩余Full/native9/natural完成。

Full1203两个已完成单元逐帧核验通过：每个1203真实decoded帧、13664原始V1候选精确一致（只剥离注册feature向量）、27拒绝帧保留、checkpoint准确、finalbinarymemory存在、无GT/future读取，官方metrics不变。MOT500 HOTA7.848/IDF5.056/IDSW3530/FP6387/FN765；追加native600 HOTA8.703/IDF5.826/IDSW3631/FP7516/FN416。只报告两个完整单元，第三teacher仍pending，非完整三臂完成和非E2E。

同源候选IoU.5一对一覆盖4341/4586GT=94.658%；Full before500/native600自身existing-anchor事件4008/2912中正确选项3952/2767，最终正确关联214/149=5.339/5.117%。原始argmax正确254/178，globalmatching丢失222/145、救回182/116；identity ranking与全局竞争均有问题，不只Ground漏检，不做跨模型同事件因果比较。新增只读已存events的UPDATE拆分初次守恒失败，原因是NEW初始化的memory_action也写UPDATE，原analyzer将NEW单列；按原分支语义排除NEW后守恒通过，未改原事件/指标。before500/native600真实known-consistent更新192/136、known-conflicting594/465、GTmatched未知birth2545/2984、unmatched5812/6648，后两类不冒充错误身份或负监督。保留原权重/预测与负结果；等待teacher和自然memory完成后再据主要原因选择最小受控修复。

研究假设：Full的native Q3在全局匹配前按candidate给UPDATE概率，Hungarian可能将candidate分配到非argmax身份，Q3未显式看到最终分配。先只读两个已完成完整视频的frame_decisions与原固定birth-anchor诊断，区分已知正确/已知冲突/未知birth/unmatched更新，统计原argmax一致性及真实选择概率。仅评估argmax一致性约束会拦下哪些已发生写入，不改任何状态/权重/阈值，不将冻结轨迹计数冒充因果rollout收益；据此决定是否值得后续受控修复。

2026-10-15：只读Full两个完整MOT轨迹原决策，确认Q3是全局匹配前的candidate观察可靠性问题，未指明最终assignedID。已知birth正确/冲突分配上Q3 AUROC before500=.513176/native600=.485748；native600两类Q3均值.852417/.849386，assigned identity概率.042690/.042338，几乎无区分。冻结轨迹若加assignedID=argmax约束，before500会拦173/192正确及518/594冲突写入，native600拦111/136正确及399/465冲突；同样大量伤害正确写入，不采纳该规则，不以冻结计数冒充因果rollout或HOTA修复。所有原始事件与状态不改。

Full fit监督实际核验：48合法fit帧/1184native候选全部Q3旧标签按visibility与correct-option availability精确重现，2460条family-matched eligible native Q3副本state/question/target与源完全一致。重要覆盖缺口：旧100步actor仅74NEW，其余1110无最终身份，0次沿用已有身份；因而该native回放没有实际wrong existing-ID memory-write上下文。训练仍有大量association/lifecycle目标，不能说完全无关联监督；也不能由eligible审计宣称实际600prefix因果已证。保留原标签与权重；后续最小可解释方向是刷新当前模型fit自身历史、显式条件于最终分配的Q3问题与监督，而不是给旧candidate问题偷换标签。此轮不增加GPU、特征缓存或新模型调用，原Fullteacher/native9/natural队列不动。

2026-10-15用户明确最终目的为达到MOT/RMOT/OVMOT常规模型水平。更新最终验收：Pilot/测评跑通/负结果审计不是最终完成；三个任务需同协议常规参照并实测缩小差距，统一同一checkpoint，不用短片/跨划分分数或三任务各挑一模型替代。公开ByteTrack/TempRMOT-v2/OVTrack为参照；实际查到TempRMOT公开train-list覆盖本地V2 validation0007/0015/0016，发布权重不能充当未见视频公平基线，公开分数只作方向参考。
执行研究计划：针对已证原生48帧旧actor0次existing assignment的支持缺口，复用全部原始1184真实DLM候选，从已完成native600权重刷新同48fit帧的真实自身历史；不发新Ground请求、不用GT纠正actor、不立即盲目延长训练。CPU先核对回放合同与预算，单一阻塞owner仅等当前GPU4 Full1203三臂成功结束后接棒一张GPU4，新增导出上限512MiB并维持全局25GiB；根据真实新rollout是否提供已有身份/冲突支持再决定下一小步监督适配。现有4/5/6/7任务不停止。

当前native600回放准备已实现并通过真实CPU合同核验：全部48帧/1184候选、三段各16原fitframes/图像存在、600checkpoint存在、原GPU4 owner2049900/start42845776仍live、注册prior+512MiB=26,262,799,802<26,843,545,600bytes。新可选replay参数保留原模式默认行为，不请求Ground、不加feature向量/不修改原课程；按当前actor真实状态产生独立新语料。新阻塞调度只在当前Full三臂成功且owner退出后接棒GPU4，未提前启动模型或任何新训练。新增目标文档明确普通水平是最终验收，公开分数须协议匹配，旧V1 TempRMOT94.679短sanity不作为V2正式水平证据。

新刷新owner2353528/start44628629/session45947已实际启动并阻塞等待原Full GPU4 owner；当前WAITING_ORIGINAL_GPU4_FULL1203_OWNER_EXIT，无新增GPU模型或训练。注册快照 `training_data/native600_fit_rollout_refresh_v1/registration.json`；原句柄70182/36847/52020及事件流58829保留，新句柄独立等待接棒。

研究计划：全9任务原生保留评测仍在运行，但已有不可变完整task_results。调用原生未改summarize只读取已完成task，核对n-original=n-effective/provider-filter0，固化原始6项与adapted200两项及共同dense200/VisDrone对照；未完成任务不补分、不汇总成全9结果。明确原始queue32与适配sync执行差异及随机requestcounter不构成bitwise配对，报告实际变化，不将差异全部归因训练。继续保留原owner/服务。

原生保留完整task阶段已固化：调用原scripts.evaluate.summarize不改scorer、不调用模型；原始6/9（31986完整task行）、adapted2002/9（1810行），各task n-original=effective且provider过滤0；原始COCO严格按既有resume复用，所有raw源仍本地保留。共同Dense200 F1@.5 78.986→78.679（-.307pp），F1mIoU66.644→66.054；VisDrone F1@.5 46.125→42.145（-3.980pp），F1mIoU29.662→27.870。后者precision/recall均下降，未见该两项保留能力改善，不推广视觉200为更好基础模型。原始queue32与adapted-sync、CUDA恢复及requestcounter不同均明确，不把差异独归因训练。其余full9继续原句柄，不将当前response进度前缀冒充完整task或完整9项分数。

2026-10-15训练规模与创新声明纠正：TRAINING_LOG顶部仍是旧500步运行状态，现以两个实际600追加阶段的完成校验、Compact训练配置及Ground200/320记录更新，保留旧快照为历史。明确仅做过共享多任务微调和分开基础模型适配，没有Ground+One端到端联合梯度训练；3595/8767秒仅150/350步恢复尾段，重复语料行不等于独立视频帧。方法设计仍待收益证据，未证明独立JEV创新或达到三任务普通模型水平。下一步仍按已登记GPU4依赖刷新48fit帧/1184原生候选，先看实际existing-ID/冲突支持再决定训练；本次不新增GPU任务、缓存或测评。原五个阻塞句柄均实际返回still-running，继续保留原owner，不因观察等待重启。

2026-10-15：原Full1203阻塞句柄70182实际exit0，第三teacher600完成官方与events，原回放owner45947自动接棒GPU4，开始48fit真实历史，不新增Ground调用。完成curator验证全部三组1203decoded帧/13664逐帧相同V1候选/27拒绝帧/checkpoint/finalmemory/官方指标，failed和pending均空。teacher追加600 HOTA7.5766/IDF5.0149/IDSW3839/FP7683/FN384，低于MOT5007.8485/5.0561；native6008.7030/5.8260亦没有稳定跨任务收益。不采用模型晋升或根据验证阈值调参。teacher已知existing-anchor1833中正确选项1588、实际正确82（4.474%），argmax77、matching丢69救74；已知一致UPDATE56、已知冲突152、未知birth2962、unmatched4977，未知不当负标签，所有候选与指标不改。保留负权重和轨迹，等待当前actor刷新真实关联/冲突支持再定下一受控训练。curator增加完整三臂终态声明并修正追加600不等于完整训练lineage的表述。

2026-10-15当前native600刷新原阻塞session45947实际exit0：三段48fit帧/1184原始DLM候选和perception逐帧精确一致，5741监督行，165806581导出bytes<512MiB，无新Ground请求/feature或GT纠正。审计CLI支持独立source/out，旧默认48/1184/2460和0existing全部计数精确回归；刷新全部Q3旧观察语义target逐条重现。实际658existing=45known正确+568known冲突+15未知birth+30unmatched；82NEW+444无最终identity守恒1184。实际UPDATE正确12/冲突41/未知12/unmatched20；300known冲突的原candidate观察Q3 target为yes，这不是selectedID正确性标签，不给旧问题偷换标签。证明可训练错误历史支持缺口已补，不证明精度提升；下一受控改动优先刷新历史并匹配多任务/问题family采样，保留RMOT/OVMOT和原runtime，用实测检验identity ranking，显式assigned-ID Q3另作独立协议。原GPU5自然和6/7保留队列继续，GPU4刷新结束，无新SFT启动。

研究计划2026-10-15：当前native600fit48已出现658真实existing关联/568已知冲突，旧actor为0。检验仅刷新MOT原生actor历史能否改善identity ranking，同时不重复仅MOT训练导致RMOT崩塌。准备两组各600追加updates/2400样本，三任务各800、相同任务/问题family逐位置配对，MOT保持400同一teacher行+400同family native行，RMOT/OVMOT逐位置内容相同。控制组旧native历史，干预组当前600新历史；两组从同一真实600 FP32 masters、seed0/lr2e-6/原loss/四行梯度累计，单GPU4串行。数据准备和CPU真实token规划先验证无超长/漏行及新导出预算，之后才启动训练。新增语料只采样引用既有图像，保持完整原始语料/权重；不是大规模unique-frame或Ground+One联合梯度训练，不改验证阈值、指标与测试GT边界。

600历史配对语料准备完成：每组2400行、800/800/800任务曝光，MOT400teacher完全同内容+400same-family native替换，实际新history400unique行覆盖48fitframes，RMOT346与OVMOT430所有源行均出现；新增48712542bytes，保守总25957225229<26843545600。官方CPU两组max25998/27960tokens且0超长/丢行。首次post-plan比较失败，根因官方plan_steps按长度重排step内部四个microbatch，raw列表顺序不相同；逐row核验所属step/index/bucket/family完全相同。修正核验为原样保持官方内部顺序、比较精确row-to-step membership，复用两个完成CPU计划；v1失败receipt/log保留，v2全部通过、958真实media路径存在、不称token/FLOP匹配。下一执行：同一GPU4顺序old/refreshed两个实际600step full SFT，均从原native600精确FP32 master初始化，原生loss/seed/clip/阈值不改，每50步原子FP32+Adam/RNG保存；每组完成后实际两次MMreload，再做同协议tracking。不是最终质量完成或大规模联合训练。

首次配对训练owner在模型启动前exit1，原因是官方CPUplan-only在正式name下预先创建args.json目录，not-exists启动检查因此停止；实际两个目录仅args.json，plan_only=true，无训练log/ckpt/模型，0新增训练。保留失败owner为full_history_refresh600_pair_v1，将两份参数目录精确归档到preflight_v2/plan_only_run_dirs，不删任何结果。未来CPU命令显式使用独立_token_plan名称；数据/种子/loss/600步预算全部不改，以新v2 owner启动同一尚未开始的实验。

配对600真实owner2448456/start45150423/session21963现运行old_FULL600_TRAINING，launch commit d938c520。GPU4启动前81027MiB空闲；真实752393024language参数/320FP32tensors从原600master精确初始化差0（相对BF16初始500278218元素改变），官方实际600步/0丢行。新phase种子0/loss/4行累计不改，控制组结束并实际MM两次reload后自动接refreshed组，原GPU5自然与6/7保留队列继续。GitHub MCP持续transport failure，最新代码/config/审计均本地提交并登记待发布，不伪称已远端同步。

后续测评预注册计划：在训练分数未知时固定same-checkpoint四scope：原V1 DT1203/ReferKITTIv2两表达各800/原始TAO BaseNovel视频40标注帧1203输入类。前模型MOT/RMOT已完整核验则仅复用，补齐原native600的完整TAO Full native-hierarchical对照；两新checkpoint各完整四scope，不按任务选权重。GPU4等当前配对训练及真实MMreload全部成功后测old；GPU5等原自然队列真正退出后先测before TAO，再等训练测refreshed。无新DLM/ROI档案，原候选及所有空/拒绝帧完整保留，所有评分用原官方程序，TAO标注时钟不称真实秒。该范围仍是注册开发视频而非全数据集普通模型目标完成。

四scope队列CPU来源核验通过：1203/800/800/40帧、13664/1412/2208/35228候选，27/36/19/0原拒绝帧，真实图像存在，TAO1203输入类/expressionNone。旧native600基线checkpoint/source/official指标一致；新worker验证函数对已有完整三scope实际逐帧候选/原拒绝列表/官方指标/final memory/video解码1203/800/800回归通过，终态pidfd依赖读取在已完成Full三臂真实owner通过。旧训练25步日志有限，未改变运行。下一步仅启动两个CPU pidfd等待worker，不提前用GPU4/5；后续同一checkpoint四scope、原scorer与输出阈值不变。

两个后续控制worker实际启动，仅CPU pidfd等待、new_model_calls_started=false/emptycompleted。登记 [{"worker": "old", "pid": 2467896, "starttick": "45243395", "blocking_session": 14226, "status": "WAITING_PAIRED_TRAINING_AND_BOTH_NATIVE_MM_RELOADS", "physical_gpu_only_after_dependencies": 4}, {"worker": "refreshed", "pid": 2467924, "starttick": "45243513", "blocking_session": 46228, "status": "WAITING_ORIGINAL_GPU5_NATURAL_OWNER_EXIT", "physical_gpu_only_after_dependencies": 5}]。原训练21963/GPU5原队列36847/native保留52020/observer58829均继续，未提前占GPU；各新worker配置与source核验均在任何新score之前固定，后续每个训练checkpoint覆盖同四scope。

常规关联基线计划：同一DLM候选的Simple不足以代表常规跟踪器，下载官方FoundationVision/ByteTrack源码commit d1bf0191adff59bc8fcfeaa0b33d3d1642552a99（直接无代理成功），只跑CPU官方BYTETracker，不下载检测器或改现有Foundation环境。固定官方CLI track_thresh.6/buffer30/match.9/mot20false；原候选无置信度，显式constant1全框输入，低分支无候选，不伪称完整ByteTrack/论文成绩。官方KF输出框与默认约1s丢失buffer保留，不按验证分数调参或附加输出面积/宽高过滤。完整DT1203/13664/27拒绝帧原官方TrackEval打分，先隔离CPU依赖，再一次完整阻塞运行。

官方ByteTrack CPU参考环境已直接无代理安装到独立reference_envs/bytetrack/site-packages，固定NumPy1.23.5兼容原np.float、SciPy1.10.1/lap0.5.12/cython-bbox.1.5/OpenCV4.10，不修改Foundation环境。只用官方tracker包、跳过无关YOLOX root初始化、上游源码不改；Torch库只读复用但不加载任何权重/不初始化CUDA。真实前21帧/266原候选含拒绝空帧21处理通过，输入boxes和unit scores完全不变、frame counter21。下一同计划完整1203帧一次运行+原TrackEval，默认KF/1s lostbuffer及constant1低分支空输入限制全部明示，不按验证分数调参。主训练50步真实FP32/Adam/scheduler/RNG已保存，原队列不变。

ByteTrack完整阻塞75056实际exit0：原官方CPU跟踪器1203帧HOTA16.4734/IDF13.9034/IDSW367/FP5234/FN395。独立逐帧核验13664全部原V1候选（只省原feature数组）、图像/框/原perception精确相同、27拒绝帧与scorer分母相同，全部predictions.txt逐行重建等于实际frame outputs，finalKFstate保留，TrackEval原12c8791未改。高于Fullnative6008.703、低于共享Compact26.316，但CompactFP579/FN1962相较ByteTrack5234/395是不同误漏权衡，且训练/特征输入/1s vs3s记忆/KF输出有差别，不作唯一JEV或普通全pipeline水平结论，不下调最终门槛。参考解释保留；不替换运行中的Full/Compact或调验证阈值。

扩展训练数据准备计划：当前Full类别语料64图/8video/9Base类，RMOT72物理帧/9fitvideo/18表达，不够描述长时间大规模联合训练。现有冻结MOT训练frames实际40904（MOT172664/Dance33772/MOT204468），TAO训练注释18274图/500video，额外freeform类不能冒充1203输入词表监督。实现纯CPU可用性索引：仅冻结MOT train_frames、V2全部9fit的本地图像和所有表达、TAOtrain映射Base(c/f)对象引用，所有无Base帧和缺文件记录保留，不当负标签/不读取eval或forbiddenGT。仅元数据128MiB预算，不开模型/feature缓存/新训练，不改变当前600对照。后续根据真实测评选择扩展课程和actor rollout，索引不冒充训练曝光。

扩大训练数据索引实际完成：MOT40904训练帧/43video，RMOT1814本地图像/全部9fit/4350表达文件，TAO18274训练图/500video/175Baseclasses/52978eligible targets；当前旧类别SFT仅9类/64图，规模缺口有实际可用数据支撑。所有index图像is_file true，RMOT所有positive表达frame都有本地图像；MOT唯一dataset/sequence/frame40904，9fit集合精确；逐一52978target核验源imageid/annoID与frozen vocab synset/frequency一致，52978+6172nonBase+50outsidevoc=59200守恒。175类别coverage sum同52978；1251零eligibleBase帧全部保留，不给Novel/unknown负标签。不称外部完整RMOT序列长度证明/实际训练曝光/联合梯度训练。原始indices约33.85MB本地保存，发布可复现builder/sourceplan与summary/coverage/validation，未创建任何feature缓存或新GPU。原训练150/600真实loss/grad有限，ckpt150和完整FP32/Adam/RNG已保存，继续原队列。

175Base语料组件实现计划：现有原生NativeCategoryHead.crop_sheet可直接按原图/精确GT框惰性生成768宽、216/432高实际像素；根/条件leaf问题原样沿用。全部52978训练Base对象构成105956问题，保留1203输入词表、1251零Base帧登记，不加Novel/unknown负标签。压缩JSONL+media裁剪recipe避免约GB的重复文本/PNG落盘；新独立训练入口仅进程内扩展gzip/load_media IO并将IO版本写入恢复协议，现有对照入口不改。先真实旧215对象像素/问题对齐及175新类代表图、CPU processor张量和官方全量token计划核验。仅准备未来平衡三任务混合的类别组件，不启动类别单任务optimizer；256MiB导出cap、合计26359878413<25GiB。

2026-10-16：175Base类别组件已实际生成并完成CPU原生全量规划session25047 exit0，52978对象/105956Root+Leaf问题、500trainvideo，1203输入类保留。gzip文本25820145bytes对比展开1005240705，17375按需裁剪recipe无PNG/ROI档案；18274图含1251零Base完整登记不造负标签。实际215旧对象/62原PNG像素完全相同，175新类代表问题/像素等同原NativeCategoryHead；768x216/432×Root/Leaf四例input_ids/mask/mmtypes/pixel_values/grid全torch.equal且实际计数token一致。全量105956target逐一对应源GT图像/类别/ROI且两种family各52978，同annotation集合。官方CPUplan0丢弃、max4238、总315781843计数tokens，26489步只是4row下假想全类别epoch，不是训练。Textrows仍一次载RAM，CPU峰3746652160bytes；原cached native reporter打印1s只含复用后的报告时间，绝不当tokenize全程速度。新独立IO入口仅进程内gzip与cropSheet读取、checkpoint协议记版本，现有200/600对照正常/FP32完整保存、其他队列未动；此组件将供未来三任务平衡课程，未发新GPU训练、不称大规模联合训练完成。

2026-10-16表达覆盖扩展计划：4350合法fit表达投影现有72真实原生帧，2213条至少有一个正目标、2137条没有；34800 contexts中15446正/19354空。直接平铺重复存在语言/正负覆盖偏斜，不能称图像规模扩大。复用原OneJev瞬时membership/empty-presence问题和既有裁剪PNG，扩展表达文本与严格train GT IoU>=.5标签，全部表达/空帧/原8拒绝帧保留；逐表达写覆盖供未来平衡课程。先与真实native build_request的所有72帧输入/裁剪像素一致，再逐条复现旧346监督行，完成后全量CPU token规划。不新增GPU/feature/PNG，不改运行中600训练，不称actor history或运动表达可观测。64MiB导出cap，保守合计26426987277<25GiB。

表达组件首次生成session35503 exit1，首个根因：0017/frame1表达individuals-donning-jeans引用ID148，但同帧release box中无148；全4350×8核验共60contexts/15表达、仅4个独立(video,frame,ID)引用缺框：0017/1/148、2/148、5/146、6/146。已有18表达/346行无此问题，不能由旧子集通过推出全表达标签可用。失败partial train.jsonl.gz与failure_audit保留，未训练/未删除任何样本。最小修正为全量target-ID/box预检前置到写行前，明确HELD并exit2，0行写入；不交集删除目标、不插值框、不偷换负标签。下一复跑只验证真实4350全量阻止无证监督，等待原始可见性/annotation合同证据后再恢复这一组件，GPU4/5/6/7队列保持。

官方TempRMOT固定commit6a65640d849fdee4a32bb055945ee34c3b0edeb1的_pre_single_frame源码证实：遍历同帧box列表，按ID是否属于expression ref_ids设is_ref；不要求expressionIDs逐个有框。实际函数回归v1因验证器只给image路径加KITTI别名、label仍本地小写路径，触发官方center而非KITTI左上角分支，exit1；这是我们的验证器路径设置错误。最小修复仅label也软链到KITTI路径，源图/label字节与官方函数完全不改，v1失败目录保留后以v2重跑60受影响contexts及旧18表达上下文。官方规则确认后才恢复新组件，且显式记录60缺框上下文，不改旧训练/评分。

固定官方函数实际198cases通过，包括全部60缺框contexts与旧18表达；6旧空帧无label文件按旧协议登记，未伪造文件。官方方法只给已有box赋is_ref，不生成未有框ID，所有60contexts仍有其他同表达正目标，所以raw presence与有框target presence逐一相同；这不是物理不可见性结论。新组件显式采用该官方released-box projection，要求固定验证receipt、记录全部缺框引用，所有34800contexts/4350表达保留，原346的state/question/target/task必须精确复现，q1实际PNG也逐个比对。此修正仅适用于尚未训练的新组件；旧训练、评测GT、loss和阈值均不变。失败v1/v2保留，新v3由原数据重建。

2026-10-16表达组件最终生成session50809 exit0：87721问题/4350表达/34800contexts，72原生帧165候选/8拒绝全部保留；6873candidate yes对77189no，presence1199yes/2460no，2213表达有sample positive/2137无。所有60缺框上下文按固定官方box projection保留并逐条记录，不读eval、不插值GT；旧346样本state/question/target/task精确，136native media与200q0/q1像素对通过。该语料是原primary-expression DLM候选的counterfactual membership，不是新表达重新检测，不称物理帧/actor history扩大。全量CPUplan session48280 exit0，87721/0drop/max2301/120061085tokens；actual plan_epoch111.924秒，CPU峰2068492288bytes，无CUDA/optimizer。上游cached reporter0s/默认4ranks不等于全tokenization速度或4GPU，args实际CPUworld1；后续仅明确PLAN_WORLD=world。21931是4row假想epoch不是已训练步数。压缩3156280bytes，coverage零计数显式填0；全部生成和失败真实文件7974645bytes<64MiB，保守合计26426987277<25GiB。保留该已验证未训练组件，下一课程须处理正负/表达分布并补真实帧覆盖；不改变当前600两组/四scope队列。

2026-10-16三臂记忆诊断补齐计划：之前full_mot_assignment_memory_v1只含MOT500/native600，两组AUROC近随机；teacher600现在全1203完成，补齐同一只读脚本的第三臂，保留旧v1结果。读取原始最终assignment与Q3值逐事件核对0.5原阈值实际UPDATE，报告known-birth一致/冲突分别的写入率、Q3/assigned概率/argmax一致AUROC；unknown与unmatched仍单独列出，不能算负类。旧两组重算所有已有统计精确复现，新的teacher与其原event计数一致才保留。只做已完成轨迹的posthoc诊断，不改任何预测、指标、阈值或运行中训练，不称因果反事实rollout或训练修复。

2026-10-16三臂记忆诊断session80514实际exit0，40992原始candidate events逐一reconcile最终ID/action/UPDATE且0.5实际gate完全匹配；旧v1两组所有字段逐项精确复现。新增teacher Q3 AUROC.499661，known正确写56/82=.682927，known冲突152/216=.703704；assigned概率AUROC.467395，argmax一致.479336。冻结teacher若加argmax约束会拦51/56正确及127/152冲突，只留5正确/25冲突，不采纳。三组共同指向候选观察Q3并未区分最终identity一致性，支持未来显式final-ID条件问题及匹配监督的研究必要性，不证明修复收益。unknown/unmatched不当负类，各模型内生事件集合不称严格同事件因果比较；无新score/GPU/缓存/推理，当前历史配对训练与四scope队列原样继续。保留新v2诊断及代码，旧v1不覆盖。

2026-10-16下一监督语义准备计划：三臂最终assignment门控AUROC近随机，当前Q3未指明最终ID。只构造新的final-assigned-existing-ID条件Noul任务，原始state/真实crop不改，附加由实际全局匹配得到的全部existing assignments（包含45未知/未匹配，不能用GT筛选输入）。按原shortlist与保存的历史别名一一对应恢复ID命名，GT仅外部标签：known birth同ID且visibility>=原.6为yes，known冲突或known同ID低可见为no；unknown/unmatched登记target null不造负类。全部48fit帧1184原候选登记；613已知监督预计29yes/584no，45existing无监督。独立共享question builder供未来post-global-matching/pre-write runtime，当前tracker和两组600训练不启用。真实全量CPU规划后才称输入准备完成，不称匹配runtime已落地或精度提升；16MiB导出上限，保守合计26443764493<25GiB。

2026-10-16final-ID组件session66147 exit0：新的纯question builder未接入当前tracker。实际48fit/1184candidates全部登记，658existing assignments全入对应group输入含45未知/未匹配，不按GT筛input；613known标签29yes/584no，300原Q3yes在明确新问题下no。用实际event shortlist对saved state/crop aliases逐一恢复命名且跨帧一致，无GT纠正。独立核验613新state去掉final_assignment后完全等于旧state、media精确、question每条命名实际alias；12个旧独立audit冲突例逐字段重现，45unknown均target null且保留在request，不造负类。完整native CPUplan0drop/max27234/13305265tokens，17.393秒，154仅假想4row epoch。当前两个600训练/四scope队列未改，无新optimizer/GPU/图片feature；实际组件约0.89MB<16MiB，保守合计26443764493<25GiB。保留该未训练未启用组件；后续须平衡正负并实现同builder的真实global-matching后/所有memory-write前runtime与因果三任务测评，不能据此声称修复收益或大规模联合训练。

2026-10-16条件写入runtime实现计划：新增独立AssignedIdentityUpdateDecision，默认OneJevDecision路径不改。在accepted native groups保存pre-write state/media；OnlineTracker全局匹配后、任一track/box/image写入前，将全部actual existing assignments传给同一final-ID问题builder，原生engine二次Noul覆盖existing update概率，关联/生命周期/NEW策略与.5阈值不变。类别wrapper只对新backend转交callback，RMOT保留原输出membership。显式--assigned-identity-updates限Full v3/gated，默认off，resume核对同一protocol；当前训练/worker未传flag。必要验证：原parent与新default在三组完整1203帧冻结模型决策、相同原nativefeatures/图像上输出/state精确；新phase在真实fit48轨迹的独立单帧分支验证pre-write边界、全部658existing含45未知进入问题、训练/运行state经合法ID别名对应相同。第二阶段用明确脚本答案只测执行逻辑，不称新模型预测/benchmark/修复收益，不额外占GPU。

2026-10-16条件runtime CPU验证session64482实际exit0：新增独立native second-pass backend，tracker在全局匹配后/任何track修改前调用，同一final-ID builder；类别wrapper转交，CLI显式Full v3/gated flag默认off，原0.5/association/lifecycle/NEW不改，保留观察Q3，resume要求同protocol。三组完整Full默认回放各1203/13664/27拒绝帧，parent blob9f092b04与新default逐帧outputs/state及最终memory精确一致，并重现原输出；未恢复trace未保存的visible life概率为原始full-memory真值。真实fit48的96独立单帧分支用明确脚本0/1Noul：1316questions/150pre-write calls，所有658existing每轮含45unknown，1316canonical state/question与150alias-history crop像素精确；长memory/可靠时间按script hold/update，输出ID/框一致、category forwarding通过。不是新模型推理、连续反事实rollout、训练收益或新HOTA；未发GPU/feature工作，原两个600及四scope不启用flag。小结果归既有16MiB reserve，总26443764493<25GiB；保留新路径，须先匹配SFT/重载和连续三任务测评才判断收益。

2026-10-16历史刷新对照首臂实际完成：原session21963输出old600/600，2400样本/三任务各800，全程15605.720秒，不是恢复尾段；最后loss1.020472/grad52.478677有限，峰21.618GiB。checkpoint600与final导出、9029140441byte完整FP32/Adam/RNG状态已保存。原owner两次真实native-MMreload474weights、概率差0通过，自动启动refreshed臂，仍从同native600FP32初始化差0、600planned/0drop。另CPU mmap完整状态检查session94584 exit0：320FP32参数/752393024元素、全部Adam一二阶矩有限，320Adam步数均600，protocol与实际args一致、scheduler/allRNG/cursor齐全；不称实际重启轨迹已验证。2400源行按原source核验800/800/800，保留完整29条训练数值记录与done，curated completion_validation.json。只更新完成证据和研究记录，不改两臂问题/loss/阈值/种子/四scope依赖，不启用新assigned-IDflag，不新增GPU或feature；当前没有新tracking指标或收益证明，完整普通水平目标继续。

2026-10-16研究计划：现有final-ID监督仅29yes/584no，先检查已有共享Compact能否在同48合法fit帧提供更充分的真实正确existing关联，再决定是否构造明确的跨actor历史监督。固定现有broad_native_training checkpoint，不用fit结果选模型；CPU2threads、原gated.5/3s/group16/shortlist32，逐帧复用全部1184原生候选/图像，真实Compact自身状态连续推进，GT仅外部posthoc一对一IoU>=.5及immutable birth标签，不修正actor。只诊断known-consistent/conflicting/unknown/unmatched以及可监督yes/no支持，不生成训练数据/PNG/feature，不称Full自身历史、泛化或tracking收益。新增16MiB小结果预算，保守合计26460541709<26843545600；不改当前GPU4两臂训练/4scope与原GPU5/6/7队列。

CPUfit支持诊断首轮session23118 exit1，首个动作根因是Full native回放没有native_region_feature，尚未产生任何frame预测。保留v1 owner失败记录。已有native_pack_v1/examples.pt由原NativeFeatureClient对完全相同48帧1184候选提取真实ROI；最小修复只按sequence/frame/batch和全部候选box精确对应，复用原float32候选特征。只排除重复option-withholding增强pack条目，不排除任何物理候选/帧；全部1184对齐、维度/有限值核验是本次针对性回归，随后v2同CPU/权重/策略完整重跑。没有补造特征、新GPU或新ROI档案，失败/修复均保存。

2026-10-16CPU Compact支持诊断v2 session34284 exit0：全部48合法fit帧1184物理候选与已有真实ROI精确对应，原gated.5/3s/group16/shortlist32、共享Compact固定权重/CPUFP32自身历史，无GT状态纠正。实际728existing=620known一致+102冲突+2unknown birth+4unmatched；82NEW+374无最终ID守恒1184。按原visibility>=.6新问题可监督377yes/345no，6unknown不造负类；对比原Full组件29yes/584no说明可补正例支持，但不同actor内生状态不是同事件因果改善或held-out精度。独立逐帧重算全部IoU匹配/birth/eligibility/targets与events输出框标签，有限概率、counts全复现，validation.json完成。2.33MB诊断原始trace/标签+汇总保留，新增16MiB reserve总26460541709<26843545600。保留该CPU证据，下一步考虑明确跨actor final-ID组件与原Full硬负例共同采样，尚未导出/训练；物理帧规模仍仅48、不改变当前两臂训练和原测评队列。

2026-10-16下一执行计划：实际Compact CPUfit48诊断已证明377yes/345no支持。构造明确cross-actor final-ID组件722known行，保留全部1184候选registry/728existing输入含6unknown无target；同原final-ID question builder，输入是Compact真实pre-write历史经Full v3 builder呈现，不称Full自身actor。不读新GT，仅引用已独立核验外部targets；同CPU2threads/固定权重/原候选ROI/策略完整再滚动并逐帧精确重现已保存outputs/state，才导出。记录每个实际anchor/long/recent的原图+框来源，用新独立按需reader重建native候选/历史cropSheet，每组与原生builder像素对照，避免PNG和feature档案。独立入口记录新input IO protocol，原category reader和当前训练/测评均不变；完整官方CPU token规划后才称输入可训练，仍0optimizer/GPU。新增16MiB预算合计26477318925<26843545600，未来须与Full硬负例平衡并做实际SFT/连续三任务测评，不据此宣称扩大物理帧或联合大训练完成。

2026-10-16cross-actor final-ID输入组件实际完成：CPU原session63984 exit0，Compact全部48帧outputs/state与已完成诊断逐帧精确；722known问题377yes/345no，1184candidate registry/728actualexisting输入含6unknown无target。80group/155native候选和历史sheet像素精确，2个31/32历史实际processor全部tensors exact；独立60712 exit0重建全部722question/target/alias/order/当前过去crop provenance，4个原ordinary/category passthrough真实像素不变。新增独立tracking recipe reader以原图+真实crop来源按需重建，8raw/192crop有限CPUcache，独立SFT入口input_io_protocol明确、旧入口与运行队列不动；无PNG/ROI档案。全量CPU plan722/0drop/max26614/14664308tokens，actual plan_epoch2.535831s、world1；181仅假想epoch不是训练。gzip3482066bytes，原始语料/registry保留server，generator/plan/receipt可重现；总reserve26477318925<26843545600。保留为未来与Full硬负例平衡的明确跨actor组件，不能称扩大物理帧/Full actor/实际SFT或精度提升。

2026-10-16原GPU5自然MOT20 fresh子单元完成并转CPU8controls：全部1657帧(1659..3315)/235232原生候选，117原parser拒绝按登记空帧保留；官方TrackEval HOTA 7.334696% / IDF1 5.959845%/IDSW1933/FP417/FN288297。独立81592 exit0逐行重建31185predictions，frame/image/event/官方记录守恒；1657FP32npz实际archive2311933077bytes<3GiB，原owner已逐帧archive/视频decode1657核验。真实native全pipeline27000.258秒/.0613698FPS，含DLM/ROI/IO/archive/状态/画图/编码完整结束，不含预加载服务启动；5秒采样ownedGPU5峰22812MiB为瞬时峰下界。posthoc候选物理GT recall219060/319062=.686575；动作NO_MATCH185585，匹配GT中169106不符既定missing-option规则，precision.0271648；158687是无存活正确birth锚点而预测NO_MATCH，194026此类观察仅212NEW，提示严重NEW/NO_MATCH及初始化学习失败，门控修复不能单独解决，不据验证集改阈值。132个3..30s返回中20known eligible，3仍存活/0恢复；112unknown单独排除不造负类。保留低分，原8CPU记忆对照继续，未来fit监督应补更广真实NEW/NO_MATCH场景。当前刷新历史训练原21963已50/600，完整FP32/Adam/RNG cursor50/200保存、loss/grad有限；所有GPU/种子/metric/scorer/原队列不变，普通三任务水平仍未达。

2026-10-16研究计划：MOT20实际NO_MATCH抑制严重，先读原536group最终训练pack发现MOT eligible2204候选中69NO_MATCH（69条均withholding增强来源）、1704existing/229NEW/202IGNORE，不能把eligible比例冒充未保存的实际抽样曝光。下一CPU诊断固定已训练checkpoint，计算全部536原训练输入上的argmax confusion/CE，按原history_geo四标量重建已完成1657帧实际pre-write输入范围，终态age/lost/reliable/conf与保存binarymemory核对，定位时间长度等分布差异。不改输入/推理/阈值/labels/当前GPU队列，不据范围直接归因或宣称性能修复；仅8MiB小证据reserve，总26485707533<26843545600。原自然CPU reference已完成HOTA7.431603%/IDF1 6.147357%，不同于freshGPU HOTA7.334696%/IDF1 5.959845%，需按同CPU reference比较后续CPU变体，不偷换基线或称CPU/GPU闭环精确等价；差异原因未隔离。

2026-10-16实际时间域审计52639 exit0：原eligible MOT历史年龄全部1/100=.01，confidence .991545.. .999416；联合所有任务年龄最大7次。完整MOT20源重建15391候选group/410796history行，95.752%年龄高于最终阶段全部eligible上界，年龄max909/中位121、confidence中位.429819，20.551%距可靠写入时间超eligible最大；final surviving所有age/lost/reliable/timestamp/conf与保存二进制memory逐值完全相等。冻结原536训练input上MOT exact ID/action accuracy.698276，existing->NO_MATCH380与NEW->NO_MATCH71（仅fit probe非泛化）。新研究计划：固定当前共享Compact checkpoint做两组CPU4各1200步/每任务400步继续训练，old物理80 MOT groups对真实Compact自身80groups，其余RMO325/OV62全同。双方明确去除69个derived-withholding增强（全部1184真实候选/48帧保留）和MOT教师蒸馏0.2项，比较真实历史刷新整体效应，不能单归因年龄或confidence。新actor数据直接捕获原Compact前向输入张量，GT仅外部target按原规则，连续48frame输出需精确重现此前CPU诊断；不造状态、不扩大物理帧。独立per-task sampler使每一步例ID和语义曝光配对，相同init/lr/seed/梯度/头/参数数，length/FLOPs可不同。预登记final1200同一checkpoint完整5scopes（DT1203/RMO800x2/rawTAO40/新MOT201657），复用已完成严格同CPU基线，原official scorer、阈值/拒绝帧全保留。新增两份pack总256MiB上限，合计26754142989<26843545600，零新GPU/DLM/ROI，不改当前GPU4/5/6/7原实验；完整普通水平目标继续。

2026-10-16原Ground九项完成及GPU7接棒：原native unchanged summary全部9tasks/91124原始=effective/provider-filter0，逐task raw results/metrics/counts完全对应，注册原8101 clients全部exit，owned service group1345024 gone；parentless runner实际exitcode未知保持未知，外层reattach仍等adapted200 GPU6。生成original_completion_validation与original_gpu7_release，GPU7实测free70229MiB，未动任何其它owner。已关闭不可再写的原MOT20 packedarchive cap3GiB到实际2311933077bytes、完成的两份Compact packs cap256MiB到实际223700168bytes，所有文件原样保留、仅释放未使用预留；保守总从26754142989降为25800115306。下一真实数据采集计划：GPU7原Ground+ROI、单客户端固定既有CompactCPU4 actor，43个冻结MOT train视频中点连续片段，共1056真实帧（MOT17 7x32/Dance32x16/MOT20两段32+两段128），不读GT选帧/不纠正actor，所有空/拒绝/候选保留。按clip exactFP32 packed ROI总上限768MiB、后续grouprequest/mixture refs各64MiB预留，全局26739639402<26843545600；不重启U2全缓存、不下载新数据、不启动未验证One训练。单阻塞owner接棒已完成原GPU7服务，后续更广物理支持与既有Base175/RMO4350用于共享SFT，不能把采集/预留称实际大规模联合训练。

原GPU7扩展源首次CPU preflight在启动任何模型/服务前exit1：直接遍历了协议中的held-out-only序列，其train_frames=[]，不属于原43个TRAINvideo计划，触发length断言；已写少量manifest和owner失败记录原样保留v1。最小修复按冻结train_frames非空选择训练序列，完全不改变43视频/1056训练帧/预算/GT边界，不遗漏任何计划训练样本。v2输出新目录，逐一全部image存在和43/1056计数通过后才准启动GPU7服务；保留原注册文件来源。

GPU7采集v2原阻塞75766实际exit1：首个run_video读取manifest时KeyError vocabulary，0frame/0新增候选；collector finally已停止其两个所属服务组，members逐组确认为空，失败manifest/log/owner及cleanup receipt原样保留。最小修复manifest显式vocabulary=[person]/expression=None，按既有MOT入口合同补齐，43train视频/1056frames、候选/种子/decoder/预算不变；v3使用新目录并重新完整预检后启动，未重跑任何已有训练/测评。Compact训练完成校验56311单独exit1的第一处实际断言是Torch sampler int keys直接比较JSON str keys；经47784实际读取归一键后值完全相等，原始失败记录另存。另实测原生Adam按条件分支参数计步profile为56参数1200/6参数1181/4参数1067，两组完全相同，不能要求未收到梯度的头也等于global1200；校验记录真实计步、不改optimizer。训练两组均已正常完成，saved model/state、每步sampler与全部RNG正确，校验修正不改变任何训练数学。

2026-10-16Compact两组真正继续训练完成并校验：70970 owner两组各1200/每任务400，逐步sample IDs完全配对；60698验证saved final模型等于checkpoint、有限、完整采样RNG/protocol，原生Adam profile56x1200/6x1181/4x1067相同、三任务reload差0。训练loop25.313669/25.500670s（1.146511M小模型，不称foundation训练）。数据两pack实际223700168bytes，refreshed监督941existing/126NEW/104IGNORE/13NO_MATCH，真实新history age最高15/conf中位.669；两组共同去69withholdaug/禁.2teacherloss，比较综合历史刷新非单scalar因果。原owner已验证11/15：DT HOTA26.3157->old9.6125/ref12.2511明显退化，RMOsingle6.0247->7.6779/10.0489，multi19.8019->20.6855/22.0629，TAOold14.517->10.426；TAOref及MOT20三单元仍待，不做模型晋升/按任务挑权重。GPU7采集v3原84427已实际完成首个MOT17-02 32帧/992候选/9818682 exactFP32bytes，所有43/1056 manifest和vocabulary/GTfree真实入口校验registration通过，继续后续clips；不是把准备当已训练。原FullGPU4已125/600 finite，既有所有队列持续。

2026-10-16扩展source转真实SFT计划：原GPU7 source仍单一84427运行；新增独立CPU watcher只读取collector声明完成的不可变clip，使用同已固定Compact/CPU4与exactF32归档重新连续滚动，全部outputs/state必须逐帧等于真实源再接受。用相同Full v3 builder投影其pre-write自身历史；按原合法GT IoU/birth规则生成identity/lifecycle/reliability/观察Q3，加独立finalID Noul（unknown无target且全部assigned仍在input）。新groupref IO只在独立SFT入口安装，一个组共享state/media/questions，逐问题refs targets外置；不重复大状态/不写PNG/ROI，与现有category/trackingrecipe reader组合并记录新resume协议。每个clip全部crop像素、question target合法性及官方全量CPU token plan必须通过，空/拒绝帧全部保留，64MiB既有export预留内，无新GPU或optimizer，不宣称Full actor/扩大语料已经训练。先完成可训练输入，既有Full175/600/Compact5scope/原natural8/nativeadapted9保持原队列。

扩展Full导出CPU92510已真实完成19clip/416frames/31745rows/allCPUplan0drop，发现producer快照是在可能耗时的clip导出前读取，source结束时应先重读terminal receipt再判断pid存活。提前最小修正该并发交接逻辑，增加--resume按验证完成clip接续；完整native CPU plan输出重定向每clip日志保留，避免模型侧被worst-row列表淹没。已按PID/cwd/script/starttick精确停止仅此CPUexporter，所有已验证clip原样复用，若有半写attempt保留另起该未完成clip目录，不动GPU7采集与其它作业、不改任何问题/GT规则/32768限制/样本。原92510的SIGTERM为明确执行修正，不是观察超时重启；新单阻塞CPU owner继续剩余clips。

2026-10-16新groupref独立原生输入验证72956 exit0：Choice/Score/final-ID Noul三类真实有历史问题，重新从原source起点按保存决策连续恢复各自前缀，state/question逐字段、所有native crop pixels和实际processor全部tensor/attention长度精确一致；6条旧ordinary/category/trackingrecipe读入不变，第一clip5301条官方CPUplan0drop。首次82003验证失败根因为复用forward-only ReplayFrames后重复请求135；失败另存、每case重建replay即通过，无源样本/标签/数学改变。无新GPU、模型前向或optimizer。

2026-10-16原Compact70970实际exit0，全部15/15五scope完成；75404独立逐帧重建全部raw prediction text与official文件/统一checkpoint核验通过。MOT20 HOTA before7.4316/old18.5218/ref17.6317，IDF1 before6.1474/old13.9389/ref14.2613；DT仍26.3157/9.6125/12.2511，RMOTsingle6.0247/7.6779/10.0489、multi19.8019/20.6855/22.0629，TAO仍退化。不按任务挑权重、不推广两新arm，不把RMOT和MOT20提升说成三任务普遍收益；新三单元原始预测额外无损gzip4,423,714bytes供GitHub发布，服务器所有原文件保留。

下一研究执行计划：固定更广43trainvideo/1056帧的真实Compact自身历史Full-v3投影，加全部4350合法RMOT表达（仍72物理帧）与175Base类别teacher-crop，组装24000行/三任务各8000、单共享OneJev6000更新。所有MOT1056帧至少出现一次；identity4000、final-ID2000、life1000、观察Q3/reliability各500；RMOT训练标签yes/no各4000且每表达至少一条；OV4000按175类分层对象成对root/leaf。仅train目标决定抽样，少数类重复需报告，不使用eval分数选样。完整CPUplan0drop和原生输入等价已证后，GPU7必须等原collector及其服务真正退出才接棒；从固定native600精确FP32 masters新phase，原Torch参考backend/loss/seed/lr、冻结vision不变，不使用已拒绝的FLA替代。6000步是计划尚未执行，不称Ground+One端到端梯度。最终同一checkpoint按固定四scope做before/default/final-ID对照，不按任务选权重或更改threshold。原600对照与自然/保留队列不动。

2026-10-16新共享6000 owner实际启动session36231，当前仅CPU pidfd等原group exporter/source collector，0新optimizer/GPU；接棒需43/1056全完成且原owned服务组消失。新phase每500步模型与完整FP32/Adam/RNG续训状态保存，预留40GiB检查点空间；不删除现有checkpoints。最终Compact三新单元逐候选events数百万行保留服务器，GitHub仅发布聚合诊断、official分数和无损压缩全部raw预测，避免误传巨大重复事件JSON。

2026-10-16共享6000课程执行前审阅发现：MOT mandatory每帧一条加入后，balanced补足池未排除已选ID，可能在同标签其余问题未用完前重复，违背已登记仅池耗尽后重复的规则。最小修正是每标签先遍历未选ID、耗尽后再循环全池，seed/目标/6000steps/各family配额均不变；添加实际最终MOT帧集合与完整1056源帧相等检查，报告聚合EXISTING计数及生成时真实commit。原owner36231仍仅等原source，未生成课程/训练，故不改任何已有优化轨迹或结果；后续全量CPU计划仍必须0drop。

实际41clip采样回归发现第二个实质问题：原800物理帧中identity/presence仅覆盖772，28空候选但仍有历史的native builder只生成life问题；原课程assert1056会提前失败，不能据772eligible帧宣称全800覆盖。保留第一回归receipt并标明范围。最小修正是每物理帧优先原identity/presence，否则必选原life问题；这些life必选行计入原1000life配额，4000identity/presence及其余family配额不变，不编造新empty问题、不改模型/标签/seed/步数。配置文字明确native问题语义例外，原source/export无修改，6000队列仍在等待。

上述修正已在真实冻结41clips的全部800物理帧上执行select_mot回归：实际选出8000行/7020unique行，3997identity+3emptyPresence、2000final-ID、1000life、500原Q3、500reliability配额精确，28个life-only帧全部保留。唯一发生重复的identity/NO_MATCH池确实仅19条，首次重复前19条全部已出现；共980重复明确可追踪，未把样本数当unique覆盖。所有原target/refs不改。完整43clips/1056帧来源仍由原队列采集，必须再全24000行CPUplan后才训练；没有增加步数或修改loss。

2026-10-16下一最小实证对照：此前old/refreshed两组共同移除69withholding和.2MOT蒸馏，不能将DT/TAO退化仅归因历史刷新。固定同一before共享Compact checkpoint，直接复用原536group pack（149MOT含69derived、325RMO、62OV）与原native_teacher flags，CPU4新phase1200步、每任务400、原lr2e-4/loss/seed0。用同独立task RNG与已完成pair保持RMOT/OVMOT逐步sample IDs精确，MOT149vs80池差异明确；新Adam/RNG重置与此前pair一致，非原optimizer trajectory resume。最终单checkpoint跑同5scope与原官方metric、阈值、全部空/拒绝帧，before复用同CPU来源。该对照检验继续原recipe是否也退化；不将withholding与distill两因素分离归因、不按验证分数调参、不做新cache/GPU或替换正式模型。完整Full6000/原采集队列不变。

原recipe对照owner实际启动session76373，已完成CPU1200/每任务400；RMOT/OVMOT逐步sample IDs与已完成history pair精确相等，三任务真实reload差0，完整FP32/Adam/RNG状态对应最终checkpoint且全部有限。原536pack/69withholding/native_teacher flags保留，无新GPU/feature文件。五scope连续官方对照正在同一原阻塞owner中运行，分数未完成不宣称收益。

原recipe续训当前已闭合DT/RMOsingle/RMOmulti/TAO四scope，均同一final1200 checkpoint。HOTA分别DT22.7949（before26.3157、去蒸馏/增强old9.6125），RMOsingle9.6612（before6.0247）、multi23.2670（before19.8019）；TAO TETA10.490（before14.517，Base12.1679/Novel7.1325且Novel ClsA仍0）。原recipe延长也不能维持所有任务，DT退化小于改recipe但TAO仍明显退化；不能由此单独归因teacher或withholding，不推广任何新checkpoint。最后MOT20仍在原76373 owner中运行；仅冻结已完整单元的partial snapshot，不宣称全部五scope完成。汇总工具新增original_recipe模式，在实际owner exit0后验证全部raw prediction text、同checkpoint及与已完成history pair的四臂分数。

2026-10-16训练加速独立探针计划：已有FLA推理drop-in概率差.021211>.001的拒绝结论及阈值不变。新试验仅比较native600同一真实FP32初始化的训练计算：5个已验证真实MM问题（有历史identity Choice/reliability Score/final-ID Noul，加RMOT yes和175Base leaf）每步各1，原生loss/seed0，native与显式FLA chunk各4更新/20行；实际CPU输入/paired index排列一致，第一次320个完整grad须有限，均匀采样每参数最多4096值用于方向比较并记录完整grad norms。事前gate sampled cosine>=.99、relativeL2<=.1、fullnormrelative<=.1；速度以steps2..4中位数比>=1.5仅作训练backend候选，不证明推理等价或tracking收益，不自动修改已有600/6000队列。GPU5启动实测free70274MiB，独立job需>=40GiB；仍只物理4/5/6/7，不停其它owner。两份小gradient diagnostics额外32MiB保守reserve使总26773193834<26843545600；无新图像/ROI档案，输出BF16权重属checkpoint保留。

FLA训练探针v1原34386实际exit1，首个reference单位在调用stateful.main(argv)时TypeError：该入口无位置参数、从sys.argv读取。未进入模型加载/SFT，0optimizer，原Full队列不受影响。最小修正为写原生CLI argv再无参调用；保留v1目录/log/失败receipt，v2新目录执行同一个尚未开始的reference单位，全部20行/seed/loss/gradient与速度gate不变。

原recipe76373实际exit0，全部五scope/10units闭合；79832实际exit0，全部raw prediction text、每帧/候选计数、official文件与同checkpoint验证通过，新五单元完整预测无损gzip共654414bytes。四臂HOTA before/old/ref/originalRecipe：DT26.3157/9.6125/12.2511/22.7949，RMO single6.0247/7.6779/10.0489/9.6612，multi19.8019/20.6855/22.0629/23.2670，MOT20 7.4316/18.5218/17.6317/8.5511；TAO TETA14.517/10.426/11.997/10.490。各续训checkpoint存在不同退化，均不推广；originalRecipe vs old不能单独分离distill/withholding因果，也非全数据集普通水平完成。更广MOT20-03实际128frame/14784候选（7原拒绝）79926行allCPU0drop/max26742/真实CPUplan323.406s；1,718,206,706仅全部源行规划token不是训练token。真实identity target NEW3384/existing6724/NO_MATCH2349/IGNORE2327，补上大量真实missing-shortlist支持；原42clips合计928frame/139101行，最后source仍运行，不称1056已全完成或已训练。

2026-10-16训练FLA v2原59743实际exit0：native与FLA各4真实更新/20行（仅5unique问题重复4次，不称大训练）；同源FP32 init差0，320全梯度均finite。均匀采样梯度cos.99954999/relativeL2 .03000655，完整norm72.596613/72.614843 relative.00025112，满足事前.99/.1/.1 gate；steps2..4 median比6.0483，循环含首步native691.5s vsFLA218.0s，组件并发条件下波动大，不外推固定整次加速倍数。66703实际exit0：FLA训练后BF16checkpoint在原reference MMengine两次真实重载概率差0、原fork check不变；49469生产入口一次5行targetedSFT真实exit0，首步loss2.9994280758182867与probe完全相等，86576仅CPU协议校验确认inputIO/pairedShuffle/显式training_compute_backend字段。新增checkpoint输出参数默认不变，不改旧500目录或任何当前训练。计划在原6000仍0步、等待source的前提下以显式v2后端接棒该未开始phase，保留v1计划/owner及交接receipt；数据课程/seed/loss/6000steps/.5阈值/原native推理/官方metrics均不改。若原6000已启动模型则不停止它；不得用采样梯度近似冒充完整向量或推理等价。

6000后端交接已实际执行：核对原3136429/start47470137仍仅pidfd CPU等待、无child/课程目录/训练目录、0更新后，只SIGTERM该waiter；原36231实际exit143。原collector84427、exporter55994、GPU4/5/6既有实验均未停止。v1原计划、pre-handoff owner状态、intent和带真实退出码的handoff全部保留，明确新v2compute protocol非原优化轨迹恢复；同一尚未开始6000phase改用已验证FLA训练，数据/种子/步骤/原生推理及官方评分不改，没有取消任何已开始训练。

2026-10-16原自然队列36847实际exit0，全部8memory controls和localization阶段结束；原GPU5 Full worker46228自动接棒beforeTAO native600，无需改调度。66881实际exit0独立重建8x1657frames/235232candidates/117rejects全部raw prediction text，全部official一致，8份原预测无损gzip共6958227bytes。CPU HOTA gated3 7.4316/frameOnly7.7886/recent8.3869/EMA7.5374/anchorAlways8.4831/anchorGated30 6.8545/Simple3 33.2021/Simple30 33.1562；Simple IDF1 29.2386/29.2325，IDSW11289/11266。仅延长TTL未提高全局HOTA，但>3s candidate-gap同birth ID恢复由3/133到5/133，短gap恢复下降；不能说所有记忆指标都无收益，也不混同既有严格3..30s自然visibility-gap分母。Simple不是learned/JEV模型，不用33.2冒充统一foundation普通水平。旧GPUfresh7.3347与CPU7.4316闭环不同，始终以同CPU reference比较CPU变体。

原collector84427实际exit0：43train clips/1056物理帧全部完成，所有原空/拒绝帧保留，packedF32 archive474314908bytes<805306368cap，所属Ground/ROI finally均在真实exit前关闭。最后MOT20-05 group导出/完整CPUplan仍在原55994运行，未称全语料可训练已闭合。已启动新6000训练owner64133（3219708/start47863228），仅CPU等待原exporter/source成功，显式fla052_training_only/referenceInference；尚未开始大phase optimizer。原600步刷新已425/600 finite、原FullTAO/适配Ground9项队列继续。

共享大phase实际开始：原export55994 exit0，全部43clips/1056frames/252258监督行的真实CPUplan0drop且全部state/output/pixels相等；64133自动完成课程24000行（MTO/RMO/OV各8000），max27346tokens、全190139221计划tokens、world1/6000updates/0drop，然后在GPU7加载真实752393024语言参数（视觉冻结），native600 FP32初始化差0。现冻结actual75steps/300rows首段（记录实际任务与图像/类别曝光，不把全课程覆盖当已训练曝光）；首段loss/gradient有限，已到LR warmup。课程独立source行8000/7475/7298，OV实际3649objects/3227images/175Base，不把4000draws或全部500sourcevideos冒充独立曝光；RMO4350表达仍72物理帧。FLA训练backend显式进续训protocol，评分/inference走原Torch。6000并未完成、仍无新tracking质量结论，500步才第一次完整FP32/Adam/RNG原子保存；旧Full刷新已450并继续。

2026-10-16研究计划：当前正式6000混合SFT已实际725步/2900行且500步完整状态已保存，保持原课程/seed/loss/原生推理及四scope终评不变。下一独立数据组件扩展RMOT物理时间支持：固定既有9个fit视频与每视频原primary/counter两条表达，遍历本地全部1814帧（1742帧超出旧72），每表达独立真实Ground+Compact因果rollout，共18条连续query视频/3628 query-frame观测；不能把两query算3628物理帧。帧选择不读GT、不挑可见性/正例；全部空/拒绝保留，GT仅后续监督。已只读审计168个无label文件帧在两表达均无raw目标、0个unboxed context；下一执行原固定TempRMOT reader逐frame/query验证盒与membership投影。先将已完成且不可再写的MOT源/导出/课程/FLA探针预算由cap关闭到实占（不删文件），再预留384MiB exactROI+48MiB group+16MiB后续课程refs；只CPU准备并登记唯一阻塞owner等待原GPU6 Ground全九项全部完成和所属服务组释放后接棒，不借用运行服务、不停止任何现有训练/测评。当前准备/排队不称新采集、联合训练或效果提升。

RMOT全fit准备已实际完成：唯一CPU47888 exit0，1814物理帧/1742新增、18独立query scope/3628上下文。执行固定TempRMOT未修改reader3292次，所有box/ID/query mask与本地标签解析一致；另外336query上下文对应168个原缺label且两表达均无目标的物理帧，完整登记保留、不造label文件、不除去帧。无unboxed目标、0GPU/模型调用/新训练。已关闭MOT源ROI/全groups/不再修改的6000课程/训练backend诊断四份cap到实际，文件零删除，关闭后26336324458bytes；新增384+48+16MiB预留后26806086506<26843545600（37459094余量）。原6000输入/优化器不动。新collector仅等待原1865651/start41803969完整九项与1450601服务组释放，随后最多原GPU6一个DLM+ROI，固定CPU actor对两表达分别从真实连续起点运行；不共用原保留评测服务、不读取GT或未来帧。

新RMOT collector19338已实际启动且CPU pidfd等待原GPU6 owner，0新query帧/0新GPU模型调用，registration冻结记录与实际进程starttick一致；原52020仍live等待原adapted200九项。正式6000原64133最新实际800步/3200行，finite，500步BF16模型及9029140633bytes完整FP32/Adam/RNG已原子保存；冻结first800 snapshot，不用数据课程总覆盖冒充已消费曝光。旧Full刷新原21963已550/600，原GPU5 TAO四scope队列继续，全部任务仍未获得普通水平稳定结论。

2026-10-16下一实现计划：在原已完成MOT exporter的可复用逐帧导出函数中增加显式RMOT context（默认MOT路径/标签不变），新独立CPU owner等待18个实际query rollout，按固定Compact真实输出逐帧复现并生成Full-v3身份/生命周期/可靠性/最终ID写入及原生membership问题。RMOT物理GT与query mask分开：identity用完整released same-frame boxes的一对一IoU>=.5与不可变birth锚点；membership用候选对query GT任意IoU>=.5；无可见性标注，可靠性/写入使用明确IoU质量代理（.7、分级.5/.7/.85），不冒称visibility。全部原候选、空/拒绝帧、unknown assigned输入保留，unknown无强造target。验证只做真实MOT17-02整clip默认路径与原导出精确对比，以及已采集合法RMOT候选的8帧CPU replay验证新分支，所有新小fixture/groups/refs在已预留48MiB scope+export预算内。新18视频export仅CPU，不改原6000或任何评测。

RMOT export回归首个4404实际exit1，0行写出/0模型训练：validator直接把相对--out传入要求绝对ROOT子路径的export_clip，relative_to(ROOT)失败；原生产调用一直为绝对路径、源/标签未改。最小修复仅validator args.out.resolve()，v1失败目录保留；v2重新执行尚未完成的第一个MOT单元，原seed/loss/GT/阈值均不改。

实际RMOT导出targeted回归v2原43891 exit0：MOT17-02-FRCNN整32帧/992候选，126个native group、5301全部targets/refs与registry在忽略唯一新文件引用路径后逐字段等于已完成旧导出，CPUplan所有counts/max/tokens/0drop精确。RMOT用既有0003 primary真实Ground候选8帧/36候选，在固定当前CPU actor独立replay（不是新Ground采集/新增物理帧）后，195监督行全native CPUplan0drop；所有实际actor outputs/state/native crop pixels精确。用全1814预检执行原官方reader的query mask独立核对membership17yes/19no；另独立IoU+physical Hungarian检查全部36reliability（0/1/2/3=20/5/2/9）。明确RMO可靠性分级仅IoU，NO_MATCH不强降reliability等级；写入仍须IoU>=.7且非NO_MATCH。source scope+全部回归实际6444542<50331648bytes，未新增GPU/模型训练；给新18rollout exporter添加该真实输入验证前置，再仅CPU等待实际源，所有18整clip仍须再次实际native CPUplan0drop后才能进后续SFT。

新18-query原生Full-v3导出owner40644实际启动，CPU-only真实load既有actor后等待源19338，无新增GPU或query帧；所有真实18段仍待采集、逐段重建/0drop后方可入训练。原正式64133最新950/6000有限、原Full刷新21963已575/600，原GPU5 nativeTAO当前第117帧阶段、原Ground全九项及GPU6 source队列继续；不据新组件输入正确性宣称tracking提升。

新增实测资源选择：GPU7一次launch前查询81920MiB total/22665used/58489free，现有FA训练实际峰18.7GiB，足够同卡独立nativeDLM .3 static+ROI并保留额外40GiB启动门槛。不增加第五卡，不借用GPU6 retention服务、不变原6000数据/优化器/任何既有测评。已核实旧RMOT source19338及export40644仍仅CPU等待、0已采query帧/0owned GPU服务/无child，按精确pid/starttick/cwd/script仅SIGTERM这两个尚未开始的数据waiter，先export后source；v1计划/状态/注册和handoff intent全部保留。下一显式v2同数据/actor/parser/decoder/预算新目录在GPU7 co-location，资源竞争导致的速度不可与无竞争实验直接相比。

SharedGPU7 launch前读固定vendor SGLang model_runner源码确认：profile_max_num_token的total_gpu_memory实参来自init_torch_distributed的启动时available GPU memory，并非整卡物理容量；rest=当前free-启动free*(1-.3)，所以.3分配是剩余可用内存的一部分，与existing SFT共存仍有KV空间。保留.3、启动free>=40GiB、原model/decoder/maxrunning1设置，未运行失败试探也未改第三方源码。新collector显式检查原3219708/start47863228 owner及其GPU7 training wrapper命令存活后方可启动；原两个零数据CPUwaiter真实exit均143，旧状态/意图/完成交接记录保留。

SharedGPU7接棒实际成功：source原17236进入真实0000_primary native Ground+ROI+固定CPUactor，export原78069仅CPU等待完整query段。冻结前16帧0..15（16query观测/16物理帧，8帧超旧72范围），候选/精确FP32archive文件逐帧存在、所有runtime GT_read=False；不是仅模型加载/排队，完整18段尚未完成。新的DLM/ROI服务均只GPU7，.3原生KV分配，原6000未停止且最新1050/6000，1000步9029140633bytes完整状态已保存；各原GPU4/5/6训练与测评不动。新数据collection并发计时不作为无竞争tracker FPS；新RMOT监督尚未进任何optimizer。

2026-10-16原Full pair21963实际exit0，两arm各600更新/2400行并完成各两次原生MM重载概率差0；GPU4旧arm四scope worker14226已自动接棒，GPU5原beforeTAO继续。下一只CPU读取新refreshed完整FP32/Adam/RNG状态，核验实际cursor/全部masters和moments有限、600 optimizer steps、BF16 inference导出与masters逐参数对应；旧arm已有完整CPU验证不重复9GB扫描。按真实两份2400行与保存paired-index协议验证每任务800、相同ID/family流和400个native-history替换；不将可重载/finite与数据曝光冒充效果。

2026-10-16首个新RMOT全段已实闭合：0000_primary154帧/289候选/0parser拒绝，原source archive2315104bytes；Full-v3导出1406行allCPU0drop/max2703/2443040计划tokens，全部actor state/output和281crop sheets精确。训练标签membership0yes/289no，identity286IGNORE/3NEW，全部102existing assignments无known GT target，不能把新增146物理帧说成正例覆盖或效果提升。实际原句designated path for people on foot的raw_sentence=pedestrain，同目标map有pedestrians/people等；原DLM输出多为左侧路面/通道框，当前原表达queryGT候选recall=0。新独立组件计划：固定全部154fit帧（含负/空帧）和原GT map，GPU4独立fresh原DLM服务.3，与已运行Full worker同卡但不共用其服务或改变权重/源。3arms各154：原表达fresh复跑、同GT map规范pedestrians、显式原provided person/car类别候选。保持官方parser/dense decoder/seed42/全帧，用posthoc queryGT IoU>=.5 proposal recall定位措辞/前端覆盖，非HOTA/TETA或完整tracker收益；baseline原source保持原样。只bbox/rawwire诊断、0ROI/图像/训练，额外16MiB reserve使global26822863722<26843545600；所有原训练/评测/新18query采集继续。

首个fit0000两query均已实际闭合：source原17236继续下一视频，export78069已完成2rollouts/308query帧但仅154unique物理帧，合计6163行，两个整clip原生CPUplan均0drop/actor state-output及pixels精确；counter原parser拒绝保留在原位置，不能丢掉或用GT补候选。新组件尚未进入optimizer。独立三arm Ground措辞probe原11590已启动自有GPU4服务并实际完成fresh_original前50帧，不借用source/retention服务，seed42/同一原checkpoint/parser/dense设置，GT仍仅后置审计；结果未完成，不先认定规范化有效。

原独立Ground prompt control11590实际exit0，三个fresh arms各完整154frames、0训练/GT候选注入，ownedGPU4服务group已空。原source/fresh原表达154/154 rawwire及box-label-score逐帧完全相同，验证不是服务/GPU变化制造的收益；同GT map规范pedestrians候选覆盖18/21=.8571429，对原0/21，provided person/car也18/21但2182候选、6原parser拒绝帧保留（规范688候选/0拒绝，原289/0）。这是fit154帧proposal recall组件，不是HOTA/TETA/held-out泛化或最终系统提升，原全部18query采集和6000phase均未改。三份全raw预测已无损gzip并逐字验证，全部原raw/结果保留；16MiB probe reserve在退出后按实际data/log/结果关闭，服务模型bundle不属feature预算。下一最小数据修复是复用已完成class候选、保留原query传给actor/JEV，补充当前原表达缺失的正视觉证据；不把GT规范表达当推理可用标签，不修改原候选源或正在训练的数据。

下一已登记最小数据修复：复用三arm probe已完成的2182个provided person/car原生框，全部154原帧/6原拒绝保留，传给同一个固定CPUCompact actor的仍为designated path for people on foot，不用GT规范表达替换用户输入；新独立GPU4 nativeROI-only服务，0新DLM请求/0GT框注入。按实际nativeF32候选+真实自身历史导出相同Full-v3身份/生命周期/原查询membership，全部状态/outputs/crop pixels与nativeCPUplan0drop必须通过，作为未来课程组件不加入正运行6000。原Ground probe已关闭实际2826086bytes，关闭后26808912592；新24MiB archive+4MiB group使26838272720<26843545600，余5272880bytes。原全部模型/评测/18query原采集不改，预留不等于已采集/已训练。

原class-proposal补充owner13102实际exit0，154相同物理帧/2182原native候选/6原拒绝帧保留，0新DLM/0GT框/0query重写/0optimizer；同原句输入的固定CPUactor完整真实轨迹已用NativeFull-v3导出14141行，所有actor state-output及crop pixels相等且全量CPUplan0drop。精确FP32archive21837488<25165824、groups2426218<4194304，ownedROI服务已停。补充实际positive membership而非靠换掉原查询达到数据修复，原各数据源和正跑6000完全未改，所有原Raw/模型检查点保留。该组件只增加相同图像的原生候选证据，不增加unique物理帧或宣称held-out跟踪提升；留作下一阶段共同课程。

当前主6000实际1575步/6300行、1500步完整FP32/Adam/RNG9029140633bytes已原子保存，log finite；新source原17236/export78069已6rollouts=1062query-frame但仅531unique物理帧（相对旧72新增507），17640监督行全nativeCPU0drop；其余12rollouts继续，不混计重复query或class组件相同154图像。新class组件下一轮先列2182条原query membership（18yes/2164no）供语义mix，所有辅助IoU/birth代理标签仍完整保留作诊断，不将unmatched released GT等同完整物理无效证据，不自动把14141行全部注入loss；当前6000输入从始至终未修改。

2026-10-16OV输入支持审计：读实际Base175生成器及24k课程，原105956行gallery人数1..8全覆盖；当前8000 OV行位置0..7分别1846/2476/1958/1004/384/158/84/90，gallery8为336行，其余1..7为462/1474/2308/1714/998/506/202。没有“只训练object0”的问题，不因假设改数据；原生布局定义/像素reader与runtime同一NativeCategoryHead。原old Full600连续3scope已完成：DT HOTA before.08702996->old.13107492，single.02506249->.03071683，multi.06740007->.06682521，TAO及refreshed未完成，不宣称三任务普遍收益。下一效率修复：NativeCategoryHead仅依赖当前image/candidates/vocabulary，与身份/记忆策略无关；新增显式、默认关闭的逐帧原生category replay，读取同一checkpoint完整默认run中的每候选root/leaf全概率，不重新分类/不复制feature archive/不读未来行或身份状态。先在实际beforeTAO前8候选用原Torch参考MM模型复算全1203层级结果，gate选择一致、概率差<=1e-6；再用实际连续2帧完整候选和保存的原生identity决定验证输出/事件/完整memory逐值不变，仅分类计时与来源元数据不同。通过前不启用到任何既有测评，训练/GT/阈值/seed不改。

2026-10-16原category probe69622实际exit0：native600原Torch/MM在GPU4重算首帧8个原始候选，全1203 root及所选conditional leaf的全部输出字段（包括概率）与原source完全相等，maxdifference0<=事前1e-6。连续TAO两帧1/30、1037+990=2027候选以独立CPU tracker、原nativeF32输入及保存identity tape回放，全部输出/events/IDs和完整state_dict数值/feature/图像逐值相等；category replay本身只提取当前帧类别，不取source identity/history，0native category calls。CPU identity明确为原决定tape，不冒称完整两帧fresh GPU重跑；只在probe允许未完成source的已完成prefix，正式CLI仍要求source完整。同ckpt native categories可复用，不能跨权重，保留所有原root和conditional leaf分布而非宣称完整flat1203概率。

原Full三scope独立冻结42493实际exit0：逐frame对原输入/全部empty-reject/counts、ffprobe完整帧数及raw predictions逐字重建；原版pinned TrackEval重新计算全部3单元的HOTA/DetA/AssA/IDF1/IDSW/FP/FN，逐数值等于原official结果。DT HOTA8.702996->13.107492%、IDF15.825959->13.571733%；RMO single HOTA2.506249->3.071683%，multi6.740007->6.682521%。完整新三份predictions无损gzip、各result/official及验证快照冻结，不发布live worker状态；TAO和refreshed仍pending，普通模型水平/全任务收益未达成。

未来6000显式效率协议已登记：launcher对两条精确consumer输出绑定生效，所有其它命令及当前已运行进程不变。final6000 defaultTAO仍fresh计算自身所有类别；assignedTAO使用相同checkpoint/manifest/candidates的已完整default逐帧类别，身份/记忆仍fresh推理，实际CLI追加--category-replay并打印注册来源。未来beforeTAO若原GPU5 baseline尚未结束，由单一blocking wrapper等原owner2467924/start45243513完成后复用完整相同run，避免另算35228候选；仅大immutable trace/predictions/video/finalmemory使用相同inode，mutable result与后续scorer独立，parent照原协议重新official评分。原19428实际exit0：精确binding选中、其它consumer不变、不同checkpoint拒绝；真实完成DT1203/13664候选复用后通过原validate_run，原图/feature零新增、源result/scorer与复用副本inode不同。所有缓存计时只作source provenance，不报告fresh E2E加速或精度收益。上述3份新小诊断cap共2MiB，保守feature上界26840369872<26843545600，剩3175728bytes；原模型/数据/检查点零删除。

主6000冻结实际2100updates/8400已消费问题，已原子保存至少2000步完整FP32/Adam/RNG；loss/gradient finite。新RMOT冻结前10已完成query rollouts、2382query观测但1191unique物理帧，相对旧72新增1151物理帧，46658监督行全nativeCPUplan0drop且actor state/output/pixels exact。两条表达不重复算物理帧，全部新RMOT组件仍未加入正运行6000，不将数据准备算训练曝光或跟踪收益；原唯一训练/采集/评测handles保持运行，整个研究goal未完成。

2026-10-16扩大联合训练的最小研究计划：现有stateful与paired sampler入口实际硬限world1，不能将原生SFT的分布式代码存在冒充已具备四卡完整恢复能力。本轮增加显式--distributed-sft，沿用上游每row全局均值loss与FP32梯度SUM，不改原world1协议；保存4rank各自Torch/当前CUDA/Python/NumPy RNG，rank0写完整模型/Adam/scheduler/cursor，所有rank在保存处同步。用同一8个真实多任务MM问题、两次global4-row更新，分别实际执行单卡连续、四卡连续、四卡step1完整保存后退出并恢复step2；不把短probe当大规模训练。事前gate：单卡/四卡全部FP32 master maxdiff<=1e-6，完整两Adam moment向量relative L2各<=1e-4；四卡全部rank模型/moments逐值相等、连续/恢复完整state逐值相等，CPU实际输入计划0drop/同global row次序。仅物理4/5/6/7 co-location，启动前各free>=32GiB，不停止现有任务；模型/优化器proof保留在/data3/user/LocateMOT-checkpoints，diagnostic另预留1MiB使保守feature upper26841418448<26843545600。当前6000和所有既有评测均不启用新分布式flag；probe通过前不发布大型四卡训练已启动或收益。

四卡probe v1原36204真实exit1：单卡2步、四卡连续2步、四卡step1完整保存退出及restore step2四个子单位均真实exit0；每一步所有4rank的320模型tensor与两份完整Adam moments共2257179072元素broadcast逐值检查maxdiff0，实际global8rows/2steps和全部token长度一致。但最终严格恢复比较失败，未通过：single/four全部FP32 master maxdiff3.8389e-6、完整Adam m/v relativeL2 .00535875/.00179744；continuous/restored maxdiff3.8494e-6、m/v .00531712/.00196006。三次fresh首步loss完全相同，但梯度norm157.9769/158.0080/158.0417已不同，差异先于任何restore，不能归因于checkpoint或单独认定FLA。v2原92390改用原Torch参考训练，同8rows/seed/loss/gates，四子单位仍各exit0且全rank状态相同，原owner最终也exit1于恢复模型逐值比较；fresh首步相同loss4.830878而norm159.4287/159.3847/159.4403，进一步排除FLA专属原因。所有失败模型/完整state/日志和原gate保留，未推广。

下一针对性执行控制：v3显式Torch deterministic algorithms/cuDNN deterministic/CUBLAS4096x8参考后端，原46094运行；v4原51437进一步验证带同样确定性设置的FLA，并显式把已登记seed0应用到Torch/CUDA/Python/NumPy全部全局RNG。源码确认原生full_finetune仅对每行rendering使用seed，torch.manual_seed原仅在LoRA分支；跨fresh进程初始全局RNG并不相同，必须在比较独立continuous/restore轨迹前统一，不改变登记seed0或样本/阈值。所有新flag默认False，原正运行6000及其它recipe不变。v1/v2已经terminal的diagnostics按实际94447/98034bytes各加16KiB bookkeeping关闭1MiB cap，零删除；额外v4一份1MiB后保守feature upper26842692273<26843545600。

新RMOT监督审计已实际CPU完成（first11 immutable单位）：2676query观测/1485unique物理帧、7926候选、50545全监督行；membership2040yes/5886no，加19条empty-presence为7945条可入下一语义课程。14042个TERMINATE标签中至少12105条可由registry证实birthGT未知；4399个unknown-birth既有assignment全部正确无强造assigned-update target，其中1251个当前候选已与released GT匹配。未知出生不等价永久离场，故不把整50545行盲加loss，先只使用query membership/presence，其余原标签完整保留作后续受控研究；不修改原正运行6000。新增source-list训练入口只读取immutable原gzip/group refs，不复制feature/图像/历史；真实8个MM问题all state/media/question/target/ID逐字段相等、原native token长度相等，实际CPUplan0drop/max26038，semantic filter2行正负均保留，原80140 exit0。

已启动唯一CPU full-source准备owner84666，等待原exporter3320470/start48408216完整18单位真实exit成功，随后冻结全部MOT252258行、原RMO87721行、全部Base175 category105956行、新18段语义行及原query class补充2182语义行的source-list。当前已验证first11支持的合计下界456062问题，不把等待/manifest准备称训练或完整corpus0drop证明；未来自然源比例明确不同于当前24k均衡课程。仅使用此前已预留16MiB未来课程budget，0新GPU/0optimizer；下一大训练仍需通过分布式数值/恢复验证和全实际输入计划，未自动启动未验证方法。

确定性控制v3原46094最终真实exit1，仅剩全局RNG equality：全部模型/Adam恢复逐值相等，single/four全FP32 max7.45e-9、两Adam moment relativeL2 6.93e-8/8.12e-8均过原门槛；4rank初始Python/NumPy/Torch/CUDA RNG各与独立fresh连续run不同，符合原full-finetune未统一全局seed的源码发现。v4原51437真实exit0，显式同seed0/确定性FLA：全部752393024 FP32参数single/four max3.7252903e-9，两完整Adam moment relativeL2 1.24950e-8/2.17795e-8，全4rank全模型及两Adam逐值差0；continuous/restored全部模型、moments、scheduler和所有rank RNG逐值相等，8真实MM rows/native token lengths/全局步骤完全匹配。没有放宽任何原gate，前3份失败记录/全部model和state保留。原45393真实exit0：四卡训练BF16导出由原Torch MMengine两次独立加载，474weights/原fork auto-check不变，重复概率差0；实际2updates不冒称foundation大训练或tracking收益。

修正真实大epoch最后一步不一定落在save_every边界的完整state保存：仅显式distributed协议，在最后非定期保存步所有rank收集自己的RNG、rank0原子保存最新FP32/Adam/scheduler/cursor；原普通world1路径不变。原13813真实exit0，真实相同8rows/2updates而save_every3从未触发periodic，仍保存final2/seen8/next2完整状态；原49373 CPU实际读取该9GB state，与v4原single的全部320 FP32 master及所有Adam moments/scheduler逐值相等。此控制不是额外训练预算或性能提升。

实际18-query全部闭合：原source17236、export78069均exit0，原自有DLM/ROI groups3320233/3320234全空；68567监督行逐段native CPUplan0drop，所有actor state/output/pixels exact。原69340独立全量验证exit0：18视频ffprobe共3628帧，逐帧原始精确F32候选对齐、所有empty/parser拒绝保留、全部raw predictions逐字重建并无损gzip，实际archive bytes111683885一致；unique1814物理帧覆盖旧72并新增1742。当前下一课程只取其中10910语义问题，辅助unknown-birth proxies未盲入loss。384MiB source archive cap在真实source退出并验证后关闭到111683885bytes，feature保守上界26551722974<26843545600、余291822626bytes，零删除/零新feature archive。

完整源清单原84666真实exit0：64个immutable源文件、459027问题=252258 MOT/100813 RMOT/105956 OVMOT，未复制原native特征/图片。新四卡大epoch owner94290（3457625/start49350320）已实际启动且live exact identity验证，进入FULL_NATIVE_CPU_PLAN，仍0optimizer；须全部实际native MM行/四rank global16计划0drop后才能启动四卡FA。同原6000一样从native600完整FP32 masters起点、独立新optimizer/scheduler/RNG，1整epoch/seed0/lr2e-6/vision frozen/global16/同一个final模型测全部任务，明确新自然源比例而非原24k均衡recipe，不声称只隔离了规模因果。保留/data3模型/checkpoints，训练前需150GiB空间及4/5/6/7各free32GiB，和所有既有任务共享总共四卡；无Ground-One端到端梯度或长视频BPTT。actual filtered-media曝光须CPU报告实测，不用raw1814源帧替代optimizer曝光。终评仍同4固定scope、原scorers/阈值/全empty-reject，TAO第二策略显式同ckpt类别复用。原主6000冻结2625updates/10500已消费问题、至少2500完整state已保存，输入未改；全部tracking/普通水平目标仍未完成。

2026-10-16下一瓶颈诊断计划：上一goal turn是真实progress（四卡严格数学/完整恢复/native inference通过、18-query全部完成、459027大epoch已注册CPU检查并发布15a02bb），原94290/64133/14226/46228/52020 handles本轮均确认live；当前大epoch仍FULL_NATIVE_CPU_PLAN，主6000已2825。读实际Full shortlist和tracker：Full身份选项纯几何top32；Compact已有固定hybrid16_16，尚只有2-frame功能证明；lifecycle即便选择LOST仍在lost_age>3s硬删除。新CPU诊断只对已完成MOT20自然1657frames/235232candidates固定reference_gated3轨迹，使用同一已留FP32候选、保存decisions逐frame因果恢复，比较geometry32与已有固定hybrid16geometry+16appearance的正确surviving birth-anchor覆盖。appearance严格复用原Compact公式（三份native memory cosine均值、组内max），不读取GT来选候选，不改变原轨迹/seed/指标/TTL，不把coverage当HOTA或新模型收益。仅posthoc读取原完整association diagnostic；保留空/拒绝帧，0GPU/0新feature archive；新report预留1MiB使feature upper26552771550<26843545600。先用覆盖差识别检索是否是真实瓶颈，再决定后续长history训练组件，不盲改当前459027或6000课程。

原40304实际exit0：完整1657frames/235232candidates/117原拒绝帧，每一步原输出/events/state精确因果恢复。25046个存在known surviving birth anchor的候选，geometry32正确ID覆盖19195=.76638984，固定hybrid32覆盖23213=.92681466，增加4292、丢失274；age1..31 8574/12080->11188/12080，age32..127 6644/8936->8021/8936，age128+3977/4030->4004/4030。仅固定轨迹candidate-option oracle，不是counterfactual准确率；已删除ID不在分母。按已有固定16/16定义启动原73681唯一完整CPU tracking control，1657全帧/同235232候选、同checkpoint/原gated-all/lost3s/原official scorer，不改TTL、不调阈值、不新增GPU/原生特征档案。另预留16MiB小报告/无损预测包，feature upper26569548766<26843545600，完整输出和唯一final identity memory属科学结果/状态检查点，逐帧不复制原feature vectors。

同时读已完成Full3scope：oldDT正确身份选项3934/3934=100%，但实际fixed-birth正确关联407/3934；single67/67、multi534/534也全覆盖但正确6/67、68/534。DT候选GTcoverage4341/4586=.9466，RMO原query coverage10/18和1133/2112，不能把MOT20检索不足当成所有Full低分的唯一原因。old multi已匹配物理GT上的membership TP277/FN855/FP58/TN49，recall.2447；这些是posthoc组件指标，非HOTA替代。

下一零GPU数据修复计划已登记并启动原56683：利用已完成18段的68567原生Full-v3 rows/真实source histories，为后续课程挑选已知不可变birth的RMO identity及final-ID write监督。只保留当前GT对应且actual shortlist中唯一正确birth anchor的原EXISTING label；NO_MATCH须证实有surviving正确anchor而非短名单内；multiple-correct ambiguity单独计数不盲用随机alias下不可区分的canonical硬标签。Q4须currentGT与被分配birth都known，并独立由原reliability等级>=2（IoU.7代理）和GT一致性核对原target；unknown assignment保持input且不造negative。state/media/questions/targets全部仍引用原group，来源birth只作离线certification，不进入模型input；报告实际filtered媒体/anchor跨度，最后所有入选rows做原生CPU计划0drop。该新组件不加入已冻结459027或正跑6000，0新Ground/ROI/原生feature/optimizer；8MiB预留使upper26577937374<26843545600。先查看真实合格数据支持，再决定下一训练，不先宣称长时身份能力或模型收益。

原73681完整CPU hybrid control实际exit0：同1657帧/235232精确原候选、每帧metadata/timestamp完全相同、原117reject保留、同checkpoint。原版official TrackEval HOTA7.431603->8.275033%、DetA9.635711->10.759073%、AssA5.741634->6.368904%、IDF16.147357->6.668855%；IDSW2060->2348、FP463->556、FN286273->282277。HOTA+0.84343pp但IDSW恶化，native candidate-gap3..30s known-birth12例仍0恢复，不能说检索解决全部问题或普通模型水平已达到。新轨迹改变birth/分母，不能把固定轨迹25046覆盖率直接等同实际counterfactual HOTA因果。保留完整科学输出、posthoc恢复和原始预测无损包，0GPU/0optimizer；未自动推广Full或变任何现有队列。

原56683 actual exit0：逐candidate核对原18段10868个registry/current GT、source actual shortlist/真实birthframe与native crop provenance、已知assigned identity；选出1428条唯一正确EXISTING（另271条multiple-correct ambiguity不纳入本新subset，所有旧row完整保留）+473yes/981no已知birth的Q4，共2882rows/977实际filtered媒体路径。原生全部CPUplan0drop/max17164、13942000计划tokens，仍0optimizer。anchor跨度identity<1s317/1..3s384/3..10s513/10..30s214；Q4<1s281/1..3s344/3..10s584/10..30s235/>30s10。state/media/question/target保持原始refs，复用原18段全部native pixels/真实actor轨迹验证，不将metadata GT证据加入模型state。此数据补充RMO任务的已知身份关联/写入，而当前459027内新RMO仅语义family；未改变正在进行的两轮训练，后续须结合6000实际测评再决定受控续训。

2026-10-16全量CPU规划效率修复计划：已读原full-source owner，459027条原生MM长度检查固定world1，输入仍在检查而非已开始四卡optimizer。新增独立CPU-only Gloo4入口，调用完全相同train.sft.token_lengths/原render seed/media counter/paired index shuffle和global16 plan；先对64个immutable source各首/中/末实际行做serial-vs4长度逐值相等gate，再对全部行原生规划且0drop，保存int32长度约1.84MB供后续显式复用研究。全量检查通过前不切换原owner；原GPU训练/测评、原serial检查均继续，0新GPU或模型权重。小报告与长度另预留4MiB，总feature上界26583180254<26843545600；不将CPU计划/短probe当大规模联合训练或长视频BPTT。

四CPU原192条native MM数值gate已实际通过：覆盖64源各首/中/末，所有token长度serial与Gloo4逐值相同，全459027检查仍运行。新增显式--validated-native-lengths入口：只允许同source manifest/immutable源文件、tokenizer checkpoint、seed/epoch0、augmentation/media/length与global batch配置，原生batch/index shuffle/rank deal保留；默认入口不变。独立真实8条MM regression实际exit0，所有长度/微批/四rank step分配完全相同，改seed拒绝；0model/GPU/optimizer。full-source owner新增可选已完成plan重验证路径，尚未切换原owner或宣称full pass。报告/新probe沿本轮4MiB预留，原6000最新4225正常。

下一自动接续计划：新增单一blocking coordinator，等待原49885实际exit0/full459027原生长度完成，复用入口另做全量CPU重验证并要求所有summary完全相同；仅原owner仍在CPU且GPU训练路径不存在时，先冻结并确认原parent状态T，再确认其exact CPU child/CUDA为空，保留handoff intent和全部旧日志，终止尚未完成CPU子单元。原parent记录真实-15/assertion退出不伪造成功；同原459027/seed0/global16/原生600FP32初始化/评测协议的新owner接续，省去重复tokenize，原GPU训练与测评不停止。若原owner已进入GPU则不切换。新独立first5 observer显式绑定新owner/config，旧observer保留原终止证据。额外元数据+observer共1MiB，feature upper26584228830<25GiB；不改变样本/metric/seed/门槛，不将等待/数据计划当训练完成。

4500实际里程碑核验计划：原64133已输出4500/18000及9029140633bytes完整FP32/Adam/RNG保存。仅CPU读取同一个atomic latest state（不复制9GB），核验320 FP32 masters/两Adam全向量finite及全部moment step/scheduler/cursor4500，读取冻结24k语料按原seed0 index-shuffle计算真实已消费任务/unique source row/首媒体路径曝光，明确重复问题不等于独立帧，仍无6000终评。小报告另预留512KiB使feature upper26584753118<25GiB；原全部训练/测评/CPU队列继续。

4500 checkpoint CPU完整核验原78686 actual exit0：全部752393024 FP32参数、1504786048 Adam moment元素finite，320optimizer step/scheduler/cursor4500与seen18000一致；185条实际日志loss/grad finite。真实sampler消费MOT5957/RMOT5977/OV6066问题，实际首媒体路径MOT980/RMO72/OV3064。首版曝光metadata把课程重新编号ID当source ID，原记录错误计数保留在exposure_count_correction；已改用supervision.source_row_id单独重算（不重复扫描9GB状态、不改训练），课程ID与source ID分列。实际unique source rows为{"rmot": 5677, "mot": 5957, "ovmot": 5624}。不称独立轨迹数量或6000完成/跟踪增益；完整latest4500已验证但未来5000会原子替换，编号BF16权重保留。

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

## 2026-10-22 — G2 terminal basis and bounded native OVMOT promotion

按最新 GroundJEV 单主干任务书重新核对：G2 Clean 已终态，不再继续 G2 正式训练。
原 64 帧 TAO 注册子集导出在 8 个完整帧记录后安全收尾，四个 shard 的部分输出和
服务日志均保留在 `tao_native_ovmot_train_subset64_v1`。新增
`promote_completed_tao_subset.py` 只验证并提升已经 flush 的 8 帧/8 视频，未填充
缺失帧、未读取测试标签、未改动候选框；产物明确标记
`registered_train_subset=true`、`full_train_image_scope=false`，不作全量 OVMOT 结论。
该子集经官方 TRAIN annotation 离线一对一 IoU join 得到 58 行 native OVMOT 监督
（29 group、29 leaf），与 131214 MOT、346 RMOT 行组成联合 manifest；manifest SHA256
为 `cc632115aa2b87161dc7b8413d08c3c471e1225179559a76509e7366190764c1`。

## 2026-10-22 — GroundJEV final-G2 coupled readout pilot

使用最终 G2 adapter step 18227（adapter SHA256
`cf1b5a6f626810ead94eb4e16adaf3feb6c6da3f99ca5023b6b5c234726f5742`，base model
SHA256 `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`），在
GPU3 以单主干 native DLM forward 运行有界 240-step pilot。训练只更新共享 JEV
readout、Yes/No、生命周期/记忆投影和 OVMOT category heads，G2 backbone 冻结；
没有加载独立 OneJev 或 YesTrack。由于 MOT 行最多含 80 个候选，native DLM 输入
采用有审计的 16 框竞争窗口，始终保留目标候选；19 行发生裁剪，原始行和策略均
保留在 receipt。

终态 owner：`groundjev_native_three_task_v1_retry_v6_gpu3`，状态
`GROUNDJEV_NATIVE_THREE_TASK_TRAINING_COMPLETE`；实际有效更新 MOT/RMOT/OVMOT
各 72 次（请求 240 步，其中 24 个 identity target 因历史 slot 缺失而合法 mask），
unique visual inputs=67、native visual forwards=67、peak allocated memory
约 9.35 GiB、耗时 74.16 s。checkpoint 为
`/data3/user/LocateMOT-checkpoints/groundjev_native_three_task_v1/groundjev-native-three-task-step000240.pt`。
before/after 只是在同一 72-row sampled pilot 上的 readout accuracy：MOT
0.4444→1.0、RMOT 0.6667→0.875、OVMOT 0.0417→0.7083；这些是训练集小样本拟合
证据，绝不是视频级 HOTA/TETA 或泛化结论。下一步必须使用该 checkpoint 做固定候选
视频 replay 和独立 held-out 评测。

## 2026-10-22 — Coupled pilot launch repairs

记录并保留三次未更新失败：首次因 `run_with_resources.py` 缺少 executable bit，
随后因终态 G2 adapter 使用 `state` 而旧 loader 只读取
`model_trainable_state`，再随后因全量 MOT 候选串入 native DLM 触发 CUDA illegal
memory access。最小修复分别为显式 Python 调用资源 runner、严格兼容两种已登记 adapter
格式、以及记录并限制 native DLM 每次最多 16 个竞争框；G2 正式目录和历史结果未改动。

## 2026-10-22 — Final-G2 coupled video diagnostic and independent queue

为使终态耦合 checkpoint 能进入已有视频级入口，修复固定候选评测器同时读取历史
`model_state` 和本轮 `groundjev_model` checkpoint schema，并保留 OVMOT heads 与单
主干约束。最终 G2 step18227 的 coupled readout 在 MOT20-03 的不可变 128 帧缓存上
完成 TrackEval：HOTA=0.015177、DetA=0.020226、AssA=0.011422、IDF1=0.016006、
IDSW=182，固定候选回放 13.265 FPS。该缓存属于注册 TRAIN scope 且早于终态 G2，
因此只登记为关联/runtime 诊断，不作 G2 独立泛化提升。

GPU7 的独立留出 owner 仍被外部作业占用；在不终止外部进程的前提下，新增 GPU3
owner `g2_clean_independent_gpu3_v1`，运行相同 G0/G2、MOT/RMOT/OVMOT、解析器、
decoder、tracker 和 scorer。owner 已进入 G0 MOT20-05 完整验证，终态官方指标尚未
生成；在 receipt 之前不宣称 held-out 结果。

## 2026-10-22 — Longer coupled run GPU5 failure and bounded retry

为避免把 240 步可行性 pilot 当成联合训练完成，启动独立的 3000 步、每任务 192
行、同一 16 框上限的 G2 三任务耦合 run。GPU5 owner 在 native DLM 预编码阶段
`optimizer_updates=0`，首个可操作错误是 Qwen3 MLP 内的 `CUDA illegal memory access`；
日志记录 GPU5 当时存在外部 PRISM 计算 PID、但显存仍有 65029 MiB。没有 checkpoint
或指标生成，失败目录和完整日志保留，未重试 G2。

在 GPU6 上对同一 442 个实际选中输入做带行号的前向诊断，前 147 个输入连续完成，
未发现某个候选框或图像格式的立即错误；诊断受 600 秒上限收尾，不能作为性能结果。
随后按相同 manifest、seed、rows、steps、候选上限和 optimizer 合同将 3000 步 run
迁移至 GPU6，输出与 GPU5 完全分离，当前等待终态 receipt。

GPU6 的未分块和“每 64 个输入重建 provider + CUDA 同步”两次重试都在 native
预编码阶段、optimizer_updates=0 处触发同类 illegal access；没有产生 checkpoint。
因此 3000 步 run 暂不作为结果，分块实现和三份失败 receipt/log 均保留。当前有效
证据仍是 GPU3 无外部计算时完成的 240 步 pilot；后续长 run 等独立 GPU3 留出 owner
终态后再在同一空闲卡恢复，避免继续在共存 GPU 上消耗失败实验。

## 2026-10-22 — G2 terminal state and current independent held-out run

按单主干 GroundJEV 任务书重新对齐后，G2 Clean 已在 step 18227 终态完成并完成
独立 merged bundle 导出；不再修改 G2 optimizer、scheduler、sampler、RNG 或历史
权重。当前 GPU3 owner `g2_clean_independent_gpu3_v1` 继续执行固定 parser、decoder、
tracker 和 scorer 的独立留出链，先跑 G0 控制再跑 G2 与 RMOT/OVMOT scope。最近核验时
`mot20_05_validation1657` 仍处于 G0 native perception（日志已到 frame 1997 的
处理入口，尚未生成官方 receipt），因此暂不报告任何 G2 独立泛化指标。held-out
完成后由同一 owner 触发隔离的 3000-step 三任务耦合恢复队列；不重启已经完成的 G2，
不把小样本 240-step pilot 或缓存 replay 诊断写成视频级性能提升。

## 2026-10-22 — Held-out owner remains live after G2 completion

复用同一 GPU3 held-out owner 等待了一个完整阻塞窗口并重新核验：owner
`g2_clean_independent_gpu3_v1` 仍为 `RUNNING_G0_mot20_05_validation1657`，
`failed=null`，G0 服务和视频推理进程均存活；日志已推进到 frame 2015 的
`native_perception` 入口。MOT20-05 validation manifest 的完整范围仍为 1657 帧，
尚未产生官方 G0 receipt，因此没有把当前帧数写成 HOTA/IDF1 结果。资源快照显示
GPU3 的 GroundAnything 服务约占 18 GiB；GPU4/6/7 均有外部计算，未启动并行副本、
未终止共存任务。G0 完成后 owner 仍会按注册顺序运行 G2、RMOT 和 OVMOT，之后才
触发隔离的 3000-step 三任务耦合恢复。

## 2026-10-22 — Held-out G0 control still progressing

同一 owner 再次经过阻塞等待后仍然存活，状态保持
`RUNNING_G0_mot20_05_validation1657`、`current_model=G0`、`failed=null`；
日志已推进至 frame 2036 的 native perception（frame 2035 已完成 association，
单帧记录约 12.98 s）。没有生成官方 metrics/receipt，也没有提前切换模型或修改
评测范围。该进度是 G0 控制的运行证据，不是 G2 性能结果；G2/RMOT/OVMOT 和后续
耦合训练仍由同一 owner 按顺序执行。

## 2026-10-22 — Parallel G2-only held-out owner launched on GPU5

G0/MOT20-05 GPU3 留出仍在长时运行（不重启、不改协议）。为缩短终态比较等待，
新增 `--models` 与显式 `--allow-shared-gpu` 入口，保留相同 held-out overlap audit、
official parser/decoder/tracker/scorer 和逐帧合同；在独立输出目录
`g2_clean_independent_gpu5_v1`、端口 8224 启动只评测终态
`GroundAnything-DLM-G2-CLEAN-V2` 的 owner。GPU5 启动快照为 65029 MiB free，外部
PRISM PID 3199661 占 16116 MiB；共享模式只在仍有至少 32768 MiB 空闲时放行，未
终止外部作业。owner 已通过 G2 terminal bundle gate，状态进入
`RUNNING_G2_CLEAN_V2_mot20_05_validation1657`；G0 owner 和历史 G2/G1 checkpoint
保持不变。该并行输出将在独立 receipt 产生后与 G0 结果按相同输入范围比较。

## 2026-10-22 — Legacy owner audit while held-out runs

只读核验 G0/G200/G1、旧 OneJev/MOT-only 和此前三任务尝试的 owner PID：所有
终态、失败或暂停 owner 的记录 PID 均已不存在，未发现旧 MOT-only 长训练或独立
OneJev 训练仍在运行；G1 GPU5 适配保持终态 `G1_GPU5_ADAPTATION_AND_CONTROLS_COMPLETE`，
旧 `real_three_task_joint_native_v1_queue` 仍为 `PAUSED_FOR_GROUNDJEV_SINGLE_BACKBONE`。
当前唯一 GroundJEV 主线运行者仍是 GPU3 的独立 G0/G2 held-out owner，故没有清理
历史 checkpoint 或覆盖对照结果。

## 2026-10-22 — Held-out G0 owner remains healthy

再次等待并核验同一进程后，GPU3 owner 仍处于
`RUNNING_G0_mot20_05_validation1657`，`current_model=G0`、`failed=null`；日志
已推进到 frame 2052（frame 2051 完成 association，记录耗时 11.33 s）。当前仍
没有官方 G0/G2 receipt 或指标文件，故不提前报告独立性能。进程句柄和服务保持原样，
后续 G2/RMOT/OVMOT 及 3000-step 耦合恢复队列不变。

## 2026-10-22 — Held-out throughput decomposition

只读性能分解显示当前慢点来自 G2 native DLM 推理而非视频编码：截至本次检查，
G0 video pipeline 中位单帧约 17.14 s（605 帧），GPU5 G2-only 约 41.48 s（64 帧），
GPU6 G2-only 约 49.03 s（32 帧）；服务端 PREFILL 本身仅约 20--38 ms，长耗时来自
官方解码/候选生成和逐帧协议。未降低 token/候选合同或把部分帧升级为正式指标，三路
owner 继续保留完整输出。

## 2026-10-22 — Parallel G2 owner entered real inference

阻塞等待后重新核验：GPU3 G0 owner 已推进到 frame 2159，GPU5 G2-only owner 已
真正进入同一 MOT20-05 manifest 的 native inference（frame 1676），两者均为
`failed=null`，尚无 official metrics。GPU5 外部 PRISM 仍在运行，所以 G2-only
吞吐低于 GPU3；保留该并行 owner 作为独立证据，不把部分帧数当成评测结果。

GPU5 共享运行受外部 PRISM 负载影响较大，故在 GPU6 即时探测到约 64.8 GiB free、
utilization=0% 且仅有 16.4 GiB 外部进程后，又启动独立 G2-only owner
`g2_clean_independent_gpu6_v1`（端口 8225）。该 owner 同样通过 bundle/overlap
审计并进入 MOT20-05 native inference（frame 1662），GPU3/GPU5/GPU6 三个输出
目录互不覆盖，外部 PID 均保留；最终只采纳产生完整 receipt 的运行。

## 2026-10-22 — G2 Clean held-out queue scope audit

复核 `queue_g2_independent_heldout_eval.py` 后确认，GPU5/GPU6 的 G2 Clean owner
在 MOT20-05 完成后会在各自 GPU 上继续执行同一套固定协议的 RMOT single、RMOT
multi 与 OVMOT TAO scope；两个 owner 彼此并行，输出目录和 receipt 分离。当前
RMOT scope 各为 800 帧，OVMOT TAO video 105 为 40 个标注帧，三份 manifest 均为
独立验证范围且 `training_on_this_scope=false`。因此不再重复启动额外进程，也不
提前把部分帧数写成指标；完成后仅采纳有完整 inference、scorer、frame-contract
和 official-metrics receipt 的结果。

G2 Clean 的 583,251 行 TRAIN manifest 还包含 AerialMind、AnimalTrack、DanceTrack、
MOT17、MOT20、RMOT26、Refer-KITTI-V2、SportsMOT 和 TAO-Amodal Base。当前注册的
其它域多数只有 TRAIN 媒体或标签；例如 MOT17-02-FRCNN 与 G2 TRAIN 有 600/600
张精确图像重叠，不能把它当成独立测试。因此本轮只把已完成 overlap 审计的
MOT20-05、Refer-KITTI-V2 RMOT 0007 两个表达式和 TAO video 105 纳入独立队列，
其它域待有合法 held-out split 和评测标签后再加入。

## 2026-10-22 — Domain-held-out continuation owners registered

为覆盖 G2 Clean 的其它合法验证域，新增并审计两个独立 scope：完整
DanceTrack validation `dancetrack0004`（1203 帧）和 AnimalTrack 官方 test
`chicken_1`（306 帧）。两者对 G2 TRAIN 做了精确 image-path 与 video-level
审计，均无重叠。GPU5 continuation owner `g2_domain_dancetrack_gpu5_v1` 和
GPU6 continuation owner `g2_domain_animaltrack_gpu6_v1` 已以独立进程注册，当前
等待对应 MOT20/RMOT/OVMOT owner 终态；终态后各自启动 G2 Clean 服务、完整视频
推理和官方 TrackEval，输出不覆盖已有结果。

对 runtime `evaluate_video.py` 做了 1 帧 AnimalTrack scorer 回归：使用真实
`chicken_1` GT、空预测和 `animaltrack` protocol，TrackEval 正常生成
`OFFICIAL_SCOPE_EVALUATION_COMPLETE`（FN=20，HOTA/IDF1=0 仅为格式回归，不能
解释为模型性能）。因此接力 owner 的完整 306 帧 scorer 路径已具备可运行合同。

新增 `collect_g2_heldout_receipts.py` 并运行一次当前状态汇总：G0/G2 MOT20
owners 均仍为 RUNNING，两个域扩展 owner 均为等待对应 owner 终态，当前
`accepted_scope_count=0`。汇总器记录了 G2 bundle SHA 和 TRAIN manifest SHA，
只接受完整 official receipt，不将运行中帧数升级为指标。

新增并启动 `queue_g2_heldout_finalize.py` 作为一次性终态收口 owner。它内部
等待 G0、两路 G2 Clean 和两个域扩展 owner 全部进入 COMPLETE/FAILED，再运行
receipt collector 写出最终汇总；当前不会改变任何 GPU 任务或推理协议。

## 2026-10-22 — Immediate co-resident domain validation start

用户要求不等待 MOT20 owner 结束，立即并行验证其它 G2 域。一次性资源探测显示
GPU5 空闲 62,919/81,920 MiB、GPU6 空闲 46,551/81,920 MiB，均高于域验证队列的
32,768 MiB 门槛；现有 GPU5/GPU6 MOT20 owner 及其服务未停止、未覆盖。清理了仅
处于 WAITING 状态的旧域接力进程，并把 `queue_g2_domain_heldout_after_existing.py`
增加为 `--start-now` 模式：保留当前 owner 证据，记录 `parallel_started=true`，
使用独立端口 8230/8231 和独立输出目录立即启动 DanceTrack `dancetrack0004`
（GPU5）及 AnimalTrack `chicken_1`（GPU6）。两路 owner 已进入
`WAITING_G2_CLEAN_V2_SERVICE_GPU5/6` 的服务启动阶段，当前 MOT20 输出仍受保护；
最终只采纳完整 inference、scorer、frame-contract 和 official-metrics receipt。

启动后回归核验：两路 `serve_groundanything.py` 均已存活并出现真实 `[PREFILL]`
请求，owner 已分别进入 `RUNNING_G2_CLEAN_V2_dancetrack0004_val1203`（GPU5，
service PID 110861）和 `RUNNING_G2_CLEAN_V2_animaltrack_chicken1_test306`
（GPU6，service PID 110862），`failed=null`。这只证明并行推理已开始，不把中间帧
或服务日志当成正式指标；待完整逐帧推理与官方 scorer receipt 结束后再收录结果。

## 2026-10-22 — GPU7 native three-task expansion

GPU7 一次性资源探测为 79,599/81,920 MiB free，只有约 1.5 GiB 外部计算占用。
原先仍停在 `WAITING_G2_CLEAN_BUNDLE_AND_GPU7` 的旧 held-out 等待 owner 没有启动
服务；已先保存为 `owner_state_pre_gpu7_native_expansion.json`，再将其安全暂停，
避免它在 GPU7 上与新实验竞争。随后启动独立的单卡真实 GroundJEV 三任务扩展
`groundjev_native_three_task_expansion_gpu7_v2`：使用最终 G2 adapter step 18227、
同一 native MOT/RMOT/OVMOT manifest（SHA256
`cc632115aa2b87161dc7b8413d08c3c471e1225179559a76509e7366190764c1`），每任务采样
72 行、1200 个 optimizer steps、16 框竞争窗口、单一冻结 G2 主干。第一次尝试在
启动 `run_with_resources.py` 时因脚本缺少 executable bit 立即失败（optimizer_updates=0，
未生成 checkpoint）；已按首个可操作原因改为显式 Python 调用并以全新 v2 输出目录
恢复。v2 owner 已进入 `GPU7_NATIVE_THREE_TASK_EXPANSION_RUNNING/ TRAINING`，不改动
GPU3/5/6 的独立 held-out 输出；完成后只接受包含 MOT、RMOT、OVMOT 三类更新和
完整 receipt 的 checkpoint。

## 2026-10-22 — GPU7 native expansion illegal-access repair and v3 restart

GPU7 v2 首次真实 1200-step 扩展在 native feature encoding 的第 34 个 unique 输入
触发 CUDA illegal memory access；逐输入上下文定位到
`native_rmot_0008_3_q1_batch0_membership_2`、MOT/RMOT/OVMOT native manifest，错误
栈落在上游 DLM `create_block_mask` 的动态 Inductor/Triton kernel，而不是 JEV
optimizer。单帧 GPU7 forward/backward 回归通过，证明 checkpoint、adapter 和设备
本身可用；4 行/任务、12 steps 的 chunk=64 回归也通过，但暴露了 receipt 中已删除
`bridge` 的 NameError，已改为读取 `feature_audit.adapter_step`。

按上游 `hybrid.py` 已提供的稳定路径，native provider 现在默认设置
`GAM_DLM_STABLE_BLOCK_MASK=1`，并让 GPU7 owner 同时启用
`CUDA_LAUNCH_BLOCKING=1`。修复后的 24 行/任务、240 steps 回归在 GPU7 完整通过：
71 个真实视觉输入、MOT/RMOT/OVMOT 各 24 行，optimizer updates 分别为
68/72/72，checkpoint 为
`/data3/user/LocateMOT-checkpoints/groundjev_native_three_task_regression_gpu7_rows24_stable/groundjev-native-three-task-step000240.pt`。
这只是稳定性与联合更新回归，不作独立性能结论。随后以相同 manifest、G2 step18227、
72 行/任务和 1200 steps 启动独立 `groundjev_native_three_task_expansion_gpu7_v3`，
当前已进入 `TRAINING`；v2 失败目录、日志和 receipt 缺失事实均保留。

修复后的一次状态读取显示 G0 MOT20-05 已写入 1633/1657 帧，G2 GPU5/GPU6 分别
写入 656/375 帧，DanceTrack/AnimalTrack 域扩展分别写入 368/147 帧；五个 held-out
owner 仍为 `failed=null`，GPU7 v3 仍在 TRAINING，尚未生成最终官方指标或三任务
扩展 receipt。

## 2026-10-22 — Held-out owners continue after stable-mask repair

从各 `frame_decisions.jsonl` 文件尾部读取的最新已提交帧号为：G0 MOT20-05
`3296`，G2 GPU5 `2317`，G2 GPU6 `2034`，DanceTrack `390`，AnimalTrack `156`。
五个 owner 仍为运行态且 `failed=null`；这些是逐帧进度和吞吐证据，尚未升级为
HOTA/IDF1 或其它正式指标。

## 2026-10-22 — Held-out pipelines advancing without terminal receipts

一次性读取当前逐帧输出（只记录已写入行，不升级为指标）：G0 MOT20-05 已写入
1592/1657 帧；G2 GPU5 已写入 625 帧、GPU6 已写入 363 帧；DanceTrack 域扩展已
写入 100/1203 帧；AnimalTrack 域扩展已写入 43/306 帧。五个 owner 均仍为
`failed=null`，官方 `official_metrics.json` 和最终收口 receipt 尚未生成。G0/G2
逐帧合同继续保留，当前部分输出仅作为运行进度证据。

## 2026-10-22 — Immediate parallel TAO OVMOT scope

用户要求不等待长 MOT20 队列，继续并行测试已经完成微调的 G2 Clean。GPU5 的一次性
资源读取显示约 44.7 GiB free（已有两个 18.2 GiB 左右的 G2 服务），因此新增独立
旁路 owner `g2_parallel_ovmot_gpu5_v1`，使用端口 8232 和 `ovmot_tao_independent105_full.json`
的 TAO video 105 / 40 帧 scope。该 manifest 已登记 `training_on_this_scope=false`，
并且独立审计为 0 个精确图像重叠；运行入口固定为 G2 Clean V2、official_scoring
parser、tracking_dense decoder、native_label category policy 和官方 multitask scorer。
新增脚本 `scripts/groundjev/run_g2_parallel_scope.py` 只负责这一 scope，写出独立
owner state、逐帧合同和 receipt，不覆盖现有 MOT20、DanceTrack、AnimalTrack 输出。
GPU6 当时仅约 28.3 GiB free，未再加服务，避免与现有 G2/AnimalTrack owner 争用显存。

## 2026-10-22 — G1 GPU5 adaptation held-out control started

为形成 G0/G2/G1 的同输入域对照，复用已经审计为独立的 DanceTrack
`dancetrack0004_val1203` 1203 帧 manifest，在 GPU3 的剩余显存上启动 G1 GPU5
适配 bundle `GroundAnything-DLM-TrackProjG1-GPU5-V2`，端口 8233，输出目录
`g1_dancetrack_gpu3_v1`。G0 MOT20 owner 未停止，G2 DanceTrack owner 未覆盖；新
owner 仍使用相同 run_video、official_scoring parser、tracking_dense decoder、
empty_and_record failure policy 和官方 MOT scorer。该运行用于同范围控制比较，
在完整 scorer receipt 生成前不宣称 G1 指标。

## 2026-10-22 — Extended held-out finalizer registered

原有 G2 finalizer 只覆盖五个早先注册的 G0/G2/domain owner。为避免新启动的 TAO
OVMOT 旁路和 G1 同域控制被遗漏，新增并启动
`queue_groundjev_heldout_finalize_extended.py`，等待七个 owner（G0 MOT20、两路
G2 MOT20、G2 DanceTrack、G2 AnimalTrack、G2 TAO OVMOT video105、G1 DanceTrack）
分别进入 COMPLETE/FAILED 后，调用同一严格 receipt collector 写出独立 extended
collection/finalization receipt；原有五 owner finalizer 和历史结果保持不变。

## 2026-10-22 — G200 same-domain control resource gate

尝试在 GPU3 的 G0/G1 共存窗口启动 `GroundAnything-DLM-TrackProj200` 的同一
DanceTrack 独立 scope，形成 G0/G200/G1/G2 同输入比较。启动前资源门检测到 GPU3
仅剩 17,949 MiB free，低于 32,768 MiB 的服务启动门槛，因此 owner
`g200_dancetrack_gpu3_v1` 在服务启动前安全失败（optimizer/inference updates=0），
没有触碰 G0/G1 运行，也没有把失败尝试当成指标。G200 已有的 48 帧官方控制 receipt
和 bundle 保留作为现有对照证据；不在当前四卡窗口继续强行加服务。

## 2026-10-22 — G0 MOT20-05 independent official result

G0 owner `g2_clean_independent_gpu3_v1` 已完成与 G2 相同输入合同的 MOT20-05
validation-half 独立评测：1657/1657 帧、MOT20-05 单视频、运行时未读取官方 TEST
标签。官方 scorer receipt 给出 HOTA `0.3426154581`、DetA `0.6661209465`、AssA
`0.1765665945`、IDF1 `0.2905332410`、IDSW `13712`、FP `19290`、FN `79814`，
端到端 FPS `0.0488079`，感知拒绝率 `0.0054315`（9 帧，且计入官方分母）。这是
独立 G0 控制结果，不是 G2 提升结论；同一 owner 已按注册顺序进入 RMOT single
0007，G2 GPU5/GPU6 MOT20 和其它域 owner 未被改动。

## 2026-10-22 — GPU7 shared-G2 three-task readout terminal

GPU7 的真实单主干联合 readout 已完成请求的 1200 optimizer steps，receipt
`GROUNDJEV_NATIVE_THREE_TASK_COUPLED_READOUT_COMPLETE`。它冻结最终 G2 adapter
step `18227`，使用同一 native manifest（SHA256
`cc632115aa2b87161dc7b8413d08c3c471e1225179559a76509e7366190764c1`）、单一 G2
视觉主干（`independent_vlm_instances=0`），178 个 unique visual inputs 只编码一次，
共 202 行：MOT 72、RMOT 72、OVMOT 58；任务更新计数为 330/432/331，102 个 identity
target 合法 mask。checkpoint 为
`/data3/user/LocateMOT-checkpoints/groundjev_native_three_task_expansion_gpu7_v3/groundjev-native-three-task-step001200.pt`
（SHA256 `fc7f0de761aeb3ede3be77babe92f82c6c47764b44d1c0c05b7d21a45650e209`）。这证明
真实 GPU 三任务共享-G2 readout 训练和可恢复 checkpoint 已完成；receipt 明确不作
视频级 HOTA/TETA 泛化或独立 OneJev 模型性能结论。

随后以 `weights_only=False` 在 CPU 重新加载该 checkpoint：`step=1200`、manifest/model
SHA 与 receipt 一致，`groundjev_model` 含 19 个 tensor、2,100,999 个参数元素，未发现
非有限值；因此该文件是可恢复 checkpoint，而不是只存在文件名的训练标记。

## 2026-10-22 — GPU7 immediate independent RMOT expansion

GPU7 在三任务 readout 完成后释放；为不等待 GPU5/GPU6 的 MOT20 长序列，立即启动
两个 G2 Clean 独立 RMOT owner：Refer-KITTI v2 sequence 0007 的 single expression
800 帧（端口 8235）和 multi expression 800 帧（端口 8236）。两者使用已登记
`training_on_this_scope=false` manifest、相同 official_scoring parser、tracking_dense
decoder、simple tracker 和 multitask scorer，输出目录互相独立；GPU3/5/6 的既有
owner 未修改。新增 `queue_groundjev_heldout_finalize_all.py`，把这两路加入全量
held-out 收口，原有五路/七路 finalizer 产物仍保留。

随后尝试在 GPU7 两路 RMOT 服务共存时再启动同一 TAO video105 OVMOT scope，资源
门检测到仅剩 29,109 MiB free，低于 32,768 MiB 门槛；该尝试在服务启动前安全
失败，未影响两路 RMOT。TAO OVMOT 仍由 GPU5 的独立 owner 运行，不把资源拒绝当作
评测结果。

为缩短 TAO OVMOT 的词表请求开销，新增 `queue_g2_ovmot_after_gpu7.py`：它在当前
两路 RMOT owner 释放至少 32,768 MiB 后自动启动独立 GPU7 OVMOT owner，保持相同
G2/parser/decoder/scorer 合同，仅把已登记的 vocabulary chunk 从 96 调到 192；
当前队列已确认处于 `WAITING_GPU7_FAST_OVMOT_MEMORY`（GPU7 free 29,109 MiB），不
抢占 RMOT，也不覆盖 GPU5 的 96-class OVMOT 证据。

生成并保留 `docs/GROUNDJEV/ACTIVE_GOAL_LIVE_20261022.json` 及其可重跑入口
`scripts/groundjev/write_live_goal_status.py`。该快照记录九个 held-out owner 的
状态、逐帧行数、GPU 显存和当前唯一完整 official metrics；明确把运行中帧数和
 服务状态标为 progress evidence，不升级为指标。

## 2026-10-22 — G2 AnimalTrack independent official result

G2 Clean GPU6 AnimalTrack owner 已完成独立 `chicken_1` test scope 的全部 306/306
帧，owner 状态为 `G2_DOMAIN_HELDOUT_COMPLETE`，frame contract 与 manifest 的
1..306 完全一致。官方 receipt 给出 HOTA `0.4014759675`、DetA `0.4986915750`、
AssA `0.3363050521`、IDF1 `0.4335730657`、IDSW `144`、FP `1231`、FN `2040`，
端到端 FPS `0.1212373`。这是 G2 Clean 的独立动物域结果，不与 G0/G1 或其它域
混合归因；live receipt collector 当前接受的完整 scope 数为 2（G0 MOT20 与该
AnimalTrack），其余 owner 仍按各自协议运行。

## 2026-10-22 — G1 AnimalTrack multi-domain control started

G2 AnimalTrack 306 帧独立 scope 完成并释放 GPU6 服务后，立即在同一 scope 启动
G1 GPU5 适配 bundle `GroundAnything-DLM-TrackProjG1-GPU5-V2`（端口 8239，输出
`g1_animaltrack_gpu6_v1`）。它沿用与 G2 AnimalTrack 完全相同的 manifest、parser、
decoder、tracker 和 official MOT scorer，与已运行的 G1 DanceTrack 组成多域控制；
GPU6 的 G2 MOT20 owner 未停止，G1 owner 在完整 receipt 前不作指标结论。

为确保新增 G1 AnimalTrack scope 不被早先九-owner finalizer 遗漏，新增并启动
`queue_groundjev_heldout_finalize_multidomain.py`，等待十个 G0/G2/G1/RMOT/TAO
owner 全部进入 COMPLETE/FAILED 后，用同一 receipt collector 写出独立的
multidomain finalization artifact；旧的五路、七路、九路收口文件不覆盖。

## 2026-10-22 — four-GPU parallel validation live snapshot

本次轻量收口确认四张研究卡仍按既定协议并行：GPU3 的 G0 RMOT single 与 G1
DanceTrack，GPU5 的 G2 MOT20、G2 DanceTrack、G2 TAO OVMOT，GPU6 的 G2 MOT20
与 G1 AnimalTrack，GPU7 的 G2 RMOT single/multi。当前逐帧进度为 G1 AnimalTrack
15/306、G1 DanceTrack 245/1203、G2 MOT20 GPU5 690/1657、GPU6 395/1657、G2
DanceTrack 691/1203、TAO 2/40、RMOT single 302/800、RMOT multi 395/800；这些
数字仅是运行证据，未升级为指标。GPU5/6 空闲显存约 24.5/28.1 GiB，低于 runner
的 32 GiB 启动门槛，因此没有强行再开服务；GPU7 的快速 OVMOT 队列继续等待两路
RMOT 释放显存。live receipt collector 当前包含 10 个 owner，仍只有已完成的
G0 MOT20 与 G2 AnimalTrack 两个 official scope（accepted_scope_count=2）。

## 2026-10-22 — G200 DanceTrack control resumed on released GPU3 capacity

GPU3 后续释放至约 44.7 GiB free，满足同一 runner 的 32 GiB resource gate；因此
在不增加 GPU 数量、不触碰 G0/G1 既有服务的前提下，启动独立 G200 DanceTrack
owner `g200_dancetrack_gpu3_v2`（端口 8240，model label
`G200_TRACKING200`）。原 `g200_dancetrack_gpu3_v1` 的 17,949 MiB 资源拒绝仍
保留为失败记录，不覆盖；新增 G200 owner 纳入独立十一-owner finalizer，只有完整
official receipt 才会进入 accepted collection。

## 2026-10-22 — TAO OVMOT slow-path diagnosis

GPU5 的 G2 TAO OVMOT owner 仍在正常运行；独立 manifest 的 40 帧是
`1,31,61,...,1171` 的稀疏帧，因此当前 `frame_decisions.jsonl` 的 3 行对应已完成
的前三个 manifest 帧，而不是 3/1203 的漏帧。每个 1203-class 请求约耗时 400 秒，
主要开销来自词表分块。模型输出多次包含重复坐标 tuple，`official_scoring` parser
按既定 strict wire contract 拒绝整帧并保留原始输出；没有把拒绝结果猜测为合法框。
GPU7 的 192-class chunk 独立队列继续等待安全显存窗口，用于比较词表分块吞吐，不能
覆盖当前 96-class 证据。

## 2026-10-22 — resumable OneJev three-task continuation registered

当前进程表没有 `run_source_manifest_sft.py`、MOT-only 或旧 J1 optimizer；历史
`onejev_full_joint_source_epoch1_v1` 在非终态 step 3500 附近退出，但 step3000 的
完整 FP32/Adam/RNG 状态和 `latest_training_state.json` 已验证存在（9,029,193,825
bytes，step=3000）。因此没有重启历史目录，也没有把中断误报成完成。

新增并启动 `queue_onejev_full_joint_resume_after_heldout.py`：它等待当前十一条
held-out owner 和 G200 finalizer 进入终态，确认 GPU3/5/6/7 无计算进程且每卡至少
32 GiB free 后，从该 step3000 状态恢复真实 MOT/RMOT/OVMOT source-manifest 联合
训练，输出到新的 `/data3/user/LocateMOT-checkpoints/onejev_full_joint_source_epoch1_after_heldout_v1`。
历史目录保持只读证据，队列当前处于 `WAITING_HELDOUT_TERMINAL_AND_EXCLUSIVE_GPU_SET`。

## 2026-10-22 — RMOT multi official receipt and fast TAO OVMOT launch

G2 Clean RMOT multi `0007` reached its complete 800/800-frame contract. The official
receipt reports HOTA `0.5028468967`, DetA `0.4257416214`, AssA `0.5942019821`, IDF1
`0.5618408437`, IDSW `108`, FP `1450`, FN `459`; the receipt also records the official
end-to-end timing. The first live collection omitted this valid result because the generic
runner stores one completed block directly at owner level while the collector only handled
the older scope-nested shape. The collector now normalizes both shapes; accepted scope
count rose from 2 to 3 without changing any result or score.

RMOT multi released GPU7 capacity. The waiting TAO OVMOT control crossed the 32 GiB gate
(observed 42,827 MiB free) and started on GPU7 with vocabulary chunk size 192, while the
original GPU5 96-class run remains untouched. A new twelve-owner finalizer and live
snapshot include this fast OVMOT control; it is a throughput comparison, not a replacement
for the original protocol.

## 2026-10-22 — G0 RMOT single official receipt and G0 multi continuation

G0 owner `g2_clean_independent_gpu3_v1` completed RMOT single `0007` with the full
800-frame contract and immediately advanced to the registered G0 RMOT multi scope. The
official single receipt reports HOTA `0.0531207079`, DetA `0.0073404994`, AssA
`0.3847650569`, IDF1 `0.0143884892`, IDSW `2`, FP `1361`, FN `7`; the owner now has two
accepted official scopes (MOT20 and RMOT single) and is not being restarted.

The live collection therefore contains four accepted scopes: G0 MOT20, G0 RMOT single,
G2 AnimalTrack, and G2 RMOT multi. The GPU7 chunk-192 TAO OVMOT owner has entered
`INFERENCE` after the RMOT multi service released memory; it has no accepted metric yet.

## 2026-10-22 02:44 UTC — user-directed parallel held-out evaluation remains active
- Action: kept the permitted research set on GPUs 3/5/6/7 and continued independent G0/G1/G2/G200/OVMOT scopes in parallel. GPU5 and GPU6 already have active G2 services plus domain/MOT runs; free memory was about 24.5 GiB and 28.1 GiB, below the 32 GiB new-server gate, so no unsafe extra model server was added.
- Current scopes: G0 RMOT-multi continuation; G2 MOT20 on GPUs 5/6; G2 DanceTrack and TAO OVMOT on GPU5; G2 AnimalTrack complete; G1 DanceTrack/AnimalTrack; G200 DanceTrack; RMOT single and fast TAO OVMOT on GPU7.
- The two independent G2 queues are already configured to proceed from MOT20 to RMOT single, RMOT multi, and independent TAO OVMOT using the same service contract after each scope. This is a progress record only; no incomplete frame count is treated as a metric.

## 2026-10-22 10:48 Asia/Shanghai — G2 Clean TAO Base/Novel video63 added in parallel
- Action: GPU7 had 39,397 MiB free while the fast TAO control was active, so a second `.30` GroundAnything G2 Clean service was started on port 8241 for the registered TAO video63 Base/Novel manifest (`ovmot_tao_base_novel63_full.json`, 40 annotation frames, 160 evaluation-only GT boxes). The run uses the same official parser, tracking_dense decoder, Simple associator, native-label policy and vocabulary chunk size 192.
- The held-out G200 finalizer, fast-OVMOT finalizer, and OneJev resume queue were restarted with the new video63 owner included in their terminal gates. Existing model/evaluation processes were left untouched.
- Scope is still running; no partial frame count is promoted to a metric. This closes the earlier gap where only TAO video105 was active for the current G2 Clean OVMOT sweep.

## 2026-10-22 10:55 Asia/Shanghai — receipt hash enrichment
- The held-out collector now resolves bundle directories to `model.safetensors` before hashing and records model/manifest SHA-256 per accepted scope. This fixes the prior directory-hash omission without changing inference or scoring.

## 2026-10-22 11:00 Asia/Shanghai — G1 adaptation evidence registered in live snapshot
- The completed GPU5 GroundAnything G1 adaptation owner is now included in the live evidence snapshot alongside its held-out DanceTrack/AnimalTrack inference owners. Its receipt records 200 native Tracking200 training steps, exported bundle SHA-256 `0efd9d7ddb241d588be85cb6b08853de5dfd18011aa477932c0580bab3265a25`, and 12 isolated control units over 96 registered frames; those controls are not promoted as full-dataset performance.

## 2026-10-22 11:05 Asia/Shanghai — held-out gate corrected for both TAO controls
- The primary G200 finalizer and OneJev resume gate now require both TAO OVMOT owners: the original chunk-96 video105 run, the fast chunk-192 video105 control, and the newly launched Base/Novel video63 run. This prevents the joint resume from starting while a parallel TAO control remains incomplete.

## 2026-10-22 11:15 Asia/Shanghai — OVMOT duplicate-coordinate root cause and repaired rerun
- Diagnosis: TAO video105 frame31 returned 100 complete pairs plus two exact duplicate coordinate tuples, but also a malformed concatenated label fragment. The strict protocol raised `duplicate coordinate tuples are not canonical`; the pinned scoring parser then recovered no safe complete sequence, so the whole chunk became an empty/rejected frame.
- Independent parser check on the saved raw response recovered 100 complete label/box pairs, removed only 2 exact duplicates, dropped 0 fragments, and filled/reordered no coordinates.
- Change: new OVMOT invocations now use `official_scoring_repaired`, which preserves strict diagnostics while retaining only complete canonical pairs. The unfinished strict TAO video63 diagnostic was preserved and marked superseded; a fresh repaired video63 owner was launched with a separate output directory. Existing strict video105 remains a diagnostic control.

## 2026-10-22 Asia/Shanghai — SGLang resource reduction
- Resource action: released three owned process groups to avoid occupying shared GPU capacity: the superseded strict TAO video105 run on GPU5 (`g2_parallel_ovmot_gpu5_v1`), the duplicate full MOT20 run on GPU6 (`g2_clean_independent_gpu6_v1`, same input as the retained GPU5 run), and the orphaned historical service on port 8101 whose scheduler was already defunct.
- Partial outputs were preserved. The two interrupted owners were marked `FAILED/RESOURCE_RECLAIMED` with `heldout_claim=false`; neither partial output is used as a metric.
- The active SGLang scheduler count dropped from 10 live schedulers plus one zombie to 8 live schedulers. Remaining services are G0, G1, G2, G200, and repaired TAO scopes; no new server will be added to GPU6 until a queued scope needs it.

## 2026-10-22 Asia/Shanghai — stale native-training waiters retired
- Two historical shell waiters were still alive: one waiting on a failed native-three-task owner and one scheduled to launch `queue_groundjev_native_three_task.py` after G0 became terminal. They were not part of the current OneJev full-joint queue and would have competed for GPU resources after the held-out gate.
- Both process groups were terminated without touching active inference owners. Existing native-training failure receipts remain unchanged; the recoverable training entry is now the explicit `queue_onejev_full_joint_resume_after_heldout.py` path.

## 2026-10-22 Asia/Shanghai — G0 RMOT multi receipt recovered
- The G0 owner produced a complete official RMOT multi `0007` receipt with all 800 registered frames. Metrics are HOTA `0.3148685730`, DetA `0.3013681168`, AssA `0.3327250738`, IDF1 `0.3742232451`, IDSW `123`, FP `1100`, FN `979`.
- A partial held-out collection was generated at `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/groundjev_heldout_partial_collection_20261022.json`; it validates four complete scopes (G0 MOT20, G0 RMOT single, G0 RMOT multi, G2 AnimalTrack). Running and resource-reclaimed owners remain status evidence only.

## 2026-10-22 Asia/Shanghai — G2 DanceTrack held-out receipt
- G2 Clean DanceTrack `dancetrack0004_val1203` completed all 1203 frames with official HOTA `0.1202769124`, DetA `0.5240129428`, AssA `0.0277880198`, IDF1 `0.0528881911`, IDSW `574`, FP `3025`, FN `96`.
- The refreshed partial collection `groundjev_heldout_partial_collection_20261022_v2.json` validates five complete scopes: G0 MOT20, G0 RMOT single/multi, G2 DanceTrack, and G2 AnimalTrack. TAO, G1, G200, and the G0 TAO control remain pending and are excluded from the accepted count.

## 2026-10-22 Asia/Shanghai — superseded strict G0 TAO control safely closed
- The legacy G0 owner had advanced to `ovmot_tao105` with the pre-repair strict parser and vocabulary chunk size 96. Its G0 MOT20/RMOT receipts were already complete, while this extra control would have held the OneJev terminal gate for hours and was not the repaired OVMOT evidence.
- Only the G0 TAO `run_video.py` child was terminated. The owner wrote `G2_CLEAN_FULL_INDEPENDENT_HELDOUT_FAILED` with the command failure and preserved partial output; existing G0 MOT20/RMOT receipts remain accepted. The repaired G2 TAO video63/video105 owners were left untouched.

## 2026-10-22 Asia/Shanghai — G1 AnimalTrack held-out receipt
- The GPU6 GroundAnything G1 adapter owner completed the independent AnimalTrack chicken_1 test scope with all 306 frames. Official metrics: HOTA `0.4661208959`, DetA `0.5490949528`, AssA `0.4067583015`, IDF1 `0.4992391856`, IDSW `145`, FP `1639`, FN `1300`.
- This is a detector-adaptation held-out result under the G1 bundle and the same parser/decoder/tracker/scorer contract; G1 DanceTrack remains pending, so no multi-domain aggregate claim is made.

## 2026-10-22 Asia/Shanghai — same-GPU SGLang schedulers serialized
- Resource snapshot found five active GroundAnything schedulers: G1 and G200 on GPU3, G2 MOT20 on GPU5, and repaired TAO video63/video105 on GPU7. The two pairs on GPU3/GPU7 each consumed roughly 18–20 GiB per scheduler and left no useful compute headroom on GPU7.
- To avoid occupying shared capacity, the G200 and repaired TAO video105 child scopes were stopped cleanly after preserving their partial outputs. Their owners wrote explicit `G2_PARALLEL_SCOPE_FAILED` receipts with return code `-15`; those partial scopes are not metrics.
- Added `queue_groundjev_serial_scope.py`. It waits for G1 DanceTrack or repaired TAO video63 to reach a terminal state and for at least 32 GiB free on the same GPU, then reruns the exact original manifest/parser/decoder/tracker/scorer contract in fresh output directories. The finalizers, OneJev gate, collector defaults and live snapshot include the serial owner paths, so the joint resume cannot skip them.
- After this change the owned scheduler count is three (one each on GPU3, GPU5 and GPU7); GPU6 remains untouched. No training checkpoint or historical result was overwritten.

## 2026-10-22 Asia/Shanghai — G200 serial retry after orphaned service cleanup
- G1 DanceTrack completed its 1203-frame official scope. The first G200 serial queue reached the service launch but failed on frame 1 because the terminated predecessor had left an owned `infer.sglang_server` orphan (PID 249526) listening on port 8240; the original owner and scheduler were gone, but the child service survived under PID 1.
- The orphan was identified by its exact TrackProj200 model path and old G200 service directory, then terminated. Its partial/failed receipt remains preserved; no unrelated process was touched.
- A fresh `g200_dancetrack_gpu3_serial_v2` queue/output directory was launched with the same manifest, model, parser, decoder, tracker and scorer contract. Finalizers, collector defaults, live status and the OneJev terminal gate now include v2 while retaining v1 as failure evidence. The v2 service passed startup and entered its inference stage.

## 2026-10-22 Asia/Shanghai — G200 and repaired TAO video63 official receipts
- G200 TrackProj200 DanceTrack `dancetrack0004_val1203` completed all 1203 registered frames under the same official parser/decoder/Simple tracker/scorer contract. Metrics: HOTA `0.1034892155`, DetA `0.2864657057`, AssA `0.0376125087`, IDF1 `0.0534199802`, IDSW `652`, FP `9155`, FN `169`, end-to-end FPS `0.3338274286`. This remains a G200 control, not a G2 claim.
- Repaired G2 Clean TAO Base/Novel video63 completed all 40 registered annotation frames. The official unmodified TETA recipe reports TETA/LocA/AssocA/ClsA all `0.0`, `predictions=0`, and perception rejection on all 40 annotation steps; this is recorded as a valid negative OVMOT result, not a performance gain.
- Both receipts passed exact frame-count checks. A compact summary was written to `docs/GROUNDJEV/G200_TAO63_RECEIPTS_20261022.json`. The serial TAO video105 queue is now the only remaining repaired TAO scope before held-out finalization can run.

## 2026-10-22 Asia/Shanghai — G200 and repaired TAO video63 official receipts
- G200 TrackProj200 DanceTrack `dancetrack0004_val1203` completed all 1203 registered frames under the same official parser/decoder/Simple tracker/scorer contract. Metrics: HOTA `0.1034892155`, DetA `0.2864657057`, AssA `0.0376125087`, IDF1 `0.0534199802`, IDSW `652`, FP `9155`, FN `169`, end-to-end FPS `0.3338274286`. This remains a G200 control, not a G2 claim.
- Repaired G2 Clean TAO Base/Novel video63 completed all 40 registered annotation frames. The official unmodified TETA recipe reports TETA/LocA/AssocA/ClsA all `0.0`, `predictions=0`, and perception rejection on all 40 annotation steps; this is recorded as a valid negative OVMOT result, not a performance gain.
- Both receipts passed exact frame-count checks. A compact summary was written to `docs/GROUNDJEV/G200_TAO63_RECEIPTS_20261022.json`. The serial TAO video105 queue is now the only remaining repaired TAO scope before held-out finalization can run.

## 2026-10-22 Asia/Shanghai — TAO video105 serial retry after orphaned service cleanup
- The first serial TAO video105 owner (`...video105_serial_v1`) reached frame 1 but failed because the superseded repaired owner had left its G2 Clean `infer.sglang_server` orphan (PID 319956) listening on port 8238. The inference traceback was a client timeout; the service log confirms the new service could not bind the occupied port.
- The exact old G2 Clean service process was terminated after checking its model path, old service directory and port ownership. Its v1 failure receipt remains unchanged and contributes no metric.
- A fresh `...video105_serial_v2` queue/output directory was started with the repaired parser, vocabulary chunk size 192, native-label policy and the original independent manifest. Finalizers, collector defaults, live status and the OneJev gate now include v2 while retaining v1 as failure evidence.

## 2026-10-22 Asia/Shanghai — repaired TAO video105 official receipt
- Repaired G2 Clean TAO independent video105 `val/HACS/Making_a_sandwich_v_6RjriKVIkR4_scene_0_0-4638` completed all 40 registered annotation frames in the v2 serial queue. The unmodified official TETA recipe reports TETA/LocA/AssocA/ClsA all `0.0`, `predictions=0`, and 40 perception-rejected annotation steps; this is a valid negative OVMOT result.
- The v1 timeout was caused by a stale service port, not a parser or model change. The v2 receipt uses the repaired parser, chunk size 192 and the same manifest/decoder/tracker/scorer contract. The compact G200/TAO receipt summary now contains G200 plus both repaired TAO scopes.
- Final held-out finalization remains gated only by the long G2 MOT20 owner; OneJev has not started early.

## 2026-10-22 Asia/Shanghai — G2 Clean MOT20-05 official receipt
- G2 Clean MOT20-05 validation completed all 1657 registered frames with the official TrackEval contract. Metrics: HOTA `0.4447386120`, DetA `0.7696739946`, AssA `0.2579811216`, IDF1 `0.4008651255`, IDSW `8464`, FP `14347`, FN `31983`; end-to-end FPS `0.0288813059`.
- The official denominator includes the 65 perception-rejected frames (`3.92275%`); no frame was removed from scoring. This is a complete single-sequence held-out result, not yet a multi-dataset aggregate claim.
- The same G2 owner immediately advanced to its registered RMOT single scope; no second scheduler was created.

## 2026-10-22 23:57 Asia/Shanghai — two additional MOT20 model controls launched in parallel
- Action: preserved the active G2 Clean GPU5 owner and launched two independent full-scope controls on the same registered MOT20-05 validation manifest (1657 frames, `training_on_this_scope=false`) using the identical parser, tracking-dense decoder, Simple associator and official TrackEval scorer.
- GPU4 runs `G1_GPU5_V2` on port 8240; GPU7 runs `G200_TRACKING200` on port 8238. Both passed the 32,768 MiB free-memory gate (51,336 and 60,638 MiB respectively).
- GPU3 and GPU6 were occupied by unrelated processes and were left untouched. No partial output is promoted as a metric; each owner must pass service startup, all-frame processing and receipt validation first.
- Launch metadata, manifest/model SHA-256 and owner paths are recorded in `docs/GROUNDJEV/MOT20_PARALLEL_CONTROL_LAUNCH_20261022.json`.

## 2026-10-22 23:58 Asia/Shanghai — two additional MOT20 model controls launched in parallel
- Action: preserved the active G2 Clean GPU5 owner and launched two independent full-scope controls on the same registered MOT20-05 validation manifest (1657 frames, `training_on_this_scope=false`) using the identical parser, tracking-dense decoder, Simple associator and official TrackEval scorer.
- GPU4 runs `G1_GPU5_V2` on port 8240; GPU7 runs `G200_TRACKING200` on port 8238. Both passed the 32,768 MiB free-memory gate (51,336 and 60,638 MiB respectively).
- GPU3 and GPU6 were occupied by unrelated processes and were left untouched. No partial output is promoted as a metric; each owner must pass service startup, all-frame processing and receipt validation first.
- Launch metadata, owner paths and resource evidence are recorded in `docs/GROUNDJEV/MOT20_PARALLEL_CONTROL_LAUNCH_20261022.json`.

## 2026-10-23 — registered the two parallel MOT20 controls in all held-out gates
- Audit found that the newly launched `g1_mot20_05_gpu4_v1` and `g200_mot20_05_gpu7_v1` owners were absent from the held-out finalizers and the OneJev resume gate. Without this, finalization could complete while those controls were still running.
- Added both owner paths to the multidomain/G200/fast-OVMOT finalizers, the receipt collector defaults, the OneJev terminal gate, and the live frame-progress snapshot.
- Restarted only the four waiting processes with the updated source; active G2/G1/G200 inference services were not restarted. Targeted `compileall` passed and the live snapshot now reports 22 owners.

## 2026-10-23 — OneJev resume GPU contract aligned
- Audit found the resumable OneJev gate still listed temporary GPUs `(3,5,6,7)`, while the registered source-epoch training contract and historical launch receipt require `(4,5,6,7)`.
- Changed the queue to use the registered set, record `previous_physical_gpus` and `gpu_contract_revision`, then restarted only the waiting gate. The owner state now records `(4,5,6,7)` and remains held out/resource gated; no optimizer process was started.
- Targeted `compileall` and live snapshot regeneration passed.

## 2026-10-23 00:19 Asia/Shanghai — MOT20 G1/G200 controls remain live but low-throughput
- A one-time diagnostic found both owner and service PIDs alive in `INFERENCE`; the registered physical frame range is 1659–3315 (1657 frames). G1 had reached frame 1672 and G200 frame 1680.
- Recent per-frame wall time was approximately 67.6s (G1 median) and 55.0s (G200 median), yielding rough remaining-time estimates of 30.9h and 25.0h. These are scheduling estimates, not results.
- No process failure was observed, so both runs remain protected under the original parser/decoder/tracker/scorer contract. Partial frames are not promoted as metrics.
- The live throughput evidence is recorded in `docs/GROUNDJEV/MOT20_CONTROL_THROUGHPUT_LIVE_20261023.json`.

## 2026-10-23 — G1 MOT20 control migrated to free GPU5 with causal resume
- GPU5 became available while the protected G200 MOT20 control continued on GPU7. The G1 GPU4 owner had a valid 18-frame prefix (`1659`–`1676`) in `g1_mot20_05_gpu4_v1`; it had no terminal result and was never promoted as a metric.
- The GPU4 owner/service were stopped after the prefix was preserved. Its owner is marked `G1_MOT20_05_PARTIAL_SUPERSEDED_FOR_GPU5_RESUME_FAILED` with `heldout_claim=false` and remains immutable evidence of the interrupted attempt.
- Added `--resume-from` to `scripts/groundjev/run_g2_parallel_scope.py`. The new owner `g1_mot20_05_gpu5_resume_v1` uses the same legal 1,657-frame manifest, G1 model, parser, dense decoder, Simple tracker and official scorer, while `tools/groundjev/run_video.py` restores the exact completed causal prefix into a fresh output and processes the suffix on GPU5.
- Updated all held-out finalizers, collector defaults, the OneJev resume gate and live status to include the resumed owner. Until the resumed owner reaches a complete official receipt, neither G1 nor the pair of controls contributes a metric claim.

## 2026-10-23 — MOT20 controls verified advancing after migration
- A fresh audit found no decoder, service, or owner failure. The G1 GPU5 resumed owner had written 31 suffix/prefix-combined rows through physical frame 1689; G200 GPU7 had written 52 rows through frame 1710.
- Recent per-frame latency was about 30.4 s (G1) and 29.8 s (G200) over the latest samples. These are throughput estimates only; neither scope has a terminal official receipt, so no partial metric is promoted.
- The live receipt was refreshed at `docs/GROUNDJEV/MOT20_CONTROL_THROUGHPUT_LIVE_20261023.json`; both finalizers and the OneJev gate remain live and continue to require terminal owners.

## 2026-10-23 — MOT20 controls continue with live progress, no metric promotion
- The original completion waiter was interrupted after repeated no-output intervals; this affected only the waiter, not either inference owner or service. A direct owner/process snapshot verified both owners and services alive in `INFERENCE`.
- The resumed G1 GPU5 output advanced to 264 registered rows through physical frame 1922; the G200 GPU7 output advanced to 310 rows through physical frame 1968. The legal manifest remains 1,657 validation-half frames (physical 1659–3315, `training_on_this_scope=false`).
- Over the latest 20 rows, median wall time was approximately 49.7 s/frame for G1 and 24.7 s/frame for G200, giving rough remaining estimates of 19.2 h and 9.2 h. These are scheduling estimates only; neither owner has a terminal official receipt, so no partial metric is claimed.
- Refreshed `docs/GROUNDJEV/MOT20_CONTROL_THROUGHPUT_LIVE_20261023.json` with the snapshot. Preserve both runs without restart; OneJev remains gated on terminal held-out owners and exclusive GPU `(4,5,6,7)`.

## 2026-10-23 — MOT20 controls verified advancing after extended wait
- A direct snapshot found both current owners and services alive: G1 GPU5 reached 397 rows through physical frame 2055, and G200 GPU7 reached 439 rows through physical frame 2097. The correct owner waiter remains active; no inference process was restarted.
- The legal scope is unchanged at 1,657 validation-half frames (physical 1659–3315, `training_on_this_scope=false`). These row counts are progress evidence only and carry no metric claim.
- The live throughput receipt was refreshed while the files were advancing; the latest recomputed snapshot reached 399/440 rows (G1/G200) with recent-20-row median estimates near 26.0/26.4 s per frame and rough remaining times near 9.1/8.9 hours. Treat these as scheduling estimates, not performance results.

## 2026-10-23 — MOT20 controls continue after extended wait
- A fresh snapshot verified both owner/service pairs still alive and advancing: G1 GPU5 reached 491 rows through physical frame 2149, and G200 GPU7 reached 526 rows through physical frame 2184.
- The refreshed live snapshot subsequently recorded 492/527 rows (G1/G200), with recent-20-row median estimates of approximately 27.8/26.8 s per frame and rough remaining times of 9.0/8.4 hours. These remain scheduling estimates only; neither scope has a terminal official receipt or metric claim.
- Preserve the current runs and corrected owner waiter; OneJev remains gated on both terminal held-out scopes and exclusive GPU `(4,5,6,7)`.

## 2026-10-23 — MOT20 controls remain live and advancing
- A new direct snapshot verified both owner/service pairs alive: G1 GPU5 reached 552 rows through physical frame 2210, and G200 GPU7 reached 588 rows through physical frame 2246.
- The live throughput receipt then recorded 553/589 rows (G1/G200). Recent-20-row median estimates were approximately 31.5/25.2 s per frame, with rough remaining times of 9.7/7.5 hours. These are scheduling estimates only; neither scope has a terminal official receipt or metric claim.
- Preserve both runs and the corrected owner waiter; no partial output is scored or promoted, and OneJev remains gated on terminal held-out owners.

## 2026-10-23 — MOT20 controls continue toward full scope
- A later snapshot verified both owner/service pairs alive: G1 GPU5 reached 609 rows through physical frame 2267, and G200 GPU7 reached 657 rows through physical frame 2315.
- The live throughput receipt then recorded 611/659 rows (G1/G200), with recent-20-row median estimates of approximately 25.7/24.3 s per frame and rough remaining times of 7.5/6.7 hours. These are scheduling estimates only; neither scope has a terminal official receipt or metric claim.
- Keep the same legal manifest and scorer contract. No partial output is scored or promoted; OneJev remains gated on terminal held-out owners.

## 2026-10-23 — MOT20 controls continue after the next progress audit
- A direct snapshot verified both owner/service pairs alive: G1 GPU5 reached 642 rows through physical frame 2300, and G200 GPU7 reached 685 rows through physical frame 2343.
- The live throughput receipt then recorded 643/688 rows (G1/G200), with recent-20-row median estimates of approximately 27.0/26.1 s per frame and rough remaining times of 7.6/7.0 hours. These remain scheduling estimates only; neither scope has a terminal official receipt or metric claim.
- Preserve both runs under the same legal manifest and official scorer contract; no partial output is scored or promoted.

## 2026-10-23 — MOT20 controls continue toward the terminal half
- A later snapshot verified both owner/service pairs alive: G1 GPU5 reached 714 rows through physical frame 2372, and G200 GPU7 reached 774 rows through physical frame 2432.
- The live throughput receipt then recorded 716/775 rows (G1/G200), with recent-20-row median estimates of approximately 29.0/22.8 s per frame and rough remaining times of 7.6/5.6 hours. These remain scheduling estimates only; neither scope has a terminal official receipt or metric claim.
- Preserve both runs under the same legal manifest and official scorer contract; no partial output is scored or promoted.

## 2026-10-23 — MOT20 controls continue with advancing row counts
- A direct snapshot found both owner/service pairs alive: G1 GPU5 reached 776 rows through physical frame 2434, and G200 GPU7 reached 843 rows through physical frame 2501.
- The live throughput receipt then recorded 779/845 rows (G1/G200), with recent-20-row median estimates of approximately 24.9/20.5 s per frame and rough remaining times of 6.1/4.6 hours. These remain scheduling estimates only; neither scope has a terminal official receipt or metric claim.
- Keep the legal manifest and official scorer contract unchanged; no partial output is scored or promoted.

## 2026-10-23 — MOT20 controls continue toward completion
- A direct snapshot verified both owner/service pairs alive: G1 GPU5 reached 875 rows through physical frame 2533, and G200 GPU7 reached 906 rows through physical frame 2564.
- The live throughput receipt then recorded 878/908 rows (G1/G200). Recent-20-row median estimates were approximately 11.2/18.8 s per frame, with rough remaining times of 2.4/3.9 hours. These remain scheduling estimates only; neither scope has a terminal official receipt or metric claim.
- Keep the same legal manifest and official scorer contract; no partial output is scored or promoted.

## 2026-10-23 — MOT20 controls continue with both owners alive
- A direct snapshot verified both owner/service pairs alive: G1 GPU5 reached 919 rows through physical frame 2577, and G200 GPU7 reached 931 rows through physical frame 2589.
- The live throughput receipt then recorded 921/934 rows (G1/G200), with recent-20-row median estimates of approximately 14.8/20.9 s per frame and rough remaining times of 3.0/4.2 hours. These remain scheduling estimates only; neither scope has a terminal official receipt or metric claim.
- Keep the same legal manifest and official scorer contract; no partial output is scored or promoted.

## 2026-10-23 — two additional evaluations share the existing GPU services
- Started `g1_rmot_shared_gpu5_v2` on the independent 800-frame RMOT single-query validation manifest through the existing G1 service at GPU5/port 8240.
- Started `g200_ovmot_shared_gpu7_v2` on the independent TAO video63 Base/Novel 40-frame OVMOT manifest through the existing G200 service at GPU7/port 8238, using the repaired official parser and vocabulary chunk size 192.
- Added `scripts/groundjev/run_shared_service_scope.py` to make the shared-service contract explicit: it never starts or stops SGLang, isolates output, records model/manifest hashes and commands, validates the full frame list, and runs the official scorer before accepting a receipt.
- Both owners are in `INFERENCE`; there are still only the original two SGLang schedulers. No partial output has a metric claim. The combined blocking waiter now covers these two owners plus the two MOT20 controls.

## 2026-10-23 — held-out finalization gate extended to the shared-service owners
- The fixed owner lists used by the receipt collector, three active finalizers and the OneJev resume gate now include `g1_rmot_shared_gpu5_v2` and `g200_ovmot_shared_gpu7_v2`; this prevents either new evaluation from being omitted from the final evidence set.
- Replaced four independent waiting processes with the single durable `scripts/groundjev/orchestrate_heldout_finalizers_and_onejev.py` waiter. It waits for all 24 registered owner directories, runs the three finalizers, then invokes the registered exclusive `(4,5,6,7)` OneJev resume queue. Current orchestrator state is `WAITING_ALL_REGISTERED_HELDOUT_OWNERS`; no OneJev optimizer process has started.

## 2026-10-23 — shared-service evaluations and MOT20 controls remain healthy
- A targeted owner audit found no failure: G1 MOT20 had 1,304 rows, G200 MOT20 had 1,148 rows, G1 RMOT had 68 rows, and G200 OVMOT had 1 row. All four owners remain in `INFERENCE` and the orchestrator remains in `WAITING_ALL_REGISTERED_HELDOUT_OWNERS`.
- These are progress rows only. No official scorer has accepted any partial output, and no metric claim is promoted until each registered manifest is complete and its receipt passes the frame contract.

## 2026-10-23 — OVMOT shared-service throughput constraint recorded
- The next audit found G1 MOT20 at 1,314 rows, G200 MOT20 at 1,150 rows, G1 RMOT at 85 rows, and G200 OVMOT at 1 row. All owners remain healthy in `INFERENCE`.
- The first G200 OVMOT TAO video63 frame took approximately 524 seconds because the 1,000-class vocabulary is processed through the same single-request service as the G200 MOT20 control. This is a scheduling limitation, not a metric result; partial outputs remain excluded from scoring. The service was not duplicated, so the scheduler count remains unchanged.

## 2026-10-23 — G200 OVMOT moved to a dedicated GPU7 service with causal resume
- The shared G200 OVMOT client was stopped after its one-frame prefix was preserved. Its owner is terminally marked `G200_OVMOT_SHARED_PARTIAL_SUPERSEDED_FOR_DEDICATED_SERVICE_FAILED` with `heldout_claim=false`; the prefix remains immutable evidence and is not scored.
- GPU7 had 56,499 MiB free before migration. A second G200 service was started on port 8242, while the original port 8238 service and G200 MOT20 control were left untouched. The new owner `g200_ovmot_dedicated_gpu7_v1` resumes from the preserved causal prefix with the same model, manifest, repaired parser, dense decoder, tracker, vocabulary chunk size 192 and official scorer.
- The finalizer/collector/OneJev owner lists now include the dedicated owner. The single orchestrator was restarted and is again waiting for all registered owners; no metric is promoted from the superseded prefix.

## 2026-10-23 — dedicated OVMOT and remaining controls advance
- A progress snapshot found the dedicated G200 OVMOT owner at 2 rows, G200 MOT20 at 1,165 rows, G1 RMOT at 166 rows, and G1 MOT20 at 1,359 rows. All four remain in `INFERENCE` with no failure receipt.
- These row counts are progress evidence only. The full-frame contracts and official metrics remain pending until each owner reaches a terminal receipt.

## 2026-10-23 — two further GroundJEV validations added without new schedulers
- Started `g1_ovmot_shared_gpu5_v1` for the independent 40-frame TAO video-63 Base/Novel OVMOT manifest through the existing G1 GroundAnything service at GPU5/port 8240, with the repaired official parser and vocabulary chunk size 192.
- Started `g200_rmot_shared_gpu7_v1` for the independent 800-frame RMOT multi-query manifest through the existing G200 GroundAnything service at GPU7/port 8238.
- Both launchers use `run_shared_service_scope.py`; they do not start or stop SGLang and keep isolated output/receipts. The owner lists and the single held-out/OneJev gate now contain 27 registered scopes. The orchestrator was restarted before the new clients were launched, so OneJev cannot start while either client is unfinished.
- No partial rows are metrics. Full-frame order, the official scorer, model/manifest hashes and terminal receipts are still required before any result is promoted.

## 2026-10-23 — OneJev gate switched to the real native three-task course
- Audited `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/real_three_task_joint_native_v1/train.sources.json`: `native_ovmot_training_rows=true`, `ovmot_training_mode=native`, 131,618 rows total (131,214 MOT identity, 346 RMOT membership/presence, 58 native TAO category group/leaf rows). The native OVMOT component is a registered TRAIN subset sourced from the preserved candidate/annotation join; it is not a full-TAO claim and has zero optimizer updates so far.
- Patched `queue_real_three_task_native.py` to use the unified held-out finalizer, accept a fresh queue/checkpoint root and explicit GPU list, and require only the measured free-memory floor while recording foreign PIDs. The new course uses GPU5/7 after the held-out services terminate; GPU4/6 foreign jobs remain untouched.
- Patched `orchestrate_heldout_finalizers_and_onejev.py` to invoke the fresh native queue after all 27 owners and three finalizers complete. The historical teacher-control/step-3000 queue remains preserved and is not invoked.
- Compile/import checks passed and the native manifest loaded 131,618 rows in the source trainer preflight before CPU token-length planning was stopped to avoid occupying the host for an extended dry run. No optimizer or GPU training update was made by this preflight.

## 2026-10-23 — held-out owners advance under the expanded shared-service batch
- A current progress snapshot found G1 MOT20 at 1,413/1,657 rows through physical frame 3,072 and G200 MOT20 at 1,206/1,657 rows through physical frame 2,864. Recent-20 medians were about 45.1 s and 34.2 s per frame, giving rough remaining estimates of 3.1 h and 4.3 h.
- The two added shared-service scopes remain active: G1 RMOT 245/800, G200 RMOT 26/800, G1 OVMOT 2/40, and dedicated G200 OVMOT 5/40. These are row-count observations only; no official metric is promoted from partial output.
- The unified orchestrator remains in `WAITING_ALL_REGISTERED_HELDOUT_OWNERS`; native OneJev has not launched. The live MOT20 receipt was refreshed with the current counts and timing estimates.

## 2026-10-23 — two additional cross-model held-out controls launched
- User requested two more tasks in parallel. To avoid increasing `sglang::scheduler` count, both jobs reuse validated shared services: G1/GPU5 on `http://127.0.0.1:8240` and G200/GPU7 on `http://127.0.0.1:8238`.
- `g1_dancetrack_shared_gpu5_v1`: DanceTrack `dancetrack0004` validation, 1,203 frames, G1 bundle `GroundAnything-DLM-TrackProjG1-GPU5-V2`, official scoring parser.
- `g200_animaltrack_shared_gpu7_v1`: AnimalTrack `chicken_1` test, 306 frames, G200 bundle `GroundAnything-DLM-TrackProj200`, official scoring parser.
- Both scopes reuse the existing independent manifests from `heldout_inputs/g2_domain_v1`; outputs are isolated and use the same input/parser/decoder/tracker/scorer contract. GPU6 was left untouched because it has another user's active workload.
- Launch receipt: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/parallel_two_shared_launch_20261023.json`; runner PIDs at launch were 3511892 and 3511893. No new SGLang service was created. Initial wrapper failure caused by precreating non-empty output dirs was cleaned without inference; relaunch uses directory-external launcher logs.
- Metrics are intentionally not claimed until each owner writes a terminal receipt and official metrics.

## 2026-10-23 — recoverable retries queued after shared-service failures
- The first G1 DanceTrack shared control reached a terminal failure at frame 1 because GPU5/8240 returned HTTP 503 while the existing MOT/RMOT/OVMOT clients occupied the service queue. The failed receipt at `heldout/g1_dancetrack_shared_gpu5_v1/failure_receipt.json` is preserved and is not treated as a metric.
- Started durable `queue_g1_dancetrack_shared_retry.py` (PID 3517597). It waits for the two extra G1 shared clients to reach terminal receipts, rechecks the existing service, and retries the identical 1,203-frame scope under `g1_dancetrack_shared_gpu5_v2` without creating a scheduler.
- The previous G200 OVMOT dedicated attempt is also preserved as failed: its log shows 500+ second OVMOT frames and a request timeout at frame 146 while GPU7 MOT/RMOT work was still active. Started durable `queue_g200_ovmot_dedicated_retry.py` (PID 3518544), which waits for G200 MOT20/RMOT termination and a 32-GiB free-memory gate, then starts one dedicated service on port 8243 and resumes the preserved causal prefix into `g200_ovmot_dedicated_gpu7_v2`.
- No existing process was killed, no failed output was overwritten, and no metric claim was made from either failed run.

## 2026-10-23 — OVMOT request-timeout fix
- The G200 OVMOT retry needs a bounded timeout larger than the previous 600-second hard-coded HTTP limit. Added an explicit `--grounding-request-timeout` option (default remains 600 seconds) to the tracked `run_video.py`/`run_g2_parallel_scope.py` path and set the queued dedicated G200 retry to 900 seconds.
- Applied the same timeout-only change to the runtime copy under `/data2/user/LocateMOT_GroundJEV`; the runtime's existing repaired parser implementation was preserved. Runtime `--help`, import, and constructor checks passed. Existing active clients keep their original default and were not restarted.

## 2026-10-23 — held-out gate extended to new parallel controls
- The durable orchestrator was restarted once after a code change (the old process was verified as our PID and was still in the pre-finalizer waiting stage). Its gate now includes the running G200 AnimalTrack owner and both retry queue state files, so native OneJev cannot claim GPU5/7 while the newly requested controls or their recoverable retries are still active.
- The restarted orchestrator is PID 3525198 and remains in `WAITING_ALL_REGISTERED_HELDOUT_OWNERS`; no finalizer or OneJev process was started by this restart.

## 2026-10-23 — receipt collector repair
- Audited `scripts/groundjev/collect_g2_heldout_receipts.py` before the finalizers can run. Two default owner entries referenced an undefined `RUNTIME` symbol, and one repaired OVMOT owner was duplicated. Replaced the stale symbol with `RUNTIME_ROOT` and removed the duplicate; the collector now compiles cleanly and preserves one record per owner. The expensive full model-hash smoke invocation was not used as an acceptance result.

## 2026-10-23 — reproducible collector for the two extra controls
- Added `scripts/groundjev/collect_extra_shared_control_receipts.py`, a non-blocking collector for the requested G1 DanceTrack/G200 AnimalTrack controls and their retry receipts. It records model/manifest hashes and official metrics only when a complete result block exists; queue states, failures, and partial outputs remain status evidence. A one-owner queue-state smoke check passed (`accepted_scope_count=0`, as required).

## 2026-10-23 — automatic post-run collection queued
- Started durable `queue_collect_extra_shared_controls.py` (PID 3535083). It waits for the G200 AnimalTrack owner and both G1/G200 retry queue states to become terminal, then writes `extra_shared_controls_final_receipt.json` through the non-blocking collector. It does not start a GPU service and does not affect the OneJev gate.

## 2026-10-23 — stopped accidental full collector smoke processes
- A prior collector smoke command that hashed the full model for every owner was still running (PIDs 3528263 and 3529284) and had never produced a receipt. Both command lines were verified as this project's `collect_g2_heldout_receipts.py` processes and were terminated explicitly; all GPU inference services, retry queues, and the durable extra-control collector were left untouched.
- The collector fix remains compile-tested; the full hash smoke is not accepted as an experiment result. Future collection is performed only by the bounded extra-control queue after its owners reach terminal states.
