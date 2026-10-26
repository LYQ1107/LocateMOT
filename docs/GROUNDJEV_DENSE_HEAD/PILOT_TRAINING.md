# Real GPU pilot training

**All four bounded trials completed1,000 optimizer updates and their complete video evaluations. The primary spatial G2 head does not reach native G2 detection quality; do not replace the front end or start full identity training from this result.**

Primary comparison: spatial-attention G0/G2 both select step1000 and threshold0.3 from internal DEV; native A0 is reused from frozen caches. Complete evaluation per model:8,370 head-held-out DEV frames /20 benchmark sequences plus1,607 frozen external frames /6 sequences. Internal DEV was in G2 TRAIN; external scope was used in prior native research, not untouched final TEST.

All four measured train+evaluation residencies: **11.6844 GPU-hours**; including the full1GPU-hour functional reserve, conservative upper estimate **12.6844**, within24. Hardware: NVIDIA A800-SXM4-80GB, at most four physical GPUs. Peak allocated GPU memory3010.7MiB per model. Overall temporary feature cache limit20GiB.

See NEXT_RESEARCH_DECISION.md for the measured negative result and eight research decisions; PILOT_RESULTS.json retains every model/video/group, including fixed0.5 operating points. Native AP is undefined; dense AP uses learned objectness under a documented class-agnostic protocol.

## Original global-query pair (retained weak baseline)

**Completed the bounded detector pilot. This is not completion of the unified temporal identity research goal.**

Architecture: frozen native MoonViT+checkpoint-specific projector (428,482,048 parameters), 256-d projection and position,100 learned object queries,3 Transformer decoder layers,3,975,829 trainable head parameters; focal objectness+5L1+2GIoU with Hungarian assignment and auxiliary layer supervision. No GT proposals, no language answer input, no original detector boxes as query seeds.

Matched optimizer: AdamW LR2e-4, weight decay1e-4,50-step warmup,0.1 LR multiplier from800;4 images/optimizer step,1,000 steps each, seed42. Alternate Sports/Animal, uniformly sample physical videos. Both initialization SHA and complete sampling-order SHA are identical. Each arm sees2,000 Sports and2,000 Animal exposures; exact unique count in receipts. The feature cache is at most6GiB per arm,12GiB combined. Native image processor max1024 merged tokens; encoder and projector never updated.

GPU4: G2, UUID GPU-44f62835-4095-23ec-d3dd-5af16a00705c. GPU6: G0, UUID GPU-883c8be3-25b0-3d19-6fa8-4c2e335e67d2. Old evaluation and all recovery launchers were stopped before these runs. Source commit and final-G2 SHA are in the registration/smoke receipts.

Checkpoint selection: macro domain AP50 on480 TRAIN-internal DEV frames every250 steps; threshold from a fixed0.05/0.1/0.2/0.3/0.5/0.7 grid maximizing macro domain F1. Both arms have identical choices and budgets. Complete8,370-frame/20-video internal DEV is then evaluated with the selected checkpoint. These DEV videos were seen by G2 fine-tuning; never call them independent of G2. A separately frozen1,607-frame/6-video comparison uses prior native G2 caches and no test-time tuning.

## Complete internal DEV (percent except counts)

|Model / dataset|Frames|R50|R75|Precision|FP/frame|FN|Duplicates|AP50 / AP75 / mAP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|g0 / animaltrack|2996|8.524|0.299|5.530|30.098|58014|511|1.105 / 0.006 / 0.179|
|g0 / sportsmot|5374|6.978|0.447|4.137|20.527|65859|101|0.907 / 0.083 / 0.209|
|g2 / animaltrack|2996|19.625|0.905|5.260|73.167|51121|934|1.328 / 0.006 / 0.216|
|g2 / sportsmot|5374|14.958|0.869|3.095|60.090|60305|148|0.651 / 0.005 / 0.121|

## Frozen video-disjoint cached comparison (percent except counts)

|Model / dataset|Frames|R50|R75|Precision|FP/frame|FN|Duplicates|AP50 / AP75 / mAP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A0 / animaltrack|809|67.582|47.732|83.433|2.801|6004|509|undefined (unscored native)|
|A0 / sportsmot|798|98.697|88.134|86.292|1.632|248|140|undefined (unscored native)|
|A1 / animaltrack|809|10.852|0.132|8.986|22.811|15594|220|3.269 / 0.007 / 0.521|
|A1 / sportsmot|798|12.470|0.687|7.120|16.429|7439|32|1.880 / 0.009 / 0.334|
|A2 / animaltrack|809|39.119|3.554|10.516|69.376|10820|1045|4.242 / 0.041 / 0.795|
|A2 / sportsmot|798|27.404|1.788|5.239|50.387|6221|32|3.026 / 0.012 / 0.738|

R50/R75 above are target coverage for compatibility with original G2 recall. One-to-one recall, FP, mean best IoU, size/density/category groups and complete per-video metrics are in PILOT_RESULTS.json. Precision/FN/FP use one-to-one matching; duplicate detections cannot manufacture matched recall. AP uses genuine dense-head objectness and class-agnostic101-point matching, not official COCO. A0 AP is undefined and not compared.

|Arm|Best step|Threshold|Peak allocated MiB|Peak reserved MiB|GPU-hours (train+eval residency)|Vision FPS / Decoder FPS / vision+decoder FPS|End-to-end wall FPS (fresh full DEV)|
|---|---:|---:|---:|---:|---:|---:|---:|
|g0|750|0.3|3010.7|3594.0|3.2740|1.394 / 101.646 / 1.375|1.321|
|g2|1000|0.2|3010.7|3594.0|2.9701|1.537 / 105.520 / 1.515|1.451|

Original global-pair measured train+evaluation residency subtotal: 6.2441 GPU-hours, plus separately bounded short smoke/feature audits. Large weights and full predictions remain under `/data3/user/LocateMOT-runs/G2_Spatial_DenseHead_20261026`. Git includes only summaries/indices/configuration. Native A0 time is historical shared-load inference; dense timing is current shared-load inference. Equal images and A800 hardware class do not imply matched concurrent load; treat speed as observed end-to-end evidence, not an isolated kernel benchmark.

Loss curves and sampled-DEV curves are exported as PILOT_CURVES.png. Loss decrease alone is not a scientific gain. No long G2 unfreezing, OneJev SFT, identity decoder training, RL or new large data downloads occurred.

Reproduce: `python tools/groundjev/dense_head/prepare_train_data.py`; `python tools/groundjev/dense_head/prepare_external_comparison.py`; run the two GPU leakage audits, compare spatial features, then `python tools/groundjev/dense_head/run_matched_pilot.py` in GroundAnything environment configuration. Local assets use existing server read-only links; external repository/environment pins are in the inherited dependency tools and environment. Single-arm resume supports optimizer/scheduler/sampler/RNG restoration via `train_dense_head.py --config ... --resume ...`; do not run two owners for an arm. The completed1,000-step runs must be archived before a new trial directory is used.

Known limits: benchmark target-set foreground is class-agnostic, not exhaustive annotation of every semantic object. TAO/Refer partial labels are not in optimization. Refer audit finds157 absent referenced IDs; future supervision must exclude/repair those documented rows. Optional SAM checkpoint/runtime was not registered, so no SAM performance claim. No JEV/HOTA gain is inferred from this detection experiment.

## Pre-frozen spatial attention controls

Weak global-query DEV AP and successful two-image overfit motivated one additional bounded matched pair. Before reading any external head metrics, sigma0.08 Gaussian additive attention bias at the model reference centers was frozen. No GT/proposal enters the prior and no trainable parameter was added. Same100 queries/3 layers/seed42/initial weights/sampling stream/1,000 steps/optimizer/LR. Entire sampling SHA, unique optimizer images and domain exposures match across all four runs. G2 usesGPU1 and G0GPU7; original global pair remainsGPU4/6. Combined feature cache limit20GiB (<25GiB). Whole-task budget monitors both pairs, with one GPU-hour conservatively reserved for functional audits. Original results/checkpoints remain preserved.

### Full internal DEV (still seen by G2 TRAIN)

|Architecture/model/dataset|Frames|R50|R75|Precision|FP/frame|FN|Duplicates|AP50 / AP75 / mAP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|G0 spatial prior/animaltrack|2996|34.150|4.100|20.967|26.678|42088|4257|10.442 / 0.182 / 2.200|
|G0 spatial prior/sportsmot|5374|21.682|2.080|13.222|18.063|55830|686|5.448 / 0.065 / 1.103|
|G2 spatial prior/animaltrack|2996|32.766|3.305|21.462|24.763|43018|4285|10.254 / 0.149 / 2.153|
|G2 spatial prior/sportsmot|5374|20.331|1.682|13.598|16.319|56818|537|5.442 / 0.060 / 1.085|

### Same frozen external videos, no external selection

|Architecture/model/dataset|Frames|R50|R75|Precision|FP/frame|FN|Duplicates|AP50 / AP75 / mAP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|G0 spatial prior/animaltrack|809|45.694|6.282|30.634|21.801|9627|2250|23.181 / 0.639 / 5.609|
|G0 spatial prior/sportsmot|798|30.317|3.198|21.436|11.340|5975|220|12.817 / 0.214 / 2.663|
|G2 spatial prior/animaltrack|809|44.700|7.499|30.235|21.717|9802|2261|22.417 / 0.822 / 5.641|
|G2 spatial prior/sportsmot|798|28.209|3.032|21.368|10.533|6160|157|12.015 / 0.165 / 2.524|

|Arm|Best step|Threshold|Allocated MiB|GPU-hours|Fresh end-to-end detection FPS|Vision FPS|Decoder FPS|
|---|---:|---:|---:|---:|---:|---:|---:|
|g0 spatial prior|1000|0.3|3010.7|2.5775|1.782|1.880|146.827|
|g2 spatial prior|1000|0.3|3010.7|2.8627|1.598|1.693|113.799|

All four measured train+evaluation residency costs total **11.6844 GPU-hours**. The recorded global upper estimate including a full1GPU-hour functional-audit reserve is 12.6844, within24. GPU diagnostics and raw weights remain on server. No external frame/query was used in an optimizer update.

Source provenance: original global heads launched at7829ff66; spatial-prior heads launched at5254d8bc. Their launch registrations are authoritative. The entrypoint also stored current Git HEAD at completion, which may be a newer reporting/control commit; it is not the launch SHA. The newer source preserves original global numerical behavior when the optional spatial prior is None. Full source SHAs and registration config are in PILOT_RESULTS.json.

All arms completed1,000 optimizer updates, but the original global G0 selected step750 while global G2 selected1000. Treat that global pair as an equal-budget/equal-selection-protocol comparison. The primary spatial-prior G0/G2 pair both selected1000 and therefore also matches evaluated update count and exact label exposures.

## Limited existing TRAIN interfaces

The frozen spatial G2 head ran two existing TRAIN images on GPU4 with zero optimizer updates and no native language decoding. This is an interface diagnostic, not a new domain benchmark. TAO uses Base-only partial positives; Refer uses all supplied physical labels.

|TRAIN image domain|Supplied target boxes|Accepted direct boxes|Coverage50 on this one TRAIN image|
|---|---:|---:|---:|
|tao_base|6|22|0.000%|
|refer_physical|6|19|33.333%|

Both coordinate outputs and positive-only losses were finite; TAO localization was not solved by bypassing the parser. Additional diagnostic residency: 0.003811 GPU-hours, included within the one-hour functional-audit reserve. See AUXILIARY_GPU_DIAGNOSTIC.json. No AP, TETA, open-category accuracy or language-satisfaction claim is made.

## Observed timing on the common external scope

Native cached animaltrack: 0.203 request FPS; 69/809 rejected frames. Native cached sportsmot: 0.253 request FPS; 0/798 rejected frames. These are historical serving request times including its concurrent load, not an isolated vision benchmark.

|Readout|Fresh external vision FPS|Decoder FPS|End-to-end wall FPS|
|---|---:|---:|---:|
|g0 global|1.371|99.410|1.247|
|g0 spatial prior|1.862|147.278|1.680|
|g2 global|1.512|107.447|1.372|
|g2 spatial prior|1.681|116.713|1.508|

The G2 spatial model spends 98.58% of measured vision+decoder time in native visual extraction. Its parallel decoder is already a small part of total latency. Different concurrent GPU loads remain a limitation; no controlled speedup ratio is inferred.

## Preserved checkpoint index

All paths below are relative to `/data3/user/LocateMOT-runs/G2_Spatial_DenseHead_20261026/`. Each directory preserves `best_dev.pt` and `last.pt`; every last checkpoint is step1000. No weights are uploaded. Backbone model identities and full-file hashes are in SPATIAL_FEATURE_AUDIT.md.

|Directory|Best step|Best SHA256|Last SHA256|
|---|---:|---|---|
|`g0`|750|`91e98e29b949224d2cc3271a5a28377ef306aaea5abeab5ddf6839969098c8b3`|`d5e3b0a31bc22024b9648e17306ca98ae0164ed6a2f38a7111ae3dd2e985facc`|
|`spatial_prior_v2/g0`|1000|`b7693376e46102a156b914781f8dabad42f617e4f5cb8289982130763fb2a832`|`03adf283e38f5e1904056b499340472f2c8570c40036384b8d3673437f61cf94`|
|`g2`|1000|`0133e58b88032e28d385a0cf766481a04e56cb7a442c7782e1f892e45a8e2d0f`|`b028f06aa1ccaa2525c707eaf8d796bad00c874463a37f38c904adbb7df89926`|
|`spatial_prior_v2/g2`|1000|`c8ebe445a291e551662333cb826d5253bfafce338a96167069bde4db09854317`|`79c834e73ce4e930a7664884afdbaa85a66375ca5bb51f01f243ebbefb00d9af`|
