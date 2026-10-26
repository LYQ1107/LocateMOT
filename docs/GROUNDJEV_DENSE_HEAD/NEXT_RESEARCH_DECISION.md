# Evidence-based next research decision

**Do not replace native final G2 or start full temporal identity training from this pilot.** The spatial attention prior improves the weak global-query readout, but coverage and false positives remain substantially worse than native G2. The matched G0 readout also has slightly higher external AP50 than G2 in both domains. This completes the bounded detector experiment, not the unified GroundJEV research goal.

Scope:20 full benchmark DEV sequences /8,370 frames, independently held out from the head optimizer but previously seen in G2 TRAIN; plus a frozen6-sequence /1,607-frame external comparison disjoint from known head/G2 TRAIN video sources. External videos were used in earlier native G2/tracking research, so they are not an untouched final TEST. External thresholds and checkpoint choices were never retuned.

1. **Do G2 spatial features support detection?** Yes for a working, trainable image-to-scored-box path; this1,000-step pilot does not demonstrate a competitive detector. Frozen428,482,048 native encoder/projector parameters feed3,975,829 trained decoder parameters. Final G2 step18227 is the actual loaded backbone. No box answer, GT-derived proposal, ROI seed or future frame enters feature extraction. Geometry, gradient, frozen-state and restore checks passed on real GPU images. Perfect memorization of two TRAIN frames is only a capacity sanity check.

|Frozen external scope|Native G2 R50 / FP-frame|Spatial G0 R50 / FP-frame|Spatial G2 R50 / FP-frame|
|---|---:|---:|---:|
|animaltrack|67.582% / 2.801|45.694% / 21.801|44.700% / 21.717|
|sportsmot|98.697% / 1.632|30.317% / 11.340|28.209% / 10.533|

2. **Which native errors improve?** Numeric parallel boxes remove the native language grammar/parser step. They recover some native misses, including rejected frames, but lose substantially more existing coverage. These are target-level diagnostics, not a union detector or GT-assisted inference.

|Spatial G2 vs native|Newly covered native misses|Lost native coverage|Recovered after native frame rejection|
|---|---:|---:|---:|
|animaltrack|1941|5926|711|
|sportsmot|17|5969|0|

3. **Is there a fair G2-specific improvement?** Not established. The primary spatial G0/G2 pair both selected step1000/threshold0.3, with identical initialization,4,000 image exposures/2,400 unique inputs, label stream,100 queries,3 layers, AdamW schedule and losses. Their encoder is identical; only five projector tensors differ. Qwen LoRA is not invoked by this image-only path. Global G0 selected750 versus global G2 at1000 under the same selection rule, so that original pair is equal-budget rather than identical evaluated-update count.

|External domain|G0 spatial AP50|G2 spatial AP50|G2 minus G0|
|---|---:|---:|---:|
|animaltrack|23.181%|22.417%|-0.763 pp|
|sportsmot|12.817%|12.015%|-0.802 pp|

The spatial prior is the sole numerical change from global queries; feature-cache limits change wall time, not updates. G2 spatial AP50 improves over G2 global from4.242% to22.417% on Animal and3.026% to12.015% on Sports, while remaining inadequate. Native A0 has no per-box scores, so its AP is undefined. A0 vs a newly supervised head also changes optimization/readout supervision; it cannot isolate architecture alone from additional supervised learning. No extra distinct GT dataset was added and no external label entered an optimizer.

4. **AnimalTrack positive result?** No net coverage improvement over native G2: 44.700% versus67.582%, Precision30.235% versus83.433%, duplicates2261 versus509. Compared with spatial G0, G2 has lower R50/AP50 but slightly higher R75/mAP; this mixed evidence is not a G2 feature advantage. The external scope covers only chicken/deer/duck.

|External animal type|GT boxes|Native G2 R50|Spatial G0 R50|Spatial G2 R50|
|---|---:|---:|---:|---:|
|chicken|6731|61.789%|25.806%|23.949%|
|deer|4861|77.618%|70.376%|71.261%|
|duck|5824|65.900%|48.077%|46.514%|

Full internal Animal DEV covers all10 types, one held-out physical video per type; these videos remain G2-TRAIN-scope.

|Internal DEV animal type|GT boxes|Spatial G0 R50|Spatial G2 R50|
|---|---:|---:|---:|
|chicken|7695|36.894%|31.332%|
|deer|14815|23.604%|21.181%|
|dolphin|8301|29.996%|29.804%|
|duck|5409|50.009%|47.402%|
|goose|3094|7.207%|5.850%|
|horse|5603|59.629%|60.361%|
|penguin|3732|9.512%|13.478%|
|pig|6761|59.311%|57.551%|
|rabbit|5496|21.852%|20.724%|
|zebra|2386|39.941%|44.216%|

5. **SportsMOT advantage retained?** No: external coverage falls from98.697% to28.209%, with FP/frame rising from1.632 to10.533. Full internal DEV small objects have0/97 coverage and dense scenes R50=15.351%; Animal medium objects R50=5.629%. These failures point to unresolved localization/readout learning and objectness, not an identity decision issue. All groups and one-to-one metrics remain in PILOT_RESULTS.json.

At fixed0.5, the spatial G2 external output contains zero Sports boxes and only19 Animal boxes. The0.3 threshold was selected from the frozen internal-DEV grid; even at that operating point coverage/FP are inadequate. Do not lower thresholds using external GT or inflate recall with duplicate boxes.

6. **Add TAO/Refer supervision now?** Keep the audited adapters for a later bounded localization study, not full open-vocabulary/identity training. Existing TAO TRAIN has177 Base classes/53,028 non-crowd annotations;1,611 Novel annotations remain excluded. Refer physical labels must not turn non-expression objects into background;157 missing referenced IDs need explicit repair/exclusion before expression training. The two real TRAIN diagnostic images produced finite direct coordinates but TAO covered0/6 targets and Refer2/6, with zero optimizer updates. This is no AP/TETA,1203-class, amodal-regression or language-satisfaction validation.

7. **Enter shared Temporal Identity Decoder?** Defer full training. The current detector loses most Sports coverage and produces too many false positives to provide a useful shared identity front end. Keep the ObjectState boxes/scores/query visual tokens/geometry and reserved language/category/history/causal-state fields. There is no need to regenerate G2 autoregressive candidates for this future path. Original J1/Coupled checkpoints are retained; no old JEV/OneJev training or new HOTA claim occurred.

8. **Unfreeze projector/LoRA?** Do not start full G2 tuning from this result. A real TRAIN detection loss backpropagated into the online G2 projector (fc2 gradient norm4.40888), with frozen encoder and zero parameter updates. That proves an available gradient path, not successful joint training. Current training used detached frozen features. If continuing, preregister one bounded study of readout optimization and spatial resolution/pre-merge features on TRAIN-internal DEV before reconsidering projector updates. Current evidence does not separate short-training underfit, representation resolution and readout design, and does not prove G2 features intrinsically unusable. Qwen LoRA still needs a genuine answer-free causal path and its own resource audit.

Four measured train+evaluation residencies total11.6844 GPU-hours; with the full1GPU-hour functional-audit reserve the conservative upper estimate is12.6844, below24. Primary G2 fresh external end-to-end FPS=1.508; native visual extraction accounts for98.58% of vision+decoder time. Observed speed is under shared load, not a controlled isolated speedup benchmark. Peak allocated memory is3010.7MiB per model on A800 GPUs.

Old full evaluation remains PAUSED_BY_USER_FOR_ARCHITECTURE_PIVOT. All old caches/weights and consistent SQLite snapshots remain intact; no automatic recovery. The bounded pilot owners exited successfully. Exact checkpoints, source-launch SHAs, failed/weak scopes and complete video metrics are preserved.
