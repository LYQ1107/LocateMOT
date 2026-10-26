# Dataset protocol

TRAIN-only supervision: SportsMOT TRAIN45 clips /12 original source IDs; AnimalTrack TRAIN32 physical videos /10 animal types. Sports clips from a common original YouTube ID are inseparable. Hash-order split with seed42 holds out two Sports source groups and one video per Animal type. This yields57 train videos and20 DEV videos; head train/dev have no source-group or representative image hash overlap. All remain within the legal original TRAIN split.

**These DEV videos were already used in G2 fine-tuning.** They are independent of detector-head optimization, not independent of G2. No claim of new-backbone generalization may use these internal DEV numbers. Frozen external cached scopes are video-disjoint from G2/head TRAIN, were used in prior native G2 research, and are evaluated once after freezing internal-DEV selection.

|Index|Frames|Targets|Videos|
|---|---:|---:|---:|
|train|3648|46423|57|
|dev|480|7997|20|
|dev_full|8370|133912|20|

Pilot: uniformly sample64 frames per training video into a read-only index; sample24 DEV frames per video for checkpoint/threshold selection. Final DEV evaluates every frame of the20 complete DEV videos (8,370 total). Optimizer alternates domains, samples a video uniformly, then a frame; no long-video domination. Both arms use identical sampler stream, seed, data, model size and optimizer steps. Actual unique images/exposure counts are recorded after training. No dataset is copied.

GT: original MOT xywh with mark/class validity for Sports, valid Animal target records; positive boxes clipped to image boundaries; original ignore objects remain ignored. Empty annotated frames are valid background. Non-exhaustive data disables unmatched negatives in the criterion. This pilot does not optimize TAO or Refer subsets, which have different annotation completeness.

TAO-Amodal: existing TRAIN Base labels only; G2 manifest records30,781 Base category rows and1,611 Novel annotations excluded. Novel labels remain excluded from training. Before expansion use category-aware negative eligibility or positive-only loss; category subsets cannot define generic background. No1203-class language decoding is launched.

Refer-KITTI-V2: legal fit videos0000/0002/0003/0006/0008/0010/0012/0014/0017. Expression labels name Track IDs and reference per-frame physical labels; xywh labels normalize to image width/height. Future detector supervision must collect all valid physical objects, not only expression targets. Text-conditioned selection is a later stage.

MOT17 detector variants share physical video; RMOT26 includes MOT17 and DanceTrack copies, including local DanceTrack validation content in historical G2 TRAIN. These sources do not enter the new head optimization. No generic no-leakage claim is made for them. Sports grouping uses raw source ID; Animal source provenance is physical file identity plus three representative SHA checks per video, not an unproven unseen-camera split.

Frozen A0/A1/A2 external protocol:3 Sports source videos and3 different animal categories,1,607 frames total. Exact IDs/index SHA in FROZEN_EXTERNAL_PROTOCOL.json. A0 uses preserved final-G2 JSONL, no generation. Common coverage/one-to-one recall/FP/duplicates use exactly the same frames and GT. Native per-box scores are absent, so A0 AP is undefined. Dense head AP uses genuine objectness under documented class-agnostic101-point protocol.

Full video IDs, GT hashes, image dimensions, representative content hashes, source groups, target/identity counts and index SHAs are in DATASET_PROTOCOL.json. Raw indexed frame-level labels stay on server. No external DEV/TEST labels enter optimization. No new datasets or large model downloads.

Auxiliary audit completed on existing data: TAO TRAIN has177 Base classes with53,028 non-crowd annotations;1,611 Novel annotations excluded. Refer legal fit physical labels contain5,508 object rows across1,646 files;4,350 expressions /379,341 query-frames.157 referenced target IDs are absent from the corresponding supplied physical-label file and are an explicit boundary for future training. This pilot optimizes neither dataset. Official KITTI coordinate contract is normalized top-left xywh (confirmed in the upstream RMOT loader/generator), not inferred from local path spelling. Full counts and examples in AUXILIARY_DATA_AUDIT.json.
