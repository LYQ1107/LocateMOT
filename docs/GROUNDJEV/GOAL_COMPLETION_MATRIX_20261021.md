# GroundJEV execution matrix — 2026-10-21

| Work item | State | Evidence / boundary |
|---|---|---|
| G0 MOT20-05 long export | Complete registered validation-half | 1,657 decoded frames; fresh native-source control retained. This is not a full MOT20 TEST claim. |
| Independent MOT-only long training | Safely complete/no live owner | No live GroundJEV MOT-only optimizer process. Historical outputs remain preserved. |
| Formal G2 Clean | Running | Latest authoritative event step 12,326/18,227; `checkpoints/adapter-step012000.pt` and full training-state checkpoint are present; final receipt absent; GPUs 3/5 protected; linear remainder about 19.0 h. |
| G1 GPU5 adaptation | Waiting | Registered owner waits for exact G2 terminal receipt; no real G1 optimizer update yet. |
| G2 clean deployment bundle | Waiting | Registered CPU owner waits for terminal G2 and G1 receipts; no overwrite/export yet. |
| Independent held-out MOT/RMOT/OVMOT | Waiting | MOT20-05, RMOT single/multi and TAO manifests preflighted; official metrics pending. |
| Legacy independent OneJev resume | Paused | Waiting owner stopped before GPU launch at durable step 3,000; checkpoint, optimizer, scheduler, sampler and RNG preserved. |
| Corrected MOT/RMOT/OVMOT source registration | v2 prepared, zero-drop course queued, not trained | v2 manifest SHA `cbdafce9…e54c`; all 131,990 rows render under 32,768 (max 28,156); compact competition transform is explicit and native OVMOT category rows remain pending. |
| GroundJEV single-backbone Phase A/B | Preliminary complete in isolated branch | Real G2 forward/backward and bounded choice/Yes/No/lifecycle pilots passed engineering gates; no full benchmark gain claim. |
| GroundJEV fixed-candidate video replay | Repair diagnostic complete / independent gate pending | Initial MOT20-03 TRAIN replay: HOTA 0.027901, IDF1 0.005261, FN 14,524. Fixed 64/64 temporal TRAIN repair: HOTA 0.302815, AssA 0.388886, IDF1 0.257729, IDSW 18, FN 5,634. This improves the action-distribution failure but is not independent generalization evidence. |
| Phase C memory-conditioned Grounding input | Contract prepared, decoder/eval pending | Opt-in causal memory-token insertion and exact disabled baseline passed 8 focused tests; no generated-box recall or tracking metric claim yet. |
| YesTrack native v2 | Queued on GPU7, not trained | Native 726-row train / 390-row held-out split, zero-drop CPU plan, base-vs-LoRA evaluator and resource-gated owner registered; GPU7 was occupied at registration, so AUROC/PR-AUC and tracking metrics are pending. |
| Overall goal | Incomplete | Must wait for terminal G2, G1, held-out receipts and valid independent metrics. |
