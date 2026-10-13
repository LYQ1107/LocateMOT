# Architecture under implementation

Current frame + task specification → GroundAnything DLM candidates and native
visual evidence → OneJev typed decisions over dynamic historical identities
→ one-to-one global assignment → reliability-controlled persistent memory.

MOT/RMOT/OVMOT share perception, decision weights and memory schema; task
inputs and output filtering differ. Future frames and evaluation GT never
enter runtime state. GroundAnything spatial coordinates use the official
0–999 grid; missing confidence remains missing rather than a fabricated
probability. OneJev's checked official Choice limit is 255 options and Score
limit is 10 levels. Dense scenes require groups and global competition, with
measured true-option coverage.

Full initializes from OneJev-0.8B pretrained weights. Compact will combine
teacher distributions and GT with fair MLP/Set Transformer controls. Native
feature extraction and model loading are pending real-runtime validation.

Initial implemented Full inputs: current real image, candidate geometry/labels,
historical state, and a sheet of real anchor/long/recent crops. Per group,
Choice/Noul/Score branches share images. Candidate batches of 16 and a past-only
32-track shortlist avoid pairwise full-model calls. GT-option coverage must be
measured after inference before interpreting dense-scene results. Native DLM
hidden features are an explicit remaining requirement, not attributed to crops.

Global assignment uses OneJev distributions plus private NEW/IGNORE columns;
memory stores model-generated IDs and physical-time motion/lost state.
Reliable Noul decisions gate long-memory writes; anchors remain fixed, recent
observations can remain uncertain. Empty frames retain/expire existing memory.
Memory serialization preserves causal images and counters, without GT IDs.
This initial implementation is pending actual pretrained online validation.


## Native feature memory (optional V5)

The DLM environment hosts a serial native MoonViT/projector RPC on8111.
The OneJev environment sends original image bytes and candidate boxes, gets
actual2560-dimensional region/global features and local-token shape, plus
optional200-step auxiliary128-dimensional temporal embedding. This module
does not load a substitute encoder or language decoder. ROIAlign coordinates
account for the official resize/padding; reference validation is exact.

OnlineTracker native mode stores fixed anchor, recent and reliability-held
long region/temporal embeddings,16-observation causal feature history, semantic
observation counts, normalized identity entropy and lifecycle survival
probability1-P(TERMINATE). V1 crop-only serialization/output remains unchanged.
V5 sends actual anchor/long/recent similarities with existing crop+motion
evidence to the same pretrained OneJev. CSV integer/1000 matrices retain all
pairs/options and avoid pretty-JSON context growth. V4's failed56426token
request is preserved. Numerical evidence formatting is a registered variant,
not a trained Compact model or proven feature benefit.

MOT/RMOT/OVMOT use the same decision class, checkpoint and identity core.
Task specifications alter Ground prompts/vocabulary, not identity mechanics.
The corrected TAO vocabulary chunks request every class before one association step;
there is no per-class identity reset or hidden GT correction.
