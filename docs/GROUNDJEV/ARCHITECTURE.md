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
