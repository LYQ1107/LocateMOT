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
