# G2 GPU3/GPU5 resource audit

The initial G2 audit found GPU3 and GPU5 idle. A fresh audit immediately before the DDP smoke showed each card had about 62 GiB free and about 19 GiB used by pre-existing `/data2/OCD/TALON` work. Those jobs were not stopped or modified. G2-A's measured single-card peak was 39,124,467,712 bytes allocated and 40,177,238,016 bytes reserved, leaving roughly 20 GiB of instantaneous headroom per card under the observed co-resident load.

The one-step DDP gate used `CUDA_VISIBLE_DEVICES=3,5`, the project `.venv-onejev-py310`, and master port `29617`. Both ranks completed one native DLM update and wrote an atomic checkpoint. Formal training must retain the same physical-card mapping and stop before the measured reserve is threatened; no other GPU owner is to be interrupted.

Receipt paths:

- environment: `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_PROJECTOR_QWEN3_LORA_FULLDATA/environment_audit.json`;
- gradient gate: `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_PROJECTOR_QWEN3_LORA_FULLDATA/g2_a_gradient_gate.json`;
- DDP smoke: `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_PROJECTOR_QWEN3_LORA_FULLDATA/ddp_smoke_1step/`.
