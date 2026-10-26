# Real GPU pilot (interim; not complete)

Old evaluation is PAUSED_BY_USER_FOR_ARCHITECTURE_PIVOT. Main matched G2/G0 image-only heads are training on GPUs4/6 for1,000 steps each, four images/step, identical seed42/initialization/order, AdamW2e-4. G2 final18227 SHA and G0 SHA were verified. Frozen visual parameters428,482,048; trainable head parameters3,975,829. No backbone or language decoder update.

At750 steps the global-query head has macro domain AP50 below1.5% on480 TRAIN-internal DEV frames (G2~0.819%, G0~1.446%); fixed0.5-threshold coverage is zero. Exact intermediate metrics are in INTERIM_TRAINING_RECEIPT.json. These20 DEV videos were seen by G2 fine-tuning. No whole-video DEV or external head results are claimed yet.

A separate real GPU1 two-TRAIN-image overfit diagnosis reached same-frame perfect R75/Precision in300 updates/75.35seconds. It is a capacity/geometry sanity check, not scientific validation. No weights from that diagnostic are transferred to either main arm.

Before seeing any external head metrics, freeze one additional matched spatial-prior control: Gaussian additive attention bias around each learned reference center, sigma0.08; same parameter count, data, initialization, optimizer and1,000-step budget. This tests spatial readout locality, with no GT/proposal query input and no new data. G0/G2 controls use GPUs7/1; at most four physical GPUs combined (4/6/1/7). Combined spatial cache limit20GiB, below25GiB. A whole-task24GPU-hour owner budget includes both original and spatial-prior pairs, with one GPU-hour reserved for functional audits. See SPATIAL_PRIOR_FROZEN_PLAN.json and SPATIAL_PRIOR_GPU_SMOKE.json.

Final training/whole-video metrics, checkpoints and research decision remain pending. Preserve original global-query results as a failed/weak intermediate, not a hidden discarded run.
