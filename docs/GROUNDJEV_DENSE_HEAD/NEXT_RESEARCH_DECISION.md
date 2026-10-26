# Research decision pending bounded controls

Global-query training shows weak DEV generalization despite successful two-image TRAIN overfit. No spatial-detection success or identity benefit is established. Freeze one spatial-attention prior control before reading external head results; maintain identical G0/G2 training protocol. Complete both1,000-step pairs and whole-video DEV/frozen external comparison within24GPU-hours before deciding whether to enter shared temporal identity training. G2 and original checkpoints remain preserved; no old evaluation is resumed.
