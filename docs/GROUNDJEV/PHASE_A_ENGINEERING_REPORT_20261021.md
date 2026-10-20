# GroundJEV Phase A engineering report — 2026-10-21

The first real single-backbone smoke completed in the isolated LocateMOT path. The official GroundAnything visual module was loaded from the G2 base checkpoint whose registered model SHA is `3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2`. One MOT17 image was encoded once, two real boxes were pooled into 2 x 2560 region tokens, and a typed GroundJEV readout ran Identity Choice, Admit, Yes/No, Lifecycle and Memory heads with a real GPU backward and optimizer step.

The receipt reports finite losses, a finite readout gradient, 3.07 GB peak allocated memory, 14.43 seconds wall time, `bridge_forward_count=1`, and zero independent VLM instances. CPU tests also passed for dynamic identity slots, safe non-destructive termination, causal memory timestamps, one-to-one assignment and the single-forward runtime contract.

This is an engineering result only. It does not claim a tracking metric, language hidden-state reuse, identity-conditioned box generation, or scientific gain. The current Phase A bridge uses the official G2 visual/region evidence and leaves contextual language states optional; integrating the native G2 DLM hidden-state return is the next bounded implementation step. The formal G2 Clean optimizer and its checkpoint files were not modified.

The first attempt in the registered G2 training environment failed before model construction because that environment lacks `pybase64`, which the vendored GroundAnything serving import requires. The successful run used the existing GroundAnything serving environment with the same isolated checkpoint. This dependency boundary remains recorded rather than hidden.
