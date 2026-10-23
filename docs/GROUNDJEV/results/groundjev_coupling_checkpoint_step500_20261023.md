# GroundJEV native joint checkpoint — step 500

- Real native MOT/RMOT/OVMOT joint training remains `REAL_NATIVE_THREE_TASK_TRAINING_RUNNING` on GPU3/GPU7.
- Immutable compact manifest: 131,618 rows, SHA-256 `7c92778fb4417ea1e49bf286cab7758ceef1443d683ab059e5e758e5bd823aa3`.
- `latest_training_state.pt`: 130383561 bytes, SHA-256 `bc5146c2d333fab05ed95fb1695f3a8531125e39e628de35cb74ed1ad58a2517`; CPU load succeeded with `weights_only=False`.
- State metadata confirms step 500, seen examples 8000, FP32 masters/Adam, scheduler and all RNG state saved.
- Adapter: `adapter-step500/adapter_model.safetensors`, 43346432 bytes, SHA-256 `e33232bdd5c1f74194d5764ef3c80e40f836b8ea2d45c8f5bd38329dfd0e5aad`, 10,822,656 trainable parameters.
- This is a recovery/training-health receipt; no independent JEV or tracking metric is claimed yet.
