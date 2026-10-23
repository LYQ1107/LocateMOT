# GroundJEV native joint checkpoint — step 1000

- Real native MOT/RMOT/OVMOT joint training remains `REAL_NATIVE_THREE_TASK_TRAINING_RUNNING` on GPU3/GPU7.
- Immutable compact manifest: 131,618 rows, SHA-256 `7c92778fb4417ea1e49bf286cab7758ceef1443d683ab059e5e758e5bd823aa3`.
- `latest_training_state.pt`: 130383561 bytes, SHA-256 `f3b37033f57963dce452b384503009fc587531ac05f6cc16742dd511103e5213`; trusted CPU load succeeded with `weights_only=False`.
- State metadata confirms step 1000, seen examples 16000, FP32 masters/Adam, scheduler and all RNG state saved.
- Adapter: `adapter-step1000/adapter_model.safetensors`, 43346432 bytes, SHA-256 `10a54906af4f001ba466b58ee3cc9c0655721594cc7de8f70f3938256fd505b8`, 10,822,656 trainable parameters; safetensors load succeeded.
- This is a recovery/training-health receipt; no independent JEV or tracking metric is claimed yet.
