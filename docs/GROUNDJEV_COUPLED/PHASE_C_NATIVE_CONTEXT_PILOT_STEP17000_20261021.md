# Phase C native G2 context pilot — adapter step 17000

- Status: `PHASE_C_NATIVE_G2_CONTEXT_READOUT_PILOT_COMPLETE`
- Backbone: one official GroundAnything G2 DLM; adapter step `17000`.
- GPU: physical GPU `7`; formal G2 two-rank run was not modified.
- Scope: `16` MOT17-02-FRCNN TRAIN frames, `8` train and `8` same-video temporal development frames.
- Native DLM forward count: `16`; bridge forward count: `16`.
- Optimizer updates: `80` on the typed readout.

| Split | Identity accuracy before | Identity accuracy after | Admit accuracy before | Admit accuracy after |
|---|---:|---:|---:|---:|
| Train prefix | 0.052419 | 0.750000 | 0.653226 | 0.927419 |
| Temporal development | 0.032258 | 0.536290 | 0.596774 | 0.915323 |

The G2 DLM was frozen and teacher-forced previous-frame GT-matched region tokens were used as causal memory slots. Therefore this receipt demonstrates that the single-backbone native-context typed readout receives finite gradients and can learn a bounded TRAIN control. It does **not** establish HOTA, IDF1, IDSW, RMOT/OVMOT performance, independent generalization, learned memory feedback into generated boxes, or a full end-to-end GroundJEV fine-tune.

Exact machine-readable receipt: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_native_context_step17000_v1/receipt.json`.
