# L88C resource cleanup — 2026-09-06

The L88C internal final replay was left running and its live inputs were
verified before cleanup. The active process uses the L88 cache, L88 epoch-30
checkpoint, and its new `/data2/usr_for_deadline/locatemot_l88c` output only.
No live input, frozen bank, manifest, report, or current checkpoint was moved.

The following clearly superseded intermediate L1 LoRA checkpoint directories
were moved, not deleted, to a recoverable quarantine on `/data2`:

| original | recoverable location | observed size |
|---|---|---:|
| `outputs/l1_c/checkpoints/lora/checkpoint-100` | `/data2/usr_for_deadline/locatemot_cleanup_20260906/l1_c_intermediate_checkpoints/checkpoint-100` | 8.0G |
| `outputs/l1_c/checkpoints/lora/checkpoint-200` | `/data2/usr_for_deadline/locatemot_cleanup_20260906/l1_c_intermediate_checkpoints/checkpoint-200` | 8.0G |
| `outputs/l1_c/checkpoints/lora/checkpoint-300` | `/data2/usr_for_deadline/locatemot_cleanup_20260906/l1_c_intermediate_checkpoints/checkpoint-300` | 8.0G |

The final `outputs/l1_c/checkpoints/lora/` files and UAF checkpoints were left
in place. No file was permanently deleted. `/data1` availability increased
from approximately 76G to 114G after the move; `/data2` remained ample.

This was a storage-only, recoverable action. It does not change any L88C
scientific result or evaluation protocol. The internal replay remained
zero-training and continued independently.
