# GroundJEV Phase C step17500 dense coupling regression

The frozen G2 adapter at step 17500 was coupled to the typed JEV readout, identity/box-state memory projector, and official `hierarchy_dynamic` DLM decoder. The run used two real DanceTrack sequences, frames 1–3, one trained transition per sequence, and two optimizer steps on GPU4. G2 itself was frozen and its formal two-rank run was untouched.

| Sequence | Image size | C0 recall@IoU .5 | C2 trained memory recall@IoU .5 |
|---|---:|---:|---:|
| DanceTrack-0001, 1→2 and 2→3 | 1280×720 | 1/7 (0.142857) | 2/7 (0.285714) |
| DanceTrack-0016, 1→2 and 2→3 | 1920×1080 | 1/6 (0.166667) | 2/6 (0.333333) |

The evaluator had a real coordinate-contract bug: DanceTrack-0016 was previously evaluated as 1280×720 even though its `seqinfo.ini` declares 1920×1080. The hardcoded dimensions were corrected before this receipt, and `py_compile` plus the official evaluator passed.

This is a bounded decoder/readout regression, not a tracking benchmark. It reports no HOTA, IDF1, TETA, or held-out generalization claim. The full receipt is at `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_g2_official_dlm_dense_coupled_dance_step17500_v1/receipt.json`.
