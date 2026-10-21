# Phase C — dense-answer GroundJEV coupling on DanceTrack

The frozen G2 Clean adapter at step 17,000 was coupled to a trainable
`IdentityBoxStateTokenProjector`, `MemoryTokenProjector`, and typed JEV
readout.  Every training row contained all legal person boxes in the frame;
no target-only answer was used.  The run used two short TRAIN clips,
DanceTrack-0001 and DanceTrack-0016, two transitions total, and two optimizer
steps.  The G2 backbone was frozen and encoded each of the six frames once.

Grounding teacher-forcing loss changed from 3.83649 to 3.83393, while the
small JEV structured loss changed from 4.11511 to 1.14514.  The four typed JEV
heads were already saturated on this two-transition sample; this is a
learnability check, not an identity benchmark.

The trained memory projector was then connected to the official
`DLMInferenceEngine` through the marker-slot adapter and compared with the
same current-frame prompt without memory:

| clip transition | dense boxes | C0 pairs / recall@0.5 | C2 pairs / recall@0.5 |
| --- | ---: | ---: | ---: |
| DanceTrack-0001 1→2 | 7 | 1 / 0.1429 | 2 / 0.2857 |
| DanceTrack-0001 2→3 (development) | 7 | 1 / 0.1429 | 0 / 0.0000 (malformed) |
| DanceTrack-0016 1→2 | 6 | 1 / 0.0000 | 0 / 0.0000 (malformed) |
| DanceTrack-0016 2→3 (development) | 6 | 1 / 0.0000 | 0 / 0.0000 (malformed) |

The single doubled recall is a bounded training-clip observation; the
development transitions and the second video fail the output contract.  No
MOT/RMOT/OVMOT gain or generalization claim is made.  Dense MOT20 rows were
also attempted, but eager causal teacher forcing reached about 63.8 GiB on
GPU7 before the external 15.4 GiB load left no allocation headroom; that
failure is recorded as a resource boundary, not hidden by shortening the
answer contract.

Training receipt:
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_g2_dense_coupling_dance_step17000_v1/receipt.json`.

Official coupled evaluation receipt:
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_c_real_g2_official_dlm_dense_coupled_dance_step17000_v1/receipt.json`.
