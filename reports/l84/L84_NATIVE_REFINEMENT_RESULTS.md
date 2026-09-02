# L84 native refinement results

Native iterative refinement was retained only as a diagnostic state family;
the refined reference points were never emitted as L69 boxes.  Mean dev
metrics were:

| state | bag hard violation | hit@1 | V2 hard | V2 hit@1 | multi-target exact |
|---|---:|---:|---:|---:|---:|
| R1 | 0.705128 | 0.407051 | 0.766143 | 0.315881 | 0.308176 |
| R4 | 0.753205 | 0.376068 | 0.766143 | 0.375218 | 0.218029 |
| R6 | 0.754274 | 0.375000 | 0.783595 | 0.335079 | 0.257862 |

R1 is numerically tied with Z1 in the paired dev output and therefore adds no
evidence for keeping native refinement as a distinct L85 semantic state.  R4
and R6 failed the complete stable criteria (R4 bootstrap lower bound was
negative; R6 failed V2/stability conditions).  The final L85 representation
is consequently the corrected Z1 state, not a refined-box output.

No native refinement state was used to modify the frozen bank, tracker IDs or
ordinary MOT/OVMOT paths.
