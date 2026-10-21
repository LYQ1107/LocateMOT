# GroundJEV Phase B ablation receipt — 2026-10-21

This receipt records three bounded readout arms on the same GroundAnything G2
step-11000 visual path and the same 24 training examples: B0 identity choice,
B1 identity choice plus Yes/No, and B2 identity choice plus Yes/No and
lifecycle. Each arm ran 200 readout updates with 1,980,682 trainable readout
parameters. The shared corpus has 1,116 registered rows from 34 videos; its
MOT-like rows use native GroundAnything candidates, the RMOT rows use the
registered native RMOT export, and the 430 OVMOT rows are explicitly Base
teacher-category candidates.

The final losses were:

| Arm | Identity loss | Yes/No loss | Lifecycle loss | Total loss |
|---|---:|---:|---:|---:|
| B0 | 2.473886 | — | — | 2.473886 |
| B1 | 2.479032 | 0.082990 | — | 2.562022 |
| B2 | 2.480030 | 0.081230 | 0.000075 | 2.642624 |

The trained readouts were replayed causally for frames 83–90 of the MOT20-03
TRAIN engineering window using the fixed candidate gate. The resulting
assignment counts were B0: `REJECT=11, NEW=1, NO_MATCH=6, ID1=6`; B1:
`NO_MATCH=9, NEW=3, ID1=7, ID2=5`; and B2: `REJECT=15, NEW=2, ID1=4,
ID2=3`. The replay used eight cached visual forwards per arm and no second
VLM. It produced one, three and two final tracks respectively.

These results verify that the typed heads and explicit `NEW`/`NO_MATCH`/
`REJECT` actions can be trained and produce different causal decisions. They
do not establish tracking quality: the replay uses a TRAIN engineering window,
has no official metric receipt, and its OVMOT rows are not native TAO TRAIN
candidate rows. The native OVMOT gate therefore remains closed.

`tools/groundjev/coupled/build_joint_identity_corpus.py` now rebuilds this
kind of normalized corpus from registered JSONL sources while preserving
candidate provenance, causal memory, target validity and forbidden-evidence
checks. `--require-native-candidates` rejects GT/teacher rows instead of
silently relabeling them as native.
