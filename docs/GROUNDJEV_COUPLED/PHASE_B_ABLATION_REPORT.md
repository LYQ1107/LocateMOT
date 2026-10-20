# Phase B fixed-candidate readout ablations

This is the first bounded Phase-B comparison after the real G2 readout pilot.
It is an engineering/development study, not a MOT/RMOT/OVMOT benchmark.

## Fixed protocol

- Backbone: frozen GroundAnything G2 step-11000 visual path, adapter SHA
  `f2b7199f57dae156a05e1efcadfa04b275a7f0399187b264e0b21ddc1b58409b`.
- Data: 24 legal TRAIN examples from the 1,116-row Phase-B corpus. The same
  examples, candidate boxes, causal histories, initialization seed, and 200
  updates were used in all arms. The set contains 7 native identity rows, 9
  native lifecycle rows, and 8 native RMOT verifier rows; it includes both
  binary labels and has no MOT20-05 or official TEST labels.
- Visual work: 24 G2 visual forwards total while preparing the fixed evidence;
  every arm then reuses detached region tokens. Each readout has 1,980,682
  trainable parameters and no second VLM.

## Arms and loss observations

| Arm | Supervision | Loss step 1 → step 200 | Additional observation |
|---|---|---:|---|
| B0 | Identity Choice only | 2.6754823 → 2.4738863 | Yes/No was measured but not optimized (0.7518284 → 0.6978400). |
| B1 | Choice + Yes/No | 3.4273107 → 2.5620224 | Yes/No fell 0.7518284 → 0.0829904; choice ended at 2.4790320. |
| B2 | Choice + Yes/No + lifecycle and registered admission proxy | 4.6023855 → 2.6426239 | Yes/No 0.7518284 → 0.0812297; lifecycle 1.8435760 → 0.0000751. |

The admission target in B2 is the explicit binary proposal-verifier label for
the same candidate. No memory UPDATE/HOLD label was available in this corpus,
so memory loss was correctly masked rather than invented. The B2 checkpoint is
therefore not yet a full lifecycle-and-memory training result.

The first attempted receipt (`PHASE_B_ABLATIONS_BIASED_RECEIPT_20261021.json`)
selected only identity-positive examples. It is preserved as a protocol
failure and excluded from conclusions. A second corrected run used the right
strata but different seeds; the final run used the same seed `20261021` for
all three arms, and that final receipt is the one used here.

## Causal replay

The three trained readouts were loaded into independent causal runtimes and
replayed on the same eight frames (83–90) of MOT20-03 TRAIN. Ground-truth
boxes supplied the registered candidate engineering gate; they were not used
as a benchmark prediction or training target. One shared G2 visual state was
created per frame, for 8 visual forwards total across all three arms.

The replay produced the following raw assignment counts:

| Arm | REJECT | NO_MATCH | NEW | Existing-ID assignments | Final tracks |
|---|---:|---:|---:|---:|---:|
| B0 | 11 | 6 | 1 | 6 | 1 |
| B1 | 0 | 9 | 3 | 12 | 3 |
| B2 | 15 | 0 | 2 | 7 | 2 |

These counts show that the trained heads execute the complete NEW/MATCH/
NO_MATCH/REJECT and memory loop, but they are not accuracy measures. The
different B1/B2 admission and lifecycle distributions expose remaining
calibration and supervision problems; they must not be presented as a
tracking improvement.
The full raw replay is retained outside GitHub at
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_b_replay_v2/phase_b_replay_receipt.json`;
the compact receipt is `PHASE_B_REPLAY_SUMMARY_20261021.json`.

## Next valid experiment

Build a larger video-disjoint TRAIN/DEV corpus with explicit memory
UPDATE/HOLD targets, calibrate admission on the same fixed candidates, and
run a complete development video with official MOT evaluation inputs before
making any HOTA/AssA/IDF1/IDSW statement. Phase C memory-conditioned
Grounding remains pending.
