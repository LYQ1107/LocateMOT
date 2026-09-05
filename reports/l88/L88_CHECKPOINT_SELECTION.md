# L88 Checkpoint and Rule Selection

## Selection boundary

Checkpoint and rule selection used only the registered video-disjoint internal
fit/dev evidence. The fixed 16 calibration and 24 validation units were not
read for this selection. Screening and official-test labels were not read.

The cheap development pass scored all 20 even checkpoints on 138 groups and
produced 9,960 complete records. The deduplicated shortlist was fixed before
full-video dev TrackEval:

| candidate | epoch | dev target-bag precision | dev target-bag F1 | dev distinct-recall precision | dev distinct target recall |
|---|---:|---:|---:|---:|---:|
| fixed checkpoint | 8 | 0.1683 | 0.2741 | 0.0805 | 0.9165 |
| fixed checkpoint | 20 | 0.1891 | 0.2842 | 0.1149 | 0.7578 |
| fixed checkpoint | 40 | 0.2365 | 0.3196 | 0.1392 | 0.7161 |
| best target-bag F1 | 30 | 0.2385 | 0.3265 | 0.1342 | 0.7286 |
| best distinct-target recall | 4 | 0.1593 | — | 0.0836 | 0.9582 |

The three registered dev rules were evaluated on each shortlisted checkpoint.
The final selection was made from full-video dev TrackEval using the
pre-registered ordering: higher dev HOTA, then DetA, AssA, distinct recall,
lower inactive false acceptance, and earlier epoch. It selected:

- checkpoint: epoch 20, phase T, optimizer step 1,320;
- rule: B;
- checkpoint SHA256:
  `7012706140cfa94278ce15bb8da3e2318eb3efcfb8d26d3b1dbc5206f1145538`;
- base digest after load:
  `cffc24672a623e9dc6fb43b55668f098da342485d0100e800402ad88105046a4`;
- selection artifact:
  `outputs/l88/dev/final_selection_attempt1/selection.json`;
- fixed validation was false in the selection provenance.

The selected rule is not being claimed as globally optimal. It is the result
of the frozen internal dev protocol and is the only rule used for the later
fixed semantic and internal validation reports.

## Provenance

The authoritative machine-readable selection files are:

- `outputs/l88/dev/shortlist_attempt1/shortlist.json`;
- `outputs/l88/dev/final_selection_attempt1/selection.json`;
- `outputs/l88/dev/final_selection_attempt1/checkpoint_selection.json`;
- `outputs/l88/dev/final_selection_attempt1/provenance.json`.

Their flags state `screening_gt_used=false`,
`official_test_labels_read=false`, and `ordinary_mot_ovmot_touched=false`.

