# G2 Clean held-out scope handoff — 2026-10-22

The existing GPU5 and GPU6 owners are already configured to run the complete
independent held-out queue. Each owner evaluates the fixed MOT20-05 scope first,
then continues on the same GPU with:

1. RMOT single: refer_kitti_v2, sequence 0007, 800 validation frames.
2. RMOT multi: refer_kitti_v2, sequence 0007, 800 validation frames.
3. OVMOT: TAO amodal validation video 105, 40 annotation frames.

All three manifests are marked training_on_this_scope=false and use the fixed
official parser, dense grounding decoder, empty-and-record perception failure
policy, and the existing scorer. GPU5 and GPU6 run independent owners in
parallel; outputs and receipts are separated by owner. No duplicate process was
started while the existing MOT20 owners were still active.

Only a complete owner receipt containing inference output, scorer output,
frame-contract validation, and official metrics is admissible as a result.
