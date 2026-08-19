# Stage L10 — Failure Analysis

Date: 2026-08-18 (partial; v2 evals pending)

Sections:

- TAO high-IDSW sequences (full-PBD wins/losses vs PBD-zero);
- TAO novel-category failures;
- RMOT worst queries (Refer-Dance and Refer-KITTI-V2);
- same-appearance crossing cases;
- which failures decreased after supervision scaling.

## Primary failure: expanded OVMOT stream collapses association

L10 v1 (15k steps, expanded 322,843-candidate stream) achieves TAO val
full-PBD TETA 26.39 / AssocA 7.26, far below L9-adapted (33.79 / 29.34).
LocA 64.5 and ClsA 7.4 are unchanged, so the failure is **identity
association only**.

Diagnosis (verified in predictions):

- L10 v1 assigns a new track id to almost every candidate: in a sample
  video, 1,612 unique ids over 1,650 prediction rows (only 7 ids ever
  reused), vs L9-ovmot's 374 unique ids with 95 reused.
- Cause: the L9 OVMOT training target marks **every unmatched candidate**
  as a positive relevance target AND a NEW birth target.  The expanded
  stream is only 3.5% matched (C-TAO base), so ~96% of training
  candidates taught the model "every detection is a new object".
- Ordinary MOT and RMOT are not collapsed by this checkpoint (Macro AssA
  ~0.504, Refer-Dance HOTA ~36.3), consistent with a stream-specific
  supervision-bias failure rather than global model damage.

## Fix tested (v2) - did not rescue

Unmatched OVMOT candidates -> relevance 0 (hard negative) and NEW birth
only for detection score >= 0.4.  Re-training from L9 final (15k steps)
still yields TAO full-PBD TETA 26.24 / AssocA 7.86.  A NEW-margin sweep
at eval (0..2) on the same shard raises AssocA from 3.6 to 6.4 only.

Root-cause boundary: candidate-GT match rate is ~3.5% in the expanded
stream (C-TAO base).  Unmatched detections include both true novel
objects and noise, and no training signal links the same novel object
across frames, so the model cannot learn association for them; it
defaults to birthing new ids.  Fixing this requires either dense
continuous GT covering detector detections (C-TAO base_and_novel is
still insufficient: +51 tracks only) or a temporal pseudo-track
self-supervision for unmatched candidates.
