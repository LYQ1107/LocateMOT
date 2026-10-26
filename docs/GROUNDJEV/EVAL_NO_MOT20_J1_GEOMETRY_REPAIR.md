# J1 competition geometry: isolated inference repair

The runtime's native G2 candidates use xyxy. TrackJEV's competition module uses
xywh for IoU, center distance and entity groups. The legacy adapter passed xyxy
directly, enlarging boxes and reporting false overlap between distinct objects.
This affects both full-frame competition and subgroup recomputation after a
native context split.

Evidence uses actual J1 TRAIN row
`MOT17-04-FRCNN_000001_c7_identity_birth`: its 44 candidates produce 520 wrong
IoU-matrix entries under the legacy interpretation. Converting only the geometry
input to xywh reproduces all saved TRAIN matrix entries exactly (zero mismatch).
A regression checks that two physically disjoint runtime objects remain separate.

The corrected inference arm is `j1_competition_xyxy_v2`. It retains J1 step1200,
the same native G2 cached candidates, parser, option vocabulary, NO_MATCH rule,
shortlist, memory policy and thresholds. Native candidate/output coordinates stay
xyxy. Only the competition geometry transport changes.

The first complete SportsMOT control has finished (226 frames, identical native
candidates). Simple HOTA is 64.7157%, legacy J1 is 13.6083%, and repaired J1 is
13.5220%. Both J1 variants have NEW=1, ACTIVE=225 and IGNORE=2434, FN=2230 and
IDSW=0. The numerical geometry error is real, but this repair does not explain
or fix the one-identity collapse on this video. No positive tracking gain is
claimed. GPU sharing precludes a controlled speed comparison for this rerun.

This v2 control changes the identity competition paths only. The separate memory
question's compact-group geometry still needs the same conversion when several
identities are assigned. This first control assigns only one identity per frame,
whose 1x1 competition matrix is identical under both conventions. Also, J1 TRAIN
rows contain a current-candidate crop sheet, while this adapter currently sends
only the full current image. These evidence-contract issues remain open and must
be evaluated in separately named follow-up arms, without rewriting v2 results.

The original `j1` directories remain historical diagnostics. Reconstruct that
contract explicitly with `replay_j1.py --arm j1 --competition-geometry legacy_xywh`.
Current tables select the subsequent current-crop/memory v4 contract through
the committed config; v2 is retained as a separate historical diagnostic.
They never mix predictions from these variants. Existing source snapshots
and the published negative results remain available in Git history.

The repair does not establish that J1 will improve over Simple. Large NO_MATCH
probability mass, missing births, and the difference between the checkpoint's
training evidence layout and this adapter remain separately measurable limits.
No new training, forced births, prediction-coordinate repair, or VAL threshold
selection is part of this experiment.
