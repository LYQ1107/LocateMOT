# External baseline real evaluation status

TempRMOT has a real legal-development prediction and TrackEval result, recorded in `outputs/unified/external_baselines/temprmot_legal_v1.json` and `reports/unified/EXTERNAL_BASELINE_LEGAL_DEV.md`. It covers two frozen V1 validation sequences and reports combined HOTA 94.679, MOTA 98.925, and IDF1 99.465.

The required two-method baseline condition is still incomplete. DKGTrack exposes only a Baidu share checkpoint (`checkpoint.pth`, fs_id `114294715583115`, 1,235,582,680 bytes); FlexHook exposes Baidu shares for `refer-kitti-best.pth` and tracker outputs. Password verification and directory listing succeeded for both shares, but the public download endpoint returned `errno 8001` (verification required), `errno 112` (share download rejected), or `errno 113` (signature rejected) without an authenticated Baidu session. No checkpoint bytes, predictions, or metrics were fabricated from those links.

DKGTrack and FlexHook remain `INCOMPLETE`; TempRMOT alone does not satisfy the two-independent-method requirement. The raw HTTP audit is outside the repository, while the reproducible state and blocking reason are recorded here.
