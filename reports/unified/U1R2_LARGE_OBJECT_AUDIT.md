# LocateMOT-U U1R2 coordinate and large-object audit

The audit used the frozen epoch-4 checkpoint
`/data2/user/reference_envs/u1_generic_adaptation_20261001/valid_work_bf16_batch2_frozen01_noacc/epoch_4.pth`
(SHA-256 `94fcc1d1b5e29ab24b8b926ae868736449832248a9415e9a3796f8b98ed3db80`)
and only legal calibration/validation frames. The official KITTI tracking
training label archive was used for same-frame geometry comparison; reserved
videos 0005, 0011, 0013, and 0019 were not read.

Refer labels are normalized top-left `xywh`. Source extents are retained for
formal metrics. Across the 33,526 legal physical GT boxes, 195 extend below the
image at `y1 > height`; none has `x0 < 0`, `y0 < 0`, or `x1 > width`. The 195
are all V1 (V1: 195/15,737; V2: 0/17,789). By size, the union is small 2/9,355,
medium 27/17,273, and large 166/6,898. No evaluation clipping was applied.

Same-frame maximum-IoU matching to official KITTI objects gives 33,522 matched
objects, mean IoU `0.7414883`, IoU >= 0.50 for 29,757, and IoU >= 0.90 for
4,922. Mean maximum coordinate delta is `14.7575` pixels. The truncation flag
counts are 30,029 at 0.0, 2,451 at 1.0, and 1,042 at 2.0. Refer and official
track-ID namespaces differ (only 787 equal IDs), so class recovery and audit
use geometry rather than ID equality.

The corrected epoch-4 formal run was low at Top150 IoU0.50 overall `0.0622800`
(V1 `0.0671666`, V2 `0.0579572`), establishing that the old apparent U1-A
values were under the repaired coordinate interpretation. The class-aware
physical-frame FP32 refinement then restored the proposal gate without
altering the source extent. The audit and the refinement are retained rather
than hiding the coordinate discrepancy.

Evidence: `outputs/unified/u1r2/generic_localization_failure_audit.json` and
`outputs/unified/u1r2/formal_generic_epoch4_corrected/result.json`.
