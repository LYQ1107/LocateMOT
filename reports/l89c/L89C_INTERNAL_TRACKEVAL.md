# L89C corrected internal TrackEval

This is a full-video internal validation-scope replay using exactly the
frozen epoch-2 / Rule-R selection from corrected developer selection. It is
not screening and does not read official-test labels. The inference summary
records 735 groups, 623 queries, all candidate rows retained, no persistent
dense cache, and predictions frozen before labels were attached.

| dataset | videos | sequences | HOTA % | DetA % | AssA % | DetRe % | DetPr % | IDSW |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Refer-KITTI-V1 | 0004, 0018 | 86 | 2.244718 | 0.990821 | 5.145246 | 1.135781 | 7.174238 | 54 |
| Refer-KITTI-V2 | 0016, 0017, 0020 | 537 | 0.583059 | 0.220838 | 1.573466 | 0.228132 | 6.437471 | 38 |

The combined TrackEval matrix is
`outputs/l89c/internal/trackeval_corrected_final_attempt1/trackeval_matrix.json`
(SHA256 `f95899b1ef7e3eadbee0e2dcfd275d8fcce6cd64f4f421dbf5dd0171588ffe2a`).
The local TrackEval checkout has no verifiable git HEAD; this provenance
limitation is recorded in the machine output.

For historical context, buggy L89 epoch-4/Rule-R internal HOTA was V1
`2.2906%` and V2 `0.5894%`; L89C corrected replay is therefore `-0.0459` and
`-0.0063` percentage points respectively. L87-A reference HOTA was V1
`28.5752%` and V2 `22.1300%`, so L89C remains far below that historical
reference. These comparisons do not convert internal validation into a
screening or official benchmark.

`hota_trackeval_run=true` applies only to this explicitly authorized internal
TrackEval. `screening_gt_used=false`, `official_test_labels_read=false`, and
`ordinary_mot_ovmot_touched=false`.
