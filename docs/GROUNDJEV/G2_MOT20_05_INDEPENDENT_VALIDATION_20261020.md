# G2 step3000 independent MOT20-05 validation — 2026-10-20

## Scope and independence

This is a complete **1,657-frame registered validation half** of `MOT20-05`, frames `1659–3315`, using the physical image paths in `mot20_05_validation_half.json`. It is video-disjoint from the 583,251-row G2 Clean TRAIN manifest: Clean MOT20 contains `MOT20-01`, `MOT20-02` and `MOT20-03`, with zero exact path/frame or same-video overlap for `MOT20-05`. The manifest is a registered validation scope, not the full MOT20 TEST set, and the runtime did not read GT or future frames.

Both arms use the same `official_scoring` parser, `tracking_dense` decoder, Simple association, `empty_and_record` rejection policy, and TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`. G2 is the preserved `G2_CLEAN_MIDTERM_STEP003000` checkpoint (checkpoint SHA `058573235114396f59ac1de05acd3700f46f230a298e31a16b3198d42dcc590f`; merged bundle SHA `d0d719ed31972f54b9ec295e69c4728a2101c386f7f7438e57a816b6cb650d43`).

## Official metrics

Fractions are used for HOTA/DetA/AssA/IDF1; IDSW/FP/FN are counts.

| arm | HOTA | DetA | AssA | IDF1 | IDSW | FP | FN | rejected | prediction rows | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 0.332021 | 0.617396 | 0.178873 | 0.292386 | 11289 | 16056 | 100036 | 117/1657 (7.06%) | 235,232 | 0.054891 |
| G2 step3000 | **0.478449** | **0.797046** | **0.287713** | **0.420743** | **11117** | 20737 | **28100** | 66/1657 (3.98%) | 311,886 | not claimed |
| G2 − G0 | +0.146428 | +0.179650 | +0.108841 | +0.128356 | -172 | +4681 | -71936 | -51 frames | +76,654 | — |

G2 improves HOTA, DetA, AssA and IDF1, reduces IDSW by 172 and FN by 71,936, and increases FP by 4,681. This separates a large perception/recall gain from a smaller association gain; it does not prove that JEV identity decisions alone caused the improvement.

## Runtime and recovery boundary

G0 processed 1,657/1,657 frames in `30,186.977 s` at end-to-end `0.054891 FPS`. The first G2 attempt was stopped after a preserved 91-frame causal prefix because the one-frame native service took about 22–25 seconds. The final run restored those 91 rows with `run_video.py --resume-from`, processed all 1,657 frames and passed official evaluation. Its successful frame-pipeline rate was `0.043444 FPS`; no complete end-to-end FPS is claimed because the initial failed-attempt timing is not part of the resumed receipt. The final output has 1,657 frame-decision rows and 311,886 prediction rows.

The initial partial directory remains untouched. The final receipt is `/data3/user/LocateMOT-runs/GroundJEV/g2_step3000_full_eval_mot20_05/comparison.json`; its SHA-256 is `2be2233c2e66323b22ccdb1b9b386646c6b792d9de5e77554cc1143b80ba42a6` for the manifest, `f09a892e2206c59e1421aabbd76f24a811f1505dc12376bc90693078f2b9f97a` for the G2 metrics, and `ae06f718a3eaac82275d56ec44af2b76fea5af3fe3cce869b43182538c78d472` for the resumed runtime result.

## Claim boundary

This is the first video-disjoint, complete registered validation-half comparison for G2 step3000. It is not a full MOT20 TEST result, does not justify test-set tuning, and does not isolate the JEV identity module from the GroundAnything perception front end. G0/G2 short controls and the earlier MOT17-02 full video remain separately labeled; the latter is TRAIN-scope only.
