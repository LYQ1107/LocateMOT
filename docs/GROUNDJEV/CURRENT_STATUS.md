# Current status — 2026-10-14

This page reports current evidence; earlier running-state snapshots are retained
in Git history and research_log.md. The full GroundJEV research goal is active.
Independent worktree: /data2/user/LocateMOT_GroundJEV. Original LocateMOT data
and U2 results remain in /data2/user/LocateMOT.

## Completed complete-video evidence

All rows below evaluate all 1203 frames of legal DanceTrack validation video
0004 with unchanged official TrackEval. Fractions are displayed as percentages.
This is one video and one training seed, not a complete-dataset result.

| Original V1 native proposals | HOTA % | IDF1 % | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|
| Compact distilled set, 300 steps | 20.157 | 17.140 | 588 | 2129 | 951 |
| Same set architecture, GT only | 5.977 | 3.406 | 1322 | 5954 | 1796 |
| Matched distilled feedforward | 11.197 | 15.118 | 161 | 549 | 3556 |
| Simple reference | 8.530 | 3.956 | 746 | 9334 | 256 |
| Original pretrained Full V1 | 2.145 | 0.483 | 3792 | 9300 | 241 |

The three trained models preserve identical original native candidate boxes,
labels, scores and all 27 rejected frames. The Compact 20-frame resume is
explicitly retained. ROI replay and resumed runs have no detector-inclusive
end-to-end FPS claim. Compact and feedforward miss more GT than Simple;
lower IDSW alone does not establish better tracking.

The fresh uninterrupted native Full V3 run and its own exact-candidate Simple
control are also terminal and validated: HOTA 2.389% / 8.257%, IDSW 3665 / 764.
Fresh V3 detector responses differ from original V1; cross-group comparisons
are not identical-candidate controls. Full V3 measured end-to-end FPS 0.118560,
including detector RPC, frame reads, identity memory, writes, rendering and
encoding; preloaded server/model initialization is excluded. Hardware was
A800-SXM4-80GB under concurrent project and foreign workloads. This establishes
an actual slow pipeline measurement, not real-time performance.
Evidence: outputs/groundjev/compact/dancetrack0004_native300_v2_empty_roi_fixed/
aggregate_results.json and both full-video owner/official-metric receipts.

## Shared semantic Compact joint stage

The same persistent track core now has trained expression membership and
full-vocabulary category heads. The 1200-step balanced stage completed with
1,146,511 parameters, real nonzero updates and exact checkpoint reload on all
three tasks. Text inputs use the actual frozen original OneJev token embedding
means; this is not language-decoder reasoning or a replacement claim for Full.

Training retains all 149 native MOT groups and adds 48 fit-v2 RMOT teacher
frames plus 64 TAO train Base teacher frames. There are 256 object-group
examples. All 112 teacher frames remain declared, including five empty frames
without a supervised region example. Teacher GT proposals/history and ideal
update/reliability proxies are explicit; they are not native detector rollouts.
No Novel GT is supervised and no unannotated TAO region is made a negative.
All 1203 supplied LVIS categories remain available at inference.

| Registered short native-proposal scope | Before | After |
|---|---:|---:|
| MOT17 4-frame HOTA % | 66.911 | 71.556 |
| MOT17 IDSW | 1 | 3 |
| RMOT single 8-frame HOTA % | 8.932 | 0 |
| RMOT multiple 12-frame HOTA % | 0 | 0 |
| TAO Base 4-frame TETA % | 14.013 | 39.272 |
| TAO Base ClsA % | 0 | 58.286 |
| TAO Base AssocA % | 21.350 | 17.971 |

All paired candidates and native ROIs match exactly; all registered empty and
rejected frames and decoded video frames were validated. These are short
functional controls, not full RMOT/OVMOT or Novel-generalization results.
The stage is not accepted as a successful unified model: RMOT regresses and
OVMOT association declines despite the category gain. Evaluation-only diagnosis
finds all 6 single-query and all 8 multi-query covered candidate observations
suppressed by the learned membership head. Their probabilities are below .008;
the registered .5 threshold is unchanged. The training-fit diagnostic classifies
all96teacher candidate observations correctly (45positive/51negative); swapping
the training expression changes probabilities, but its target correctness was
not scored. Training fit does not establish native validation generalization.
These counts are candidate observations,
not unique GT coverage counts. No GT correction or threshold tuning was applied.
Evidence: outputs/groundjev/compact/joint1200_controls_v1/owner_state.json and
validation_and_diagnostic.json. Full 1203-frame joint-core retention evaluation also completed with all candidate
and ROI rows identical and1203decoded video frames. HOTA20.157->22.103%,
IDF1 17.140->21.413%, IDSW588->346, FP2129->1653, FN951->1485. The association
gain comes with more missed targets. All27original rejected frames remain in
the official denominator. This is a same-video retention result, not complete
RMOT/OVMOT success or a full-dataset claim. ReplayFPS22.747 excludes detector
and ROI extraction and must not be compared to nativeFull E2EFPS. Evidence:
outputs/groundjev/compact/joint1200_dancetrack0004_v1/owner_state.json.

## Current training and services

The original corrected OneJev500 trainer stopped without a terminal receipt.
It saved BF16 weights at 250 and last logged 275, but no FP32 master weights,
Adam state or RNG state. Its external termination cause is not established.
The old intermediate is preserved and is not declared a completed 500-step run.

A new finite replay of the original unfinished 500-step protocol starts from
its original valid 100-step initialization, with unchanged corpus, seed0,
lr5e-6, 32768 context, native fork tolerance and zero dropped rows. It runs on
physical GPU4; the latest returned live report was step50/500. It does not
pretend to reconstruct missing Adam state at old step250. Checkpoint handling
now atomically saves FP32 masters, Adam, scheduler, all RNG and exact planned
step cursor every125 steps, retaining one latest full training-state checkpoint.
A real-native-ROI functional resume regression gives identical next-step loss
and parameters; this small probe is not itself an actual OneJev training result.
Config: configs/groundjev/onejev_own_history_v3_exact500_replay_stateful.json.

Physical GPU5 runs only the owned native visual-feature RPC on port8111.
The original Ground decoder service on8101 is currently absent. GPU7
was released after the finite full-video retention owner completed. GroundJEV does not own the fufu
GPU6 job or UDMT GPU4/5 jobs and has not stopped them. At most four physical
GPUs are allowed. No old U2 queue or full-dataset feature extraction is running.
Temporary feature caches remain bounded to25GiB; datasets and training-state
checkpoints are separate preserved artifacts.

## Earlier verified foundation and remaining requirements

Old U2 owners/queues were stopped and 39.89GiB of regenerable features removed;
raw data, models and results were preserved. Both official original model
bundles and isolated DLM/OneJev environments are installed and verified.
OneJev SFT100 completed with actual changed weights and exact multimodal
reload, but its controlled tracking results regressed. The separate genuine
native-Ground own-history corpus completed 48 train frames and 5661 rows; the
current OneJev500 replay still uses its declared teacher-proposal corpus.

Native Ground projector adaptation completed three warmup steps and 48-frame
fresh dense/referring retention checks; formal tracking adaptation and full
original Ground benchmarks remain incomplete. The frozen visual temporal
adapter completed200 steps without proving a tracking gain. Native category
and earlier Full controls remain preserved, including failures and collapses.

Still required: complete formal OneJev500 and actual reload/before-after
tracking evaluation; adaptation using native detector-rollout training data;
formal Ground tracking adaptation and full dense/referring retention;
shared model RMOT generalization, Base/Novel OVMOT and complete legal scopes;
long-memory/reappearance/NO_MATCH and permutation ablations; controlled joint
negative-transfer studies; matched Full/Compact/MLP/SetTransformer accuracy,
end-to-end speed and resource comparisons; and continued GitHub publication.
Pilot and these short controls do not complete the research goal.
