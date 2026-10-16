## Isolated global identity increment — 2026-10-16

This branch adds CPU research at `/data2/user/LocateMOT_GlobalIdentity`; original
native training remains in its original worktree, without changed code or workers.
R1 whole-frame audit complete. A1/A2 completed4056/4064updates; A3 completed3602
memory-head updates with unchanged A2 physical tensors. Full-video A2 HOTA%:
DanceTrack24.752, RMOT single0, multi5.685. A0 references26.316/6.025/19.802.
These regressions are retained. All five A1/A2 scopes, four A3 memory policies and
real own-state8/16/32-frame V2 training continue under blocking CPU owners. No new
GPU process or feature bank. Detailed evidence and pending limitations:
[global identity report](GLOBAL_IDENTITY_DECISION_RESEARCH_REPORT.md).
The original status below records the source branch state at isolation, not a
fresh observation of those native jobs. Both research goals remain incomplete.

## Matched native RMOT physical-supervision foundation training started — 2026-10-16

Both complete three-task24000question courses passed their real native multimodal
CPU plans with zero drops and6000global4updates each. All8000MOT/8000OV rows
remain exact; both arms retain the same5118oldRMOTrows and append2882paired
actual-history questions in the same positions. Their native rendered state/media
prefixes match at every actual global index; only the registered physical versus
membership questions/targets differ. Native within-update length ordering may
differ, so this is not token/FLOP matched. Maximum27480tokens in both courses;
planned tokens202072473physical/201659237semantic. Rebuild code, manifests and
complete CPU scalar receipts are retained. Evidence:
`training_data/paired_rmot_physical_semantic24000_inputs_v1/completion_validation.json`.

The new blocking owner starts physical then semantic native750M SFT, each an
additional6000updates/24000questions from the identical verified native6000FP32
masters, new Adam/scheduler/RNG and seed0. First physical optimization has begun;
initialization maxdifference0 across752393024parameters, vision frozen. Actual
updates1..5 consumed20questions (9MOT/6RMOT/5OVMOT), including2newRMOTidentity
rows; losses/gradients are finite and gradients positive. First5peakTorchallocated
17.87594GiB, not full-run peak or controlled speed. Semantic arm0at observation.
This is neither Ground+One joint gradient nor long-video BPTT, and has no new
tracking accuracy result yet. Evidence:
`training/paired_rmot_physical_semantic6000_owner_v1/first5_physical_optimizer_snapshot.json`.
Registration:
`configs/groundjev/paired_rmot_physical_semantic6000_v1.json`.

The previous GPU4native owner actually exited0 after its final TAO unit. The new
stage uses only GPU4 alongside the unchanged four-GPU rank; launch free37341MiB
and Torch allocator cap.33/28081638604bytes. GPUs0-3 and all older workers/course
definitions remain untouched; there is no exclusive GPU claim. Both final models
must use the same assigned-ID memory-update protocol across all4original scopes.
After both trainings/reloads, the owner waits on the original native6000assigned
baseline dependency without retaining a GPU model; no native600baseline substitute
or per-task checkpoint selection. Numbered exports/full state and scientific
tracking predictions are preserved separately from temporary features.

New course2MiB and owner/report4MiB reservations raise the conservative combined
feature/metadata upper to26840605662bytes<26843545600(25GiB). No new image/ROI/
activation bank. Current four-GPU459027 and paired-short/long jobs continue their
registered courses; the ordinary-model-level three-task goal is still unmet.

## Paired RMOT supervision inputs ready; full readout study closed — 2026-10-16

The complete new actual-history paired component has2882question pairs from the
already certified18RMOT TRAINrollouts:1428existing-identity and1454assigned-write
questions(981no/473yes) versus the corresponding original membership questions
(2237yes/645no). Both arms share original physical states/media and paired row
order/IDs; semantic source questionIDs are separately recorded. Assigned-state
controls use the original actual selected-identity context, not GT-repaired tracks.

Native augmentation chooses JSON formatting after option shuffling, so different
question option counts would produce different state formatting in698pairs despite
the same seed. This derivative component explicitly fixes both state blocks to
source-equivalent prettyJSONstrings; decoded original input trees/media/questions/
targets remain exact and all2882native-rendered state prefixes match. Native
per-question option permutation remains. This is a declared new component format,
not an unannounced edit of the original courses. Source pixel validation is reused.

Both complete2882-row native multimodal CPU plans returned0with no dropped row:
physical maximum18096tokens/planned15639410tokens; semantic17227/15226174.
No model forward, GPU, optimizer, new ROI/image bank or current course modification.
The questions differ, so this is not token/FLOP matched. Files total about1.67MB
under the new4MiB metadata reservation. Evidence:
`training_data/paired_rmot_physical_semantic_component_v1/{registration,input_validation,
physical_CPU_validation,semantic_CPU_validation,completion_validation,coverage_and_budget_validation}.json`.
Rebuild code/config are published; full derivative group/row metadata stay server-side.

The registered459027course has97112RMOTmembership and3701presence questions;
it has not silently acquired this new2882physical component. This is curriculum
inventory, not actual consumed counts or a claim that no earlier initialization
ever received physical RMOT supervision. The prepared component enables a later
matched learning test; no new training recipe/optimizer is launched by it.

The frozen shared-body task-readout study now completed all5scopes/all15units,
original owner51823exit0. Final official results:

|Registered scope/metric|Original|Canonical/untrained|Task-conditioned readout|
|---|---:|---:|---:|
|DT1203 HOTA %|26.3157|26.3157|26.3157|
|RMOT single800 HOTA %|6.0247|2.0050|2.4619|
|RMOT multi800 HOTA %|19.8019|11.7532|11.9169|
|TAO40 TETA|14.517|26.125|30.818|
|MOT20full1657 HOTA %|7.4316|7.4316|7.4316|

TAO newreadout LocA33.075/AssocA35.637 equal canonical; ClsA9.6637→23.743,
still below original30.314. BaseTETA31.9085, NovelTETA28.637but NovelClsA0.
This one-video40annotation-frame/three-GT-class scope is not full TAO or evidence
of novel-class recognition. RMOT regressions prevent common policy promotion;
no metric-component stitching or per-task checkpoint selection. Evidence:
`compact/canonical_task_readout_v1/all_five_scopes_completed.json`.

A separate complete trajectory check validates all4500frames/287744candidates/
331063events: every physical candidate decision, lifecycle, memory write and ID
matches canonical, as does final full physical memory including pixels/features/
timestamps. Only final label/semantic-count fields are excluded from memory equality.
All35228TAOcategory decisions and3620RMOTmembership probabilities differ;
emitted outputs change in501frames. Thus restoring semantic context does not
repair the canonical physical coverage loss in RMOT. No new forward/scorer/GT calls.
Evidence: `validation/task_readout_full_rollouts_v1/validation.json`.

The older native750M history-refresh comparison has all9 complete first-three-scope
units frozen, with exact raw/official prediction reconstruction and native candidate,
frame, video and causal-input checks. Each old/refreshed arm adds600updates/
2400questions from the same native600FP32 masters;400history questions differ,
so the comparison is not token/FLOP matched. It is separate from native6000.

|HOTA %|Before600|Old history +600|Refreshed history +600|
|---|---:|---:|---:|
|DT1203|8.7030|13.1075|16.4112|
|RMOT single800|2.5062|3.0717|2.8779|
|RMOT multi800|6.7400|6.6825|7.5062|

Refreshed history improves DT/multi relative to old history but reduces single-RMOT
HOTA and raises DT identity switches2351→2611. No shared-model promotion.
Evidence: `training/full_history_refresh600_pair_first3_v1/validation.json`.

Original old-history worker14226 has now returned exit0 and completed its TAO40
unit: TETA27.748before→25.161old, LocA21.573→23.965, AssocA2.8136→2.9972,
ClsA58.856→48.522; NovelClsA remains0. All40frames/35228candidates and5112
official predictions validate, including exact raw/official tracker-input transport
and unmapped-category retention. Original scorer result is preserved without new
inference or rescoring. Refreshed TAO remains running, so its comparison is pending.
The first3 and old-TAO snapshots jointly use875796bytes of the existing1MiB
history-report reservation. Evidence: `training/full_history_refresh600_old_tao_v1/validation.json`.

Original fourGPU/source and paired-short training continue unchanged; exact observed
log rows2750/44000questions and1950/7800questions are frozen separately. Full-state
numeric gates remain2000/1000, not these observed rows; longbranch had not started.
Evidence: `training/observed_full2750_short1950_v1/snapshot.json`.
With the4MiB paired-component and1MiB native-history report reservations, current
conservative feature/metadata upper is26,834,314,206bytes<26,843,545,600(25GiB).
No new GPU/model-state copy/native feature bank. Overall goal remains incomplete.

## Both input/constraint studies complete; task-conditioned readout running — 2026-10-16

Original owners90021(grouping) and48863(NO_MATCH/lifecycle factorial) have both
returned exit0. All5scopes/all10and20official units are frozen; no pending arm
remains in either study. This supersedes their pending descriptions below.

|Registered scope/metric|Original|Overlap-neighbor16|
|---|---:|---:|
|DT1203 HOTA %|26.3157|28.6921|
|RMOT single800 HOTA %|6.0247|6.0247|
|RMOT multi800 HOTA %|19.8019|19.8019|
|TAO40 TETA|14.517|13.997|
|MOT20full1657 HOTA %|7.4316|7.0507|

MOT20grouping IDF16.1474→5.6306%, IDSW2060→1992, FP463→341,
FN286273→287351. Better DT and TRAIN competitor co-exposure do not establish
a common-model improvement. Preserve the default contiguous grouping.

|Registered scope/metric|Original|Timeout only|Feasibility only|Both|
|---|---:|---:|---:|---:|
|DT1203 HOTA %|26.3157|26.3157|12.9762|12.9762|
|RMOT single800 HOTA %|6.0247|6.4816|6.0247|6.4816|
|RMOT multi800 HOTA %|19.8019|19.5578|20.6629|20.2362|
|TAO40 TETA|14.517|14.493|14.517|14.493|
|MOT20full1657 HOTA %|7.4316|7.6054|9.8980|9.7962|

Final MOT20both arm: IDF110.0231%, IDSW3447, FP513, FN265498. Increased recall
comes with more identity switches and the DT collapse; neither constraint policy
is adopted across tasks. No per-task best-policy selection. Evidence:
`compact/{overlap_neighbor_grouping_v1,no_match_feasibility_factorial_v1}/all_five_scopes_completed.json`.

The next intervention separates physical and semantic contexts using the SAME
frozen set-core weights. Completed untrained canonical controls preserve MOT,
raise TAO LocA/AssocA to33.075/35.637, but lower ClsA to9.6637and RMOTHOTA
to2.0050/11.7532(single/multi). Their semantic heads consumed canonical physical
vectors. New opt-in `task_conditioned_semantic_readout` makes a second pass of
the shared body with actual task/query on the same current/history features;
physical association/update/reliability/lifecycle remain canonical. There is one
identity memory, no new parameter tensor, GT input, model family or optimizer.

Real61native TRAINcontexts plus6Baseobservations/402declared query interventions
validate exact physical logits against canonical, exact category/membership logits
against original query-coupled model on identical inputs, and all-head MOT parity.
143238output elements checked,61groups have history;335interventions actually
change canonical semantics. All66original tensors/1146511elements are bit-exact,
only the previously existing canonical physical-spec buffer and a runtime flag
distinguish model artifacts. These are conditional tensor proofs, not new scores or
invariance across different frontend candidate sets. Evidence:
`validation/canonical_task_readout_v1/{validation,parameter_identity_validation}.json`.

CPU owner/session51823has launched all5scopes with original/canonical-untrained/
newreadout arms:10explicit completed references and5new actual trajectories. Same
frozen checkpoint parameters, original16grouping/shortlist32/thresholds/clocks/
frames/candidates/scorers. Do not construct a synthetic TETA from old components
or claim training gain. Registration: `configs/groundjev/canonical_task_readout_v1.json`
and `compact/canonical_task_readout_v1/{registration,launch_snapshot}.json`.
Native fourGPU/paired training untouched; no new GPU or image/ROI bank.

First3scopes/all9official units have now completed and are frozen:

|HOTA %|Original coupled|Canonical physical/late semantic head|Task-conditioned shared-body readout|
|---|---:|---:|---:|
|DT1203|26.3157|26.3157|26.3157|
|RMOT single800|6.0247|2.0050|2.4619|
|RMOT multi800|19.8019|11.7532|11.9169|

Single newreadout FP47/FN17 versus canonical80/17 and original323/13; multi
FP160/FN1704 versus canonical196/1703 and original619/1054. Restoring original
semantic functions on canonical histories only slightly alleviates the RMOT loss;
it does not fix physical coverage/association. No promotion, and no conclusion
about pending TAO/MOT20. Evidence:
`compact/canonical_task_readout_v1/first_three_scopes_completed.json`.

The full459027row input-reuse audit also completed. Native per-row augmentation,
all64sources/family filters, seed0shuffle and exact native global16/fourrank
placement give97556distinct serialized-state/media contexts and24560media-recipe
lists. Only150same-global-update repeated contexts (23same-rank) occur. Even
overestimating each saved prefix by the entire shortest row gives2945176tokens/
0.05222%of planned tokens globally,390967tokens/0.006933%within ranks. These are
bounds for recognized identical metadata within an update, not runtime speed or
all possible equivalent pixels. Whole-media-list co-occurrences do not measure
vision compute. Across-update language activations cannot assume constant weights.

Upstream shared-prefix training packs text IDs/masks without native multimodal
tensors and is singleGPU; Qwen3.5 GDN needs branch-state isolation. Do not toggle
this flag in current jobs. A future grouped curriculum/branch engine is a separate
recipe, not a drop-in acceleration or completed optimization. Evidence:
`diagnostics/full_source_context_reuse_v1/{registration,summary}.json`. Only small
scalar reports use existing reservations; feature upper remains26,829,071,326bytes
<25GiB. The ordinary-model-level three-task goal remains incomplete.

## Short native1000 validated; TAO grouping and Ground retention remain mixed — 2026-10-16

The paired-short branch has an actual1000update/4000question full-state numeric
gate, not just a log observation: all752393024FP32language parameters and
1504786048Adam-moment elements finite;320optimizer counters, scheduler/epoch/
cursor exact, Torch/CUDA/Python/NumPy RNG fields saved. ExactFP32initialization
from native6000 is retained. The9,029,140,889byte atomiclatest state was inspected
and the numberedBF16step1000 retained; later full saves replace latest. No actual
milestone restart/newnative inference or completion of the6000short/long study.

Actual seed0index-shuffle exposure:1288MOT/1327RMOT/1385OVMOTquestions.
MOTincludes673teacher rows without an older `source_row_id`; their original rowIDs
are explicitly counted separately from615consumed old-sourceIDs, not invented or
folded into the old-source count. First-media paths1017MOT/71RMOT/1180OVMOT
are current/source-media counts, not all historical frames or independent tracks;
teacher images draw from the registered wider legal TRAIN temporal course. Vision
frozen; longbranch still not started at this observation. Last full-source fourGPU
log observed2350/37600questions, while its complete numeric gate remains2000.
Evidence: `training/paired_temporal_short_actual1000_v1/validation.json`.

Overlap-neighbor grouping first4scopes/all8official units are now frozen. TAO40
TETA14.517→13.997, LocA8.088→6.6283, AssocA5.1478→10.2, ClsA30.314→25.162;
NovelClsA0inboth. Thus higher association does not imply higher total quality.
The unchanged shared weights/grouping intervention affects every learned head and
later history; no mix-and-match of original category scores with new association.
MOT20newgrouping remains with original90021owner. Evidence:
`compact/overlap_neighbor_grouping_v1/first_four_scopes_completed.json`.

Separately, the NO_MATCH feasibility owner has completed its full1657frame/
235232candidate MOT20feasibility-only arm: HOTA7.4316→9.8980%,
IDF16.1474→10.1300%, IDSW2060→3292, FP463→564, FN286273→267117.
This is a completed one-arm result, not completion of the4arm factorial; the
feasibility+timeout arm is still running under original48863. Its DT regression
26.32→12.98prevents common policy adoption. Evidence:
`compact/no_match_feasibility_factorial_v1/first_MOT20_feasible_unit_completed.json`.

Ground200native retention now has6completed whole tasks/31986rows against the
completed original9/91124rows, all effective counts equal original, provider
filtering0. F1mIoU % on the six common tasks:

|Native retention task|Rows|Original|Adapted200|
|---|---:|---:|---:|
|COCO|4952|47.9781|47.5713|
|Dense200|200|66.6435|66.0540|
|VisDrone|1610|29.6624|27.8704|
|RefCOCOg val|4889|91.7038|89.3700|
|RefCOCOg test|9577|91.1193|88.5215|
|RefCOCOplus|10758|84.3763|81.6416|

All six are lower. Preserve each completed task's exact official score/sample
fields and the earlier partial snapshot; do not promote Ground200. LVIS, RefCOCO
and RefCOCOg remain missing for adapted200; no full9aggregate or missing score
inferred. Originalqueue32 and adapted-sync execution differ after recorded CUDA
recovery, so this is not bitwise paired inference or exclusive causal attribution
to weights. These grounding retention metrics are not tracking HOTA/ordinary
three-task parity. Original clients/services and reattached owner remain unchanged.
Evidence: `ground_retention/partial_completed_tasks_20261016_v2/summary.json`.

All new scalar proofs reuse existing report reservations; temporary feature upper
remains26,829,071,326bytes<25GiB. No new ROI/image bank, GPU, model-state copy or
training update from validators. Overall research goal remains incomplete.

## Overlap-neighbor candidate context study launched — 2026-10-16

The preceding TRAIN audit motivates changing which candidates share a context,
rather than dropping or merging proposals. New opt-in `overlap_neighbors` keeps
capacity16. Seed the lowest remaining original index; repeatedly add the remaining
candidate with greatest IoU to any current group member, ties by original index.
Zero-overlap ties also retain original order. No IoU threshold, NMS, topK, class
filter or GT decision rule. One-vs-N geometry is streamed, not an N-squared feature
archive. Every original candidate remains exactly once; model decisions are mapped
back to their original row indices before the unchanged global matcher.

Re-auditing all43TRAIN units preserves every original count/GT binding/IGNORE
target. Only-outside-group winners fall4804→298; GT-supported IGNORE with an
inside-group winner rises1024→5530. Positive candidates with outside-group rivals
fall5319→320. These are evidence co-exposure counts, not new labels or tracking
quality. Full-frame imagery was always visible and is unchanged. Original frozen
audits remain retained. Evidence:
`diagnostics/candidate_group_competition_overlap_neighbors_v1/summary.json`.

Real3frameprefixes on all5registeredscopes/3339candidates pass exact default
outputs and entire states. The regrouped causal prefixes check native input feature
ownership and association/update/query/category output alignment to original rows
against direct regrouped neural outputs. No GPU or GT input. Evidence:
`validation/overlap_neighbor_grouping_v1/validation.json`.

Only after both gates, CPU owner/session90021 launched the registered all5scope
before/regrouped study:5explicit completed original references plus5new rollouts.
The same original shared Compact checkpoint, 16capacity, geometryshortlist32,
learned heads, original lifecycle/NO_MATCH policies, clocks/frames/candidates and
official scorers remain fixed. Input grouping can change every learned output,
geometric shortlists, first-group lifecycle selection and later causal memory;
this is not an isolated association-only or probability-conserving intervention.
No new optimizer or native foundation training is claimed. One policy across all
tasks; final full-scope scores decide whether to retain it. Registration:
`configs/groundjev/overlap_neighbor_grouping_v1.json` and
`compact/overlap_neighbor_grouping_v1/registration.json`.

First completeDT1203official control is now frozen: HOTA26.3157→28.6921%,
DetA45.1842→46.2724%, AssA15.3476→17.8072%, IDF130.6329→33.2758%,
IDSW165→159, FP579→736, FN1962→1794. This is a modest positive on one
registered scope with increased FP, not all-task or full-dataset parity. Remaining
RMOT/TAO/MOT20 controls stay registered; no task-specific model/policy promotion.
Evidence: `compact/overlap_neighbor_grouping_v1/first_DT_scope_completed.json`.

Both registered RMOT800 scopes have also completed: single HOTA6.0247% and
multi19.8019%, with every reported official metric exactly equal to the original
reference. This does not assert every probability/state is identical. First3scopes/
all6official units are frozen in `compact/overlap_neighbor_grouping_v1/first_three_scopes_completed.json`.
TAO and MOT20 remain pending with the original owner.

At most1MiB streamed grouping geometry plus256KiBreports uses the remaining
existing2MiB audit/control reservation. Feature upper stays26,829,071,326bytes
<25GiB, no new ROI/image bank or GPU. Native Full/paired training and original
MOT20NO_MATCH owner48863 continue unchanged. Overall goal remains incomplete.

## TRAIN candidate-group competition audit completed — 2026-10-16

The new read-only audit checks all43MOT TRAIN units/1056frames/48051candidates
against actual saved native request groups, original candidate registry and original
identity-row targets. Every original GT binding and IGNORE/non-IGNORE target matches
the reproduced whole-frame Hungarian-before-threshold rule. No source row changes.

Of7992IGNORE candidates,5832have GT IoU≥0.5 and2160do not. Among the5832,
4804(82.37%)have assigned winners only outside the actual explicit candidate group;
4723of those support exactly one GT. There are1024with an inside-group winner,
4829with an outside-group winner (categories overlap),4with no assigned winner,
and261with multiple supported GT. Positive candidates with any supported rival:
6754;5319have a rival outside their group. Actual3401nonempty request groups have
at most16candidates, verified against the saved group geometry and local indices.

This quantifies a mismatch between whole-frame one-to-one exclusion labels and
group-local explicit candidate evidence. Full image pixels remain visible, so it
does not prove labels are unobservable or explain all failure. GT-supported overlap
also does not certify every proposal as a valid unique output. No validation data
or threshold tuning was used. A future admission/duplicate objective should expose
its competition evidence explicitly; do not silently reinterpret IGNORE as background
or retarget the current full-source/paired courses. No additional experiment is
registered from this audit alone. Evidence:
`diagnostics/candidate_group_competition_v1/summary.json` (63350bytes), generated
by `tools/groundjev/audit_candidate_group_competition.py`. Scalar metadata draws
≤128KiB from the existing2MiB reservation; global feature upper remains unchanged.

NO_MATCH factorial first4scopes/all16official units are now frozen. TAO feasibility
TETA14.517 equals original; both14.493 equals timeout-only; allNovelClsA0. Final
MOT20 arms remain with originalowner48863. Evidence:
`compact/no_match_feasibility_factorial_v1/first_four_scopes_completed.json`.

## Complete appearance controls; NO_MATCH feasibility regresses DT — 2026-10-16

This entry supersedes the pending appearance/timeout completion descriptions below.
The original appearance-rank owner completed all20official units, temporal
appearance1000 all20, and timeout-only all10. All-five-scope receipts recheck each
completed unit against official score/runtime, frame/candidate/rejection counts
and final identity state. No frames or arms removed.

|Five registered scopes|DT HOTA %|RMOT single HOTA %|RMOT multi HOTA %|TAO TETA|MOT20 HOTA %|
|---|---:|---:|---:|---:|---:|
|Original shared Compact|26.3157|6.0247|19.8019|14.517|7.4316|
|Timeout-only deletion|26.3157|6.4816|19.5578|14.493|7.6054|
|Raw-appearance rank|30.2544|5.7087|18.9930|14.104|8.0009|
|Random-projection rank|19.4111|5.8205|19.6370|14.324|7.9101|
|Trained-projection rank|24.7555|6.1548|20.1424|14.967|8.0870|
|Fixed fusion, raw appearance|13.3108|5.5377|16.6224|13.964|31.3078|
|Fixed fusion, random projection|13.4881|6.6298|14.3821|14.194|31.5077|
|Fixed fusion, trained projection|8.4607|9.2325|31.5524|14.395|31.7503|

The trained projector has341184parameters/1000updates, not foundation SFT.
Fixed fusion also changes admission/auxiliary execution relative to the original
learned actor. Its MOT20 jump7.43→31.75 cannot be attributed to learned appearance:
raw/random fixed-fusion controls already reach31.31/31.51. Trained fixed-fusion
MOT20 has8689IDSW/16049FP versus2060/463original; trained rank has3317IDSW
despite HOTA8.087. AllTAO NovelClsA remain0. Mixed outcomes and DT regressions
prevent common-model promotion or an ordinary-model parity claim.

The next intervention is an explicit active-live-pool execution contract. The
exporter labels NO_MATCH when a correct live identity is omitted from the shortlist;
the old native prompt says previously observed without explicitly qualifying live
versus retired. The all14unit audit finds10406DT/112129MOT20Compact NO_MATCH
selections with every live identity offered. Unknown birth bindings remain unknown,
and GT-unmatched candidates are not universally negative. Opt-in `only_if_omitted`
leaves raw probabilities/confidence/entropy/auxiliary/semantics untouched on
identical inputs, excluding only a private NO_MATCH cost when no live ID is omitted.
No probability renormalization, GT decision rule or proposal filtering is introduced.

All339596original candidate assignments across14units reproduce exactly under the
default extracted matcher; raw probability immutability and opt-in feasibility
pass on those same real inputs. Actual CPU rollouts have frozen first3scopes/all12
official units; these supersede the earlier shadow-assignment-only observation:

|HOTA %|Original|Timeout only|Feasibility only|Both|
|---|---:|---:|---:|---:|
|DT1203|26.3157|26.3157|12.9762|12.9762|
|RMOT single800|6.0247|6.4816|6.0247|6.4816|
|RMOT multi800|19.8019|19.5578|20.6629|20.2362|

DT feasibility FP579→3509/FN1962→1550/IDSW165→459 shows additional admission
does not repair identity association. Preserve the failure and all4arms; no adoption
based on small multiRMOT gains. TAO/MOT20 new-arm rollouts remain with original
owner/session48863: all5scopes, checkpoint, timestamps, candidates and scorers
unchanged. Two reference arms explicitly reuse completed original/timeout controls.

Evidence: `compact/{appearance_rank_study_v1/owner,native_temporal_appearance1000_owner_v1,
timeout_only_lifecycle_controls_v1}/all_five_scopes_completed.json`,
`diagnostics/no_match_pool_exposure_v1/summary.json`,
`validation/no_match_feasibility_v1/validation.json`,
`compact/no_match_feasibility_factorial_v1/{registration,first_three_scopes_completed}.json`.
New2MiB scalar/working reservation raises conservative temporary feature upper
to26,829,071,326bytes <26,843,545,600. No new ROI/image bank, GPU or optimizer.

Independent genuine752393024-parameter language adaptation has frozen observed
log rows2250/36000questions (full source) and875/3500questions (paired short).
These are progress, not completed epochs or another full-state numerical gate;
last complete full-state gate remains2000. Paired-long was not started at that
observation. Vision frozen; no Ground/One joint gradients or long-video BPTT.
Evidence: `training/observed_full2250_short875_v1/snapshot.json`. Training unchanged.

## Shared native600 baseline now complete on all four registered scopes — 2026-10-16

The original TAO40baseline has completed under the old registered GPU5owner;
its explicit identical-baseline consumer has validated the full40frames/35228
native candidates. All four original before scopes now use the same native600
checkpoint: DT1203HOTA8.7030%, RMOTsingle8002.5062%, multi8006.7400%, and
rawTAO40TETA27.748/LocA21.573/AssocA2.8136/ClsA58.856. TAOBaseTETA37.437,
NovelTETA8.3686/NovelClsA0; its GT classes are knife/baby(Base),peeler(Novel).
These are one-video registered controls, not full-dataset ordinary-model parity.
TAO association remains weak despite higher Base classification than Compact;
no new native6000after-training result or model promotion is inferred.

Native6000defaultTAO has actually advanced from baseline dependency waiting to
`WAITING_REGISTERED_GPU7_TEMPORAL_TRAINING_WINDOW`. Its future GPU7launch waits
the current paired study's window; original four-GPU training and pairedshort
training continue (latest observed short425steps). OriginallegacyGPU4oldcontrol
is still in its TAO run; GPU5refreshedcontrol has moved on to its DT1203run.
All original handles remain live, no worker restarted. No latency estimate is
inferred from the cached consumer. Frozen completebaseline receipt:
`training/broader_shared6000_fla_owner_v2/native600_before_all_four_scopes_completed.json`.

Clock clarification: MOT/RMOT use physical seconds; TAO explicitly uses annotation
steps because source FPS is unavailable. Its legacy `lost_seconds=3` field therefore
means3annotationsteps. Earlier generic3s descriptions of the lifecycle control
were inaccurate for TAO. No timestamp/timeout value, prediction, model input or
metric changed. Clock-corrected provenancev2 preserves all14units' counts/causes/
numeric gaps exactly against retainedv1; only unit metadata and the misleading
`seconds_since_last_assignment` field name change. Evidence:
`diagnostics/lifecycle_clock_clarification_v1/{clarification,validation}.json` and
`diagnostics/identity_survival_provenance_v2_clock/summary.json`.

## Lifecycle provenance measured; explicit deletion controls running — 2026-10-16

Two complete posthoc audits now constrain the next intervention. All five
original learned Compact scopes/287744candidates show probability dilution
(sum existing-ID mass beats an action but every ID individually loses) in
65DT/0singleRMOT/0multiRMOT/0TAO/3040MOT20positive-shortlist events. This does
not explain the missing RMOT appearance gain; the proposed conditional-mass
policy was NOT implemented or launched. The audit is an oracle potential
diagnostic, not tracking improvement or calibration evidence.

The second audit reconstructs immutable birth bindings, survival and actual
retirement causes on those five scopes plus nine completed native controls.
Every reconstructed surviving correct ID matches the original frozen external
diagnostic; all frames/candidates remain included. CompactMOT20's219060matched
observations split into first646/neverGTboundbirth147769/allboundbirthsretired
45599/alivebutshortlistmiss5851/correctshortlist19195. Unknown births remain
unknown: these counts do not prove no physical identity existed or replace HOTA.

Native DT600→6000 has0→138GT-bound premature learned retirements,131withGT
still present; native multiRMOT0→55,41withGTpresent. Compact's known early
retirements are DT0/single20/multi7/TAO0/MOT2014. These are provenance counts,
not proof deletion alone caused accuracy changes. The source exporter labels
unknown GT-at-birth bindings TERMINATE, a proxy that does not establish permanent
exit. Literal lifecycle target inventory: full-sourceMOT4682/93624=5.00085%
TERMINATE; completedbalanced24k250/1000=25%; bothpairedcourses130/500=26%.
The running full-source natural course has not inherited balanced oversampling;
the controlled paired courses remain unchanged. Inventory is not already-trained
exposure or independent per-row GT certification. No current row/target/job edited.

Opt-in `timeout_only` changes only physical deletion execution: learned early
TERMINATE leaves the identity available until the originalTTL3in registered units
(physical seconds for MOT/RMOT, annotationsteps for TAO). All model
distributions/matching/auxiliary/semantic outputs are retained; future states
may differ. Default `learned_or_timeout` and all registered original protocols
remain unchanged. Real30frameCompact and19frame native prefixes reproduce default
state/outputs exactly, retain precisely the early-deleted IDs under intervention,
and preserve all common identity states. Original3s expiry also passes a labelled
functional empty-observation counterfactual, which is not a scored video frame.

The actual Compact all5scope control owner82298has completed first4/all8units:

|HOTA %|Original|Timeout-only|
|---|---:|---:|
|DT1203|26.3157|26.3157|
|RMOT single800|6.0247|6.4816|
|RMOT multi800|19.8019|19.5578|

RawTAO40TETA14.517→14.493,NovelClsA0inboth; no general recognition gain.

MultiIDF1%=21.0082→21.0638,IDSW165→157,FP619→597,FN1054→1061.
Mixed evidence, no policy promotion; finalMOT20 remains running. A genuine native
same6000checkpoint counterpart is queued (owner4149961/start52754749,session90700)
behind original6000all-controls completion and the current GPU7training window.
It runs all4originalscopes, keepsTTL3in registeredunits/defaultmatching/thresholds/scorers, takes a
serializedGPU7window, and reuses TAO native image-only category outputs only from
the same complete checkpoint/current-image/candidate/vocabulary source. All
identity/lifecycle inference reruns; no fresh E2E/speed claim. It has0newcalls now.
Its v1CPUwaiter had a statically discovered category summary field-name error;
exactCPU-only handoff/exit143/code snapshot are preserved before any inference.
Only this new waiter was replaced; original training/inference jobs were untouched.

Evidence: `diagnostics/{existing_probability_dilution_v1,identity_survival_provenance_v2_clock,
lifecycle_proxy_inventory_v1}/summary.json`, `validation/timeout_only_lifecycle_v1/validation.json`,
`compact/timeout_only_lifecycle_controls_v1/first_four_scopes_completed.json`,
`training/native6000_timeout_only_controls_v2/queued_snapshot.json`.
Shared3MiBaudit/control reservation raises feature upper to26,826,974,174bytes
<26,843,545,600; no new ROI/image bank or GPU. Original Full four-GPU and paired
short/long training continue. The one-shot CPU observer49603has now completed
and verified the actual2000save/32000questions:17590MOT/7107RMOT/7303OVMOT,
all752393024FP32parameters and1504786048Adam-moment elements finite, exact
optimizer/scheduler/cursor2000 and allfourrankTorch/CUDA/Python/NumPyRNG states.
The9,029,193,825byte latest full state and numberedBF16ckpt-step2000exist;
later full-state saves replace the atomiclatest, numberedBF16exports persist.
This is actual checkpoint inspection, not an actual2000restart, completedepoch
or newtrackingresult. Evidence:
`training/full_joint_source_epoch1_actual2000_snapshot_v1/validation.json`.
Ordinary-model parity across all three tasks is unmet.

## Paired temporal foundation study now actually optimizing — 2026-10-16

The prepared short/long study no longer waits for unrelated native6000 TAO
controls before training. V3reuses the exact v2indexed24000question curricula
and training configs; same native6000FP32masters, independent newAdam/RNGseed0,
final6000steps per arm, short thenlong, same four unchanged tracking scopes.
Only scheduling/allocation changed. All43TRAINvideos/9789selected images and
short<=1s versus long3..104.2s teacher pairing remain unchanged. These are GT
teacher snapshots, not native actor rollout, deployedTTL extension or long-BPTT.

Real short-branch updates1..5 are verified, examples4/8/12/16/20, finite losses
and positive finite gradients. Actual320FP32/752393024initial parameters match
the saved native6000masters exactly (maxdiff0); vision remains frozen. The long
branch has0updates so far. First5peakTorchallocation16.8548GiB fits the actual
28,081,638,604byte Torch allocation cap (.33of reported device capacity); this
is not a full-run peak/driver-memory guarantee or quality evidence. The owner
counter increments only completed6000step stages; its native log.jsonl gives
intermediate progress. Evidence:
`training/paired_temporal_teacher6000_owner_v3_early_window/first5_short_optimizer_snapshot.json`.

The oldCPU-only waiter4002071/start52168753had0updates/nochildren/emptyCUDA
visibility; exactpidfd SIGTERM and originalsession60582exit143 are preserved
in `training/paired_temporal_teacher_v2_CPU_handoff_v1/{intent,receipt}.json`.
No training or inference worker was stopped. The newowner4118405/start52597639
and originalblocking session65852 hold an exclusiveGPU7window across both
training/reload stages. Own children inherit the lock descriptor; future GPU7
resource-wrapper launches wait. The already-running four-GPU epoch remains
permitted and untouched. Release this window BEFORE waiting original6000
control completion, then perform the original before/short/long comparisons.
This avoids a circular wait; existing GPU4/5TAO inference continues. No speed
comparison is inferred under changed co-resident load. The expandable allocator
environment persists for newowner training/reload/eval; reference arithmetic
and native inference backend remain unchanged.

RealCPUprocess wrapper regression proves ownFD entry, GPU4passthrough,
GPU7wait/release, and unchanged course/config references:
`validation/gpu7_temporal_window_v1/validation.json`. New4MiBscheduling metadata
raises the bounded temporary feature upper to26,823,828,446bytes<26,843,545,600;
no new image/ROI bank. No checkpoint/metric/seed/acceptance change or promotion.

Both Compact appearance studies completed first4scopes/all16units each. TAO40
TETA original14.517; raw/random/trained ranks14.104/14.324/14.967, fixed-affinity
native/random/trained13.964/14.194/14.395. AllNovelClsA0, same single peeler class
in this video; rank gain is small and does not establish general recognition.
Final naturalMOT20controls remain running under original owners73067/68145.
Receipts: `compact/{appearance_rank_study_v1/owner,native_temporal_appearance1000_owner_v1}/first_four_scopes_completed.json`.

## Native shared6000 RMOT gains verified; DT still regresses — 2026-10-16

All nine original native OneJev before/default/assigned-write controls have
completed and passed full frame/candidate/rejection/checkpoint/video integrity
validation on DT1203 and both RMOT800 scopes. This is genuine pretrained
752M language-model adaptation, separate from Compact projection/rank controls.

|HOTA %|Native600 before|Shared6000 default|Same6000 assigned-ID write|
|---|---:|---:|---:|
|DT1203|8.7030|7.8406|7.8406|
|RMOT single800|2.5062|3.3287|3.4052|
|RMOT multi800|6.7400|15.6144|15.8936|

MultiRMOT IDF1%=6.6932→15.6747→16.7555; default→assigned FP467→457,
FN1297→1303 and IDSW475→471 (before185). Thus recall/precision/identity
tradeoffs remain, and improved HOTA does not imply fewer switches than before.
SingleRMOT contains only18GTboxes and remains poor; do not generalize.

The assigned-ID write mechanism has real same-causal-input threshold effects
on RMOT:61calls/78existing assignments/5gate flips on single,620/1190/7 on
multi. Single has4HOLD→UPDATE/1UPDATE→HOLD, multi1/6. Each compares observation
and assigned-identity probabilities within the same assigned arm before memory
writes, not probabilities from divergent default histories. Exported output
frames are exact796/800(single) and388/800(multi); DT remains1203/1203exact,
1350calls/4940assignments/0gateflips. The earlier statement of no assigned-write
benefit applies to DT; both RMOT scopes now show small measured gains. Do not
select different checkpoints/policies by task or promote a common model yet.
Native6000TAO controls now wait the pairedGPU7trainingwindow after baseline completion;
the earlier0update short/long queue is superseded by v3actual training above. Original owners,
source input, thresholds and all failed samples are unchanged.

Receipt: `training/broader_shared6000_fla_owner_v2/first_three_native_scopes_completed.json`.
The new posthoc freezer first failed on normal lifecycle events without a
candidate field; failed code/receipt are preserved alongside it. The corrected
audit includes lifecycle events in signatures and limits write-probability
comparisons to candidate events, then passes all9original units/every event.
No model, inference output, metric or experimental acceptance rule changed.
No full dataset or ordinary-model parity, Ground+One joint gradient or long-BPTT.

## Appearance rank controls measured; native graph candidate rejected — 2026-10-16

The registered rank intervention is implemented and passed actual67group
checks:566candidate rows/8286existing-ID options per arm, exact original action
values/full probability multisets and auxiliary/semantic/lifecycle outputs.
It only permutes original existing-ID values by log-prior plus mean appearance
cosine/.07; global matching and future histories can change. These are derived
Compact categorical values, not native OneJev decoder probabilities.

The first three full scopes/all12official units have completed:

|HOTA %|Original learned|Raw native ranks|Random projection ranks|Trained projection ranks|
|---|---:|---:|---:|---:|
|DT1203|26.3157|30.2544|19.4111|24.7555|
|RMOT single800|6.0247|5.7087|5.8205|6.1548|
|RMOT multi800|19.8019|18.9930|19.6370|20.1424|

Raw-rank DT IDF1=30.6329→38.2885%, IDSW165→148, but FP579→720/FN1962→2156.
This isolated gain does not select different policies per task. The trained
projection's earlier fixed-affinity RMOT gain does not transfer at comparable
magnitude to this rank-preserving policy; retain the representation and policy
interaction as mixed evidence. Full TAO/MOT20 controls continue under the
original owner, no promotion. Proofs:
`compact/appearance_rank_study_v1/owner/regression/validation.json` and
`compact/appearance_rank_study_v1/owner/first_three_scopes_completed.json`.

The official Torch CUDA Graph component probe completed all four real requests.
Sparse MOT/root1203/fixedleaf128 answer differences0.011445/0.013179/0.013894
exceed the unchanged1e-3gate; measured median ratios1.044/1.057/1.061 are below
the registered1.5gate. Dense16candidate/32history graph requests fall back to
eager and take30.754s versus13.942s (ratio0.453); fallback is not graph success.
The same saved-answer posthoc audit finds no changes in68literal Choice answers,
17Noul>=.5gates or17row global Hungarian decisions. This does not prove trajectory
or memory parity, and dense answers are from fallback. No RMOT membership or
assigned-write questions were in this component probe. No acceptance criterion
or original inference backend was changed. The installed Qwen3.5 multimodal
forward contains no deepstack injection; missing deepstack is not an established
cause. Numerical root cause remains unresolved; do not patch on speculation.
Evidence: `optimization/native_graph_real_requests_v1/{result,eager_results,
graphs_results,runtime_decision_audit}.json`. Shared GPU, warmup excluded,
three repeats; no E2E/FPS claim. Upper temporary feature reservation is now
26,819,634,142bytes<26,843,545,600bytes; no new image/ROI bank.

Original four-GPU joint training has actual1700updates/27200questions logged,
elapsed20688.803s (5.747h), while the459027question/28690update epoch remains
incomplete. This is log evidence, not a1700checkpoint-state inspection; last
full numerical state proof remains1000. Original owner/training, queued paired
long/short teacher stages (0newupdates), and all pending controls remain intact.
Evidence: `training/full_joint_source_epoch1_progress1700_snapshot_v1/snapshot.json`.
No Ground+One joint gradient, long-BPTT, or three-task ordinary-model parity.

## Earlier projection snapshot — learned appearance helps RMOT under fixed affinity, hurts DT

The same temporal projection/fixed fusion recipe completed both full800-frame
RMOT scopes. Its effect is mixed, not an overall failed representation:

|HOTA %|Original learned|Fixed native cosine|Random projection|Trained projection|
|---|---:|---:|---:|---:|
|DT1203|26.3157|13.3108|13.4881|8.4607|
|RMOT single800|6.0247|5.5377|6.6298|9.2325|
|RMOT multi800|19.8019|16.6224|14.3821|31.5524|

MultiRMOT IDF1=21.0082→40.0107%, IDSW165→53 (fixednative105/random106).
The three fixed arms share fusion/auxiliary/semantic functions and differ only
in appearance representation; comparison to the original learned arm additionally
changes association and auxiliary policies. These are learned-representation
controls, not native OneJev decoder gains. No new thresholds or task-specific
checkpoint selection. DT fixed arms emit essentially every native candidate
(NEW.3 dominates IGNORE0), FP579→9328/9329/9330; trainedAssA also decreases
0.06641native→0.02618trained. The DT problem includes both proposal rejection
and association; restoring rejection alone is not assumed sufficient. No promotion.
The rank-control hypothesis in this earlier snapshot has now been implemented;
see measured results above. TAO/MOT20 projection controls still run.
Receipt: `native_temporal_appearance1000_owner_v1/first_three_scopes_completed.json`.

## Closed controls and paired long-history foundation inputs ready — 2026-10-16

Native residual study completed all15official arms/scopes; disabledMOT20
restores originalHOTA7.4316/IDF1=6.1474% exactly, versus enabled19.5518/17.4310%
with30874IDSW. DT/singleRMOT regressions remain; no common model promotion.
The paired physical/semantic study also completed all25controls. Canonical
and coupled MOT20 remain exactly original7.4316HOTA; TAO40canonical31.323
versus original14.517 remains isolated, with RMOT regressions. All originals,
final checkpoints, inputs and scorers remain unchanged.

The temporal appearance projection completed three full scopes/all12arms.
DT1203 HOTA% is26.3157original /13.3108fixednative /13.4881random /8.4607trained;
trainedIDF1=3.2767%, no promotion. Remaining TAO/MOT20 controls continue.

The category-evidence audit uses the exact official annotation+fullLVISsynset
aliases. This40frame video has3GT classes:knife/baby(Base),peeler(Novel).
Of40NovelGTboxes,37have native box support but only1has any correct native
box+label support. Of6one-to-one GTmatched emitted Novel boxes, neither raw
frontend labels nor learned head is correct. These are posthoc oracle/support
counts, not TETA/ClsA. They do not characterize allNovelclasses or justify a
raw-label restoration policy. No labels or runtime behavior were changed.

Paired native OneJev temporal-teacher curricula are now fully prepared and
CPU input-validated:24000rows each,8000per task, including4000original native
MOT rows and4000matched teacher positives. Same current images/boxes/aliases/
questions/targets and16000semantic rows/order; past teacher evidence changes.
Both cover43legalTRAINvideos with9789selectedsourceimages across current/
short/long observations. Short gap min/median/p95/max=0.0333/0.5/0.96/1s;
long=3/7.95/38.65/104.2s. Actual86native request proofs and172pixel sheets match
native rendering exactly. Teacher snapshots are not actor rollout, long-BPTT,
new runtimeTTL/archive or native detector recall; RMOT remains72physicalframes
in this controlled pair, while the independent459027epoch is broader.

All24000short/24000long native MM rows pass with0dropped,
6000planned optimizer updates each, maximum27319/27319tokens
and124050808/124044895plannedinputtokens. These are planned
inputs, not trained tokens. Both stages have0new optimizer updates and are
queued behind the verified original GPU7native6000owner; no old job restarted.
Same native6000FP32initial weights, new optimizer/RNG seed0, all752Mlanguage
parameters trainable, vision frozen; training-onlyFLA and originalTorch inference.
Both final6000models will precede unchanged four-scope default comparisons.

Preparationv1 failed its16MiBmetadata gate (22,748,803bytes from copied source
mixtures). All failed inputs and exact source snapshots are preserved. V2
reuses original selected row indices under an explicit new source protocol;
8legacy and3indexed real rows remain exact. V2metadata is2,849,670bytes before
summary/examples. Original row IDs and common20000+teacher4000 ordering are
explicitly different from untrainedfailedv1, matched across paired arms. No v1
training trajectory existed. V2keeps its16MiBcap;24MiBseparately reserves the
failed input preservation. Conservative combined upper26,811,245,534bytes
<25GiB(26,843,545,600); no new image/ROI feature bank or GPU allocated.

Evidence: `native_relation_head8000_owner_v1/all_five_controls_completed.json`,
`physical_factorization_pair8000_v1/all_five_paired_controls_completed.json`,
`native_temporal_appearance1000_owner_v1/first_three_scopes_completed.json`,
`diagnostics/frozen_TAO40_category_evidence_v1/summary.json`,
`training/paired_temporal_teacher6000_owner_v2_indexed/prepared_and_CPU_validated.json`.

## Canonical physical stream gains on TAO40; RMOT regresses — 2026-10-16

Paired physical-factorization studies completed four unchanged scopes; natural
MOT20full1657 remains running. Same final8000 models, no task-specific choice:

|Scope/metric|Original|Coupled heads8000|Canonical heads8000|Canonical untrained|
|---|---:|---:|---:|---:|
|DT1203 HOTA %|26.3157|26.3157|26.3157|26.3157|
|RMOT single800 HOTA %|6.0247|0.0000|3.1383|2.0050|
|RMOT multi800 HOTA %|19.8019|19.1420|12.0955|11.7532|
|Raw TAO40 TETA|14.517|14.871|31.323|26.125|

TAO canonical LocA8.088→33.075 and AssocA5.1478→35.637 are shared with the
untrained canonical intervention; semantic training raises its ClsA9.6637→25.259
(original30.314). NovelClsA remains0 in all arms; the NovelTETA gain5.679→28.637
comes from localization/association, not Novel category recognition. This is
an isolated registered40frame component result, not fullTAO or three-task
ordinary-model performance. The physical intervention trades recall84.833→55.208
for precision8.2072→45.298 and alters trajectories. RMOT regressions prevent
promotion. Receipt: `physical_factorization_pair8000_v1/first_four_scopes_completed.json`.

The native-relation residual final model also completed naturalMOT20full1657:
HOTA7.4316→19.5518%, IDF1=6.1474→17.4310%, FP463→5592, FN286273→159383,
IDSW2060→30874. Detection gains coexist with strong identity fragmentation;
final residual-disabled MOT20 ablation still runs. Its DT/single-RMOT regressions
remain. Receipt: `native_relation_head8000_owner_v1/all_five_enabled_scopes_ablation_last_pending.json`.

## Cross-frame native identity representation trained; causal controls running — 2026-10-16

A task/query-independent341,184parameter projection was trained on immutable
TRAIN registry GT identities, independent of actor birth aliases/actions:
43units/1056frames,48051raw candidates/40059matched observations,887IDs,
873cross-frameIDs and14singletons preserved. All43units are contrastive-eligible.
No labels, original corpus or validation samples were changed. Same-video
supervised contrastive sampling uses up-to8IDs/up-to4distinctframes each.
The fixed1000-step seed0 recipe consumed27797ROI exposures across all43units
in36.587s; model/Adam finite/steps exact, export reload exact. Actual8vs4+4
model/Adam/Torch/sampler/exposure state matches bit exactly. These are short
TRAIN clips and small projection training, not long-BPTT or foundation SFT.

On67real native-feature groups, published default affinity/semantic outputs
remain exact;16572projected pair scores match the registered fusion formula
exactly and semantic outputs remain unchanged. Geometry was consistently
rescaled for this representative formula regression, not full-video parity.
Original/native-cosine/random-initial/final1000 controls now run all five
complete original scopes with unchanged fixedfusion/NEW/IGNORE/TTL and scorers.
They are affinity scores, not native JEV probabilities. No tracking gain yet.
Reference metadata104862bytes, no new persistent ROI bank or GPU. A48MiB
reservation includes16MiBmetadata and32MiBbounded streaming buffers, raising
feature upper to26,769,302,494<25GiB(26,843,545,600). Model/Adam/scientific final
identity states remain separately protected. Original native750M four-GPU
joint training, native6000controls, Ground retention and legacy workers continue.
Evidence: `native_temporal_identity_refs_v1/summary.json` and
`native_temporal_appearance1000_owner_v1/{prefix_validation,completed_training_summary,runtime_validation/validation}.json`.

## Paired factorization training complete; native final-ID DT control has no gain — 2026-10-16

Both independently trained semantic-head-only8000-step models completed all
16000questions (8000RMOT+8000OV): coupled115.290s, canonical116.032s. Each
updates65,922of1,146,511parameters; all59original fixed tensors/1,080,589elements
are exact unchanged. Full model/Adam finite, exports equal saved states and
all three task reloads exact. This is small-head adaptation with frozen shared
representations; the original native750M four-GPU joint training continues.
Official full five-scope controls now run for both final checkpoints, with
original/untrained-canonical controls. Results and promotion remain pending.
Receipt: `outputs/groundjev/compact/physical_factorization_pair8000_v1/paired_training_validation.json`.

The native OneJev6000 final-ID DT1203 control completed: HOTA7.840637%,
IDF1=3.534137%, same as default and below native600HOTA8.702996%. There were
1350actual second-pass native calls on4940existing assignments. All4940update
probabilities changed, maximum absolute difference0.018702, but none crossed
the registered0.5gate; every exported tracking output and all official metrics
remain exact between default/assigned arms. This explains the absent measured
DT effect without changing thresholds. Remaining RMOT/TAO scopes continue.
Receipt: `outputs/groundjev/training/broader_shared6000_fla_owner_v2/DT1203_default_and_assigned_completed_comparison.json`.

## Physical/semantic factorization paired study — 2026-10-16

A new opt-in Compact physical stream uses the original fixed MOT/person
specification, while the actual query/task remains in semantic readout.
On67real observation groups (one from each61native units plus6Base objects),
all335task/query interventions leave match/update/reliability/lifecycle tensors
exactly unchanged; all335alter original coupled physical tensors. All67
canonical MOT forwards equal the original for every output tensor, while
membership changes with queries in all335interventions. This is conditional
input invariance, not a tracking improvement, and does not cover changed
frontend proposals or query-based output filtering. Complete DT1203 replay also passed: all1203frames/13664candidates retain
every original output/state/shadow assignment exactly, and every canonical
model output tensor equals the original. Paired real resume checks precede
training under the single owner. Both paired8vs4+4model/Adam/RNG
resume checks passed exactly (16real semantic questions per compared branch);
both8000-step production studies have launched.
The paired recipes freeze the same physical weights and train only semantic
heads on identical16000RMOT/OV questions/order/seed/loss/hyperparameters,
8000updates each; no MOT or shared-backbone gradient. Coupled and canonical
final checkpoints, original and untrained-canonical controls use all five
unchanged scopes. The original native750M four-GPU joint owner is untouched.
Registered8MiB metadata reserve raises feature upper to26,718,970,846bytes
<25GiB; no new ROI bank or GPU. Code/plans/initial proof live under
`physical_factorization_pair8000_v1` and `compact_physical_*_semantic8000_v1`.

Native residual TAO40 controls are now complete: TETA14.517original→14.320
enabled versus14.646same-final semantic weights/residual disabled; NovelClsA
remains0 in all arms. Together with DT/single-RMOT regressions this prevents
promotion despite the isolated multi-RMOT gain. Natural MOT20 is still pending.
Receipt: `outputs/groundjev/compact/native_relation_head8000_owner_v1/first_four_scopes_completed.json`.

## Four-GPU joint training: actual1000 full state verified — 2026-10-16

The original four-GPU owner saved an actual1000-step/16000-question full
training state. CPU inspection verified all320FP32 parameter tensors
(752,393,024elements), both full Adam moments (1,504,786,048elements), exact
optimizer/scheduler/cursor1000 and all four ranks' Torch/CUDA/Python/NumPy RNG.
Actual consumed task questions are8817MOT/3537RMOT/3646OVMOT. The retained
numbered BF16 export is `ckpt-step1000`; the latest full state is atomically
replaced at later saves. This observer performed no restart or native inference.
The original progress follower subsequently reported1125steps/18000questions,
14248.87s elapsed. The459027-question epoch and its final tracking controls
remain unfinished. Shared OneJev language parameters are trained; visual encoder
is frozen, GroundAnything has no joint gradient and long-video BPTT is absent.
Evidence: `outputs/groundjev/training/full_joint_source_actual1000_observer_v1/validation.json`.

## Native-relation adapter8000 training complete; first controlled results — 2026-10-16

The frozen-core adapter completed8000updates/24000questions in1040.460s
(17.34min). All59fixed original weight tensors remain exact; full model/Adam
are finite, final export equals saved training state and all three reloads
agree exactly. The same-final-semantic-weights residual-disabled checkpoint
is an inference ablation, not a separately trained model.

|Scope/metric|Original|Native residual|Residual disabled|
|---|---:|---:|---:|
|DT1203 HOTA %|26.3157|15.3059|26.3157|
|RMOT single800 HOTA %|6.0247|0.0000|1.6577|
|RMOT multi800 HOTA %|19.8019|24.1906|19.1828|

Multi-RMOT IDF1 improves21.0082→29.2044%, IDSW165→97. DT regression and zero
single-query HOTA prevent promotion; TAO and natural MOT20 controls continue.
This establishes a component gain on one registered multi-query scope, not
ordinary-model performance or native OneJev decoder gains. Evidence:
`outputs/groundjev/compact/native_relation_head8000_owner_v1/`
`{training_validation,frozen_core_and_ablation_validation,first_three_scopes_completed}.json`.

## Completed controls, failed repair hypotheses and native-relation adapter — 2026-10-16

Shared and matched MOT-only8000 studies completed all five unchanged causal
scopes, including natural MOT20full1657. Neither is promoted:

|Scope/metric|Original|Shared8000|MOT-only8000|TRAIN-prior8000|
|---|---:|---:|---:|---:|
|DT1203 HOTA %|26.3157|5.5800|1.6800|2.5567|
|RMOT single800 HOTA %|6.0247|1.0579|2.4910|3.5421|
|RMOT multi800 HOTA %|19.8019|12.5237|17.0572|15.7803|
|TAO40 TETA (0–100)|14.5170|19.2520|3.6599|6.6607|
|MOT20full1657 HOTA %|7.4316|11.4780|5.6407|3.1083|

SharedMOT20 IDF1 is7.8200%, IDSW58632 versus original2060; sharedTAO NovelClsA
remains0. More detections/localization do not establish stable identity or
three-task gains. TRAIN-prior8000 completed8000questions/944.302s and all five scopes; its
naturalMOT20 score also remains below ordinary-model performance.
The separate native OneJev6000 defaultDT1203 also regresses: HOTA8.7030→7.8406%,
IDF15.8260→3.5341%, with all1203frames/13664candidates/27rejections retained.
Assigned-ID and remaining task controls continue under the original owner.

The exact-original-prefix diagnostic reproduces every original output/state
and shadow assignment on DT1203. With1768correct-existing options, argmax
correct is651original/6shared/55MOT-only/3prior-weighted; new models already
reject valid existing options before their history changes. Zero-age or
zero-confidence/four-scalar one-step interventions do not repair this
(43/1/26correct respectively). These are conditional diagnoses, not HOTA.
A report-scope correction explicitly marks altered scalar tensors; visual/box/
pair/specification tensors remain identical.

A read-only birth-support audit matches all48051original TRAINregistry targets.
It identifies2/844TRAIN births with uniquely GT-supported actual emissions left
unbound, and4isolated valid edges lost by assignment-before-threshold. These
are conservative unambiguous-case counts; remaining ambiguous births are not
certified correct and the broader birth-label mechanism is not ruled out. No
existing labels or running native foundation protocol were changed. Its first
run failed only at final JSON printing; failure retained and completev2 rerun
exited0.

A new frozen-core Compact adapter is implemented and actually training onCPU:
1,147,379total parameters,66,790trainable, including868zero-initialized native
ROI relation/action residual parameters plus category/membership/scale heads.
Original backbone, text conditioning, association and auxiliary weights are
frozen. New inputs use real anchor/recent/long cosine and existing IoU/distance,
not GT. All24000new questions/options validated (8000identity/8000membership/
8000Basecategory); eligible images1016MOT/1795RMOT/3227OV, unsupported263query-
frame contexts explicitly inventoried and original sources retained. This is
a new identity-onlyMOT recipe, not a compute-matched continuation.

Zero-initialized outputs agree exactly with the original on every captured
DT1203group, and original causal states/assignments remain exact across1203
frames/13664candidates. Actual three-task8versus4+4resume is bit exact for
model/Adam/RNG. Production passed at least3000updates/9000questions; its observed saved
state verifies all59fixed tensors/1,080,589original elements exactly unchanged
and all model/Adam values finite. Final8000 and tracking gains remain pending. The original actor, final residual-enabled model and a
same-final-semantic-weights residual-disabled inference ablation will use all
five existing scopes. The ablation is not separately trained. This adapter
freezes shared representations and is not native OneJev decoder reasoning or
joint shared-backbone fine-tuning. The independent native four-GPU459027-
question epoch continues (last verified825updates/13200questions).

No new ROI bank or GPU was allocated; an8MiBmetadata reservation raises the
conservative temporary-feature upper bound to26,710,582,238bytes below25GiB
(26,843,545,600). Ordinary-model performance across all three tasks remains
unachieved.

## Earlier registration and first100 snapshot — 2026-10-16 (superseded)

The balanced shared native OneJev study completed6000updates/24000questions
(8000per task),29432.9s. Final full FP32/Adam/RNG state and native-MM reload
are validated; official tracking controls are running, no new gains established.
Evidence: `outputs/groundjev/training/shared6000_actual6000_snapshot_v1/validation.json`.
The independent four-GPU459027-question epoch remains unfinished.

Selected175Base teacher feature collection completed all3227images/3649objects
in a single37,365,888-byte original FP32 native ROI bank. The private ROI-only
service stopped; no DLM proposal or optimizer calls were made by collection.
All175class representatives match original Compact inputs and frozen forward
outputs exactly (216,205,500input tensor elements). Train-Base GT boxes are
declared teachers; this bank is not native detector proposal recall.

A fresh CPU shared Compact study is registered and launched:8000updates with
one MOT, one RMOT and one OV typed question per update,24000total. It reuses
existing native appearance banks and the single selectedBase bank, keeps the
existing architecture/runtime, and selects one final checkpoint for five
unchanged causal control scopes. Actual eligible first-media coverage is
1053MOT/1795RMOT/3227OVimages. Original1056MOT/1814RMOT frames and unsupported
questions remain preserved;226query-frame contexts with no selected supported
head are explicitly inventoried. It excludes RMOT unknown-birth auxiliary
proxies and invents no temporal labels for teacher OV objects. The balanced
question-level recipe differs from old whole-group training, so the planned
before/after is not an isolated data-only ablation. All24000selected source
labels/options passed validation, and a real8update versus4+4resume prefix
matches every model/Adam/counter/RNG value exactly. Full8000 completion and
tracking results remain pending. Evidence under
`outputs/groundjev/compact/broader_question_curriculum_v1/`; owner
`outputs/groundjev/compact/broader_question_joint8000_owner_v1/`.

Current registered temporary-feature upper bound is26,702,193,630bytes,
below25GiB(26,843,545,600). No original running training/evaluation job was
stopped, and all project GPU work remains within physical4/5/6/7.

# GroundJEV-MOT current evidence — 2026-10-16

## Actual four-GPU shared training; final quality still pending

Full-source shared OneJev training has reached25actual updates/400questions;
see [the recorded update](../../outputs/groundjev/training/full_joint_source_epoch1_follow_v1/step25_snapshot.json).
Its first five real updates consumed80questions across all three tasks:
49MOT/12RMOT/19OVMOT. The
one-shot observer exited0 after verifying finite loss/gradients, exact native600
FP32 initialization and live ranks on physicalGPUs4–7. The459,027-question
epoch plans28,690updates; it is running, not complete. See
[the immutable execution snapshot](../../outputs/groundjev/training/full_joint_source_epoch1_first5_observer_v1/snapshot.json)
and [the current training log](TRAINING_LOG.md).

Both full native CPU plans have completed: zero dropped questions, identical
serial/four-process summaries,5,639,441,074planned multimodal input tokens.
Filtered source images are1,056MOT/1,796RMOT/17,023OVMOT; these describe the
whole input corpus, not the first80questions' actual image exposure. RMOT's
raw1,814-image source includes18images whose original questions are only
lifecycle, outside the registered semantic-only selection. The source
frames/rows remain intact. Full proof and coverage explanation are in
[completion_validation.json](../../outputs/groundjev/training/full_source_parallel_cpu_plan_v1/completion_validation.json).

The independent balanced24k/6000-step study has reached at least5300updates;
its complete5000-step FP32/Adam/RNG save was observed. Original Full old/
refreshed controls and Ground adapted200 retention continue under their
existing owners. The proposed CPU-only handoff did not signal or replace any
job: original four-GPU training had already started. There is no Ground+One
joint gradient or long-video BPTT. Neither a completed large epoch nor
ordinary-model parity on MOT/RMOT/OVMOT is established.

Completed old-history Full600 controls include DT1203 HOTA8.703→13.107%,
RMOTsingle800 2.506→3.072%, and RMOTmulti800 6.740→6.683%; TAO and refreshed
arm measurements are still pending. Compact hybrid32 improves the fixed
MOT20 CPU control HOTA7.432→8.275% while ID switches increase2060→2348.
These partial/mixed findings do not promote a new universal model. The2882
known-birth RMOT identity/write rows are prepared but were not inserted into
either frozen running curriculum. Next decisions depend on actual completed
tracking comparisons, not training loss or input-plan size.

## Broader Compact input references prepared, not a trained model

The61completed native MOT/RMOT source rollouts now have a reference-only
Compact input corpus:4,684query-frame contexts,1,056MOT/1,814RMOT unique
images,7,071ordinary groups and5,870existing-assignment conditions. About21MB
of geometry/reference metadata reuses the original FP32 ROI arrays without
copying per-group history vectors or images. All45empty n=m=0contexts are
retained; the existing Compact model does not execute a forward there, and no
new presence head is claimed.

On nine actual source frames, all36native forward groups reproduce6,784,728
input tensor elements and every frozen forward output exactly, with the
actual source tracking outputs/state unchanged. This numerical regression
does not claim a newly trained tracking gain or a whole-corpus forward check.
See [completion evidence](../../outputs/groundjev/compact/broader_native_reference_inputs_v1/summary.json).
The175Base OV feature component and a new shared Compact training recipe
remain unfinished; this CPU-only preparation launched no GPU or optimizer.

## Earlier source-collection snapshot — superseded

The source collection described below has since completed; its former active
state is retained as history. Current execution evidence is above.

All nine original GroundAnything tasks now have their full unchanged native
summaries: 91,124 original/effective rows, zero provider-policy filtering and
no missing task. Every raw result's metrics and sample counts match the
summary. Original registered clients have exited and service group 1345024
is gone. The original parentless runner's exit code remains unknown; it is
not invented as zero. Adapted200 full9 on GPU6 is still pending under the
same original reattached owner, so this is not a complete before/after result.

GPU7 release is verified separately for the completed original variant.
The new source collector uses the original DLM and real ROI service there,
with the fixed existing sharedCompact actor on CPU4. All 43 frozen MOT TRAIN
videos and 1,056 source images are registered: MOT17 7x32 frames, DanceTrack
32x16, and MOT20 two32/two128-frame clips. Intervals are chosen chronologically
from the middle of registered train frames, without GT/visibility screening.
The first complete real clip has 32 frames/992 native candidates/9,818,682
archive bytes and no rejected frames. Collection is active, not complete;
there are zero new OneJev training updates from this source yet.

Two startup failures are retained: v1 included held-out-only protocol sequences
with empty train_frames and failed before launching a model; v2 omitted the
required person vocabulary and stopped before its first frame. Its owned
services were fully stopped. v3 fixes these input contracts, validates all43
TRAIN manifests, and now runs actual native frames. No planned train frame is
omitted or substituted. All future parser rejections remain registered empties.

The completed immutable MOT20 archive and two finalized Compact packs now
use their measured bytes rather than unused caps; no file was deleted.
New collection reserves 768MiB exact FP32 archives plus64MiB future group
export/64MiB future mixture references. Conservative upper is
26,739,639,402 < 26,843,545,600 bytes. This expands physical support before
broader shared SFT, rather than restarting the old U2 whole-dataset cache.
Evidence: ground_retention/reattached_full9_owner_v1/original_* and
training_data/broader_native_MOT1056_gpu7_v3_complete_manifest/ under
outputs/groundjev/. Original collection handle84427, owner3032116/start47104873.

## Actual Compact own-history1200 pair trained; partial tracking regressions

Both new CPU sharedCompact arms completed 1,200 updates with 400 sampled
groups per task. Their actual per-step sample IDs match exactly. Each has
1,146,511 parameters; three actual task reload probes have maxdifference0.
The final training-state model tensors equal the inference checkpoint and
are finite, with sampler/RNG/protocol saved. Native per-parameter Adam step
histogram is identical:56 tensors at1200,6 at1181,4 at1067. Conditional heads
do not all receive gradients on every group. The first receipt check failed
because Torch integer sampler keys were compared directly with JSON string
keys; normalized values match exactly. No optimizer state was changed.

Each arm has80 physical MOT groups/1,184 candidates/48 fit frames plus the
same325RMOT/62OV groups. Both explicitly omit69 derived-withholding duplicate
groups and disable the MOT .2 distillation term in this new continuation.
The old arm retains original Full100 histories; the new arm captures actual
pre-write Compact forward tensors and re-evaluates the same label rules.
All48 instrumented actor outputs/states reproduce the prior diagnostic
exactly; image/candidate/specification tensors match the original physical
groups. History/targets differ, and sequence lengths/FLOPs are not matched.
The comparison is not an isolated age or confidence intervention.
Both feature packs total223,700,168 bytes; models/full optimizer states are
preserved separately. Training-loop timers25.314/25.501s are for this tiny
Compact continuation, not foundation training or pipeline throughput.

Snapshot of already-validated CPU official results (percent):

| Scope | Before | Old-history1200 | Own-history1200 |
| --- | ---: | ---: | ---: |
| DT1203 HOTA |26.316|9.612|12.251|
| RMOT single800 HOTA |6.025|7.678|10.049|
| RMOT multi800 HOTA |19.802|20.686|22.063|
| Raw TAO40 TETA |14.517|10.426|pending|

The five-scope cohort also includes the same full MOT20-05 source. Eleven
of fifteen units are in this frozen snapshot; pending units are explicit.
Both new checkpoints regress badly on DT despite some RMOT improvement.
No checkpoint is promoted or selected per task. The original single CPU
owner70970 continues all registered units. The source collector deliberately
keeps the existing baseline actor rather than promoting these partial models.
Evidence: compact/own_history_refresh1200_pair_v1/
{training_completion_validation,partial_tracking_results}.json.

## Frozen action-support and actual temporal-domain diagnosis

The final-stage eligible MOT pack has2,204 candidate targets:1,704existing,
229NEW,202IGNORE,69NO_MATCH. It is not dominated by NO_MATCH. Historical
per-step sampled IDs were not saved for that old stage; eligible proportions
are not asserted as actual optimizer exposure. The fixed checkpoint's CPU
fit-input exact ID/action accuracy is69.83% MOT/80.97% RMOT/92.56% OV; MOT
includes380existing->NO_MATCH and71NEW->NO_MATCH errors. These are fit probes.

All MOT eligible history ages are1 observation, confidence .991545.. .999416.
Across all three task inputs the maximum age is7. Actual completed MOT20
candidate groups contain410,796 history rows:95.752% have ages above that
joint eligible maximum, max909/median121 observations. Confidence median is
.429819;20.551% of rows exceed the eligible maximum time since reliable write.
The audit reconstructs all1,657 frames/235,232 candidates; surviving final
age/lost/reliable/timestamp/confidence fields match saved binary memory exactly.
Empty-candidate groups do not expose saved shortlist inputs and are excluded
only from scalar-domain rows; their frames still update reconstructed state.
No feature/input intervention was made; these shifts are not causal proof.
Evidence: diagnostics/compact_action_temporal_domain_v1/summary.json.

Original GPU4 Full history-refresh training remains separate, last observed
125/600 with finite loss/gradient. The original GPU5 natural CPU controls and
GPU6 adapted retention also remain active. Natural CPU gated3 reference HOTA
7.4316 differs from fresh GPU7.3347; CPU variants are compared with their CPU
reference. No CPU/GPU exact closed-loop equivalence or ordinary-model success
is assumed. Broader data and controlled actual shared foundation training
remain required for the full goal.

## Fresh MOT20-05 completed; severe NO_MATCH suppression — 2026-10-16

The original natural-study owner completed fresh original-DLM+ROI+sharedCompact
inference on all1,657frames(1659..3315),235,232native candidates, retaining all
117parser-rejected frames as registered empty predictions. Official TrackEval
12c8791 gives HOTA 7.334696%, IDF1 5.959845%,IDSW1,933,FP417,FN288,297.
All31,185raw prediction lines reconstruct exactly from the frame outputs.
The bounded packed archive contains1,657FP32feature files/2,311,933,077bytes
under its original3GiBcap. Final binary memory and1,657-frame video are retained;
the original owner already verified every archive candidate and video decoding.
Fresh source validation81592exit0 independently checks frame/image/event/
prediction conservation, official/source records and actual archive sizes.

The actual native pipeline took27,000.258s(about7.50h),.0613698FPS including
DLM/ROI/IO/archive/state/drawing/encoder completion; preloaded remote service
startup is excluded. Five-second sampled ownedGPU5sum peaks22,812MiB, a lower
bound on instantaneous peak. This is one dense legal validation half, not full
MOT20 benchmark or standalone/realtime throughput.

Posthoc one-to-one physical candidate recall atIoU.5 is219,060/319,062=68.6575%.
Actual candidate actions:287NEW/27,604ACTIVE/3,294REACTIVATE/185,585NO_MATCH/
18,462IGNORE. Among GT-matched candidates,169,106NO_MATCH decisions violate
the existing missing-correct-option diagnostic; posthoc precision2.7165%.
158,687of these have no surviving known-correct birth anchor, where the
registered diagnostic expectsNEW. Only212of194,026such observations becomeNEW.
This identifies a major learned action/initiation failure alongside detector
coverage limits; it is not an official FP count or a causal threshold intervention.
Future fit supervision must addressNEW/NO_MATCH, beyond memory-write gating.
The eight original CPU memory controls are now running with unchanged policy.

Of132natural3..30s candidate returns,20have eligible known birth anchors;
the old correct ID survives in3and is recovered in0. The112unknown/no-correct-
anchor events remain excluded from this particular eligibility denominator.
No long-gap gain or ordinary-model parity is established. Published artifacts:
completed_fresh_source_validation.json, fresh official/result/raw predictions,
association_summary.json and natural_recovery_case.json under
outputs/groundjev/natural_memory/mot20_05_full_validation_gpu5_v1/.
The multi-GB native archive and full event trace remain on the server.

## Cross-actor final-ID component with streamed crops — 2026-10-16

Prepared722supervised questions(377yes/345no) from the fixed sharedCompact's
actual causal48fit-frame history. All1,184candidates are registered; all728
actual existing assignments, including6unknown/unmatched without target,
remain in exported input states. Full v3 is a rendering of Compact actor
state, explicitly not Full own-history or GT-corrected history. A fresh CPU
rollout reproduces every completed diagnostic output/state exactly before
export. Shared final-ID builder/alias/history/crop provenance round trips pass
for all722questions; images are always current/past legal-fit observations.

New independent tracking recipe reader reconstructs exact native candidate
and anchor/long/recent sheets from original images and real crop provenance.
All80groups/155sheets match native builder pixels; two actual31/32-history
native processor comparisons match every inputid/mask/mmtype/pixel/grid tensor.
Four ordinary-legacy/Base-category media cases pass unchanged passthrough.
The separate SFT entry records a new explicit IO protocol; existing default
entry/category IO and registered GPU workers remain untouched. Bounded CPU
caches hold at most8original images/192observation crops per process.

Full native CPU planning63984exit0 retains722rows/0drops,max26,614tokens,
14,664,308total planned tokens,actual plan_epoch2.536s,world1CPU.
181is only a hypothetical4-row component epoch; no optimizer or GPU training.
Independent target/alias/provenance validation60712exit0 passes. Raw gzip
3,482,066bytes/registry retained locally; generator, reader, plan and small
receipts are published. NoPNG/feature archive;16MiBreserve makes conservative
26,477,318,925<26,843,545,600bytes. This supplements positive support beside
existing Full hard negatives; physical coverage stays48frames and no tracking
gain is established. Evidence: training_data/compact_actor_assigned_update_component_v1/
under outputs/groundjev/. Current refreshed-history600training is ongoing;
the original handle last reported50/600with finite loss/gradient and saved
complete FP32/Adam/RNG cursor50/200.

## CPU Compact fit-history support diagnosis — 2026-10-16

The final-ID component has only29positive versus584negative known Full
assignments. To test a concrete source of richer support, the existing fixed
sharedCompact checkpoint ran causally onCPU/FP32/2threads over the SAME48
legal-fit frames/1,184native proposals, with original gated.5/3s memory,
group16/shortlist32. No GT enters or corrects the actor. All image/proposal
records, source perception and empty/rejected-frame handling are retained.

The first attempt(session23118exit1) stopped before any frame output because
the Full replay lacks native_region_feature. The repaired v2 joins the
already-existing native_pack_v1 ROI tensors by exact sequence/frame/batch
order and every candidate box. Only duplicate option-withholding augmentations
are excluded from this physical-ROI lookup; all1,184physical proposals remain.
No new detector, ROI request, image/feature archive or GPU job was created.
The original failure is preserved under compact_fit48_identity_support_v1.

The complete CPU actor(session34284exit0) produces728existing assignments:
620known-consistent,102known-conflicting,2unknown birth and4unmatched; plus
82NEW and374without final identity, conserving1,184. Under the unchanged
identity-consistency/visibility>=.6 rule,722known assignments could supply
377yes/345no final-ID targets. The six unknown/unmatched remain without targets.
Independent reconstruction validates all48frames and1,184external IoU matches,
immutable birth anchors, eligibility/targets, saved event/output boxes/labels
and finite candidate probabilities.

These are different actors' endogenous histories on the same small fit scope,
not a same-event causal comparison, held-out tracking score or Full gain.
No SFT component has been exported or trained from this diagnosis. The result
supports considering an explicitly cross-actor final-ID component alongside
the original Full hard negatives; it does not expand physical-frame coverage.
The current old/refreshed600training and four-scope queues remain unchanged.
All small raw traces/labels/receipts total2.33MB; new16MiB reserve gives
conservative26,460,541,709<26,843,545,600bytes. Evidence:
outputs/groundjev/diagnostics/compact_fit48_identity_support_v2_existing_ROIs/
{summary,validation,owner_state}.json plus actor_frames.jsonl and
posthoc_candidate_labels.jsonl; reproducible code/plan under tools/configs.

## Old-history600 arm completed; refreshed arm started — 2026-10-16

The original paired owner completed the old-history arm:600updates/2400rows,
800per task,15,605.720s for the whole uninterrupted training process
(about4h20m). Logged losses/gradient norms remain finite; final logged loss
1.020472 and peak allocated memory21.618GiB. These training values do not
establish tracking accuracy or standalone end-to-end throughput.

Checkpoint600/final exports are saved. Two actual native multimodal reloads
retain474weight entries and give identical probabilities(maxdifference0).
A separate CPU mmap read of the9,029,140,441-byte full training state verifies
all320FP32parameter entries/752,393,024elements and both Adam moment tensors
per parameter are finite; all320Adam counters equal600. Scheduler, all saved
RNG states and cursor600/2400 are present, and saved protocol matches actual
args. This is structural/numerical checkpoint validation, not a new exact
restart-trajectory experiment. Small raw records and curated receipt are
published; multi-GB model/master files remain on the server.

The same original owner automatically started refreshed-history training on
GPU4. Its320FP32entries initialize from the same native600source with maximum
difference0;600planned updates and0overlength drops are recorded. Both arms'
current observation questions/runtime remain unchanged, including default-OFF
assigned-ID writes. The old/refreshed four-scope tracking workers retain their
original dependencies and have no new metrics from this completion.
No new GPU queue or feature archive was created by this validation.
Evidence: outputs/groundjev/training/onejev_history_refresh600_old_v1/
{completion_validation,checkpoint600_validation,initial_fp32_state_receipt,latest_training_state}.json
and log.jsonl.

## Opt-in post-assignment native write path — 2026-10-16

Implemented `AssignedIdentityUpdateDecision` and explicit
`run_video.py --assigned-identity-updates` (Full v3/gated only, default OFF).
The original association/lifecycle pass and global matching run first. Before
any current-frame track mutation, the new path asks native Noul questions using
the SAME final-ID builder as the prepared613-row component. All assigned existing
IDs enter the request; there is no runtime GT filtering. These answers govern
existing long-memory writes at the unchanged.5threshold. Original observation
Q3 probabilities remain logged, and NEW initialization, boxes/recent memory,
association, lifecycle, RMOT membership and category policies remain as before.
The category wrapper forwards the callback for this explicit backend. Resuming
an existing run requires its recorded observation/final-ID protocol to match.
Current registered workers do not enable the new flag.

CPU execution validation completed(session64482exit0): the parent tracker blob
`9f092b04` and new default path reproduce all three saved complete Full MOT
trajectories exactly:1,203frames/13,664native candidates/all27rejected frames per
arm, identical outputs/per-frame state, and parent/current final memory equality.
Native original features and actual images are used; no new model forward.
Visible-track lifecycle probabilities absent from traces are not invented as
original full-memory truth. This is a frozen-decision compatibility check.

On the native600fit48original prefixes,96single-frame branches with explicitly
scripted gate answers0/1 exercise the new path:1,316questions,150pre-write calls,
all658existing assignments per gate value including all45unknown/unmatched.
All1,316training/runtime states/questions match after the declared identity
alias mapping;150historical crop-sheet pixel comparisons pass with saved aliases.
Long memory and last-reliable timestamps hold/update as scripted, original
association outputs remain identical, and category callback forwarding passes.
This is execution/schema/pixel evidence, NOT native second-pass model inference,
a causal continuous rollout, trained gain or a new tracking score.

The new questions still require matching SFT, balanced known positive/negative
support, fresh native-MM reload and controlled continuous three-task evaluation.
No new GPU job or feature archive was launched. All new small receipts fit the
already registered16MiBreserve; conservative upper26,443,764,493<25GiB. Evidence:
`outputs/groundjev/validation/assigned_identity_runtime_v1/{owner_state,validated_summary}.json`.

## Final-assigned-ID update component — 2026-10-16

Implemented a separate question builder, not enabled in the current tracker:
`locatemot/groundjev/assigned_identity_update.py`. It preserves the original
causal visual state and adds all actual existing-ID assignments from final
global matching, including unknown/unmatched assignments. Each new Noul question
explicitly asks whether this candidate should enter that specific identity's
long-term memory. No GT eligibility controls which assignments enter the input.
The builder is now shared with the opt-in post-global-matching/pre-write path
above; matching SFT and fresh causal tracking evaluation are still required.

Prepared the component from the already-completed native600actor's48legal-fit
frames/1,184native candidates. All candidates are registered:82NEW,444without
final ID,45known-consistent existing,568known-conflicting existing,15unknown
birth and30unmatched existing. Actor IDs map consistently to saved randomized
history/crop aliases using actual shortlist order. The613known assignments
supply29yes/584no labels, using immutable birth-ID consistency and the original
visibility>=.6proxy. The45unknown/unmatched existing assignments have no
supervised target and remain present in the input assignments. The300old
candidate-observation Q3yes/new assigned-IDno cases are different questions,
not silently changed labels. All original questions/targets and actor states
remain untouched; no GT state correction, new image artifact or model call.

All613new questions retain exact old state apart from the declared final
assignment, retain exact media references and name the actual selected input
alias. Twelve independent audit conflict examples reproduce exactly. Full CPU
native planning exits0:613rows,0drops,max27,234tokens,total13,305,265tokens,
actual plan_epoch17.393s;154updates is only a hypothetical4-row component epoch.
All component files are under16MiB; conservative total26,443,764,493<25GiB.
This tiny, highly imbalanced component is not trained or deployed and does not
establish accuracy, runtime/training equivalence or ordinary-model performance.
Current paired600training/questions/loss and four-scope control queues remain
unchanged. Broader balanced data and matched post-assignment runtime are needed
before a future causal comparison.

Evidence: `outputs/groundjev/training_data/assigned_identity_update_component_v1/{validated_component_summary,validation,files_manifest}.json`.

## Three-arm final-assignment memory diagnosis — 2026-10-16

Read-only full-video diagnosis now includes all three completed Full arms,
not just MOT500/native600. All13,664candidate events per arm reconcile with
saved final assignments and actual0.5-gated UPDATE/HOLD decisions; every
historical v1 statistic for the original two arms reproduces exactly.

| Completed arm | Q3 AUROC, known consistent vs conflicting assignment | Consistent writes / assignments | Conflicting writes / assignments |
| --- | ---: | ---: | ---: |
| MOT500 | .513176 | 192/214 (89.72%) | 594/696 (85.34%) |
| native additional600 | .485748 | 136/149 (91.28%) | 465/514 (90.47%) |
| teacher additional600 | .499661 | 56/82 (68.29%) | 152/216 (70.37%) |

Teacher's assigned-identity probability AUROC is.467395; its argmax-agreement
AUROC is.479336. Requiring final-ID=argmax on already-observed writes would
hold51/56known-consistent and127/152known-conflicting teacher writes, retaining
only5consistent/25conflicting writes. This is not an accepted policy or causal
rollout. These values describe each model's own endogenous known-anchor event
set, not a same-event training effect; unknown-birth and unmatched candidates
remain separate rather than being labeled negative. No universal memory-write
precision, threshold sweep, modified prediction or new HOTA is claimed.

All three trajectories support the diagnosis that the current pre-assignment,
candidate-observation Q3 does not discriminate correctness of the ultimately
assigned identity. Explicit final-ID-conditioned training/runtime supervision
remains a separate hypothesis; the ongoing paired history-refresh study keeps
its original questions, loss and runtime unchanged. Evidence:
`outputs/groundjev/diagnostics/full_mot_assignment_memory_v2_all_three/summary.json`.

## RMOT expression component — 2026-10-16

Prepared and validated all4,350legal-fit expression files on the same72
recorded native frames/165candidates, retaining all8rejected frames and34,800
query-frame contexts. The87,721questions comprise84,062candidate membership
and3,659empty-frame presence questions; candidate positives6,873 versus77,189
negatives. Only2,213expressions have a positive target in these sampled frames;
2,137do not. Per-expression positive/negative coverage is preserved for a
future balanced mixed-task curriculum. This expands language coverage, not
physical-image coverage or identity-history supervision. Proposals still come
from their original recorded primary-expression DLM requests; no detector was
rerun for the new expressions, so this is counterfactual membership supervision.

The initial generator stopped on missing same-frame boxes for expression IDs:
60contexts/15expressions/four unique video-frame-ID references, all in fit0017.
An early-preflight rerun correctly held before exporting any rows. Pinned
TempRMOT6a65640's unchanged `_pre_single_frame` was then executed on198actual
cases, including all60affected contexts: it iterates released boxes and marks
membership by expression ID; it does not invent boxes for unboxed IDs. Six
legacy empty-frame missing-label-file cases remain explicitly recorded. The
first official-reader verification failed because our symlink setup gave only
the image path the required KITTI token; fixing the label alias too selects
the official KITTI geometry branch. All failed attempts are preserved locally.

The final component explicitly records the official box-projection rule and
all60affected contexts; none were skipped, interpolated or silently relabeled.
All346legacy samples reproduce exact state/question/target/task, with136native
input-media pixel comparisons and200legacy query-media pixel pairs checked.
Full native CPU planning retains all87,721rows, max2,301tokens,120,061,085total
planned tokens,0overlength drops; actual `plan_epoch` took111.924s and peakCPU
RSS2,068,492,288bytes. The cached native reporter's0s and default4-rank label
are not full tokenization time or GPU use: actual world size was1onCPU; future
entry explicitly sets PLAN_WORLD to match. The21,931steps are a hypothetical
4-row accumulation epoch, not an optimizer run. Gzip training text3,156,280bytes;
all generated/failed-attempt real files total about7.98MB, within64MiB reserve.
Combined conservative temporary upper26,426,987,277bytes remains below25GiB.
No new image/feature archive or GPU training was created. Existing paired600
training and queued four-scope controls remain unchanged; ordinary-model-level
three-task performance and large-scale joint foundation training remain unproven.

Sources: `configs/groundjev/rmot_query_diversity_component_v1.json`,
`tools/groundjev/{prepare_rmot_query_diversity_component,check_rmot_release_box_projection,plan_rmot_query_diversity_component}.py`,
`outputs/groundjev/training_data/rmot_query_diversity_component_v3_official_box_projection/validated_component_summary.json`,
`outputs/groundjev/validation/rmot_query_diversity_full_cpu_plan_v1/owner_state.json`.

Independent worktree `/data2/user/LocateMOT_GroundJEV`; original LocateMOT data
and preserved U2 results remain under `/data2/user/LocateMOT`. The full goal
remains active. These are registered legal validation scopes, one seed; no
complete-dataset result or successful unified-model promotion is claimed.
Earlier snapshots/failures remain in Git, `STATUS_HISTORY_20261015.md` and
`research_log.md`. Official artifacts remain authoritative.

User clarification2026-10-15: the final purpose is ordinary-model-level
MOT/RMOT/OVMOT performance, not completion of the Pilot or experiment queue.
That target is NOT achieved. ByteTrack,TempRMOT-v2 andOVTrack are the initial
conventional references; see `ordinary_model_targets_v1.json` andFINAL_GOAL.
Published figures require matching scope/splits before formal use. In
particular, released TempRMOT training includes all three localV2validation
videos, so its released weights cannot establish unbiased held-out parity
there. The old TempRMOT V1 sanity report is neither a currentV2validation
result nor proof that the new performance goal has been reached.

Currentnative600 actual own-history refresh completed with its original
blocking owner exit0 after the successful Full1203cohort. All48fit frames/
1184original DLM candidates and perception records exactly match the original;
5741supervised rows,165806581exported bytes, no new DLM calls or GT-corrected
actor states. No automatic further training; export cap512MiB keeps the registered
combined upper26,262,799,802bytes below25GiB. Review actual existing-identity
and conflicting-anchor support before changing Q3 supervision. Plan:
`configs/groundjev/native600_fit_rollout_refresh_v1.json`.

Fit-only audit now reproduces all1184observation Q3 targets on refreshed
histories. Actual existing-ID assignments rise from0(original100actor) to658:
45known-consistent,568known-conflicting,15unknown-birth and30unmatched. Of
these, actual existing-memory UPDATE counts are12/41/12/20respectively.
There are82NEW initializations and444events without an assigned identity;
all counts conserve1184. The original observation-question target is yes for
300known-conflicting assigned events, which does not make that original
candidate-only question an assigned-ID correctness label. Unknown/unmatched
cases are not relabeled negative. Audit:
`diagnostics/full_fit_update_supervision_native600_v1/audit.json`.

Refreshed versus old actor history SFT is now running serially onGPU4,
retaining RMOT/OVMOT content and the existing runtime policy. Each arm adds
600updates/2400sampled rows from the same native600FP32 masters:320tensors/
752393024trainable parameters, first-arm initialization difference0. Each
arm uses800MOT/800RMOT/800OVMOT rows; only400native MOT histories change,
while400teacher MOT and all semantic rows remain equal. Actual row-to-step
membership and question families match; native length ordering within steps
can differ. CPU plans drop0rows, max25998/27960tokens; not token/FLOP matched.
New JSONL48712542bytes; combined registered upper25957225229<25GiB.
The first owner stopped before loading models because plan-only had created
args-only directories; these are preserved under the CPU validation, and
the corrected live owner is `training/full_history_refresh600_pair_v2`.
Original blocking session21963, PID2448456/start45150423; no subsequent
tracking gains are claimed. Explicit assigned-ID Q3 would be a
separate declared training/runtime protocol, rather than silently replacing
the current observation labels.

Preregistered post-training controls are queued through two CPU pidfd owners:
GPU4old worker waits for both training arms and native-MM reloads; GPU5worker
waits for the original natural-study queue to exit, fills the missing initial
native600 Full TAO40baseline, then waits for training and measures refreshed.
Each trained checkpoint covers DT1203, RMOTsingle800/multi800 and rawTAO40;
the initial checkpoint's already-completed exact MOT/RMOT units are reused.
All13664/1412/2208/35228native candidates and27/36/19/0rejected frames retain
their original denominators. TAO uses all1203inputclasses/UNKNOWN and the
annotation clock, not physical video seconds. No new DLM/ROI archive or
per-task checkpoint selection. Plan `full_history_refresh600_controls_v1.json`,
CPU source/real-baseline-validator evidence under
`validation/full_history_refresh600_controls_registration_v1`.
No post-training model calls or new accuracy results exist at registration.

## Complete DanceTrack video

Same original V1 Ground proposals, all1203frames of validation0004,
13664candidates/27parser-rejected frames; unchanged official TrackEval.
HOTA/IDF1 below are percentages, IDSW/FP/FN counts.

|Method|HOTA|IDF1|IDSW|FP|FN|
|---|---:|---:|---:|---:|---:|
|Simple|8.530|3.956|746|9334|256|
|Official ByteTrack association, explicit constant-score inputs|16.473|13.903|367|5234|395|
|Distilled Compact300|20.157|17.140|588|2129|951|
|Matched MLP300|11.197|15.118|161|549|3556|
|Same Set architecture, GT-only300|5.977|3.406|1322|5954|1796|
|New original U2 core/bridge300|1.243|0.625|250|455|4191|
|Shared Compact, broad native history|26.316|30.633|165|579|1962|
|Full MOT500, V3 native heads|7.848|5.056|3530|6387|765|
|Full MOT500 + own-history600, V3 native heads|8.703|5.826|3631|7516|416|
|Full MOT500 + teacher-history600, V3 native heads|7.577|5.015|3839|7683|384|

Shared Compact has1146511learnable resident parameters plus original frozen
OneJev CPU token lookup254279680parameters/508559360nominal bytes and external
Ground perception. It is not a replacement claim for the0.8B decoder. The
shared stage has a larger/different training budget than the matched300arms.
Lower IDSW alone does not establish improvement when FN rises.

ByteTrack uses untouched official tracker code
[`d1bf019`](https://github.com/FoundationVision/ByteTrack/tree/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99),
CPU-only with no weights/CUDA. All13664native boxes enter with explicit
constant1because DLM scores are absent; the low-score branch is empty.
Official CLI defaults .6track threshold/30buffer/.9match threshold are fixed,
giving20lost frames at20Hz (1s), versus GroundJEV3s. Native Kalman output
boxes and all returned active tracks remain, with no extra area/aspect filter.
All1203frames/27rejects and the exact unchanged input geometry are validated,
and MOT prediction text agrees with every saved frame output. This is a
same-proposal association control, not the full YOLOX+ByteTrack pipeline,
published-score reproduction, matched training budget or proof of unique JEV
benefit/parity. Compact has fewer FP but more FN; Fullnative600 is weaker
than this conventional association reference. Receipt:
`baselines/bytetrack_unscored_dancetrack1203_v1/validated_summary.json`.

Fresh pretrained Full V3 uses a different actual source: HOTA2.389 versus its
own matched Simple8.257. Actual unresumed native pipeline FPS.118560, under
concurrent workloads, excluding preloaded service/model initialization. Its
historical final-checkpoint timer definition is preserved. Full native600 and
MOT500 and teacher600 controls above each have all1203decoded frames, exact13664V1candidates,
all27parser-rejected frames, unchanged official metrics and final memory
validated. All three registered units are complete, without failed/pending
arms. Teacher600 lowers HOTA versus MOT500 despite fewer FN and has more
ID switches/FP; neither600branch establishes stable multi-task improvement.
These are replay measurements with no E2E FPS claim. Receipt:
`training/onejev_full1203_controls_v2_wait_fix/completed_units_validated_summary.json`.

Same-source posthoc diagnostics align4341/4586GT observations (94.658%) to
candidate boxes atIoU.5. MOT500/native600 correct-option coverage is98.603%/
95.021% of their respective4008/2912existing-anchor events, but actual correct
assignments are214/149 (5.339%/5.117%). Per-candidate argmax is correct254/178
times; global matching loses222/145 and rescues182/116, giving the same214/149
actual assignments. Low native identity ranking and global competition both
contribute; different endogenous event sets prevent a strict causal comparison.

Completed teacher600 has1588correct options among1833existing-anchor events
(86.634% coverage), but82correct final assignments (4.474%). Raw argmax is
correct77times; matching loses69and rescues74, yielding82. The same candidate
GT recall is94.658%; proposal misses alone do not explain this weak ranking.

Existing-memory UPDATE breakdown, excluding NEW initialization:

|Birth-anchor diagnostic|MOT500|Own-history600|Teacher-history600|
|---|---:|---:|---:|
|Matched GT, known consistent anchor|192|136|56|
|Matched GT, known conflicting anchor|594|465|152|
|Matched GT, unknown birth anchor|2545|2984|2962|
|Unmatched candidate, not universal negative truth|5812|6648|4977|

Known conflicting writes show real memory contamination in each run. Unknown
birth anchors and unmatched/ignore-region candidates are kept separate; they
are not relabeled wrong identities or supervised negatives. Counts preserve
the original analyzer, metrics and uncorrected causal tracking states.

Saved-decision diagnosis of the memory gate: native Q3 is asked per candidate
before global matching and does not name the final assigned identity. On
known-consistent/conflicting existing-ID assignments, Q3 AUROC is0.513(MOT500)
and0.486(native600). Native600 mean Q3 is0.852/0.849 for those two classes;
mean probability of the actually assigned identity is0.04269/0.04234. The
observation gate does not distinguish those identity outcomes here.

Requiring assigned-ID/argmax agreement on these frozen histories would hold
173/192correct and518/594conflicting MOT500writes, and111/136correct and399/465
conflicting native600writes. It discards most correct writes too, so this
heuristic is not adopted. These counts do not simulate future causal states
or establish HOTA gains. Receipt:
`diagnostics/full_mot_assignment_memory_v1/summary.json`.

Fit-only audit reproduces all1184original native Q3 labels on48legal training
frames and all2460eligible family-matched copies. The actual old100-step
native actor produced74NEW and1110events with no assigned identity, with zero
existing-ID assignments. Thus these native actor states contain no actual
conflicting existing-ID write contexts. Association/lifecycle supervision
targets still exist; this is not absence of association labels throughout the
training corpus. Q3 labels express visibility/correct-option availability,
not final assigned-ID correctness. These observations motivate refreshing
own-history support and an explicit assignment-conditioned gate; no existing
label, checkpoint or policy is silently changed, and no causal training-failure
claim follows from this eligible-corpus audit alone. Receipt:
`diagnostics/full_fit_update_supervision_v1/audit.json`.

## All12complete800frame RMOT controls

Both expressions cover all frames0..799 of legal Refer-KITTI-v2 validation0007.
Single has18target boxes/782empty-target frames; multiple2112/247. Same actual
native sources1412/2208candidates,36/19parser rejections and every empty/rejected
frame retained. Official parsing unchanged; strict-wire diagnostic failures are
not relabeled as valid or omitted. All12actual official/runtime/video receipts
validate; the original blocking owner exited0 and released its own services.

|Method|Single HOTA %|Single IDF1 %|Single FN|Multi HOTA %|Multi IDF1 %|Multi IDSW|Multi FP|Multi FN|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Simple|4.966|1.259|8|31.877|37.685|99|1076|980|
|Shared Compact|6.025|2.890|13|19.802|21.008|165|619|1054|
|Pretrained Full|16.119|17.391|16|8.621|7.635|215|214|1766|
|MOT500 Full|0|0|18|0|0|0|11|2112|
|Teacher-history additional600 Full|6.847|4.839|14|4.578|4.228|108|104|1963|
|Own-history additional600 Full|2.506|2.062|17|6.740|6.693|185|145|1859|

Both additional600stages start from the actual completed MOT500 weights and
have600further updates/2400examples each. They partly recover MOT500's RMOT
collapse but do not beat pretrainedFull here. PretrainedSingle's higher HOTA
still accompanies16/18misses. No training/unified-model success is claimed.

Compact fresh runs used live pristine DLM and real native2560ROI, association,
causal memory, archive/prediction writes, rendering/encoding and final export:
actual pipeline FPS.678005/.882942, wall1179.932/906.062s. Core initialization
5.232/5.362s and remote preloading excluded. Simple/Full in this table replay
these exact sources, so their replay FPS is not comparable to native Compact
FPS. Archive total197218901bytes<2GiB. Core Torch allocator peaks14970368/
14855168bytes and core RSS1467027456/1478381568bytes exclude external services.

|Empty-target frames with any output|Single|Multiple|
|---|---:|---:|
|Compact|200/782=25.575%|143/247=57.895%|
|Simple|747/782=95.524%|243/247=98.381%|

Compact reduces false alarms but loses multiple-target identity accuracy.
False-alarm frame counts and detection counts are separate. Live-video paths:
`multitask_runs/full_rmot_shared600_v1/{single,multi}800_shared_compact_fresh/tracks.mp4`.
Receipt: `multitask_runs/full_rmot_shared600_v1/all12_validated_summary.json`.

Paired fresh Fullnative600/Compact800same-video/same-GPU5 E2E has completed.
All800frame/image/box/label/native-score/score-kind/parser-rejection records are
exactly equal:1412candidates and36rejected frames. Official accuracy exactly
matches the prior same-candidate native600/Compact controls. Both decoded800.

|Fresh same-video method|Actual pipeline FPS|Wall seconds|Sampled owned VRAM GiB|HOTA %|FN /18|
|---|---:|---:|---:|---:|---:|
|Full native600|0.495914|1613.184|36.184|2.506|17|
|Shared Compact|0.687099|1164.316|19.166|6.025|13|

Observed pipeline FPS ratio1.3855x, wall reduction27.825%. Sampled total owned
GPU peaks37052/19626MiB (47.031%lower), five-second samples are lower bounds
on instantaneous peak, not exact maxima. Full core Torch allocator8426480128
bytes/RSS3305140224bytes; Compact14970368/1466036224bytes exclude external
services. Compact also has original frozen CPU text lookup; the1.146M core
alone does not describe whole-pipeline model/memory cost.

Actual pipeline includes imageIO/nativeDLM/ROIwhenrequired/association/state/
archive/prediction/render/video-encoder completion/final export. Private
service startup26.052/26.046s and core initialization8.010/5.004s are excluded
and separately recorded. Full uses real images/crops; Compact additionally
pays nativeROI cost. Per-frame inference p50/p95seconds Full1.810/2.930,
Compact1.202/2.136 exclude trailing output/state/video writes that are included
in whole-pipeline wall time. Both remain below1FPS with poor18-box recall.

Empty-target alarm frames Full67/782 versusCompact200/782; fewer misses/higher
HOTA here still trade against more false alarms. One single-expression pair,
fixed Full-then-Compact order, one seed, concurrent other CPU/project loads:
no counterbalanced/statistical speed or real-time claim. Historical other-video
or replay FPS is unchanged and not used for this ratio. Evidence:
`e2e/full_compact_rmot800_v1/validated_summary.json` and both run artifacts.
All prior owned groups exited before GPU5followup allocation; localization-only
training and its96native-frame/18Full controls have completed on GPU5. The
serial queue has advanced to fresh naturalMOT20; see the completed comparison
below. Original start/device evidence is preserved.

## Same-source saved-output diagnosis

All2208candidate outputIDs/boxes reconstruct exactly from saved actions and
original membership>=.5, with no new model calls, corrected states, thresholds
or predictions. Same1132one-to-one aligned query-positive candidates:

|Method|No physical assignment|Assigned then membership rejects|Actually emitted aligned positives|
|---|---:|---:|---:|
|PretrainedFull|105|687|340|
|MOT500Full|263|869|0|
|Teacher600Full|544|445|143|
|Native600Full|330|555|247|
|SharedCompact|4|71|1057|

These disjoint candidate funnels are not an officialFN decomposition across
HOTA thresholds. Per-target proposal coverage1133/2112uses a different matching
rule. Unknown/unmatched candidates are not fabricated negativeGT. Full query
rejection and association decisions both need work; the membership gate alone
is not the whole teacher/native600failure. Different causal birth-anchor event
sets prevent strict same-event identity-accuracy comparisons.
Receipt: `diagnostics/multi800_full_compact_output_funnel_v1/summary.json`.

## Completed matched input-evidence study14.2

All12full-video controls complete: three modes on fullDT1203/rawTAO40/single800/
multi800. Two new1200step models and reused completed all_three1200 full-input
weights share actual initialization, all1200sample IDs, seed0,536source groups,
GT/native teacher targets and1146511resident parameters. Three-task reload
delta0. Full-input official scores exactly reproduce all four prior baseline
scopes despite changed state-storage implementation.

|Runtime evidence|DT HOTA %|RawTAO TETA %|Single800 HOTA %|Multi800 HOTA %|
|---|---:|---:|---:|---:|
|Geometry/motion, ROI/specification branches masked|9.144|13.568|0|14.462|
|Native appearance, specification condition masked|26.061|11.444|2.491|12.900|
|Native appearance plus specification condition|19.312|10.081|7.860|18.674|

Appearance helps this MOT scope; specification conditioning helps both RMOT
expressions but hurts MOT/TAO here. All NovelClsA0, and every multi800arm is
below same-candidate Simple31.877HOTA. Keep negative transfer; no universal
multimodal/JEV superiority claim or validation-threshold tuning.

Geometry-motion disables382208visual/history-projection parameters, so equal
resident count does not mean equal active capacity. All modes retain the native
semantic frontend, softteacher knowledge and output-only class-vocabulary text
prototypes. These are association-input ablations on common older-actor training
histories, not language-free perception or unique JEV proof. Full native-image
JEV differs in architecture/training budget and is separately reported.
Receipt: `compact/input_evidence1200_v1/controls/validated_summary.json`.

## Complete TAO annotation-frame scope and policy studies

One legal Base/Novel video:40annotation frames/160GTboxes, all1203LVIS input
categories. Raw source35228proposals, separate exact geometry-alias protocol
7343groups retaining every alias. Physical source FPS unknown; annotation-step
timing is not physical seconds. Official unchanged TETA, all NovelClsA0.

|Proposal protocol/method|TETA %|AssocA %|ClsA %|
|---|---:|---:|---:|
|Raw/Simple|18.640|28.618|20.290|
|Raw/shared Compact|14.517|5.148|30.314|
|Aliases/Simple|22.556|26.472|18.325|
|Aliases/shared Compact|21.773|9.627|32.419|

Raw versus alias changes proposal representation/representative labels, not
association alone. Coverage147/160atIoU.5. Frozen-trajectory correct-option
coverage32/64/128 is15.672/31.343/44.776% raw and53.097/60.177/74.336% alias;
these are ranking diagnostics, not counterfactual tracking scores.

Original five-policy cohort25units and both full800RMOT five-policy cohorts
10units complete. Same-input DT HOTA learned/IoU/cosine/cosine+learnedaux/
learned+fixedaux=26.316/7.488/13.311/13.009/24.283. But fullmulti800 gives
19.802/32.813/16.622/22.035/19.558. Current learned association loses to fixed
IoU plus identical semantic functions here. Histories/confidence/action priors
and filtering evolve by policy; not optimal retrained baselines or JEV proof.

Actual context study12units, four32references/eight new64/128trajectories:

|Context IDs|DT learned HOTA %|DT fixed cosine HOTA %|AliasTAO learned TETA %|AliasTAO fixed cosine TETA %|
|---|---:|---:|---:|---:|
|32|26.316|13.311|21.773|23.913|
|64|26.316|13.289|23.100|26.099|
|128|26.316|12.491|24.749|29.062|

Larger context helps this TAO scope but exceeds training32options and changes
semantic normalization/histories; fixed cosine still leads. No isolated JEV gain.

Hybrid32retrieval12units, six new/six geometry references, all complete:

|Scope|Learned geometry|Learned16geometry+16appearance|Fixed cosine geometry|Fixed cosine hybrid|
|---|---:|---:|---:|---:|
|DT HOTA %|26.316|26.316|13.311|12.811|
|AliasTAO TETA %|21.773|21.466|23.913|24.435|
|RawTAO TETA %|14.517|15.178|13.964|15.652|

Mixed gains; geometry remains the default. All receipts under respective
`compact/{fixed_affinity_controls_v1,context_sizes_v1,hybrid_shortlist32_v1}/validated_summary.json`.

## Completed task-subset and memory studies

Four new task subsets share MOT300+seed0semantic initialization,1200totalupdates,
536groups and1146511resident core parameters. Per-task exposures differ:
MOT1200; pair600each; triple400each. All24controls complete:

|Tasks|DT HOTA %|RawTAO TETA %|Single800 HOTA %|Multi800 HOTA %|
|---|---:|---:|---:|---:|
|MOT|19.639|12.421|0|7.673|
|MOT+RMO|6.760|19.284|0|18.255|
|MOT+OV|21.229|12.850|0|15.794|
|All three|19.312|10.081|7.860|18.674|

All NovelClsA0. No universal transfer; older shared26.316DT/19.802multi model
has different initialization and joint1200+native1200 budget. MOT400/600extra
controls HOTA13.317/13.979 have exact actual sample/loss prefixes versus1200,
but different total compute prevents pure gradient-interference attribution.
Receipt: `compact/task_subsets1200_cpu_v2_feature_sources/validated_summary.json`.

Complete DT1203 memory interventions HOTA11.441(frame-only),16.394(recent),
14.643(EMA),24.245(anchor/always),26.316(anchor/gated3s),18.552(gated30s).
These are frozen policy interventions; no successful >3sDTrecovery evidence.
SingleRMOT800has two natural >3s candidate returns, recovered0 by both Compact/
Simple under immutable correct-birth-anchor diagnostics.

Natural MOT20-05validation1657frames(1659..3315,25Hz) is now queued on GPU5
after pairedE2E and localization200 complete. OriginalGPU7retention continues
independently; this execution-device/dependency amendment occurred before any
natural inference in the superseded waiting owner. Eight same-source CPU memory/Simple
controls, packed FP32 archive3GiBcap. Registered ID854's29.76s low-visibility
interval retains743intermediate annotated frames; it is not a proven missing-
candidate interval. No synthetic occlusion/GT correction/visibility filtering.
Real21frame exact archive regression preserves266candidates/rejected empty
frame/full state/next-frame restore:14998172JSON-equivalent→3409159packed bytes.
No new accuracy/speed claim from this storage test.

## Training, adaptation and remaining work

Broader legal TRAIN-source availability is now indexed without new model
calls/features: MOT40904frames/43videos (2664MOT17,33772Dance,4468MOT20),
RMOT1814local frames/9fit videos/4350expression files, andTAO18274annotation
images/500train videos. All indexed image paths exist. TAO provides52978
eligible Base annotation references across175frozen-vocabulary classes,
versus only9classes/64images in the existing category SFT. All175class
synsets/frequencies and52978image/annotation links validate against actual
train-only sources;6172nonBase and50outside-vocabulary annotations are not
supervised or converted into negatives. All1251images with no eligible Base
label remain indexed. Full1203input vocabulary stays unchanged.

The index uses33.85MB of metadata under a128MiB export reserve; conservative
registered upper26091442957bytes<25GiB. It is source availability, not rendered
JEV SFT, native proposals, actual training exposure, new large-scale training,
or joint Ground+One gradient optimization. RMOT local-frame listing is not an
external sequence-length completeness proof. Current600paired runs continue
unchanged. Index/coverage/validation live under
`training_data/broader_legal_scale_index_v1`; generator and source plan are
`build_training_scale_index.py` and `training_scale_index_v1.json`.

The175Base category component is now prepared and input-validated:52978train
objects produce105956native root/conditional-leaf questions across500videos.
Expanded JSONL1005240705bytes is stored as25820145gzip bytes; real-image crop
sheets are generated only when requested through the unchanged native
`NativeCategoryHead.crop_sheet`, with no PNG/ROI dataset cache. Typed questions,
1203input categories and conditional-root teacher forcing match the earlier
category component. No Novel/unknown negative labels are added; all18274images
are registered, including1251without eligible Base labels.

Actual regressions reproduce all215earlier Base object pixels/62sheets, native
question/pixel evidence on175class representatives and exact processor tensors
for root/leaf at both768x216/768x432sizes. All105956annotation/image/class/box
targets validate. Official CPU planning drops0rows, max4238tokens and total
315781843planned tokens; these are not training exposure. The reported26489
steps are hypothetical at4rows/update for this category-only component, not
an optimizer launch. Text rows still load into RAM (actual CPU peak3746652160
bytes); media generation is on demand. The native cached-plan reporter's1s
message is not complete tokenization timing or throughput. Existing normal
rows/images and native optimizer/rendering math remain unchanged; the separate
streamed-input entry records its IO version in checkpoint resume protocols.
Use this component within a future balanced MOT/RMOT/OVMOT curriculum, not an
automatic category-only training stage. Original600paired training continues.
Receipt: `training_data/base175_streamed_component_v1/validated_component_summary.json`.

Original One100real uncorrected rollout→500SFT and Compactjoint1200real72frame
rollout→native1200 provide offline iterative own-state training. Not per-gradient
on-policy resampling; the latter is a research direction, not an invented gate.
Both additionalFull600FP32/Adam/scheduler/RNG states saved,2400examples each,
nativeMM reload delta0. Teacher resumed450, native250 exactly; teacher replayed
475loss differs.001734814 and native has no saved-log overlap. Neither called
bitwise training replay; resumed-tail timings are not whole-stage throughput.
All12same-policy shortFull controls completed, but complete-video results above
show the short multiRMO gains do not establish full-video improvement.

Fit72update audit reproduces original labels:28positive,12unknown birth owners,
0two-known-identity-conflicting positives. One known conflicting write already
has target0. Initial unknown→negative diagnostic suggestion was invalid and
never applied. No pack/model changed; do not claim this label hypothesis proved.
Authoritative audit: `diagnostics/causal_update_supervision_fit72_v3_known_conflicts/audit.json`.

Ground200trained native27267584parameter projector, native dense/referring plus
identity losses, exported569tensors with remaining weights untouched. Ground320
adds actual absence supervision. Isolated original/200/200repeat/320controls
preserve request order/repeat raw outputs exactly, show mixed MOT/RMO behavior;
no promotion. Original/Ground200nativefull9retention each targets91124released
rows, not all30paper benchmarks; neither complete while native clients remain.
GPU6native CUDA failure/retry evidence retained; first failing kernel unknown.

Available completed full-task artifacts are now frozen independently of the
still-running9task owners: original6/9tasks (31,986native rows), adapted2002/9
(1,810rows). Every completed task has original=effective sample count and
provider-policy-filtered0. These are complete individual tasks, not completion
of either full9run or a result extrapolated from active response prefixes.

|Common completed grounding task|Rows|Original F1@.5 %|Adapted200 F1@.5 %|Delta pp|Original F1mIoU %|Adapted200 F1mIoU %|
|---|---:|---:|---:|---:|---:|---:|
|Dense200|200|78.986|78.679|-0.307|66.644|66.054|
|VisDrone|1610|46.125|42.145|-3.980|29.662|27.870|

Dense200 recall76.086→74.350% and precision82.116→83.544%; VisDrone recall
43.534→39.578%, precision49.044→45.068%. These measured grounding-retention
results do not establish tracking gains or overall retention. Originalqueue32
versus adapted-sync execution and recorded CUDA recovery prevent a bitwise/
paired-request or exclusively weight-causal interpretation. OriginalCOCO is
reused through its existing registered resume receipt. All original native
files, owner/service processes and metrics remain unchanged. Receipt:
`ground_retention/partial_completed_tasks_20261015_v1/summary.json` and the
eight exact native metric/sample-count extracts under that directory.

Localization-only200 and all96native-frame/18Full controls are complete and
official-artifact validated. Same112rows/56pairs/200actual draws/seed42/lr2e-6
as completedGround200, both identity coefficients0. All200logged total losses
equal native loss;27,267,584trainable parameters,16,164,180BF16elements changed,
checkpoint reload delta0. Export569native tensors; all other weights unchanged.
Localization repeat reproduces all raw-wire candidates/outputs on4/8/12frames.

|Native frontend|MOT4 HOTA %|Single8 HOTA %|Multi12 HOTA %|
|---|---:|---:|---:|
|Original|75.069|30.385|22.664|
|Localization200|84.099|18.776|23.355|
|Tracking200|83.467|24.876|23.675|

Full policy HOTA %, each frontend's exact candidates reused in bothOne arms:

|One weights + frontend|MOT4|Single8|Multi12|
|---|---:|---:|---:|
|Pretrained + original|33.913|0|17.636|
|Pretrained + localization200|36.858|0|1.754|
|Pretrained + tracking200|35.199|0|6.697|
|Native600 + original|31.932|0|7.605|
|Native600 + localization200|34.709|0|7.313|
|Native600 + tracking200|34.934|0|5.504|

ShortMOT improves after either visual adaptation; single-expression native
results worsen and Full misses all5single-scope GT boxes. Composition has no
general RMOT gain. These4/8/12frame controls do not establish complete-video
or dataset quality, full9retention, Novel capability or model promotion.
Separately adapted weights are composed, not jointly optimized. Frontend
adaptation changes proposals; exact input pairing holds within each twoOne
comparison. All new frontend/factorial runs useGPU5; reference tracking200
training usedGPU6 (same A800), so no bitwise cross-device trajectory claim.
Receipt: `ground_adaptation/localization200_control_gpu5_v1/validated_summary.json`.

|Queue/device|Remaining actual work|
|---|---|
|GPU4|Paired old/refreshed history sharedFull600SFT running; queuedold four-scope worker follows successful training/MMreload|
|GPU5|Fresh naturalMOT20 then8CPUcontrols; queuedbeforeTAO/refreshed four-scope worker follows original queue release|
|GPU6|AdaptedGround200nativefull9 remains unchanged|
|GPU7|OriginalGroundnativefull9 remains unchanged|
|CPU|Policy25, tasksubset24, context12, hybrid12, input12 all complete|

At most four physical GPUs4/5/6/7combined; foreign jobs and stopped U2 queues
untouched. No new GPU/feature cache for completed CPU studies. Conservative
prior feature upper22471148986+3GiB+32MiBregression reserve=25725928890bytes
<25GiB(26843545600). Data/checkpoints/results retained. Historical old-cache
cleanup freed39.89GiB; resource-snapshot limits in `RESOURCE_CLEANUP.md`.

Compact batches FP64 geometry and caches same-timestamp predicted boxes in
new processes. Real591×2365=1397715overlaps/37groups32/64/128ordering and full
model outputs/lifecycle/metadata match scalar exactly. Three warm16candidate
post-state probe medians147.08→51.11ms geometry and784.82→675.81ms hybrid;
not causal-video or E2E FPS. Historical timings unchanged, actual childGit
revisions recorded. Receipt: `validation/dense_geometry_batching_v1/validation.json`.


Execution scheduling amendment: both old localization/natural owners were
verified still CPU-only waiters, no children/completed experimental units, and
superseded with preserved manager-handoff receipts. The old localization shell
returned241 from intentionalSIGTERM; this is a canceled wait, not a completed
measurement or hidden experimental failure. New owner2207841/start43963616
(blocking session36847) waits original paired owner1953808/start42350971 before
any GPU5 allocation. It verifies successful paired completion and all owned
process groups gone, then runs both existing studies serially. No active
training/inference/retention jobs stopped. Real112row/200draw/native-proposal
alignment matches, all other scientific plan fields and96/18/1657/8scopes
unchanged; GPU5/6/7 allNVIDIA A800-SXM4-80GB. Cache bound unchanged.
Receipts: `validation/localization200_contract_gpu5_v1/validation.json` and
`validation/gpu5_followup_registration_v1/validation.json`.
New plans: `configs/groundjev/gpu5_followups_v1.json` and referenced GPU5variants.


The GPU5handoff executed after paired blocking session87824 exited0 and all
five owned model process groups were gone. Serial owner2207841 completed
localization child2213633 and advanced to naturalMOT20. Original retention6/7
and Full1203owner remain independent.

Progress visibility: `tools/groundjev/stream_owner_events.py` uses blocking
Linux inotify/pidfd events for the three existing owners and their registered
child state files. It prints actual state changes and process exits without
periodic filesystem/process polling or model calls. Original blocking commands
retain authoritative exit codes; an exit event alone is not scientific success.
Registration snapshot: `progress_events/remaining_v1/registration.json`.

Pilot documentation is now reconciled with the actual12execution deliverables
in usertask17; see `FOUR_GPU_PILOT.md` and
`pilot_milestone_20261015/evidence_audit.json`. This is a limited-development-scope
execution milestone, not completion of the full research goal or model promotion.
The durable original cleanup snapshots directly show14042MiB net all-device
memory decrease; concurrent other jobs prevent exclusive U2 attribution. The
historically reported17584MiB owned-worker inventory is kept as historical,
not reconstructed from those per-device tables. No old model is restarted.

A read-only original-owner checkpoint observer is registered at
`outputs/groundjev/training/full_joint_source_actual1000_observer_v1/`;
1000-step full FP32/Adam/all-rank-RNG verification remains pending until its
actual save is available. It neither restarts training nor allocates a GPU.
