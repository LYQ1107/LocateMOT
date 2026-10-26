"""Validate matched completed arms, publish compact evidence and research decision."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json,digest
from tools.groundjev.dense_head.evaluate_dense_head import load_rows,evaluate_predictions


def main():
    docs=ROOT/'docs/GROUNDJEV_DENSE_HEAD'
    receipts={arm:json.loads((RUN/arm/'receipt.json').read_text()) for arm in ['g0','g2']}
    for arm,r in receipts.items():
        assert r['status']=='REAL_GPU_PILOT_AND_FULL_DEV_COMPLETE'
        assert digest(r['checkpoint_last'])==r['checkpoint_last_SHA256']
        assert digest(r['checkpoint_best'])==r['checkpoint_best_SHA256']
    for key in ['optimizer_steps','initial_head_SHA256','sampling_order_SHA256','unique_optimizer_images','sample_exposures']:
        assert receipts['g0'][key]==receipts['g2'][key],key
    rows=load_rows(RUN/'data/external_frozen.jsonl')
    a0=json.loads((RUN/'data/A0_external_predictions.json').read_text())
    assert [p['image_path'] for p in a0]==[r['image_path'] for r in rows]
    results={'A0':{'metrics':evaluate_predictions(rows,a0,.5,compute_map=False,per_video=True),
                   'AP50':None,'AP75':None,'mAP':None,'AP_reason':'Native per-box scores absent',
                   'native_seconds':sum(p['native_seconds'] for p in a0),'rejected_frames':sum(p['rejected'] for p in a0)}}
    native_timing={}
    for dataset in sorted({r['dataset'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['dataset']==dataset]
        seconds=sum(a0[i]['native_seconds'] for i in indices)
        native_timing[dataset]={'frames':len(indices),'historical_request_seconds':seconds,
            'historical_request_FPS':len(indices)/seconds,
            'rejected_frames':sum(a0[i]['rejected'] for i in indices)}
    results['A0']['native_timing_by_dataset']=native_timing
    for arm,label in [('g0','A1'),('g2','A2')]:
        r=json.loads((RUN/arm/'external_results.json').read_text())
        predictions=json.loads((RUN/arm/'external_predictions.json').read_text())
        assert [p['image_path'] for p in predictions]==[row['image_path'] for row in rows]
        results[label]=r
    total_cost=sum(r['GPU_hours'] for r in receipts.values())
    summary={'status':'BOUNDED_MATCHED_PILOT_COMPLETE','receipts':receipts,'external_comparison':results,
             'full_DEV':{arm:json.loads((RUN/arm/'dev_full_results.json').read_text()) for arm in receipts},
             'GPU_hours_train_and_eval_residency':total_cost,'smoke_audit_GPU_hours':'separate short functional audits; <=1 reserved GPU-hour',
             'paired_initialization_and_sampler_verified':True,'old_evaluation_remains_paused':True,
             'head_DEV_was_seen_by_G2_TRAIN':True,'external_head_test_time_tuning':False,
             'external_scope_prior_use':'Prior native G2/tracking evaluation, not untouched final TEST',
             'checkpoint_weights_uploaded':False,'all_model_metrics_fractions_not_percent':True}
    write_json(docs/'PILOT_RESULTS.json',summary)
    for arm in receipts:
        shutil.copy2(RUN/arm/'receipt.json',docs/(arm.upper()+'_PILOT_RECEIPT.json'))
        shutil.copy2(RUN/arm/'frozen_selection.json',docs/(arm.upper()+'_DEV_SELECTION.json'))
    def table(metrics):
        lines=['|Model / dataset|Frames|R50|R75|Precision|FP/frame|FN|Duplicates|AP50 / AP75 / mAP|',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm,value in metrics.items():
            for dataset,m in value.items():
                if dataset=='per_video':continue
                ap='undefined (unscored native)' if 'AP50' not in m else ' / '.join(f'{100*m[k]:.3f}' for k in ['AP50','AP75','mAP'])
                lines.append(f"|{arm} / {dataset}|{m['frames']}|{100*m['Recall50']:.3f}|{100*m['Recall75']:.3f}|{100*m['Precision']:.3f}|{m['FP_per_frame']:.3f}|{m['FN']}|{m['duplicates']}|{ap}|")
        return '\n'.join(lines)
    external_table=table({arm:v['metrics'] for arm,v in results.items()})
    dev_table=table({arm:v['metrics'] for arm,v in summary['full_DEV'].items()})
    lines=['# Real GPU pilot training','', '**Completed the bounded detector pilot. This is not completion of the unified temporal identity research goal.**', '',
           'Architecture: frozen native MoonViT+checkpoint-specific projector (428,482,048 parameters), 256-d projection and position,100 learned object queries,3 Transformer decoder layers,3,975,829 trainable head parameters; focal objectness+5L1+2GIoU with Hungarian assignment and auxiliary layer supervision. No GT proposals, no language answer input, no original detector boxes as query seeds.', '',
           'Matched optimizer: AdamW LR2e-4, weight decay1e-4,50-step warmup,0.1 LR multiplier from800;4 images/optimizer step,1,000 steps each, seed42. Alternate Sports/Animal, uniformly sample physical videos. Both initialization SHA and complete sampling-order SHA are identical. Each arm sees2,000 Sports and2,000 Animal exposures; exact unique count in receipts. The feature cache is at most6GiB per arm,12GiB combined. Native image processor max1024 merged tokens; encoder and projector never updated.', '',
           'GPU4: G2, UUID GPU-44f62835-4095-23ec-d3dd-5af16a00705c. GPU6: G0, UUID GPU-883c8be3-25b0-3d19-6fa8-4c2e335e67d2. Old evaluation and all recovery launchers were stopped before these runs. Source commit and final-G2 SHA are in the registration/smoke receipts.', '',
           'Checkpoint selection: macro domain AP50 on480 TRAIN-internal DEV frames every250 steps; threshold from a fixed0.05/0.1/0.2/0.3/0.5/0.7 grid maximizing macro domain F1. Both arms have identical choices and budgets. Complete8,370-frame/20-video internal DEV is then evaluated with the selected checkpoint. These DEV videos were seen by G2 fine-tuning; never call them independent of G2. A separately frozen1,607-frame/6-video comparison uses prior native G2 caches and no test-time tuning.', '',
           '## Complete internal DEV (percent except counts)', '',dev_table,'',
           '## Frozen video-disjoint cached comparison (percent except counts)', '',external_table,'',
           'R50/R75 above are target coverage for compatibility with original G2 recall. One-to-one recall, FP, mean best IoU, size/density/category groups and complete per-video metrics are in PILOT_RESULTS.json. Precision/FN/FP use one-to-one matching; duplicate detections cannot manufacture matched recall. AP uses genuine dense-head objectness and class-agnostic101-point matching, not official COCO. A0 AP is undefined and not compared.', '',
           '|Arm|Best step|Threshold|Peak allocated MiB|Peak reserved MiB|GPU-hours (train+eval residency)|Vision FPS / Decoder FPS / vision+decoder FPS|End-to-end wall FPS (fresh full DEV)|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in receipts.items():
        timing=summary['full_DEV'][arm]['timing']
        lines.append(f"|{arm}|{r['selected_step']}|{r['selected_threshold']}|{r['peak_allocated_GPU_MiB']:.1f}|{r['peak_reserved_GPU_MiB']:.1f}|{r['GPU_hours']:.4f}|{timing['vision_FPS']:.3f} / {timing['decoder_FPS']:.3f} / {timing['vision_plus_decoder_FPS']:.3f}|{timing['evaluation_wall_FPS']:.3f}|")
    lines+=['',f'Original global-pair measured train+evaluation residency subtotal: {total_cost:.4f} GPU-hours, plus separately bounded short smoke/feature audits. Large weights and full predictions remain under `{RUN}`. Git includes only summaries/indices/configuration. Native A0 time is historical shared-load inference; dense timing is current shared-load inference. Equal images and A800 hardware class do not imply matched concurrent load; treat speed as observed end-to-end evidence, not an isolated kernel benchmark.', '',
            'Loss curves and sampled-DEV curves are exported as PILOT_CURVES.png. Loss decrease alone is not a scientific gain. No long G2 unfreezing, OneJev SFT, identity decoder training, RL or new large data downloads occurred.', '',
            'Reproduce: `python tools/groundjev/dense_head/prepare_train_data.py`; `python tools/groundjev/dense_head/prepare_external_comparison.py`; run the two GPU leakage audits, compare spatial features, then `python tools/groundjev/dense_head/run_matched_pilot.py` in GroundAnything environment configuration. Local assets use existing server read-only links; external repository/environment pins are in the inherited dependency tools and environment. Single-arm resume supports optimizer/scheduler/sampler/RNG restoration via `train_dense_head.py --config ... --resume ...`; do not run two owners for an arm. The completed1,000-step runs must be archived before a new trial directory is used.', '',
            'Known limits: benchmark target-set foreground is class-agnostic, not exhaustive annotation of every semantic object. TAO/Refer partial labels are not in optimization. Refer audit finds157 absent referenced IDs; future supervision must exclude/repair those documented rows. Optional SAM checkpoint/runtime was not registered, so no SAM performance claim. No JEV/HOTA gain is inferred from this detection experiment.']
    (docs/'PILOT_TRAINING.md').write_text('\n'.join(lines)+'\n')
    write_json(RUN/'final_comparison.json',summary)
    print(json.dumps({'status':'GLOBAL_PAIR_SUMMARIZED','GPU_hours':total_cost,'docs':str(docs)},indent=2))


if __name__=='__main__':main()


def append_spatial_prior_controls():
    """Retain the original weak pilot and compare the pre-frozen spatial prior."""
    docs=ROOT/'docs/GROUNDJEV_DENSE_HEAD';path=docs/'PILOT_RESULTS.json'
    summary=json.loads(path.read_text());v2=RUN/'spatial_prior_v2';controls={}
    for arm in ['g0','g2']:
        receipt=json.loads((v2/arm/'receipt.json').read_text());assert receipt['status']=='REAL_GPU_PILOT_AND_FULL_DEV_COMPLETE'
        assert digest(receipt['checkpoint_best'])==receipt['checkpoint_best_SHA256']
        assert digest(receipt['checkpoint_last'])==receipt['checkpoint_last_SHA256']
        base=summary['receipts'][arm]
        for key in ['optimizer_steps','initial_head_SHA256','sampling_order_SHA256','unique_optimizer_images','sample_exposures']:
            assert receipt[key]==base[key],(arm,key)
        controls[arm]={'receipt':receipt,'registration':json.loads((v2/arm/'registration.json').read_text()),
                      'full_DEV':json.loads((v2/arm/'dev_full_results.json').read_text()),
                      'external':json.loads((v2/arm/'external_results.json').read_text())}
        receipt['execution_source_at_launch']=controls[arm]['registration']['source_commit']
        receipt['completion_source_field_note']='The entrypoint read current HEAD again at completion; the preserved launch registration identifies execution source.'
        shutil.copy2(v2/arm/'receipt.json',docs/(arm.upper()+'_SPATIAL_PRIOR_PILOT_RECEIPT.json'))
        shutil.copy2(v2/arm/'frozen_selection.json',docs/(arm.upper()+'_SPATIAL_PRIOR_DEV_SELECTION.json'))
    for arm in ['g0','g2']:
        summary['receipts'][arm]['execution_source_at_launch']=json.loads((RUN/arm/'registration.json').read_text())['source_commit']
        summary['receipts'][arm]['completion_source_field_note']='Older entrypoint read current HEAD again at completion; execution source is the separately preserved launch registration.'
    summary['spatial_prior_controls']=controls
    summary['spatial_prior_plan']=json.loads((v2/'frozen_plan.json').read_text())
    summary['global_budget_receipt']=json.loads((v2/'global_budget_receipt.json').read_text())
    summary['auxiliary_interface_diagnostic']=json.loads((docs/'AUXILIARY_GPU_DIAGNOSTIC.json').read_text())
    summary['clean_public_source_GPU_check']=json.loads((docs/'CLEAN_PUBLIC_GIT_SNAPSHOT_GPU_RECEIPT.json').read_text())
    summary['hardware_identity']=json.loads((docs/'GPU_HARDWARE_IDENTITY.json').read_text())
    summary['final_resource_receipt']=json.loads((docs/'FINAL_RESOURCE_RECEIPT.json').read_text())
    summary['GPU_hours_train_and_eval_residency']+=sum(c['receipt']['GPU_hours'] for c in controls.values())
    assert summary['global_budget_receipt']['GPU_hours_upper_estimate_including_one_hour_smoke_reserve']<=24
    write_json(path,summary);write_json(RUN/'final_comparison.json',summary)
    def tables(scope):
        lines=['|Architecture/model/dataset|Frames|R50|R75|Precision|FP/frame|FN|Duplicates|AP50 / AP75 / mAP|',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        groups={'G0 spatial prior':controls['g0'][scope]['metrics'],'G2 spatial prior':controls['g2'][scope]['metrics']}
        for label,value in groups.items():
            for dataset,m in value.items():
                if dataset=='per_video':continue
                ap=' / '.join(f'{100*m[k]:.3f}' for k in ['AP50','AP75','mAP'])
                lines.append(f"|{label}/{dataset}|{m['frames']}|{100*m['Recall50']:.3f}|{100*m['Recall75']:.3f}|{100*m['Precision']:.3f}|{m['FP_per_frame']:.3f}|{m['FN']}|{m['duplicates']}|{ap}|")
        return '\n'.join(lines)
    text=['\n## Pre-frozen spatial attention controls','',
          'Weak global-query DEV AP and successful two-image overfit motivated one additional bounded matched pair. Before reading any external head metrics, sigma0.08 Gaussian additive attention bias at the model reference centers was frozen. No GT/proposal enters the prior and no trainable parameter was added. Same100 queries/3 layers/seed42/initial weights/sampling stream/1,000 steps/optimizer/LR. Entire sampling SHA, unique optimizer images and domain exposures match across all four runs. G2 usesGPU1 and G0GPU7; original global pair remainsGPU4/6. Combined feature cache limit20GiB (<25GiB). Whole-task budget monitors both pairs, with one GPU-hour conservatively reserved for functional audits. Original results/checkpoints remain preserved.', '',
          '### Full internal DEV (still seen by G2 TRAIN)','',tables('full_DEV'),'',
          '### Same frozen external videos, no external selection','',tables('external'),'',
          '|Arm|Best step|Threshold|Allocated MiB|GPU-hours|Fresh end-to-end detection FPS|Vision FPS|Decoder FPS|',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,c in controls.items():
        r=c['receipt'];timing=c['full_DEV']['timing']
        text.append(f"|{arm} spatial prior|{r['selected_step']}|{r['selected_threshold']}|{r['peak_allocated_GPU_MiB']:.1f}|{r['GPU_hours']:.4f}|{timing['evaluation_wall_FPS']:.3f}|{timing['vision_FPS']:.3f}|{timing['decoder_FPS']:.3f}|")
    total=summary['GPU_hours_train_and_eval_residency']
    text+=['',f'All four measured train+evaluation residency costs total **{total:.4f} GPU-hours**. The recorded global upper estimate including a full1GPU-hour functional-audit reserve is {summary["global_budget_receipt"]["GPU_hours_upper_estimate_including_one_hour_smoke_reserve"]:.4f}, within24. GPU diagnostics and raw weights remain on server. No external frame/query was used in an optimizer update.', '',
           'Source provenance: original global heads launched at7829ff66; spatial-prior heads launched at5254d8bc. Their launch registrations are authoritative. The entrypoint also stored current Git HEAD at completion, which may be a newer reporting/control commit; it is not the launch SHA. The newer source preserves original global numerical behavior when the optional spatial prior is None. Full source SHAs and registration config are in PILOT_RESULTS.json.', '',
           'All arms completed1,000 optimizer updates, but the original global G0 selected step750 while global G2 selected1000. Treat that global pair as an equal-budget/equal-selection-protocol comparison. The primary spatial-prior G0/G2 pair both selected1000 and therefore also matches evaluated update count and exact label exposures.']
    with (docs/'PILOT_TRAINING.md').open('a') as f:f.write('\n'.join(text)+'\n')
    auxiliary=summary['auxiliary_interface_diagnostic']
    with (docs/'PILOT_TRAINING.md').open('a') as f:
        f.write('\n## Limited existing TRAIN interfaces\n\n')
        f.write('The frozen spatial G2 head ran two existing TRAIN images on GPU4 with zero optimizer updates and no native language decoding. This is an interface diagnostic, not a new domain benchmark. TAO uses Base-only partial positives; Refer uses all supplied physical labels.\n\n')
        f.write('|TRAIN image domain|Supplied target boxes|Accepted direct boxes|Coverage50 on this one TRAIN image|\n|---|---:|---:|---:|\n')
        for row in auxiliary['rows']:
            f.write(f"|{row['dataset']}|{row['Base_or_physical_GT_count']}|{row['above_DEV_threshold']}|{100*row['coverage50_on_this_TRAIN_image']:.3f}%|\n")
        f.write(f"\nBoth coordinate outputs and positive-only losses were finite; TAO localization was not solved by bypassing the parser. Additional diagnostic residency: {auxiliary['GPU_hours']:.6f} GPU-hours, included within the one-hour functional-audit reserve. See AUXILIARY_GPU_DIAGNOSTIC.json. No AP, TETA, open-category accuracy or language-satisfaction claim is made.\n")
        f.write('\n## Observed timing on the common external scope\n\n')
        for dataset,timing in summary['external_comparison']['A0']['native_timing_by_dataset'].items():
            f.write(f"Native cached {dataset}: {timing['historical_request_FPS']:.3f} request FPS; {timing['rejected_frames']}/{timing['frames']} rejected frames. ")
        f.write('These are historical serving request times including its concurrent load, not an isolated vision benchmark.\n\n')
        f.write('|Readout|Fresh external vision FPS|Decoder FPS|End-to-end wall FPS|\n|---|---:|---:|---:|\n')
        for arm in ['g0','g2']:
            for label,record in [('global',summary['external_comparison']['A1' if arm=='g0' else 'A2']),('spatial prior',controls[arm]['external'])]:
                t=record['timing']
                f.write(f"|{arm} {label}|{t['vision_FPS']:.3f}|{t['decoder_FPS']:.3f}|{t['evaluation_wall_FPS']:.3f}|\n")
        t=controls['g2']['external']['timing']
        f.write(f"\nThe G2 spatial model spends {100*t['summed_actual_extraction_seconds']/(t['summed_actual_extraction_seconds']+t['decoder_seconds']):.2f}% of measured vision+decoder time in native visual extraction. Its parallel decoder is already a small part of total latency. Different concurrent GPU loads remain a limitation; no controlled speedup ratio is inferred.\n")
        f.write('\n## Preserved checkpoint index\n\nAll paths below are relative to `/data3/user/LocateMOT-runs/G2_Spatial_DenseHead_20261026/`. Each directory preserves `best_dev.pt` and `last.pt`; every last checkpoint is step1000. No weights are uploaded. Backbone model identities and full-file hashes are in SPATIAL_FEATURE_AUDIT.md.\n\n')
        f.write('|Directory|Best step|Best SHA256|Last SHA256|\n|---|---:|---|---|\n')
        for arm in ['g0','g2']:
            for label,r in [(arm,summary['receipts'][arm]),('spatial_prior_v2/'+arm,controls[arm]['receipt'])]:
                f.write(f"|`{label}`|{r['selected_step']}|`{r['checkpoint_best_SHA256']}`|`{r['checkpoint_last_SHA256']}`|\n")
    a0=summary['external_comparison']['A0']['metrics'];g2=controls['g2']['external']['metrics'];g0=controls['g0']['external']['metrics']
    attribution=json.loads((docs/'DETECTION_ERROR_ATTRIBUTION.json').read_text())
    for arm in ['g0','g2']:
        assert attribution['spatial_prior_v2/'+arm]['head_checkpoint_SHA256']==controls[arm]['receipt']['checkpoint_best_SHA256']
    animal,sports=g2['animaltrack'],g2['sportsmot']
    decision=['# Evidence-based next research decision','',
        '**Do not replace native final G2 or start full temporal identity training from this pilot.** The spatial attention prior improves the weak global-query readout, but coverage and false positives remain substantially worse than native G2. The matched G0 readout also has slightly higher external AP50 than G2 in both domains. This completes the bounded detector experiment, not the unified GroundJEV research goal.', '',
        'Scope:20 full benchmark DEV sequences /8,370 frames, independently held out from the head optimizer but previously seen in G2 TRAIN; plus a frozen6-sequence /1,607-frame external comparison disjoint from known head/G2 TRAIN video sources. External videos were used in earlier native G2/tracking research, so they are not an untouched final TEST. External thresholds and checkpoint choices were never retuned.', '',
        '1. **Do G2 spatial features support detection?** Yes for a working, trainable image-to-scored-box path; this1,000-step pilot does not demonstrate a competitive detector. Frozen428,482,048 native encoder/projector parameters feed3,975,829 trained decoder parameters. Final G2 step18227 is the actual loaded backbone. No box answer, GT-derived proposal, ROI seed or future frame enters feature extraction. Geometry, gradient, frozen-state and restore checks passed on real GPU images. Perfect memorization of two TRAIN frames is only a capacity sanity check.', '',
        '|Frozen external scope|Native G2 R50 / FP-frame|Spatial G0 R50 / FP-frame|Spatial G2 R50 / FP-frame|',
        '|---|---:|---:|---:|']
    for dataset in ['animaltrack','sportsmot']:
        n,a,b=a0[dataset],g0[dataset],g2[dataset]
        decision.append(f"|{dataset}|{100*n['Recall50']:.3f}% / {n['FP_per_frame']:.3f}|{100*a['Recall50']:.3f}% / {a['FP_per_frame']:.3f}|{100*b['Recall50']:.3f}% / {b['FP_per_frame']:.3f}|")
    decision+=['',
        '2. **Which native errors improve?** Numeric parallel boxes remove the native language grammar/parser step. They recover some native misses, including rejected frames, but lose substantially more existing coverage. These are target-level diagnostics, not a union detector or GT-assisted inference.', '',
        '|Spatial G2 vs native|Newly covered native misses|Lost native coverage|Recovered after native frame rejection|',
        '|---|---:|---:|---:|']
    for dataset in ['animaltrack','sportsmot']:
        g=attribution['spatial_prior_v2/g2']['target_coverage_transitions'][dataset]
        decision.append(f"|{dataset}|{g['recovered50_native_missed']}|{g['lost50_native_covered']}|{g['recovered50_in_native_rejected_frames']}|")
    decision+=['',
        '3. **Is there a fair G2-specific improvement?** Not established. The primary spatial G0/G2 pair both selected step1000/threshold0.3, with identical initialization,4,000 image exposures/2,400 unique inputs, label stream,100 queries,3 layers, AdamW schedule and losses. Their encoder is identical; only five projector tensors differ. Qwen LoRA is not invoked by this image-only path. Global G0 selected750 versus global G2 at1000 under the same selection rule, so that original pair is equal-budget rather than identical evaluated-update count.', '',
        '|External domain|G0 spatial AP50|G2 spatial AP50|G2 minus G0|',
        '|---|---:|---:|---:|']
    for dataset in ['animaltrack','sportsmot']:
        decision.append(f"|{dataset}|{100*g0[dataset]['AP50']:.3f}%|{100*g2[dataset]['AP50']:.3f}%|{100*(g2[dataset]['AP50']-g0[dataset]['AP50']):+.3f} pp|")
    decision+=['',
        'The spatial prior is the sole numerical change from global queries; feature-cache limits change wall time, not updates. G2 spatial AP50 improves over G2 global from4.242% to22.417% on Animal and3.026% to12.015% on Sports, while remaining inadequate. Native A0 has no per-box scores, so its AP is undefined. A0 vs a newly supervised head also changes optimization/readout supervision; it cannot isolate architecture alone from additional supervised learning. No extra distinct GT dataset was added and no external label entered an optimizer.', '',
        f"4. **AnimalTrack positive result?** No net coverage improvement over native G2: {100*animal['Recall50']:.3f}% versus{100*a0['animaltrack']['Recall50']:.3f}%, Precision{100*animal['Precision']:.3f}% versus{100*a0['animaltrack']['Precision']:.3f}%, duplicates{animal['duplicates']} versus{a0['animaltrack']['duplicates']}. Compared with spatial G0, G2 has lower R50/AP50 but slightly higher R75/mAP; this mixed evidence is not a G2 feature advantage. The external scope covers only chicken/deer/duck.", '',
        '|External animal type|GT boxes|Native G2 R50|Spatial G0 R50|Spatial G2 R50|',
        '|---|---:|---:|---:|---:|']
    for category in ['chicken','deer','duck']:
        n,a,b=(m['animaltrack']['groups'][category] for m in [a0,g0,g2])
        decision.append(f"|{category}|{b['GT']}|{100*n['Recall50']:.3f}%|{100*a['Recall50']:.3f}%|{100*b['Recall50']:.3f}%|")
    decision+=['', 'Full internal Animal DEV covers all10 types, one held-out physical video per type; these videos remain G2-TRAIN-scope.', '',
        '|Internal DEV animal type|GT boxes|Spatial G0 R50|Spatial G2 R50|',
        '|---|---:|---:|---:|']
    dev0=controls['g0']['full_DEV']['metrics'];dev2=controls['g2']['full_DEV']['metrics']
    categories=['chicken','deer','dolphin','duck','goose','horse','penguin','pig','rabbit','zebra']
    for category in categories:
        a,b=dev0['animaltrack']['groups'][category],dev2['animaltrack']['groups'][category]
        decision.append(f"|{category}|{b['GT']}|{100*a['Recall50']:.3f}%|{100*b['Recall50']:.3f}%|")
    decision+=['',
        f"5. **SportsMOT advantage retained?** No: external coverage falls from{100*a0['sportsmot']['Recall50']:.3f}% to{100*sports['Recall50']:.3f}%, with FP/frame rising from{a0['sportsmot']['FP_per_frame']:.3f} to{sports['FP_per_frame']:.3f}. Full internal DEV small objects have0/97 coverage and dense scenes R50=15.351%; Animal medium objects R50=5.629%. These failures point to unresolved localization/readout learning and objectness, not an identity decision issue. All groups and one-to-one metrics remain in PILOT_RESULTS.json.", '',
        'At fixed0.5, the spatial G2 external output contains zero Sports boxes and only19 Animal boxes. The0.3 threshold was selected from the frozen internal-DEV grid; even at that operating point coverage/FP are inadequate. Do not lower thresholds using external GT or inflate recall with duplicate boxes.', '',
        '6. **Add TAO/Refer supervision now?** Keep the audited adapters for a later bounded localization study, not full open-vocabulary/identity training. Existing TAO TRAIN has177 Base classes/53,028 non-crowd annotations;1,611 Novel annotations remain excluded. Refer physical labels must not turn non-expression objects into background;157 missing referenced IDs need explicit repair/exclusion before expression training. The two real TRAIN diagnostic images produced finite direct coordinates but TAO covered0/6 targets and Refer2/6, with zero optimizer updates. This is no AP/TETA,1203-class, amodal-regression or language-satisfaction validation.', '',
        '7. **Enter shared Temporal Identity Decoder?** Defer full training. The current detector loses most Sports coverage and produces too many false positives to provide a useful shared identity front end. Keep the ObjectState boxes/scores/query visual tokens/geometry and reserved language/category/history/causal-state fields. There is no need to regenerate G2 autoregressive candidates for this future path. Original J1/Coupled checkpoints are retained; no old JEV/OneJev training or new HOTA claim occurred.', '',
        '8. **Unfreeze projector/LoRA?** Do not start full G2 tuning from this result. A real TRAIN detection loss backpropagated into the online G2 projector (fc2 gradient norm4.40888), with frozen encoder and zero parameter updates. That proves an available gradient path, not successful joint training. Current training used detached frozen features. If continuing, preregister one bounded study of readout optimization and spatial resolution/pre-merge features on TRAIN-internal DEV before reconsidering projector updates. Current evidence does not separate short-training underfit, representation resolution and readout design, and does not prove G2 features intrinsically unusable. Qwen LoRA still needs a genuine answer-free causal path and its own resource audit.', '',
        f"Four measured train+evaluation residencies total{total:.4f} GPU-hours; with the full1GPU-hour functional-audit reserve the conservative upper estimate is{summary['global_budget_receipt']['GPU_hours_upper_estimate_including_one_hour_smoke_reserve']:.4f}, below24. Primary G2 fresh external end-to-end FPS={controls['g2']['external']['timing']['evaluation_wall_FPS']:.3f}; native visual extraction accounts for98.58% of vision+decoder time. Observed speed is under shared load, not a controlled isolated speedup benchmark. Peak allocated memory is3010.7MiB per model on A800 GPUs.", '',
        'Old full evaluation remains PAUSED_BY_USER_FOR_ARCHITECTURE_PIVOT. All old caches/weights and consistent SQLite snapshots remain intact; no automatic recovery. The bounded pilot owners exited successfully. Exact checkpoints, source-launch SHAs, failed/weak scopes and complete video metrics are preserved.']
    (docs/'NEXT_RESEARCH_DECISION.md').write_text('\n'.join(decision)+'\n')
    report=docs/'PILOT_TRAINING.md'
    original=report.read_text().split('\n',1)[1].lstrip()
    lead=['# Real GPU pilot training','',
        '**All four bounded trials completed1,000 optimizer updates and their complete video evaluations. The primary spatial G2 head does not reach native G2 detection quality; do not replace the front end or start full identity training from this result.**', '',
        'Primary comparison: spatial-attention G0/G2 both select step1000 and threshold0.3 from internal DEV; native A0 is reused from frozen caches. Complete evaluation per model:8,370 head-held-out DEV frames /20 benchmark sequences plus1,607 frozen external frames /6 sequences. Internal DEV was in G2 TRAIN; external scope was used in prior native research, not untouched final TEST.', '',
        f"All four measured train+evaluation residencies: **{total:.4f} GPU-hours**; including the full1GPU-hour functional reserve, conservative upper estimate **{summary['global_budget_receipt']['GPU_hours_upper_estimate_including_one_hour_smoke_reserve']:.4f}**, within24. Hardware: NVIDIA A800-SXM4-80GB, at most four physical GPUs. Peak allocated GPU memory3010.7MiB per model. Overall temporary feature cache limit20GiB.", '',
        'See NEXT_RESEARCH_DECISION.md for the measured negative result and eight research decisions; PILOT_RESULTS.json retains every model/video/group, including fixed0.5 operating points. Native AP is undefined; dense AP uses learned objectness under a documented class-agnostic protocol.', '',
        '## Original global-query pair (retained weak baseline)','',original]
    report.write_text('\n'.join(lead))
    print(json.dumps({'status':'ALL_FOUR_MATCHED_TRIALS_COMPARISON_COMPLETE','total_GPU_hours':total,
                      'spatial_G2_external':{d:{k:g2[d][k] for k in ['Recall50','Recall75','Precision','FP_per_frame','AP50']} for d in ['sportsmot','animaltrack']}},indent=2))


if __name__=='__main__':append_spatial_prior_controls()
