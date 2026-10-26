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
           '|Arm|Best step|Threshold|Peak allocated MiB|Peak reserved MiB|GPU-hours (train+eval residency)|Vision FPS / Decoder FPS / total FPS (fresh full DEV)|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in receipts.items():
        timing=summary['full_DEV'][arm]['timing']
        lines.append(f"|{arm}|{r['selected_step']}|{r['selected_threshold']}|{r['peak_allocated_GPU_MiB']:.1f}|{r['peak_reserved_GPU_MiB']:.1f}|{r['GPU_hours']:.4f}|{timing['vision_FPS']:.3f} / {timing['decoder_FPS']:.3f} / {timing['vision_plus_decoder_FPS']:.3f}|")
    lines+=['',f'Total measured pilot train+evaluation residency: {total_cost:.4f} GPU-hours, plus separately bounded short smoke/feature audits. Large weights and full predictions remain under `{RUN}`. Git includes only summaries/indices/configuration. Native A0 time is historical shared-load inference; dense timing is current shared-load inference. Equal images and A100 hardware class do not imply matched concurrent load; treat speed as observed end-to-end evidence, not an isolated kernel benchmark.', '',
            'Loss curves and sampled-DEV curves are exported as PILOT_CURVES.png. Loss decrease alone is not a scientific gain. No long G2 unfreezing, OneJev SFT, identity decoder training, RL or new large data downloads occurred.', '',
            'Reproduce: `python tools/groundjev/dense_head/prepare_train_data.py`; `python tools/groundjev/dense_head/prepare_external_comparison.py`; run the two GPU leakage audits, compare spatial features, then `python tools/groundjev/dense_head/run_matched_pilot.py` in GroundAnything environment configuration. Local assets use existing server read-only links; external repository/environment pins are in the inherited dependency tools and environment. Single-arm resume supports optimizer/scheduler/sampler/RNG restoration via `train_dense_head.py --config ... --resume ...`; do not run two owners for an arm. The completed1,000-step runs must be archived before a new trial directory is used.', '',
            'Known limits: benchmark target-set foreground is class-agnostic, not exhaustive annotation of every semantic object. TAO/Refer partial labels are not in optimization. Refer audit finds157 absent referenced IDs; future supervision must exclude/repair those documented rows. Optional SAM checkpoint/runtime was not registered, so no SAM performance claim. No JEV/HOTA gain is inferred from this detection experiment.']
    (docs/'PILOT_TRAINING.md').write_text('\n'.join(lines)+'\n')
    # Research decision is based on both independent coverage and false positives.
    animal0=results['A0']['metrics']['animaltrack'];animal2=results['A2']['metrics']['animaltrack']
    sports0=results['A0']['metrics']['sportsmot'];sports2=results['A2']['metrics']['sportsmot']
    animal_gain=animal2['Recall50']-animal0['Recall50'];sports_gain=sports2['Recall50']-sports0['Recall50']
    clear_gain=animal_gain>.02 and animal2['FP_per_frame']<=animal0['FP_per_frame']+1 and sports_gain>-.02
    decisions=['# Next research decision','', 'The first matched parallel-head pilot is complete; the broader unified GroundJEV research is not complete.', '',
        f"1. **Spatial feature usability:** real GPU training and full DEV detection demonstrate an operational image-to-box path with a frozen G2 vision/projector. Whether it is competitive is determined by the following metrics, not its training loss. Animal external R50={100*animal2['Recall50']:.3f}%, Sports={100*sports2['Recall50']:.3f}%.", '',
        f"2. **Relative to native G2:** Animal coverage delta={100*animal_gain:+.3f} percentage points, FP/frame {animal0['FP_per_frame']:.3f}->{animal2['FP_per_frame']:.3f}; Sports coverage delta={100*sports_gain:+.3f}pp, FP/frame {sports0['FP_per_frame']:.3f}->{sports2['FP_per_frame']:.3f}. Native syntax/box parsing is bypassed by direct numeric coordinates, but bypassing a parser is not proof of better detection.", '',
        '3. **Attribution:** A1 and A2 share head size, training labels, sampling stream, initialization, optimizer, steps and evaluation. Their difference measures the checkpoint-specific projector effect, because the native visual encoder is unchanged. The language LoRA contributes nothing to this image-only path. A0 vs a newly supervised head also changes the trainable prediction mechanism/optimization; this experiment cannot isolate a pure architectural cause from supervised readout learning. No new distinct GT dataset was added, and external GT is evaluation-only.', '',
        f"4. **AnimalTrack:** {'A positive coverage signal exists under the recorded FP constraint' if clear_gain else 'The pilot has not established a practically useful recall improvement under the joint FP/Sports constraints'}. Category/size/density breakdown is preserved; the external Animal scope contains3 categories, not all10 or the whole dataset.", '',
        f"5. **SportsMOT:** {'The frozen external scope retains native coverage within2pp' if sports_gain>-.02 else 'The parallel pilot does not preserve the very high native recall; do not replace the native detector yet'}. Report selected-threshold and fixed0.5 metrics; do not lower thresholds on external labels.", '',
        '6. **TAO/Refer extension:** keep Base-only and all-physical-object adapters, with non-exhaustive negative masking; complete the recorded157-ID Refer boundary before training expressions. A class-agnostic two-domain head does not establish1203-class open-vocabulary detection. The small interface diagnostic is not a TAO benchmark.', '',
        '7. **Shared temporal identity decoder:** '+('Proceed only after repeating the bounded detection signal across more frozen categories/videos and checking objectness calibration.' if clear_gain else 'Defer full identity training. Detection coverage/precision must first become useful on the frozen development protocol. Keep the ObjectState interface for later causal IDs; do not restart old JEV training.'), '',
        '8. **Projector/LoRA unfreezing:** not justified by a decoder-only pilot alone. The audited online projector gradient exists, but current training used detached frozen features. Next, if continuing, use one pre-budgeted readout-capacity/optimization study on internal DEV before considering projector adaptation. Qwen LoRA requires a genuine answer-free causal language path; never claim that frozen cached features jointly fine-tuned G2.', '',
        'No native A0 regeneration, test-set tuning, forced NEW identity action, mock training, threshold-based invented AP or automatic old-evaluation resumption occurred. All weights and raw results are preserved.']
    (docs/'NEXT_RESEARCH_DECISION.md').write_text('\n'.join(decisions)+'\n')
    write_json(RUN/'final_comparison.json',summary)
    print(json.dumps({'status':summary['status'],'GPU_hours':total_cost,'animal_delta_pp':100*animal_gain,
                      'sports_delta_pp':100*sports_gain,'ready_for_identity':clear_gain,'docs':str(docs)},indent=2))


if __name__=='__main__':main()
