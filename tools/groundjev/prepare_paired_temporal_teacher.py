"""Matched short/long legal TRAIN teacher anchors, native image recipes only."""
import argparse
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import gzip
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    import numpy as np
    from PIL import Image
    from locatemot.groundjev.decision import OneJevDecision
    from locatemot.groundjev.tracker import Track
    from tools.groundjev.prepare_tracking_decisions import PROTOCOL,annotations
    from tools.groundjev.streamed_tracking_media import CANDIDATE_KEY,HISTORY_KEY,render_tracking_media
    from tools.groundjev.group_reference_rows import REFERENCE_KEY
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());rng=random.Random(plan['seed'])
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    index=json.loads((ROOT/plan['train_index']).read_text())
    units=[(u,v) for u,v in index['units'].items() if u.startswith('mot/')];assert len(units)==43
    class MetadataBuilder(OneJevDecision):
        # These dimensions describe on-demand native crop recipes; no synthetic
        # images are passed to training. Pixel parity is checked below.
        def _candidate_sheet(self,image,candidates):
            return SimpleNamespace(width=640,height=max(1,math.ceil(len(candidates)/4))*192)
        def _history_sheet(self,tracks):
            return SimpleNamespace(width=1152,height=max(1,math.ceil(len(tracks)/4))*144)
    metadata=MetadataBuilder.__new__(MetadataBuilder);metadata.evidence_format='v3_crop_motion'
    native=OneJevDecision.__new__(OneJevDecision);native.evidence_format='v3_crop_motion'
    specification=dict(task='mot',vocabulary=['person'],expression=None)
    groups={};teacher_rows={};gap_values=defaultdict(list);coverage={};paired=[];pixel_checks=0
    for arm in ('short','long'):
        directory=out/arm;directory.mkdir();groups[arm]=gzip.open(directory/'groups.jsonl.gz','wt');teacher_rows[arm]=[]
    for unit_index,(unit,info) in enumerate(units):
        manifest=json.loads((ROOT/info['manifest']).read_text())
        protocol=json.loads(Path(manifest['source_train_protocol']).read_text())
        sequence=next(s for s in protocol['sequences'] if s['sequence']==manifest['sequence'])
        names={int(Path(f).stem):f for f in sequence['train_frames']};allowed=set(names)
        truth=annotations(sequence,allowed);by_frame={f:{r['gt_id']:r for r in values} for f,values in truth.items()}
        observations=defaultdict(list)
        for frame in sorted(truth):
            for obj in truth[frame]:observations[obj['gt_id']].append(frame)
        fps=sequence['fps'];long_distance=math.ceil(plan['minimum_long_seconds']*fps)
        feasible={}
        for identity,frames in observations.items():
            possibilities=[]
            for current in frames:
                short_start=bisect_left(frames,current-max(1,int(plan['maximum_short_seconds']*fps)))
                short_end=bisect_left(frames,current)
                long_end=bisect_right(frames,current-long_distance)
                if short_end>short_start and long_end>0:possibilities.append((current,short_start,short_end,long_end))
            if possibilities:feasible[identity]=possibilities
        assert feasible,('No legal long-history positive for unit',unit)
        quota=plan['teacher_rows']//len(units)+(unit_index<plan['teacher_rows']%len(units))
        selected_frames=set()
        for sample in range(quota):
            identity=rng.choice(list(feasible));current,lo,hi,end=rng.choice(feasible[identity])
            frames=observations[identity];short=rng.choice(frames[lo:hi]);long=rng.choice(frames[:end])
            shared=set(by_frame[current])&set(by_frame[short])&set(by_frame[long])
            others=sorted(shared-{identity});rng.shuffle(others);history_ids=[identity]+others[:plan['history_tracks']-1];rng.shuffle(history_ids)
            aliases=dict(zip(history_ids,rng.sample(range(100,99999),len(history_ids))))
            current_path=str(Path(sequence['image_dir'])/names[current]);candidate=by_frame[current][identity]
            assert Path(current_path).is_file();selected_frames.add(current)
            identifier=sequence['sequence']+f'_teacher{sample:04d}'
            condition={}
            for arm,past in [('short',short),('long',long)]:
                past_path=str(Path(sequence['image_dir'])/names[past]);assert past<current and past in allowed
                assert Path(past_path).is_file();selected_frames.add(past)
                tracks=[];recipes=[]
                for gt_id in history_ids:
                    previous=by_frame[past][gt_id];box=previous['box'];alias=aliases[gt_id]
                    tracks.append(Track(alias,box,'person',None,None,long_image=None,
                        timestamp=past/fps,last_reliable_timestamp=past/fps,lost_age=(current-past)/fps,state='LOST'))
                    observation=dict(path=past_path,box=box,sequence=sequence['sequence'],frame=past)
                    recipes.append(dict(id=alias,anchor=observation,long=observation,recent=observation))
                image=SimpleNamespace(width=sequence['width'],height=sequence['height'])
                request,sheets=metadata.build_request(image,[candidate],tracks,current/fps,specification)
                media=[dict(type='image',path=current_path,w=image.width,h=image.height),
                    dict(type='image',path=current_path,w=sheets[1]['image'].width,h=sheets[1]['image'].height,**{CANDIDATE_KEY:[candidate['box']]}),
                    dict(type='image',path=current_path,w=sheets[2]['image'].width,h=sheets[2]['image'].height,**{HISTORY_KEY:recipes})]
                # Only identity supervision; other native questions stay absent
                # from this teacher component and existing native rows remain.
                questions={'match_0':request.questions['match_0'].model_dump(exclude_none=True)}
                group=dict(id=identifier,state=request.state,media=media,questions=questions)
                groups[arm].write(json.dumps(group)+'\n')
                row=dict(id=identifier,split='train',source=sequence['sequence'],task='identity',
                    target={str(aliases[identity]):1.},**{REFERENCE_KEY:dict(path=str((out/arm/'groups.jsonl.gz').relative_to(ROOT)),group=identifier,question='match_0')},
                    supervision=dict(tracking_task_bucket='mot',component='paired_GT_temporal_identity_teacher',
                        teacher_current=current,teacher_past=past,history_gap_seconds=(current-past)/fps,
                        GT_teacher_not_actual_actor_rollout=True,GT_identity_labels_outside_model_inputs=True))
                teacher_rows[arm].append(row);gap_values[arm].append((current-past)/fps);condition[arm]=group
                if sample==0:
                    with Image.open(current_path) as src:actual_image=src.convert('RGB')
                    with Image.open(past_path) as src:past_image=src.convert('RGB')
                    for track,gt_id in zip(tracks,history_ids):
                        crop=past_image.crop(tuple(by_frame[past][gt_id]['box'])).convert('RGB');crop.thumbnail((192,192))
                        track.anchor_image=crop;track.recent_image=crop;track.long_image=crop
                    actual,actual_media=native.build_request(actual_image,[candidate],tracks,current/fps,specification)
                    assert actual.state==request.state
                    assert actual.questions['match_0'].model_dump(exclude_none=True)==questions['match_0']
                    for item,pixels in zip(media[1:],actual_media[1:]):
                        rendered=render_tracking_media(item,ROOT)
                        assert rendered.size==pixels['image'].size and rendered.tobytes()==pixels['image'].tobytes()
                        pixel_checks+=1
            assert condition['short']['state']['candidates']==condition['long']['state']['candidates']
            assert condition['short']['questions']==condition['long']['questions']
            assert teacher_rows['short'][-1]['target']==teacher_rows['long'][-1]['target']
            paired.append(dict(id=identifier,current_image=current_path,history_aliases=list(aliases.values()),
                short_gap_seconds=(current-short)/fps,long_gap_seconds=(current-long)/fps))
        coverage[unit]=dict(legal_train_frames=len(allowed),selected_teacher_images=len(selected_frames),
            original_native_clip_seconds=(max(manifest['frame_numbers'])-min(manifest['frame_numbers']))/fps,teacher_rows=quota)
        print(json.dumps(dict(status='PAIRED_LEGAL_TEMPORAL_TEACHERS_PREPARED',units=len(coverage),rows=len(paired))),flush=True)
    for file in groups.values():file.close()
    assert len(paired)==plan['teacher_rows']==4000 and pixel_checks==43*2*2
    with gzip.open(ROOT/plan['original_curriculum'],'rt') as f:original=[json.loads(l) for l in f]
    original_mot=[r for r in original if r['supervision']['tracking_task_bucket']=='mot']
    selected=[]
    for family,count in plan['retained_native_MOT_families'].items():
        pool=[r for r in original_mot if (r['task'] in ('identity','empty_presence') if family=='identity_or_empty_presence' else r['task']==family)]
        selected.extend(rng.sample(pool,count))
    rmot=[r for r in original if r['supervision']['tracking_task_bucket']=='rmot']
    ovmot=[r for r in original if r['supervision']['tracking_task_bucket']=='ovmot']
    assert len(selected)==4000 and len(rmot)==len(ovmot)==8000
    order=list(range(8000));random.Random(plan['seed']).shuffle(order)
    # Keep selected source rows in their exact native form and order. Both
    # arms have the same20000 immutable rows followed by4000 matched teachers;
    # the registered index-shuffle sampler consumes all24000 once.
    positions={r['id']:i for i,r in enumerate(original)}
    common=[]
    for i in order:
        for rows in (selected,rmot,ovmot):
            if i<len(rows):common.append(positions[rows[i]['id']])
    assert len(common)==20000 and len(set(common))==20000
    for arm in ('short','long'):
        directory=out/arm;path=directory/'train.sources.json';teachers=directory/'teacher_rows.jsonl.gz'
        with gzip.open(teachers,'wt') as f:
            for row in teacher_rows[arm]:f.write(json.dumps(row)+'\n')
        sources=dict(protocol='immutable-native-explicit-row-index-sources-v1',rows=24000,
            sources=[dict(path=plan['original_curriculum'],families=None,rows=20000,source_rows=len(original),row_indices=common),
                dict(path=str(teachers.relative_to(ROOT)),families=None,rows=4000)])
        path.write_text(json.dumps(sources,separators=(',',':'))+'\n')
        config=json.loads((ROOT/plan['base_training_config']).read_text())
        config.update(model=str(ROOT/plan['initial_checkpoint']),train=str(path),name=plan['training_names'][arm],
            runs_dir=plan['runs_dir'],max_steps=6000,save_every=1000)
        (directory/'training_config.json').write_text(json.dumps(config,indent=2)+'\n')
    report=dict(status='PAIRED_SHORT_LONG_TEMPORAL_NATIVE_TEACHER24000_CURRICULA_PREPARED_NOT_TRAINED',
        teacher_rows_each=4000,total_rows_each=24000,rows_per_task_each=dict(mot=8000,rmot=8000,ovmot=8000),
        source_units=43,coverage=coverage,gap_quantile_levels=[0,.5,.95,1],
        gap_seconds={a:np.quantile(v,[0,.5,.95,1]).tolist() for a,v in gap_values.items()},
        native_reference_pixel_sheets_exact=pixel_checks,native_request_state_question_proof_groups=86,
        paired_all_current_images_boxes_aliases_questions_targets_exact=True,
        identical_retained_native4000_and_semantic16000_rows_order_and_hyperparameters=True,
        same_model_all_tasks_and_same_FP32_native6000_initialization=True,
        teacher_scope='Positive identity teacher episodes with known selected anchors provided across gaps; not actual own-history online rollout, long-BPTT or implemented long-TTL archive. Existing native action/auxiliary supervision retained in declared4000rows.',
        new_images_or_ROI_feature_archive=False,production_optimizer_updates=0,full_goal_complete=False)
    with gzip.open(out/'paired_episode_index.json.gz','wt') as f:json.dump(paired,f)
    metadata_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    assert metadata_bytes<plan['new_metadata_budget_bytes']
    report['generated_metadata_bytes_before_summary']=metadata_bytes
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
