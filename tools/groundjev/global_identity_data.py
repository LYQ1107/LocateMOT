"""Stream original causal actor states; GT labels remain outside model inputs."""
from collections import Counter
import json
from pathlib import Path


def training_frames(plan):
    import numpy as np
    import torch
    from PIL import Image
    from locatemot.groundjev.tracker import Track
    from tools.groundjev.analyze_tracking_events import load_truth
    from tools.groundjev.audit_global_identity_labels import frame_support
    source=Path(plan['immutable_source_root']);index=json.loads((source/plan['source_index']).read_text())
    for unit_name,unit in index['units'].items():
        if unit['task']!='mot':continue
        manifest=json.loads((source/unit['manifest']).read_text())
        assert manifest['scope']=='registered_train_only_native_source_not_benchmark_evaluation'
        manifest['gt_path_for_evaluation_only']=manifest.get('gt_path_for_evaluation_only',manifest.get('gt_path'))
        truth,_=load_truth(manifest);tracks={};anchors={};seen=set()
        with Image.open(manifest['images'][0]) as image:size=image.size
        with (source/unit['original_run']/'frame_decisions.jsonl').open() as trace:
            for row in map(json.loads,trace):
                number=row['frame'];timestamp=row['state']['timestamp'];candidates=row['candidates']
                with np.load(source/unit['frames'][str(number)]['feature_path'],allow_pickle=False) as archive:
                    regions=archive['regions']
                assert len(regions)==len(candidates)
                candidates=[dict(c,native_region_feature=regions[i]) for i,c in enumerate(candidates)]
                gt=truth[number];matrix,matched,support=frame_support(candidates,gt)
                live=list(tracks.values());by_id={t.track_id:i for i,t in enumerate(live)}
                winners={j:i for i,j in matched.items()};options=[];identities=[];qualities=[]
                for i,supported in enumerate(support):
                    identity=gt[supported[0]]['identity'] if len(supported)==1 else None
                    identities.append(identity if identity is not None else -1)
                    qualities.append(float(matrix[i,supported[0]]) if identity is not None else 0.)
                    choices=[]
                    if identity is not None:
                        if winners.get(supported[0])!=i:
                            if supported[0] in winners:choices=[len(live)+1]
                        else:
                            choices=[by_id[tid] for tid in by_id if anchors.get(tid)==identity]
                            if not choices and identity not in seen and not any(anchors.get(tid) is None for tid in by_id):
                                choices=[len(live)]
                    options.append(choices)
                labels=dict(action_options=options,unique_GT=torch.tensor(identities),quality=torch.tensor(qualities),
                    GT_supervision_only=True,full_truth=gt,birth_anchors=dict(anchors),support=support,matched=matched)
                yield dict(unit=unit_name,frame=number,timestamp=timestamp,image_size=size,candidates=candidates,
                    tracks=live,labels=labels,manifest=manifest,source_row=row)
                # Reconstruct original predicted histories, including wrong bindings;
                # never replace them with GT trajectories or future observations.
                active=set(row['state']['persistent_ids'])
                for event in row['state']['events']:
                    if 'candidate' not in event or 'track_id' not in event:continue
                    i=event['candidate'];tid=event['track_id'];candidate=candidates[i];feature=regions[i].copy()
                    if event['action']=='NEW':
                        t=Track(tid,list(candidate['box']),candidate['label'],None,None,
                            timestamp=timestamp,last_reliable_timestamp=timestamp)
                        t.anchor_feature=feature.copy();t.long_feature=feature.copy();tracks[tid]=t
                        anchors[tid]=gt[support[i][0]]['identity'] if len(support[i])==1 and i in matched else None
                    else:
                        t=tracks[tid];dt=timestamp-t.timestamp
                        t.velocity=[(a-b)/dt for a,b in zip(candidate['box'],t.box)]
                        t.box=list(candidate['box']);t.timestamp=timestamp;t.age+=1
                    t.recent_feature=feature
                    if event.get('memory_action')=='UPDATE':
                        t.long_feature=feature.copy();t.last_reliable_timestamp=timestamp
                    t.lost_age=0;t.state='ACTIVE'
                    t.confidence=event['decision']['probabilities'].get(str(tid),event['decision']['probabilities']['NEW'])
                for tid in list(tracks):
                    if tid not in active:del tracks[tid]
                    else:tracks[tid].lost_age=timestamp-tracks[tid].timestamp
                seen.update(g['identity'] for g in gt)
