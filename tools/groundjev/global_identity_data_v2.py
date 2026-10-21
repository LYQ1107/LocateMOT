"""Entity identity is independent of which equivalent proposal wins allocation."""
from collections import Counter


def entity_options(candidates,support,truth,tracks,anchors,ever_bound):
    options=[];counts=Counter();by_id={t.track_id:i for i,t in enumerate(tracks)}
    for supported in support:
        choices=[]
        if len(supported)==1:
            identity=truth[supported[0]]['identity']
            choices=[by_id[tid] for tid in by_id if anchors.get(tid)==identity]
            if choices:counts['MATCH_supported_proposal']+=1
            elif identity not in ever_bound and not any(anchors.get(tid) is None for tid in by_id):
                choices=[len(tracks)];counts['NEW_first_effective_birth_proposal']+=1
            else:counts['UNKNOWN_retired_or_unbound']+=1
        else:counts['UNKNOWN_annotation']+=1
        options.append(choices)
    return options,counts


def training_frames_v2(plan):
    from tools.groundjev.global_identity_data import training_frames
    unit=None;anchors={};ever_bound=set()
    for frame in training_frames(plan):
        if unit!=frame['unit']:unit=frame['unit'];anchors={};ever_bound=set()
        labels=dict(frame['labels']);labels['birth_anchors']=dict(anchors)
        options,counts=entity_options(frame['candidates'],labels['support'],labels['full_truth'],frame['tracks'],anchors,ever_bound)
        labels.update(action_options=options,label_version='entity_identity_separate_from_proposal_winner-v2',
            entity_counts=dict(counts),original_v1_action_options=frame['labels']['action_options'])
        yield dict(frame,labels=labels)
        for event in frame['source_row']['state']['events']:
            if event.get('action')!='NEW':continue
            supported=labels['support'][event['candidate']]
            identity=labels['full_truth'][supported[0]]['identity'] if len(supported)==1 else None
            # This is external birth supervision. The model's predicted ID,
            # pixels, features, motion and memory remain exactly unchanged.
            anchors[event['track_id']]=identity
            if identity is not None:ever_bound.add(identity)
