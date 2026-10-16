"""Check complete canonical/readout physical trajectories and final memories."""
import argparse
from collections import Counter
from itertools import zip_longest
import json
import os
from pathlib import Path


def physical_event(event):
    if 'decision' not in event:return event
    return dict(event,decision={k:v for k,v in event['decision'].items()
        if k not in ('category_decision','query_membership_probability')})


def main():
    import torch
    parser=argparse.ArgumentParser();parser.add_argument('--completed',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--registered-feature-upper',type=int,required=True)
    args=parser.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    frozen=json.loads(args.completed.read_text());cases={};totals=Counter()
    for scope in frozen['scopes']:
        before=frozen['completed'][scope+'_canonical_untrained'];after=frozen['completed'][scope+'_task_readout']
        a,b=Path(before['run']),Path(after['run']);counts=Counter()
        with (a/'frame_decisions.jsonl').open() as old,(b/'frame_decisions.jsonl').open() as new:
            for first,second in zip_longest(old,new):
                assert first is not None and second is not None
                x,y=json.loads(first),json.loads(second)
                assert x['frame']==y['frame'] and x['image_path']==y['image_path'] and x['candidates']==y['candidates']
                sx,sy=x['state'],y['state']
                for key in ('persistent_ids','next_id','timestamp','visual_memory_source'):assert sx[key]==sy[key]
                assert [physical_event(e) for e in sx['events']]==[physical_event(e) for e in sy['events']]
                counts['frames']+=1;counts['candidates']+=len(x['candidates']);counts['events']+=len(sx['events'])
                for e,f in zip(sx['events'],sy['events']):
                    if 'candidate' not in e:continue
                    counts['changed_category_decisions']+=e['decision'].get('category_decision')!=f['decision'].get('category_decision')
                    counts['changed_query_membership_probabilities']+=e['decision'].get('query_membership_probability')!=f['decision'].get('query_membership_probability')
                counts['changed_emitted_output_frames']+=x['outputs']!=y['outputs']
        assert counts['frames']==before['decoded_frames']==after['decoded_frames']
        assert counts['candidates']==before['native_candidates']==after['native_candidates']
        rx=json.loads((a/'result.json').read_text());ry=json.loads((b/'result.json').read_text())
        mx=torch.load(rx['final_identity_memory_path'],map_location='cpu',weights_only=False)
        my=torch.load(ry['final_identity_memory_path'],map_location='cpu',weights_only=False)
        def strip_semantics(state):
            return dict(state,tracks=[{k:v for k,v in t.items() if k not in ('label','semantic_counts')} for t in state['tracks']])
        assert strip_semantics(mx)==strip_semantics(my)
        cases[scope]=dict(counts,all_candidate_physical_decisions_lifecycle_writes_and_IDs_exact=True,
            final_full_physical_memory_including_pixels_features_and_timestamps_exact=True,
            only_final_label_and_semantic_counts_excluded_from_memory_equality=True)
        totals.update(counts)
    assert len(cases)==5 and totals['frames']==4500 and totals['candidates']==287744
    assert totals['changed_category_decisions'] and totals['changed_query_membership_probabilities']
    assert not torch.cuda.is_initialized()
    report=dict(status='ALL5_COMPLETE_READOUT_ROLLOUTS_PHYSICAL_EVENTS_AND_FINAL_MEMORY_EXACT',cases=cases,
        totals=dict(totals),source_completed_receipt=str(args.completed),
        scope='Complete saved canonical versus task-conditioned readout trajectories on identical original candidates. Every physical decision/event/ID and final physical memory matches; semantic label/count fields intentionally differ and query filtering may change outputs. No new model, GT analysis, decoder or scorer call.',
        combined_registered_feature_upper_bytes=args.registered_feature_upper,
        global_temporary_feature_budget_bytes=26843545600,new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],totals=report['totals'])),flush=True)


if __name__=='__main__':main()
