"""Prepare small native Compact input references, without copying ROI vectors."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from tools.groundjev.compact_reference_data import PROTOCOL,geometry
    from tools.groundjev.streamed_tracking_media import HISTORY_KEY
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=dict(status='PREPARING_EXACT_NATIVE_COMPACT_INPUT_REFERENCES',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        completed_units=0,new_GPU_or_training_updates=0,full_goal_complete=False)
    def save():
        (out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    save();units={};counts=Counter();images=defaultdict(set)
    try:
        with gzip.open(out/'groups.jsonl.gz','wt') as target:
            for source in plan['sources']:
                exported=json.loads((ROOT/source['export']/'owner_state.json').read_text())
                assert len(exported['completed'])==(43 if source['task']=='mot' else 18)
                for name,item in exported['completed'].items():
                    unit=source['task']+'/'+name;group_file=Path(item['row_references']).parent/'groups.jsonl.gz'
                    original=ROOT/source['source']/'clips'/name
                    runtime=json.loads((original/'result.json').read_text())
                    assert runtime['all_input_frames_processed']
                    archive=ROOT/runtime['native_candidate_archive'];frames={};lookup={}
                    with archive.open() as file:
                        for line in file:
                            frame=json.loads(line);number=frame['frame'];assert number not in frames
                            feature=archive.parent/frame['native_region_features_npz'];assert feature.is_file()
                            frames[number]=dict(feature_path=str(feature.relative_to(ROOT)),candidates=len(frame['candidates']),image_path=frame['image_path'])
                            boxes=defaultdict(list)
                            for i,candidate in enumerate(frame['candidates']):boxes[tuple(candidate['box'])].append(i)
                            lookup[number]=boxes
                            images[source['task']].add(str(Path(frame['image_path']).resolve()))
                    assert len(frames)==runtime['frames']
                    counts['source_frame_contexts']+=len(frames)
                    units[unit]=dict(task=source['task'],archive=str(archive.relative_to(ROOT)),
                        original_run=str(original.relative_to(ROOT)),manifest=str((ROOT/source['manifests']/(name+'.json')).relative_to(ROOT)),
                        full_source_rows=item['row_references'],frames=frames)
                    rows={}
                    def reference(frame,box):
                        matches=lookup[frame][tuple(box)];assert matches
                        if len(matches)>1:
                            with np.load(ROOT/frames[frame]['feature_path'],allow_pickle=False) as file:vectors=file['regions']
                            assert all(np.array_equal(vectors[matches[0]],vectors[index]) for index in matches[1:])
                            counts['duplicate_box_reference_exact_equal_features']+=1
                        return [frame,matches[0]]
                    with gzip.open(group_file,'rt') as file:
                        for line in file:
                            group=json.loads(line);identifier=group['id'];body=group['state']
                            if identifier.endswith('_assigned'):
                                base=rows[identifier.removesuffix('_assigned')]
                                base['assigned_existing_alias_by_candidate']=body['final_assignment']['selected_existing_identity_by_candidate']
                                counts['source_assigned_condition_groups']+=1;continue
                            match=re.fullmatch(r'(.+)_(\d+)_batch(\d+)',identifier);assert match and match[1]==name
                            frame=int(match[2]);media=group['media'][0]
                            assert Path(media['path']).resolve()==Path(frames[frame]['image_path']).resolve()
                            geo=geometry(body,media['w'],media['h'])
                            history=next((m[HISTORY_KEY] for m in group['media'] if HISTORY_KEY in m),[])
                            assert [str(t['id']) for t in history]==[str(t['id']) for t in body['historical_tracks']]
                            history_refs=[]
                            for track in history:
                                triple=[]
                                for kind in ('anchor','recent','long'):
                                    old=track[kind];assert old['sequence']==name and old['frame']<=frame
                                    triple.append(reference(old['frame'],old['box']))
                                history_refs.append(triple)
                            rows[identifier]=dict(id=identifier,unit=unit,frame=frame,image_path=media['path'],
                                specification=body['task'],alias_ids=[str(t['id']) for t in body['historical_tracks']],
                                candidate_refs=[reference(frame,c['box']) for c in body['candidates']],history_refs=history_refs,
                                **{k:v.tolist() for k,v in geo.items()},assigned_existing_alias_by_candidate={},
                                compact_native_skips_empty_context=not body['candidates'] and not body['historical_tracks'])
                    for row in rows.values():
                        target.write(json.dumps(row,separators=(',',':'))+'\n')
                        counts['ordinary_input_groups']+=1;counts['empty_context_groups']+=row['compact_native_skips_empty_context']
                    state['completed_units']+=1;save()
        assert len(units)==plan['expected_units'] and counts['source_frame_contexts']==plan['expected_query_frame_contexts']
        actual_images={k:len(v) for k,v in images.items()};assert actual_images==plan['expected_unique_physical_images']
        (out/'index.json').write_text(json.dumps(dict(protocol=PROTOCOL,units=units),indent=2)+'\n')
        assert not torch.cuda.is_initialized()
        state.update(status='NATIVE_COMPACT_REFERENCE_INPUTS_PREPARED_NUMERICAL_REGRESSION_PENDING',
            counts=dict(counts),unique_physical_images=actual_images,source_FP32_vectors_copied=False,
            original_Full_supervision_rows_and_frames_preserved=True,
            empty_context_scope='Retained in index/records; existing native Compact skips forward when n=m=0. No new presence head or loss is claimed.',
            metadata_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
        assert state['metadata_bytes']<plan['new_metadata_cap_bytes'];save()
    except Exception as error:
        state.update(status='NATIVE_COMPACT_REFERENCE_INPUT_PREPARATION_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
