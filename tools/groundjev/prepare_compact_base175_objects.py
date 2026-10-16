"""Select exactly the existing balanced-curriculum Base teacher observations."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    frames={}
    with (ROOT/plan['source_index']).open() as file:
        for line in file:
            frame=json.loads(line);assert frame['split']=='train' and frame['image_available']
            frames[frame['image_id']]=frame
    annotation=json.loads(Path(next(iter(frames.values()))['annotation_source']).read_text())
    labels={a['id']:a for a in annotation['annotations']}
    vocabulary=json.loads(Path(plan['vocabulary']).read_text())['categories'];assert len(vocabulary)==1203
    objects={};draws=[];root_count=Counter();leaf_count=Counter()
    with gzip.open(ROOT/plan['source_curriculum'],'rt') as file:
        for line in file:
            row=json.loads(line)
            if row['supervision']['tracking_task_bucket']!='ovmot':continue
            info=row['supervision'];aid=info['annotation_id'];label=labels[aid];frame=frames[info['image_id']]
            assert label['image_id']==info['image_id'] and not info['Novel_GT_supervision']
            eligible={r['annotation_id']:r['vocabulary_index'] for r in frame['eligible_Base_annotation_references']}
            target=info['Base_vocabulary_index'];assert eligible[aid]==target and vocabulary[target]['frequency'] in ('c','f')
            match=re.fullmatch(r'base175_(\d+)_(\d+)_(group|class)_(\d+)',info['source_row_id']);assert match
            assert int(match[1])==frame['image_id']
            local=int(match[4]);x,y,w,h=label['bbox'];box=[x,y,x+w,y+h]
            media=row['media'][0];assert media['groundjev_native_category_crops'][local]==box
            assert Path(media['path']).resolve()==Path(frame['image_path']).resolve()
            assert Path(media['path']).is_file()
            value=dict(annotation_id=aid,image_id=frame['image_id'],video_id=frame['video_id'],
                image_path=frame['image_path'],image_size=frame['image_size'],box=box,
                class_index=target,Base_only=True,teacher_GT_proposal_not_native_DLM=True)
            if aid in objects:assert objects[aid]==value
            else:objects[aid]=value
            if row['task']=='category_group':
                root_count[aid]+=1;draws.append(aid);assert row['target']=={str(target//128):1.}
            else:
                assert row['task']=='category_leaf' and row['target']=={str(target%128):1.}
                leaf_count[aid]+=1
    assert root_count==leaf_count and len(draws)==plan['expected_sampled_root_questions']
    classes=Counter(objects[aid]['class_index'] for aid in draws)
    assert len(classes)==plan['expected_Base_classes']
    ordered=[]
    for index,aid in enumerate(sorted(objects)):
        ordered.append(dict(objects[aid],feature_row=index,sampled_draws=root_count[aid]))
    with (out/'objects.jsonl').open('w') as file:
        for row in ordered:file.write(json.dumps(row)+'\n')
    (out/'sampled_draws.json').write_text(json.dumps(draws)+'\n')
    report=dict(status='EXISTING_BASE175_TEACHER_OBJECTS_VALIDATED_NATIVE_FEATURES_NOT_YET_COLLECTED',
        sampled_root_and_leaf_questions_each=len(draws),unique_annotation_objects=len(ordered),
        unique_physical_images=len({row['image_path'] for row in ordered}),supervised_Base_classes=len(classes),
        full_input_vocabulary=1203,draws_per_class=dict(classes),all_original_GT_boxes_classes_and_row_pairs_exact=True,
        approximate_FP32_bank_bytes=len(ordered)*2560*4,new_model_or_GPU_or_optimizer_calls=0,full_goal_complete=False)
    (out/'object_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<plan['new_component_cap_bytes']
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
