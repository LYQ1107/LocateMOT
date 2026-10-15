"""All indexed train-Base objects as real native category questions, no PNG cache."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    from locatemot.groundjev.category import NativeCategoryHead
    from tools.groundjev.streamed_category_media import RECIPE_KEY,IO_PROTOCOL
    from train import common
    from qev.prompt import render_question
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    index=ROOT/plan['source_index']
    proof=json.loads((index/'validation.json').read_text())
    assert proof['status']=='ACTUAL_LEGAL_TRAINING_INDEX_COUNTS_AND_ALL_BASE_TARGET_REFERENCES_VALIDATED'
    annotation=json.loads(Path(plan['annotation']).read_text());labels={a['id']:a for a in annotation['annotations']}
    categories=json.loads(Path(plan['vocabulary']).read_text())['categories']
    vocabulary=[c['name'].replace('_',' ') for c in categories];assert len(vocabulary)==1203
    args.out.mkdir(parents=True,exist_ok=False)
    class RecipeHead(NativeCategoryHead):
        def crop_sheet(self,image,candidates):
            return {'type':'image','path':image,'w':4*192,
                'h':max(1,math.ceil(len(candidates)/4))*216,
                RECIPE_KEY:[list(c['box']) for c in candidates]}
    head=RecipeHead(None,group_size=plan['group_size'],batch_candidates=plan['batch_candidates'])
    counts,classes,videos=Counter(),Counter(),set()
    expanded_bytes=0
    # Compression changes storage only. No blank/substitute pixels are made:
    # crop recipes are evaluated on the real image by the native crop method.
    with gzip.open(args.out/'train.jsonl.gz','wt',encoding='utf-8',compresslevel=6) as output, \
            (args.out/'frame_scope.jsonl').open('w') as scope:
        for line in (index/'tao_train_frames.jsonl').open():
            frame=json.loads(line);targets=frame['eligible_Base_annotation_references']
            assert frame['split']=='train' and frame['image_available']
            counts['registered_images']+=1;videos.add(frame['video_id'])
            counts['zero_Base_images_no_negative_category_labels']+=not targets
            batches=0
            for offset in range(0,len(targets),head.batch_candidates):
                selected=targets[offset:offset+head.batch_candidates]
                boxes=[];indices=[]
                for target in selected:
                    label=labels[target['annotation_id']];assert label['image_id']==frame['image_id']
                    x,y,w,h=label['bbox'];boxes.append({'box':[x,y,x+w,y+h]})
                    index_value=target['vocabulary_index'];assert categories[index_value]['frequency'] in ('c','f')
                    indices.append(index_value);classes[index_value]+=1
                request,media,groups=head.build_root_request(frame['image_path'],boxes,vocabulary)
                roots={i:str(target//head.group_size) for i,target in enumerate(indices)}
                leaf=head.build_leaf_request(request.state,groups,roots)
                for i,target in enumerate(indices):
                    for kind,req,key,answer in [('category_group',request,f'group_{i}',roots[i]),
                            ('category_leaf',leaf,f'class_{i}',str(target%head.group_size))]:
                        question=req.questions[key]
                        common.target_vector({answer:1.},render_question('q',question).labels)
                        row={'id':f'base175_{frame["image_id"]}_{offset}_{key}',
                            'split':'train','source':f'tao_train_{frame["video_id"]}','task':kind,
                            'state':req.state,'question':question.model_dump(exclude_none=True),
                            'target':{answer:1.},'media':[media[0]['image']],
                            'supervision':{'frame':frame['frame_index'],'image_id':frame['image_id'],
                                'annotation_id':selected[i]['annotation_id'],'Base_vocabulary_index':target,
                                'candidate_source':'registered train-Base GT teacher crops',
                                'GT_class_or_identity_in_state':False,'Novel_GT_supervision':False,
                                'all1203_input_categories_retained':True,'unknown_supervision':False,
                                'tracking_task_bucket':'ovmot','input_io_protocol':IO_PROTOCOL,
                                'leaf_teacher_forcing':'conditional correct root for SFT; native model chooses root at runtime',
                                'native_DLM_or_causal_actor_history_claim':False}}
                        text=json.dumps(row,ensure_ascii=False)+'\n';output.write(text)
                        expanded_bytes+=len(text.encode());counts[kind]+=1
                counts['Base_objects']+=len(selected);batches+=1
            counts['crop_sheet_recipes']+=batches
            scope.write(json.dumps({'image_id':frame['image_id'],'video_id':frame['video_id'],
                'image_path':frame['image_path'],'Base_objects':len(targets),'crop_sheet_recipes':batches})+'\n')
    assert counts['Base_objects']==plan['eligible_objects']==52978
    assert counts['category_group']==counts['category_leaf']==52978 and len(classes)==175
    size=sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    assert size<plan['new_export_budget_bytes']
    report={'status':'ALL175_BASE_NATIVE_CATEGORY_GZIP_COMPONENT_PREPARED_NOT_TRAINED',
        'counts':dict(counts),'rows':105956,'classes_with_targets':len(classes),
        'training_videos':len(videos),'input_vocabulary_classes':1203,'class_counts':dict(classes),
        'uncompressed_JSONL_bytes':expanded_bytes,'export_bytes_before_summary':size,
        'compressed_JSONL_bytes':(args.out/'train.jsonl.gz').stat().st_size,
        'input_io_protocol':IO_PROTOCOL,'generated_PNG_or_native_features_bytes':0,
        'new_model_calls_or_training_steps':0,'source_plan':str(args.plan),
        'combined_registered_upper_bytes':plan['combined_registered_upper_bytes'],
        'GT_teacher_proposals_explicit_not_native_DLM':True,
        'leaf_teacher_forcing_matches_existing_category_SFT':True,
        'Novel_or_evaluation_GT_supervision':False,'unknown_or_Novel_negative_labels_added':False,
        'all_zero_Base_images_registered_without_category_labels':True,
        'role':'Category component for a future balanced MOT/RMOT/OVMOT mixture; not a category-only optimizer launch.',
        'native_pixels_questions_and_processor_equivalence':'pending real-image regression',
        'full_goal_complete':False}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())<plan['new_export_budget_bytes']
    print(json.dumps({k:v for k,v in report.items() if k!='class_counts'},indent=2),flush=True)


if __name__=='__main__':
    main()
