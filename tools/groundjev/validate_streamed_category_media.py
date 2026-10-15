"""Real-pixel, native-question and processor equivalence for streamed Base data."""
import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    import torch
    from PIL import Image
    from train import common,mm
    from locatemot.groundjev.category import NativeCategoryHead
    from tools.groundjev.streamed_category_media import install,RECIPE_KEY,IO_PROTOCOL
    parser=argparse.ArgumentParser()
    parser.add_argument('--component',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((args.component/'summary.json').read_text())
    plan=json.loads(Path(summary['source_plan']).read_text())
    assert summary['rows']==105956 and summary['classes_with_targets']==175
    native_rows,native_media=install();mm.set_root(ROOT)
    legacy=ROOT/'outputs/groundjev/training_data/native_ground_own_history_v1/train.jsonl'
    assert common.load_rows(legacy,limit=2)==native_rows(legacy,limit=2)
    for row in native_rows(legacy,limit=2):
        old=native_media(row['media']);new=mm.load_media(row['media'])
        assert not old[1] and not new[1]
        assert [im.tobytes() for im in old[0]]==[im.tobytes() for im in new[0]]
    source=ROOT/'outputs/groundjev/compact/joint_pack_v3'
    scope={(r['scope'],r['frame']):r for r in json.loads((source/'frame_scope.json').read_text()) if r['task']=='ovmot'}
    pack=torch.load(source/'examples.pt',map_location='cpu',weights_only=True)
    head=NativeCategoryHead(None)
    prior_objects,prior_sheets=0,0
    for example in pack['examples']:
        if example['task_index']!=2:
            continue
        frame=scope[example['source_sequence'],example['frame']]
        boxes=example['candidate_boxes']
        for offset in range(0,len(boxes),head.batch_candidates):
            selected=boxes[offset:offset+head.batch_candidates]
            old_path=ROOT/'outputs/groundjev/training_data/onejev_base_categories_v1/evidence'/f"{example['id']}_{offset}.png"
            with Image.open(old_path) as image:
                old=image.convert('RGB')
            item={'type':'image','path':frame['image_path'],'w':old.width,'h':old.height,RECIPE_KEY:selected}
            images,_,_=mm.load_media([item])
            assert images[0].size==old.size and images[0].tobytes()==old.tobytes()
            prior_objects+=len(selected);prior_sheets+=1
    assert prior_objects==215
    annotation=json.loads(Path(plan['annotation']).read_text());labels={a['id']:a for a in annotation['annotations']}
    source_categories={c['id']:c for c in annotation['categories']}
    categories=json.loads(Path(plan['vocabulary']).read_text())['categories']
    vocabulary=[c['name'].replace('_',' ') for c in categories]
    image_by_id={i['id']:i for i in annotation['images']}
    expected_ids={a['annotation_id'] for line in (ROOT/plan['source_index']/'tao_train_frames.jsonl').open()
        for a in json.loads(line)['eligible_Base_annotation_references']}
    counts=Counter();seen={'category_group':set(),'category_leaf':set()};representatives={};first=[]
    with gzip.open(args.component/'train.jsonl.gz','rt',encoding='utf-8') as rows:
        for line in rows:
            row=json.loads(line);family=row['task'];target=row['supervision']['Base_vocabulary_index']
            label=labels[row['supervision']['annotation_id']];category=source_categories[label['category_id']]
            assert category['synset']==categories[target]['synset'] and category['frequency'] in ('c','f')
            assert categories[target]['frequency'] in ('c','f')
            assert label['image_id']==row['supervision']['image_id']
            assert label['id'] not in seen[family];seen[family].add(label['id'])
            i=int(row['id'].rsplit('_',1)[1]);x,y,w,h=label['bbox']
            assert row['media'][0][RECIPE_KEY][i]==[x,y,x+w,y+h]
            image=image_by_id[label['image_id']]
            assert row['media'][0]['path']==str(Path('/data2/user/LocateMOT/data/TAO-Amodal/frames')/image['file_name'])
            answer=str(target//128) if family=='category_group' else str(target%128)
            assert row['target']=={answer:1.} and answer in row['question']['criteria']
            counts[family]+=1
            if family=='category_group' and target not in representatives:
                representatives[target]=[row]
            elif family=='category_leaf' and len(representatives.get(target,[]))==1 and representatives[target][0]['supervision']['annotation_id']==label['id']:
                representatives[target].append(row)
            if len(first)<2:
                first.append(row)
    assert seen['category_group']==seen['category_leaf']==expected_ids and len(expected_ids)==52978
    assert counts=={'category_group':52978,'category_leaf':52978} and len(representatives)==175
    assert common.load_rows(args.component/'train.jsonl.gz',limit=2,split='train')==first
    processor_cases={}
    for target,pair in representatives.items():
        assert len(pair)==2
        row=pair[0];item=row['media'][0];i=int(row['id'].rsplit('_',1)[1])
        with Image.open(item['path']) as source_image:
            image=source_image.convert('RGB')
        request,media,groups=head.build_root_request(image,[{'box':box} for box in item[RECIPE_KEY]],vocabulary)
        leaf=head.build_leaf_request(request.state,groups,{i:str(target//128)})
        assert request.state==row['state'] and request.questions[f'group_{i}'].model_dump(exclude_none=True)==row['question']
        assert leaf.state==pair[1]['state'] and leaf.questions[f'class_{i}'].model_dump(exclude_none=True)==pair[1]['question']
        native=media[0]['image'];images,_,_=mm.load_media(row['media'])
        assert images[0].size==native.size and images[0].tobytes()==native.tobytes()
        processor_cases.setdefault(native.size,(pair,native))
    bundle=common.load_bundle(str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600'),
        device='cpu',dtype='float32',train=True,multimodal=True,load_model=False)
    tensor_checks=[]
    for size,(pair,image) in processor_cases.items():
        path=args.out/f'ordinary_native_sheet_{size[0]}x{size[1]}.png';image.save(path)
        for row in pair:
            ordinary=copy.deepcopy(row);ordinary['media']=[{'type':'image','path':str(path.resolve()),'w':size[0],'h':size[1]}]
            def render(r):
                return common.render_example(r,bundle.template_for(r),0,seed=0,augment=True).prompt
            eager,lazy=render(ordinary),render(row);assert eager==lazy
            a,_=mm.mm_inputs(bundle.processor,[eager],[ordinary['media']],'cpu')
            b,_=mm.mm_inputs(bundle.processor,[lazy],[row['media']],'cpu')
            assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
            assert bundle.counter.length(lazy,row['media'])==int(b['attention_mask'].sum())
            tensor_checks.append({'size':size,'family':row['task'],'all_processor_tensors_exact':True,
                'tokens':int(b['attention_mask'].sum()),'keys':list(b)})
    assert not torch.cuda.is_initialized()
    report={'status':'REAL_STREAMED_CATEGORY_PIXELS_QUESTIONS_AND_PROCESSOR_TENSORS_EXACT',
        'input_io_protocol':IO_PROTOCOL,'ordinary_legacy_rows_and_media_unchanged':True,
        'previous_Base_objects_pixels_exact':prior_objects,'previous_Base_crop_sheets_exact':prior_sheets,
        'all_rows_validated':sum(counts.values()),'all_Base_annotation_ids_paired_exact':52978,
        'actual_175_classes_native_questions_and_pixels_exact':len(representatives),
        'processor_tensors_exact_cases':tensor_checks,'GPU_or_model_weights_loaded':False,
        'new_training_steps':0,'full_goal_complete':False}
    (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
