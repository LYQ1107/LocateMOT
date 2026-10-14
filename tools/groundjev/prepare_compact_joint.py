"""Mixed native MOT plus explicitly declared fit/Base teacher semantic data."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/data2/user/LocateMOT/data')
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.feature_client import NativeFeatureClient
    from locatemot.groundjev.text_features import OneJevTextFeatures
    from locatemot.groundjev.decision import overlap
    from locatemot.unified.data.legal_scope import assert_legal_video,LEGAL_SPLITS,LEGAL_SPLIT_VERSION
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'outputs/groundjev/compact/joint_pack_v1')
    out=parser.parse_args().out;out.mkdir(parents=True,exist_ok=False)
    state={'status':'BUILDING_DECLARED_JOINT_SEMANTIC_TRAIN_DATA','frames':0,'teacher_examples':0,
        'MOT_native_examples':149,'RMOT_split':'fit-v2','TAO_split':'train-Base-only',
        'Novel_GT_supervised':False,'GT_labels_in_input':False,'new_proposal_source':'teacherGT; not native detector'}
    def save():
        (out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    save()
    text=OneJevTextFeatures();vocabulary=json.loads((DATA.parent/'outputs/unified/u2/protocol/lvis_v1_vocabulary.json').read_text())['categories']
    names=[c['name'].replace('_',' ') for c in vocabulary];class_by_synset={c['synset']:i for i,c in enumerate(vocabulary)}
    class_text=text.vocabulary(names);person=class_by_synset['person.n.01']
    client=NativeFeatureClient('http://127.0.0.1:8111')
    native=torch.load(ROOT/'outputs/groundjev/compact/native_pack_v1/examples.pt',weights_only=True)
    source_rows=[json.loads(s) for s in (ROOT/'outputs/groundjev/training_data/native_ground_own_history_v1/train.jsonl').open()]
    class_labels={}
    for row in source_rows:
        if row['task']=='identity':class_labels[row['id']]=person if 'IGNORE' not in row['target'] else -100
    mot_spec=text.specification({'task':'mot','vocabulary':['person'],'expression':None})
    for e in native:
        stem=e['id'].split('_withheld',1)[0]
        e.update(spec_text=mot_spec,class_target=torch.tensor([class_labels[stem+f'_match_{i}'] for i in range(len(e['candidates']))]),
            membership_target=torch.ones(len(e['candidates'])),native_teacher=True)
    examples=list(native);frames=[]
    def add_clip(name,task,observations,specification):
        memory={};timestamp_last=None;spec_text=text.specification(specification)
        for path,number,timestamp,targets in observations:
            with Image.open(path) as raw:image=raw.convert('RGB')
            candidates=[{'box':t['box'],'label':'object','native_score':None} for t in targets]
            enriched,meta=client(image,candidates,image_path=path)
            current=torch.tensor([c['native_region_feature'] for c in enriched]).reshape(len(targets),2560)
            predicted={t:[x+v*(timestamp-memory[t]['timestamp']) for x,v in zip(memory[t]['box'],memory[t]['velocity'])] for t in memory}
            for offset in range(0,len(targets),16):
                group=targets[offset:offset+16];n=len(group)
                ranked=sorted(memory,key=lambda identity:(-max(overlap(c['box'],predicted[identity]) for c in group),identity))[:32]
                m=len(ranked);history=torch.stack([torch.stack([memory[t][k] for k in ('anchor','recent','long')]) for t in ranked]) if m else torch.empty(0,3,2560)
                pair=torch.tensor([[[round(overlap(c['box'],predicted[t]),4),round(((c['box'][0]+c['box'][2]-predicted[t][0]-predicted[t][2])**2+(c['box'][1]+c['box'][3]-predicted[t][1]-predicted[t][3])**2)**.5/(2*(image.width**2+image.height**2)**.5),4)] for t in ranked] for c in group]).reshape(n,m,2)
                target=torch.tensor([ranked.index(c['identity']) if c['identity'] in ranked else m+2 if c['identity'] in memory else m for c in group])
                present={c['identity'] for c in targets}
                examples.append({'id':name+f'_{number}_{offset}','candidates':current[offset:offset+n],
                    'history':history,'candidate_geo':torch.tensor([[c['box'][0]/image.width,c['box'][1]/image.height,c['box'][2]/image.width,c['box'][3]/image.height] for c in group]),
                    'history_geo':torch.tensor([[*(x/s for x,s in zip(predicted[t],(image.width,image.height,image.width,image.height))),memory[t]['age']/100,(timestamp_last-memory[t]['timestamp'])/3,(timestamp-memory[t]['timestamp'])/3,1.] for t in ranked]).reshape(m,8),
                    'pair_geometry':pair,'task_index':1 if task=='rmot' else 2,'target':target,
                    'update_target':torch.ones(n),'reliability_target':torch.full((n,),3),
                    'life_target':torch.tensor([2 if t in present and timestamp_last>memory[t]['timestamp'] else 0 if t in present else 1 for t in ranked]),
                    'spec_text':spec_text,'class_target':torch.tensor([c['class'] for c in group]),
                    'membership_target':torch.tensor([float(c['member']) for c in group]),
                    'native_teacher':False,'source_sequence':name,'frame':number,'candidate_boxes':[c['box'] for c in group],
                    'proposal_source':'teacherGT','history_source':'causal teacherGT observations',
                    'update_reliability_source':'ideal teacher observation proxy; not native detection confidence'})
            for t,feature in zip(targets,current):
                identity=t['identity']
                if identity not in memory:memory[identity]={'anchor':feature,'age':0,'velocity':[0.]*4}
                else:memory[identity]['velocity']=[(x-y)/(timestamp-memory[identity]['timestamp']) for x,y in zip(t['box'],memory[identity]['box'])]
                memory[identity].update(recent=feature,long=feature,box=t['box'],timestamp=timestamp,age=memory[identity]['age']+1)
            memory={t:v for t,v in memory.items() if timestamp-v['timestamp']<=3}
            timestamp_last=timestamp
            frames.append({'scope':name,'task':task,'frame':number,'image_path':str(path),'teacher_GT_regions':len(targets)})
            state.update(frames=len(frames),teacher_examples=len(examples)-len(native));save()
    queries=[('0000','automobiles-colored-silver-are-located-on-the-right'),('0000','the-right-side-features-women'),
        ('0002','automobiles-applying-the-brakes'),('0002','standing-women'),
        ('0003','automobiles-are-driving-in-the-same-direction-as-our-automobile'),('0003','automobiles-are-moving-in-the-same-direction-as-our-vehicle')]
    for video,query in queries:
        assert_legal_video(video,dataset='refer_kitti_v2');assert video in LEGAL_SPLITS['refer_kitti_v2']['fit']
        expression=json.loads((DATA/'refer_kitti_v2/expression'/video/(query+'.json')).read_text());assert not expression.get('ignore')
        first=min(map(int,expression['label']));observations=[]
        for number in range(first,first+8):
            path=DATA/'kitti_tracking_training/image_02'/video/f'{number:06d}.png';assert path.is_file()
            with Image.open(path) as im:w,h=im.size
            targets=[];labels=DATA/'refer_kitti_v2/labels_with_ids/image_02'/video/f'{number:06d}.txt'
            for line in labels.read_text().splitlines() if labels.exists() else []:
                f=line.split();identity=int(f[1]);x,y,bw,bh=map(float,f[2:6]);targets.append({'identity':identity,'box':[x*w,y*h,(x+bw)*w,(y+bh)*h],
                    'class':-100,'member':identity in expression['label'].get(str(number),[])})
            assert set(expression['label'].get(str(number),[]))<={c['identity'] for c in targets}
            observations.append((path,number,number/10,targets))
        add_clip('rmot_'+video+'_'+query,'rmot',observations,{'task':'rmot','vocabulary':['person','car'],'expression':expression['sentence']})
    annotation=json.loads(Path('/data2/user/TAO-Amodal/annotations/train_with_freeform.json').read_text());categories={c['id']:c for c in annotation['categories']};by_image=defaultdict(list)
    for a in annotation['annotations']:
        category=categories[a['category_id']]
        if category.get('frequency') in ('c','f') and category['synset'] in class_by_synset:by_image[a['image_id']].append(a)
    for video in (0,1,7,9,13,16,38,47):
        images=sorted([i for i in annotation['images'] if i['video_id']==video],key=lambda i:i['frame_index'])[:8];observations=[]
        for step,im in enumerate(images):
            path=DATA/'TAO-Amodal/frames'/im['file_name'];assert path.is_file();targets=[]
            for a in by_image[im['id']]:
                x,y,w,h=a['bbox'];targets.append({'identity':a['track_id'],'box':[x,y,x+w,y+h],
                    'class':class_by_synset[categories[a['category_id']]['synset']],'member':True})
            observations.append((path,im['frame_index']+1,float(step),targets))
        add_clip('tao_train_'+str(video),'ovmot',observations,{'task':'ovmot','vocabulary':names,'expression':None})
    torch.save({'examples':examples,'class_text':class_text,'vocabulary':names},out/'examples.pt')
    size=(out/'examples.pt').stat().st_size;assert size<25*2**30
    (out/'frame_scope.json').write_text(json.dumps(frames,indent=2)+'\n')
    state.update(status='DECLARED_JOINT_SEMANTIC_TRAIN_PACK_COMPLETE',examples=len(examples),feature_file_bytes=size,
        semantic_encoder='original real OneJev frozen input token mean, not language-decoder reasoning',
        text_embedding_shape=list(text.embedding.shape),class_choices=len(names),KITTI_legal_split_version=LEGAL_SPLIT_VERSION,
        teacher_empty_frames=sum(f['teacher_GT_regions']==0 for f in frames),teacher_proposal_stage_not_native_rollout=True);save()


if __name__=='__main__':main()
