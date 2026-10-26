"""Audit existing legal TAO Base and Refer-KITTI physical annotations, no fit."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PIL import Image
from locatemot.unified.data.legal_scope import LEGAL_SPLITS
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json,digest


def main():
    annotation=Path('/data2/user/LocateMOT/data/TAO-Amodal/annotations/train.json')
    tao=json.loads(annotation.read_text());categories={c['id']:c for c in tao['categories']}
    by_image=defaultdict(list);base=Counter();novel_excluded=0
    for item in tao['annotations']:
        category=categories[item['category_id']]
        if category['frequency']=='r':novel_excluded+=1;continue
        if item.get('iscrowd',0):continue
        by_image[item['image_id']].append(item);base[category['name']]+=1
    example=None
    for image in tao['images']:
        if not by_image[image['id']]:continue
        path=next((Path(f'/data{i}/user/TAO-Amodal/frames')/image['file_name'] for i in range(4)
                   if (Path(f'/data{i}/user/TAO-Amodal/frames')/image['file_name']).exists()),None)
        if path is not None:
            boxes=[]
            for item in by_image[image['id']]:
                x,y,w,h=item['bbox'];boxes.append([x,y,x+w,y+h])
            example={'dataset':'tao_base','sequence':str(image['video_id']),'category':'Base objects',
                     'image_path':str(path),'frame':image['frame_index'],'image_size':[image['width'],image['height']],
                     'boxes':boxes,'ignored':[],'exhaustive':False,'source_split':'TRAIN',
                     'category_ids':[a['category_id'] for a in by_image[image['id']]],'novel_GT_used':False}
            break
    refer=Path('/data2/user/LocateMOT/data/refer_kitti_v2');images=Path('/data2/user/LocateMOT/data/kitti_tracking_training/image_02')
    stats=Counter();video_stats={};refer_example=None
    for video in LEGAL_SPLITS['refer_kitti_v2']['fit']:
        labels={}
        for p in sorted((refer/'labels_with_ids/image_02'/video).glob('*.txt')):
            frame=int(p.stem);current={}
            for line in p.read_text().splitlines():
                fields=line.split();identity=int(fields[1]);assert identity not in current
                current[identity]=list(map(float,fields[2:6]));stats['physical_objects']+=1
            labels[frame]=current;stats['physical_frames']+=1
        local=Counter()
        for p in sorted((refer/'expression'/video).glob('*.json')):
            query=json.loads(p.read_text());local['expressions']+=1
            ignore={int(v) for v in query.get('ignore',[])}
            if ignore:local['expressions_with_ignored_frames']+=1
            assert query.get('sentence')
            for f,identities in query.get('label',{}).items():
                frame=int(f)
                if frame in ignore:local['ignored_query_frames']+=1;continue
                missing=set(map(int,identities))-set(labels.get(frame,{}))
                local['missing_expression_target_ids']+=len(missing);local['query_frames']+=1
                if missing or not identities or refer_example is not None:continue
                path=images/video/(f'{frame:06d}.png')
                if not path.exists():local['missing_images']+=1;continue
                with Image.open(path) as im:width,height=im.size
                # Official RMOT KITTI branch uses normalized top-left xywh;
                # do not select representation based on this local path's case.
                boxes=[[x*width,y*height,(x+w)*width,(y+h)*height] for x,y,w,h in labels[frame].values()]
                refer_example={'dataset':'refer_physical','sequence':video,'category':'all physical objects',
                    'image_path':str(path),'frame':frame,'image_size':[width,height],'boxes':boxes,'ignored':[],
                    'exhaustive':False,'source_split':'TRAIN_fit','physical_IDs':sorted(labels[frame]),
                    'example_expression':query['sentence'],'example_expression_target_IDs':identities,
                    'GT_format':'official KITTI normalized top-left xywh','expression_file':str(p),
                    'partial_annotation_caution':'All supplied physical boxes included; exhaustiveness not asserted for all visible semantic categories'}
        video_stats[video]=dict(local);stats.update(local)
    assert example and refer_example
    write_json(RUN/'data/auxiliary_rows.json',[example,refer_example])
    receipt={'status':'EXISTING_TRAIN_INTERFACE_AUDIT','TAO':{'annotation':str(annotation),'SHA256':digest(annotation),
        'TRAIN_images':len(tao['images']),'TRAIN_videos':len(tao['videos']),
        'Base_annotated_images':len([k for k,v in by_image.items() if v]),'Base_classes_with_annotations':len(base),
        'Base_annotation_counts':dict(base),'Novel_annotations_excluded':novel_excluded,'non_exhaustive':True,
        'optimizer_use_this_pilot':False,'example_category_IDs':example['category_ids']},
        'Refer_KITTI':{'legal_fit_videos':list(LEGAL_SPLITS['refer_kitti_v2']['fit']),
        'counts':dict(stats),'per_video':video_stats,'optimizer_use_this_pilot':False,
        'coordinate_reference':'https://github.com/wudongming97/RMOT/blob/master/datasets/refer_kitti.py',
        'coordinate_contract':'KITTI normalized x1,y1,w,h; convert using actual image width/height',
        'historical_builder_boundary':'Old clean builder skips entire expressions when ignore is nonempty; preserve historical corpus, future adapter must ignore only listed frames',
        'physical_vs_expression_targets':{'physical_objects':len(refer_example['boxes']),'expression_targets':len(refer_example['example_expression_target_IDs'])}},
        'no_new_downloads':True,'future_training_requires_negative_eligibility':True,
        'SAM_optional':'No registered ready SAM3.1 checkpoint/environment; optional comparison skipped without downloading weights'}
    write_json(RUN/'audits/auxiliary_data.json',receipt)
    print(json.dumps({'TAO_Base_classes':len(base),'TAO_Base_annotations':sum(base.values()),'TAO_Novel_excluded':novel_excluded,'Refer_counts':dict(stats),'examples':str(RUN/'data/auxiliary_rows.json')}),flush=True)


if __name__=='__main__':main()
