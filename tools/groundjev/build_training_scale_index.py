"""Index broader legal training sources without proposals, features or SFT claims."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from locatemot.unified.data.legal_scope import LEGAL_SPLITS,LEGAL_SPLIT_VERSION,assert_legal_video


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert Path(plan['project_root'])==Path('/data2/user/LocateMOT')
    assert plan['combined_registered_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    args.out.mkdir(parents=True,exist_ok=False)
    counts,missing,sequence_counts=Counter(),Counter(),Counter()
    def available(path,task):
        exists=path.is_file();counts[task+'_indexed_images']+=1
        missing[task]+=not exists
        return exists
    def emit(output,row):
        output.write(json.dumps(row,ensure_ascii=False)+'\n')
    mot_details={}
    with (args.out/'mot_frames.jsonl').open('w') as output:
        for name in plan['mot_protocols']:
            path=Path(plan['protocol_root'])/name
            protocol=json.loads(path.read_text());seen=set();videos=set()
            for sequence in protocol['sequences']:
                assert set(sequence['train_frames']).isdisjoint(sequence.get('validation_frames',[]))
                assert set(sequence['train_frames']).isdisjoint(sequence.get('calibration_frames',[]))
                for image in sequence['train_frames']:
                    key=sequence['sequence'],image;assert key not in seen;seen.add(key)
                    file=Path(sequence['image_dir'])/image;videos.add(sequence['sequence'])
                    emit(output,{'dataset':protocol['dataset'],'task':'mot','sequence':sequence['sequence'],
                        'frame':int(Path(image).stem),'image_path':str(file),'fps':sequence['fps'],
                        'annotation_source':sequence['gt_path'],'frozen_protocol':str(path),
                        'split':'frozen train_frames only','image_available':available(file,'mot')})
            mot_details[protocol['dataset']]={'frames':len(seen),'videos':len(videos)}
            sequence_counts['mot']+=len(videos)
    rmot_details={}
    with (args.out/'rmot_videos.jsonl').open('w') as videos,(args.out/'rmot_queries.jsonl').open('w') as queries:
        for video in LEGAL_SPLITS[plan['RMOT_dataset']]['fit']:
            assert_legal_video(video,dataset=plan['RMOT_dataset'])
            directory=Path(plan['RMOT_image_root'])/video
            images=sorted(directory.glob('*.png'))
            assert images,'No local physical-frame listing for registered fit video '+video
            frame_ids={int(p.stem) for p in images}
            query_files=sorted((Path(plan['RMOT_expression_root'])/video).glob('*.json'))
            assert query_files
            positive,nonempty_ignore,query_missing=0,0,set()
            for file in query_files:
                query=json.loads(file.read_text())
                positive_frames={int(f) for f,ids in query['label'].items() if ids}
                absent=positive_frames-frame_ids;query_missing.update(absent)
                nonempty_ignore+=bool(query.get('ignore'))
                emit(queries,{'dataset':plan['RMOT_dataset'],'video':video,'split':'fit',
                    'expression_file':str(file),'sentence':query['sentence'],
                    'positive_annotation_frames':len(positive_frames),
                    'local_physical_frames':len(images),'nonempty_ignore_payload':bool(query.get('ignore')),
                    'positive_labeled_frames_missing_images':sorted(absent),
                    'target_annotation_payload_remains_in_original_release':True})
                positive+=len(positive_frames)
            local=[{'frame':int(p.stem),'image_path':str(p),'image_available':available(p,'rmot')} for p in images]
            emit(videos,{'dataset':plan['RMOT_dataset'],'task':'rmot','video':video,
                'legal_split_version':LEGAL_SPLIT_VERSION,'split':'fit','fps':10.,
                'frames':local,'expression_files':[str(p) for p in query_files],
                'label_directory':str(Path(plan['RMOT_label_root'])/video),
                'ignore_rules_must_be_read_from_release_before_generating_supervision':True,
                'empty_target_frames_are_not_filtered':True})
            rmot_details[video]={'local_physical_frames':len(images),'expression_files':len(query_files),
                'query_positive_annotation_frames':positive,'queries_with_nonempty_ignore':nonempty_ignore,
                'positive_labeled_frames_missing_images':sorted(query_missing)}
            counts['rmot_expressions']+=len(query_files);sequence_counts['rmot']+=1
    vocabulary=json.loads(Path(plan['input_vocabulary']).read_text())['categories']
    assert len(vocabulary)==1203
    class_index={c['synset']:i for i,c in enumerate(vocabulary)}
    annotation=json.loads(Path(plan['TAO_train_annotation']).read_text())
    categories={c['id']:c for c in annotation['categories']}
    image_info={i['id']:i for i in annotation['images']}
    by_image=defaultdict(list);classes=Counter();class_images=defaultdict(set);class_videos=defaultdict(set)
    excluded=Counter()
    for label in annotation['annotations']:
        category=categories[label['category_id']]
        if category.get('frequency') not in ('c','f'):
            excluded['non_Base_annotations_not_supervised']+=1;continue
        if category.get('synset') not in class_index:
            excluded['outside_frozen_vocabulary_not_supervised']+=1;continue
        index=class_index[category['synset']]
        assert vocabulary[index]['frequency'] in ('c','f')
        by_image[label['image_id']].append({'annotation_id':label['id'],'vocabulary_index':index})
        classes[index]+=1;class_images[index].add(label['image_id']);class_videos[index].add(image_info[label['image_id']]['video_id'])
    tao_videos=set()
    with (args.out/'tao_train_frames.jsonl').open('w') as output:
        for image in sorted(annotation['images'],key=lambda i:(i['video_id'],i['frame_index'])):
            assert Path(image['file_name']).parts[0]=='train'
            file=Path(plan['TAO_image_root'])/image['file_name'];targets=by_image[image['id']]
            tao_videos.add(image['video_id']);counts['tao_Base_target_references']+=len(targets)
            counts['tao_zero_eligible_Base_target_images']+=not targets
            emit(output,{'task':'ovmot','split':'train','video_id':image['video_id'],'image_id':image['id'],
                'frame_index':image['frame_index'],'image_path':str(file),
                'image_size':[image['width'],image['height']],'image_available':available(file,'tao'),
                'eligible_Base_annotation_references':targets,'annotation_source':plan['TAO_train_annotation'],
                'input_vocabulary':plan['input_vocabulary'],'all1203_input_classes_retained':True,
                'zero_Base_labels_do_not_mean_background_or_negative_Novel_labels':True})
    sequence_counts['tao']=len(tao_videos)
    coverage=[{'vocabulary_index':i,'name':vocabulary[i]['name'],'synset':vocabulary[i]['synset'],
        'frequency':vocabulary[i]['frequency'],'training_targets':classes[i],
        'training_images':len(class_images[i]),'training_videos':len(class_videos[i])} for i in sorted(classes)]
    (args.out/'Base_class_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    size=sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    assert size<plan['metadata_export_budget_bytes']
    report={'status':'BROADER_LEGAL_TRAINING_SOURCE_INDEX_COMPLETE_NOT_SFT_OR_TRAINING',
        'plan':str(args.plan),'counts':dict(counts),'missing_indexed_image_counts':dict(missing),
        'video_counts':dict(sequence_counts),'MOT':mot_details,'RMOT':rmot_details,
        'TAO_training_Base_classes_with_targets':len(coverage),'TAO_non_supervised_annotations':dict(excluded),
        'all_TAO_training_annotation_images_retained':len(annotation['images']),
        'all_RMOT_expressions_in_legal_fit_videos_retained':True,
        'RMOT_physical_frames_are_local_listing_not_external_sequence_completeness_proof':True,
        'no_GT_evaluation_or_forbidden_video_labels_read':True,'Novel_GT_supervised_or_negative':False,
        'metadata_bytes_before_summary':size,'registered_metadata_cap_bytes':plan['metadata_export_budget_bytes'],
        'combined_registered_upper_bytes':plan['combined_registered_upper_bytes'],
        'new_native_feature_cache_bytes':0,'new_model_calls':False,'new_training_steps':0,
        'ready_as_existing_SFT_training_JSONL':False,'new_training_protocol_or_gpu_job_started':False,
        'full_goal_complete':False}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())<plan['metadata_export_budget_bytes']
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
