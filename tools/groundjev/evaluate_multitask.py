"""Unmodified TrackEval/TETA evaluation of explicit legal RMOT/OVMOT scopes."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def check_scope(manifest, result, rows):
    if result['status'] not in ('VIDEO_INFERENCE_COMPLETE','VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION'):
        raise ValueError('Incomplete inference cannot be evaluated as a completed registered scope')
    if result['frames']!=len(manifest['images']) or len(rows)!=len(manifest['images']):
        raise ValueError('Every registered frame, including empty/rejected frames, is required')
    if result['scope']!=manifest['scope'] or result['task']!=manifest['task']:
        raise ValueError('Task or registered scope changed')
    if [r['frame'] for r in rows]!=manifest['frame_numbers'] or [r['image_path'] for r in rows]!=manifest['images']:
        raise ValueError('Predictions belong to different physical frames')
    expected={'task':manifest['task'],'vocabulary':manifest['vocabulary'],'expression':manifest.get('expression')}
    if result['specification']!=expected: raise ValueError('Evaluated language/vocabulary differs from the registered specification')


def rmot(manifest, rows, directory):
    import numpy as np
    from locatemot.unified.data.legal_scope import assert_legal_video, LEGAL_SPLITS, LEGAL_SPLIT_VERSION
    from locatemot.unified.runtime.u2_ordinary_evaluation import verify_trackeval_source
    assert_legal_video(manifest['sequence'],dataset=manifest['dataset'])
    if manifest['legal_split_version']!=LEGAL_SPLIT_VERSION or manifest['sequence'] not in LEGAL_SPLITS[manifest['dataset']]['validation']:
        raise ValueError('RMOT evaluation must use the frozen current validation split')
    source=verify_trackeval_source()
    import trackeval
    query=json.loads(Path(manifest['expression_file_for_evaluation_only']).read_text())
    if query['sentence']!=manifest['expression']: raise ValueError('Expression source changed')
    if query.get('ignore'): raise ValueError('This first registered RMOT protocol has empty release ignore fields')
    width,height=manifest['image_size']
    sequence='query_'+manifest['sequence']
    gt_dir=directory/'gt'/sequence; gt_dir.mkdir(parents=True,exist_ok=True)
    tracker_dir=directory/'trackers/groundjev/data';tracker_dir.mkdir(parents=True,exist_ok=True)
    gt_lines,pred_lines,target_counts=[],[],[]
    proposal_covered=0
    from locatemot.groundjev.decision import overlap
    for time_index,row in enumerate(rows,1):
        frame=row['frame']; target_ids=set(query['label'].get(str(frame),[]))
        available={}
        path=Path(manifest['label_dir_for_evaluation_only'])/f'{frame:06d}.txt'
        # The release omits YOLO files for empty frames. The expression defines
        # an empty target set there, so no target boxes need to be looked up.
        # A nonempty target set still requires its real release label file.
        for line in (path.read_text().splitlines() if target_ids else []):
            fields=line.split();identity=int(fields[1]);x,y,w,h=map(float,fields[2:6])
            # This release is normalized TOP-LEFT xywh, despite the
            # labels_with_ids directory name. Preserve its source extent.
            available[identity]=(x*width,y*height,w*width,h*height)
        if not target_ids<=available.keys(): raise ValueError(f'Release expression target lacks its GT box at frame {frame}')
        target_counts.append(len(target_ids))
        for identity in sorted(target_ids):
            x,y,w,h=available[identity]
            proposal_covered+=any(overlap([x,y,x+w,y+h],candidate['box'])>=.5 for candidate in row['candidates'])
            gt_lines.append(f'{time_index},{identity},{x:.6f},{y:.6f},{w:.6f},{h:.6f},1,1,1\n')
        for prediction in row['outputs']:
            x,y,x2,y2=prediction['box']
            pred_lines.append(f'{time_index},{prediction["track_id"]},{x:.6f},{y:.6f},{x2-x:.6f},{y2-y:.6f},1,-1,-1,-1\n')
    (gt_dir/'gt.txt').write_text(''.join(gt_lines))
    (gt_dir/'seqinfo.ini').write_text(f'[Sequence]\nname={sequence}\nimDir=img1\nframeRate=10\nseqLength={len(rows)}\nimWidth={width}\nimHeight={height}\nimExt=.png\n')
    (tracker_dir/f'{sequence}.txt').write_text(''.join(pred_lines))
    adapter=trackeval.datasets.MotChallenge2DBox({'GT_FOLDER':str(directory/'gt'),
        'TRACKERS_FOLDER':str(directory/'trackers'),'TRACKERS_TO_EVAL':['groundjev'],
        'TRACKER_SUB_FOLDER':'data','SKIP_SPLIT_FOL':True,'BENCHMARK':'MOT17',
        'PRINT_CONFIG':False,'DO_PREPROC':True,'SEQ_INFO':{sequence:len(rows)},
        'GT_LOC_FORMAT':'{gt_folder}/{seq}/gt.txt'})
    metrics=[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),trackeval.metrics.Identity({'PRINT_CONFIG':False})]
    evaluator=trackeval.Evaluator({'PRINT_CONFIG':False,'PRINT_RESULTS':False,'TIME_PROGRESS':False,
        'OUTPUT_SUMMARY':False,'OUTPUT_DETAILED':False,'PLOT_CURVES':False,'BREAK_ON_ERROR':True})
    raw,messages=evaluator.evaluate([adapter],metrics)
    if messages[adapter.get_name()]['groundjev']!='Success': raise RuntimeError('TrackEval failed')
    values=raw[adapter.get_name()]['groundjev'][sequence]['pedestrian']
    measured={k:float(np.mean(values['HOTA'][k])) for k in ('HOTA','DetA','AssA')}
    measured.update(IDF1=float(values['Identity']['IDF1']),IDSW=int(values['CLEAR']['IDSW']),
                    FP=int(values['CLEAR']['CLR_FP']),FN=int(values['CLEAR']['CLR_FN']))
    empty=[row for row,count in zip(rows,target_counts) if not count]
    return {'evaluator':'unmodified official TrackEval HOTA/CLEAR/Identity','evaluator_commit':source['commit'],
        'metrics':measured,'metric_scale':'fraction; counts for IDSW/FP/FN',
        'release_query_GT_boxes':sum(target_counts),'multi_target_frames':sum(n>1 for n in target_counts),
        'posthoc_proposal_coverage_IoU_05':{'covered':proposal_covered,'GT_boxes':sum(target_counts),
            'recall':proposal_covered/sum(target_counts) if sum(target_counts) else None,
            'scope':'evaluation-only target coverage; not HOTA; never used to correct runtime state'},
        'empty_target_frames':len(empty),'empty_target_frame_predictions':sum(len(r['outputs']) for r in empty),
        'GT_transport_class':'class1 is a query-target transport convention, including cars and people',
        'GT_policy':'only release expression target IDs; empty frames retained; exact release boxes without clipping',
        'release_box_format':'normalized top-left xywh',
        'ignore_policy':'registered expressions have empty release ignore fields',
        'official_complete_dataset_result':False}


def normalize_label(name):
    return ' '.join(name.lower().replace('_',' ').split())


def ovmot(manifest, rows, directory):
    import numpy as np
    source=Path('/data2/user/LocateMOT/third_party/TETA')
    sys.path.insert(0,str(source))
    from teta.datasets import TAO
    from teta.eval import Evaluator
    from teta.metrics import TETA
    source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    annotation=json.loads(Path(manifest['annotation_path_for_evaluation_only']).read_text())
    selected=set(manifest['image_ids'])
    annotations=[a for a in annotation['annotations'] if a['image_id'] in selected]
    track_ids={a['track_id'] for a in annotations}
    gt={k:annotation[k] for k in ('info','categories','licenses')}
    gt.update(videos=[v for v in annotation['videos'] if v['id']==manifest['video_id']],
              images=[i for i in annotation['images'] if i['id'] in selected],annotations=annotations,
              tracks=[t for t in annotation['tracks'] if t['id'] in track_ids])
    gt_path=directory/'gt.json';gt_path.write_text(json.dumps(gt))
    category_by_synset={c['synset']:c['id'] for c in annotation['categories']}
    aliases={}
    for category in annotation['categories']:
        for alias in [category['name']]+category.get('synonyms',[]):
            aliases[normalize_label(alias)]=category['id']
    vocabulary=json.loads(Path(manifest['vocabulary_path']).read_text())['categories']
    for category in vocabulary:
        if category['synset'] in category_by_synset:
            for alias in [category['name']]+category['synonyms']:
                aliases[normalize_label(alias)]=category_by_synset[category['synset']]
    predictions,unmapped=[],[]
    for row,image_id in zip(rows,manifest['image_ids']):
        for prediction in row['outputs']:
            category_id=aliases.get(normalize_label(prediction['label']),-1)
            if category_id==-1: unmapped.append({'frame':row['frame'],'label':prediction['label']})
            x,y,x2,y2=prediction['box']
            predictions.append({'image_id':image_id,'video_id':manifest['video_id'],
                'track_id':prediction['track_id'],'category_id':category_id,
                'bbox':[x,y,x2-x,y2-y], 'score':1.})
    tracker=directory/'trackers/groundjev/data';tracker.mkdir(parents=True,exist_ok=True)
    (tracker/'predictions.json').write_text(json.dumps(predictions))
    adapter=TAO({'GT_FOLDER':str(gt_path),'TRACKERS_FOLDER':str(directory/'trackers'),
        'TRACKERS_TO_EVAL':['groundjev'],'TRACKER_SUB_FOLDER':'data','PRINT_CONFIG':False,
        'SPLIT_TO_EVAL':'val','MAX_DETECTIONS':0})
    evaluator=Evaluator({'USE_PARALLEL':False,'BREAK_ON_ERROR':True,'RETURN_ON_ERROR':False,
        'LOG_ON_ERROR':str(directory/'teta_error.log'),
        'PRINT_CONFIG':False,'PRINT_RESULTS':False,'TIME_PROGRESS':False,
        'OUTPUT_SUMMARY':True,'OUTPUT_TEM_RAW_DATA':True,'OUTPUT_PER_SEQ_RES':True})
    _,messages=evaluator.evaluate([adapter],[TETA(exhaustive=False)])
    if messages[adapter.get_name()]['groundjev']!='Success': raise RuntimeError('Official TETA failed')
    with (directory/'trackers/groundjev/teta_summary_results.pth').open('rb') as handle: summary=pickle.load(handle)
    summary=summary.get('COMBINED_SEQ',summary)
    fields=TETA().float_array_fields
    overall={name:float(value) for name,value in zip(fields,summary['average']['TETA'][50])}
    groups={'base':[],'novel':[]}
    measured_categories=[]
    for category in annotation['categories']:
        if category['name'] not in summary: continue
        group='novel' if category['frequency']=='r' else 'base'
        value=np.asarray(summary[category['name']]['TETA'][50],dtype=float)
        groups[group].append(value)
        measured_categories.append({'name':category['name'],'group':group})
    grouped={group:({name:float(value) for name,value in zip(fields,np.mean(np.stack(values),axis=0))} if values else None)
        for group,values in groups.items()}
    return {'evaluator':'unmodified official TETA OVMOT recipe; non-exhaustive TAO','evaluator_commit':source_commit,
        'metrics':overall,'base_novel':grouped,'metric_scale':'official TETA summary percent',
        'evaluated_categories':measured_categories,'GT_boxes':len(annotations),'predictions':len(predictions),
        'unmapped_predicted_labels':unmapped,'unmapped_policy':'retain boxes with unknown category -1; never discard predictions or GT',
        'prediction_score':'constant unscored sentinel 1; no detector confidence and no score-based truncation',
        'GT_policy':manifest['GT_policy'],'novel_evaluation_GT_used_for_training':False,
        'official_complete_dataset_result':False,'timestamp_policy':manifest['timestamp_policy']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.video.read_text());result=json.loads((args.run/'result.json').read_text())
    rows=[json.loads(line) for line in (args.run/'frame_decisions.jsonl').read_text().splitlines()]
    check_scope(manifest,result,rows)
    directory=args.run/'official_evaluation';directory.mkdir(parents=True,exist_ok=True)
    measured=rmot(manifest,rows,directory) if manifest['task']=='rmot' else ovmot(manifest,rows,directory)
    report={'status':'OFFICIAL_REGISTERED_MULTITASK_SCOPE_EVALUATION_COMPLETE','task':manifest['task'],
        'dataset':manifest['dataset'],'sequence':manifest['sequence'],'scope':manifest['scope'],
        'frames':len(rows),'all_registered_frames_included':True,'GT_access':'evaluation only',
        'official_test_labels_read':False,'end_to_end_fps':result['end_to_end_fps'],
        'association_replay_fps':result.get('association_replay_fps'),
        'perception_rejected_frames':result['perception_rejected_frames'],**measured}
    (args.run/'official_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    result['official_metrics']={'status':'COMPLETE','path':str(args.run/'official_metrics.json'),'scope':manifest['scope']}
    (args.run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
