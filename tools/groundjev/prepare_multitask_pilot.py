"""Register legal RMOT/OVMOT scopes with no evaluation targets in runtime inputs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = Path('/data2/user/LocateMOT/data')


def save(name, manifest, out):
    from PIL import Image
    if len(manifest['images']) != len(manifest['frame_numbers']):
        raise ValueError('Physical frame scope is inconsistent')
    for path in manifest['images']:
        if not Path(path).is_file(): raise FileNotFoundError(path)
    with Image.open(manifest['images'][0]) as image: manifest['image_size'] = list(image.size)
    manifest.update(training_on_this_scope=False, official_test_labels_read=False, runtime_GT_fields=False)
    (out/f'{name}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {k:v for k,v in manifest.items() if k not in ('images','frame_numbers','vocabulary','timestamps')} | {
        'frames':len(manifest['images']), 'vocabulary_size':len(manifest['vocabulary']),
        'first':manifest['images'][0], 'last':manifest['images'][-1]}


def main():
    from locatemot.unified.data.legal_scope import assert_legal_video, LEGAL_SPLITS, LEGAL_SPLIT_VERSION
    out=ROOT/'outputs/groundjev/multitask_pilot_data'
    out.mkdir(parents=True,exist_ok=True)
    summary={}
    dataset,video='refer_kitti_v2','0007'
    assert_legal_video(video,dataset=dataset)
    if video not in LEGAL_SPLITS[dataset]['validation']: raise ValueError('Expected frozen v2 validation video')
    image_dir=DATA/'kitti_tracking_training/image_02'/video
    # Both descriptions and intervals are frozen explicit functional checks.
    # Only the human expression is copied, never its label/ignore fields.
    scopes=[('rmot_single_empty','a-red-coat-is-being-worn-by-the-woman',range(150,158)),
            ('rmot_multi_empty','arranging-automobiles',range(32,44))]
    for name,query,frames in scopes:
        path=DATA/'refer_kitti_v2/expression'/video/f'{query}.json'
        expression=json.loads(path.read_text())['sentence']
        manifest={'dataset':dataset,'sequence':video,'task':'rmot','expression':expression,
            'expression_file_for_evaluation_only':str(path),
            'label_dir_for_evaluation_only':str(DATA/'refer_kitti_v2/labels_with_ids/image_02'/video),
            'legal_split_version':LEGAL_SPLIT_VERSION,'split':'validation',
            'images':[str(image_dir/f'{n:06d}.png') for n in frames], 'frame_numbers':list(frames),
            'fps':10,'vocabulary':['person','car'], 'scope':'registered_short_functional_clip',
            'timestamp_policy':'KITTI 10 Hz frame clock; source zero-based frame numbers retained',
            'GT_policy':'release expression target IDs intersect exact per-frame release boxes; no visibility or clipping filter'}
        summary[name]=save(name,manifest,out)
        # Register the full physical sequence now for subsequent full-video runs.
        full=dict(manifest); paths=sorted(image_dir.glob('*.png'))
        full.update(images=[str(p) for p in paths],frame_numbers=[int(p.stem) for p in paths],scope='full_legal_validation_video')
        summary[name+'_full']=save(name+'_full',full,out)
    source=Path('/data2/user/TAO-Amodal/annotations/validation.json')
    annotation=json.loads(source.read_text())
    video=min(annotation['videos'],key=lambda v:v['id'])
    images=sorted((im for im in annotation['images'] if im['video_id']==video['id']),key=lambda im:im['frame_index'])
    vocabulary_path=Path('/data2/user/LocateMOT/outputs/unified/u2/protocol/lvis_v1_vocabulary.json')
    categories=json.loads(vocabulary_path.read_text())['categories']
    vocabulary=[category['name'].replace('_',' ') for category in categories]
    # Physical source FPS is absent from this local annotation/archive. Do not
    # infer it from a 30-frame annotation gap. Use declared annotation-step time
    # for this sparse functional protocol, never call these units seconds.
    manifest={'dataset':'tao_amodal','sequence':video['name'],'video_id':video['id'],'task':'ovmot',
        'annotation_path_for_evaluation_only':str(source),'vocabulary_path':str(vocabulary_path),
        'vocabulary':vocabulary,'vocabulary_policy':'all 1203 fixed LVIS-v1 names; no GT-driven class selection',
        'images':[str(DATA/'TAO-Amodal/frames'/im['file_name']) for im in images],
        'image_ids': [im['id'] for im in images], 'frame_numbers':[im['frame_index']+1 for im in images],
        'timestamps':list(range(len(images))),'fps':1,
        'timestamp_policy':'annotation-step units; physical source FPS unavailable; display at one step per second',
        'physical_video_fps_verified':False,'scope':'full_legal_validation_video_annotation_frames',
        'GT_policy':'all original modal xywh boxes; no clipping, visibility threshold, or category exclusion',
        'base_novel_policy':'official TETA frequency c/f=Base, r=Novel; retain all original categories',
        'novel_evaluation_GT_used_for_training':False}
    summary['ovmot_tao_full']=save('ovmot_tao_full',manifest,out)
    short=dict(manifest)
    for key in ('images','frame_numbers','image_ids','timestamps'): short[key]=short[key][:4]
    short['scope']='registered_short_functional_clip'
    summary['ovmot_tao_first4']=save('ovmot_tao_first4',short,out)
    report={'status':'LEGAL_MULTITASK_VIDEO_INPUTS_REGISTERED','scopes':summary,
        'shared_tracker':'locatemot.groundjev.tracker.OnlineTracker',
        'shared_decision_checkpoint':str(ROOT/'models/OneJev-0.8B'),
        'official_test_labels_read':False, 'native_tracking_results':'pending real inference'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
