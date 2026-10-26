"""Freeze complete available evaluation units after cross-source TRAIN audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import configparser
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.groundjev.eval_no_mot20.common import (GROUND, RUN, G2_SHA, G2_STEP, TRAIN_ROOT,
    TRAIN_SHA, assert_no_mot20, fingerprint, load, save, sha_file)

DATA = Path('/data2/user/LocateMOT/data')
FULL = Path('/data2/user/LocateMOT_FullDataV2_spill/datasets')
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}


def images_in(path):
    return sorted((p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.stem.isdigit()),
                  key=lambda p: int(p.stem))


def training_audit(out):
    manifest = load(TRAIN_ROOT / 'g2_clean_train_manifest.json')
    actual_sha = sha_file(TRAIN_ROOT / 'g2_clean_train_rows.jsonl')
    if actual_sha != TRAIN_SHA or manifest['row_count'] != 583251:
        raise ValueError('G2 TRAIN source digest/row count mismatch')
    coverage = {k: set(v) for k, v in manifest['coverage_videos_or_sequences'].items()}
    aerial = set()
    for line in (TRAIN_ROOT / 'g2_clean_train_rows.jsonl').open():
        row = json.loads(line)
        if row['source'].startswith('AerialMind'):
            for item in row['images']:
                aerial.add(Path(item['path']).parent.name)
    coverage['AerialMind'] = aerial
    j1_path = Path('/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/three_task_sft_v2b/rows.jsonl')
    j1_sources = defaultdict(set)
    for line in j1_path.open():
        row = json.loads(line)
        j1_sources[row['task']].add(str(row['source']))
    receipt = {'status': 'TRAIN_SOURCE_AUDIT_COMPLETE', 'G2_rows': 583251,
               'G2_rows_sha256': actual_sha, 'G2_video_ids_by_source': {k: sorted(v) for k, v in coverage.items()},
               'J1_manifest': str(j1_path), 'J1_manifest_sha256': sha_file(j1_path),
               'J1_video_ids_by_task': {k: sorted(v) for k, v in j1_sources.items()},
               'cross_source_policy': 'exclude RMOT26 copies and same source video; half-val is not video-disjoint',
               'MOT20_registered': False}
    save(out / 'train_overlap_audit.json', receipt)
    return coverage, j1_sources


class Registrar:
    def __init__(self, out):
        self.out = out
        self.jobs, self.exclusions = [], []

    def exclude(self, dataset, sequence, reason, **detail):
        self.exclusions.append(dict(dataset=dataset, sequence=sequence, reason=reason, **detail))

    def add(self, manifest, evaluation):
        assert_no_mot20(manifest)
        if not manifest['images']:
            raise ValueError('Empty video scope')
        manifest.update(model_sha256=G2_SHA, model_step=G2_STEP, parser='official_scoring',
                        decoder='tracking_dense', prompt_fallback=False, training_on_this_scope=False,
                        sampling_seed=42, sampling_seed_scope='request_local_v1',
                        gt_read_by_runtime=False, scope='registered_complete_video',
                        independent_video=True)
        manifest['query_hash'] = fingerprint({'expression': manifest.get('expression'),
                                             'vocabulary': manifest['vocabulary']})
        identity = fingerprint(manifest)
        name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(manifest['sequence']))[:95]
        job_id = f"{manifest['dataset']}__{name}__{identity[:16]}"
        job = self.out / 'jobs' / job_id
        if (job / 'manifest.json').exists() and load(job / 'manifest.json') != manifest:
            raise ValueError('Attempted to overwrite a frozen evaluation manifest')
        save(job / 'manifest.json', manifest)
        save(job / 'evaluation.json', evaluation)
        self.jobs.append({'id': job_id, 'path': str(job), 'dataset': manifest['dataset'],
                          'task': manifest['task'], 'sequence': manifest['sequence'],
                          'frames': len(manifest['images']), 'query_hash': manifest['query_hash'],
                          'manifest_sha256': sha_file(job / 'manifest.json'),
                          'expression_id': manifest.get('expression_id'),
                          'planned_requests': len(manifest['images']) *
                            ((len(manifest['vocabulary']) + manifest.get('vocabulary_chunk_size', 1203)-1)
                             // manifest.get('vocabulary_chunk_size', 1203))})


def add_mot(reg, coverage, j1):
    roots = [('mot17', DATA / 'MOT17/MOT17/train', 'MOT17'),
             ('dancetrack', DATA / 'DanceTrack/val', 'DanceTrack'),
             ('sportsmot', FULL / 'SportsMOT/val', 'SportsMOT')]
    rmot26 = coverage['RMOT26']
    for dataset, root, source in roots:
        for seq in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
            video = seq.name
            reasons = []
            if video in coverage[source]: reasons.append('G2 same training video')
            if video in rmot26: reasons.append('G2 RMOT26 cross-dataset training copy')
            if video in j1.get('identity', set()): reasons.append('J1 training video')
            if source == 'SportsMOT' and video.rsplit('_c',1)[0] in {
                    value.rsplit('_c',1)[0] for value in coverage[source]}:
                reasons.append('G2 same original sports source video')
            if reasons:
                reg.exclude(dataset, video, '; '.join(reasons),
                            half_val_independent=False if dataset == 'mot17' else None)
                continue
            paths = images_in(seq / 'img1')
            config = configparser.ConfigParser(); config.read(seq / 'seqinfo.ini')
            fps = config.getfloat('Sequence', 'frameRate')
            expected = config.getint('Sequence', 'seqLength')
            if len(paths) != expected or [int(p.stem) for p in paths] != list(range(1, expected+1)):
                reg.exclude(dataset, video, 'incomplete physical video frames', available=len(paths), expected=expected)
                continue
            protocol_path = reg.out / 'protocols' / f'{dataset}.json'
            save(protocol_path, {'dataset': dataset, 'split': 'validation',
                                 'official_source': 'https://github.com/MCG-NJU/SportsMOT' if dataset == 'sportsmot' else dataset})
            reg.add({'dataset': dataset, 'sequence': video, 'video_id': video, 'task': 'mot',
                     'split': 'validation', 'vocabulary': ['person'], 'expression': None,
                     'images': [str(p) for p in paths], 'frame_numbers': [int(p.stem) for p in paths],
                     'image_size': [config.getint('Sequence','imWidth'),config.getint('Sequence','imHeight')],
                     'fps': fps, 'physical_frames_complete': True},
                    {'protocol': str(protocol_path), 'gt_path_for_evaluation_only': str(seq / 'gt/gt.txt'),
                     'gt_format': 'motchallenge', 'result_scope': 'OFFICIAL_FULL_VAL'})


def add_animal(reg, coverage):
    root = FULL / 'AnimalTrack'
    for name in (root / 'splits/videos_test.txt').read_text().splitlines():
        video = Path(name).stem
        if video in coverage['AnimalTrack']:
            reg.exclude('animaltrack',video,'G2 training video'); continue
        source = root / 'videos_all' / name
        gt = root / 'gt_all' / f'{video}_gt.txt'
        if not source.exists() or not gt.exists():
            reg.exclude('animaltrack',video,'missing raw video or public GT'); continue
        raw = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','json',str(source)],text=True))['streams'][0]
        count = int(raw['nb_frames']); a,b = raw['r_frame_rate'].split('/'); fps = float(a)/float(b)
        frame_root = root / 'frames_eval' / video
        paths = [str(frame_root / f'{frame:06d}.png') for frame in range(1,count+1)]
        protocol_path = reg.out / 'protocols/animaltrack.json'
        save(protocol_path, {'dataset': 'animaltrack', 'split': 'public_test',
                             'gt_classes': 'single animal class per sequence, unchanged public MOT fields'})
        reg.add({'dataset':'animaltrack','sequence':video,'video_id':video,'task':'mot',
                 'split':'public_test','vocabulary':[video.rsplit('_',1)[0]], 'expression':None,
                 'images':paths,'frame_numbers':list(range(1,count+1)),
                 'image_size':[int(raw['width']),int(raw['height'])],'fps':fps,
                 'physical_frames_complete':True,'raw_video':str(source),
                 'frame_decode':{'codec':'png','pixel_format':'rgb24','start_number':1,'lossless':True}},
                {'protocol':str(protocol_path),'gt_path_for_evaluation_only':str(gt),
                 'gt_format':'motchallenge','result_scope':'REGISTERED_COMPLETE_VIDEO',
                 'public_test_no_tuning':True})


def add_refer(reg, coverage, j1):
    from PIL import Image
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS, LEGAL_SPLIT_VERSION
    root = DATA / 'refer_kitti_v2'
    train = set(coverage['Refer-KITTI-V2']) | j1.get('rmot',set())
    for video in LEGAL_SPLITS['refer_kitti_v2']['validation']:
        if video in train:
            reg.exclude('refer_kitti_v2',video,'G2 or J1 TRAIN overlap'); continue
        image_root = DATA / 'kitti_tracking_training/image_02' / video
        paths = images_in(image_root)
        with Image.open(paths[0]) as image: size = list(image.size)
        if [int(p.stem) for p in paths] != list(range(len(paths))):
            reg.exclude('refer_kitti_v2',video,'incomplete physical video'); continue
        for expression in sorted((root / 'expression' / video).glob('*.json')):
            if expression.name.startswith('.'): continue
            query = load(expression)
            counts = [len(query['label'].get(str(int(p.stem)),[])) for p in paths]
            if query.get('ignore'):
                reg.exclude('refer_kitti_v2',video,'nonempty release ignore needs official semantic adapter',
                            expression_id=expression.stem); continue
            reg.add({'dataset':'refer_kitti_v2','sequence':video,'video_id':video,'task':'rmot',
                     'split':'validation','vocabulary':['person','car'],'expression':query['sentence'],
                     'expression_id':expression.stem,'images':[str(p) for p in paths],
                     'frame_numbers':[int(p.stem) for p in paths],'image_size':size,'fps':10.,
                     'legal_split_version':LEGAL_SPLIT_VERSION,'physical_frames_complete':True},
                    {'expression_file_for_evaluation_only':str(expression),
                     'label_dir_for_evaluation_only':str(root / 'labels_with_ids/image_02' / video),
                     'positive_frames':sum(n>0 for n in counts),'empty_target_frames':sum(n==0 for n in counts),
                     'multi_target_frames':sum(n>1 for n in counts),'GT_boxes':sum(counts),
                     'gt_format':'refer_kitti_v2_top_left_xywh','result_scope':'REGISTERED_COMPLETE_VIDEO'})


def add_tao(reg, j1):
    annotation_path = Path('/data1/user/TAO-Amodal/amodal_annotations/validation_lvis_v1.json')
    annotation = load(annotation_path)
    train_annotation = load(Path('/data1/user/TAO-Amodal/amodal_annotations/train.json'))
    train_names = {v['name'].split('/',1)[-1] for v in train_annotation['videos']}
    train_names.update(v.split('/',1)[-1] for v in j1.get('ovmot',set()))
    vocabulary_path = Path('/data2/user/LocateMOT/outputs/unified/u2/protocol/lvis_v1_vocabulary.json')
    vocabulary = load(vocabulary_path)['categories']
    if len(vocabulary) != 1203 or len(annotation['categories']) != 1203:
        raise ValueError('TAO evaluation requires the genuine LVIS-v1 1203-category annotation/vocabulary')
    by_video = defaultdict(list)
    for image in annotation['images']: by_video[image['video_id']].append(image)
    roots = [Path(f'/data{i}/user/TAO-Amodal/frames') for i in range(4)]
    for video in annotation['videos']:
        name = video['name']
        if name.split('/',1)[-1] in train_names:
            reg.exclude('tao_amodal',name,'TRAIN video overlap'); continue
        images = sorted(by_video[video['id']],key=lambda i:i['frame_index'])
        paths, missing = [], []
        for image in images:
            candidates = [root / image['file_name'] for root in roots]
            path = next((p for p in candidates if p.is_file()),None)
            if path is None: missing.append(image['file_name'])
            else: paths.append(str(path))
        if missing or not images:
            reg.exclude('tao_amodal',name,'missing annotated frame inputs',video_id=video['id'],
                        expected=len(images),missing=len(missing),examples=missing[:3]); continue
        reg.add({'dataset':'tao_amodal','sequence':name,'video_id':video['id'],'task':'ovmot',
                 'split':'validation','vocabulary':[c['name'] for c in vocabulary],
                 'vocabulary_chunk_size':96,'expression':None,'images':paths,
                 'frame_numbers':[i['frame_index'] for i in images],'image_size':[images[0]['width'],images[0]['height']],
                 'fps':30.,'timestamps':[i['frame_index']/30. for i in images],
                 'timestamp_policy':'TAO source frame_index / 30; only sparse annotated frames inferred',
                 'physical_frames_complete':False,'image_ids':[i['id'] for i in images],
                 'vocabulary_path':str(vocabulary_path)},
                {'annotation_path_for_evaluation_only':str(annotation_path),'vocabulary_path':str(vocabulary_path),
                 'GT_policy':'official amodal LVIS-v1 validation annotations; non-exhaustive categories retained',
                 'gt_format':'tao_amodal_lvis_v1','result_scope':'REGISTERED_COMPLETE_VIDEO',
                 'physical_fps_not_available':True})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=RUN);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    coverage,j1=training_audit(args.out)
    reg=Registrar(args.out)
    add_mot(reg,coverage,j1);add_animal(reg,coverage);add_refer(reg,coverage,j1);add_tao(reg,j1)
    aerial_root=FULL/'AerialMind/image_02'
    aerial_videos={p.name for p in aerial_root.iterdir() if p.is_dir()}
    reg.exclude('aerialmind','ALL','all available videos used in G2 dense training',
                videos=len(aerial_videos),train_overlap=len(aerial_videos & coverage['AerialMind']),
                official_query_source='JSON filename stem, https://github.com/shawnliang420/AerialMind')
    reg.exclude('rmot26','ALL','local test release lacks decoded images and complete per-frame tracking GT; initial boxes alone do not define MOT HOTA',
                official_source='https://github.com/gaash-lab/QTrack')
    reg.exclude('burst','ALL','no locally registered independent frames; download is not required for this evaluation')
    summary={}
    for dataset in sorted({j['dataset'] for j in reg.jobs}):
        values=[j for j in reg.jobs if j['dataset']==dataset]
        summary[dataset]={'videos':len({j['sequence'] for j in values}),'jobs':len(values),
                          'query_frames':sum(j['frames'] for j in values),
                          'planned_DLM_requests':sum(j['planned_requests'] for j in values)}
    save(args.out/'jobs.json',{'status':'FROZEN_NON_MOT20_EVALUATION_SCOPE','jobs':reg.jobs,
                             'summary':summary,'excluded':reg.exclusions,'MOT20_jobs':0})
    print(json.dumps({'summary':summary,'excluded_units':len(reg.exclusions),'MOT20_jobs':0}),flush=True)


if __name__=='__main__':main()
