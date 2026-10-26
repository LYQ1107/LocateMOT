"""Freeze TRAIN-internal physical-video splits and a bounded pilot frame index."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

RUN = Path('/data3/user/LocateMOT-runs/G2_Spatial_DenseHead_20261026')
FULL = Path('/data2/user/LocateMOT_FullDataV2_spill/datasets')
G2_TRAIN = Path('/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/g2_clean_train_manifest.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def physical_source(dataset, sequence):
    return re.sub(r'_c\d+$', '', sequence) if dataset == 'sportsmot' else sequence


def uniform(rows, count):
    if len(rows) <= count:
        return rows
    return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]


def parse_gt(path, animal, width, height):
    valid, ignored = defaultdict(list), defaultdict(list)
    ids, audit = set(), Counter()
    for line in path.read_text().splitlines():
        f = [v.strip() for v in line.split(',')]
        if len(f) < 7:
            raise ValueError(f'incomplete GT row in {path}')
        frame, identity = int(float(f[0])), int(float(f[1]))
        x, y, w, h = map(float, f[2:6])
        box = [max(0., x), max(0., y), min(float(width), x+w), min(float(height), y+h)]
        if box[2] <= box[0] or box[3] <= box[1]:
            audit['degenerate_after_clip'] += 1
            continue
        okay = float(f[6]) > 0 and (animal or len(f) < 8 or int(float(f[7])) == 1)
        (valid if okay else ignored)[frame].append(box)
        audit['valid' if okay else 'ignored'] += 1
        if okay:
            ids.add(identity)
    return valid, ignored, len(ids), dict(audit)


def prepare(out=RUN, train_frames_per_video=64, dev_check_frames=24):
    out = Path(out)
    data = out/'data'
    data.mkdir(parents=True, exist_ok=True)
    records, videos = [], []
    for dataset in ['sportsmot', 'animaltrack']:
        root = FULL/('SportsMOT' if dataset == 'sportsmot' else 'AnimalTrack')
        if dataset == 'sportsmot':
            sequences = sorted(p.name for p in (root/'train').iterdir() if p.is_dir() and not p.name.startswith('.'))
        else:
            sequences = sorted(Path(x.strip()).stem for x in (root/'splits/videos_train.txt').read_text().splitlines() if x.strip())
        for seq in sequences:
            frame_root = root/'train'/seq/'img1' if dataset == 'sportsmot' else root/'frames_all'/seq
            gt_path = root/'train'/seq/'gt/gt.txt' if dataset == 'sportsmot' else root/'gt_all'/(seq+'_gt.txt')
            images = sorted((p for p in frame_root.iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg'} and p.stem.isdigit()), key=lambda p:int(p.stem))
            with Image.open(images[0]) as image:
                width, height = image.size
            gt, ignored, identities, audit = parse_gt(gt_path, dataset == 'animaltrack', width, height)
            assert set(gt) <= {int(p.stem) for p in images}, (dataset, seq, 'missing annotated frame')
            group = physical_source(dataset, seq)
            category = 'person' if dataset == 'sportsmot' else seq.rsplit('_',1)[0]
            hashes = {str(p.resolve()):digest(p) for p in uniform(images, 3)}
            videos.append({'dataset':dataset,'sequence':seq,'source_group':group,'category':category,
                           'GT_path':str(gt_path.resolve()),'GT_sha256':digest(gt_path),
                           'frames':len(images),'identities':identities,'image_size':[width,height],
                           'target_boxes':sum(map(len,gt.values())),'GT_audit':audit,
                           'representative_image_sha256':hashes})
            for path in images:
                frame = int(path.stem)
                records.append({'dataset':dataset,'sequence':seq,'source_group':group,'category':category,
                                'image_path':str(path.resolve()),'frame':frame,'image_size':[width,height],
                                'boxes':gt[frame],'ignored':ignored[frame],'exhaustive':True})
    # Hash order is fixed before model fitting, independent of GT/difficulty/results.
    order = lambda s: hashlib.sha256(('42:'+s).encode()).hexdigest()
    sports_groups = sorted({v['source_group'] for v in videos if v['dataset']=='sportsmot'},key=order)
    sports_dev = set(sports_groups[:2])
    animal_dev = set()
    for category in sorted({v['category'] for v in videos if v['dataset']=='animaltrack'}):
        names = [v['sequence'] for v in videos if v['dataset']=='animaltrack' and v['category']==category]
        animal_dev.add(sorted(names,key=order)[0])
    grouped = defaultdict(list)
    for row in records:
        is_dev = row['source_group'] in sports_dev if row['dataset']=='sportsmot' else row['sequence'] in animal_dev
        row['split'] = 'dev' if is_dev else 'train'
        grouped[(row['dataset'],row['sequence'])].append(row)
    splits = {'train':[],'dev':[],'dev_full':[]}
    for rows in grouped.values():
        if rows[0]['split']=='train':
            splits['train'].extend(uniform(rows,train_frames_per_video))
        else:
            splits['dev'].extend(uniform(rows,dev_check_frames))
            splits['dev_full'].extend(rows)
    train_groups = {(r['dataset'],r['source_group']) for r in splits['train']}
    dev_groups = {(r['dataset'],r['source_group']) for r in splits['dev']}
    assert not train_groups & dev_groups
    sample_hashes = defaultdict(set)
    for video in videos:
        split = 'train' if (video['dataset'],video['source_group']) in train_groups else 'dev'
        video['head_split']=split
        sample_hashes[split].update(video['representative_image_sha256'].values())
    assert not sample_hashes['train'] & sample_hashes['dev'], 'representative duplicate image across splits'
    manifest = json.loads(G2_TRAIN.read_text())
    g2_coverage = manifest['coverage_videos_or_sequences']
    for video in videos:
        key = 'SportsMOT' if video['dataset']=='sportsmot' else 'AnimalTrack'
        video['seen_in_G2_TRAIN'] = video['sequence'] in g2_coverage[key]
    for name, rows in splits.items():
        with (data/(name+'.jsonl')).open('w') as f:
            for row in rows:
                f.write(json.dumps(row,separators=(',',':'))+'\n')
    train_counts = [len(r['boxes']) for r in records if r['split']=='train']
    maximum = max(train_counts)
    queries = max(100,min(300,((maximum*5//4+31)//32)*32))
    assert queries >= maximum, 'query capacity smaller than observed TRAIN targets'
    receipt = {'status':'FROZEN_TRAIN_INTERNAL_VIDEO_SPLIT','seed':42,'videos':videos,
               'split_source_groups':{'train':sorted(train_groups),'dev':sorted(dev_groups)},
               'source_overlap':[],'representative_image_duplicates':[],
               'G2_training_manifest':str(G2_TRAIN),'G2_training_rows_sha256':manifest['rows_sha256'],
               'head_DEV_is_not_independent_of_G2_finetuning':True,
               'query_selection':{'max_TRAIN_boxes':maximum,'queries':queries,'policy':'ceil(1.25*TRAIN_max/32)*32, bounded100..300'},
               'pilot_train_frames_per_video':train_frames_per_video,'dev_check_frames_per_video':dev_check_frames,
               'indices':{k:{'path':str(data/(k+'.jsonl')),'sha256':digest(data/(k+'.jsonl')),
                              'frames':len(v),'boxes':sum(len(r['boxes']) for r in v),
                              'videos':len({(r['dataset'],r['sequence']) for r in v}),
                              'by_domain':dict(Counter(r['dataset'] for r in v))} for k,v in splits.items()},
               'cross_dataset_audit':'MOT17 detector variants canonicalize to same MOT17-XX; RMOT26 aliases overlap MOT17/DanceTrack G2 training. None enters this head optimization. Sports clips grouped by original YouTube ID; Animal physical files disjoint, three representative contents hashed per video; no proof of unseen-camera independence.',
               'TAO_policy':'Existing TRAIN Base only; Novel excluded. Non-exhaustive: positive-only or category-qualified negatives required. Not in this pilot optimizer.',
               'Refer_policy':'Existing TRAIN physical labels required; expression-selected boxes cannot define exhaustive background. Not in this pilot optimizer.'}
    write_json(data/'protocol.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['videos','split_source_groups']},indent=2))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=RUN)
    args=parser.parse_args();prepare(args.out)
