"""Small legal-train native ROI pack with exact uncorrected Full histories."""
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.feature_client import NativeFeatureClient
    source = ROOT/'outputs/groundjev/training_data/native_ground_own_history_v1'
    out = ROOT/'outputs/groundjev/compact/native_pack_v1'
    out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'PACKING_REAL_NATIVE_ROI_HISTORY', 'scope': '48 legal training frames only',
             'source': str(source), 'frames': 0, 'examples': 0, 'GT_corrected_actor': False}
    def save():
        (out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)
    save()
    train = [json.loads(line) for line in (source/'train.jsonl').open()]
    groups = defaultdict(dict)
    for row in train:
        batch = int(re.search(r'_batch(\d+)_', row['id'])[1])
        groups[(row['source'], row['supervision']['frame'])].setdefault(batch, []).append(row)
    client = NativeFeatureClient('http://127.0.0.1:8111')
    examples, memory, previous = [], {}, None
    try:
        for frame in map(json.loads, (source/'native_rollout.jsonl').open()):
            if frame['sequence'] != previous:
                memory = {}; previous = frame['sequence']
            with Image.open(frame['image_path']) as raw:
                image = raw.convert('RGB')
            enriched, feature_meta = client(image, frame['candidates'], image_path=frame['image_path'])
            current = torch.tensor([c['native_region_feature'] for c in enriched], dtype=torch.float32)
            events = {e['candidate']: e for e in frame['state']['events'] if 'candidate' in e}
            life_events = {e['track_id']: e.get('lifecycle_decision') for e in frame['state']['events'] if 'candidate' not in e}
            offset = 0
            for batch, rows in sorted(groups[(frame['sequence'], frame['frame'])].items()):
                row = rows[0]; context = row['state']; n = len(context['candidates'])
                assert [c['box'] for c in context['candidates']] == [c['box'] for c in frame['candidates'][offset:offset+n]]
                ranked = events[offset]['decision']['shortlisted_ids']
                aliases = [h['id'] for h in context['historical_tracks']]
                assert len(ranked) == len(aliases)
                m = len(ranked); option_ids = [str(i) for i in aliases]+['NEW', 'IGNORE', 'NO_MATCH']
                history = torch.stack([torch.stack([memory[t][k] for k in ('anchor', 'recent', 'long')]) for t in ranked]) if m else torch.empty(0, 3, 2560)
                hist_geo = []
                for h in context['historical_tracks']:
                    b = h['predicted_box']; w, height = image.size
                    hist_geo.append([b[0]/w, b[1]/height, b[2]/w, b[3]/height,
                        h['age']/100, h['lost_age']/3, (context['timestamp']-h['last_reliable_timestamp'])/3,
                        h['confidence'] or 0.])
                candidate_geo = torch.tensor([[c['box'][0]/image.width, c['box'][1]/image.height,
                    c['box'][2]/image.width, c['box'][3]/image.height] for c in context['candidates']])
                geometry = torch.stack([torch.tensor(context[k]) for k in
                    ('motion_iou_rows_by_candidate', 'center_distance_rows_by_candidate')], dim=-1).reshape(n,m,2)
                labels = {r['id'].split(f'_batch{batch}_', 1)[1]: r for r in rows}
                target = torch.tensor([option_ids.index(next(iter(labels[f'match_{i}']['target']))) for i in range(n)])
                teacher = torch.zeros(n,m+3)
                for i in range(n):
                    p = events[offset+i]['decision']['probabilities']
                    assert events[offset+i]['decision']['shortlisted_ids'] == ranked
                    for j,t in enumerate(ranked): teacher[i,j] = p[str(t)]
                    for j,name in enumerate(('NEW','IGNORE','NO_MATCH')): teacher[i,m+j] = p.get(name,0.)
                teacher /= teacher.sum(-1,keepdim=True)
                life_target = torch.tensor([['ACTIVE','LOST','REACTIVATE','TERMINATE'].index(
                    next(iter(labels[f'life_{a}']['target']))) for a in aliases])
                life_teacher = torch.zeros(m,4); life_mask = torch.zeros(m,dtype=torch.bool)
                for j,t in enumerate(ranked):
                    if life_events.get(t):
                        life_teacher[j] = torch.tensor([life_events[t]['probabilities'][k] for k in ('ACTIVE','LOST','REACTIVATE','TERMINATE')])
                        life_teacher[j] /= life_teacher[j].sum(); life_mask[j] = True
                example = {'id':f"{frame['sequence']}_{frame['frame']}_batch{batch}",
                    'candidates':current[offset:offset+n].clone(), 'history':history,
                    'candidate_geo':candidate_geo, 'history_geo':torch.tensor(hist_geo).reshape(m,8),
                    'pair_geometry':geometry, 'task_index':0, 'target':target, 'teacher':teacher,
                    'update_target':torch.tensor([float('yes' in labels[f'update_{i}']['target']) for i in range(n)]),
                    'update_teacher':torch.tensor([events[offset+i]['decision']['update_probability'] for i in range(n)]),
                    'reliability_target':torch.tensor([int(next(iter(labels[f'reliability_{i}']['target']))) for i in range(n)]),
                    'reliability_teacher':torch.tensor([[events[offset+i]['decision']['reliability_probabilities'][str(k)] for k in range(4)] for i in range(n)]),
                    'life_target':life_target,'life_teacher':life_teacher,'life_teacher_mask':life_mask,
                    'source_sequence':frame['sequence'], 'frame':frame['frame'],
                    'candidate_boxes':[c['box'] for c in context['candidates']], 'derived_withholding':False}
                examples.append(example)
                for key,r in labels.items():
                    if not key.startswith('withheld_'): continue
                    i = int(key.split('_')[1]); removed = int(target[i]); assert removed < m
                    keep = [j for j in range(m) if j != removed]
                    augmented = dict(example, id=example['id']+'_withheld'+str(i), derived_withholding=True)
                    for name in ('history','history_geo','life_target','life_teacher','life_teacher_mask'): augmented[name] = example[name][keep]
                    augmented['pair_geometry'] = geometry[:,keep]
                    augmented['target'] = torch.tensor([len(keep)+2 if int(t)==removed else int(t)-(int(t)>removed) for t in target])
                    augmented['teacher'] = teacher[:,keep+list(range(m,m+3))].clone()
                    augmented['teacher'][:,-1] += teacher[:,removed]
                    examples.append(augmented)
                offset += n
            assert offset == len(frame['candidates'])
            for i,event in events.items():
                if 'track_id' not in event: continue
                tid = event['track_id']; feature = current[i].clone()
                if event['action']=='NEW': memory[tid] = dict(anchor=feature, recent=feature, long=feature)
                else:
                    memory[tid]['recent'] = feature
                    if event['memory_action']=='UPDATE': memory[tid]['long'] = feature
            memory = {t:v for t,v in memory.items() if t in frame['state']['persistent_ids']}
            state.update(frames=state['frames']+1, examples=len(examples), feature_source=feature_meta['feature_source']); save()
        assert state['frames']==48
        torch.save(examples, out/'examples.pt')
        size = (out/'examples.pt').stat().st_size; assert size < 25*2**30
        state.update(status='REAL_NATIVE_COMPACT_TRAIN_PACK_COMPLETE', examples=len(examples),
            feature_file_bytes=size, feature_cache_budget_bytes=25*2**30,
            native_feature_dimension=2560, actual_Full_teacher=True,
            derived_option_withholding=sum(e['derived_withholding'] for e in examples),
            validation_GT_used=False, full_source_dataset_cache=False); save()
    except Exception as error:
        state.update(status='REAL_NATIVE_COMPACT_PACK_FAILED', error=str(error)); save(); raise


if __name__=='__main__':main()
