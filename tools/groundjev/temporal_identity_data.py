"""TRAIN cross-frame identity references, independent of actor birth aliases."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


class TemporalIdentityData:
    def __init__(self,plan):
        self.plan=plan
        directory=ROOT/plan['output']
        with gzip.open(directory/'identities.json.gz','rt') as f:self.records=json.load(f)
        index=json.loads((ROOT/plan['train_index']).read_text())
        self.units=index['units'];self.eligible=defaultdict(list)
        for index,row in enumerate(self.records):
            if len(row['observations'])>=2:self.eligible[row['unit']].append(index)
        self.eligible={u:ids for u,ids in self.eligible.items() if len(ids)>=2}

    def batch(self,rng):
        import numpy as np
        import torch
        unit=rng.choice(list(self.eligible));available=self.eligible[unit]
        identities=rng.sample(available,min(self.plan['identities_per_batch'],len(available)))
        rows=[]
        for identity in identities:
            record=self.records[identity]
            for frame,candidate in rng.sample(record['observations'],min(self.plan['observations_per_identity'],len(record['observations']))):
                rows.append((frame,candidate,identity))
        result=np.empty((len(rows),2560),dtype=np.float32)
        groups=defaultdict(list)
        for index,(frame,candidate,_) in enumerate(rows):groups[frame].append((index,candidate))
        for frame,selections in groups.items():
            info=self.units[unit]['frames'][str(frame)]
            with np.load(ROOT/info['feature_path'],allow_pickle=False) as f:
                native=f['regions']
                assert native.dtype==np.float32 and native.shape==(info['candidates'],2560)
                for index,candidate in selections:result[index]=native[candidate]
        return torch.from_numpy(result),torch.tensor([r[2] for r in rows]),unit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    proof=json.loads((ROOT/plan['registry_GT_alignment_proof']).read_text())
    assert proof['all_original_TRAIN_registry_candidate_targets_exact']
    index=json.loads((ROOT/plan['train_index']).read_text());counts=Counter();groups=defaultdict(list)
    units={u:v for u,v in index['units'].items() if u.startswith('mot/')}
    max_frame_bytes=max(f['candidates']*2560*4 for v in units.values() for f in v['frames'].values())
    assert max_frame_bytes+plan['identities_per_batch']*plan['observations_per_identity']*2560*4<plan['streaming_working_bytes']
    for unit,info in units.items():
        source=Path(info['full_source_rows']).parent/'candidate_registry.jsonl.gz'
        with gzip.open(source,'rt') as f:
            for line in f:
                row=json.loads(line);counts['all_original_registry_candidates']+=1
                frame,candidate=row['frame'],row['candidate']
                assert 0<=candidate<info['frames'][str(frame)]['candidates']
                if row['posthoc_GT_identity'] is None:
                    counts['unmatched_candidates_not_contrastive_labels']+=1;continue
                groups[unit,row['posthoc_GT_identity']].append([frame,candidate])
                counts['original_GT_matched_observations']+=1
    records=[];eligible=Counter();singletons=Counter()
    for (unit,identity),observations in groups.items():
        assert len({frame for frame,_ in observations})==len(observations)
        records.append(dict(unit=unit,GT_identity=identity,observations=observations))
        if len(observations)>=2:eligible[unit]+=1
        else:singletons[unit]+=1
    assert len(units)==43 and sum(len(v['frames']) for v in units.values())==1056
    assert counts['all_original_registry_candidates']==48051
    with gzip.open(out/'identities.json.gz','wt') as f:json.dump(records,f,separators=(',',':'))
    result=dict(status='TRAIN_NATIVE_CROSS_FRAME_IDENTITY_REFERENCE_CORPUS_COMPLETE',source_units=len(units),
        source_frames=1056,counts=dict(counts),source_identities=len(records),
        cross_frame_identities=sum(eligible.values()),eligible_identities_by_unit=dict(eligible),
        single_observation_identities_by_unit=dict(singletons),
        noncontrastive_units={u:dict(cross_frame_identities=eligible[u],singletons=singletons[u]) for u in units if eligible[u]<2},
        largest_original_frame_feature_bytes=max_frame_bytes,new_persistent_feature_bytes=0,
        new_metadata_bytes_before_summary=sum(p.stat().st_size for p in out.iterdir()),
        labels='Unmodified original TRAIN candidate-level one-to-one IoU>=.5 GT identities; actor birth/action/alias labels never define this contrastive target.',
        singleton_policy='Preserved in corpus/counts; cannot form cross-frame positives. Uniform eligible unit, up-to8distinctIDs and up-to4distinct observations perID, at least2frames each.',
        training_scope='Declared TRAIN1056 source frames, not full videos, long-BPTT or benchmark validation. All official validation frames retained.',
        new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    assert result['new_metadata_bytes_before_summary']<plan['metadata_bytes']
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
