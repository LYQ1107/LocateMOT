"""Exact Compact tensors from immutable native FP32 observation references."""
from functools import lru_cache
import gzip
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL='native-compact-geometry-and-existing-fp32-observation-references-v1'


def geometry(state,width,height):
    import torch
    from locatemot.groundjev.decision import overlap
    candidates=state['candidates'];tracks=state['historical_tracks']
    n,m=len(candidates),len(tracks);timestamp=state['timestamp']
    return dict(candidate_geo=torch.tensor([[c['box'][0]/width,c['box'][1]/height,
        c['box'][2]/width,c['box'][3]/height] for c in candidates]).reshape(n,4),
        history_geo=torch.tensor([[*(x/s for x,s in zip(t['predicted_box'],(width,height,width,height))),
            t['age']/100,t['lost_age']/3,(timestamp-t['last_reliable_timestamp'])/3,t['confidence'] or 0.]
            for t in tracks]).reshape(m,8),
        pair_geometry=torch.tensor([[[overlap(c['box'],t['predicted_box']),
            ((c['box'][0]+c['box'][2]-t['predicted_box'][0]-t['predicted_box'][2])**2+
             (c['box'][1]+c['box'][3]-t['predicted_box'][1]-t['predicted_box'][3])**2)**.5/
             (2*(width*width+height*height)**.5)] for t in tracks] for c in candidates]).reshape(n,m,2))


class CompactReferenceInputs:
    def __init__(self,directory):
        self.directory=Path(directory)
        summary=json.loads((self.directory/'index.json').read_text())
        assert summary['protocol']==PROTOCOL
        self.units=summary['units'];self.records={}
        with gzip.open(self.directory/'groups.jsonl.gz','rt') as file:
            for line in file:
                row=json.loads(line);key=(row['unit'],row['id'])
                assert key not in self.records;self.records[key]=row
        self.text=None;self.specs={};self.classes={}

    @lru_cache(maxsize=4)
    def frame_features(self,unit,frame):
        import numpy as np
        info=self.units[unit]
        with np.load(ROOT/info['frames'][str(frame)]['feature_path'],allow_pickle=False) as file:
            vectors=file['regions']
        assert vectors.dtype==np.float32 and vectors.shape==(info['frames'][str(frame)]['candidates'],2560)
        return vectors

    def inputs(self,unit,identifier):
        import numpy as np
        import torch
        from locatemot.groundjev.text_features import OneJevTextFeatures
        row=self.records[unit,identifier.removesuffix('_assigned')]
        def vectors(refs):
            return np.stack([self.frame_features(unit,frame)[index] for frame,index in refs]) if refs else np.empty((0,2560),dtype=np.float32)
        n,m=len(row['candidate_refs']),len(row['history_refs'])
        current=vectors(row['candidate_refs'])
        history=vectors([ref for triple in row['history_refs'] for ref in triple]).reshape(m,3,2560)
        result=dict(candidates=torch.from_numpy(current),history=torch.from_numpy(history),
            candidate_geo=torch.tensor(row['candidate_geo']).reshape(n,4),
            history_geo=torch.tensor(row['history_geo']).reshape(m,8),
            pair_geometry=torch.tensor(row['pair_geometry']).reshape(n,m,2),
            task_index=('mot','rmot','ovmot').index(row['specification']['task']))
        if self.text is None:self.text=OneJevTextFeatures()
        key=json.dumps(row['specification'],sort_keys=True);vocabulary=tuple(row['specification']['vocabulary'])
        if key not in self.specs:self.specs[key]=self.text.specification(row['specification'])
        if vocabulary not in self.classes:self.classes[vocabulary]=self.text.vocabulary(vocabulary)
        return dict(result,spec_text=self.specs[key],class_text=self.classes[vocabulary])
