"""Measure legal TRAIN duplicate supervision priors without changing any loss."""
import json
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from locatemot.groundjev.global_competition import sparse_competition
    from tools.groundjev.global_identity_data import training_frames
    from tools.groundjev.global_identity_fit_observations import rmot_fit_observations
    plan=json.loads((ROOT/'configs/groundjev/global_identity_v1.json').read_text());totals={}
    for task,frames in [('mot',training_frames(plan)),('rmot',rmot_fit_observations(plan))]:
        c=Counter()
        for frame in frames:
            candidates=frame['candidates'];features=np.asarray([r['native_region_feature'] for r in candidates],np.float32).reshape(-1,2560)
            edges,_=sparse_competition(np.asarray([r['box'] for r in candidates],np.float32).reshape(-1,4),features,32)
            gt=frame['labels']['unique_GT'].numpy();c['frames']+=1;c['candidates']+=len(candidates);c['sparse_directed_edges']+=len(edges)
            if len(edges):
                a,b=edges.T;known=(gt[a]>=0)&(gt[b]>=0);same=known&(gt[a]==gt[b])
                c['known_duplicate_positive_edges']+=int(same.sum());c['known_other_entity_negative_edges']+=int(known.sum()-same.sum());c['unknown_masked_edges']+=int((~known).sum())
        denominator=c['known_duplicate_positive_edges']+c['known_other_entity_negative_edges']
        totals[task]=dict(counts=dict(c),known_duplicate_positive_fraction=c['known_duplicate_positive_edges']/denominator if denominator else None)
        print(json.dumps(dict(task=task,summary=totals[task])),flush=True)
    out=dict(status='FIXED_SPARSE_TRAIN_GRAPH_DUPLICATE_SUPERVISION_PRIORS_AUDITED',tasks=totals,
        scope='Same raw observed proposals and fixed32-neighbor union as training; GT outside model input. Measures binary edge class balance, not learned discrimination or causal proof.',loss_weights_or_running_experiments_changed=False,new_GPU_processes=0,new_feature_bank_bytes=0,full_goal_complete=False)
    (ROOT/plan['output']/'TRAIN_competition_class_balance.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
