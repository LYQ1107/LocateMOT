"""Measure the actual per-frame BCE supervision prior, without fitting any threshold."""
import json
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from locatemot.groundjev.global_competition import sparse_competition
    from tools.groundjev.global_identity_data import training_frames
    from tools.groundjev.global_identity_fit_observations import rmot_fit_observations
    plan=json.loads((ROOT/'configs/groundjev/global_identity_v1.json').read_text());result={}
    old=json.loads((ROOT/plan['output']/'TRAIN_competition_class_balance.json').read_text())
    for task,iterator in [('mot',training_frames(plan)),('rmot',rmot_fit_observations(plan))]:
        counts=Counter();fraction_sum=0.
        for frame in iterator:
            candidates=frame['candidates'];edges,_=sparse_competition(
                np.asarray([c['box'] for c in candidates],np.float32).reshape(-1,4),
                np.asarray([c['native_region_feature'] for c in candidates],np.float32).reshape(-1,2560),32)
            counts['frames']+=1
            if not len(edges):continue
            gt=frame['labels']['unique_GT'].numpy();a,b=edges.T;known=(gt[a]>=0)&(gt[b]>=0)
            n=int(known.sum());positive=int((known&(gt[a]==gt[b])).sum())
            counts['positive_edges']+=positive;counts['known_edges']+=n
            if n:
                fraction_sum+=positive/n;counts['frames_with_nonzero_duplicate_BCE_supervision']+=1
                counts['frames_with_positive_duplicate_edges']+=positive>0
        assert counts['positive_edges']==old['tasks'][task]['counts']['known_duplicate_positive_edges']
        assert counts['known_edges']==old['tasks'][task]['counts']['known_duplicate_positive_edges']+old['tasks'][task]['counts']['known_other_entity_negative_edges']
        result[task]=dict(counts=dict(counts),per_supervised_frame_mean_positive_fraction=fraction_sum/counts['frames_with_nonzero_duplicate_BCE_supervision'],micro_positive_fraction=counts['positive_edges']/counts['known_edges'])
        print(json.dumps(dict(task=task,summary=result[task])),flush=True)
    combined=sum(result[k]['per_supervised_frame_mean_positive_fraction']*result[k]['counts']['frames_with_nonzero_duplicate_BCE_supervision'] for k in result)/sum(result[k]['counts']['frames_with_nonzero_duplicate_BCE_supervision'] for k in result)
    out=dict(status='ACTUAL_PER_FRAME_DUPLICATE_LOSS_SUPERVISION_PRIORS_AUDITED',tasks=result,mixed_per_supervised_frame_prior=combined,
        scope='Same fixed whole-frame TRAIN proposal graphs/unique GT labels. BCE is mean over known edges in each supervised frame; this reports that normalization, not a learned or deployment posterior calibration.',new_optimizer_updates=0,new_GPU_processes=0,threshold_optimized=False,full_goal_complete=False)
    (ROOT/plan['output']/'TRAIN_competition_per_frame_prior.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
