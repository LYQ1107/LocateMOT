"""Global one-to-one assignment with an explicit optional active-pool contract."""
import math


def assign_candidates(decisions,tracks,*,native_probabilities,explicit_no_match=False,no_match_policy='learned'):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    n,m=len(decisions),len(tracks)
    actions=('NEW','IGNORE','NO_MATCH') if explicit_no_match else ('NEW','IGNORE')
    width=len(actions);costs=np.full((n,m+width*n),1e9);masked=[]
    for i,d in enumerate(decisions):
        probabilities=d['probabilities']
        for j,t in enumerate(tracks):
            if str(t.track_id) in probabilities:
                p=probabilities[str(t.track_id)]
                costs[i,j]=-math.log(max(p,1e-12)) if native_probabilities else -p
        for j,option in enumerate(actions):
            if option not in probabilities:continue
            if option=='NO_MATCH' and no_match_policy=='only_if_omitted' and all(str(t.track_id) in probabilities for t in tracks):
                masked.append(i);continue
            p=probabilities[option]
            costs[i,m+width*i+j]=-math.log(max(p,1e-12)) if native_probabilities else -p
    rows,columns=linear_sum_assignment(costs) if n else ([],[])
    return dict(zip(rows,columns)),actions,masked
