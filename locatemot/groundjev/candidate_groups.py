"""Candidate context grouping without suppression, truncation or GT inputs."""


def candidate_group_indices(candidates,capacity,policy='contiguous'):
    if policy=='contiguous':
        return [list(range(i,min(i+capacity,len(candidates)))) for i in range(0,len(candidates),capacity)] or [[]]
    import numpy as np
    from locatemot.groundjev.geometry import overlap_matrix
    remaining=np.ones(len(candidates),dtype=bool);groups=[]
    while remaining.any():
        seed=int(np.flatnonzero(remaining)[0]);group=[seed];remaining[seed]=False
        scores=np.zeros(len(candidates))
        while len(group)<capacity and remaining.any():
            indices=np.flatnonzero(remaining)
            affinity=overlap_matrix([candidates[group[-1]]['box']],[candidates[i]['box'] for i in indices])[0]
            scores[indices]=np.maximum(scores[indices],affinity)
            chosen=int(indices[np.argmax(scores[indices])])
            group.append(chosen);remaining[chosen]=False
        groups.append(group)
    return groups or [[]]
