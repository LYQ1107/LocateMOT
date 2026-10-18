"""Global one-to-one assignment with an explicit optional active-pool contract.

The verification hook in this module is deliberately additive.  With no
``verification_results`` or with the default disabled policy, the exact
Choice-only matrix is built.  When enabled, Yes/No evidence is applied to
shortlisted Track/Detection pairs *before* the single Hungarian solve; it
does not create identities, perform a second assignment, or write track
memory.  The caller remains responsible for lifecycle and post-assignment
memory updates.
"""
import math


def _verification_result_for(verification_results, row_index, track_id):
    """Resolve a pair result from the supported row/track-keyed mappings."""

    if verification_results is None:
        return None
    for key in ((row_index, track_id), (row_index, str(track_id)), (str(row_index), track_id), str(track_id), track_id):
        try:
            if key in verification_results:
                return verification_results[key]
        except TypeError:
            continue
    return None


def assign_candidates(
    decisions,
    tracks,
    *,
    native_probabilities,
    explicit_no_match=False,
    no_match_policy='learned',
    verification_results=None,
    verification_policy=None,
    return_audit=False,
):
    """Solve the physical assignment once, optionally fusing Yes/No evidence.

    ``verification_results`` is keyed by ``(decision_row, track_id)`` (with a
    few scalar-key fallbacks for callers that already use track IDs).  Values
    are ``YesNoResult`` instances or ``None``.  Only pairs that already have a
    native Choice probability are eligible.  A verifier ``No`` therefore
    lowers a compatible pair through the configured policy; it is never
    converted into ``NEW``.  The returned audit records raw and fused values
    so callers can count corrected and introduced association events.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    n,m=len(decisions),len(tracks)
    actions=('NEW','IGNORE','NO_MATCH') if explicit_no_match else ('NEW','IGNORE')
    width=len(actions);costs=np.full((n,m+width*n),1e9);masked=[]
    audit=[]
    for i,d in enumerate(decisions):
        probabilities=d['probabilities']
        for j,t in enumerate(tracks):
            if str(t.track_id) in probabilities:
                raw_p=float(probabilities[str(t.track_id)])
                p=raw_p
                result=_verification_result_for(verification_results, i, t.track_id)
                if verification_policy is not None and result is not None:
                    p=float(verification_policy.fuse_association_score(raw_p, result))
                    audit.append({
                        'row': i,
                        'track_id': int(t.track_id),
                        'raw_choice_probability': raw_p,
                        'p_yes': getattr(result, 'p_yes', None),
                        'result_status': getattr(result, 'status', None),
                        'used': bool(getattr(result, 'used', False)),
                        'fused_probability': p,
                    })
                costs[i,j]=-math.log(max(p,1e-12)) if native_probabilities else -p
        for j,option in enumerate(actions):
            if option not in probabilities:continue
            if option=='NO_MATCH' and no_match_policy=='only_if_omitted' and all(str(t.track_id) in probabilities for t in tracks):
                masked.append(i);continue
            p=probabilities[option]
            costs[i,m+width*i+j]=-math.log(max(p,1e-12)) if native_probabilities else -p
    rows,columns=linear_sum_assignment(costs) if n else ([],[])
    assignment=dict(zip(rows,columns))
    if return_audit:
        return assignment,actions,masked,audit
    return assignment,actions,masked
