"""Explicit post-assignment questions shared by training and opt-in runtime."""
from copy import deepcopy

PROTOCOL = 'final-assigned-existing-identity-memory-update-v1'


def build_assigned_update_request(state, selected_existing_ids):
    """Reuse unmodified causal visual evidence and name actual final IDs.

    All assigned existing IDs are supplied, including those lacking training
    labels. GT eligibility must never select which assignments enter this state.
    IDs use the same namespace as the supplied historical tracks/crop sheet.
    """
    from qev.schema import NoulQuestion, SystemOneRequest
    history = {int(row['id']) for row in state['historical_tracks']}
    assert set(selected_existing_ids) <= {int(c['index']) for c in state['candidates']}
    assert all(int(identity) in history for identity in selected_existing_ids.values())
    current = deepcopy(state)
    current['final_assignment'] = {
        'phase': 'after_global_matching_before_any_current_frame_memory_write',
        'selected_existing_identity_by_candidate': {str(i): int(identity) for i, identity in selected_existing_ids.items()},
        'assignment_source': 'model decisions and global matching; not GT identity correction',
        'memory_update_protocol': PROTOCOL}
    questions = {
        f'assigned_update_{i}': NoulQuestion(type='noul', instructions=
            f'Should candidate {i} be written into long-term memory of the finally assigned '
            f'existing identity {identity}? Answer yes only if it is a clear, reliable observation '
            'of that specific identity, supported by the historical anchor/long/recent evidence '
            'and current geometry. Answer no when identity evidence conflicts or is ambiguous; '
            'a clear object alone is insufficient.')
        for i, identity in selected_existing_ids.items()}
    return SystemOneRequest(state=current, model='jev-latest', questions=questions)
