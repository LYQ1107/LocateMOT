"""Opt-in native second-pass write decisions after actual global assignment."""
from locatemot.groundjev.assigned_identity_update import PROTOCOL, build_assigned_update_request
from locatemot.groundjev.decision import OneJevDecision


class AssignedIdentityUpdateDecision(OneJevDecision):
    memory_update_protocol = PROTOCOL

    def decide(self, image, candidates, tracks, timestamp, specification):
        self.assignment_batches = []
        return super().decide(image, candidates, tracks, timestamp, specification)

    def accepted_batch(self, image, group, ranked, timestamp, specification):
        # Accepted groups have passed native context planning. No track has
        # been mutated yet; cache the same causal state and real visual media.
        request, media = self.build_request(image, group, ranked, timestamp, specification)
        offset = sum(batch['size'] for batch in self.assignment_batches)
        self.assignment_batches.append({'offset': offset, 'size': len(group),
                                       'state': request.state, 'media': media})

    def decide_assigned_updates(self, selected_existing_ids):
        updates, reports = {}, []
        for batch in self.assignment_batches:
            offset = batch['offset']
            selected = {i: selected_existing_ids[offset+i] for i in range(batch['size'])
                        if offset+i in selected_existing_ids}
            if not selected:
                continue
            request = build_assigned_update_request(batch['state'], selected)
            response, metadata = self.engine.decide(request, media=batch['media'])
            for local in selected:
                updates[offset+local] = response.answers[f'assigned_update_{local}'].noul
            reports.append({'offset': offset, 'questions': len(request.questions),
                            'selected_existing_identity_by_candidate': dict(selected), 'metadata': metadata})
        assert updates.keys() == selected_existing_ids.keys()
        self.assignment_batches = []
        return updates, {'protocol': PROTOCOL, 'phase': 'after_global_matching_before_any_memory_write',
                         'existing_assignments': len(updates), 'native_second_pass_calls': len(reports),
                         'batches': reports, 'GT_read': False, 'NEW_initialization_policy': 'unchanged'}
