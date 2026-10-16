"""Causal persistent identity memory driven by structured decision distributions."""
from dataclasses import dataclass, field
import math


@dataclass
class Track:
    track_id: int
    box: list
    label: str
    anchor_image: object
    recent_image: object
    long_image: object = None
    velocity: list = field(default_factory=lambda: [0., 0., 0., 0.])
    age: int = 1
    lost_age: float = 0.
    state: str = 'ACTIVE'
    confidence: float | None = None
    timestamp: float = 0.
    last_reliable_timestamp: float = 0.
    recent_boxes: list = field(default_factory=list)
    anchor_feature: list | None = None
    recent_feature: list | None = None
    long_feature: list | None = None
    anchor_temporal_feature: list | None = None
    recent_temporal_feature: list | None = None
    long_temporal_feature: list | None = None
    feature_history: list = field(default_factory=list)
    alive_probability: float | None = None
    identity_uncertainty: float | None = None
    semantic_counts: dict = field(default_factory=dict)

    def predicted_box(self, timestamp):
        return [x + v*(timestamp-self.timestamp) for x, v in zip(self.box, self.velocity)]


class OnlineTracker:
    def __init__(self, decision, lost_seconds=3., update_threshold=.5, memory_policy='gated', ema_alpha=.1,
                 lifecycle_termination_policy='learned_or_timeout',no_match_policy='learned'):
        self.decision = decision
        self.lost_seconds, self.update_threshold = lost_seconds, update_threshold
        self.tracks, self.next_id, self.previous_timestamp = {}, 1, None
        self.memory_policy,self.ema_alpha = memory_policy,ema_alpha
        self.lifecycle_termination_policy=lifecycle_termination_policy
        self.no_match_policy=no_match_policy

    def step(self, image, candidates, timestamp, specification):
        from locatemot.groundjev.assignment import assign_candidates
        if self.previous_timestamp is not None and timestamp <= self.previous_timestamp:
            raise ValueError('Video timestamps must increase; future frames cannot alter past states')
        tracks = list(self.tracks.values())
        decisions, lifecycle, meta = self.decision.decide(image, candidates, tracks, timestamp, specification)
        n, m = len(candidates), len(tracks)
        is_jev = meta['native_probabilities']
        assigned,actions,masked=assign_candidates(decisions,tracks,native_probabilities=is_jev,
            explicit_no_match=meta.get('explicit_no_match',False),no_match_policy=self.no_match_policy)
        width=len(actions)
        assigned_update = getattr(self.decision, 'decide_assigned_updates', None)
        if assigned_update is not None:
            selected_existing = {i: tracks[column].track_id for i, column in assigned.items() if column < m}
            updates, report = assigned_update(selected_existing)
            for i, probability in updates.items():
                decisions[i] = dict(decisions[i], observation_update_probability=decisions[i]['update_probability'],
                                    update_probability=probability, memory_update_probability_source=report['protocol'])
            meta['assigned_identity_update'] = report
        seen, outputs, events = set(), [], []
        for i, candidate in enumerate(candidates):
            column = assigned[i]
            if column >= m and (column-m) % width:
                events.append({'candidate': i, 'action': actions[(column-m)%width], 'decision': decisions[i]})
                continue
            category = decisions[i].get('category_decision')
            label = category['label'] if category else candidate['label']
            reliable = decisions[i]['update_probability'] >= self.update_threshold
            write_memory = reliable or self.memory_policy in ('always','ema','frame_only')
            crop = image.crop(tuple(candidate['box'])).convert('RGB')
            crop.thumbnail((192, 192))
            if column >= m:
                tid = self.next_id
                self.next_id += 1
                track = Track(tid, list(candidate['box']), label, crop.copy(), crop.copy(),
                              long_image=crop.copy(),
                              timestamp=timestamp, last_reliable_timestamp=timestamp,
                              recent_boxes=[list(candidate['box'])])
                self.tracks[tid] = track
                action = 'NEW'
            else:
                track = tracks[column]
                action = 'REACTIVATE' if track.lost_age else 'ACTIVE'
                dt = timestamp-track.timestamp
                track.velocity = [(x-y)/dt for x, y in zip(candidate['box'], track.box)]
                track.box, track.timestamp, track.label = list(candidate['box']), timestamp, label
                track.age += 1
                track.recent_boxes = (track.recent_boxes + [list(candidate['box'])])[-16:]
                track.recent_image = crop.copy()
                if write_memory:
                    track.long_image = crop.copy()
                if reliable:
                    track.last_reliable_timestamp = timestamp
            if candidate.get('native_region_feature') is not None:
                feature=list(candidate['native_region_feature'])
                temporal=candidate.get('temporal_region_feature')
                temporal=list(temporal) if temporal is not None else None
                if track.anchor_feature is None:
                    track.anchor_feature=feature.copy();track.long_feature=feature.copy()
                    track.anchor_temporal_feature=temporal;track.long_temporal_feature=temporal
                track.recent_feature=feature
                track.recent_temporal_feature=temporal
                if write_memory:
                    if self.memory_policy=='ema' and column<m:
                        track.long_feature=[(1-self.ema_alpha)*old+self.ema_alpha*new
                            for old,new in zip(track.long_feature,feature)]
                        if temporal is not None:
                            track.long_temporal_feature=[(1-self.ema_alpha)*old+self.ema_alpha*new
                                for old,new in zip(track.long_temporal_feature,temporal)]
                    else:
                        track.long_feature=feature.copy();track.long_temporal_feature=temporal
                track.feature_history=(track.feature_history+[{'timestamp':timestamp,'feature':feature,'temporal_feature':temporal}])[-16:]
                if not category:
                    track.semantic_counts[label]=track.semantic_counts.get(label,0)+1
                if is_jev:
                    probabilities=list(decisions[i]['probabilities'].values())
                    track.identity_uncertainty=-sum(p*math.log(max(p,1e-12)) for p in probabilities)/math.log(len(probabilities))
                    life=lifecycle.get(track.track_id)
                    track.alive_probability=1.-life['probabilities']['TERMINATE'] if life else None
            if category:
                track.semantic_counts[label] = track.semantic_counts.get(label, 0) + category['probability']
            track.state, track.lost_age = 'ACTIVE', 0.
            probability = decisions[i]['probabilities'].get(str(track.track_id), decisions[i]['probabilities']['NEW'])
            track.confidence = probability if is_jev else None
            seen.add(track.track_id)
            outputs.append({'track_id': track.track_id, 'box': track.box, 'label': track.label,
                            'association_probability': track.confidence, 'native_detection_score': candidate['native_score']})
            if category:
                outputs[-1].update(category_probability=category['probability'],
                    category_probability_kind=category['probability_kind'])
            membership=decisions[i].get('query_membership_probability')
            if membership is not None:
                outputs[-1]['query_membership_probability']=membership
                if membership<.5:outputs.pop()
            events.append({'candidate': i, 'track_id': track.track_id, 'action': action,
                           'memory_action': 'UPDATE' if write_memory else 'HOLD', 'decision': decisions[i]})
        for track in tracks:
            if track.track_id in seen:
                continue
            track.lost_age = timestamp-track.timestamp
            decision = lifecycle.get(track.track_id)
            if track.anchor_feature is not None and decision:
                track.alive_probability=1.-decision['probabilities']['TERMINATE']
            terminate = (decision and decision['choice'] == 'TERMINATE' and
                         self.lifecycle_termination_policy=='learned_or_timeout') or self.memory_policy=='frame_only'
            track.state = 'TERMINATE' if terminate or track.lost_age > self.lost_seconds else 'LOST'
            events.append({'track_id': track.track_id, 'action': track.state, 'lifecycle_decision': decision})
            if self.lifecycle_termination_policy=='timeout_only':
                events[-1]['learned_termination_deferred']=bool(decision and decision['choice']=='TERMINATE' and track.state!='TERMINATE')
            if track.state == 'TERMINATE':
                self.tracks.pop(track.track_id)
        self.previous_timestamp = timestamp
        if self.memory_policy=='frame_only':
            for track in self.tracks.values():
                track.anchor_image=track.recent_image
                track.long_image=track.recent_image
                track.anchor_feature=track.recent_feature
                track.long_feature=track.recent_feature
                track.anchor_temporal_feature=track.recent_temporal_feature
                track.long_temporal_feature=track.recent_temporal_feature
                track.feature_history=track.feature_history[-1:]
                track.recent_boxes=track.recent_boxes[-1:]
        native=any(t.anchor_feature is not None for t in self.tracks.values())
        if self.memory_policy!='gated':meta.update(memory_update_policy=self.memory_policy,ema_alpha=self.ema_alpha)
        if self.lifecycle_termination_policy!='learned_or_timeout':
            meta['lifecycle_termination_policy']=self.lifecycle_termination_policy
        if self.no_match_policy!='learned':
            meta.update(no_match_policy=self.no_match_policy,no_match_infeasible_candidate_indices=masked,
                original_model_probabilities_retained=True,
                no_match_execution_contract='Only an omitted live ID permits NO_MATCH; constraint uses the actual live pool/options, no GT. Raw probabilities, confidence and entropy remain original, not renormalized.')
        return outputs, {'events': events, 'decision_metadata': meta, 'persistent_ids': list(self.tracks),
                         'next_id': self.next_id, 'timestamp': timestamp,
                         'visual_memory_source': 'real_causal_crops_and_native_DLM_region_features' if native else 'real_causal_crops; native feature adapter pending'}

    def state_dict(self):
        import base64
        import io
        from dataclasses import fields
        rows = []
        empty_images = False
        native=any(t.anchor_feature is not None for t in self.tracks.values())
        native_fields={'anchor_feature','recent_feature','long_feature','anchor_temporal_feature','recent_temporal_feature',
            'long_temporal_feature','feature_history','alive_probability','identity_uncertainty','semantic_counts'}
        for track in self.tracks.values():
            row = {}
            # Retain at most one encoding per live track/image field. Anchor
            # and lost-track pixels do not change each frame; repeated PNG
            # compression otherwise dominates persistent-memory export.
            encoded_images = getattr(track, '_encoded_images', {})
            for field in fields(track):
                if not native and field.name in native_fields and not (field.name == 'semantic_counts' and track.semantic_counts): continue
                value = getattr(track, field.name)
                if field.name.endswith('_image'):
                    if value.width == 0 or value.height == 0:
                        row[field.name] = {'empty_image_size': list(value.size), 'mode': value.mode}
                        empty_images = True
                        continue
                    cached = encoded_images.get(field.name)
                    if cached is not None and cached[0] is value:
                        value = cached[1]
                    else:
                        buffer = io.BytesIO()
                        value.save(buffer, format='PNG')
                        encoded = base64.b64encode(buffer.getvalue()).decode()
                        encoded_images[field.name] = (value, encoded)
                        value = encoded
                row[field.name] = value
            track._encoded_images = encoded_images
            rows.append(row)
        state = {'format': 'groundjev-causal-memory-v2-native' if native else 'groundjev-causal-memory-v1', 'tracks': rows, 'next_id': self.next_id,
                'previous_timestamp': self.previous_timestamp, 'lost_seconds': self.lost_seconds,
                'update_threshold': self.update_threshold}
        if empty_images:state['format']='groundjev-causal-memory-v3-empty-images'
        if self.memory_policy!='gated':state.update(memory_policy=self.memory_policy,ema_alpha=self.ema_alpha)
        return state

    def load_state_dict(self, state):
        import base64
        import io
        from PIL import Image
        if state['format'] not in ('groundjev-causal-memory-v1','groundjev-causal-memory-v2-native','groundjev-causal-memory-v3-empty-images'):
            raise ValueError('Unknown identity-memory checkpoint format')
        self.tracks = {}
        for source in state['tracks']:
            row = dict(source)
            for key in ('anchor_image', 'recent_image', 'long_image'):
                if isinstance(row[key],dict):
                    row[key]=Image.new(row[key]['mode'],tuple(row[key]['empty_image_size']))
                    continue
                with Image.open(io.BytesIO(base64.b64decode(row[key]))) as image:
                    row[key] = image.convert('RGB')
            track = Track(**row)
            self.tracks[track.track_id] = track
        self.next_id, self.previous_timestamp = state['next_id'], state['previous_timestamp']
        self.lost_seconds, self.update_threshold = state['lost_seconds'], state['update_threshold']
        self.memory_policy=state.get('memory_policy','gated');self.ema_alpha=state.get('ema_alpha',.1)
