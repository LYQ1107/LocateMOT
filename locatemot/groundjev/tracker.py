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

    def predicted_box(self, timestamp):
        return [x + v*(timestamp-self.timestamp) for x, v in zip(self.box, self.velocity)]


class OnlineTracker:
    def __init__(self, decision, lost_seconds=3., update_threshold=.5):
        self.decision = decision
        self.lost_seconds, self.update_threshold = lost_seconds, update_threshold
        self.tracks, self.next_id, self.previous_timestamp = {}, 1, None

    def step(self, image, candidates, timestamp, specification):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        if self.previous_timestamp is not None and timestamp <= self.previous_timestamp:
            raise ValueError('Video timestamps must increase; future frames cannot alter past states')
        tracks = list(self.tracks.values())
        decisions, lifecycle, meta = self.decision.decide(image, candidates, tracks, timestamp, specification)
        n, m = len(candidates), len(tracks)
        actions = ('NEW','IGNORE','NO_MATCH') if meta.get('explicit_no_match') else ('NEW','IGNORE')
        width = len(actions)
        costs = np.full((n, m+width*n), 1e9)
        is_jev = meta['native_probabilities']
        for i, d in enumerate(decisions):
            probabilities = d['probabilities']
            for j, t in enumerate(tracks):
                if str(t.track_id) in probabilities:
                    p = probabilities[str(t.track_id)]
                    costs[i, j] = -math.log(max(p, 1e-12)) if is_jev else -p
            for j, option in enumerate(actions):
                if option not in probabilities:
                    continue
                p = probabilities[option]
                costs[i, m+width*i+j] = -math.log(max(p, 1e-12)) if is_jev else -p
        rows, columns = linear_sum_assignment(costs) if n else ([], [])
        assigned = dict(zip(rows, columns))
        seen, outputs, events = set(), [], []
        for i, candidate in enumerate(candidates):
            column = assigned[i]
            if column >= m and (column-m) % width:
                events.append({'candidate': i, 'action': actions[(column-m)%width], 'decision': decisions[i]})
                continue
            reliable = decisions[i]['update_probability'] >= self.update_threshold
            crop = image.crop(tuple(candidate['box'])).convert('RGB')
            crop.thumbnail((192, 192))
            if column >= m:
                tid = self.next_id
                self.next_id += 1
                track = Track(tid, list(candidate['box']), candidate['label'], crop.copy(), crop.copy(),
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
                track.box, track.timestamp, track.label = list(candidate['box']), timestamp, candidate['label']
                track.age += 1
                track.recent_boxes = (track.recent_boxes + [list(candidate['box'])])[-16:]
                track.recent_image = crop.copy()
                if reliable:
                    track.long_image = crop.copy()
                    track.last_reliable_timestamp = timestamp
            track.state, track.lost_age = 'ACTIVE', 0.
            probability = decisions[i]['probabilities'].get(str(track.track_id), decisions[i]['probabilities']['NEW'])
            track.confidence = probability if is_jev else None
            seen.add(track.track_id)
            outputs.append({'track_id': track.track_id, 'box': track.box, 'label': track.label,
                            'association_probability': track.confidence, 'native_detection_score': candidate['native_score']})
            events.append({'candidate': i, 'track_id': track.track_id, 'action': action,
                           'memory_action': 'UPDATE' if reliable else 'HOLD', 'decision': decisions[i]})
        for track in tracks:
            if track.track_id in seen:
                continue
            track.lost_age = timestamp-track.timestamp
            decision = lifecycle.get(track.track_id)
            terminate = decision and decision['choice'] == 'TERMINATE'
            track.state = 'TERMINATE' if terminate or track.lost_age > self.lost_seconds else 'LOST'
            events.append({'track_id': track.track_id, 'action': track.state, 'lifecycle_decision': decision})
            if track.state == 'TERMINATE':
                self.tracks.pop(track.track_id)
        self.previous_timestamp = timestamp
        return outputs, {'events': events, 'decision_metadata': meta, 'persistent_ids': list(self.tracks),
                         'next_id': self.next_id, 'timestamp': timestamp,
                         'visual_memory_source': 'real_causal_crops; native feature adapter pending'}

    def state_dict(self):
        import base64
        import io
        from dataclasses import fields
        rows = []
        for track in self.tracks.values():
            row = {}
            for field in fields(track):
                value = getattr(track, field.name)
                if field.name.endswith('_image'):
                    buffer = io.BytesIO()
                    value.save(buffer, format='PNG')
                    value = base64.b64encode(buffer.getvalue()).decode()
                row[field.name] = value
            rows.append(row)
        return {'format': 'groundjev-causal-memory-v1', 'tracks': rows, 'next_id': self.next_id,
                'previous_timestamp': self.previous_timestamp, 'lost_seconds': self.lost_seconds,
                'update_threshold': self.update_threshold}

    def load_state_dict(self, state):
        import base64
        import io
        from PIL import Image
        if state['format'] != 'groundjev-causal-memory-v1':
            raise ValueError('Unknown identity-memory checkpoint format')
        self.tracks = {}
        for source in state['tracks']:
            row = dict(source)
            for key in ('anchor_image', 'recent_image', 'long_image'):
                with Image.open(io.BytesIO(base64.b64decode(row[key]))) as image:
                    row[key] = image.convert('RGB')
            track = Track(**row)
            self.tracks[track.track_id] = track
        self.next_id, self.previous_timestamp = state['next_id'], state['previous_timestamp']
        self.lost_seconds, self.update_threshold = state['lost_seconds'], state['update_threshold']
