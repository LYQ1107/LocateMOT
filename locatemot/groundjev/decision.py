"""Real OneJev multimodal decisions with shared images and dynamic identities."""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'external/OneJev'))


def overlap(a, b):
    w, h = max(0., min(a[2], b[2]) - max(a[0], b[0])), max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = max(0., a[2]-a[0]) * max(0., a[3]-a[1]), max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return w * h / max(area_a + area_b - w*h, 1e-8)


class SimpleDecision:
    """Explicit geometry-only reference; never called JEV or learned."""
    def decide(self, image, candidates, tracks, timestamp, specification):
        decisions = []
        for candidate in candidates:
            scores = {str(t.track_id): overlap(candidate['box'], t.predicted_box(timestamp)) for t in tracks}
            scores.update(NEW=.3, IGNORE=0.)
            decisions.append({'probabilities': scores, 'update_probability': 1., 'reliability': None})
        return decisions, {}, {'backend': 'simple_iou_reference', 'native_probabilities': False}


class OneJevDecision:
    def __init__(self, checkpoint, device='cuda:0', batch_candidates=16, shortlist=32, evidence_format='v1'):
        from qev.mm_engine import MMDecisionEngine
        self.checkpoint = str(Path(checkpoint).resolve())
        self.engine = MMDecisionEngine(str(checkpoint), device=device, dtype='bfloat16', cuda_graphs=False)
        self.batch_candidates, self.shortlist = batch_candidates, shortlist
        self.evidence_format = evidence_format

    def _candidate_sheet(self, image, candidates):
        from PIL import Image, ImageDraw
        sheet = Image.new('RGB',(4*160,max(1,math.ceil(len(candidates)/4))*192),'white')
        draw = ImageDraw.Draw(sheet)
        for i,candidate in enumerate(candidates):
            x,y = (i%4)*160,(i//4)*192
            draw.text((x+2,y+2),f'Candidate {i}',fill='black')
            crop = image.crop(tuple(candidate['box'])).convert('RGB')
            crop.thumbnail((156,164))
            sheet.paste(crop,(x+2,y+22))
        return sheet

    def _history_sheet(self, tracks):
        from PIL import Image, ImageDraw
        columns, cell_w, cell_h = 4, 288, 144
        sheet = Image.new('RGB', (columns * cell_w, max(1, math.ceil(len(tracks) / columns)) * cell_h), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, track in enumerate(tracks):
            x, y = i % columns * cell_w, i // columns * cell_h
            draw.text((x+2, y+2), f'ID {track.track_id} anchor / long / recent', fill='black')
            for j, crop in enumerate((track.anchor_image, track.long_image, track.recent_image)):
                resized = crop.copy()
                resized.thumbnail((92, 112))
                sheet.paste(resized, (x + j*96, y+22))
        return sheet

    def build_request(self, image, group, ranked, timestamp, specification):
        from qev.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
        history = [{'id': t.track_id, 'box': t.box, 'predicted_box': t.predicted_box(timestamp),
                    'age': t.age, 'lost_age': t.lost_age, 'velocity': t.velocity,
                    'last_reliable_timestamp': t.last_reliable_timestamp, 'state': t.state,
                    'confidence': t.confidence} for t in ranked]
        state = {'current_frame': '<image:1>', 'timestamp': timestamp, 'task': specification,
                 'candidates': [{'index': i, 'box': c['box'], 'label': c['label']} for i, c in enumerate(group)],
                 'historical_tracks': history}
        media = [{'type': 'image', 'image': image}]
        evidence_format = getattr(self,'evidence_format','v1')
        v2 = evidence_format in ('v2_crop_motion','v3_crop_motion')
        if v2 and group:
            state['current_candidate_crop_sheet'] = '<image:2>'
            media.append({'type':'image','image':self._candidate_sheet(image,group)})
            state['geometry_columns_track_ids'] = [t.track_id for t in ranked]
            state['motion_iou_rows_by_candidate'] = [[round(overlap(c['box'],t.predicted_box(timestamp)),4) for t in ranked] for c in group]
            scale = math.hypot(image.width,image.height)
            state['center_distance_rows_by_candidate'] = [[round(math.hypot(
                (c['box'][0]+c['box'][2]-t.predicted_box(timestamp)[0]-t.predicted_box(timestamp)[2])/2,
                (c['box'][1]+c['box'][3]-t.predicted_box(timestamp)[1]-t.predicted_box(timestamp)[3])/2)/scale,4) for t in ranked] for c in group]
        if ranked:
            state['historical_anchor_and_recent_crops'] = f'<image:{len(media)+1}>'
            media.append({'type': 'image', 'image': self._history_sheet(ranked)})
        questions = {}
        for i, candidate in enumerate(group):
            options = {str(t.track_id): f'Existing persistent identity {t.track_id}, evidence in history and visual crop sheet' for t in ranked}
            options.update(NEW='A valid newly appearing object that belongs to no existing identity',
                           IGNORE='Invalid/duplicate detection, do not create an identity')
            if evidence_format == 'v2_crop_motion':
                options['NO_MATCH'] = 'A valid object has no supported matching identity among the shortlisted options; hold the unresolved association without creating a false ID'
            elif evidence_format == 'v3_crop_motion' and ranked:
                options['NO_MATCH'] = 'This object was previously observed in the video, but no listed identity supports its match; hold the unresolved association. A genuinely new object must use NEW instead.'
            questions[f'match_{i}'] = ChoiceQuestion(type='choice', instructions=
                f'Which identity owns current candidate {i}? Compare visual content, history, motion and competing identities. Never use a class label alone as identity.', criteria=options)
            questions[f'update_{i}'] = NoulQuestion(type='noul', instructions=
                f'Is candidate {i} a clear, reliable identity observation suitable for long-term memory update? Reject occlusion, ambiguous identity and drifting boxes.')
            questions[f'reliability_{i}'] = ScoreQuestion(type='score', instructions=
                f'How reliable is the identity evidence for candidate {i}?', criteria=['contradictory', 'ambiguous', 'supported', 'strong'])
        for t in ranked:
            questions[f'life_{t.track_id}'] = ChoiceQuestion(type='choice', instructions=
                f'What is the current lifecycle of identity {t.track_id} given only current and past evidence?',
                criteria={'ACTIVE': 'currently visible and associated', 'LOST': 'temporarily unobserved, retain memory',
                          'REACTIVATE': 'returned after a gap', 'TERMINATE': 'permanently leave this tracked video'})
        if not questions:
            questions['empty'] = NoulQuestion(type='noul', instructions='Does the current image contain a valid tracking target?')
        return SystemOneRequest(state=state, model='jev-latest', questions=questions), media

    def decide(self, image, candidates, tracks, timestamp, specification):
        results, life, reports = [], {}, []
        groups = [candidates[i:i+self.batch_candidates] for i in range(0, len(candidates), self.batch_candidates)] or [[]]
        for offset, group in enumerate(groups):
            # The shortlist uses only predicted past geometry. Coverage of GT
            # identity options is measured downstream, never used to choose it.
            ranked = sorted(tracks, key=lambda t: (-max((overlap(c['box'], t.predicted_box(timestamp)) for c in group), default=0.),
                                                 t.lost_age, t.track_id))[:self.shortlist]
            request, media = self.build_request(image, group, ranked, timestamp, specification)
            response, meta = self.engine.decide(request, media=media)
            for i in range(len(group)):
                results.append({'probabilities': response.answers[f'match_{i}'].probabilities,
                    'update_probability': response.answers[f'update_{i}'].noul,
                    'reliability': response.answers[f'reliability_{i}'].score,
                    'reliability_probabilities': response.answers[f'reliability_{i}'].probabilities,
                    'shortlisted_ids': [t.track_id for t in ranked]})
            for t in ranked:
                # Lifecycle is recorded once, from the first group that sees
                # this identity; group order is deterministic and causal.
                life.setdefault(t.track_id, response.answers[f'life_{t.track_id}'].model_dump())
            reports.append({'batch': offset, 'metadata': meta, 'questions': len(request.questions)})
        return results, life, {'backend': 'onejev_full', 'checkpoint': self.checkpoint, 'native_probabilities': True,
                               'evidence_format':self.evidence_format,
                               'explicit_no_match':self.evidence_format in ('v2_crop_motion','v3_crop_motion'),
                               'shortlist_limit': self.shortlist, 'batches': reports}
