"""Official GroundAnything DLM HTTP output adapter; no invented confidences."""
import base64
import io
import importlib.util
import json
import mimetypes
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'external/GroundAnything'))
from grounding_anything.client import parse_response
from grounding_anything.protocol import build_dense_bbox_prompt, build_refer_bbox_prompt


class GroundingProtocolError(RuntimeError):
    def __init__(self, metadata):
        self.metadata = metadata
        super().__init__('GroundAnything invalid/truncated output: '+str(metadata['result']))


# A rejected dense decode is often vocabulary-string sensitive rather than a
# model failure (DanceTrack: "person" can collapse to full-frame boxes while
# "pedestrian" grounds correctly). Retry a short, fixed chain and record it.
FALLBACK_VOCABULARIES = (('pedestrian',), ('person .',), ('person',))


class GroundAnythingAdapter:
    def __init__(self, base_url='http://127.0.0.1:8101/v1', max_tokens=8192, parser_policy='strict_client', decoder_policy='generic_greedy', vocabulary_chunk_size=None):
        self.base_url, self.max_tokens = base_url.rstrip('/'), max_tokens
        self.parser_policy = parser_policy
        self.decoder_policy = decoder_policy
        self.vocabulary_chunk_size = vocabulary_chunk_size
        self.decode_parameters = None
        if decoder_policy == 'tracking_dense':
            profiles = json.loads((ROOT/'external/GroundAnything/infer/decode/configs/task_profiles.json').read_text())
            self.decode_parameters = dict(profiles['shared_decode'])
            tier = profiles['tiers']['dense']
            self.decode_parameters.update(temperature=tier['temperature'], top_p=tier['top_p'])
            self.max_tokens = tier['max_tokens']

    def adapt_response(self, response, image, image_path, mime, prompt, elapsed):
        result = parse_response(response, 'bbox')
        parsed = [(p.label, coordinates) for p in result.predictions for coordinates in p.coordinates or ()]
        repair_report = None
        if (not result.valid and self.parser_policy in ('official_scoring', 'official_scoring_repaired')
                and result.finish_reason == 'stop'):
            source = ROOT/'external/GroundAnything/eval/utils/prompt_mode.py'
            name = '_groundjev_official_scoring'
            if name not in sys.modules:
                spec = importlib.util.spec_from_file_location(name, source)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            if self.parser_policy == 'official_scoring_repaired':
                # Preserve every complete, independently valid pair while
                # dropping malformed fragments. This never fills coordinates
                # or reorders tuples; strict validity remains recorded below.
                from locatemot.groundjev.native_output_repair import repair_gam_bbox_output
                parsed, repair_report = repair_gam_bbox_output(
                    result.raw_output,
                    label_validator=sys.modules[name].canonicalize_phrase,
                )
            else:
                # Use the pinned official evaluator's native spatial scoring
                # parser unchanged. Keep strict wire-format diagnostics
                # independently.
                parsed = sys.modules[name]._parse_gam_predictions_deduplicated(result.raw_output, 'bbox')
        metadata = {'seconds': elapsed, 'prompt': prompt, 'result': result.to_dict(),
                    'parser_policy': self.parser_policy, 'strict_wire_valid': result.valid,
                    'decoder_policy': self.decoder_policy, 'request_decode_parameters': self.decode_parameters,
                    'official_scoring_recovery': not result.valid and bool(parsed),
                    'parser_repair': ({
                        'version': 'complete_pair_salvage_v1',
                        'applied': bool(repair_report and parsed),
                        'scope': 'complete object_ref/box pairs only; malformed fragments dropped; coordinates never filled or reordered',
                        'strict_parse_error': result.parse_error,
                        **repair_report,
                    } if repair_report is not None else None),
                    'source_image_path': str(image_path) if image_path else None,
                    'source_encoding': mime, 'lossy_reencoding_applied': False,
                    'coordinate_rule': 'official_grid_0_999_times_image_dimension_minus_one'}
        if not result.valid and not parsed:
            raise GroundingProtocolError(metadata)
        candidates = []
        for label, coordinates in parsed:
            box = [v / 999 * (image.width - 1 if i % 2 == 0 else image.height - 1)
                   for i, v in enumerate(coordinates)]
            candidates.append({'box': box, 'label': label,
                'native_score': None, 'score_kind': 'missing_native_per_box_score',
                'feature_source': 'native_hidden_feature_not_exported_yet'})
        return candidates, metadata

    def __call__(self, image, task, vocabulary, expression=None, image_path=None):
        if task == 'ovmot' and self.vocabulary_chunk_size and len(vocabulary) > self.vocabulary_chunk_size:
            start = time.perf_counter()
            candidates, reports, failures, outcomes = [], [], [], []
            for offset in range(0, len(vocabulary), self.vocabulary_chunk_size):
                chunk = vocabulary[offset:offset+self.vocabulary_chunk_size]
                try:
                    current, report = self._request(image, task, chunk, expression, image_path)
                except GroundingProtocolError as error:
                    failures.append({'offset':offset,'vocabulary':chunk,'metadata':error.metadata})
                    outcomes.append({'offset':offset,'classes':len(chunk),'status':'REJECTED','metadata':error.metadata})
                    continue
                candidates.extend(current)
                reports.append(report)
                outcomes.append({'offset':offset,'classes':len(chunk),'status':'SCORED','metadata':report})
            if failures:
                metadata=dict(failures[0]['metadata'],vocabulary_chunk_offset=failures[0]['offset'],
                    vocabulary_chunk=failures[0]['vocabulary'],completed_chunk_reports=reports,
                    seconds=time.perf_counter()-start,chunk_outcomes=outcomes,
                    requested_vocabulary_size=len(vocabulary),all_vocabulary_classes_requested=True,
                    rejected_chunk_count=len(failures),
                    chunk_failure_policy='request every registered class; retain every raw outcome; reject the entire frame if any chunk fails')
                raise GroundingProtocolError(metadata)
            return candidates, {'seconds':time.perf_counter()-start, 'mode':'full_vocabulary_chunks',
                'vocabulary_size':len(vocabulary), 'vocabulary_chunk_size':self.vocabulary_chunk_size,
                'chunk_count':len(reports), 'chunks':reports,
                'all_vocabulary_classes_requested':True,
                'strict_wire_valid':all(r['strict_wire_valid'] for r in reports),
                'decoder_policy':self.decoder_policy,
                'candidate_merge':'concatenate all native outputs; no GT filtering or cross-label NMS'}
        try:
            return self._request(image, task, vocabulary, expression, image_path)
        except GroundingProtocolError as first_error:
            if task != 'mot':
                raise
            for alternative in FALLBACK_VOCABULARIES:
                if list(alternative) == list(vocabulary):
                    continue
                try:
                    candidates, metadata = self._request(image, task, list(alternative),
                                                         expression, image_path)
                except GroundingProtocolError:
                    continue
                metadata['vocabulary_fallback'] = {
                    'from': list(vocabulary), 'to': list(alternative),
                    'reason': str(first_error)[:160],
                    'policy': 'fixed chain, recorded per frame; no GT involved'}
                return candidates, metadata
            raise

    def _request(self, image, task, vocabulary, expression, image_path):
        if image_path is not None:
            payload = Path(image_path).read_bytes()
            mime = mimetypes.guess_type(str(image_path))[0]
        else:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            payload, mime = buffer.getvalue(), 'image/png'
        prompt = build_refer_bbox_prompt(expression) if task == 'rmot' else build_dense_bbox_prompt(vocabulary)
        body = {'model': 'groundinganything', 'temperature': 0, 'max_tokens': self.max_tokens,
                'skip_special_tokens': False, 'spaces_between_special_tokens': False,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,' + base64.b64encode(payload).decode()}},
                    {'type': 'text', 'text': prompt}]}]}
        if self.decode_parameters is not None:
            body.update(temperature=self.decode_parameters['temperature'], top_p=self.decode_parameters['top_p'],
                        custom_params={'gam_dlm_decode': self.decode_parameters})
        start = time.perf_counter()
        request = Request(self.base_url + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urlopen(request, timeout=600) as response:
            response = json.load(response)
        elapsed = time.perf_counter() - start
        return self.adapt_response(response, image, image_path, mime, prompt, elapsed)
