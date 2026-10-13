"""Official GroundAnything DLM HTTP output adapter; no invented confidences."""
import base64
import io
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


class GroundAnythingAdapter:
    def __init__(self, base_url='http://127.0.0.1:8101/v1', max_tokens=8192):
        self.base_url, self.max_tokens = base_url.rstrip('/'), max_tokens

    def __call__(self, image, task, vocabulary, expression=None, image_path=None):
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
        start = time.perf_counter()
        request = Request(self.base_url + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urlopen(request, timeout=600) as response:
            result = parse_response(json.load(response), 'bbox')
        elapsed = time.perf_counter() - start
        if not result.valid:
            raise RuntimeError(f'GroundAnything invalid/truncated output: {result.to_dict()}')
        candidates = []
        for prediction in result.predictions:
            for coordinates in prediction.coordinates or ():
                box = [v / 999 * (image.width - 1 if i % 2 == 0 else image.height - 1)
                       for i, v in enumerate(coordinates)]
                candidates.append({'box': box, 'label': prediction.label,
                    'native_score': None, 'score_kind': 'missing_native_per_box_score',
                    'feature_source': 'native_hidden_feature_not_exported_yet'})
        return candidates, {'seconds': elapsed, 'prompt': prompt, 'result': result.to_dict(),
                            'source_image_path': str(image_path) if image_path else None,
                            'source_encoding': mime, 'lossy_reencoding_applied': False,
                            'coordinate_rule': 'official_grid_0_999_times_image_dimension_minus_one'}
