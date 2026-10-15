"""Exact native Full tracking sheets from causal crop provenance, on demand."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

IO_PROTOCOL = 'gzip-jsonl-native-category-and-tracking-crop-recipes-v1'
CANDIDATE_KEY = 'groundjev_native_tracking_candidate_boxes'
HISTORY_KEY = 'groundjev_native_tracking_history_crops'


@lru_cache(maxsize=8)
def original_image(path):
    from PIL import Image
    with Image.open(path) as source:
        return source.convert('RGB')


@lru_cache(maxsize=192)
def observation_crop(path, box):
    image = original_image(path).crop(box).convert('RGB')
    image.thumbnail((192, 192))
    return image


def render_tracking_media(item, root):
    from locatemot.groundjev.decision import OneJevDecision
    builder = OneJevDecision.__new__(OneJevDecision)
    if CANDIDATE_KEY in item:
        image = original_image(str(Path(root)/item['path']))
        result = builder._candidate_sheet(image, [{'box': box} for box in item[CANDIDATE_KEY]])
    elif HISTORY_KEY in item:
        tracks = []
        for row in item[HISTORY_KEY]:
            crops = {key+'_image': observation_crop(str(Path(root)/row[key]['path']), tuple(row[key]['box']))
                     for key in ('anchor', 'long', 'recent')}
            tracks.append(SimpleNamespace(track_id=row['id'], **crops))
        result = builder._history_sheet(tracks)
    else:
        raise ValueError('Expected a registered native tracking crop recipe')
    assert result.size == (item['w'], item['h'])
    return result


def install():
    from train import mm
    native_media = mm.load_media

    def load_media(media):
        if not any(CANDIDATE_KEY in item or HISTORY_KEY in item for item in media):
            return native_media(media)
        images, videos, metas = [], [], []
        for item in media:
            if CANDIDATE_KEY in item or HISTORY_KEY in item:
                assert item['type'] == 'image'
                images.append(render_tracking_media(item, mm.MM_ROOT))
            else:
                im, vi, md = native_media([item])
                images += im; videos += vi; metas += md
        return images, videos, metas

    mm.load_media = load_media
    return native_media
