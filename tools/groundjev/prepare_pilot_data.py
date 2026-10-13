"""Freeze first legal video scopes from preserved protocols; never read test GT."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/data2/user/LocateMOT/outputs/unified/u2/protocol')


def main():
    from PIL import Image
    clips = {}
    for name, filename, seq_name in [('mot17', 'mot17_half.json', 'MOT17-02-FRCNN'),
                                    ('dancetrack', 'dancetrack.json', 'dancetrack0004'),
                                    ('mot20', 'mot20_half.json', 'MOT20-01')]:
        protocol = json.loads((SOURCE / filename).read_text())
        seq = next(s for s in protocol['sequences'] if s['sequence'] == seq_name)
        frames = seq['validation_frames']
        images = [str(Path(seq['image_dir']) / f) for f in frames]
        missing = [p for p in images if not Path(p).is_file()]
        if missing or not Path(seq['gt_path']).is_file():
            raise RuntimeError(f'{name} legal pilot files missing: {missing[:3]}')
        with Image.open(images[0]) as image:
            image.load()
            size = image.size
        clips[name] = {'protocol': str(SOURCE / filename), 'sequence': seq_name,
                      'images': images, 'frame_numbers': [int(Path(f).stem) for f in frames],
                      'fps': seq.get('fps', 30), 'image_size': size,
                      'gt_path_for_evaluation_only': seq['gt_path'],
                      'gt_policy': protocol.get('gt_policy'), 'task': 'mot', 'vocabulary': ['person'],
                      'scope': 'full_validation_video' if name == 'dancetrack' else 'full_registered_development_half',
                      'training_on_this_scope': False, 'official_test_labels_read': False}
    out = ROOT / 'outputs/groundjev/pilot_data'
    out.mkdir(parents=True, exist_ok=True)
    for name, manifest in clips.items():
        (out / f'{name}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = {'status': 'REAL_LEGAL_VIDEO_FILES_VERIFIED',
               'clips': {name: {k: v for k, v in d.items() if k not in ('images', 'frame_numbers')} |
                         {'frames': len(d['images']), 'first': d['images'][0], 'last': d['images'][-1]}
                         for name, d in clips.items()}, 'runtime_gt_access': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
