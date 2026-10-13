"""Load the actual local pretrained model and decide on a verified MOT frame."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'external/OneJev')]


def main():
    import torch
    from PIL import Image
    from qev.mm_engine import MMDecisionEngine
    from qev.schema import SystemOneRequest, ChoiceQuestion, NoulQuestion, ScoreQuestion
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/OneJev-0.8B')
    parser.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    engine = MMDecisionEngine(str(args.checkpoint), device='cuda:0', dtype='bfloat16', cuda_graphs=False)
    load_seconds = time.perf_counter()-start
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    request = SystemOneRequest(state={'current_frame': '<image:1>', 'task': 'Locate visible people for online MOT',
                                      'historical_tracks': [], 'future_frames_available': False},
        questions={'identity': ChoiceQuestion(type='choice', instructions='How should a valid visible new person be handled?',
                                             criteria={'NEW': 'Create a new persistent identity', 'IGNORE': 'Ignore an invalid detection'}),
                   'visible': NoulQuestion(type='noul', instructions='Does the image contain visible people?'),
                   'quality': ScoreQuestion(type='score', instructions='How reliable is person identity evidence?',
                                           criteria=['none', 'ambiguous', 'good', 'strong'])})
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    response, metadata = engine.decide(request, media=[{'type': 'image', 'image': image}])
    torch.cuda.synchronize()
    result = {'status': 'REAL_PRETRAINED_MODEL_FRAME_DECISION_COMPLETE_NOT_TRACKING_BENCHMARK',
              'checkpoint': str(args.checkpoint), 'image': str(args.image), 'physical_gpu': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
              'load_seconds': load_seconds, 'decision_seconds': time.perf_counter()-start,
              'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'response': response.model_dump(),
              'metadata': metadata, 'official_test_labels_read': False, 'training_steps': 0}
    out = ROOT / 'outputs/groundjev/model_setup/onejev_real_frame.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
