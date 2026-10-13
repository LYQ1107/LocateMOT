"""CPU memory/causality sanity on real annotated frames, not tracking accuracy."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from PIL import Image
    from locatemot.groundjev.decision import SimpleDecision
    from locatemot.groundjev.tracker import OnlineTracker
    manifest = json.loads((ROOT/'outputs/groundjev/pilot_data/mot17.json').read_text())
    numbers = manifest['frame_numbers'][:2]
    annotations = {}
    for line in Path(manifest['gt_path_for_evaluation_only']).read_text().splitlines():
        fields = list(map(float, line.split(',')))
        if int(fields[0]) in numbers and fields[6] == 1 and fields[7] == 1:
            x,y,w,h = fields[2:6]
            annotations.setdefault(int(fields[0]), {})[int(fields[1])] = [x,y,x+w,y+h]
    identity = min(set(annotations[numbers[0]]) & set(annotations[numbers[1]]))
    inputs = []
    for path, number in zip(manifest['images'][:2], numbers):
        with Image.open(path) as source:
            image = source.convert('RGB')
        inputs.append((image, [{'box': annotations[number][identity], 'label': 'person', 'native_score': None}], number/manifest['fps']))
    original = OnlineTracker(SimpleDecision())
    first, _ = original.step(*inputs[0], specification={'task':'mot'})
    state = original.state_dict()
    resumed = OnlineTracker(SimpleDecision())
    resumed.load_state_dict(json.loads(json.dumps(state)))
    expected, report = original.step(*inputs[1], specification={'task':'mot'})
    actual, _ = resumed.step(*inputs[1], specification={'task':'mot'})
    assert actual == expected and resumed.state_dict() == original.state_dict()
    assert first[0]['track_id'] == expected[0]['track_id']
    assert state['previous_timestamp'] == inputs[0][2]
    before = original.state_dict()
    try:
        original.step(*inputs[0], specification={'task':'mot'})
    except ValueError:
        pass
    else:
        raise AssertionError('Past timestamp accepted')
    assert original.state_dict() == before
    result = {'status':'CPU_MEMORY_SERIALIZATION_AND_PAST_TIME_REJECTION_PASS',
              'frames':manifest['images'][:2], 'source':'real GT boxes as explicit annotated fixture',
              'detector_used':False, 'pretrained_onejev_used':False, 'benchmark':False,
              'gt_used_in_fixture':True, 'identical_resumed_state':True,
              'identity_retained':True, 'metric_claim':None}
    out = ROOT/'outputs/groundjev/validation/memory_fixture.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
