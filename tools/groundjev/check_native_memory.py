"""Real native proposals/features through persistent memory, without GT or a benchmark claim."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from PIL import Image
    from locatemot.groundjev.feature_client import NativeFeatureClient
    from locatemot.groundjev.decision import SimpleDecision
    from locatemot.groundjev.tracker import OnlineTracker
    source=ROOT/'outputs/groundjev/runs/mot17_first_native_clip_repair2/frame_decisions.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines()][:2]
    client=NativeFeatureClient('http://127.0.0.1:8111')
    inputs,receipts=[],[]
    for row in rows:
        with Image.open(row['image_path']) as source_image: image=source_image.convert('RGB')
        candidates,receipt=client(image,row['candidates'],image_path=row['image_path'])
        assert all(len(c['native_region_feature'])==2560 and len(c['temporal_region_feature'])==128 for c in candidates)
        inputs.append((image,candidates,row['frame']/30));receipts.append(receipt)
    class HoldDecision(SimpleDecision):
        def decide(self,*args):
            decisions,lifecycle,metadata=super().decide(*args)
            for d in decisions:d['update_probability']=0.
            return decisions,lifecycle,metadata
    tracker=OnlineTracker(HoldDecision())
    first,_=tracker.step(*inputs[0],specification={'task':'mot'})
    state=tracker.state_dict()
    restored=OnlineTracker(HoldDecision());restored.load_state_dict(json.loads(json.dumps(state)))
    expected,_=tracker.step(*inputs[1],specification={'task':'mot'})
    actual,_=restored.step(*inputs[1],specification={'task':'mot'})
    assert expected==actual and tracker.state_dict()==restored.state_dict()
    first_tracks={t['track_id']:t for t in state['tracks']}
    retained=set(first_tracks)&set(tracker.tracks)
    assert retained
    for tid in retained:
        assert tracker.tracks[tid].anchor_feature==first_tracks[tid]['anchor_feature']
        assert tracker.tracks[tid].long_feature==first_tracks[tid]['long_feature']
        assert tracker.tracks[tid].anchor_temporal_feature==first_tracks[tid]['anchor_temporal_feature']
    snapshot=tracker.state_dict()
    try:tracker.step(*inputs[0],specification={'task':'mot'})
    except ValueError:pass
    else:raise AssertionError('Past timestamp accepted')
    assert snapshot==tracker.state_dict()
    receipt={'status':'REAL_NATIVE_FEATURE_MEMORY_ROUNDTRIP_AND_HOLD_PASS',
        'images':[r['image_path'] for r in rows],'candidate_counts':[len(r['candidates']) for r in rows],
        'feature_receipts':receipts,'retained_ids':len(retained),'state_format':state['format'],
        'serialized_final_memory_bytes':len(json.dumps(snapshot).encode()),'GT_read':False,
        'features':'real native DLM 2560 + trained auxiliary temporal128',
        'identical_resume':True,'immutable_anchor':True,'held_long_memory_unchanged':True,
        'past_timestamp_rejected_before_mutation':True,
        'decision_backend':'explicit fixed geometry sanity control, not OneJev','benchmark_claim':False}
    out=ROOT/'outputs/groundjev/validation/native_memory.json'
    out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
