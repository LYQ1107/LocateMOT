"""Verify exact geometry and model decisions on a completed dense native state."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from PIL import Image
    from locatemot.groundjev.compact import CompactDecision
    from locatemot.groundjev.decision import overlap
    from locatemot.groundjev.geometry import overlap_matrix, rank_tracks
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.run_video import ReplayFrames
    torch.set_num_threads(4)
    run = ROOT/'outputs/groundjev/compact/hybrid_shortlist32_v1/tao_raw_full40_learned_hybrid32'
    result = json.loads((run/'result.json').read_text())
    manifest = json.loads(Path(result['video_manifest']).read_text())
    plan = json.loads((ROOT/'configs/groundjev/compact_hybrid_shortlist32_v1.json').read_text())
    replay = ReplayFrames(ROOT/plan['scopes']['tao_raw_full40']['candidates'])
    row = replay.frame(manifest['frame_numbers'][-1])
    replay.close()
    namespace = {}
    revision = 'ba94b948373ea5e2d2891130c502d48e2acd3e86'
    source = subprocess.check_output(['git','show',revision+':locatemot/groundjev/compact.py'], cwd=ROOT, text=True)
    exec(compile(source, 'scalar_compact_reference.py', 'exec'), namespace)
    tracker = OnlineTracker(None)
    tracker.load_state_dict(torch.load(ROOT/result['final_identity_memory_path'], map_location='cpu', weights_only=False))
    tracks, candidates = list(tracker.tracks.values()), row['candidates']
    timestamp = tracker.previous_timestamp
    predicted = [t.predicted_box(timestamp) for t in tracks]
    start = time.perf_counter()
    scalar = np.asarray([[overlap(c['box'], box) for box in predicted] for c in candidates])
    scalar_seconds = time.perf_counter()-start
    start = time.perf_counter()
    batched = overlap_matrix([c['box'] for c in candidates], predicted)
    batched_seconds = time.perf_counter()-start
    assert np.array_equal(scalar, batched)
    groups = 0
    for offset in range(0,len(candidates),16):
        scores = scalar[offset:offset+16].max(0)
        expected = sorted(range(len(tracks)), key=lambda i:(-float(scores[i]),tracks[i].lost_age,tracks[i].track_id))
        for limit in (32,64,128):
            actual = rank_tracks(candidates[offset:offset+16],tracks,predicted,limit)
            assert [t.track_id for t in actual] == [tracks[i].track_id for i in expected[:limit]]
        groups += 1
    empty_expected = sorted(tracks,key=lambda t:(t.lost_age,t.track_id))[:32]
    assert rank_tracks([],tracks,predicted,32) == empty_expected
    assert rank_tracks(candidates,[],[],32) == []
    image = Image.open(row['image_path']).convert('RGB')
    specification = {'task':'ovmot','vocabulary':manifest['vocabulary'],'expression':None}
    cases = []
    timings = {}
    for policy in ('geometry','hybrid16_16'):
        old = namespace['CompactDecision'](result['checkpoint'],device='cpu',shortlist_policy=policy)
        new = CompactDecision(result['checkpoint'],device='cpu',shortlist_policy=policy)
        for label,group,history in [('dense_two_groups',candidates[:32],tracks),
                                     ('empty_observation',[],tracks),('no_history',candidates[:16],[])]:
            before = old.decide(image,group,history,timestamp,specification)
            after = new.decide(image,group,history,timestamp,specification)
            assert before == after
            cases.append({'shortlist_policy':policy,'case':label,'all_decisions_lifecycle_metadata_exact':True})
        measurements = {'scalar': [], 'batched': []}
        for _ in range(3):
            for label,model in [('scalar',old),('batched',new)]:
                start = time.perf_counter()
                model.decide(image,candidates[:16],tracks,timestamp,specification)
                measurements[label].append(time.perf_counter()-start)
        timings[policy] = {k:{'samples_seconds':v,'median_seconds':statistics.median(v)} for k,v in measurements.items()}
    out = ROOT/'outputs/groundjev/validation/dense_geometry_batching_v1'
    out.mkdir(parents=True,exist_ok=False)
    report = {'status':'REAL_DENSE_GEOMETRY_AND_COMPACT_DECISIONS_EXACT',
        'scalar_reference_revision':revision,'candidates':len(candidates),'live_tracks':len(tracks),
        'pairwise_overlaps_exact':int(scalar.size),'all_group_shortlists32_64_128_exact':groups,
        'model_checks':cases,'one_shot_overlap_matrix_seconds':{'scalar':scalar_seconds,'batched':batched_seconds},
        'warm16candidate_decision_probe':timings,'new_GPU_processes':0,'new_feature_cache_files':False,
        'limits':'Actual completed final state and actual image/candidates; post-state numerical/performance '
            'probe, not a new causal video trajectory or official accuracy result. Cold input/state/model '
            'loading excluded from timings, three unprofiled warm repeats per policy, concurrent workload '
            'not controlled; no full-pipeline FPS/speedup claim. Existing processes retain imported old code.'}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
