"""Export real query-conditioned histories through native Full-v3 question schemas."""
import argparse
from collections import defaultdict
from copy import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def context_for_manifest(manifest, original, label_root, alias_index):
    from PIL import Image
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS, assert_legal_video
    video = assert_legal_video(manifest['sequence'], dataset='refer_kitti_v2')
    assert video in LEGAL_SPLITS['refer_kitti_v2']['fit']
    query = json.loads(Path(manifest['expression_file']).read_text())
    assert manifest['task'] == 'rmot' and query['sentence'] == manifest['expression']
    truth = defaultdict(list)
    for frame, path in zip(manifest['frame_numbers'], manifest['images']):
        requested = set(map(int, query['label'].get(str(frame), [])))
        label = Path(label_root)/video/f'{frame:06d}.txt'
        if not label.exists():
            assert not requested
            continue
        with Image.open(path) as image:
            width, height = image.size
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split(); identity = int(fields[1]); x, y, w, h = map(float, fields[2:6])
            truth[frame].append(dict(gt_id=identity,
                box=[x*width, y*height, (x+w)*width, (y+h)*height], is_query=identity in requested))
        assert requested <= {g['gt_id'] for g in truth[frame]}
    return dict(manifest=manifest, truth=truth, original=str(original), alias_index=alias_index,
        specification=dict(task='rmot', vocabulary=manifest['vocabulary'], expression=manifest['expression']))


def main():
    import torch
    from locatemot.groundjev.compact import CompactDecision
    from tools.groundjev.export_broader_native_full_groups import export_clip
    from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
    from tools.groundjev.run_unified600_controls import alive

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT'
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    input_validation = json.loads((ROOT/plan['input_validation']).read_text())
    assert input_validation['status'] == 'MOT_DEFAULT_EXACT_AND_REAL_RMOT8_NATIVE_INPUTS_LABELS_PIXELS_CPU_PLAN_VALIDATED'
    assert input_validation['all_crop_pixels_and_actor_outputs_states_exact']
    assert input_validation['MOT_default']['dropped_rows'] == input_validation['RMOT_CPU_replay']['dropped_rows'] == 0
    source = ROOT/plan['source']; scope = ROOT/plan['source_scope']
    validation = json.loads((scope/'validation.json').read_text())
    assert validation['status'] == 'ALL_FIT1814_PHYSICAL_FRAMES_18_NATIVE_QUERY_SCOPES_VALIDATED_NOT_COLLECTED'
    by_name = {item['name']: (i, item) for i, item in enumerate(validation['scopes'])}
    out = ROOT/plan['output']; out.mkdir(parents=True, exist_ok=args.resume)
    previous = json.loads((out/'owner_state.json').read_text()) if args.resume else None
    state = dict(status='CPU_WAITING_COMPLETED_NATIVE_RMOT_QUERY_ROLLOUTS',
        owner_pid=os.getpid(), owner_starttick=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        completed={}, new_GPUs=0, new_OneJev_training_updates=0, full_goal_complete=False)
    if previous is not None:
        assert all(v['question_evidence_format'] == plan['evidence_format'] for v in previous['completed'].values())
        state.update(completed=previous['completed'], resumed_completed_rollouts=len(previous['completed']),
            previous_owner=[previous['owner_pid'], previous['owner_starttick']])

    def save():
        temporary = out/'owner_state.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(out/'owner_state.json')
        print(json.dumps(dict(status=state['status'], completed_rollouts=len(state['completed']),
            query_frame_contexts=sum(r['counts']['frames'] for r in state['completed'].values()),
            rows=sum(r['counts']['rows'] for r in state['completed'].values()))), flush=True)

    class Recorder(CompactDecision):
        def accepted_batch(self, group, ranked, inputs):
            self.captured.append((group, [copy(t) for t in ranked]))

    save()
    try:
        torch.set_num_threads(plan['actor_threads']); torch.manual_seed(0)
        stateful = configure_group_reference_sft()
        actor = Recorder(ROOT/plan['actor_checkpoint'], device='cpu')
        while True:
            producer = json.loads((source/'owner_state.json').read_text())
            for name in producer['completed']:
                if name in state['completed']:
                    continue
                index, item = by_name[name]
                manifest = json.loads((ROOT/item['manifest']).read_text())
                assert len(manifest['images']) == item['frames']
                context = context_for_manifest(manifest, source/'clips'/name, plan['label_root'], index)
                state.update(status='EXPORTING_AND_PLANNING_'+name); save()
                destination = out/'clips'/name
                attempt = 1
                while destination.exists():
                    destination = out/'clips'/(name+f'_resume{attempt}'); attempt += 1
                result = export_clip(name, source, destination, plan, actor, stateful, context=context)
                assert result['status'] == 'REAL_CLIP_FULL_V3_GROUP_REFS_PIXELS_AND_ALL_NATIVE_CPU_ROWS_VALIDATED'
                result.update(tracking_task_bucket='rmot', query_role=item['query_role'], video=item['video'],
                    supervision_quality='GT matched candidate IoU proxy, not visibility or native confidence',
                    original_expression_conditioned_rollout=True)
                (destination/'validated_summary.json').write_text(json.dumps(result, indent=2)+'\n')
                state['completed'][name] = result
                export_bytes = sum(p.stat().st_size for directory in (out, scope) for p in directory.rglob('*')
                    if p.is_file() and not p.is_symlink())
                assert export_bytes < plan['scope_and_export_budget_bytes']
                save()
            # Exporting a full clip may outlast its producer's final transition.
            producer = json.loads((source/'owner_state.json').read_text())
            if set(producer['completed'])-state['completed'].keys():
                continue
            if producer['status'] == 'ALL1814_FIT_PHYSICAL_FRAMES_18_NATIVE_QUERY_ROLLOUTS_COMPLETE':
                assert len(state['completed']) == plan['expected_query_rollouts']
                assert sum(v['counts']['frames'] for v in state['completed'].values()) == plan['expected_query_frame_contexts']
                break
            assert producer['status'] != 'FULL_FIT_RMOT_COLLECTION_FAILED', producer.get('error')
            assert alive([producer['owner_pid'], producer['owner_starttick']]), 'Native RMOT source owner exited without complete receipt'
            state['status'] = 'CPU_WAITING_MORE_COMPLETED_NATIVE_RMOT_QUERY_ROLLOUTS'; save(); time.sleep(20)
        state.update(status='ALL18_NATIVE_RMOT_QUERY_ROLLOUTS_FULL_V3_GROUPS_AND_CPU_PLANS_VALIDATED',
            unique_physical_frames=plan['expected_physical_frames'])
        save()
    except Exception as error:
        state.update(status='NATIVE_RMOT_FULL_GROUP_EXPORT_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
