"""Validate completed localization-only training and controlled Ground/One scopes."""
from itertools import zip_longest
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    directory = ROOT/'outputs/groundjev/ground_adaptation/localization200_control_gpu5_v1'
    owner = json.loads((directory/'owner_state.json').read_text())
    assert owner['status'] == 'LOCALIZATION200_AND_CONTROLLED_VISUAL_ONE_COMPOSITION_COMPLETE'
    parent = json.loads((ROOT/'outputs/groundjev/gpu5_followups_v1/owner_state.json').read_text())
    assert parent['completed']['LOCALIZATION200']['status'] == owner['status']
    trained = json.loads((directory/'training/owner_state.json').read_text())
    assert trained['status'] == 'NATIVE_GROUND_LOCALIZATION200_TRAINING_COMPLETE'
    assert trained['steps'] == 200 and trained['trainable_parameters'] == 27267584
    assert trained['physical_gpu'] == 5 and trained['seed'] == 42
    assert trained['identity_loss_coefficients'] == {'teacher':0.,'native_proposals':0.}
    assert trained['changed_BF16_elements'] > 0 and trained['checkpoint_reload_max_difference'] == 0
    logs = [json.loads(line) for line in (directory/'training/log.jsonl').open()]
    assert [r['step'] for r in logs] == list(range(1,201))
    assert all(r['total_loss'] == r['native_loss'] and math.isfinite(r['total_loss']) for r in logs)
    reference = torch.load(ROOT/'outputs/groundjev/ground_adaptation/native_tracking200_v7_boundary_trace/projector_delta.pt',map_location='cpu',weights_only=True)
    assert [r['pair_index'] for r in logs] == reference['schedule']
    del reference
    exported = json.loads((directory/'bundle_export.json').read_text())
    assert exported['native_tensor_count'] == 569 and exported['all_other_native_tensors_unmodified']
    assert not exported['source_bundle_overwritten']
    front = json.loads((directory/'native_frontends/owner_state.json').read_text())
    assert front['status'] == 'ISOLATED_ORDERED_GROUND_CONTROLS_COMPLETE' and len(front['completed']) == 12
    assert not front['shared_full9_services_touched']
    assert all(r['same_raw_wire_candidates_and_outputs'] for r in front['repeat_checks'].values())
    clips = {'dense_mot4':4,'refer_single8':8,'refer_multi12':12}
    front_rows = {}
    for variant,root in front['variant_roots'].items():
        for scope,n in clips.items():
            run = Path(root)/scope
            actual = json.loads((run/'result.json').read_text())
            official = json.loads((run/'official_metrics.json').read_text())
            assert official == front['completed'][variant+'_'+scope]
            assert actual['frames'] == official['frames'] == n and actual['all_input_frames_processed']
            assert not actual['gt_read_by_runtime'] and not actual['future_frames_read_by_runtime']
            assert (run/'tracks.mp4').is_file()
            front_rows[variant+'_'+scope] = {'run':str(run),'frames':n,'metrics':official['metrics'],
                'parser_rejected_frames':official['perception_rejected_frames']}
    factorial = json.loads((directory/'factorial/owner_state.json').read_text())
    assert factorial['status'] == 'SAME_POLICY_GROUND_ONE_FACTORIAL_CONTROLS_COMPLETE'
    assert len(factorial['completed']) == 18 and not factorial['failed']
    plan = json.loads((ROOT/'configs/groundjev/ground_one_factorial_gpu5_v1.json').read_text())
    full_rows = {}
    for one in plan['arms']:
        for scope,job in plan['scopes'].items():
            name = one+'_'+scope
            entry = factorial['completed'][name]
            run = Path(entry['run'])
            actual = json.loads((run/'result.json').read_text())
            official = json.loads((run/'official_metrics.json').read_text())
            assert official == entry['official']
            assert actual['frames'] == official['frames'] == entry['decoded_frames']
            assert actual['all_input_frames_processed'] and entry['all_exact_native_candidates_retained']
            assert not actual['gt_read_by_runtime'] and not actual['future_frames_read_by_runtime']
            assert actual['end_to_end_fps'] is None
            assert Path(actual['checkpoint']).resolve() == (ROOT/plan['arms'][one]['checkpoint']).resolve()
            count = 0
            with (ROOT/job['candidates']).open() as source,(run/'frame_decisions.jsonl').open() as output:
                for before,after in zip_longest(source,output):
                    assert before is not None and after is not None
                    before,after = json.loads(before),json.loads(after)
                    assert before['frame'] == after['frame'] and before['image_path'] == after['image_path']
                    assert before['candidates'] == after['candidates']
                    count += 1
            assert count == actual['frames']
            full_rows[name] = {'run':str(run),'frames':count,'metrics':official['metrics'],
                'same_exact_frontend_candidates_in_both_One_arms':True,
                'parser_rejected_frames':official['perception_rejected_frames']}
    report = {'status':'LOCALIZATION200_NATIVE96_AND_FULL18_OFFICIAL_ARTIFACTS_VALIDATED',
        'training':{'steps':200,'seed':42,'physical_gpu':5,'parameters':27267584,
            'actual200_pair_indices_equal_completed_tracking200':True,'all200_total_losses_equal_native_loss':True,
            'BF16_changed_elements':trained['changed_BF16_elements'],'checkpoint_reload_max_difference':0},
        'export':exported,'native_frontends':front_rows,'localization_repeat_checks':front['repeat_checks'],
        'Full_composition':full_rows,'full_goal_complete':False,
        'limits':'Short MOT4/RMOT8/RMOT12 scopes only;96native frontend frames and18Full scopes. '
            'Tracking200 reference trainedGPU6/localizationGPU5, same A800/scientific rows/schedule '
            'but not bitwise trajectories. Frontends legitimately change proposals; same candidates '
            'within each two-One comparison. Composed separately adapted weights, not joint-gradient '
            'optimization. No full-dataset/Novel/retention/promotion or E2E speed claim from these scopes.'}
    (directory/'validated_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])


if __name__ == '__main__':
    main()
