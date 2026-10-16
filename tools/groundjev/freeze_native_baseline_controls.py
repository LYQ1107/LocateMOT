"""Freeze all four original shared-native checkpoint baselines after completion."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--owner',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    state=json.loads((args.owner/'owner_state.json').read_text());plan=json.loads(args.plan.read_text())
    completed={};categories=None
    for scope,job in plan['scopes'].items():
        record=state['completed']['before_'+scope];source=Path(record['run'])
        runtime=json.loads((source/'result.json').read_text());official=json.loads((source/'official_metrics.json').read_text())
        assert Path(record['checkpoint']).resolve()==Path(runtime['checkpoint']).resolve()==args.checkpoint.resolve()
        assert runtime['frames']==official['frames']==record['frames']==job['frames']
        assert record['native_candidates']==job['candidates_count'] and record['same_exact_all_candidates']
        assert record['official_metrics']==official['metrics'] and record['base_novel']==official.get('base_novel')
        assert runtime['all_input_frames_processed'] and not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
        assert runtime['perception_rejected_frames']==official['perception_rejected_frames']
        assert runtime['end_to_end_fps'] is None and record['actual_end_to_end_FPS'] is None
        completed[scope]=record
        if scope=='tao_raw_full40':categories=official['evaluated_categories']
    assert len(completed)==4
    report=dict(status='SAME_NATIVE600_CHECKPOINT_ALL_FOUR_REGISTERED_BASELINE_SCOPES_COMPLETE',
        checkpoint=str(args.checkpoint),completed=completed,TAO_evaluated_categories=categories,
        scope='Four original legal video scopes, not complete validation datasets. Original owner validated every native candidate, decoded frame, rejected frame and causal state before completing each unit; this freezer rechecks runtime/official summaries. TAO source is explicit completed baseline reuse, no new inference or E2E timing claim. TAO time is annotation steps with unverified source FPS; MOT/RMOT use physical seconds.',
        native6000_TAO_after_training_complete=False,ordinary_model_performance_achieved=False,
        new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],scopes={k:v['official_metrics'] for k,v in completed.items()})))


if __name__=='__main__':main()
