"""Freeze completed official units already validated by their original owner."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--owner',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scopes',nargs='+',required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    state=json.loads((args.owner/'owner_state.json').read_text())
    completed={}
    for scope in args.scopes:
        job=plan['scopes'][scope]
        for arm,policies in plan['arms'].items():
            key=scope+'_'+arm;record=state['completed'][key];run=Path(record['run'])
            official=json.loads((run/'official_metrics.json').read_text())
            runtime=json.loads((run/'result.json').read_text())
            assert record['policies']==policies
            assert record['decoded_frames']==official['frames']==runtime['frames']==job['frames']
            assert record['native_candidates']==job['candidates_count']
            assert record['all_native_geometry_labels_scores_order_and_frames_retained']
            assert record['metrics']==official['metrics'] and record['base_novel']==official.get('base_novel')
            assert record['native_rejected_frames']==official['perception_rejected_frames']==runtime['perception_rejected_frames']
            assert runtime['all_input_frames_processed'] and not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
            assert runtime['end_to_end_fps'] is None and record['speed_comparison_or_E2E_claim'] is False
            assert Path(runtime['final_identity_memory_path']).is_file()
            completed[key]=record
    result=dict(status='ALL_REQUESTED_COMPLETED_OFFICIAL_AFFINITY_UNITS_FROZEN',
        plan=str(args.plan),original_owner=str(args.owner),scopes=args.scopes,completed=completed,
        integrity_scope='Recheck exact original owner-completed receipts against official metrics/runtime/frame/candidate/rejection/final-state records. Full candidate-trace equality and decoded-video coverage were checked by original owner before completing each unit; no repeated inference/scoring/video decode.',
        remaining_scopes=[s for s in plan['scopes'] if s not in args.scopes],
        common_model_promoted=False,new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],units=len(completed),remaining_scopes=result['remaining_scopes'])))


if __name__=='__main__':main()
