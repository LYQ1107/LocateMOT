"""Audit a new label version; retain all V1 labels and results unchanged."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.global_identity_data_v2 import training_frames_v2
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());total=Counter();examples=[]
    for frame in training_frames_v2(plan):
        labels=frame['labels'];m=len(frame['tracks']);total.update(labels['entity_counts'])
        total['frames']+=1;total['candidates']+=len(frame['candidates'])
        for i,(before,after) in enumerate(zip(labels['original_v1_action_options'],labels['action_options'])):
            total['old_duplicate_REJECT_now_valid_entity_identity']+=before==[m+1] and bool(after)
            total['old_UNKNOWN_now_first_effective_NEW']+=not before and after==[m]
            total['old_UNKNOWN_now_confirmed_MATCH']+=not before and bool(after) and after!=[m]
            if not before and after==[m] and len(examples)<6:
                examples.append(dict(unit=frame['unit'],frame=frame['frame'],candidate=i,
                    unique_GT=int(labels['unique_GT'][i]),old_options=before,new_options=after,
                    source_predicted_history_unchanged=True))
    assert total['frames']==1056 and total['candidates']==48051
    report=dict(status='ALL43_ENTITY_VS_REPRESENTATIVE_AND_EFFECTIVE_BIRTH_V2_AUDITED',totals=dict(total),examples=examples,
        label_version='entity_identity_separate_from_proposal_winner-v2',
        mechanism='V1 penalizes duplicate-supported proposals as identity REJECT, while StageA should own proposal exclusivity. V1 also marks GT seen without a confirmed model birth as previously tracked. V2 supervises identity on every uniquely supported proposal; binds external birth GT for any unique-support predicted birth, not just the Hungarian winning proposal; and uses confirmed effective births rather than annotation presence. Competition ranking still selects one output entity. Unmatched/ambiguous and retired/unbound cases remain masked; no model GT repair.',
        original_V1_labels_checkpoints_or_predictions_modified=False,new_GPU_processes=0,new_feature_bank_bytes=0,
        V2_training_or_tracking_gain_proven=False,full_goal_complete=False)
    path=ROOT/plan['output']/'effective_birth_v2_audit.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
