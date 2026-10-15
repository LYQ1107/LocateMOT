"""Paired finite shared-task curricula, replacing only native MOT actor history."""
import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return [json.loads(line) for line in path.open()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    refresh = ROOT/'outputs/groundjev/training_data/native600_fit_rollout_refresh_v1'
    receipt = json.loads((refresh/'owner_state.json').read_text())
    assert receipt['status']=='CURRENT_NATIVE600_FIT48_REAL_ROLLOUT_COMPLETE'
    assert receipt['rollout_exit_code']==0 and not receipt['new_DLM_requests']
    audit = json.loads((ROOT/'outputs/groundjev/diagnostics/full_fit_update_supervision_native600_v1/audit.json').read_text())
    assert audit['actual_existing_identity_assignments']==658 and audit['counts']['frames']==48
    source = ROOT/'outputs/groundjev/training_data/onejev_native_mixture200_v2_family_matched/train.jsonl'
    selected = set(random.Random(0).sample(range(20881),400))
    mot = []
    with source.open() as lines:
        for index,pair in enumerate(zip_longest(lines,lines)):
            assert None not in pair
            if index not in selected:
                continue
            teacher,native = map(json.loads,pair)
            assert teacher['id'].startswith('teacher_') and native['id'].startswith('native_')
            assert teacher['task']==native['task']
            mot.extend([teacher,native])
    assert index+1==20881 and len(mot)==800
    pools = defaultdict(list)
    fresh_rows = load(refresh/'rollout/train.jsonl')
    assert len(fresh_rows)==5741
    for row in fresh_rows:
        pools[row['task']].append(row)
    rng = random.Random(0)
    for family in sorted(pools):
        rng.shuffle(pools[family])
    replacement,used = [],Counter()
    for row in mot:
        if row['id'].startswith('native_'):
            family = row['task']
            replacement.append(pools[family][used[family]%len(pools[family])])
            used[family]+=1
        else:
            replacement.append(row)
    semantic_sources = {
        'rmot':ROOT/'outputs/groundjev/training_data/onejev_native_rmot_membership_v2_registered_release_root/train.jsonl',
        'ovmot':ROOT/'outputs/groundjev/training_data/onejev_base_categories_v1/train.jsonl'}
    semantic = {k:load(p) for k,p in semantic_sources.items()}
    assert {k:len(v) for k,v in semantic.items()}=={'rmot':346,'ovmot':430}
    for key in sorted(semantic):
        rng.shuffle(semantic[key])
    variants = {}
    for name,history in [('old',mot),('refreshed',replacement)]:
        rows=[]
        for i in range(800):
            for bucket,pool in [('mot',history),*semantic.items()]:
                original=pool[i%len(pool)]
                row=dict(original,id=f'history_refresh_{bucket}_{i}')
                row['supervision']=dict(original.get('supervision',{}),
                    tracking_task_bucket=bucket,source_row_id=original['id'],
                    history_refresh_study='matched_native_actor_history_v1')
                rows.append(row)
        variants[name]=rows
    for old,new in zip(variants['old'],variants['refreshed']):
        assert old['id']==new['id'] and old['task']==new['task']
        original=old['supervision']['source_row_id']
        changed=old!=new
        assert changed == (old['supervision']['tracking_task_bucket']=='mot' and original.startswith('native_'))
    serialized={name:''.join(json.dumps(row)+'\n' for row in rows) for name,rows in variants.items()}
    prior=25725928890+receipt['exported_bytes']
    new_bytes=sum(len(s.encode()) for s in serialized.values())
    upper=prior+new_bytes+16*2**20
    assert upper<25*2**30,(prior,new_bytes,upper)
    args.out.mkdir(parents=True,exist_ok=False)
    configs={}
    for name,rows in variants.items():
        directory=args.out/name;directory.mkdir()
        (directory/'train.jsonl').write_text(serialized[name])
        config=json.loads((ROOT/'configs/groundjev/onejev_unified600_native_v2.json').read_text())
        config.update(model=str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600'),
            train=str((directory/'train.jsonl').resolve()),name=f'onejev_history_refresh600_{name}_v1')
        path=ROOT/f'configs/groundjev/{config["name"]}.json'
        assert not path.exists()
        path.write_text(json.dumps(config,indent=2)+'\n');configs[name]=str(path)
    report={'status':'MATCHED_HISTORY_REFRESH_CURRICULA_PREPARED_NOT_TRAINED',
        'hypothesis':'Refresh actual native MOT actor states while preserving paired shared-task supervision.',
        'configs':configs,'rows_each':2400,'steps_each':600,'seed':0,
        'task_counts_each':dict(Counter(r['supervision']['tracking_task_bucket'] for r in variants['old'])),
        'question_families_each':dict(Counter(r['task'] for r in variants['old'])),
        'replaced_native_MOT_rows':400,'unchanged_teacher_MOT_rows':400,
        'actual_refreshed_family_exposures':dict(used),
        'unique_refreshed_source_rows':len({r['id'] for r in replacement if not r['id'].startswith('teacher_')}),
        'unique_refreshed_fit_frames':len({(r['source'],r['supervision']['frame']) for r in replacement if not r['id'].startswith('teacher_')}),
        'all_common_semantic_payloads_and_per_index_families_equal':True,
        'all_RMOT_OVMOT_source_rows_included':{k:len({r['supervision']['source_row_id'] for r in variants['old'] if r['supervision']['tracking_task_bucket']==k}) for k in semantic},
        'initial_FP32_state':str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/latest_training_state.pt'),
        'expected_initial_step':600,'optimizer_scheduler_RNG':'explicit same new-stage recipe reset in both arms',
        'source_old_native_mixture':str(source),'source_refreshed':str(refresh/'rollout/train.jsonl'),
        'native_observation_Q3_labels_or_runtime_policy_changed':False,
        'new_JSONL_bytes':new_bytes,'metadata_reserve_bytes':16*2**20,
        'combined_registered_upper_bytes':upper,'global_temporary_feature_budget_bytes':25*2**30,
        'new_model_calls':False,'new_feature_or_image_cache':False,
        'token_or_FLOP_matched':False,'large_scale_unique_video_or_joint_foundation_training':False,
        'ordinary_model_performance_achieved':False,'full_goal_complete':False}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
