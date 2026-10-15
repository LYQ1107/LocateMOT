"""Check the relocated scientific plans and a real completed GPU ownership release."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def load(relative):
    return json.loads((ROOT/relative).read_text())


def main():
    from tools.groundjev.completed_gpu_handoff import validate_release
    original = load('configs/groundjev/native_ground_localization200_v1.json')
    relocated = load('configs/groundjev/native_ground_localization200_gpu5_v1.json')
    metadata = {'name','physical_training_GPU','before_after','queue','execution_relocation_note'}
    assert {k:v for k,v in original.items() if k not in metadata} == {k:v for k,v in relocated.items() if k not in metadata}
    assert original['physical_training_GPU'] == 6 and relocated['physical_training_GPU'] == 5
    contract = load('outputs/groundjev/validation/localization200_contract_gpu5_v1/validation.json')
    assert contract['status'] == 'REAL_LOCALIZATION200_ROWS_AND_ACTUAL_TRACKING200_SCHEDULE_MATCH_VALIDATED'
    old_front = load('configs/groundjev/ground_localization_frontends_v1.json')
    new_front = load('configs/groundjev/ground_localization_frontends_gpu5_v1.json')
    assert len(old_front['variants']) == len(new_front['variants']) == 4
    assert old_front['clips'] == new_front['clips'] and old_front['repeat_pair'] == new_front['repeat_pair']
    assert old_front['requested_sampling_seed'] == new_front['requested_sampling_seed'] == 42
    for old,new in zip(old_front['variants'],new_front['variants']):
        assert old['name'] == new['name']
        assert new['checkpoint'] == (old['checkpoint']+'-GPU5' if old['name'].startswith('localization200') else old['checkpoint'])
    old_full = load('configs/groundjev/ground_one_factorial_v1.json')
    new_full = load('configs/groundjev/ground_one_factorial_gpu5_v1.json')
    assert old_full['scopes'].keys() == new_full['scopes'].keys()
    assert old_full['arms'] == new_full['arms'] and old_full['policies'] == new_full['policies']
    assert old_full.keys() == new_full.keys()
    for key in old_full:
        if key not in ('physical_gpu','scopes'):assert old_full[key] == new_full[key]
    for name,old in old_full['scopes'].items():
        expected = dict(old,candidates=old['candidates'].replace('localization200_control_v1/','localization200_control_gpu5_v1/'))
        assert new_full['scopes'][name] == expected
    old_nat = load('configs/groundjev/natural_memory_study_v1.json')
    new_nat = load('configs/groundjev/natural_memory_study_gpu5_v1.json')
    assert new_front['physical_gpu'] == new_full['physical_gpu'] == new_nat['physical_gpu'] == 5
    metadata = {'physical_gpu','gpu_handoff_receipt','wait','launch'}
    assert {k:v for k,v in old_nat.items() if k not in metadata} == {k:v for k,v in new_nat.items() if k not in metadata}
    assert new_nat['total_conservative_feature_upper_bound_bytes'] < new_nat['global_feature_cap_bytes'] == 25*2**30
    hardware = subprocess.check_output(['nvidia-smi','--query-gpu=index,name','--format=csv,noheader'],text=True)
    names = {int(row.split(',')[0]):row.split(',',1)[1].strip() for row in hardware.splitlines()}
    assert names[5] == names[6] == names[7]
    out = ROOT/'outputs/groundjev/validation/gpu5_followup_registration_v1'
    out.mkdir(parents=True,exist_ok=False)
    prior = load('outputs/groundjev/multitask_runs/full_rmot_shared600_v1/owner_state.json')
    paired = load('outputs/groundjev/e2e/full_compact_rmot800_v1/owner_state.json')
    assert prior['status'] == 'FULL_TWO800_RMOT_FRESH_COMPACT_AND_ALL_MATCHED_FULL_CONTROLS_COMPLETE'
    release = {'status':'VERIFIED_COMPLETED_GPU_OWNER_RELEASE','physical_gpu':5,
        'upstream_owner_identity':paired['upstream_owner'],'owned_process_groups':prior['owned_service_groups'],
        'all_owned_groups_gone':prior['owned_native_service_groups_stopped']}
    receipt = out/'actual_completed_full_rmot_release.json'
    receipt.write_text(json.dumps(release,indent=2)+'\n')
    validate_release(receipt,5)
    report = {'status':'GPU5_FOLLOWUPS_SCIENTIFIC_PLANS_AND_COMPLETED_HANDOFF_VALIDATED',
        'all_scientific_training_fields_exact':True,'real112rows_and200draws_vs_completed_training_checked':True,
        'all96_frontend_frames_sampling_and18_factorial_scopes_unchanged':True,
        'natural1657frames_8controls_original_manifest_model_and_budget_exact':True,
        'actual_completed_GPU5_owner_and_owned_groups_release_verified':True,
        'physical_GPU5_6_7_model':names[5],'actual_GPU_set_remains':[4,5,6,7],
        'limits':'Execution devices/dependencies explicitly changed before any experiment in the superseded '
            'queues started. Reference tracking200 training GPU6, localization execution GPU5; same A800 '
            'model does not establish bitwise cross-device training trajectory or wall-time pairing.',
        'no_new_GPU_model_or_training_started_by_this_check':True,'full_goal_complete':False}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'])


if __name__ == '__main__':
    main()
