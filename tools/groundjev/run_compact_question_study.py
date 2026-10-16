"""One CPU owner for broader shared Compact training and five causal controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--control-protocol', type=Path, default=ROOT/'configs/groundjev/compact_own_history_refresh1200_pair_v1.json')
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    prefix = json.loads((ROOT/plan['output']/'training_prefix_validation.json').read_text())
    assert prefix['all_model_parameters_Adam_moments_counters_and_RNG_bit_exact']
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    controls_source = json.loads(args.control_protocol.read_text())
    assert (ROOT/controls_source['checkpoint']).resolve() == (ROOT/plan['initial_checkpoint']).resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'registration.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = dict(status='REGISTERED_BROADER_COMPACT_CPU_JOINT_STUDY', owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        plan=str(args.plan), physical_GPUs=[], completed={}, full_goal_complete=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONPATH=str(ROOT/'external/OneJev'), PYTHONUNBUFFERED='1')
    python = str(ROOT/'.venv-onejev-py310/bin/python')
    def save():
        temporary=args.out/'owner_state.tmp'; temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(args.out/'owner_state.json')
        print(json.dumps(dict(status=state['status'], completed=list(state['completed']))), flush=True)
    def run(stage, command):
        state.update(status=stage, command=command); save()
        with (args.out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['child_pid']=child.pid; save()
            for line in child.stdout: log.write(line); log.flush(); print(line,end='',flush=True)
            code=child.wait()
        state.pop('child_pid',None)
        assert code==0, f'{stage} exited{code}; retain first failure, checkpoint and completed scopes'
    save()
    try:
        run('BROADER_SHARED8000_CPU_TRAINING',[python,'tools/groundjev/train_compact_questions.py','--plan',str(args.plan)])
        training=Path(plan['training_output']); result=json.loads((training/'owner_state.json').read_text())
        assert result['status']=='BROADER_SHARED_COMPACT_TYPED_QUESTION_TRAINING_COMPLETE'
        assert result['step']==8000 and result['actual_questions']==24000
        assert all(v==0 for v in result['three_task_reload_probability_max_difference'].values())
        checkpoint=torch.load(training/'checkpoint.pt',weights_only=True,map_location='cpu')
        latest=torch.load(training/'latest_training_state.pt',weights_only=False,map_location='cpu')
        assert latest['step']==8000 and latest['state_dict'].keys()==checkpoint['state_dict'].keys()
        for name,tensor in checkpoint['state_dict'].items():
            assert torch.equal(tensor,latest['state_dict'][name]) and torch.isfinite(tensor).all()
        for entry in latest['optimizer']['state'].values():
            assert 1<=int(entry['step'])<=8000
            assert torch.isfinite(entry['exp_avg']).all() and torch.isfinite(entry['exp_avg_sq']).all()
        assert sum(latest['exposure'].values())==24000
        state['completed']['training']=dict(step=8000,questions=24000,exposure=latest['exposure'],
            full_model_and_optimizer_finite=True,final_export_equals_full_state=True,
            three_task_reload_max_difference=0,checkpoint=str(training/'checkpoint.pt'),
            training_seconds=result['training_seconds'],source_scope=result['source_scope'])
        (args.out/'training_validation.json').write_text(json.dumps(state['completed']['training'],indent=2)+'\n')
        del latest,checkpoint; save()
        references=args.out/'reference_runs'; references.mkdir()
        for scope,source in controls_source['before_CPU_reference_runs'].items():
            runtime=json.loads((ROOT/source/'result.json').read_text())
            assert runtime['decision_device']=='cpu' and runtime['frames']==controls_source['scopes'][scope]['frames']
            assert Path(runtime['checkpoint']).resolve()==(ROOT/plan['initial_checkpoint']).resolve()
            (references/(scope+'_before')).symlink_to((ROOT/source).resolve(),target_is_directory=True)
        policy=dict(association='learned',auxiliary='learned',memory_policy='gated',feature_view='all',lost_seconds=3.,shortlist=32)
        controls=dict(regression=controls_source['control_regression'],checkpoint=plan['initial_checkpoint'],
            reference_root=str(references.resolve()),scopes=controls_source['scopes'],cpu_blas_threads=4,
            arms=dict(before=dict(policy,reference_arm='before'),broader=dict(policy,checkpoint=str(training/'checkpoint.pt'))),
            comparison_limit='Broader supervised shared Compact versus its original frozen actor; new balanced question-level recipe, not isolated data-only or token/compute matched. Same final checkpoint and policies across five registered scopes; no full-benchmark parity or E2E speed claim.',
            memory_storage='Reuse immutable native source banks; keep scientific decision traces and final binary identity state, no new candidate feature archives.')
        path=args.out/'resolved_controls_plan.json'; path.write_text(json.dumps(controls,indent=2)+'\n')
        run('FIVE_SCOPE_OFFICIAL_CONTROLS',[python,'tools/groundjev/run_fixed_affinity_controls.py',
            '--plan',str(path),'--out',str(args.out/'controls')])
        final=json.loads((args.out/'controls/owner_state.json').read_text())
        assert final['status']=='ALL_FIXED_AFFINITY_CONTROLS_COMPLETE' and len(final['completed'])==10
        state.update(status='BROADER_COMPACT_SHARED8000_AND_ALL_FIVE_CAUSAL_CONTROLS_VALIDATED')
        state['completed']['controls']=final['completed']; save()
    except Exception as error:
        state.update(status='BROADER_COMPACT_QUESTION_STUDY_FAILED',error=repr(error)); save(); raise


if __name__=='__main__': main()
