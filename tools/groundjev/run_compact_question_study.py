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
    parser.add_argument('--initial-parity-proof',type=Path)
    parser.add_argument('--control-protocol', type=Path, default=ROOT/'configs/groundjev/compact_own_history_refresh1200_pair_v1.json')
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    prefix_path = ROOT/plan.get('training_prefix_validation',plan['output']+'/training_prefix_validation.json')
    prefix = json.loads(prefix_path.read_text())
    assert prefix['all_model_parameters_Adam_moments_counters_and_RNG_bit_exact']
    assert prefix['actual_questions_per_branch']==8*len(plan.get('tasks',['mot','rmot','ovmot']))
    if plan.get('native_relation_residual'):
        assert args.initial_parity_proof is not None
        initial=json.loads(args.initial_parity_proof.read_text())
        assert initial['all_declared_untrained_model_output_equalities_exact']
        assert initial['original_every_output_state_and_assignment_exact'] and initial['frames']==1203 and initial['candidates']==13664
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    controls_source = json.loads(args.control_protocol.read_text())
    assert (ROOT/controls_source['checkpoint']).resolve() == (ROOT/plan['initial_checkpoint']).resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'registration.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = dict(status='REGISTERED_BROADER_COMPACT_CPU_JOINT_STUDY', owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        plan=str(args.plan), physical_GPUs=[], selected_tasks=plan.get('tasks',['mot','rmot','ovmot']),
        completed={}, full_goal_complete=False)
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
        actual_questions=8000*len(plan.get('tasks',['mot','rmot','ovmot']))
        assert result['step']==8000 and result['actual_questions']==actual_questions
        assert all(v==0 for v in result['three_task_reload_probability_max_difference'].values())
        checkpoint=torch.load(training/'checkpoint.pt',weights_only=True,map_location='cpu')
        latest=torch.load(training/'latest_training_state.pt',weights_only=False,map_location='cpu')
        assert latest['step']==8000 and latest['state_dict'].keys()==checkpoint['state_dict'].keys()
        for name,tensor in checkpoint['state_dict'].items():
            assert torch.equal(tensor,latest['state_dict'][name]) and torch.isfinite(tensor).all()
        for entry in latest['optimizer']['state'].values():
            assert 1<=int(entry['step'])<=8000
            assert torch.isfinite(entry['exp_avg']).all() and torch.isfinite(entry['exp_avg_sq']).all()
        assert sum(latest['exposure'].values())==actual_questions
        state['completed']['training']=dict(step=8000,questions=actual_questions,exposure=latest['exposure'],
            full_model_and_optimizer_finite=True,final_export_equals_full_state=True,
            three_task_reload_max_difference=0,checkpoint=str(training/'checkpoint.pt'),
            training_seconds=result['training_seconds'],source_scope=result['source_scope'])
        (args.out/'training_validation.json').write_text(json.dumps(state['completed']['training'],indent=2)+'\n')
        if plan.get('native_relation_residual'):
            prefixes=tuple(plan['trainable_parameter_prefixes'])
            original=torch.load(ROOT/plan['initial_checkpoint'],map_location='cpu',weights_only=True)
            fixed=[k for k in original['state_dict'] if not k.startswith(prefixes)]
            assert all(torch.equal(original['state_dict'][k],checkpoint['state_dict'][k]) for k in fixed)
            ablation=dict(checkpoint,native_relation_residual=False,
                state_dict={k:v for k,v in checkpoint['state_dict'].items() if not k.startswith(('native_pair.','native_actions.'))},
                inference_ablation_of_checkpoint=str(training/'checkpoint.pt'),separate_training_claim=False)
            torch.save(ablation,training/'residual_disabled_checkpoint.pt')
            state['completed']['frozen_core_and_ablation']=dict(all_declared_fixed_weights_exact=True,
                fixed_parameter_tensors=len(fixed),ablation=str(training/'residual_disabled_checkpoint.pt'),
                same_final_semantic_head_weights=True,separate_training=False)
            (args.out/'frozen_core_and_ablation_validation.json').write_text(json.dumps(state['completed']['frozen_core_and_ablation'],indent=2)+'\n')
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
            comparison_limit=plan.get('comparison_limit','Broader supervised shared Compact versus its original frozen actor; new balanced question-level recipe, not isolated data-only or token/compute matched. Same final checkpoint and policies across five registered scopes; no full-benchmark parity or E2E speed claim.'),
            memory_storage='Reuse immutable native source banks; keep scientific decision traces and final binary identity state, no new candidate feature archives.')
        if plan.get('native_relation_residual'):
            controls['arms']['broader']['native_relation_residual']=True
            controls['arms']['residual_disabled']=dict(policy,checkpoint=str(training/'residual_disabled_checkpoint.pt'),native_relation_residual=False)
        path=args.out/'resolved_controls_plan.json'; path.write_text(json.dumps(controls,indent=2)+'\n')
        run('FIVE_SCOPE_OFFICIAL_CONTROLS',[python,'tools/groundjev/run_fixed_affinity_controls.py',
            '--plan',str(path),'--out',str(args.out/'controls')])
        final=json.loads((args.out/'controls/owner_state.json').read_text())
        assert final['status']=='ALL_FIXED_AFFINITY_CONTROLS_COMPLETE' and len(final['completed'])==len(controls['arms'])*len(controls['scopes'])
        state.update(status='BROADER_COMPACT_SHARED8000_AND_ALL_FIVE_CAUSAL_CONTROLS_VALIDATED')
        state['completed']['controls']=final['completed']; save()
    except Exception as error:
        state.update(status='BROADER_COMPACT_QUESTION_STUDY_FAILED',error=repr(error)); save(); raise


if __name__=='__main__': main()
