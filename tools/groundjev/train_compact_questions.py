"""CPU shared Compact adaptation, three original typed questions per update."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from locatemot.groundjev.compact import CompactCore
    from tools.groundjev.compact_question_data import CompactQuestionData
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, help='explicit prefix probe; never a completed full curriculum')
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    tasks = plan.get('tasks', ['mot', 'rmot', 'ovmot'])
    assert tasks and len(set(tasks)) == len(tasks) and set(tasks) <= {'mot', 'rmot', 'ovmot'}
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and plan['device'] == 'cpu'
    torch.set_num_threads(plan['cpu_threads']); torch.manual_seed(plan['seed']); random.seed(plan['seed'])
    torch.use_deterministic_algorithms(True)
    dataset = CompactQuestionData(ROOT / plan['output'])
    proof = json.loads((ROOT / plan['output'] / 'target_validation.json').read_text())
    assert proof['all_selected_original_labels_and_native_question_options_exact']
    out = args.out or Path(plan['training_output'])
    if not args.resume: out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(ROOT / plan['initial_checkpoint'], map_location='cpu', weights_only=True)
    model = CompactCore(checkpoint['architecture'], semantic=checkpoint['semantic'],
        input_evidence=checkpoint.get('input_evidence', 'geometry_appearance_language'),
        native_relation_residual=plan.get('native_relation_residual',False))
    if model.native_relation_residual:
        missing,unexpected=model.load_state_dict(checkpoint['state_dict'],strict=False)
        assert not unexpected and set(missing)=={k for k in model.state_dict() if k.startswith(('native_pair.','native_actions.'))}
    else:model.load_state_dict(checkpoint['state_dict'], strict=True)
    trainable_prefixes=plan.get('trainable_parameter_prefixes')
    if trainable_prefixes:
        for name,parameter in model.named_parameters():parameter.requires_grad_(name.startswith(tuple(trainable_prefixes)))
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
    protocol = dict(plan=plan, curriculum_summary=dataset.summary, initialization=str((ROOT / plan['initial_checkpoint']).resolve()),
        deterministic_CPU=True, task_order=['mot', 'rmot', 'ovmot'])
    start_step = 0; elapsed_before = 0.; exposure = Counter()
    if args.resume:
        saved = torch.load(out / 'latest_training_state.pt', map_location='cpu', weights_only=False)
        assert saved['protocol'] == protocol
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']); random.setstate(saved['python_rng'])
        start_step = saved['step']; elapsed_before = saved['training_seconds']; exposure.update(saved['exposure'])
    stop = args.stop_after or plan['optimizer_steps']
    assert start_step < stop <= plan['optimizer_steps']
    state = dict(status='TRAINING_BROADER_SHARED_COMPACT_TYPED_QUESTIONS', owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        step=start_step, target_steps=plan['optimizer_steps'], actual_questions=len(tasks)*start_step,
        CPU_only=True, new_GPU_processes=0, full_goal_complete=False)
    def save_state():
        temporary = out / 'owner_state.tmp'; temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(out / 'owner_state.json'); print(json.dumps(state), flush=True)
    save_state(); started = time.perf_counter(); model.train()
    try:
        with (out / 'log.jsonl').open('a') as log:
            for step in range(start_step + 1, stop + 1):
                optimizer.zero_grad(); losses = {}
                for row in dataset.rows[3*(step-1):3*step]:
                    if row['unit'].split('/')[0] not in tasks: continue
                    inputs = dataset.inputs(row); outputs = model(inputs)
                    loss = dataset.loss(outputs, row) * plan['loss_coefficients'][row['task']]
                    if row['task'] in plan.get('label_loss_weights',{}):
                        loss = loss * plan['label_loss_weights'][row['task']][row['label']]
                    assert torch.isfinite(loss)
                    (loss / 3).backward(); losses[row['unit'].split('/')[0]] = float(loss.detach())
                    exposure[row['unit'].split('/')[0] + ':' + row['task']] += 1
                grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), plan['gradient_clip']))
                assert math.isfinite(grad) and grad > 0
                optimizer.step()
                elapsed = elapsed_before + time.perf_counter() - started
                if step == 1 or step % 100 == 0 or step == stop:
                    record = dict(step=step, actual_questions=len(tasks)*step, weighted_task_losses=losses,
                        mean_loss=sum(losses.values())/3, unclipped_gradient_norm=grad, training_seconds=elapsed)
                    log.write(json.dumps(record) + '\n'); log.flush(); print(json.dumps(record), flush=True)
                if step % plan['save_every'] == 0 or step == stop:
                    temporary = out / 'latest_training_state.tmp'
                    torch.save(dict(step=step, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(), python_rng=random.getstate(), exposure=dict(exposure),
                        training_seconds=elapsed, protocol=protocol), temporary)
                    temporary.replace(out / 'latest_training_state.pt')
                    state.update(step=step, actual_questions=len(tasks)*step); save_state()
        if stop != plan['optimizer_steps']:
            state.update(status='EXPLICIT_COMPACT_TRAINING_PREFIX_PROBE_COMPLETE_FULL_CURRICULUM_INCOMPLETE')
            save_state(); return
        model.eval()
        result = dict(architecture=checkpoint['architecture'], semantic=True, state_dict=model.state_dict(),
            steps=stop, seed=plan['seed'], tasks=tasks, Base_only_OV_supervision=True,
            text_features='real original OneJev frozen token mean; not language decoder reasoning',
            training_protocol='native-compact-typed-question-balanced-three-task-v1')
        if model.native_relation_residual:result['native_relation_residual']=True
        torch.save(result, out / 'checkpoint.pt')
        fresh = CompactCore(checkpoint['architecture'], semantic=True,native_relation_residual=model.native_relation_residual)
        fresh.load_state_dict(torch.load(out / 'checkpoint.pt', weights_only=True)['state_dict']); fresh.eval()
        differences = {}
        for row in dataset.rows[-3:]:
            inputs = dataset.inputs(row)
            with torch.no_grad(): before = model(inputs); after = fresh(inputs)
            delta = max(float((before[k]-after[k]).abs().max()) for k in before if before[k].numel())
            assert delta == 0; differences[row['unit'].split('/')[0]] = delta
        changed = {k: int((initial[k] != v).sum()) for k, v in model.state_dict().items()}
        expected = [] if trainable_prefixes else ['visual.1.weight', 'text.1.weight']
        if 'mot' in tasks: expected.append('native_pair.0.weight' if model.native_relation_residual else 'pair.0.weight')
        if model.native_relation_residual:expected.append('native_actions.0.weight')
        if 'rmot' in tasks: expected.append('membership.0.weight')
        if 'ovmot' in tasks: expected.append('category.weight')
        assert all(changed[k] > 0 for k in expected)
        if trainable_prefixes:
            assert all(changed[k]==0 for k in changed if not k.startswith(tuple(trainable_prefixes)))
        if 'rmot' not in tasks: assert all(v==0 for k,v in changed.items() if k.startswith('membership.'))
        if 'ovmot' not in tasks: assert all(v==0 for k,v in changed.items() if k.startswith('category.'))
        assert not torch.cuda.is_initialized()
        state.update(status='BROADER_SHARED_COMPACT_TYPED_QUESTION_TRAINING_COMPLETE',
            exposure=dict(exposure), training_seconds=elapsed, parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_backbone_and_original_association_unchanged=bool(trainable_prefixes),
            changed_elements=sum(changed.values()), changed_elements_by_parameter=changed,
            same_final_checkpoint_for_all_tasks=str(out / 'checkpoint.pt'),
            three_task_reload_probability_max_difference=differences, official_tracking_results='pending',
            source_scope={k:v for k,v in dataset.summary['unique_physical_first_media_images'].items() if k in tasks})
        save_state()
    except Exception as error:
        state.update(status='BROADER_COMPACT_QUESTION_TRAINING_FAILED', error=repr(error)); save_state(); raise


if __name__ == '__main__': main()
