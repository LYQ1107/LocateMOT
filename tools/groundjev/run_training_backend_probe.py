"""One blocking GPU5 owner compares native and FLA four-step training units."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    import torch
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    plan = {'name': 'Real multimodal OneJev training-only FLA probe', 'seed': 0, 'physical_gpu': 5,
        'steps_each': 4, 'accum_rows': 5, 'examples_each': 20, 'sampling': 'same5 real rows per step; same native seed/index-shuffle/rendering/loss',
        'first_gradient_gate': {'sampled_cosine_min': .99, 'sampled_relative_L2_max': .1, 'full_norm_relative_difference_max': .1},
        'steady_step_speed_ratio_min': 1.5, 'timing': 'median steps2..4, first compilation/warmup step excluded; concurrent workloads recorded, component timing only',
        'diagnostic_budget_bytes': 32*2**20, 'combined_registered_feature_upper_bytes': 26739639402+32*2**20,
        'global_temporary_feature_budget_bytes': 25*2**30,
        'previous_inference_parity_rejection': 'outputs/groundjev/optimization/onejev_fla_multimodal_v1/result.json',
        'previous_inference_tolerance_unchanged': .001, 'production_reference6000_queue_unchanged': True,
        'no_tracking_accuracy_or_Ground_One_joint_gradient_claim': True, 'full_goal_complete': False}
    (args.out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    def rows(path):
        with gzip.open(path, 'rt', encoding='utf-8') as source:
            for line in source: yield json.loads(line)
    data = ROOT/'outputs/groundjev/training_data'
    mot = list(rows(data/'broader_native_MOT1056_full_v3_groups_v1/clips/MOT17-02-FRCNN/train_refs.jsonl.gz'))
    from_ref = lambda r: r['groundjev_group_question_ref']['group']
    cases = [next(r for r in mot if r['task'] == family and '_136_' in from_ref(r))
             for family in ('identity', 'reliability', 'assigned_identity_memory_update')]
    cases += [next(r for r in rows(data/'rmot_query_diversity_component_v3_official_box_projection/train.jsonl.gz') if r['target'] == {'yes': 1.}),
              next(r for r in rows(data/'base175_streamed_component_v1/train.jsonl.gz') if r['task'] == 'category_leaf')]
    order = list(range(20)); random.Random(0).shuffle(order)
    examples = [None]*20
    for position, index in enumerate(order):
        row = cases[position % 5]
        examples[index] = dict(row, id=f'training_backend_probe_{position}',
                               supervision=dict(row.get('supervision', {}), source_row_id=row['id'], probe_only=True))
    train = args.out/'train.jsonl.gz'
    with gzip.open(train, 'wt', encoding='utf-8') as target:
        for row in examples: target.write(json.dumps(row)+'\n')
    state = {'status': 'REGISTERED_TRAINING_ONLY_BACKEND_PROBE', 'owner_pid': os.getpid(),
        'owner_starttick': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'owner_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'completed': {}, 'physical_gpu': 5, 'production_training_changed': False, 'full_goal_complete': False}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)
    save()
    try:
        assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
        free = int(subprocess.check_output(['nvidia-smi', '-i', '5', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        assert free >= 40*1024, 'Insufficient free GPU5 memory for independent probe; do not stop existing owners'
        state['GPU5_free_MiB_before_launch'] = free
        for mode in ('reference', 'fla'):
            config = json.loads((ROOT/'configs/groundjev/onejev_history_refresh600_refreshed_v1.json').read_text())
            config.update(train=str(train.resolve()), name=mode, runs_dir=str(args.out.resolve()), max_steps=4,
                          accum_rows=5, save_every=0, log_every=1)
            config_path = args.out/(mode+'_config.json'); config_path.write_text(json.dumps(config, indent=2)+'\n')
            command = ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '5', '--',
                str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/probe_onejev_training_backend.py',
                '--mode', mode, '--config', str(config_path)]
            state.update(status=mode+'_REAL_NATIVE_SFT4_STEPS', command=command); save()
            with (args.out/(mode+'.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                state['child_pid'] = child.pid; save()
                for line in child.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                code = child.wait()
            state.pop('child_pid', None)
            assert code == 0, f'{mode} failed; preserve source, first failure and completed unit'
            records = [json.loads(line) for line in (args.out/mode/'log.jsonl').open()]
            done = next(r for r in records if r.get('event') == 'done')
            assert done['steps'] == 4 and done['examples'] == 20
            assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in records if 'loss' in r)
            initial = json.loads((args.out/mode/'initial_fp32_state_receipt.json').read_text())
            assert initial['initialization_max_difference'] == 0 and initial['source_step'] == 600
            state['completed'][mode] = {'done': done, 'steady_step_seconds': [r['sec_per_step'] for r in records if r.get('step', 0) > 1 and 'loss' in r]}; save()
        a, b = [torch.load(args.out/mode/'first_gradient_samples.pt', map_location='cpu', weights_only=True) for mode in ('reference', 'fla')]
        assert a['shapes'] == b['shapes'] and a['samples'].keys() == b['samples'].keys()
        x, y = [torch.cat([record['samples'][name].double() for name in sorted(record['samples'])]) for record in (a, b)]
        cosine = float(torch.dot(x, y)/(x.norm()*y.norm()))
        relative = float((x-y).norm()/x.norm())
        norms = [math.sqrt(sum(v*v for v in record['full_parameter_gradient_norms'].values())) for record in (a, b)]
        norm_difference = abs(norms[0]-norms[1])/norms[0]
        speed = statistics.median(state['completed']['reference']['steady_step_seconds'])/statistics.median(state['completed']['fla']['steady_step_seconds'])
        gradient_ok = cosine >= .99 and relative <= .1 and norm_difference <= .1
        report = {'status': 'TRAINING_BACKEND_CANDIDATE' if gradient_ok and speed >= 1.5 else 'TRAINING_BACKEND_PROBE_REJECTED',
                  'sampled_gradient_cosine': cosine, 'sampled_gradient_relative_L2': relative,
                  'full_gradient_norms': norms, 'full_norm_relative_difference': norm_difference,
                  'steady_step_component_speed_ratio': speed, 'source_plan': plan,
                  'actual_optimizer_steps_each': 4, 'actual_examples_each': 20,
                  'sampled_not_full_gradient_vector_comparison': True,
                  'old_inference_rejection_preserved': True, 'production_training_changed': False, 'full_goal_complete': False}
        (args.out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        state.update(status=report['status'], result=report); save()
    except Exception as error:
        state.update(status='TRAINING_BACKEND_PROBE_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
