"""Real three-task prefix versus restored full Compact optimizer trajectory."""
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
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text()); tasks = plan.get('tasks', ['mot', 'rmot', 'ovmot'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    for name, stop, resume in [('continuous', 8, False), ('restored', 4, False), ('restored', 8, True)]:
        command = [sys.executable, 'tools/groundjev/train_compact_questions.py', '--plan', str(args.plan),
            '--out', str(args.out / name), '--stop-after', str(stop)]
        if resume: command.append('--resume')
        with (args.out / f'{name}_{stop}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps(dict(status='REAL_COMPACT_THREE_TASK_PREFIX_FINISHED', branch=name, step=stop)), flush=True)
    a = torch.load(args.out / 'continuous/latest_training_state.pt', weights_only=False, map_location='cpu')
    b = torch.load(args.out / 'restored/latest_training_state.pt', weights_only=False, map_location='cpu')
    assert a['step'] == b['step'] == 8 and a['protocol'] == b['protocol'] and a['exposure'] == b['exposure']
    def equal(x, y):
        assert type(x) is type(y)
        if torch.is_tensor(x): assert x.dtype == y.dtype and torch.equal(x, y) and torch.isfinite(x).all()
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x: equal(x[k], y[k])
        elif isinstance(x, (tuple, list)):
            assert len(x) == len(y)
            for v, w in zip(x, y): equal(v, w)
        else: assert x == y
    for field in ('state_dict', 'optimizer', 'torch_rng', 'python_rng'): equal(a[field], b[field])
    assert not torch.cuda.is_initialized()
    result = dict(status=('REAL_THREE_TASK_COMPACT_FULL_MODEL_ADAM_RNG_RESUME_BIT_EXACT' if len(tasks)==3
        else 'REAL_COMPACT_TASK_SUBSET_FULL_MODEL_ADAM_RNG_RESUME_BIT_EXACT'),
        actual_updates_per_compared_branch=8, actual_questions_per_branch=8*len(tasks), exposure=a['exposure'],
        all_model_parameters_Adam_moments_counters_and_RNG_bit_exact=True,
        no_GPU_used=True, full8000_training_complete=False, full_goal_complete=False)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    proof_path = ROOT / plan.get('training_prefix_validation', plan['output']+'/training_prefix_validation.json')
    proof_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
