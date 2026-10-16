"""Real native SFT with full model/Adam equality checks across all ranks."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True); parser.add_argument('--label', required=True)
    parser.add_argument('--initial-training-state', type=Path); parser.add_argument('--resume-state', type=Path)
    parser.add_argument('--stop-after-saved-step', type=int)
    parser.add_argument('--backend', choices=['reference','fla'], default='fla')
    parser.add_argument('--deterministic-execution', action='store_true')
    parser.add_argument('--seed-global-rngs', action='store_true')
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from train import sft
    torch.set_num_threads(4)
    world = int(os.environ.get('WORLD_SIZE', '1')); rank = int(os.environ.get('RANK', '0'))
    step_reports, planning = [], []
    native_step = torch.optim.AdamW.step
    native_plan = sft.plan_steps

    def plan_steps(batches, lengths, world, accum_rows):
        planned = native_plan(batches, lengths, world, accum_rows)
        planning.append(dict(global_rows=[sorted(i for per in step for batch in per for i in batch) for step in planned],
            local_rows=[[i for batch in step[rank] for i in batch] for step in planned],
            lengths=lengths, world=world))
        return planned

    def synchronized_step(optimizer, *values, **kwargs):
        result = native_step(optimizer, *values, **kwargs)
        comparisons = []; tensors = 0; elements = 0
        for group in optimizer.param_groups:
            for parameter in group['params']:
                for tensor in (parameter, optimizer.state[parameter]['exp_avg'], optimizer.state[parameter]['exp_avg_sq']):
                    tensors += 1; elements += tensor.numel()
                    if world > 1:
                        reference = tensor.detach().clone()
                        dist.broadcast(reference, 0)
                        comparisons.append((tensor.detach()-reference).abs().max())
                        del reference
        if world > 1:
            maximum = torch.stack(comparisons).max()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            difference = float(maximum)
            assert difference == 0., difference
        else:
            difference = 0.
        assert tensors == 960 and elements == 3*752393024
        step_reports.append(dict(all320_parameters_and_both_full_Adam_moments_checked=world > 1,
            compared_tensors=tensors, compared_elements=elements, all_rank_max_difference=difference))
        return result

    sft.plan_steps = plan_steps
    torch.optim.AdamW.step = synchronized_step
    if args.backend == 'fla':
        from tools.groundjev.run_fla_group_reference_sft import configure_fla_group_reference_sft
        stateful = configure_fla_group_reference_sft()
    else:
        from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
        stateful = configure_group_reference_sft()
    command = [str(Path(__file__).resolve()), '--config', str(args.config), '--paired-index-shuffle']
    if world > 1:
        command += ['--distributed-sft']
    if args.deterministic_execution:
        command += ['--deterministic-execution']
    if args.seed_global_rngs:
        command += ['--seed-global-rngs']
    if args.initial_training_state:
        command += ['--initial-training-state', str(args.initial_training_state), '--expected-initial-step', '600']
    if args.resume_state:
        command += ['--resume-state', str(args.resume_state)]
    if args.stop_after_saved_step:
        command += ['--stop-after-saved-step', str(args.stop_after_saved_step)]
    sys.argv = command
    code = stateful.main()
    assert code == 0 and step_reports and planning
    receipt = dict(rank=rank, world=world, optimizer_steps_this_invocation=len(step_reports),
        backend=args.backend,
        deterministic_execution=args.deterministic_execution,
        seed_global_rngs=args.seed_global_rngs,
        steps=step_reports, plans=planning, resumed=bool(args.resume_state),
        registered_partial_stop=bool(args.stop_after_saved_step), full_goal_complete=False)
    args.receipts.mkdir(parents=True, exist_ok=True)
    (args.receipts/f'{args.label}_rank{rank}.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(rank=rank, world=world, steps=len(step_reports),
        all_rank_max_difference=max(s['all_rank_max_difference'] for s in step_reports))), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
