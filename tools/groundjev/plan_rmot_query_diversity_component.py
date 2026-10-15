"""Official CPU token planning for the complete, untrained expression component."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    from tools.groundjev.run_streamed_media_sft import configure_streamed_sft
    stateful = configure_streamed_sft()
    from train import sft
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    component = json.loads((args.component/'summary.json').read_text())
    assert component['status'] == 'RMOT_ALL_FIT_EXPRESSION_COMPONENT_GENERATED_NOT_TRAINED'
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'configs/groundjev/onejev_unified600_native_v2.json').read_text())
    config.update(model=str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600'),
        train=str((args.component/'train.jsonl.gz').resolve()), name='official_RMOT_component_CPU_plan',
        runs_dir=str(args.out.resolve()), plan_only=True, device='cpu', max_steps=0)
    path = args.out/'official_plan_config.json'; path.write_text(json.dumps(config, indent=2)+'\n')
    state = {'status': 'RUNNING_FULL_RMOT_EXPRESSION_COMPONENT_CPU_PLAN', 'owner_pid': os.getpid(),
        'component': str(args.component), 'expected_rows': component['counts']['rows'],
        'physical_frames': component['physical_frames'], 'expression_files': component['counts']['expressions'],
        'new_model_or_GPU_loaded': False, 'training_started': False,
        'text_rows_loaded_in_RAM_by_native_trainer': True,
        'native_reporter_timing_scope': 'cached-plan reporter only; separate full token planning timer below',
        'full_goal_complete': False}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)
    native_plan, native_epoch = sft.plan_only, sft.plan_epoch
    def capture(bundle, rows, settings, world):
        started = time.monotonic()
        cached = native_epoch(bundle, rows, settings, 0, 0, 1)
        seconds = time.monotonic() - started
        batches, dropped, lengths = cached
        assert len(rows) == component['counts']['rows'] and dropped == 0 and len(lengths) == len(rows)
        families = Counter(row['task'] for row in rows)
        for family, count in families.items():
            assert count == sum(component['counts'].get(family+':'+target, 0) for target in ('yes', 'no'))
        assert len({r['supervision']['expression_file'] for r in rows}) == component['counts']['expressions']
        protocol = stateful.protocol(settings)
        assert protocol['input_io_protocol'] == 'gzip-jsonl-and-native-category-crop-recipes-v1'
        steps = sft.plan_steps(batches, lengths, 1, settings.accum_rows)
        # The upstream reporter defaults PLAN_WORLD to 4 independently of its
        # CPU world argument. Keep its hypothetical distribution explicit.
        os.environ['PLAN_WORLD'] = str(world)
        sft.plan_epoch = lambda *unused, **kwargs: cached
        try:
            code = native_plan(bundle, rows, settings, world)
        finally:
            sft.plan_epoch = native_epoch
        assert code == 0 and not torch.cuda.is_initialized()
        state.update(status='FULL_RMOT_EXPRESSION_COMPONENT_CPU_PLAN_COMPLETE_NOT_TRAINED',
            all_rows=len(rows), overlength_dropped_rows=dropped, max_actual_tokens=max(lengths),
            total_actual_tokens=sum(lengths), question_families=dict(families),
            hypothetical_full_component_epoch_updates=len(steps), actual_new_training_updates=0,
            actual_CPU_world_size=world,
            input_io_resume_protocol=protocol['input_io_protocol'],
            actual_full_native_plan_epoch_wall_seconds=seconds,
            process_CPU_RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save()
        return code
    sft.plan_only = capture; save()
    try:
        assert sft.main(['--config', str(path), '--plan-only']) == 0
    except Exception as error:
        state.update(status='RMOT_EXPRESSION_COMPONENT_CPU_PLAN_FAILED', error=repr(error)); save()
        raise
    finally:
        sft.plan_only = native_plan


if __name__ == '__main__':
    main()
