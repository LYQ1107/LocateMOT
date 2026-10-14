"""Official CPU-only SFT plan plus the actual finite training-prefix exposure."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'external/OneJev'))


def main():
    from train import sft
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--paired-index-shuffle', action='store_true')
    options, argv = parser.parse_known_args()
    args = sft.parse_args(argv)
    assert args.plan_only
    native_plan_only, native_epoch = sft.plan_only, sft.plan_epoch
    if options.paired_index_shuffle:
        from paired_sft_sampler import index_shuffle_plan
        native_epoch = index_shuffle_plan(native_epoch)

    def capture(bundle, rows, config, world):
        cached = native_epoch(bundle, rows, config, 0, 0, 1)
        batches, dropped, lengths = cached
        assert dropped == 0 and len(rows) == 125286
        steps = sft.plan_steps(batches, lengths, 1, config.accum_rows)
        prefix = steps[:config.max_steps]
        buckets, families, conditional_categories, labels = Counter(), Counter(), Counter(), Counter()
        selected = []
        for step, ranks in enumerate(prefix, 1):
            for micro in ranks[0]:
                for index in micro:
                    row = rows[index]
                    bucket = row['supervision']['tracking_task_bucket']
                    buckets[bucket] += 1; families[row['task']] += 1
                    answer = next(iter(row['target']))
                    labels[row['task']+':'+answer] += 1
                    if row['task'] == 'category_leaf':
                        conditional_categories[row['question']['criteria'][answer]] += 1
                    selected.append({'step': step, 'index': index, 'id': row['id'],
                                     'bucket': bucket, 'family': row['task'], 'tokens': lengths[index]})
        assert len(prefix) == config.max_steps == 600 and len(selected) == 2400
        assert set(buckets) == {'mot', 'rmot', 'ovmot'}
        # The native reporter consumes the exact plan already computed above.
        # This cache is local to this CPU-only reporting call, never training.
        sft.plan_epoch = lambda *unused, **kwargs: cached
        try:
            code = native_plan_only(bundle, rows, config, world)
        finally:
            sft.plan_epoch = native_epoch
        assert code == 0
        receipt = {'status': 'OFFICIAL_CPU_PLAN_AND_FINITE_PREFIX_COMPLETE', 'eligible_rows': len(rows),
            'overlength_dropped_rows': dropped, 'max_context': config.max_len, 'max_actual_tokens': max(lengths),
            'total_eligible_tokens': sum(lengths), 'planned_updates': len(steps),
            'registered_new_stage_updates': config.max_steps, 'actual_prefix_rows': len(selected),
            'actual_prefix_task_counts': dict(buckets), 'actual_prefix_question_families': dict(families),
            'actual_prefix_label_counts': dict(labels), 'actual_prefix_conditional_Base_category_counts': dict(conditional_categories),
            'actual_prefix_tokens': sum(r['tokens'] for r in selected), 'prefix': selected,
            'official_rendering_augmentation_step_planning_and_loss_unchanged': True,
            'official_length_bucket_sampler_unchanged': not options.paired_index_shuffle,
            'sampling_protocol': 'paired-index-shuffle-v1' if options.paired_index_shuffle else 'native-length-buckets',
            'one_CPU_plan_reused_for_native_report_only': True, 'model_or_GPU_loaded': False,
            'new_training_complete': False, 'config': vars(config)}
        options.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps({k: v for k, v in receipt.items() if k not in ('prefix', 'actual_prefix_label_counts', 'config')}, indent=2), flush=True)
        return code

    sft.plan_only = capture
    try:
        return sft.main(argv)
    finally:
        sft.plan_only = native_plan_only


if __name__ == '__main__':
    raise SystemExit(main())
