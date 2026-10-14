"""Matched eligible task buckets for an explicit new shared Full curriculum."""
from collections import Counter
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'teacher': ROOT/'outputs/groundjev/training_data/mixed_onejev_own_history_v3_lifecycle_fixed2_500/train.jsonl',
    'native': ROOT/'outputs/groundjev/training_data/onejev_native_mixture200_v2_family_matched/train.jsonl',
    'rmot': ROOT/'outputs/groundjev/training_data/onejev_native_rmot_membership_v2_registered_release_root/train.jsonl',
    'ovmot': ROOT/'outputs/groundjev/training_data/onejev_base_categories_v1/train.jsonl',
}


def feature_bytes():
    excluded = {'checkpoint.pt', 'projector_delta.pt', 'latest_training_state.pt'}
    return sum(p.stat().st_size for p in (ROOT/'outputs/groundjev').rglob('*')
               if p.is_file() and p.suffix in ('.json', '.jsonl', '.pt', '.png', '.jpg')
               and p.name not in excluded
               and not any(part.startswith('ckpt-') or part == 'final' for part in p.parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--version', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    data = {k: [json.loads(line) for line in p.open()] for k, p in SOURCES.items()}
    assert len(data['teacher']) == 20881 and len(data['native']) == 41762
    assert len(data['rmot']) == 346 and len(data['ovmot']) == 430
    for domain in ('teacher', 'native'):
        assert all(r['state']['task']['task'] == 'mot' for r in data[domain])
    teacher_families = Counter(r['task'] for r in data['teacher'])
    native_families = Counter(r['task'] for r in data['native'])
    assert native_families == Counter({k: 2*v for k, v in teacher_families.items()})
    n = len(data['native'])
    teacher_rows = data['teacher']*2 if args.version == 1 else [row for row in data['teacher'] for _ in range(2)]
    variants = {'teacher': teacher_rows, 'native': data['native']}
    if args.version == 2:
        assert all(a['task'] == b['task'] for a, b in zip(teacher_rows, data['native']))
    existing = feature_bytes()
    # Serialize the exact planned rows once for a byte estimate; this also
    # accounts for all explicit repeat identifiers and provenance fields.
    def row_for(row, bucket, index):
        result = dict(row)
        result['id'] = f"unified_{bucket}_{index}_{row['id']}"
        result['supervision'] = dict(row.get('supervision', {}),
                                     tracking_task_bucket=bucket,
                                     explicit_new_stage=f'matched_unified600_v{args.version}',
                                     source_row_id=row['id'], repetition_index=index)
        return result
    def rows(mot):
        for i in range(n):
            for bucket, source in [('mot', mot), ('rmot', data['rmot']), ('ovmot', data['ovmot'])]:
                yield row_for(source[i % len(source)], bucket, i)
    serialized_bytes = {k: sum(len((json.dumps(r)+'\n').encode()) for r in rows(v)) for k, v in variants.items()}
    budget = 25*2**30
    assert existing+sum(serialized_bytes.values())+10*2**20 < budget, (existing, serialized_bytes, budget)
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'PREPARING_NEW_UNIFIED_FULL_CANDIDATE_DOMAIN_STUDY',
        'source_rows': {k: len(v) for k, v in data.items()}, 'sources': {k: str(v) for k, v in SOURCES.items()},
        'conservative_existing_feature_and_text_bytes': existing,
        'planned_new_JSONL_bytes': serialized_bytes, 'temporary_feature_budget_bytes': budget,
        'common_semantics': 'same real native fit RMOT rows and registered Base-only teacher crop category rows',
        'Base_category_ground_native_proposal_claim': False,
        'native_RMOT_One500_history_claim': False,
        'same_initialization': 'actual original500 FP32 masters; new optimizer/scheduler reset',
        'steps_each': 600, 'seed': 0, 'lr': 2e-6, 'accum_rows': 4,
        'official_loss_and_step_planning_unchanged': True,
        'sampling_protocol': 'native-length-buckets' if args.version == 1 else 'paired-index-shuffle-v1',
        'matched_budget': 'optimizer updates and eligible bucket counts; actual prefix exposures recorded; v2 also matches every per-step semantic row and MOT question family; not token/FLOP matched',
        'new_training_started': False, 'Novel_or_eval_GT_supervised': False,
        'launch_requirement': 'actual500 matched controls reviewed and queued real native Noul probe completed',
        'existing500_or_MOTonly200_protocol_changed': False, 'variants': {}}
    for name, mot in variants.items():
        directory = args.out/name; directory.mkdir()
        buckets, families, unique_source_ids = Counter(), Counter(), {k: set() for k in ('mot', 'rmot', 'ovmot')}
        actual_bytes = 0
        with (directory/'train.jsonl').open('w') as output:
            for row in rows(mot):
                bucket = row['supervision']['tracking_task_bucket']
                buckets[bucket] += 1; families[row['task']] += 1
                unique_source_ids[bucket].add(row['supervision']['source_row_id'])
                line = json.dumps(row)+'\n'; output.write(line); actual_bytes += len(line.encode())
        assert buckets == Counter({'mot': n, 'rmot': n, 'ovmot': n})
        expected_unique_mot = len(mot) if name == 'native' else len(data['teacher'])
        assert len(unique_source_ids['mot']) == expected_unique_mot
        assert len(unique_source_ids['rmot']) == 346 and len(unique_source_ids['ovmot']) == 430
        assert actual_bytes == serialized_bytes[name]
        receipt = {'eligible_rows': 3*n, 'tracking_task_buckets': dict(buckets), 'question_families': dict(families),
                   'unique_source_row_ids': {k: len(v) for k, v in unique_source_ids.items()},
                   'JSONL_bytes': actual_bytes, 'all_registered_sources_retained': True,
                   'tracking_task_field': 'supervision.tracking_task_bucket; native category request state intentionally retains its deployed descriptive task string',
                   'native_category_question_templates_and_all1203_vocabulary_unchanged': True}
        (directory/'summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
        summary['variants'][name] = receipt
        config = json.loads((ROOT/'configs/groundjev/onejev_teacher_continuation200_v1.json').read_text())
        config.update(train=str((directory/'train.jsonl').resolve()), name=f'onejev_unified600_{name}_v{args.version}', max_steps=600)
        path = ROOT/f'configs/groundjev/onejev_unified600_{name}_v{args.version}.json'
        assert not path.exists()
        path.write_text(json.dumps(config, indent=2)+'\n')
        preflight = dict(config, model=str(ROOT/'outputs/groundjev/training/onejev_teacher_check_100/final'),
                         name=f'onejev_unified600_{name}_token_preflight_v{args.version}')
        suffix = '' if args.version == 1 else f'_v{args.version}'
        (ROOT/f'configs/groundjev/onejev_unified600_{name}_token_preflight{suffix}.json').write_text(json.dumps(preflight, indent=2)+'\n')
    assert summary['variants']['teacher']['question_families'] == summary['variants']['native']['question_families']
    summary['status'] = 'NEW_UNIFIED_FULL_MATCHED_ELIGIBLE_CURRICULA_COMPLETE_NOT_TRAINED'
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
