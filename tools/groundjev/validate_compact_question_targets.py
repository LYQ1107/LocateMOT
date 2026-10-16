"""Check selected losses against original native question options and GT targets."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tools.groundjev.compact_reference_data import CompactReferenceInputs
    parser = argparse.ArgumentParser(); parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads((args.data / 'registration.json').read_text())
    inputs = CompactReferenceInputs(ROOT / plan['inputs']); selected = defaultdict(list); base = []
    with gzip.open(args.data / 'questions.jsonl.gz', 'rt') as file:
        for offset, line in enumerate(file):
            row = json.loads(line)
            assert row['step'] == offset // 3 + 1
            assert row['unit'].split('/')[0] == ('mot', 'rmot', 'ovmot')[offset % 3]
            if row['task'] == 'category': base.append(row)
            else: selected[row['unit']].append(row)
    counts = Counter(); examples = {}
    for unit, rows in selected.items():
        info = inputs.units[unit]; wanted = {r['id'] for r in rows}; original = {}
        with gzip.open(info['full_source_rows'], 'rt') as file:
            for line in file:
                source = json.loads(line)
                if source['id'] in wanted: original[source['id']] = source
        assert set(original) == wanted
        groups = {}
        wanted_groups = {r['group'] for r in rows}
        with gzip.open(Path(info['full_source_rows']).parent / 'groups.jsonl.gz', 'rt') as file:
            for line in file:
                source = json.loads(line)
                if source['id'] in wanted_groups: groups[source['id']] = source
        for row in rows:
            source = original[row['id']]; ref = source['groundjev_group_question_ref']
            assert source['target'] == row['source_target'] and source['task'] == row['task']
            assert ref['question'] == row['question'] and ref['group'] == row['group']
            group = groups[row['group']]; question = group['questions'][row['question']]
            record = inputs.records[unit, row['group']]; task = row['task']; slot = row['slot']
            aliases = [str(t['id']) for t in group['state']['historical_tracks']]
            assert aliases == record['alias_ids']
            if task == 'identity':
                assert list(question['criteria']) == aliases + ['NEW', 'IGNORE'] + (['NO_MATCH'] if aliases else [])
                # Existing Compact retains its third, hard-masked action column at m=0.
                # This padded zero target is not an additional legal native option.
                names = aliases + ['NEW', 'IGNORE', 'NO_MATCH']
                counts['zero_history_original_masked_NO_MATCH_columns'] += not aliases
                assert row['question'] == f'match_{slot}'
                assert not (not aliases and source['target'].get('NO_MATCH', 0)), 'Original Compact masks NO_MATCH with no history'
            elif task == 'lifecycle':
                names = list(question['criteria']); assert names == ['ACTIVE', 'LOST', 'REACTIVATE', 'TERMINATE']
                assert row['question'] == 'life_' + aliases[slot]
            elif task == 'reliability':
                assert question['type'] == 'score' and len(question['criteria']) == 4
                names = ['0', '1', '2', '3']; assert row['question'] == f'reliability_{slot}'
            else:
                assert question['type'] == 'noul'; names = ['no', 'yes']
                prefix = 'update' if task == 'memory_update' else 'membership'
                assert row['question'] == f'{prefix}_{slot}'
            assert dict(zip(names, row['target'])) == {name: float(source['target'].get(name, 0)) for name in names}
            counts[unit.split('/')[0] + ':' + task] += 1
            examples.setdefault(task + ':' + row['label'], dict(unit=unit, id=row['id'], group=row['group'], slot=slot))
    objects = [json.loads(line) for line in (ROOT / plan['base'] / 'objects.jsonl').open()]
    actual_draws = Counter()
    for row in base:
        source = objects[row['object_row']]
        assert row['class_target'] == source['class_index'] and row['annotation_id'] == source['annotation_id']
        actual_draws[row['annotation_id']] += 1
    original_draws = Counter(json.loads((ROOT / plan['base'] / 'sampled_draws.json').read_text()))
    assert actual_draws == Counter({k: 2*v for k, v in original_draws.items()})
    assert len(base) == 8000 and len({r['class_target'] for r in base}) == 175
    temporal_count = sum(len(rows) for rows in selected.values())
    assert temporal_count == 16000
    result = dict(status='ALL_SELECTED_COMPACT_LABELS_AND_NATIVE_HEAD_OPTIONS_VALIDATED',
        all_selected_original_labels_and_native_question_options_exact=True,
        original_temporal_questions_checked=temporal_count, questions_by_family=dict(counts),
        original_OV_teacher_draws_preserved_exactly_twice=True, OV_flat_category_exposures=len(base),
        representative_source_questions=examples, total_questions=temporal_count+len(base),
        GPU_or_optimizer_updates_by_validation=0, full_goal_complete=False)
    (args.data / 'target_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
