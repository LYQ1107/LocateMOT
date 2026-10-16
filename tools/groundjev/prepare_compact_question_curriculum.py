"""Explicit typed-head curriculum from original native question labels and references."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PROTOCOL = 'native-compact-typed-question-balanced-three-task-v1'


def encode(row, unit, record):
    task = row['task']; question = row['groundjev_group_question_ref']['question']
    names = None
    if task == 'identity':
        slot = int(question.removeprefix('match_'))
        names = record['alias_ids'] + ['NEW', 'IGNORE', 'NO_MATCH']
    elif task in ('memory_update', 'query_membership', 'reliability'):
        prefix = dict(memory_update='update_', query_membership='membership_', reliability='reliability_')[task]
        slot = int(question.removeprefix(prefix))
        names = ['0', '1', '2', '3'] if task == 'reliability' else ['no', 'yes']
    else:
        assert task == 'lifecycle' and question.startswith('life_')
        slot = record['alias_ids'].index(question.removeprefix('life_'))
        names = ['ACTIVE', 'LOST', 'REACTIVATE', 'TERMINATE']
    assert 0 <= slot < len(record['history_refs'] if task == 'lifecycle' else record['candidate_refs'])
    assert set(row['target']) <= set(names)
    target = [float(row['target'].get(name, 0.)) for name in names]
    assert all(0 <= p <= 1 for p in target) and abs(sum(target) - 1) < 1e-6
    label = max(row['target'], key=row['target'].get)
    if task == 'identity' and label not in ('NEW', 'IGNORE', 'NO_MATCH'): label = 'EXISTING'
    return dict(id=row['id'], unit=unit, group=record['id'], task=task, slot=slot,
                target=target, source_target=row['target'], question=question, label=label)


def fill(pool, chosen, count, rng, balanced=True):
    """Use unseen labels/rows before explicit repetitions; mandatory context rows stay."""
    assert len(chosen) <= count and pool
    used = {r['id'] for r in chosen}; frequency = Counter(r['label'] for r in chosen)
    by_label = defaultdict(list)
    for row in pool: by_label[row['label']].append(row)
    for rows in by_label.values(): rng.shuffle(rows)
    unseen = {label: [r for r in rows if r['id'] not in used] for label, rows in by_label.items()}
    while len(chosen) < count:
        available = [label for label, rows in unseen.items() if rows]
        if not available: available = list(by_label)
        label = min(available, key=lambda k: (frequency[k], k)) if balanced else rng.choice(available)
        row = unseen[label].pop() if unseen[label] else rng.choice(by_label[label])
        chosen.append(row); frequency[label] += 1
    return chosen


def main():
    from tools.groundjev.compact_reference_data import CompactReferenceInputs
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text()); rng = random.Random(plan['seed'])
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    out = ROOT / plan['output']; out.mkdir(parents=True, exist_ok=False)
    (out / 'registration.json').write_text(json.dumps(plan, indent=2) + '\n')
    inputs = CompactReferenceInputs(ROOT / plan['inputs'])
    proof = json.loads((ROOT / plan['inputs'] / 'native_input_regression.json').read_text())
    assert proof['all_reconstructed_model_input_tensors_and_frozen_forward_outputs_exact']
    inventory = Counter(); excluded = Counter(); pools = defaultdict(list); contexts = defaultdict(list)
    for unit, info in inputs.units.items():
        bucket = unit.split('/')[0]
        with gzip.open(info['full_source_rows'], 'rt') as file:
            for line in file:
                row = json.loads(line); task = row['task']; inventory[bucket + ':' + task] += 1
                if task not in (plan['MOT_family_questions'] if bucket == 'mot' else plan['RMOT_family_questions']):
                    excluded[bucket + ':' + task] += 1; continue
                record = inputs.records[unit, row['groundjev_group_question_ref']['group']]
                entry = encode(row, unit, record)
                pools[bucket, task].append(entry); contexts[unit, record['frame']].append(entry)
    mandatory = defaultdict(list)
    for entries in contexts.values():
        # Prefer identity on MOT; membership on RMOT, with an existing positive if available.
        row = min(entries, key=lambda r: (r['task'] != 'identity', r['label'] != 'yes', r['id']))
        mandatory[row['unit'].split('/')[0], row['task']].append(row)
    selected = []
    for task, count in plan['MOT_family_questions'].items():
        selected.extend(fill(pools['mot', task], mandatory['mot', task], count, rng))
    rmot = mandatory['rmot', 'query_membership']; rmot_pool = pools['rmot', 'query_membership']
    for label in ('yes', 'no'):
        selected.extend(fill([r for r in rmot_pool if r['label'] == label],
                             [r for r in rmot if r['label'] == label], plan['RMOT_yes_no_questions_each'], rng))
    base = ROOT / plan['base']
    objects = [json.loads(line) for line in (base / 'objects.jsonl').open()]
    by_annotation = {r['annotation_id']: r for r in objects}
    draws = json.loads((base / 'sampled_draws.json').read_text())
    assert len(draws) * 2 == plan['task_questions']
    for exposure in range(2):
        for i, annotation in enumerate(draws):
            row = by_annotation[annotation]
            selected.append(dict(id=f'Base175_draw{i}_exposure{exposure}', unit='ovmot/teacher',
                task='category', object_row=row['feature_row'], class_target=row['class_index'],
                annotation_id=annotation, label=str(row['class_index'])))
    streams = defaultdict(list)
    for row in selected: streams[row['unit'].split('/')[0]].append(row)
    for i, task in enumerate(('mot', 'rmot', 'ovmot')):
        assert len(streams[task]) == plan['task_questions']
        random.Random(plan['seed'] + i).shuffle(streams[task])
    with gzip.open(out / 'questions.jsonl.gz', 'wt') as file:
        for step in range(plan['optimizer_steps']):
            for task in ('mot', 'rmot', 'ovmot'):
                file.write(json.dumps(dict(streams[task][step], step=step + 1), separators=(',', ':')) + '\n')
    trained = {(r['unit'], r['group']) for r in selected if r['task'] != 'category'}
    eligible_contexts = set(contexts)
    assert {(unit, inputs.records[unit, group]['frame']) for unit, group in trained} == eligible_contexts
    unrepresented = []
    for unit, info in inputs.units.items():
        for frame, source in info['frames'].items():
            if (unit, int(frame)) not in eligible_contexts:
                unrepresented.append(dict(unit=unit, frame=int(frame), image_path=source['image_path'],
                    reason='No original question supported by the explicitly selected task heads'))
    images = defaultdict(set)
    for row in selected:
        bucket = row['unit'].split('/')[0]
        path = objects[row['object_row']]['image_path'] if bucket == 'ovmot' else inputs.records[row['unit'], row['group']]['image_path']
        images[bucket].add(str(Path(path).resolve()))
    result = dict(status='BROADER_COMPACT_TYPED_QUESTION_CURRICULUM_PREPARED', protocol=PROTOCOL,
        task_questions={k: len(v) for k, v in streams.items()}, actual_optimizer_steps=0,
        source_question_inventory=dict(inventory), questions_outside_declared_head_selection=dict(excluded),
        selected_families=dict(Counter(r['unit'].split('/')[0] + ':' + r['task'] for r in selected)),
        selected_labels=dict(Counter(r['unit'].split('/')[0] + ':' + r['task'] + ':' + r['label'] for r in selected)),
        unique_original_question_IDs={k: len({r['id'] for r in v}) for k, v in streams.items() if k != 'ovmot'},
        unique_physical_first_media_images={k: len(v) for k, v in images.items()},
        eligible_query_frame_contexts=len(eligible_contexts), contexts_without_selected_head=unrepresented,
        all_original_frames_and_questions_preserved=True, GT_teacher_categories_Base_only=True,
        OV_flat_category_twice_per_original_object_draw_explicit=True,
        no_invented_RMOT_identity_lifecycle_or_OV_temporal_targets=True, new_feature_vectors_copied=False,
        training_input_validation_and_optimizer_updates_pending=True, full_goal_complete=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < plan['new_metadata_cap_bytes']
    print(json.dumps({k: v for k, v in result.items() if k != 'contexts_without_selected_head'}), flush=True)


if __name__ == '__main__': main()
