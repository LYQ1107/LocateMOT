"""Finite, declared train-only three-task curriculum; preserve all source corpora."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import random
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def load_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        for line in source: yield json.loads(line)


def reservoir(pool, key, row, counts, capacity, rng):
    counts[key] += 1
    if len(pool[key]) < capacity:
        pool[key].append(row)
    else:
        index = rng.randrange(counts[key])
        if index < capacity: pool[key][index] = row


def label(row):
    return next(iter(row['target']))


def balanced(pool, total, rng, initial=(), key=label):
    selected = list(initial)
    groups = defaultdict(list)
    for row in pool: groups[key(row)].append(row)
    assert groups and len(selected) <= total
    for values in groups.values(): rng.shuffle(values)
    initial_ids = {row['id'] for row in selected}
    unused = {k: [row for row in values if row['id'] not in initial_ids] for k, values in groups.items()}
    exposed = Counter(key(r) for r in selected); cursors = Counter()
    while len(selected) < total:
        target = min(sorted(groups), key=lambda k: exposed[k])
        cursor = cursors[target]
        available = unused[target]
        row = (available[cursor] if cursor < len(available) else
               groups[target][(cursor-len(available)) % len(groups[target])])
        selected.append(row)
        cursors[target] += 1; exposed[target] += 1
    return selected


def identity_label(row):
    if row['task'] == 'empty_presence': return 'empty_presence'
    value = label(row)
    return 'EXISTING' if value.isdigit() else value


def select_mot(pools, frame_rows, quotas, rng):
    mandatory, mandatory_lifecycle = [], []
    for _, rows in sorted(frame_rows.items()):
        preferred = [row for row in rows if row['task'] in ('identity', 'empty_presence')]
        if preferred:
            mandatory.append(rng.choice(preferred))
        else:
            # Empty candidates with surviving history have native life questions only.
            mandatory_lifecycle.append(rng.choice([row for row in rows if row['task'] == 'lifecycle']))
    mot = balanced(pools['identity'], quotas['identity_and_all_empty_frames'], rng, mandatory, identity_label)
    for family, count in quotas.items():
        if family != 'identity_and_all_empty_frames':
            mot.extend(balanced(pools[family], count, rng,
                                mandatory_lifecycle if family == 'lifecycle' else ()))
    return mot, len(mandatory_lifecycle)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    rng = random.Random(plan['seed'])
    source = ROOT/plan['native_groups']
    state = json.loads((source/'owner_state.json').read_text())
    assert state['status'] == 'ALL43_NATIVE_TRAIN_CLIPS_FULL_V3_GROUP_REFS_AND_CPU_PLANS_VALIDATED'
    assert len(state['completed']) == 43
    proof = json.loads((ROOT/plan['input_io_validation']).read_text())
    assert proof['status'] == 'NATIVE_CAUSAL_REQUEST_AND_REAL_PROCESSOR_GROUP_REFERENCES_VALIDATED'
    mot_pools, frame_rows = defaultdict(list), defaultdict(list)
    for name, unit in sorted(state['completed'].items()):
        assert unit['dropped_rows'] == 0 and unit['all_actor_outputs_states_exact']
        for row in load_rows(Path(unit['row_references'])):
            mot_pools[row['task']].append(row)
            group = row['groundjev_group_question_ref']['group'].removesuffix('_assigned')
            seq, frame, _ = re.fullmatch(r'(.+)_(\d+)_batch(\d+)', group).groups()
            frame_rows[(seq, int(frame))].append(row)
    assert len(frame_rows) == 1056
    mot, mandatory_lifecycle = select_mot(mot_pools, frame_rows, plan['MOT_family_rows'], rng)

    # Keep only bounded reservoirs while streaming the full 87,721-row RMOT component.
    expression, global_pool, counts = defaultdict(list), defaultdict(list), Counter()
    physical = {}; source_rows = 0
    for row in load_rows(ROOT/plan['rmot_source']):
        source_rows += 1
        target = label(row); assert target in ('yes', 'no')
        expr = row['supervision']['expression_file']
        reservoir(expression, (expr, target), row, counts, 1, rng)
        reservoir(global_pool, target, row, counts, 4000, rng)
        physical.setdefault(row['media'][0]['path'], row)
    expressions = sorted({key[0] for key in expression})
    assert len(expressions) == 4350 and len(physical) == 72 and source_rows == 87721
    rmot = [expression[(expr, 'yes' if (expr, 'yes') in expression else 'no')][0] for expr in expressions]
    represented = {r['media'][0]['path'] for r in rmot}
    rmot.extend(row for path, row in sorted(physical.items()) if path not in represented)
    for target in ('yes', 'no'):
        needed = 4000-sum(label(r) == target for r in rmot)
        assert needed >= 0
        rng.shuffle(global_pool[target]); rmot.extend(global_pool[target][:needed])
        assert sum(label(r) == target for r in rmot) == 4000

    # Stream root/leaf pairs; a bounded per-class object reservoir prevents loading all rows.
    objects, object_counts = defaultdict(list), Counter()
    with gzip.open(ROOT/plan['category_source'], 'rt', encoding='utf-8') as rows:
        while (line := rows.readline()):
            root, leaf = json.loads(line), json.loads(next(rows))
            assert root['task'] == 'category_group' and leaf['task'] == 'category_leaf'
            assert root['supervision']['annotation_id'] == leaf['supervision']['annotation_id']
            category = root['supervision']['Base_vocabulary_index']
            reservoir(objects, category, (root, leaf), object_counts, 23, rng)
    assert len(objects) == 175
    ovmot = []
    for index, category in enumerate(sorted(objects)):
        pool = objects[category]; rng.shuffle(pool)
        for j in range(22+(index < 150)):
            ovmot.extend(pool[j % len(pool)])
    tasks = {'mot': mot, 'rmot': rmot, 'ovmot': ovmot}
    assert all(len(rows) == 8000 for rows in tasks.values())
    sampled_frames = {re.fullmatch(r'(.+)_(\d+)_batch(\d+)',
                      row['groundjev_group_question_ref']['group'].removesuffix('_assigned')).groups()[:2]
                      for row in mot}
    assert sampled_frames == {(seq, str(frame)) for seq, frame in frame_rows}
    out = ROOT/plan['output']; out.mkdir(parents=True, exist_ok=False)
    train = out/'train.jsonl.gz'
    for rows in tasks.values(): rng.shuffle(rows)
    ids = defaultdict(set); counts = Counter(); actual_frame_sets = defaultdict(set)
    with gzip.open(train, 'wt', encoding='utf-8') as target:
        for i in range(8000):
            for bucket, rows in tasks.items():
                original = rows[i]
                row = dict(original, id=f'broader6000_{bucket}_{i}')
                row['supervision'] = dict(original.get('supervision', {}), tracking_task_bucket=bucket,
                                           source_row_id=original['id'], curriculum='broader_shared6000_v1')
                target.write(json.dumps(row, ensure_ascii=False)+'\n')
                ids[bucket].add(original['id'])
                target_label = identity_label(row) if bucket == 'mot' and row['task'] == 'identity' else label(row)
                counts[(bucket, row['task'], target_label)] += 1
                if bucket != 'mot': actual_frame_sets[bucket].add(row['media'][0]['path'])
    assert train.stat().st_size <= plan['new_mixture_budget_bytes']
    config = json.loads((ROOT/'configs/groundjev/onejev_history_refresh600_refreshed_v1.json').read_text())
    config.update(train=str(train), name=plan.get('training_name', 'onejev_broader_shared6000_v1'), max_steps=6000, save_every=plan['save_every'],
                  model=str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600'))
    (out/'training_config.json').write_text(json.dumps(config, indent=2)+'\n')
    summary = {'status': 'BROADER_SHARED24000_ROWS_PREPARED_NOT_YET_TRAINED',
               'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'rows_per_task': {k: len(v) for k, v in tasks.items()},
               'unique_source_rows_per_task': {k: len(v) for k, v in ids.items()},
               'MOT_physical_frames': len(sampled_frames), 'MOT_videos': len(state['completed']),
               'MOT_zero_candidate_frames_using_native_lifecycle_supervision': mandatory_lifecycle,
               'RMOT_expressions': len(expressions), 'RMOT_physical_frames': len(actual_frame_sets['rmot']),
               'OVMOT_Base_classes': len(objects), 'OVMOT_unique_images': len(actual_frame_sets['ovmot']),
               'OVMOT_unique_objects': len({r['supervision']['annotation_id'] for r in ovmot}),
               'family_labels': [{'task_bucket': k[0], 'question_family': k[1], 'target': k[2], 'rows': v}
                                 for k, v in sorted(counts.items())],
               'compressed_training_bytes': train.stat().st_size, 'actual_new_optimizer_updates': 0,
               'train': str(train), 'config': str(out/'training_config.json'),
               'all_source_corpora_preserved': True, 'full_goal_complete': False}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
