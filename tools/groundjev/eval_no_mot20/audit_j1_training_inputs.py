"""Audit the frozen J1 TRAIN inputs; never train or modify the original rows."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'external/OneJev'))
from tools.groundjev.eval_no_mot20.common import RUN, load, save


def audit(rows, recipe):
    from train.common import render_example

    counts = Counter()
    targets = Counter()
    image_roles = Counter()
    groups = {}
    for line in rows.open():
        row = json.loads(line)
        counts['total_rows'] += 1
        state = row.get('state') or {}
        counts['rows_with_nonempty_historical_tracks'] += bool(state.get('historical_tracks'))
        counts['rows_with_more_than_two_media'] += len(row.get('media') or []) > 2
        for key, value in state.items():
            if isinstance(value, str) and re.fullmatch(r'<image:\d+>', value):
                image_roles[key] += 1
        if state.get('current_candidate_crop_sheet') == '<image:2>':
            counts['rows_declaring_current_crop_image'] += 1
            counts['declared_crop_with_two_media'] += int(len(row.get('media') or []) >= 2)
        if (row.get('supervision') or {}).get('question_kind') != 'birth':
            continue
        counts['birth_rows'] += 1
        target = json.dumps(row['target'], sort_keys=True, separators=(',', ':'))
        targets[target] += 1
        instruction = row['question'].get('instructions', '')
        if len(state.get('candidates') or []) > 1 and not re.search(r'candidate\s+\d+', instruction):
            counts['birth_rows_with_multiple_candidates_and_no_question_index'] += 1
        # render_example is the actual upstream TRAIN renderer. Row id,
        # supervision metadata and sample index are not rendered into its
        # state/question prompt. Permutation changes slots, not candidate ID.
        example = render_example(row, lambda state, suffix: state+'\n'+suffix, 0, seed=None)
        media = json.dumps(row.get('media') or [], separators=(',', ':'))
        key = hashlib.sha256((example.prompt+'\n'+media).encode()).hexdigest()
        group = groups.setdefault(key, {'count':0,'targets':Counter(),'examples':{},'rows':{}})
        group['count'] += 1
        group['targets'][target] += 1
        if target not in group['examples']:
            group['examples'][target] = row['id']
            group['rows'][target] = row
    conflicting = [group for group in groups.values() if len(group['targets']) > 1]
    examples = []
    for group in conflicting[:5]:
        pair = list(group['rows'].values())[:2]
        first = render_example(pair[0], lambda state, suffix: state+'\n'+suffix, 0, seed=None)
        second = render_example(pair[1], lambda state, suffix: state+'\n'+suffix, 0, seed=None)
        examples.append({'row_ids':[row['id'] for row in pair],
            'targets':[row['target'] for row in pair],
            'same_frozen_upstream_rendered_prompt':first.prompt == second.prompt,
            'same_media_entries':pair[0]['media'] == pair[1]['media'],
            'candidate_count':len(pair[0]['state'].get('candidates') or []),
            'question':pair[0]['question'], 'source':pair[0]['source']})
    return {'status':'FROZEN_J1_TRAIN_INPUT_AUDIT_COMPLETE','rows_file':str(rows),
        'training_recipe':recipe,'counts':dict(counts),'birth_targets':dict(targets),
        'declared_image_roles':dict(image_roles),
        'unique_birth_inputs':len(groups),'birth_input_groups_with_conflicting_targets':len(conflicting),
        'birth_rows_in_conflicting_input_groups':sum(group['count'] for group in conflicting),
        'conflict_examples':examples,'runtime_GT_read':False,'new_training':False,
        'interpretation':'Identical state/question/media with different labels is conflicting supervision. '
                         'This audit does not establish how much it caused the measured tracking failure. '
                         'TRAIN random slot permutations do not supply a missing candidate index.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=Path, default=Path('/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/three_task_sft_v2b/rows.jsonl'))
    parser.add_argument('--recipe', type=Path, default=Path('/data3/user/LocateMOT-runs/TrackJEV_FullDataV2/trackjev_3task_J1_v4_birth/args.json'))
    parser.add_argument('--out', type=Path, default=RUN/'engineering_checks/j1_input_contract/train_input_audit.json')
    args = parser.parse_args()
    result = audit(args.rows, load(args.recipe))
    save(args.out, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
