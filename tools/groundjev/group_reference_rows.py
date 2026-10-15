"""Native training rows referencing immutable shared request groups."""
from functools import lru_cache
import gzip
import json
from pathlib import Path

REFERENCE_KEY = 'groundjev_group_question_ref'
IO_PROTOCOL = 'gzip-jsonl-native-tracking-category-and-group-question-refs-v1'


@lru_cache(maxsize=64)
def request_groups(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    groups = {}
    with opener(path, 'rt', encoding='utf-8') as source:
        for line in source:
            group = json.loads(line)
            assert group['id'] not in groups
            groups[group['id']] = group
    return groups


def hydrate(row, root):
    if REFERENCE_KEY not in row:
        return row
    reference = row[REFERENCE_KEY]
    group = request_groups(str(Path(root)/reference['path']))[reference['group']]
    return dict(row, state=group['state'], media=group['media'],
                question=group['questions'][reference['question']])


def install(root):
    from train import common
    native_rows = common.load_rows

    def load_rows(path, limit=None, split=None):
        return [hydrate(row, root) for row in native_rows(path, limit=limit, split=split)]

    common.load_rows = load_rows
    return native_rows
