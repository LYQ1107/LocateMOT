"""Execute the pinned, unchanged official frame reader on actual fit annotations."""
import argparse
import ast
import json
import os.path as osp
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
COMMIT = '6a65640d849fdee4a32bb055945ee34c3b0edeb1'


def main():
    import numpy as np
    import torch
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    assert audit['status'] == 'RMOT_QUERY_COMPONENT_HELD_MISSING_GT_TARGET_BOXES'
    source = Path('/data2/user/reference_envs/temprmot')/f'refer_kitti_{COMMIT}.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DetMOTDetection')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_pre_single_frame')
    # Compile exactly this original method, avoiding unrelated model/training imports.
    namespace = {'np': np, 'torch': torch, 'Image': Image, 'osp': osp}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    reader = namespace['_pre_single_frame']
    clips = json.loads((ROOT/'configs/groundjev/rmot_native72_20261014.json').read_text())['clips']
    by_video = {c['video']: c for c in clips}
    native = {(r['sequence'], r['frame']): r for r in map(json.loads,
        (ROOT/'outputs/groundjev/compact/rmot_native72_v1/native_rollout.jsonl').open())}
    cases = {(r['video'], r['frame'], r['expression_file']) for r in audit['affected_contexts']}
    for clip in clips:
        for key in ('primary_expression_file', 'counter_expression_file'):
            for frame in clip['frames']:
                cases.add((clip['video'], frame, clip[key]))
    args.out.mkdir(parents=True, exist_ok=False)
    label_root = Path('/data2/user/LocateMOT/data/refer_kitti_v2/labels_with_ids/image_02')
    results = []
    for video, frame, expression in sorted(cases):
        query = json.loads(Path(expression).read_text())
        row = native[video, frame]
        # The official reader selects KITTI's xywh convention by this literal
        # path token. Symlink to the original bytes without copying images.
        alias = args.out.resolve()/'KITTI'/video/f'{frame:06d}.png'
        alias.parent.mkdir(parents=True, exist_ok=True)
        if not alias.exists():
            alias.symlink_to(Path(row['image_path']))
        label = label_root/video/f'{frame:06d}.txt'
        # The pre-existing 0000 empty frames have no released label file. They
        # are covered by the component's existing empty-frame protocol; the
        # official loader requires a file, so do not fabricate one for this check.
        if not label.exists():
            assert not query['label'].get(str(frame), [])
            results.append({'video': video, 'frame': frame, 'expression_file': expression,
                            'missing_label_file_and_no_expression_IDs': True})
            continue
        label_alias = alias.with_suffix('.txt')
        if not label_alias.exists():
            label_alias.symlink_to(label)
        owner = SimpleNamespace(img_files=[str(alias)], label_files=[str(label_alias)],
                                video_dict={str(label_alias.parent): 0})
        image, targets = reader(owner, 0, [query])
        labels = [line.split() for line in label.read_text().splitlines()]
        ids = [int(fields[1]) for fields in labels]
        requested = set(query['label'].get(str(frame), []))
        projected = [identity for identity, flag in zip(targets['obj_ids'].tolist(), targets['is_ref'].tolist()) if flag]
        assert projected == [identity for identity in ids if identity in requested]
        assert targets['obj_ids'].tolist() == ids
        assert len(targets['boxes']) == len(ids)
        for fields, box in zip(labels, targets['boxes'].tolist()):
            x, y, w, h = map(float, fields[2:6])
            expected = [x*image.width, y*image.height, (x+w)*image.width, (y+h)*image.height]
            assert max(abs(a-b) for a, b in zip(expected, box)) < .001
        results.append({'video': video, 'frame': frame, 'expression_file': expression,
            'official_released_box_IDs': ids, 'official_positive_box_IDs': projected,
            'expression_IDs_without_released_box': sorted(requested-set(ids))})
    affected = {(r['video'], r['frame'], r['expression_file']) for r in results
                if r.get('expression_IDs_without_released_box')}
    assert affected == {(r['video'], r['frame'], r['expression_file']) for r in audit['affected_contexts']}
    assert not torch.cuda.is_initialized()
    report = {'status': 'PINNED_OFFICIAL_RMOT_RELEASE_BOX_PROJECTION_VALIDATED',
        'official_commit': COMMIT, 'official_reader': 'datasets/refer_kitti.py:DetMOTDetection._pre_single_frame',
        'source_url': f'https://github.com/zyn213/TempRMOT/blob/{COMMIT}/datasets/refer_kitti.py',
        'source_path': str(source), 'unmodified_official_method_executed': True,
        'actual_official_method_cases': sum('official_released_box_IDs' in r for r in results),
        'legacy_missing_empty_label_file_cases_recorded_without_fabricating_files': sum('missing_label_file_and_no_expression_IDs' in r for r in results),
        'all_affected_contexts_validated': len(affected), 'unique_missing': audit['unique_missing'],
        'rule': 'iterate released same-frame boxes; mark each as a query target iff its ID is listed in the expression; unboxed IDs create no extra box',
        'physical_visibility_or_annotation_completeness_inferred': False,
        'component_question_and_existing_candidate_IoU_proxy_are_our_protocol_not_official_tracker': True,
        'training_or_evaluation_protocol_of_running_jobs_changed': False, 'new_training_started': False,
        'all_cases': results}
    (args.out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'all_cases'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
