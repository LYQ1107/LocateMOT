"""Prepare every available legal fit frame for two independent native query rollouts."""
import argparse
import ast
from collections import Counter
import json
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OFFICIAL_COMMIT = '6a65640d849fdee4a32bb055945ee34c3b0edeb1'


def main():
    import numpy as np
    import torch
    from PIL import Image
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS, assert_legal_video

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT'
    out = ROOT/plan['scope_output']; out.mkdir(parents=True, exist_ok=False)
    manifests = out/'manifests'; manifests.mkdir()
    old = json.loads((ROOT/plan['original_query_plan']).read_text())
    assert {c['video'] for c in old['clips']} == set(LEGAL_SPLITS['refer_kitti_v2']['fit'])
    stats, contexts, scopes = Counter(), [], []
    # Execute the exact previously pinned method. The KITTI token is required
    # by that reader to interpret top-left xywh, so use directory symlinks.
    source = Path('/data2/user/reference_envs/temprmot')/f'refer_kitti_{OFFICIAL_COMMIT}.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DetMOTDetection')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_pre_single_frame')
    namespace = dict(np=np, torch=torch, Image=Image, osp=osp)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    official_reader = namespace['_pre_single_frame']
    kit = out/'KITTI'; kit.mkdir()
    image_root = Path(plan['image_root']); label_root = Path(plan['label_root'])
    (kit/'images').symlink_to(image_root, target_is_directory=True)
    (kit/'labels').symlink_to(label_root, target_is_directory=True)

    for clip in old['clips']:
        video = assert_legal_video(clip['video'], dataset='refer_kitti_v2')
        assert video in LEGAL_SPLITS['refer_kitti_v2']['fit']
        images = sorted((image_root/video).glob('*.png'), key=lambda p: int(p.stem))
        frames = [int(p.stem) for p in images]
        assert frames == list(range(frames[0], frames[-1]+1))
        assert set(clip['frames']) <= set(frames)
        stats['physical_frames'] += len(frames)
        stats['new_physical_frames_beyond_original72'] += len(set(frames)-set(clip['frames']))
        stats['videos'] += 1
        for role in ('primary', 'counter'):
            expression_path = Path(clip[role+'_expression_file'])
            query = json.loads(expression_path.read_text())
            assert not query.get('ignore') and query['sentence'] == clip[role+'_expression']
            name = f'{video}_{role}'
            manifest = dict(task='rmot', dataset='refer_kitti_v2', sequence=video,
                query_role=role, fps=clip['fps'], images=list(map(str, images)), frame_numbers=frames,
                vocabulary=['person', 'car'], expression=query['sentence'],
                expression_file=str(expression_path),
                scope='all_available_fit_frames_independent_native_query_rollout_not_benchmark',
                frame_selection='all available image frames in chronological order; no GT selection',
                physical_video_fps_verified=True)
            (manifests/(name+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')
            positive = 0
            for image_path, frame in zip(images, frames):
                label = label_root/video/(image_path.stem+'.txt')
                requested = set(map(int, query['label'].get(str(frame), [])))
                if not label.exists():
                    # Existing empty-frame convention: preserve the actual
                    # frame, record missing source annotation, fabricate none.
                    assert not requested
                    stats['missing_empty_label_query_contexts'] += 1
                    contexts.append(dict(video=video, frame=frame, role=role,
                        released_boxes=0, projected_positive_IDs=[], missing_empty_label_file=True))
                    continue
                with Image.open(image_path) as image:
                    width, height = image.size
                fields = [line.split() for line in label.read_text().splitlines() if line.strip()]
                ids = [int(f[1]) for f in fields]
                label_alias = str(kit/'labels'/video/label.name)
                owner = SimpleNamespace(img_files=[str(kit/'images'/video/image_path.name)],
                    label_files=[label_alias], video_dict={str(Path(label_alias).parent): 0})
                official_image, target = official_reader(owner, 0, [query])
                official_image.close()
                assert target['obj_ids'].tolist() == ids
                expected_boxes = []
                for f in fields:
                    x, y, w, h = map(float, f[2:6])
                    expected_boxes.append([x*width, y*height, (x+w)*width, (y+h)*height])
                assert len(target['boxes']) == len(expected_boxes)
                assert all(max(abs(a-b) for a, b in zip(actual, expected)) < .001
                    for actual, expected in zip(target['boxes'].tolist(), expected_boxes))
                projected = [identity for identity, flag in zip(ids, target['is_ref'].tolist()) if flag]
                assert projected == [i for i in ids if i in requested]
                assert not requested-set(ids)
                positive += bool(projected)
                stats['official_reader_query_contexts'] += 1
                contexts.append(dict(video=video, frame=frame, role=role,
                    released_boxes=len(ids), projected_positive_IDs=projected,
                    missing_empty_label_file=False))
            scopes.append(dict(name=name, video=video, query_role=role, frames=len(frames),
                first_frame=frames[0], last_frame=frames[-1], positive_query_frames=positive,
                expression_file=str(expression_path), expression=query['sentence'],
                manifest=str((manifests/(name+'.json')).relative_to(ROOT))))
            stats['query_frame_contexts'] += len(frames)
    assert stats['physical_frames'] == plan['expected_physical_frames']
    assert stats['query_frame_contexts'] == plan['expected_query_frame_contexts']
    assert len(contexts) == stats['query_frame_contexts'] and len(scopes) == 18
    assert not torch.cuda.is_initialized()
    (out/'annotation_projection_contexts.json').write_text(json.dumps(contexts, indent=2)+'\n')
    report = dict(status='ALL_FIT1814_PHYSICAL_FRAMES_18_NATIVE_QUERY_SCOPES_VALIDATED_NOT_COLLECTED',
        counts=dict(stats), scopes=scopes, official_commit=OFFICIAL_COMMIT,
        unmodified_official_reader_executed=True, frame_selection_GT_read=False,
        GT_read_only_for_supervision_contract_audit=True, new_GPU_model_calls=0,
        new_training_updates=0, original6000_training_inputs_changed=False,
        full_goal_complete=False)
    (out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'scopes'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
