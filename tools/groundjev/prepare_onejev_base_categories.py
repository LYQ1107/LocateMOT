"""Full native category question supervision from previously registered train-Base crops."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.category import NativeCategoryHead
    from qev.prompt import render_question
    from train import common
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'outputs/groundjev/compact/joint_pack_v3'
    owner = json.loads((source/'owner_state.json').read_text())
    assert owner['TAO_split'] == 'train-Base-only' and not owner['Novel_GT_supervised']
    frames = [r for r in json.loads((source/'frame_scope.json').read_text()) if r['task'] == 'ovmot']
    frame_by_key = {(r['scope'], r['frame']): r for r in frames}
    categories = json.loads(Path('/data2/user/LocateMOT/outputs/unified/u2/protocol/lvis_v1_vocabulary.json').read_text())['categories']
    vocabulary = [c['name'].replace('_', ' ') for c in categories]
    pack = torch.load(source/'examples.pt', map_location='cpu', weights_only=True)
    assert vocabulary == pack['vocabulary'] and len(vocabulary) == 1203
    head = NativeCategoryHead(None)
    counts, class_counts, scope = Counter(), Counter(), []
    with (args.out/'train.jsonl').open('w') as output:
        for example in pack['examples']:
            if example['task_index'] != 2:
                continue
            assert example['proposal_source'] == 'teacherGT'
            registered = frame_by_key[example['source_sequence'], example['frame']]
            assert '/frames/train/' in registered['image_path']
            with Image.open(registered['image_path']) as raw:
                image = raw.convert('RGB')
            labels = example['class_target'].tolist()
            assert all(categories[i]['frequency'] in ('c', 'f') for i in labels)
            boxes = example['candidate_boxes']
            assert len(boxes) == len(labels)
            for offset in range(0, len(boxes), head.batch_candidates):
                batch = [{'box': b} for b in boxes[offset:offset+head.batch_candidates]]
                targets = labels[offset:offset+head.batch_candidates]
                request, media, groups = head.build_root_request(image, batch, vocabulary)
                roots = {i: str(target//head.group_size) for i, target in enumerate(targets)}
                leaf = head.build_leaf_request(request.state, groups, roots)
                directory = args.out/'evidence'; directory.mkdir(exist_ok=True)
                file = directory/f"{example['id']}_{offset}.png"
                media[0]['image'].save(file)
                files = [{'type': 'image', 'path': str(file), 'w': media[0]['image'].width, 'h': media[0]['image'].height}]
                for i, target in enumerate(targets):
                    class_counts[vocabulary[target]] += 1
                    for kind, req, key, answer in [('category_group', request, f'group_{i}', roots[i]),
                                                  ('category_leaf', leaf, f'class_{i}', str(target % head.group_size))]:
                        question = req.questions[key]
                        common.target_vector({answer: 1.}, render_question('q', question).labels)
                        sample = {'id': f"base_category_{example['id']}_{offset}_{key}", 'split': 'train',
                            'source': example['source_sequence'], 'task': kind, 'state': req.state,
                            'question': question.model_dump(exclude_none=True), 'target': {answer: 1.}, 'media': files,
                            'supervision': {'frame': example['frame'], 'candidate_source': 'registered train-Base GT teacher crops',
                                'GroundAnything_prediction_claim': False, 'GT_class_or_identity_in_state': False,
                                'all1203_input_categories_retained': True, 'Novel_GT_supervision': False,
                                'leaf_teacher_forcing': 'conditional correct root during supervised training; runtime root chosen by native model',
                                'unknown_supervision': False, 'temporal_or_actor_history_claim': False}}
                        output.write(json.dumps(sample)+'\n'); counts[kind] += 1
                    counts['Base_objects'] += 1
                scope.append({'source_group': example['id'], 'offset': offset, 'objects': len(batch)})
    assert len(frames) == 64 and counts['category_group'] == counts['category_leaf'] == counts['Base_objects']
    summary = {'status': 'REGISTERED_BASE_NATIVE_CATEGORY_QUESTION_SUPERVISION_COMPLETE',
        'counts': dict(counts), 'rows': counts['category_group']+counts['category_leaf'],
        'registered_frames': frames, 'source_groups': scope, 'class_counts': dict(class_counts),
        'original_pack': str(source/'examples.pt'), 'all_input_vocabulary_classes': 1203,
        'Novel_or_validation_GT_supervised': False, 'new_native_DLM_rollout_claim': False,
        'teacher_GT_crop_stage_explicit': True, 'empty_frames_retained_in_registration': True,
        'empty_frames_with_no_crop_have_no_category_rows': True,
        'unknown_negative_supervision_added': False, 'native_runtime_or_training_complete': False,
        'evidence_png_bytes': sum(p.stat().st_size for p in (args.out/'evidence').glob('*.png'))}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('registered_frames', 'source_groups')}, indent=2))


if __name__ == '__main__':
    main()
