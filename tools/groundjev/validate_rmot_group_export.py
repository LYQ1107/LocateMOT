"""Actual unchanged MOT clip and existing-native RMOT CPU replay export checks."""
import argparse
from collections import Counter
from copy import copy
import gzip
from itertools import zip_longest
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def rows(path):
    with gzip.open(path, 'rt') as source:
        yield from map(json.loads, source)


def main():
    import numpy as np
    import torch
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.compact import CompactDecision
    from tools.groundjev.export_broader_native_full_groups import export_clip
    from tools.groundjev.export_rmot_full_fit_groups import context_for_manifest
    from tools.groundjev.group_reference_rows import REFERENCE_KEY
    from tools.groundjev.run_group_reference_sft import configure_group_reference_sft

    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(0)
    stateful = configure_group_reference_sft()

    class Recorder(CompactDecision):
        def accepted_batch(self, group, ranked, inputs):
            self.captured.append((group, [copy(t) for t in ranked]))

    mot_plan = json.loads((ROOT/'configs/groundjev/broader_native_full_v3_groups_v1.json').read_text())
    source = ROOT/mot_plan['source']
    name = next(p.stem for p in (source/'manifests').glob('MOT17-02*.json'))
    actor = Recorder(ROOT/mot_plan['actor_checkpoint'], device='cpu')
    mot = export_clip(name, source, args.out/'mot_default', mot_plan, actor, stateful)
    old = ROOT/mot_plan['output']/'clips'/name
    compared = Counter()
    for filename in ('groups.jsonl.gz', 'candidate_registry.jsonl.gz', 'train_refs.jsonl.gz'):
        for a, b in zip_longest(rows(old/filename), rows(args.out/'mot_default'/filename)):
            assert a is not None and b is not None
            if filename == 'train_refs.jsonl.gz':
                a[REFERENCE_KEY]['path'] = b[REFERENCE_KEY]['path']
            assert a == b
            compared[filename] += 1
    prior = json.loads((old/'validated_summary.json').read_text())
    assert all(mot[k] == prior[k] for k in ('counts', 'max_tokens', 'total_planned_tokens', 'dropped_rows'))
    print(json.dumps(dict(status='OLD_MOT_DEFAULT_ALL_GROUPS_TARGETS_REGISTRY_AND_CPU_PLAN_EXACT',
        sequence=name, compared=dict(compared))), flush=True)

    rmot_plan = json.loads((ROOT/'configs/groundjev/rmot_allfit1814_query18_full_v3_groups_v1.json').read_text())
    collector = json.loads((ROOT/rmot_plan['source_collector_plan']).read_text())
    validation = json.loads((ROOT/rmot_plan['source_scope']/'validation.json').read_text())
    name = '0003_primary'
    index, item = next((i, x) for i, x in enumerate(validation['scopes']) if x['name'] == name)
    manifest = json.loads((ROOT/item['manifest']).read_text())
    native = [r for r in map(json.loads,
        (ROOT/'outputs/groundjev/compact/rmot_native72_v1/native_rollout.jsonl').open()) if r['sequence'] == '0003']
    assert len(native) == 8
    manifest.update(images=[r['image_path'] for r in native], frame_numbers=[r['frame'] for r in native],
        scope='existing_native_candidate_CPU_replay_export_validation_only_not_new_DLM_collection')
    manifest_path = args.out/'rmot8_manifest.json'; manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    original_input = args.out/'existing_native_rmot8_candidates.jsonl'
    with original_input.open('w') as output:
        for r in native:
            output.write(json.dumps({k: r[k] for k in ('frame', 'image_path', 'candidates', 'perception')})+'\n')
    original = args.out/'rmot8_CPU_replay'
    command = [sys.executable, 'tools/groundjev/run_video.py', '--task', 'rmot', '--video', str(manifest_path),
        '--backend', 'compact', '--compact-device', 'cpu', '--checkpoint', str(ROOT/collector['checkpoint']),
        '--candidate-replay', str(original_input), '--record-native-candidates',
        '--native-candidate-format', 'packed_npz_f32', '--native-candidate-byte-budget', str(8*2**20),
        '--omit-replay-feature-vectors', '--memory-checkpoint-format', 'binary', '--memory-export-interval', '0',
        '--out', str(original)]
    with (args.out/'rmot8_CPU_replay.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    context = context_for_manifest(manifest, original, rmot_plan['label_root'], index)
    rmot = export_clip(name, ROOT/rmot_plan['source'], args.out/'rmot8_export', rmot_plan,
        actor, stateful, context=context)

    # Independently check the new task's labels against the full-frame audit's
    # actual official query masks and a separate IoU implementation.
    masks = {r['frame']: set(r['projected_positive_IDs']) for r in json.loads(
        (ROOT/rmot_plan['source_scope']/'annotation_projection_contexts.json').read_text())
        if r['video'] == '0003' and r['role'] == 'primary'}

    def iou(a, b):
        left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0., right-left)*max(0., bottom-top)
        return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

    expected, qualities = {}, {}
    for r in native:
        frame = r['frame']
        with Image.open(r['image_path']) as im:
            w, h = im.size
        boxes, query_boxes = [], []
        label = Path(rmot_plan['label_root'])/'0003'/f'{frame:06d}.txt'
        for line in label.read_text().splitlines():
            fields = line.split(); x, y, bw, bh = map(float, fields[2:6])
            box = [x*w, y*h, (x+bw)*w, (y+bh)*h]; boxes.append(box)
            if int(fields[1]) in masks[frame]:
                query_boxes.append(box)
        candidates = r['candidates']
        matrix = np.asarray([[iou(c['box'], box) for box in boxes] for c in candidates])
        matching = {}
        if candidates and boxes:
            rr, cc = linear_sum_assignment(-matrix)
            matching = {int(i): float(matrix[i, j]) for i, j in zip(rr, cc) if matrix[i, j] >= .5}
        for i, candidate in enumerate(candidates):
            expected[frame, tuple(candidate['box'])] = 'yes' if any(iou(candidate['box'], b) >= .5 for b in query_boxes) else 'no'
            qualities[frame, tuple(candidate['box'])] = matching.get(i, 0.)
    groups = {r['id']: r for r in rows(args.out/'rmot8_export/groups.jsonl.gz')}
    labels = Counter()
    for row in rows(args.out/'rmot8_export/train_refs.jsonl.gz'):
        ref = row[REFERENCE_KEY]; group = groups[ref['group']]
        frame = int(group['id'][len(name)+1:].split('_batch')[0])
        assert group['state']['task'] == context['specification']
        if row['task'] not in ('query_membership', 'reliability'):
            continue
        local = int(ref['question'].rsplit('_', 1)[1])
        box = tuple(group['state']['candidates'][local]['box'])
        actual = next(iter(row['target']))
        if row['task'] == 'query_membership':
            assert actual == expected[frame, box]
            labels['membership:'+actual] += 1
        else:
            quality = qualities[frame, box]
            level = '0' if quality < .5 else '1' if quality < .7 else '2' if quality < .85 else '3'
            assert actual == level
            labels['reliability:'+actual] += 1
    assert labels['membership:yes'] == 17 and labels['membership:no'] == 19
    assert sum(v for k, v in labels.items() if k.startswith('reliability:')) == 36
    assert not torch.cuda.is_initialized()
    total_bytes = sum(p.stat().st_size for directory in (ROOT/rmot_plan['source_scope'], ROOT/rmot_plan['output'])
        if directory.exists() for p in directory.rglob('*') if p.is_file() and not p.is_symlink())
    assert total_bytes < rmot_plan['scope_and_export_budget_bytes']
    report = dict(status='MOT_DEFAULT_EXACT_AND_REAL_RMOT8_NATIVE_INPUTS_LABELS_PIXELS_CPU_PLAN_VALIDATED',
        MOT_default=mot, MOT_unchanged_rows_and_groups=dict(compared), RMOT_CPU_replay=rmot,
        RMOT_independent_official_mask_membership_and_quality_labels=dict(labels),
        actual_new_DLM_queries=0, actual_new_Physical_frames=0, actual_new_training_updates=0,
        all_crop_pixels_and_actor_outputs_states_exact=True, new_GPUs=0,
        scope_and_export_bytes=total_bytes, scope_and_export_cap=rmot_plan['scope_and_export_budget_bytes'],
        real1814_source_not_yet_collected=True, full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('MOT_default', 'RMOT_CPU_replay')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
