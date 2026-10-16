"""Posthoc native/learned category evidence, using official evaluator aliases."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.decision import overlap
    from tools.groundjev.evaluate_multitask import normalize_label
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest=json.loads((ROOT/plan['manifest']).read_text())
    source=ROOT/plan['run'];annotation=json.loads((source/'official_evaluation/gt.json').read_text())
    categories={c['id']:c for c in annotation['categories']}
    by_synset={c['synset']:c['id'] for c in annotation['categories']}
    aliases={normalize_label(alias):c['id'] for c in annotation['categories'] for alias in [c['name']]+c.get('synonyms',[])}
    vocabulary=json.loads(Path(manifest['vocabulary_path']).read_text())['categories']
    for category in vocabulary:
        if category['synset'] in by_synset:
            for alias in [category['name']]+category['synonyms']:
                aliases[normalize_label(alias)]=by_synset[category['synset']]
    truth=defaultdict(list)
    for obj in annotation['annotations']:truth[obj['image_id']].append(obj)
    counts=Counter();examples=[];frames=0
    with (source/'frame_decisions.jsonl').open() as f:
        for image_id,line in zip(manifest['image_ids'],f):
            row=json.loads(line);assert row['frame']==manifest['frame_numbers'][frames]
            frames+=1;candidates=row['candidates'];objects=truth[image_id]
            counts['native_candidates']+=len(candidates);counts['GT_boxes']+=len(objects)
            if not candidates or not objects:continue
            matrix=np.asarray([[overlap(c['box'],[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]]) for a in objects] for c in candidates])
            labels=[aliases.get(normalize_label(c['label']),-1) for c in candidates]
            for j,obj in enumerate(objects):
                group='novel' if categories[obj['category_id']]['frequency']=='r' else 'base'
                counts[group+':GT_boxes']+=1
                covered=matrix[:,j]>=.5
                counts[group+':any_native_box_support']+=bool(covered.any())
                counts[group+':any_native_box_and_label_support']+=any(covered[i] and label==obj['category_id'] for i,label in enumerate(labels))
            rr,cc=linear_sum_assignment(-matrix)
            emitted={e['candidate']:e for e in row['state']['events'] if e.get('action') in ('NEW','ACTIVE','REACTIVATE')}
            for i,j in zip(rr,cc):
                if matrix[i,j]<.5:continue
                obj=objects[j];group='novel' if categories[obj['category_id']]['frequency']=='r' else 'base'
                counts[group+':one_to_one_GTmatched']+=1
                counts[group+':one_to_one_native_label_correct']+=labels[i]==obj['category_id']
                if i not in emitted:continue
                counts[group+':emitted_one_to_one_GTmatched']+=1
                counts[group+':emitted_native_label_correct']+=labels[i]==obj['category_id']
                learned=emitted[i]['decision']['category_decision']['label']
                counts[group+':emitted_learned_label_correct']+=aliases.get(normalize_label(learned),-1)==obj['category_id']
                if len(examples)<20 and group=='novel':
                    examples.append(dict(frame=row['frame'],candidate=int(i),GT_category=categories[obj['category_id']]['name'],
                        native_label=candidates[i]['label'],learned_label=learned,IoU=float(matrix[i,j])))
        assert f.readline()==''
    assert frames==len(manifest['images'])==40 and counts['native_candidates']==35228
    result=dict(status='FROZEN_TAO40_NATIVE_AND_LEARNED_CATEGORY_EVIDENCE_AUDIT_COMPLETE',frames=frames,
        counts=dict(counts),examples=examples,
        category_alias_protocol='Exact normalize/annotation+fullLVISsynset alias construction from unchanged official TETA wrapper; no class filtering.',
        selected_GT_categories=[dict(name=categories[k]['name'],frequency=categories[k]['frequency']) for k in sorted({a['category_id'] for a in annotation['annotations']})],
        scope='External posthoc IoU>=.5 diagnostic only; same global class-agnostic one-to-one assignment as previous event diagnostics. Any-box+label support is an oracle coverage bound, not class accuracy, ClsA or tracking metrics.',
        GT_used_by_model_or_runtime=False,predictions_source_labels_and_metrics_changed=False,
        new_GPU_or_optimizer_updates=0,metadata_suballocation='Within existing temporal projection16MiB metadata reservation',
        full_goal_complete=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
