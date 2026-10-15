"""Freeze completed native task artifacts while the full9 owners keep running."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'external/GroundAnything'
sys.path[:0] = [str(ROOT),str(SOURCE)]


def main():
    from scripts.evaluate import summarize
    runs = {'original':'groundjev_original_dense_ref9_queue32_v3_20261014',
        'adapted200':'groundjev_trackproj200_dense_ref9_sync_v3_20261015'}
    output = ROOT/'outputs/groundjev/ground_retention/partial_completed_tasks_20261015_v1'
    output.mkdir(parents=True,exist_ok=False)
    variants = {}
    for name,directory in runs.items():
        root = SOURCE/'outputs/groundjev_retention'/directory
        run = json.loads((root/'run.json').read_text())
        tasks = run['plan']['tasks']
        assert len(tasks)==9
        native = summarize(root,tasks)
        records = {}
        for record in native['records']:
            task = record['task']
            assert task not in records
            path = root/record['source']
            actual = json.loads(path.read_text())
            n = actual['n-samples'][task]
            assert n['original']==n['effective'] and n['original']>0 and n['provider_policy_filtered']==0
            assert actual['results'][task]==record['metrics']
            saved = output/name/(task+'.json');saved.parent.mkdir(exist_ok=True)
            artifact = {'source':str(path),'results':{task:actual['results'][task]},'n-samples':{task:n},
                'scope':'Exact metric/sample-count fields extracted from the completed original native task result; full source artifact retained locally.'}
            saved.write_text(json.dumps(artifact,indent=2)+'\n')
            records[task] = {'source':str(path),'snapshot':str(saved),'metrics':record['metrics'],'n-samples':n}
        variants[name] = {'run':str(root),'registered_tasks':tasks,'completed':records,
            'missing_tasks':native['missing_tasks'],'completed_tasks':len(records),
            'completed_native_rows':sum(r['n-samples']['original'] for r in records.values()),
            'all9_complete':len(records)==9 and not native['missing_tasks']}
    a,b = variants['original']['completed'],variants['adapted200']['completed']
    common = {}
    keys = ('recall_iou_50,none','precision_iou_50,none','f1_iou_50,none','F1mIoU,none')
    for task in sorted(set(a)&set(b)):
        assert a[task]['n-samples']==b[task]['n-samples']
        common[task] = {'rows':a[task]['n-samples']['original'],'metrics':{
            key:{'original':a[task]['metrics'][key],'adapted200':b[task]['metrics'][key],
                'difference_percentage_points':100*(b[task]['metrics'][key]-a[task]['metrics'][key])}
            for key in keys}}
    report = {'status':'AVAILABLE_COMPLETED_NATIVE_TASK_ARTIFACTS_VALIDATED_FULL9_PENDING',
        'variants':variants,'common_completed_tasks':common,'full_goal_complete':False,
        'new_model_calls_or_modified_sources':False,
        'limits':'Only completed whole tasks, not partial prefixes or an aggregate full9 result. '
            'OriginalCOCO is reused through the registered native resume receipt. Originalqueue32 '
            'and adapted-sync execution differ after recorded CUDA recovery; no bitwise or '
            'paired-stochastic-request-counter claim and no exclusive causal attribution to weights. '
            'These are grounding retention metrics, not HOTA/IDF1 or ordinary-model-level MOT/RMOT/OVMOT parity. '
            'Original owners/services continue; missing task scores are not inferred.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'completed':{k:v['completed_tasks'] for k,v in variants.items()},'common':common}))


if __name__=='__main__':
    main()
