"""Compare real native Compact forward tensors with reference-only hydration."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.compact import CompactDecision
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.compact_reference_data import CompactReferenceInputs
    from tools.groundjev.run_video import ReplayFrames
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args();data=args.data
    plan=json.loads((data/'registration.json').read_text())
    prepared=json.loads((data/'owner_state.json').read_text())
    assert prepared['status']=='NATIVE_COMPACT_REFERENCE_INPUTS_PREPARED_NUMERICAL_REGRESSION_PENDING'
    torch.set_num_threads(4);torch.manual_seed(0)
    reader=CompactReferenceInputs(data)
    class Recorder(CompactDecision):
        def accepted_batch(self,group,ranked,inputs):
            self.captured.append({k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in inputs.items()})
    actor=Recorder(ROOT/plan['actor_checkpoint'],device='cpu')
    cases=[];counts=Counter()
    for unit in ('mot/MOT17-02-FRCNN','mot/MOT20-05','rmot/0000_primary'):
        info=reader.units[unit];name=unit.split('/',1)[1]
        manifest=json.loads((ROOT/info['manifest']).read_text())
        runtime=json.loads((ROOT/info['original_run']/'result.json').read_text())
        tracker=OnlineTracker(actor,lost_seconds=runtime['lost_seconds'],memory_policy=runtime['memory_policy'])
        specification=dict(task=manifest['task'],vocabulary=manifest['vocabulary'],expression=manifest.get('expression'))
        replay=ReplayFrames(ROOT/info['archive'])
        try:
            with (ROOT/info['original_run']/'frame_decisions.jsonl').open() as trace:
                for ordinal,line in enumerate(trace):
                    if ordinal==3:break
                    saved=json.loads(line);frame=saved['frame'];raw=replay.frame(frame)
                    actor.captured=[]
                    with Image.open(raw['image_path']) as source:image=source.convert('RGB')
                    outputs,state=tracker.step(image,raw['candidates'],frame/manifest['fps'],specification)
                    assert outputs==saved['outputs'] and state==saved['state'],(unit,frame,'source trajectory')
                    if not actor.captured:
                        record=reader.records[unit,f'{name}_{frame}_batch0']
                        assert record['compact_native_skips_empty_context']
                        counts['native_empty_contexts_retained_without_forward']+=1
                    for batch,expected in enumerate(actor.captured):
                        identifier=f'{name}_{frame}_batch{batch}'
                        hydrated=reader.inputs(unit,identifier)
                        assert hydrated.keys()==expected.keys()
                        for key,value in expected.items():
                            if torch.is_tensor(value):
                                assert value.dtype==hydrated[key].dtype and torch.equal(value,hydrated[key]),(unit,frame,batch,key)
                                counts['exact_tensor_elements']+=value.numel()
                            else:assert value==hydrated[key],(unit,frame,batch,key)
                        with torch.no_grad():a=actor.model(expected);b=actor.model(hydrated)
                        assert all(torch.equal(a[k],b[k]) for k in a),(unit,frame,batch,'forward')
                        counts['exact_native_forward_groups']+=1
                        counts['groups_with_history']+=bool(len(expected['history']))
                    cases.append(dict(unit=unit,frame=frame,candidates=len(raw['candidates']),
                        captured_groups=len(actor.captured),actual_outputs_and_state_exact=True))
                    counts['source_frames']+=1
        finally:replay.close()
    assert counts['groups_with_history']>0 and counts['source_frames']==9 and not torch.cuda.is_initialized()
    report=dict(status='REAL_NATIVE_COMPACT_REFERENCE_INPUTS_AND_FORWARD_TENSORS_VALIDATED',
        source_units_prepared=prepared['completed_units'],source_frame_contexts=prepared['counts']['source_frame_contexts'],
        unique_physical_images=prepared['unique_physical_images'],counts=dict(counts),cases=cases,
        all_reconstructed_model_input_tensors_and_frozen_forward_outputs_exact=True,
        original_export_all_source_actor_and_crop_pixel_validation_reused=True,
        numerical_regression_scope='First3 actual chronological frames in each of3 real sources, including dense MOT20 and referring input. Whole-source preparation preserves all61 units; this nine-frame numerical regression does not claim a retrained tracking improvement.',
        native_FP32_feature_vectors_copied=False,new_GPU_or_optimizer_updates=0,
        complete_three_task_training_pack=False,full_goal_complete=False)
    (data/'native_input_regression.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sum(p.stat().st_size for p in data.rglob('*') if p.is_file())<plan['new_metadata_cap_bytes']
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
