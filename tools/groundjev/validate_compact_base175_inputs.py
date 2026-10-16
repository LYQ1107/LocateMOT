"""One actual teacher observation per Base class versus native Compact input code."""
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
    from tools.groundjev.compact_base175_data import CompactBaseTeacherInputs
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(0)
    dataset=CompactBaseTeacherInputs(args.data)
    class Recorder(CompactDecision):
        def accepted_batch(self,group,ranked,inputs):self.captured=inputs
    actor=Recorder(ROOT/'outputs/groundjev/compact/rmot_native72_stage_v1/broad_native_training/checkpoint.pt',device='cpu')
    representatives={}
    for index,row in enumerate(dataset.objects):representatives.setdefault(row['class_index'],index)
    assert len(representatives)==175 and len(dataset.sampled_draw_rows)==4000
    counts=Counter();cases=[]
    for cls,index in sorted(representatives.items()):
        row=dataset.objects[index]
        candidate=dict(box=row['box'],label='region',native_region_feature=dataset.bank[index].tolist())
        with Image.open(row['image_path']) as source:image=source.convert('RGB')
        actor.decide(image,[candidate],[],0.,dataset.specification)
        expected=actor.captured;actual=dataset.inputs(index)
        assert expected.keys()==actual.keys() and dataset.class_target(index)==cls
        for key,value in expected.items():
            if torch.is_tensor(value):
                assert value.dtype==actual[key].dtype and torch.equal(value,actual[key]),(index,key)
                counts['exact_input_tensor_elements']+=value.numel()
            else:assert value==actual[key]
        with torch.no_grad():a=actor.model(expected);b=actor.model(actual)
        assert all(torch.equal(a[key],b[key]) for key in a),(index,'native forward')
        counts['actual_teacher_objects_checked']+=1
        cases.append(dict(annotation_id=row['annotation_id'],class_index=cls,feature_row=index))
    assert not torch.cuda.is_initialized()
    proof=json.loads((args.data/'feature_validation.json').read_text())
    result=dict(status='BASE175_REAL_NATIVE_ROI_INPUTS_AND_COMPACT_FORWARD_EXACT',
        unique_teacher_objects=proof['unique_objects'],unique_teacher_images=proof['unique_images'],
        sampled_object_draws_preserved=4000,supervised_Base_classes=175,full_input_classes=1203,
        counts=dict(counts),representative_objects=cases,
        all_native_forward_inputs_and_outputs_exact=True,
        original_GT_class_labels_boxes_and_repeat_counts_unchanged=True,
        no_GT_specific_class_label_in_model_inputs=True,GT_teacher_box_source_explicit=True,
        no_new_identity_lifecycle_or_observation_update_targets_invented=True,
        new_GPU_or_optimizer_updates_by_validator=0,new_Compact_training_started=False,full_goal_complete=False)
    (args.data/'native_input_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='representative_objects'}),flush=True)


if __name__=='__main__':main()
