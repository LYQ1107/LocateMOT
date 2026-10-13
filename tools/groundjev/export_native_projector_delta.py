"""Export a native projector delta into a fresh complete DLM bundle, preserving all other weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    import torch
    from safetensors.torch import load_file,save_file
    parser=argparse.ArgumentParser();parser.add_argument('--delta',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve the existing adapted bundle')
    source=ROOT/'models/GroundAnything-DLM'
    delta=torch.load(args.delta,map_location='cpu',weights_only=True)
    assert Path(delta['initial_DLM_checkpoint']).resolve()==source.resolve()
    state=load_file(source/'model.safetensors',device='cpu')
    before_keys=set(state);prefix='base_model.model.visual.projector.'
    changed=0;replacements=[]
    for name,tensor in delta['native_projector_state_dict'].items():
        key=prefix+name
        assert key in state and tensor.shape==state[key].shape
        value=tensor.to(state[key].dtype).contiguous()
        changed+=int((state[key]!=value).sum());state[key]=value;replacements.append(key)
    assert changed>0 and set(state)==before_keys
    args.out.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name!='model.safetensors':shutil.copyfile(path,args.out/path.name)
    save_file(state,args.out/'model.safetensors')
    digest=hashlib.sha256()
    with (args.out/'model.safetensors').open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    report={'status':'COMPLETE_NATIVE_DLM_PROJECTOR_DELTA_BUNDLE_EXPORTED','source':str(source),
        'delta':str(args.delta.resolve()),'output':str(args.out.resolve()),'native_tensor_count':len(state),
        'replaced_native_projector_keys':replacements,'BF16_changed_elements':changed,
        'all_other_native_tensors_unmodified':True,'source_bundle_overwritten':False,
        'tokenizer_code_and_configuration':'copied unchanged from complete released DLM bundle',
        'bytes':(args.out/'model.safetensors').stat().st_size,'sha256':digest.hexdigest(),
        'training_steps':delta['steps'],'seed':delta['seed'],'dense_referring_retention':'pending actual before/after',
        'benchmark_improvement_claim':False}
    args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
