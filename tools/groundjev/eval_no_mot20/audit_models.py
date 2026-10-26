"""Read-only validation of the exact immutable checkpoints used by this round."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import GROUND,RUN,TRAIN_ROOT,G2_SHA,TRAIN_SHA,load,save,sha_file


def main():
    import torch
    from locatemot.groundjev.coupled.groundjev_model import GroundJEVModel
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=RUN/'model_audit.json');args=parser.parse_args()
    root=TRAIN_ROOT/'formal_clean_coverage_v2_final'
    final=load(root/'final_receipt.json')
    assert final['steps']==18227 and final['status']=='G2_EXACT_COVERAGE_TRAINING_COMPLETE'
    assert final['manifest_sha256']==TRAIN_SHA
    export=load(GROUND/'outputs/groundjev/ground_adaptation/g2_clean_bundle_export_v1/export_receipt.json')
    checks={}
    for name,path,expected in [
        ('G2',Path(export['output_bundle'])/'model.safetensors',G2_SHA),
        ('G2_adapter',root/'adapter_projector_final.pt',export['adapter_sha256']),
        ('J1_adapter',Path('/data3/user/LocateMOT-runs/TrackJEV_FullDataV2/trackjev_3task_J1_v4_birth/adapter/adapter_model.safetensors'),
         '386f1a48729a050793053b5d970938b1c657d0a175081e2ea32040f97e360437')]:
        actual=sha_file(path)
        if actual!=expected:raise ValueError(f'{name} checkpoint digest differs')
        checks[name]={'path':str(path),'sha256':actual,'bytes':path.stat().st_size}
    checkpoint=GROUND/'outputs/groundjev/coupled/fixed_candidate_repair_mot20_03_v1/adapter-step000180.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=GroundJEVModel(2560,256);model.load_state_dict(saved['model_state'],strict=True)
    checks['coupled']={'path':str(checkpoint),'sha256':sha_file(checkpoint),'step':saved['step'],
        'feature_dim':2560,'hidden_dim':256,'parameters':sum(t.numel() for t in model.parameters()),
        'supported_tasks':['mot'],'training_scope':'MOT20-03 first 64 frames; fixed native base-G0 features; 180 updates',
        'checkpoint_base_sha':saved['model_sha256'],
        'evaluation_feature_shift':'final G2 visual projector; provider SHA recorded; no claim of retraining on G2 final'}
    merged=Path('/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/post_j1_heldout_gate_v1/merged_3task_J1_v4_birth')
    receipt=load(merged/'merge_receipt.json')
    assert receipt['unmatched_count']==0 and receipt['tensors_updated']==186
    weights={str(p.name):sha_file(p) for p in merged.glob('*.safetensors')}
    assert weights
    checks['J1_merged']={'path':str(merged),'receipt':receipt,'weight_sha256':weights,'step':1200}
    checks['G2_contract']={'step':18227,'manifest_sha256':TRAIN_SHA,'export':export}
    checks['status']='IMMUTABLE_MODELS_CPU_LOAD_AND_DIGEST_VALIDATED_GPU_PREFLIGHT_PENDING'
    save(args.out,checks)
    print(json.dumps({'status':checks['status'],'G2_SHA':G2_SHA,'coupled_load':'strict_success','out':str(args.out)}),flush=True)


if __name__=='__main__':main()
