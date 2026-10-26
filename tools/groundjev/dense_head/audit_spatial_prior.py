import sys,json,os,time
from pathlib import Path
root=Path('/data2/user/LocateMOT_GroundJEV_DenseHead');sys.path.insert(0,str(root))
import torch
from PIL import Image
from locatemot.groundjev.dense_head.spatial_backbone import SpatialBackbone
from locatemot.groundjev.dense_head.object_decoder import ObjectDecoder
from locatemot.groundjev.dense_head.hungarian_criterion import HungarianCriterion
from locatemot.groundjev.dense_head.model import targets_on_grid,valid_ratio
from tools.groundjev.dense_head.prepare_train_data import RUN,write_json

def main():
    started=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(42)
    b=SpatialBackbone(root/'models/GroundAnything-DLM-G2-CLEAN-V2');r=json.loads((RUN/'data/train.jsonl').open().readline())
    with Image.open(r['image_path']) as im:f,m=b.encode(im.convert('RGB'))
    h=ObjectDecoder(queries=100,spatial_attention_sigma=.08).cuda();opt=torch.optim.AdamW(h.parameters(),lr=1e-4);criterion=HungarianCriterion();target=targets_on_grid(r['boxes'],m,'cuda:0');losses=[]
    for _ in range(3):
     opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.bfloat16):p=h(f,valid_ratio(m));l=criterion(p,target)['loss']
     assert torch.isfinite(l);l.backward();opt.step();losses.append(float(l.detach()))
    grads={n:float(p.grad.norm()) for n,p in h.named_parameters() if p.grad is not None};assert len(grads)==len(list(h.parameters())) and all(v>0 for v in grads.values())
    assert all(p.grad is None and not p.requires_grad for p in b.visual.parameters())
    h.eval()
    with torch.no_grad():a=h(f,valid_ratio(m))
    path=RUN/'spatial_prior_v2/smoke_head.pt';torch.save(h.state_dict(),path)
    z=ObjectDecoder(queries=100,spatial_attention_sigma=.08).cuda();z.load_state_dict(torch.load(path,weights_only=True));z.eval()
    with torch.no_grad():v=z(f,valid_ratio(m))
    assert torch.equal(a['boxes'],v['boxes']) and torch.equal(a['logits'],v['logits'])
    receipt={'status':'SPATIAL_PRIOR_REAL_GPU_SMOKE_PASSED','GPU':os.environ['CUDA_VISIBLE_DEVICES'],'feature_shape':list(f.shape),'sigma':.08,'trainable_parameters':sum(p.numel() for p in h.parameters()),'losses':losses,'all_gradients_nonzero':True,'reference_logits_gradient_norm':grads['reference_logits'],'checkpoint_restore_exact':True,'frozen_backbone':True,'GT_in_attention_prior':False,'elapsed_seconds':time.perf_counter()-started}
    write_json(RUN/'spatial_prior_v2/spatial_prior_smoke.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=="__main__":main()
