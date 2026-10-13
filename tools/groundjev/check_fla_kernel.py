"""Compare isolated FLA GDN kernels to the immutable Torch reference on A800."""
import inspect
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.tools/onejev-fla052'))


def main():
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    reference=inspect.unwrap(torch_chunk_gated_delta_rule)
    torch.manual_seed(0)
    results=[]
    for length in [257,1024,4096]:
        q=torch.randn((1,length,16,128),device='cuda:0',dtype=torch.bfloat16)
        k=torch.randn_like(q);v=torch.randn_like(q)
        g=-torch.rand((1,length,16),device='cuda:0',dtype=torch.float32)
        beta=torch.randn((1,length,16),device='cuda:0',dtype=torch.bfloat16).sigmoid()
        def call(function,gradient=False):
            values=[x.detach().clone().requires_grad_(gradient) for x in (q,k,v,g,beta)]
            torch.cuda.synchronize();start=time.perf_counter()
            output,state=function(*values[:3],g=values[3],beta=values[4],output_final_state=True,use_qk_l2norm_in_kernel=True)
            if gradient:output.float().square().mean().backward()
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            return output.detach(),state.detach(),[x.grad.detach() for x in values] if gradient else [],seconds
        # One compilation warmup is excluded from steady-state kernel timings.
        call(chunk_gated_delta_rule)
        call(reference)
        a,sa,_,ta=call(reference)
        b,sb,_,tb=call(chunk_gated_delta_rule)
        record={'length':length,'shape':list(q.shape),'reference_seconds':ta,'fla_seconds':tb,
            'speed_ratio':ta/tb,'output_max_abs_difference':float((a-b).abs().max()),
            'output_RMSE':float((a.float()-b.float()).square().mean().sqrt()),
            'state_max_abs_difference':float((sa-sb).abs().max())}
        if length==257:
            _,_,ga,_=call(reference,True);_,_,gb,_=call(chunk_gated_delta_rule,True)
            record['gradient_max_abs_differences']=[float((x-y).abs().max()) for x,y in zip(ga,gb)]
            record['finite_nonzero_gradients']=all(bool(torch.isfinite(x).all()) and bool((x!=0).any()) for x in gb)
        results.append(record);print(json.dumps(record),flush=True)
    report={'status':'ISOLATED_FLA_KERNEL_REFERENCE_COMPARISON_COMPLETE','seed':0,
        'fla_version':'0.5.2','torch_version':torch.__version__,'GPU':torch.cuda.get_device_name(0),
        'results':results,'actual_model_probability_verification':'pending',
        'adopted_in_running_training':False,'end_to_end_speed_claim':False,
        'original_reference_environment_modified':False}
    out=ROOT/'outputs/groundjev/optimization/fla_kernel_reference.json'
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
