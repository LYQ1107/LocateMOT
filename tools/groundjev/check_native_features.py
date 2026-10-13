"""Real image: native serving-entry/reference agreement and ROI geometry."""
import copy
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.native_features import NativeGroundFeatures
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from safetensors import safe_open
    extractor=NativeGroundFeatures()
    row=json.loads((ROOT/'outputs/groundjev/runs/mot17_first_native_clip_repair2/frame_decisions.jsonl').read_text().splitlines()[0])
    with Image.open(row['image_path']) as source: image=source.convert('RGB')
    torch.cuda.reset_peak_memory_stats()
    spatial,metadata=extractor.encode(image)
    pooled,local=extractor.pool(spatial,metadata,[c['box'] for c in row['candidates']])
    assert torch.isfinite(pooled).all() and pooled.shape==(31,2560)
    from sglang_gam.gam_qwen3 import FastDVLMForConditionalGeneration
    pixels=extractor.processor(images=[image],return_tensors='pt')
    items=[SimpleNamespace(feature=pixels['pixel_values'].to('cuda:0'),image_grid_thw=pixels['image_grid_thw'].to('cuda:0'))]
    native=SimpleNamespace(visual=extractor.visual,config=extractor.config,use_data_parallel=False)
    with torch.inference_mode():
        routed=FastDVLMForConditionalGeneration.get_image_feature(native,items)
    expected=spatial[0].permute(1,2,0).reshape(-1,2560)
    serving_delta=float((routed-expected).abs().max())
    assert serving_delta==0.,serving_delta
    reference_class=get_class_from_dynamic_module('modeling_groundinganything.GroundAnythingVLMVisionModel',str(extractor.checkpoint),local_files_only=True)
    config=copy.deepcopy(extractor.config.vision_config)
    config._attn_implementation='eager'
    reference=reference_class(config)
    prefix='base_model.model.visual.'
    with safe_open(extractor.checkpoint/'model.safetensors',framework='pt') as source:
        state={key[len(prefix):]:source.get_tensor(key) for key in source.keys() if key.startswith(prefix)}
    reference.load_state_dict(state,strict=True,assign=True)
    reference.to(device='cuda:0',dtype=torch.bfloat16).eval().requires_grad_(False)
    with torch.inference_mode():
        reference_tokens=reference(pixels['pixel_values'].to('cuda:0'),grid_thw=pixels['image_grid_thw'].to('cuda:0')).last_hidden_state
    reference_delta=float((routed-reference_tokens).abs().max())
    assert reference_delta==0.,reference_delta
    resize=metadata['resize_and_padding'];gh,gw=metadata['merged_grid']
    cell=extractor.processor.patch_size*extractor.processor.merge_size
    # A real spatial map with one constant field and one x/y ramp gives an
    # independent check of ROI coordinates through native resize and padding.
    y,x=torch.meshgrid(torch.arange(gh,device='cuda:0'),torch.arange(gw,device='cuda:0'),indexing='ij')
    ramp=torch.stack([torch.ones_like(x),x,y]).float().unsqueeze(0)
    fx,fy,fw,fh=5.,6.,2.,2.
    scales=[resize['new_width']/image.width/cell,resize['new_height']/image.height/cell]
    box=[fx/scales[0],fy/scales[1],(fx+fw)/scales[0],(fy+fh)/scales[1]]
    value,_=extractor.pool(ramp,metadata,[box])
    expected_pool=torch.tensor([[1.,fx+fw/2-.5,fy+fh/2-.5]],device='cuda:0')
    geometry_delta=float((value-expected_pool).abs().max())
    assert geometry_delta<1e-5,geometry_delta
    result={'status':'REAL_NATIVE_DLM_FEATURES_VALIDATED','image':row['image_path'],'metadata':metadata,
        'region_shape':list(pooled.shape),'local_tokens_shape':list(local.shape),
        'unmodified_serving_entry_max_abs_difference':serving_delta,
        'immutable_checkpoint_reference_max_abs_difference':reference_delta,
        'independent_ROI_coordinate_ramp_max_abs_difference':geometry_delta,
        'reference_scope':'same real image and exact serving visual function, not a live-scheduler RPC',
        'language_decoder_not_loaded':True,'substituted_encoder':False,'GT_read':False,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'physical_gpu':5}
    out=ROOT/'outputs/groundjev/validation/native_ground_features.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
