"""Native DLM MoonViT/projector features from the official serving implementation.

Run in GroundAnything's isolated environment. No language decoder, substitute
encoder, or full-dataset feature cache is created by this adapter.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
# The formal G2 runtime is kept in an isolated worktree.  The environment
# override lets a clean LocateMOT checkout use that immutable upstream mirror
# without copying the full GroundAnything source tree into this repository.
SOURCE=Path(os.environ.get('LOCATEMOT_GROUNDANYTHING_ROOT', ROOT/'external/GroundAnything')).resolve()


class NativeGroundFeatures:
    def __init__(self,checkpoint=ROOT/'models/GroundAnything-DLM',device='cuda:0'):
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig,AutoProcessor
        checkpoint=Path(checkpoint).resolve()
        self.checkpoint,self.device=checkpoint,device
        sys.path[:0]=[str(SOURCE/'vendor/sglang'),str(SOURCE/'infer/engines'),str(SOURCE)]
        os.environ['GAM_SGLANG_MODEL_CODE_PATH']=str(checkpoint)
        os.environ['GAM_SGLANG_VISION_ATTN']='eager'
        from sglang_gam.gam_qwen3 import GroundAnythingVLMVisionModel
        self.config=AutoConfig.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
        self.feature_dimension=self.config.vision_config.out_hidden_size
        self.processor=AutoProcessor.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True).image_processor
        self.visual=GroundAnythingVLMVisionModel(self.config.vision_config)
        prefix='base_model.model.visual.'
        with safe_open(checkpoint/'model.safetensors',framework='pt') as source:
            state={name[len(prefix):]:source.get_tensor(name) for name in source.keys() if name.startswith(prefix)}
        self.weight_entries=len(state)
        if not state:
            raise RuntimeError('DLM checkpoint has no native visual weights')
        self.visual.load_state_dict(state,strict=True,assign=True)
        self.visual.to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
        self.parameter_count=sum(p.numel() for p in self.visual.parameters())

    def encode(self,image):
        import torch
        start=time.perf_counter()
        pixels=self.processor(images=[image],return_tensors='pt')
        resized=self.processor.get_resize_config({'type':'image','image':image})
        grid=pixels['image_grid_thw'].to(self.device)
        with torch.inference_mode():
            tokens=self.visual(pixels['pixel_values'].to(self.device),grid)
        t,h,w=grid[0].tolist()
        merge=self.visual.spatial_merge_size
        gh,gw=h//merge,w//merge
        if t!=1 or tokens.shape!=(gh*gw,self.feature_dimension):
            raise RuntimeError('Native merged feature layout disagrees with the official image grid')
        spatial=tokens.reshape(gh,gw,-1).permute(2,0,1).unsqueeze(0).contiguous()
        metadata={'feature_source':'official_DLM_native_MoonViT_and_trained_projector',
            'checkpoint':str(self.checkpoint),'weight_prefix':'base_model.model.visual.',
            'native_parameters':self.parameter_count,'native_weight_entries':self.weight_entries,
            'image_size':[image.width,image.height],'grid_thw':[t,h,w],'merged_grid':[gh,gw],
            'resize_and_padding':resized,'feature_dimension':tokens.shape[-1],
            'attention':'official eager','seconds':time.perf_counter()-start}
        return spatial,metadata

    def pool(self,spatial,metadata,boxes,local_size=3):
        import torch
        from torchvision.ops import roi_align
        if not boxes:
            # Nonempty ROIAlign interpolates spatial.float(); empty outputs
            # must obey the same dtype contract for the FP32 temporal adapter.
            return (spatial.new_empty((0,spatial.shape[1]),dtype=torch.float32),
                    spatial.new_empty((0,spatial.shape[1],local_size,local_size),dtype=torch.float32))
        width,height=metadata['image_size']
        resize=metadata['resize_and_padding']
        cell=self.processor.patch_size*self.processor.merge_size
        box_tensor=torch.tensor(boxes,device=spatial.device,dtype=torch.float32)
        scales=box_tensor.new_tensor([resize['new_width']/width/cell,resize['new_height']/height/cell]*2)
        feature_boxes=box_tensor*scales
        rois=torch.cat([torch.zeros((len(boxes),1),device=spatial.device),feature_boxes],dim=1)
        # ROIAlign uses float32 interpolation; native BF16 features remain the
        # source. Padding is explicit, so it is never mistaken for image area.
        local=roi_align(spatial.float(),rois,output_size=local_size,spatial_scale=1.,sampling_ratio=2,aligned=True)
        pooled=local.mean(dim=(-1,-2))
        return pooled,local

    def __call__(self,image,boxes):
        spatial,metadata=self.encode(image)
        pooled,local=self.pool(spatial,metadata,boxes)
        return pooled,local,metadata
