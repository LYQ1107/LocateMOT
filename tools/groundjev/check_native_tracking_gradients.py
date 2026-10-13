"""Three-step native DLM projector adaptation with dense spatial and temporal identity losses."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/GroundAnything'),str(ROOT/'.tools/ground-train')]


def main():
    import torch
    from torch._functorch import config as functorch_config
    # Independent gradient probes retain the same graph. Native compiled
    # backward donation is incompatible with retain_graph, so disable only
    # this task-local buffer-reuse optimization, as directed by its error.
    functorch_config.donated_buffer=False
    import torch.nn.functional as F
    from PIL import Image
    from safetensors.torch import load_file
    from safetensors import safe_open
    from accelerate import init_empty_weights
    from transformers import AutoConfig,AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    os.environ['GAM_DLM_FLEX_CUDAGRAPHS']='0'
    from models.dlm.vlm import GAMQwen3DLM
    from train.dlm.data import DLMDataCollator
    from grounding_anything.protocol import GroundingEntry,serialize_answer,build_dense_bbox_prompt
    from locatemot.groundjev.native_features import NativeGroundFeatures
    from tools.groundjev.prepare_tracking_decisions import annotations,PROTOCOL
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'outputs/groundjev/ground_adaptation/native_tracking_grad3')
    args=parser.parse_args();out=args.out
    if out.exists() and any(out.iterdir()):raise FileExistsError('Preserve the prior native-gradient experiment')
    out.mkdir(parents=True)
    receipt={'status':'NATIVE_TRACKING_GRADIENT_CHECK_STARTING','steps_budget':3,'seed':42,
        'native_loss':'official GAMQwen3DLM MDM0.5 + causal0.5',
        'auxiliary_loss':'same-ID cross-frame ROI contrastive, weight0.1',
        'trainable_policy':'native multimodal projector only; language and native vision tower frozen',
        'official_stock_full_language_recipe_claim':False,'formal_adaptation_complete':False}
    def save():
        (out/'owner_state.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    save()
    try:
        # Keep the exact native FlexAttention operator/mask, but avoid the
        # observed CUDA-graph replay lifetime error with checkpointed backward.
        receipt['native_FlexAttention_runtime']='official GAM_DLM_FLEX_CUDAGRAPHS=0; same compiled operator and masks'
        receipt['gradient_probe_runtime']='task-local AOT donated_buffer=False for independently retained gradient probes'
        torch.manual_seed(42)
        checkpoint=ROOT/'models/GroundAnything-DLM'
        raw=AutoConfig.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
        Config=get_class_from_dynamic_module('configuration_groundinganything.GroundAnythingVLMConfig',str(checkpoint),local_files_only=True)
        Model=get_class_from_dynamic_module('modeling_groundinganything.GroundAnythingVLMForConditionalGeneration',str(checkpoint),local_files_only=True)
        config=Config.from_dict(raw.to_dict());config._attn_implementation='eager'
        processor=AutoProcessor.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
        with safe_open(checkpoint/'model.safetensors',framework='pt') as source:
            config.text_config.vocab_size=source.get_slice('base_model.model.language_model.embed_tokens.weight').get_shape()[0]
        assert config.text_config.vocab_size==len(processor.tokenizer)
        mask=processor.tokenizer.encode('|<MASK>|',add_special_tokens=False);assert len(mask)==1
        # Learned parameters stay empty until strict checkpoint assignment;
        # nonpersistent rotary buffers must be initialized from the native
        # model's own formulas, since they are not in the checkpoint state.
        with init_empty_weights(include_buffers=False):
            base=Model(config)
            model=GAMQwen3DLM(base,mask[0],processor.tokenizer.convert_tokens_to_ids('<|im_end|>'),block_size=32)
        model.load_state_dict(load_file(checkpoint/'model.safetensors'),strict=True,assign=True)
        # Checkpoint parameters already carry their released dtypes; keep
        # native nonpersistent rotary buffers in their initialized FP32 dtype.
        model.to('cuda:0').requires_grad_(False)
        projector=model.base_model.model.visual.projector
        projector.float().requires_grad_(True)
        initial={name:value.detach().clone() for name,value in projector.state_dict().items()}
        trainable=sum(p.numel() for p in projector.parameters())
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==trainable
        model.train();model.base_model.model.visual.vision_tower.eval()
        model.gradient_checkpointing_enable({'use_reentrant':False})
        protocol=json.loads((PROTOCOL/'mot17_half.json').read_text())
        sequence=next(s for s in protocol['sequences'] if s['sequence']=='MOT17-02-FRCNN')
        numbers=[209,210];names={int(Path(p).stem):p for p in sequence['train_frames']}
        assert set(numbers)<=set(names)
        gt=annotations(sequence,set(numbers));rows=[];images=[]
        for number in numbers:
            path=Path(sequence['image_dir'])/names[number]
            with Image.open(path) as source:image=source.convert('RGB')
            images.append(image)
            coordinates=[]
            for target in gt[number]:
                # All selected targets are within the representable native
                # image grid. No clipping, sample removal or target filtering.
                assert 0<=target['box'][0]<target['box'][2]<=image.width-1
                assert 0<=target['box'][1]<target['box'][3]<=image.height-1
                coordinates.append(tuple(round(v/(image.width-1 if i%2==0 else image.height-1)*999) for i,v in enumerate(target['box'])))
            assert len(set(coordinates))==len(gt[number])
            response=serialize_answer([GroundingEntry('person',tuple(coordinates))],'bbox')
            rows.append({'id':f'mot17_train_{number}','images':[{'path':str(path),'bytes':path.read_bytes()}],
                'messages':[{'role':'user','content':'<image>\n'+build_dense_bbox_prompt(['person'])},
                            {'role':'assistant','content':response}]})
        collator=DLMDataCollator(processor,max_length=2048,max_joint_length=4096,model_family='groundinganything_qwen3')
        batch=collator(rows);batch={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in batch.items()}
        pooler=NativeGroundFeatures.__new__(NativeGroundFeatures);pooler.processor=processor.image_processor
        common=sorted(set(t['gt_id'] for t in gt[numbers[0]])&set(t['gt_id'] for t in gt[numbers[1]]))
        assert common
        captured=[]
        def capture(module,args,output):captured.append(output)
        hook=projector.register_forward_hook(capture)
        optimizer=torch.optim.AdamW(projector.parameters(),lr=2e-6,weight_decay=.01)
        logs=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for step in range(1,4):
            captured.clear();optimizer.zero_grad()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(**batch)
                assert len(captured)==1
                # Native training projector returns one token tensor per image.
                features=torch.cat(captured[0],dim=0);grids=batch['image_grid_thw'].tolist();offset=0;regions=[]
                for number,image,(t,h,w) in zip(numbers,images,grids):
                    gh,gw=h//2,w//2;count=gh*gw
                    spatial=features[offset:offset+count].reshape(gh,gw,-1).permute(2,0,1).unsqueeze(0);offset+=count
                    resize=processor.image_processor.get_resize_config({'type':'image','image':image})
                    metadata={'image_size':[image.width,image.height],'resize_and_padding':resize}
                    by_id={target['gt_id']:target for target in gt[number]}
                    pooled,local=pooler.pool(spatial,metadata,[by_id[i]['box'] for i in common]);regions.append(F.normalize(pooled.float(),dim=-1))
                assert offset==features.shape[0]
                logits=regions[0]@regions[1].T/.07
                target=torch.arange(len(common),device='cuda:0')
                temporal=.5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))
                loss=output.loss+.1*temporal
            if step==1:
                parameters=list(projector.parameters())
                dense_grad=torch.autograd.grad(output.loss,parameters,retain_graph=True,allow_unused=True)
                temporal_grad=torch.autograd.grad(temporal,parameters,retain_graph=True,allow_unused=True)
                dense_norm=float(sum(g.float().square().sum() for g in dense_grad if g is not None).sqrt())
                temporal_norm=float(sum(g.float().square().sum() for g in temporal_grad if g is not None).sqrt())
                assert dense_norm>0 and temporal_norm>0
                receipt['independent_native_dense_gradient_norm']=dense_norm
                receipt['independent_temporal_identity_gradient_norm']=temporal_norm
                del dense_grad,temporal_grad
            loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(projector.parameters(),1.))
            assert torch.isfinite(loss) and norm>0
            assert all(p.grad is None for name,p in model.named_parameters() if not p.requires_grad)
            optimizer.step()
            record={'step':step,'native_loss':float(output.loss),'mdm_loss':float(output.mdm_loss),
                'causal_loss':float(output.causal_loss),'temporal_identity_loss':float(temporal),
                'total_loss':float(loss),'projector_gradient_norm':norm,'elapsed_seconds':time.perf_counter()-start}
            logs.append(record);print(json.dumps(record),flush=True)
        hook.remove()
        changed=sum(int((value!=initial[name]).sum()) for name,value in projector.state_dict().items());assert changed>0
        bf16_changed=sum(int((value.to(torch.bfloat16)!=initial[name].to(torch.bfloat16)).sum()) for name,value in projector.state_dict().items())
        state={'native_projector_state_dict':projector.state_dict(),'optimizer':optimizer.state_dict(),
            'initial_DLM_checkpoint':str(checkpoint),'steps':3,'seed':42,'trainable_policy':'native_projector_only'}
        torch.save(state,out/'projector_delta.pt')
        reloaded=torch.load(out/'projector_delta.pt',map_location='cuda:0',weights_only=True)
        difference=max(float((value-reloaded['native_projector_state_dict'][name]).abs().max()) for name,value in projector.state_dict().items());assert difference==0
        digest=hashlib.sha256((out/'projector_delta.pt').read_bytes()).hexdigest()
        receipt.update(status='REAL_NATIVE_DLM_DENSE_AND_IDENTITY_GRADIENTS_VERIFIED',steps=3,
            trainable_native_projector_parameters=trainable,changed_projector_elements=changed,
            BF16_changed_projector_elements=bf16_changed,
            source_frames=[str(Path(sequence['image_dir'])/names[n]) for n in numbers],
            dense_GT_targets=[len(gt[n]) for n in numbers],shared_identity_positives=len(common),
            original_native_weight_tensors=569,tokenizer_and_checkpoint_rows=config.text_config.vocab_size,
            language_and_ViT_gradients_present=False,training_log=logs,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_reload_max_difference=difference,
            checkpoint={'path':str((out/'projector_delta.pt').resolve()),'bytes':(out/'projector_delta.pt').stat().st_size,'sha256':digest},
            complete_video_tracking_gain_claim=False,dense_and_referring_retention='pending exported delta/bundle protocol runs')
        (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');save()
    except Exception as error:
        receipt.update(status='NATIVE_TRACKING_GRADIENT_CHECK_FAILED',error_type=type(error).__name__,error=str(error));save();raise


if __name__=='__main__':main()
