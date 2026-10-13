"""Finite identity-contrastive adaptation of frozen native DLM region features."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from torch import nn
    import torch.nn.functional as F
    from PIL import Image
    from locatemot.groundjev.native_features import NativeGroundFeatures
    from tools.groundjev.prepare_tracking_decisions import annotations,PROTOCOL
    out=ROOT/'outputs/groundjev/ground_adaptation/native_temporal200'
    if out.exists() and any(out.iterdir()): raise RuntimeError('Preserve the existing finite adapter experiment')
    out.mkdir(parents=True)
    torch.manual_seed(0);rng=random.Random(0)
    extractor=NativeGroundFeatures()
    # These are legal TRAIN images. The calibration video is disjoint from
    # all feature-adaptation training videos; no official validation/test GT.
    scopes=[('mot17_half.json','MOT17-02-FRCNN',list(range(209,225)),'train'),
            ('dancetrack.json','dancetrack0001',list(range(129,145)),'train'),
            ('mot20_half.json','MOT20-01',list(range(185,201)),'train'),
            ('dancetrack.json','dancetrack0006',list(range(1,17)),'calibration')]
    train_features,train_labels,calibration,summary=[],[],[],[]
    identity_ids={}
    for filename,name,frames,split in scopes:
        protocol=json.loads((PROTOCOL/filename).read_text())
        sequence=next(s for s in protocol['sequences'] if s['sequence']==name)
        names={int(Path(f).stem):f for f in sequence['train_frames']}
        if not set(frames)<=set(names): raise RuntimeError('Native adaptation entered a non-training frame')
        gt=annotations(sequence,set(frames))
        observations=0
        for number in frames:
            path=Path(sequence['image_dir'])/names[number]
            with Image.open(path) as source: image=source.convert('RGB')
            targets=sorted(gt[number],key=lambda g:g['gt_id'])[:32]
            features,local,metadata=extractor(image,[g['box'] for g in targets])
            features=features.detach().float().cpu()
            del local
            labels=[]
            for target in targets:
                key=(name,target['gt_id'])
                if key not in identity_ids: identity_ids[key]=len(identity_ids)
                labels.append(identity_ids[key])
            observations+=len(labels)
            if split=='train':
                train_features.append(features);train_labels.extend(labels)
            else:
                calibration.append({'frame':number,'features':features,'labels':labels})
            print(json.dumps({'stage':'extract_real_native_regions','sequence':name,'frame':number,
                              'regions':len(labels),'scope':split}),flush=True)
        summary.append({'sequence':name,'split':split,'frames':frames,'region_observations':observations})
    x=torch.cat(train_features).to('cuda:0')
    y=torch.tensor(train_labels,device='cuda:0')
    groups=defaultdict(list)
    for index,label in enumerate(train_labels): groups[label].append(index)
    groups={label:indices for label,indices in groups.items() if len(indices)>=2}
    # This adapter is not called OneJev or a replacement JEV decision model.
    adapter=nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,256),nn.GELU(),nn.Linear(256,128)).to('cuda:0')
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-3,weight_decay=1e-4)
    initial={name:value.detach().clone() for name,value in adapter.state_dict().items()}
    def retrieval(model=None):
        correct,total,missing=0,0,0
        with torch.no_grad():
            for previous,current in zip(calibration,calibration[1:]):
                a,b=previous['features'].to('cuda:0'),current['features'].to('cuda:0')
                if model is not None: a,b=model(a),model(b)
                similarity=F.normalize(b,dim=-1)@F.normalize(a,dim=-1).T
                chosen=similarity.argmax(dim=1).tolist()
                for i,label in enumerate(current['labels']):
                    if label not in previous['labels']:
                        missing+=1;continue
                    total+=1;correct+=previous['labels'][chosen[i]]==label
        return {'correct':correct,'eligible_events':total,'correct_option_missing':missing,
                'accuracy':correct/total if total else None,
                'scope':'GT-region adjacent-frame retrieval on disjoint train calibration video, not HOTA'}
    baseline=retrieval()
    adapter.eval();before=retrieval(adapter);adapter.train()
    log=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(1,201):
        sampled=rng.sample(list(groups),min(64,len(groups)))
        indices=[index for label in sampled for index in rng.sample(groups[label],2)]
        batch=x[indices];labels=y[indices]
        z=F.normalize(adapter(batch),dim=-1)
        logits=z@z.T/.07
        diagonal=torch.eye(len(indices),device=z.device,dtype=torch.bool)
        same=(labels[:,None]==labels[None,:])&~diagonal
        targets=same.float()/same.sum(dim=1,keepdim=True)
        logp=F.log_softmax(logits.masked_fill(diagonal,-1e4),dim=-1)
        loss=-(targets*logp).sum(dim=1).mean()
        optimizer.zero_grad();loss.backward()
        gradient=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.))
        if not torch.isfinite(loss) or not (gradient>0): raise RuntimeError('Native temporal adaptation has invalid loss/gradient')
        optimizer.step()
        if step==1 or step%20==0:
            record={'step':step,'loss':float(loss),'gradient_norm':gradient,'elapsed_seconds':time.perf_counter()-start}
            log.append(record);print(json.dumps(record),flush=True)
    adapter.eval();after=retrieval(adapter)
    changed=sum(int((initial[name]!=value).sum()) for name,value in adapter.state_dict().items())
    if not changed: raise RuntimeError('Temporal adapter weights did not change')
    checkpoint={'adapter_state_dict':adapter.state_dict(),'architecture':'LayerNorm2560-Linear256-GELU-Linear128',
                'optimizer':optimizer.state_dict(),'steps':200,'seed':0,'native_checkpoint':str(extractor.checkpoint)}
    torch.save(checkpoint,out/'checkpoint.pt')
    checkpoint_digest=hashlib.sha256()
    with (out/'checkpoint.pt').open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):checkpoint_digest.update(chunk)
    fresh=nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,256),nn.GELU(),nn.Linear(256,128)).to('cuda:0')
    fresh.load_state_dict(torch.load(out/'checkpoint.pt',weights_only=True)['adapter_state_dict']);fresh.eval()
    with torch.no_grad(): reload_difference=float((adapter(x[:8])-fresh(x[:8])).abs().max())
    if reload_difference!=0: raise RuntimeError('Native adapter checkpoint reload differs')
    report={'status':'REAL_NATIVE_GROUND_TEMPORAL_ADAPTER_TRAINING_COMPLETE','steps':200,'seed':0,
        'native_DLM_visual_parameters':extractor.parameter_count,'native_backbone_frozen':True,
        'native_backbone_gradients_present':any(p.grad is not None for p in extractor.visual.parameters()),
        'trainable_adapter_parameters':sum(p.numel() for p in adapter.parameters()),'changed_weight_elements':changed,
        'checkpoint_reload_max_abs_difference':reload_difference,'scopes':summary,
        'native_feature_cache_bytes':int(x.numel()*x.element_size()),'full_dataset_cache':False,
        'native_unadapted_calibration_retrieval':baseline,'random_adapter_before_training':before,
        'trained_adapter_calibration_retrieval':after,'training_log':log,
        'peak_adapter_training_allocated_bytes':torch.cuda.max_memory_allocated(),
        'checkpoint':{'path':str((out/'checkpoint.pt').resolve()),'bytes':(out/'checkpoint.pt').stat().st_size,
                      'sha256':checkpoint_digest.hexdigest()},
        'feature_cache_accounting':{'reported_training_device_region_bytes':int(x.numel()*x.element_size()),
            'CPU_training_region_bytes':sum(int(t.numel()*t.element_size()) for t in train_features),
            'CPU_calibration_region_bytes':sum(int(row['features'].numel()*row['features'].element_size()) for row in calibration),
            'persistent_feature_files':0,'additional_encoder_ROI_workspace':'transient, released each image; not a full-dataset cache'},
        'Ground_DLM_decoder_weights_changed':False,'dense_referring_retention_evaluation':'pending actual protocol runs; frozen decoder alone is not reported as a benchmark',
        'complete_video_tracking_improvement_claim':False,'actual_DLM_candidate_rollout_training':False}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
