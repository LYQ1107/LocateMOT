"""Trainable physical identity allocation with sparse frame-wide competition.

This is a Compact model initialized from Shared Compact, not native OneJev.
Hungarian is used only by the runtime; gradients train logits and ranking losses.
"""
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F


class GlobalIdentityCore(nn.Module):
    def __init__(self,base,competition=True):
        super().__init__();self.competition=competition;w=base.width;self.width=w
        for name in ('visual','history_merge','candidate_geo','history_geo','body','pair','actions','update'):
            setattr(self,name,deepcopy(getattr(base,name)))
        self.edge=nn.Sequential(nn.Linear(3*w+6,64),nn.GELU(),nn.Linear(64,2))
        self.message=nn.Linear(w,w,bias=False)
        self.quality=nn.Sequential(nn.Linear(w,64),nn.GELU(),nn.Linear(64,1))
        self.relation=nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,1))
        self.appearance_scale=nn.Parameter(torch.tensor(2.))
        self.write=nn.Sequential(nn.Linear(4*w+6,64),nn.GELU(),nn.Linear(64,1))
        self.register_buffer('physical_specification',base.task.weight[0].detach().clone())

    def encode_blocks(self,x,size):
        if not len(x):return x
        return torch.cat([self.body((x[i:i+size]+self.physical_specification).unsqueeze(0)).squeeze(0)
            for i in range(0,len(x),size)],0)

    def forward(self,e):
        n,m=len(e['candidates']),len(e['history']);w=self.width
        c=self.visual(e['candidates'])+self.candidate_geo(e['candidate_geo'])
        h=self.history_merge(self.visual(e['history']).reshape(m,3*w))+self.history_geo(e['history_geo'])
        edges=e['edges'];edge_logits=c.new_empty((0,2))
        if len(edges):
            a,b=c[edges[:,0]],c[edges[:,1]]
            edge_logits=self.edge(torch.cat([a,b,a*b,e['edge_geometry']],-1))
            if self.competition:
                weights=edge_logits[:,0].sigmoid();message=self.message(b)*weights[:,None]
                aggregate=torch.zeros_like(c).index_add(0,edges[:,0],message)
                denominator=c.new_zeros((n,)).index_add(0,edges[:,0],weights)
                c=c+aggregate/denominator.clamp_min(1)[:,None]
        c=self.encode_blocks(c,16);h=self.encode_blocks(h,32)
        logits=[]
        history_norm=F.normalize(e['history'],dim=-1)
        current_norm=F.normalize(e['candidates'],dim=-1)
        for start in range(0,n,16):
            current=c[start:start+16];count=len(current)
            a=current[:,None].expand(count,m,w);b=h[None].expand(count,m,w)
            geometry=e['pair_geometry'][start:start+count]
            cosine=torch.einsum('nd,mkd->nmk',current_norm[start:start+count],history_norm)
            pair=self.pair(torch.cat([a,b,a*b,a-b,geometry],-1)).squeeze(-1)
            pair=pair+self.relation(torch.cat([cosine,geometry],-1)).squeeze(-1)+self.appearance_scale*cosine.mean(-1)
            context=torch.cat([current,h],0).mean(0,keepdim=True)
            actions=self.actions(torch.cat([current,context.expand(count,-1)],-1))[:,:2]
            logits.append(torch.cat([pair,actions],-1))
        scores=torch.cat(logits,0) if logits else c.new_empty((0,m+2))
        return dict(logits=scores,candidate_hidden=c,history_hidden=h,edge_logits=edge_logits,competition_enabled=self.competition,
            quality=self.quality(c).squeeze(-1),update=self.update(c).squeeze(-1))

    def assigned_write(self,out,rows,columns,confidence,margin,extra):
        a=out['candidate_hidden'][rows];b=out['history_hidden'][columns]
        evidence=torch.cat([confidence[:,None],margin[:,None],extra],-1)
        return self.write(torch.cat([a,b,a*b,a-b,evidence],-1)).squeeze(-1)


def physical_inputs(image_size,candidates,tracks,timestamp,neighbors=32):
    import numpy as np
    from locatemot.groundjev.global_competition import box_iou,sparse_competition
    w,h=image_size;n,m=len(candidates),len(tracks)
    boxes=np.asarray([c['box'] for c in candidates],np.float32).reshape(n,4)
    predicted=np.asarray([t.predicted_box(timestamp) for t in tracks],np.float32).reshape(m,4)
    current=np.asarray([c['native_region_feature'] for c in candidates],np.float32).reshape(n,2560)
    history=np.asarray([[t.anchor_feature,t.recent_feature,t.long_feature] for t in tracks],np.float32).reshape(m,3,2560)
    edges,edge_geometry=sparse_competition(boxes,current,neighbors)
    center=(boxes[:,:2]+boxes[:,2:])/2;past=(predicted[:,:2]+predicted[:,2:])/2
    distance=np.linalg.norm(center[:,None]-past[None],axis=-1)/max(float(np.hypot(w,h)),1.)
    return dict(candidates=torch.from_numpy(current),history=torch.from_numpy(history),
        candidate_geo=torch.from_numpy(boxes/np.asarray([w,h,w,h],np.float32)),
        history_geo=torch.tensor([[*(x/s for x,s in zip(t.predicted_box(timestamp),(w,h,w,h))),
            t.age/100,t.lost_age/3,(timestamp-t.last_reliable_timestamp)/3,t.confidence or 0.] for t in tracks]).reshape(m,8),
        pair_geometry=torch.from_numpy(np.stack([box_iou(boxes,predicted),distance],-1)),
        edges=torch.from_numpy(edges),edge_geometry=torch.from_numpy(edge_geometry))


def physical_loss(out,e,labels):
    """Masked action CE + pairwise duplicate BCE + GT-quality competition rank."""
    zero=out['update'].sum()*0+out['logits'].sum()*0;identity=[];ranking=[]
    n,m=out['logits'].shape[0],out['logits'].shape[1]-2
    for i,options in enumerate(labels['action_options']):
        if options:
            identity.append(torch.logsumexp(out['logits'][i],0)-torch.logsumexp(out['logits'][i,options],0))
    duplicate=zero
    if out['competition_enabled'] and len(e['edges']):
        a,b=e['edges'].T;gt=labels['unique_GT'];known=(gt[a]>=0)&(gt[b]>=0)
        if known.any():
            target=(gt[a[known]]==gt[b[known]]).float()
            duplicate=F.binary_cross_entropy_with_logits(out['edge_logits'][known,1],target)
            same=known&(gt[a]==gt[b])&(labels['quality'][a]>labels['quality'][b]+1e-6)
            if same.any():ranking.append(F.softplus(-(out['quality'][a[same]]-out['quality'][b[same]])).mean())
    ce=torch.stack(identity).mean() if identity else zero
    rank=torch.stack(ranking).mean() if ranking else zero
    return ce+.5*duplicate+.2*rank,dict(identity=float(ce.detach()),duplicate=float(duplicate.detach()),
        ranking=float(rank.detach()),supervised_actions=len(identity),masked_actions=n-len(identity))
