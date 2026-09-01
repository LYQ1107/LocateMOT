"""Teacher-anchored, antisymmetric current-frame pair residual for L43."""
from __future__ import annotations

import torch
from torch import nn


class L43TeacherAnchoredPairwiseResidual(nn.Module):
    def __init__(self, image_dim=768, text_dim=768, numeric_dim=36,
                 hidden=128, heads=4, layers=1):
        super().__init__()
        if hidden % heads: raise ValueError("hidden must be divisible by heads")
        self.config={"image_dim":image_dim,"text_dim":text_dim,"numeric_dim":numeric_dim,"hidden":hidden,"heads":heads,"layers":layers}
        self.image_proj=nn.Sequential(nn.LayerNorm(image_dim),nn.Linear(image_dim,hidden),nn.GELU())
        self.text_proj=nn.Sequential(nn.LayerNorm(text_dim),nn.Linear(text_dim,hidden),nn.GELU())
        self.cross=nn.ModuleList([nn.MultiheadAttention(hidden,heads,batch_first=True,dropout=0.0) for _ in range(layers)])
        self.cross_norm=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        # Pair features are antisymmetrized explicitly below.  The network is
        # never exposed to source/pool/group/state identifiers.
        pair_dim=2*hidden+2*numeric_dim+1
        self.pair_core=nn.Sequential(nn.Linear(pair_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))

    @staticmethod
    def masked_mean(x, mask):
        w=mask.to(x.dtype).unsqueeze(-1); return (x*w).sum(1)/w.sum(1).clamp_min(1.0)

    def candidate_features(self, patches, text, text_mask=None):
        # patches [N,P,D], text [T,D]
        image=self.image_proj(torch.nan_to_num(patches.float())); q=self.text_proj(torch.nan_to_num(text.float()))
        q=q.unsqueeze(0).expand(image.shape[0],-1,-1); kv=image
        if text_mask is None:
            text_mask=torch.ones(text.shape[0],dtype=torch.bool,device=text.device)
        qm=text_mask.bool().unsqueeze(0).expand(image.shape[0],-1)
        for attn,norm in zip(self.cross,self.cross_norm):
            z,_=attn(q,kv,kv,need_weights=False); q=norm(q+z)
        return self.masked_mean(q,qm)

    def forward(self, patch_tokens, text_tokens, numeric, teacher,
                candidate_mask=None, text_mask=None):
        # patch_tokens [N,P,D], text_tokens [T,D], numeric [N,F], teacher [N]
        if patch_tokens.ndim!=3 or numeric.ndim!=2 or teacher.ndim!=1: raise ValueError("expected unbatched current-frame set")
        n=patch_tokens.shape[0]; valid=torch.ones(n,dtype=torch.bool,device=patch_tokens.device) if candidate_mask is None else candidate_mask.bool()
        if text_tokens.ndim==3:
            text_tokens=text_tokens[0]
        if text_mask is not None and text_mask.ndim==2:
            text_mask=text_mask[0]
        cand=self.candidate_features(patch_tokens,text_tokens,text_mask)
        idx=torch.arange(n,device=patch_tokens.device); ii=idx[:,None].expand(n,n); jj=idx[None,:].expand(n,n); pair_valid=valid[:,None]&valid[None,:]&(ii!=jj)
        dc=cand[:,None,:]-cand[None,:,:]; dn=numeric[:,None,:]-numeric[None,:,:]; td=teacher[:,None]-teacher[None,:]
        base=torch.cat((dc,torch.abs(dc),dn,torch.abs(dn),td.unsqueeze(-1)),dim=-1)
        swapped=torch.cat((-dc,torch.abs(dc),-dn,torch.abs(dn),(-td).unsqueeze(-1)),dim=-1)
        raw=self.pair_core(base).squeeze(-1); raw_swap=self.pair_core(swapped).squeeze(-1)
        residual=0.025*(torch.tanh(raw)-torch.tanh(raw_swap)); residual=residual.masked_fill(~pair_valid,0.0)
        denom=pair_valid.to(residual.dtype).sum(1).clamp_min(1.0); delta=residual.sum(1)/denom; final=teacher+delta; final=final.masked_fill(~valid,-20.0)
        return {"final_score":final,"teacher_score":teacher,"residual":residual,"delta_score":delta,"pair_valid":pair_valid,"candidate_features":cand}
