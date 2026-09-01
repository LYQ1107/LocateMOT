"""Frame-set token correspondence model for the isolated L25 experiments."""
from __future__ import annotations
import torch
from torch import nn


class L25TokenCorrespondence(nn.Module):
    """Query-token to ROI/context token matcher with optional set decoding.

    A forward call represents all candidates of one frame.  No source, pool,
    group or tracker identifiers are accepted by the model.
    """
    def __init__(self, stage='D0', hidden=96):
        super().__init__()
        if stage not in {'D0','D1','D2','D3','D4','F1','F2','F3','F4','F5','F6'}: raise ValueError(stage)
        self.stage=stage;self.hidden=hidden
        self.q=nn.Linear(512,hidden,bias=False);self.k=nn.Linear(512,hidden,bias=False);self.v=nn.Linear(512,hidden,bias=False);self.coord=nn.Linear(2,hidden,bias=False)
        self.cross=nn.MultiheadAttention(hidden,4,batch_first=True)
        self.context_cross=nn.MultiheadAttention(hidden,4,batch_first=True)
        self.history_cross=nn.MultiheadAttention(hidden,4,batch_first=True)
        self.cond=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden*2),nn.GELU(),nn.Linear(hidden*2,hidden*2))
        self.set_decoder=nn.MultiheadAttention(hidden,4,batch_first=True)
        self.motion=nn.Linear(16,16,bias=False)
        self.head=nn.Sequential(nn.LayerNorm(hidden+16),nn.Linear(hidden+16,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,q_tokens,roi_tokens,roi_coords,context_tokens=None,prev_tokens=None,motion=None,teacher_score=None):
        # Inputs are [N,L,D], [N,K,D], and [N,K,2].
        q=self.q(torch.nan_to_num(q_tokens.float()));roi=self.k(torch.nan_to_num(roi_tokens.float()));val=self.v(torch.nan_to_num(roi_tokens.float()));roi=roi+self.coord(roi_coords.float())
        if self.stage == 'F6' and self.training:
            # Attribute/token masking is training-only; evaluation always uses
            # the complete word-token sequence.
            keep=(torch.rand(q.shape[:2],device=q.device) > .15).unsqueeze(-1)
            q=q*keep
        if self.stage in {'D3','D4','F4','F5'}:
            gamma,beta=self.cond(q.mean(1)).chunk(2,-1);roi=roi*(1+.1*torch.tanh(gamma[:,None,:]))+.1*beta[:,None,:]
        out,_=self.cross(q,roi,val,need_weights=False)
        fused=out.mean(1)
        if context_tokens is not None and self.stage in {'D1','D2','D3','D4','F3','F4','F5'}:
            c=self.k(torch.nan_to_num(context_tokens.float()));co,_=self.context_cross(q,c,self.v(torch.nan_to_num(context_tokens.float())),need_weights=False);fused=fused+.5*co.mean(1)
        if prev_tokens is not None and self.stage in {'D2','D3','D4','F3','F4','F5'}:
            h=self.k(torch.nan_to_num(prev_tokens.float()));ho,_=self.history_cross(q,h,self.v(torch.nan_to_num(prev_tokens.float())),need_weights=False);fused=fused+.5*ho.mean(1)
        if motion is None:motion=torch.zeros(q.shape[0],16,device=q.device)
        fused=torch.cat((fused,self.motion(torch.nan_to_num(motion.float()))),-1)
        residual=self.head(fused).squeeze(-1)
        if self.stage == 'F2':
            if teacher_score is None: raise ValueError('F2 requires a frozen token teacher')
            scores=teacher_score.detach()+0.1*torch.tanh(residual)
        else:
            scores=residual
        if self.stage in {'D4','F5'} and len(scores)>1:
            set_in=fused[:,:self.hidden].unsqueeze(0);set_out,_=self.set_decoder(set_in,set_in,set_in,need_weights=False);scores=self.head(torch.cat((set_out[0],fused[:,self.hidden:]),-1)).squeeze(-1)
        return scores


__all__=['L25TokenCorrespondence']
