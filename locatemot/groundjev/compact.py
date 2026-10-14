"""Trainable structured Compact and controls; not the pretrained OneJev model."""
import torch
from torch import nn


class FeedForwardBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,6*width), nn.GELU(), nn.Linear(6*width,width))
    def forward(self, x): return x+self.net(x)


class CompactCore(nn.Module):
    def __init__(self, architecture='set', width=128, semantic=False):
        super().__init__()
        self.architecture, self.width = architecture, width
        self.visual = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560,width), nn.GELU())
        self.history_merge = nn.Linear(3*width,width)
        self.candidate_geo = nn.Linear(4,width); self.history_geo = nn.Linear(8,width)
        self.task = nn.Embedding(3,width)
        if architecture=='set':
            self.body = nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,4*width,dropout=0.,
                activation='gelu',batch_first=True,norm_first=True),2,enable_nested_tensor=False)
        else: self.body = nn.Sequential(FeedForwardBlock(width),FeedForwardBlock(width))
        self.pair = nn.Sequential(nn.Linear(4*width+2,width),nn.GELU(),nn.Linear(width,1))
        self.actions = nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,3))
        self.update = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.reliability = nn.Linear(width,4)
        self.life = nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,4))
        self.semantic=semantic
        if semantic:
            self.text = nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,width),nn.GELU(),nn.Linear(width,width))
            self.category = nn.Linear(width,width)
            self.membership = nn.Sequential(nn.Linear(3*width,width),nn.GELU(),nn.Linear(width,1))
            self.class_scale = nn.Parameter(torch.tensor(2.3025851))

    def forward(self, e):
        n,m = len(e['candidates']),len(e['history'])
        c = self.visual(e['candidates'])+self.candidate_geo(e['candidate_geo'])
        h = self.history_merge(self.visual(e['history']).flatten(1))+self.history_geo(e['history_geo'])
        specification = self.task.weight[e['task_index']]
        if self.semantic:specification=specification+self.text(e['spec_text'])
        tokens = self.body((torch.cat([c,h])+specification).unsqueeze(0)).squeeze(0)
        c,h = tokens[:n],tokens[n:]
        context = tokens.mean(0,keepdim=True)
        a = c[:,None].expand(n,m,self.width); b = h[None].expand(n,m,self.width)
        pair = self.pair(torch.cat([a,b,a*b,a-b,e['pair_geometry']],dim=-1)).squeeze(-1)
        actions = self.actions(torch.cat([c,context.expand(n,-1)],dim=-1))
        if not m:
            actions = actions.clone(); actions[:,-1] = -1e4
        result={'match':torch.cat([pair,actions],dim=-1),'update':self.update(c).squeeze(-1),
            'reliability':self.reliability(c),'life':self.life(torch.cat([h,context.expand(m,-1)],dim=-1))}
        if self.semantic:
            import torch.nn.functional as F
            result['category']=F.normalize(self.category(c),dim=-1)@F.normalize(self.text(e['class_text']),dim=-1).T*self.class_scale.exp()
            s=specification.expand(n,-1)
            result['membership']=self.membership(torch.cat([c,s,c*s],dim=-1)).squeeze(-1)
        return result


class CompactDecision:
    def __init__(self, checkpoint, device='cuda:0', batch_candidates=16, shortlist=32, memory_feature_view='all'):
        from pathlib import Path
        data = torch.load(checkpoint,map_location=device,weights_only=True)
        self.model = CompactCore(data['architecture'],semantic=data.get('semantic',False)).to(device)
        self.model.load_state_dict(data['state_dict']); self.model.eval()
        self.checkpoint = str(Path(checkpoint).resolve()); self.device = device
        self.batch_candidates,self.shortlist = batch_candidates,shortlist
        self.memory_feature_view = memory_feature_view
        if self.model.semantic:
            from locatemot.groundjev.text_features import OneJevTextFeatures
            self.text_features=OneJevTextFeatures();self.text_cache={};self.class_cache={}

    def decide(self,image,candidates,tracks,timestamp,specification):
        from locatemot.groundjev.decision import overlap
        results,life = [],{}
        groups = [candidates[i:i+self.batch_candidates] for i in range(0,len(candidates),self.batch_candidates)] or [[]]
        for group in groups:
            ranked = sorted(tracks,key=lambda t:(-max((overlap(c['box'],t.predicted_box(timestamp)) for c in group),default=0.),t.lost_age,t.track_id))[:self.shortlist]
            n,m = len(group),len(ranked); w,height = image.size
            if not n and not m: continue
            history = [[t.anchor_feature,t.recent_feature,t.long_feature] for t in ranked]
            if self.memory_feature_view=='recent':history=[[t.recent_feature]*3 for t in ranked]
            elif self.memory_feature_view=='ema':history=[[t.long_feature]*3 for t in ranked]
            e = {'candidates':torch.tensor([c['native_region_feature'] for c in group]).reshape(n,2560),
                'history':torch.tensor(history).reshape(m,3,2560),
                'candidate_geo':torch.tensor([[c['box'][0]/w,c['box'][1]/height,c['box'][2]/w,c['box'][3]/height] for c in group]).reshape(n,4),
                'history_geo':torch.tensor([[*(x/s for x,s in zip(t.predicted_box(timestamp),(w,height,w,height))),
                    t.age/100,t.lost_age/3,(timestamp-t.last_reliable_timestamp)/3,t.confidence or 0.] for t in ranked]).reshape(m,8),
                'pair_geometry':torch.tensor([[[overlap(c['box'],t.predicted_box(timestamp)),
                    ((c['box'][0]+c['box'][2]-t.predicted_box(timestamp)[0]-t.predicted_box(timestamp)[2])**2+
                     (c['box'][1]+c['box'][3]-t.predicted_box(timestamp)[1]-t.predicted_box(timestamp)[3])**2)**.5/(2*(w*w+height*height)**.5)]
                    for t in ranked] for c in group]).reshape(n,m,2),
                'task_index':('mot','rmot','ovmot').index(specification['task'])}
            if self.model.semantic:
                key=(specification['task'],specification.get('expression'),tuple(specification['vocabulary']))
                if key not in self.text_cache:self.text_cache[key]=self.text_features.specification(specification).to(self.device)
                vocabulary=tuple(specification['vocabulary'])
                if vocabulary not in self.class_cache:self.class_cache[vocabulary]=self.text_features.vocabulary(vocabulary).to(self.device)
                e.update(spec_text=self.text_cache[key],class_text=self.class_cache[vocabulary])
            e = {k:v.to(self.device) if torch.is_tensor(v) else v for k,v in e.items()}
            with torch.no_grad(): output = self.model(e)
            p = output['match'].softmax(-1).cpu();update = output['update'].sigmoid().cpu(); reliability = output['reliability'].softmax(-1).cpu()
            names = [str(t.track_id) for t in ranked]+['NEW','IGNORE','NO_MATCH']
            for i in range(n):results.append({'probabilities':dict(zip(names,p[i].tolist())),
                'update_probability':float(update[i]),'reliability':float(reliability[i]@torch.arange(4.)),
                'reliability_probabilities':dict(zip(map(str,range(4)),reliability[i].tolist())),
                'shortlisted_ids':[t.track_id for t in ranked]})
            if self.model.semantic:
                membership=output['membership'].sigmoid().cpu();classes=output['category'].softmax(-1).cpu()
                for i in range(n):
                    if specification['task']=='rmot':results[-n+i]['query_membership_probability']=float(membership[i])
                    if specification['task']=='ovmot':
                        j=int(classes[i].argmax());results[-n+i]['category_decision']={
                            'label':specification['vocabulary'][j],'probability':float(classes[i,j]),
                            'probability_kind':'learned full supplied-vocabulary softmax; not detection confidence'}
            for t,probabilities in zip(ranked,output['life'].softmax(-1).cpu().tolist()):
                names_life=['ACTIVE','LOST','REACTIVATE','TERMINATE'];life.setdefault(t.track_id,
                    {'type':'choice','choice':names_life[max(range(4),key=lambda i:probabilities[i])],
                     'probabilities':dict(zip(names_life,probabilities))})
        metadata={'backend':'trained_compact_'+self.model.architecture,'native_probabilities':True,
            'probability_source':'learned structured Compact distributions, not native OneJev decoder',
            'explicit_no_match':True,'checkpoint':self.checkpoint,'shortlist_limit':self.shortlist,
            'MOT_only_initial_training':not self.model.semantic,'GT_read':False}
        if self.model.semantic:metadata.update(shared_semantic_heads=True,
            query_output_policy='RMOT learned membership>=.5; physical identity state preserved before output filtering',
            class_policy='OVMOT full supplied-vocabulary learned softmax; no GT-based class or box filtering')
        if self.memory_feature_view!='all':metadata['memory_feature_view']=self.memory_feature_view
        return results,life,metadata
