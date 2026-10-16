"""Global physical allocation + frozen task semantics + assigned-ID callback."""
from pathlib import Path
import torch

from locatemot.groundjev.assigned_identity_update import PROTOCOL,build_assigned_update_request
from locatemot.groundjev.tracker import OnlineTracker


class GlobalIdentityDecision:
    def __init__(self,checkpoint,memory_policy='original'):
        from locatemot.groundjev.compact import CompactCore,CompactDecision
        from locatemot.groundjev.global_identity_core import GlobalIdentityCore
        data=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if memory_policy=='assigned':
            assert data.get('assignment_conditioned_memory_trained',False),'A3 requires trained assigned-ID write weights'
        initial=torch.load(data['initial_checkpoint'],map_location='cpu',weights_only=True)
        base=CompactCore(initial['architecture'],semantic=initial['semantic']);base.load_state_dict(initial['state_dict'])
        self.model=GlobalIdentityCore(base,data['competition']);self.model.load_state_dict(data['state_dict']);self.model.eval()
        self.model.semantic=True;self.checkpoint=str(Path(checkpoint).resolve());self.plan=data['plan']
        self.semantic=CompactDecision('/data3/user/LocateMOT-checkpoints/compact_canonical_task_readout_initial_v1.pt',device='cpu')
        self.text_features=self.semantic.text_features;self.memory_policy=memory_policy;self.cache=None

    def decide(self,image,candidates,tracks,timestamp,specification):
        from locatemot.groundjev.global_identity_core import physical_inputs
        n,m=len(candidates),len(tracks)
        e=physical_inputs(image.size,candidates,tracks,timestamp,self.plan['competition_neighbors_each_metric'])
        retrieval=[]
        if m:
            current=torch.nn.functional.normalize(e['candidates'],dim=-1)
            history=torch.nn.functional.normalize(e['history'],dim=-1)
            cosine=(current@history.reshape(-1,2560).T).reshape(n,m,3).mean(-1)
            for i in range(n):
                motion=sorted(range(m),key=lambda j:(-float(e['pair_geometry'][i,j,0]),tracks[j].track_id))
                appearance=sorted(range(m),key=lambda j:(-float(cosine[i,j]),tracks[j].track_id))
                chosen=motion[:16]
                chosen.extend(j for j in appearance if j not in chosen)
                retrieval.append([tracks[j].track_id for j in chosen[:32]])
        else:retrieval=[[] for _ in range(n)]
        with torch.no_grad():out=self.model(e)
        semantic,_,semantic_report=self.semantic.decide(image,candidates,tracks,timestamp,specification)
        p=out['logits'].softmax(-1);names=[str(t.track_id) for t in tracks]+['NEW','IGNORE']
        suppressed={};components=[]
        if self.model.competition and n:
            parent=list(range(n))
            def find(i):
                while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
                return i
            # Only spatially overlapping pairs can form a duplicate component.
            # Both edge directions are observed; either positive is explicit.
            for (i,j),q,g in zip(e['edges'].tolist(),out['edge_logits'][:,1].sigmoid().tolist(),e['edge_geometry'].tolist()):
                if q>=.5 and g[0]>0:
                    a,b=find(i),find(j)
                    if a!=b:parent[b]=a
            groups={}
            for i in range(n):groups.setdefault(find(i),[]).append(i)
            for group in groups.values():
                if len(group)<2:continue
                winner=max(group,key=lambda i:(float(out['quality'][i]),tuple(-float(x) for x in candidates[i]['box']),-i))
                components.append(dict(candidates=group,selected=winner))
                for i in group:
                    if i!=winner:suppressed[i]=winner
        results=[]
        for i in range(n):
            raw=dict(zip(names,p[i].tolist()));probabilities=raw
            if i in suppressed:probabilities=dict(NEW=1e-12,IGNORE=1.-1e-12)
            decision=dict(probabilities=probabilities,update_probability=semantic[i]['update_probability'],
                reliability=semantic[i]['reliability'],reliability_probabilities=semantic[i]['reliability_probabilities'],
                shortlisted_ids=[t.track_id for t in tracks],retrieval_pool_size=m,
                initial_retrieved_top32_ids=retrieval[i],expanded_to_all_live_before_NEW=True,
                admission_action='DUPLICATE_COMPETITOR' if i in suppressed else 'ALLOCATE',
                pre_competition_probabilities=raw)
            if i in suppressed:decision['duplicate_representative_candidate']=suppressed[i]
            for key in ('query_membership_probability','category_decision'):
                if key in semantic[i]:decision[key]=semantic[i][key]
            results.append(decision)
        self.cache=dict(out=out,inputs=e,decisions=results,tracks=tracks,
            state=dict(historical_tracks=[dict(id=t.track_id) for t in tracks],
                candidates=[dict(index=i,box=c['box']) for i,c in enumerate(candidates)]))
        return results,{},dict(backend='global_identity_compact',native_probabilities=True,
            probability_source='learned Compact physical ID/NEW/REJECT logits; not native OneJev',
            GT_read=False,explicit_no_match=False,checkpoint=self.checkpoint,physical_query_input=False,
            retrieval_policy='All available ACTIVE/LOST identities scored in16-row blocks; no TopK omission becomes NEW.',
            all_live_identity_count=m,competition_components=components,duplicate_rejected_candidates=len(suppressed),
            semantic_checkpoint=self.semantic.checkpoint,semantic_readout=semantic_report,
            memory_update_policy=self.memory_policy)

    def decide_assigned_updates(self,selected_existing_ids):
        cached=self.cache
        request=build_assigned_update_request(cached['state'],selected_existing_ids) if selected_existing_ids else None
        tracks=cached['tracks'];by_id={t.track_id:i for i,t in enumerate(tracks)};updates={};flips=0
        if selected_existing_ids:
            rows=torch.tensor(list(selected_existing_ids));columns=torch.tensor([by_id[t] for t in selected_existing_ids.values()])
            confidence=[];margin=[];extra=[]
            for i,tid in selected_existing_ids.items():
                column=by_id[tid];scores=cached['out']['logits'][i];rivals=torch.cat([scores[:column],scores[column+1:]])
                confidence.append(float(scores.softmax(0)[column]));margin.append(float(scores[column]-rivals.max()))
                t=tracks[column];extra.append([cached['decisions'][i]['update_probability'],t.age/100,t.lost_age/3,
                    float(torch.nn.functional.cosine_similarity(cached['inputs']['candidates'][i],cached['inputs']['history'][column,2],dim=0))])
            with torch.no_grad():assigned=self.model.assigned_write(cached['out'],rows,columns,
                torch.tensor(confidence),torch.tensor(margin),torch.tensor(extra)).sigmoid().tolist()
            for index,(i,tid) in enumerate(selected_existing_ids.items()):
                original=cached['decisions'][i]['update_probability']
                if self.memory_policy=='always':value=1.
                elif self.memory_policy=='confidence':value=float(confidence[index]>=.7)
                elif self.memory_policy=='assigned':value=assigned[index]
                else:value=original
                updates[i]=value;flips+=((value>=.5)!=(original>=.5))
        return updates,dict(protocol=PROTOCOL,phase=request.state['final_assignment']['phase'] if request else 'after_global_matching_before_any_current_frame_memory_write',
            existing_assignments=len(updates),update_HOLD_flips_vs_original=flips,policy=self.memory_policy,
            probability_source='Compact assigned-identity sigmoid' if self.memory_policy=='assigned' else self.memory_policy,
            native_second_pass_calls=0,GT_read=False,NEW_initialization_policy='unchanged original causal initialization')


class GlobalIdentityTracker(OnlineTracker):
    """Reuse the original identity memory and post-assignment update order."""
    def __init__(self,*args,**kwargs):
        kwargs['lifecycle_termination_policy']='timeout_only';super().__init__(*args,**kwargs)

    def step(self,image,candidates,timestamp,specification):
        outputs,state=super().step(image,candidates,timestamp,specification)
        for event in state['events']:
            if event['action']=='ACTIVE':event['action']='MATCH'
            elif event['action']=='TERMINATE':event['action']='RETIRED';event['retirement_reason']='registered_fixed_timeout'
            if event.get('action')=='NEW':event['confirmation_state']='TENTATIVE'
        for track in self.tracks.values():
            if track.lost_age==0:track.state='TENTATIVE' if track.age<2 else 'ACTIVE'
        state['decision_metadata'].update(lifecycle='TENTATIVE/ACTIVE/LOST/RETIRED; fixed timeout only',
            tentative_confirmation_hits=2,tentative_output='Emit current provisional box immediately; no delayed or backfilled output.',
            learned_permanent_exit_supervision=False)
        return outputs,state
