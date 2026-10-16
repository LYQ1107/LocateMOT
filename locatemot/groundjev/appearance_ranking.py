"""Appearance-informed identity ranks with original learned probability values."""
import math
import numpy as np


class AppearanceRankDecision:
    def __init__(self,learned,projection_checkpoint=None,temperature=.07):
        self.learned=learned;self.model=learned.model;self.text_features=learned.text_features
        self.temperature=temperature;assert temperature>0
        self.projector=None
        if projection_checkpoint:
            from locatemot.groundjev.temporal_appearance import AppearanceProjector
            self.projector=AppearanceProjector(projection_checkpoint)

    def memories(self,track):
        view=self.learned.memory_feature_view
        if view=='recent':return [track.recent_feature]*3
        if view=='ema':return [track.long_feature]*3
        return [track.anchor_feature,track.recent_feature,track.long_feature]

    def decide(self,image,candidates,tracks,timestamp,specification):
        decisions,life,metadata=self.learned.decide(image,candidates,tracks,timestamp,specification)
        by_id={t.track_id:t for t in tracks}
        used={tid for row in decisions if len(row['shortlisted_ids'])>1 for tid in row['shortlisted_ids']}
        current=memory=None;native_norms={};changed=0
        if self.projector is not None and used:
            current=self.projector([c['native_region_feature'] for c in candidates])
            ordered=[t for t in tracks if t.track_id in used]
            vectors=self.projector([v for t in ordered for v in self.memories(t)]).reshape(len(ordered),3,64)
            memory={t.track_id:v for t,v in zip(ordered,vectors)}
        result=[]
        for index,(candidate,decision) in enumerate(zip(candidates,decisions)):
            ids=decision['shortlisted_ids'];row=dict(decision)
            if len(ids)>1:
                if self.projector is None:
                    a=np.asarray(candidate['native_region_feature'],dtype=float)
                    a_norm=max(float(np.linalg.norm(a)),1e-12)
                scores=[]
                for tid in ids:
                    if self.projector is None:
                        cosines=[]
                        for column,vector in enumerate(self.memories(by_id[tid])):
                            b=np.asarray(vector,dtype=float);key=(tid,column)
                            if key not in native_norms:native_norms[key]=max(float(np.linalg.norm(b)),1e-12)
                            cosines.append(float(a@b/(a_norm*native_norms[key])))
                        cosine=sum(cosines)/3
                    else:cosine=float((memory[tid]@current[index]).mean())
                    scores.append(math.log(max(decision['probabilities'][str(tid)],1e-30))+cosine/self.temperature)
                order=sorted(range(len(ids)),key=lambda j:(-scores[j],j))
                values=sorted((decision['probabilities'][str(tid)] for tid in ids),reverse=True)
                probabilities=dict(decision['probabilities'])
                for rank,j in enumerate(order):probabilities[str(ids[j])]=values[rank]
                changed+=any(probabilities[str(tid)]!=decision['probabilities'][str(tid)] for tid in ids)
                row['probabilities']=probabilities
            result.append(row)
        metadata=dict(metadata,backend='appearance_rank_compact_control',native_probabilities=True,
            association_policy='rank_appearance',auxiliary_policy='learned',
            appearance_ranking_temperature=self.temperature,appearance_rank_changed_rows=changed,
            original_action_probabilities_and_probability_multiset_preserved=True,
            probability_source='Original learned Compact probability values permuted across existing identity options by appearance-informed ranks; not native OneJev decoder or a calibrated contrastive posterior.',
            intervention_scope='Preserves per-row action values and best-existing/action odds on identical inputs; global matching and future causal states may change.')
        if self.projector is not None:
            metadata.update(appearance_projection_checkpoint=self.projector.checkpoint,
                appearance_projection_training_steps=self.projector.training_steps)
        return result,life,metadata
