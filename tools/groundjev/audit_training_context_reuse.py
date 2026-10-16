"""Measure same-update context reuse in the immutable native index-shuffle plan."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    import numpy as np
    from train import common,sft
    from tools.groundjev.group_reference_rows import REFERENCE_KEY
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    measured=json.loads((ROOT/plan['native_length_report']).read_text());manifest=measured['manifest']
    assert measured['rows']==manifest['rows']==459027 and measured['dropped_rows']==0
    assert manifest==json.loads(Path(measured['native_args']['train_file']).read_text())
    assert measured['native_args']['augment'] and measured['native_args']['pad_options']==0
    assert measured['native_args']['seed']==plan['seed']==0 and measured['paired_index_shuffle']
    assert measured['global_rows_per_update']==plan['global_rows_per_update']==16
    assert measured['planned_world_size']==plan['world_size']==4
    lengths=np.load(measured['length_array'],allow_pickle=False,mmap_mode='r')
    assert lengths.dtype==np.int32 and lengths.shape==(459027,)
    assert int(lengths.sum(dtype=np.int64))==measured['planned_tokens']==5639441074
    identifiers=np.empty(len(lengths),dtype=np.uint32);media_identifiers=np.empty_like(identifiers)
    contexts={};media_keys={};counts=Counter();source_records=[];position=0
    for source in manifest['sources']:
        path=ROOT/source['path'];groups=None;group_path=None;selected=0
        with gzip.open(path,'rt') as f:
            for row in map(json.loads,f):
                if source['families'] is not None and row['task'] not in source['families']:continue
                if REFERENCE_KEY in row:
                    ref=row[REFERENCE_KEY]
                    if ref['path']!=group_path:
                        group_path=ref['path']
                        with gzip.open(ROOT/group_path,'rt') as g:
                            groups={r['id']:r for r in map(json.loads,g)}
                    group=groups[ref['group']]
                    row=dict(row,state=group['state'],media=group['media'],question=group['questions'][ref['question']])
                # This is the native augmentation/option rendering, not a reimplemented RNG.
                example=common.render_example(row,lambda state,suffix:state,position,
                    seed=position,augment=True,pad_prob=0.)
                media=json.dumps(row.get('media') or [],ensure_ascii=False,separators=(',',':'))
                payload=json.dumps([example.prompt,media],ensure_ascii=False,separators=(',',':')).encode()
                key=hashlib.blake2b(payload,digest_size=16).digest()
                media_key=hashlib.blake2b(media.encode(),digest_size=16).digest()
                identifiers[position]=contexts.setdefault(key,len(contexts))
                media_identifiers[position]=media_keys.setdefault(media_key,len(media_keys))
                position+=1;selected+=1;counts[source['bucket']]+=1
        assert selected==source['rows']
        source_records.append(dict(path=source['path'],bucket=source['bucket'],rows=selected))
        if len(source_records)%8==0:print(json.dumps(dict(status='NATIVE_CONTEXT_IDENTITY_AUDIT',sources=len(source_records),rows=position)),flush=True)
    assert position==len(lengths) and dict(counts)==manifest['rows_per_task']
    order=list(range(position));random.Random(0).shuffle(order)
    totals=Counter();rank_totals=Counter();tail=None
    def opportunities(indices,keys):
        buckets={}
        for i in indices:buckets.setdefault(int(keys[i]),[]).append(i)
        repeated=[rows for rows in buckets.values() if len(rows)>1]
        return dict(repeated_contexts=len(repeated),additional_rows=sum(len(r)-1 for r in repeated),
            rows_in_repeated_contexts=sum(len(r) for r in repeated),
            optimistic_full_row_token_saving_upper=sum((len(r)-1)*min(int(lengths[i]) for i in r) for r in repeated))
    media_totals=Counter();media_rank_totals=Counter()
    for begin in range(0,len(order),16):
        indices=order[begin:begin+16]
        native=sft.plan_steps([[i] for i in indices],lengths,4,16)
        assert len(native)==1
        per_rank=native[0];tail=[sum(len(b) for b in rank) for rank in per_rank]
        assert sorted(i for rank in per_rank for b in rank for i in b)==sorted(indices)
        totals.update(opportunities(indices,identifiers));media_totals.update(opportunities(indices,media_identifiers))
        totals['steps']+=1
        for rank in per_rank:
            rows=[i for b in rank for i in b]
            rank_totals.update(opportunities(rows,identifiers));media_rank_totals.update(opportunities(rows,media_identifiers))
    assert totals['steps']==measured['planned_updates']==28690 and tail==[1,1,1,0]
    for result in (totals,rank_totals):
        result['full_row_token_saving_upper_fraction']=result['optimistic_full_row_token_saving_upper']/measured['planned_tokens']
    report=dict(status='FULL459027_NATIVE_CONTEXT_REUSE_OPPORTUNITY_AUDIT_COMPLETE',
        plan=str(args.plan),native_length_report=plan['native_length_report'],rows=position,task_rows=dict(counts),
        source_files=source_records,unique_native_serialized_state_and_media_contexts=len(contexts),
        unique_media_recipe_lists=len(media_keys),same_global_update_context_opportunities=dict(totals),
        same_rank_update_context_opportunities=dict(rank_totals),
        same_global_update_media_recipe_cooccurrences=dict(media_totals),
        same_rank_update_media_recipe_cooccurrences=dict(media_rank_totals),
        native_global16_fourrank_dealing_and_tail_exact=True,planned_native_multimodal_tokens=measured['planned_tokens'],
        identity_scope='Native per-row serialized-state augmentation and identical ordered media recipes. Compact digests identify metadata equality; pixels/vision/LM activations are not computed. Does not detect distinct recipes that happen to render identical pixels.',
        upper_bound_scope='For these recognized identical contexts within one optimizer update only: (duplicates-1)*minimum full-row length overestimates saved prefix tokens. Global-update figure optimistically allows cross-rank sharing. Neither is measured runtime speed or a universal bound for all possible pixel equivalences.',
        media_scope='Media-only cooccurrences ignore text-state differences; full-row token quantities attached to these counts are not a bound on vision-compute savings. Frozen-vision reuse across updates would need its own bounded cache and numeric validation.',
        training_scope='No row regrouping, optimizer update, weights/seed/metrics change or forward computation. Across-update language activations cannot be reused as if weights were constant; grouping future courses changes optimizer chronology.',
        existing_shared_entry='Upstream shared_readout_logits packs text IDs/masks without native multimodal tensors and is single-GPU; current Qwen3.5 GDN needs branch-state handling, not only full-attention masks.',
        combined_registered_feature_upper_bytes=26829071326,global_temporary_feature_budget_bytes=26843545600,
        metadata_suballocation='Scalar report≤128KiB from existing full-source/report reserve; no new image/ROI/native activation bank.',
        new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    content=json.dumps(report,indent=2)+'\n';assert len(content.encode())<=128*1024
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n');(out/'summary.json').write_text(content)
    print(json.dumps({k:v for k,v in report.items() if k!='source_files'}),flush=True)


if __name__=='__main__':main()
