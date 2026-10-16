"""Actual native multimodal planning through the future source-list input path."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    import torch
    from tools.groundjev.run_source_manifest_sft import configure_source_manifest_sft,PROTOCOL
    from train import common,sft
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    configure_source_manifest_sft()
    source='outputs/groundjev/optimization/distributed_native_sft_probe_v1/train.jsonl.gz'
    original=common.load_rows(ROOT/source)
    full=dict(protocol=PROTOCOL,sources=[dict(path=source,families=None,rows=8)],rows=8)
    full_path=args.out/'full.sources.json';full_path.write_text(json.dumps(full,indent=2)+'\n')
    actual=common.load_rows(full_path)
    assert len(actual)==8 and actual==original
    semantic=dict(protocol=PROTOCOL,sources=[dict(path=source,families=['query_membership','empty_query_presence'],rows=2)],rows=2)
    filtered_path=args.out/'semantic.sources.json';filtered_path.write_text(json.dumps(semantic,indent=2)+'\n')
    filtered=common.load_rows(filtered_path)
    assert filtered==[r for r in original if r['task']=='query_membership']
    assert {next(iter(r['target'])) for r in filtered}=={'yes','no'}
    config=ROOT/'outputs/groundjev/optimization/distributed_native_sft_probe_v1/single_config.json'
    settings=sft.parse_args(['--config',str(config),'--plan-only'])
    bundle=common.load_bundle(settings.model,device='cpu',dtype=settings.dtype,train=True,
        multimodal=True,chat_template=settings.chat_template,load_model=False)
    batches,dropped,lengths=sft.plan_epoch(bundle,actual,settings,0,0,1)
    assert dropped==0 and sorted(i for b in batches for i in b)==list(range(8))
    recorded=json.loads((ROOT/'outputs/groundjev/optimization/distributed_native_sft_probe_v1/rank_receipts/single_rank0.json').read_text())
    assert lengths==recorded['plans'][0]['lengths']
    assert not torch.cuda.is_initialized()
    report=dict(status='NATIVE_MM_SOURCE_MANIFEST_HYDRATION_AND_FAMILY_FILTER_INPUTS_VALIDATED',
        full_rows_exact=8,all_original_states_media_questions_targets_and_IDs_exact=True,
        native_CPU_plan_rows=8,dropped_rows=0,max_tokens=max(lengths),native_token_lengths_exact=True,
        semantic_filter_rows_exact=2,positive_and_negative_preserved=True,
        model_or_GPU_loaded=False,new_image_or_feature_archive=False,
        whole_future_corpus_CPU_plan_still_required=True,full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
