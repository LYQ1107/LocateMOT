"""Regression on actual native rows: legacy manifest and explicit indices."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    from tools.groundjev.run_source_manifest_sft import configure_source_manifest_sft,PROTOCOL,INDEXED_PROTOCOL
    from train import common
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    args.out.mkdir(parents=True,exist_ok=False);configure_source_manifest_sft()
    source='outputs/groundjev/optimization/distributed_native_sft_probe_v1/train.jsonl.gz'
    original=common.load_rows(ROOT/source);assert len(original)==8
    legacy=dict(protocol=PROTOCOL,rows=8,sources=[dict(path=source,families=None,rows=8)])
    path=args.out/'legacy.sources.json';path.write_text(json.dumps(legacy))
    assert common.load_rows(path)==original
    indices=[0,3,6]
    indexed=dict(protocol=INDEXED_PROTOCOL,rows=3,sources=[dict(path=source,families=None,rows=3,source_rows=8,row_indices=indices)])
    path=args.out/'indexed.sources.json';path.write_text(json.dumps(indexed))
    actual=common.load_rows(path);assert actual==[original[i] for i in indices]
    report=dict(status='REAL_NATIVE_LEGACY_AND_INDEXED_SOURCE_ROWS_EXACT',legacy_rows_exact=8,indexed_rows_exact=3,
        all_original_IDs_states_media_questions_targets_and_order_exact=True,new_GPU_or_training_updates=0,full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
