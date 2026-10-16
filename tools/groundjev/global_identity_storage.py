"""Read-only physical storage accounting before the isolated CPU experiments."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path


def inventory(root):
    totals=Counter();counts=Counter();seen=set()
    protected={'frame_decisions.jsonl','association_events.json','predictions.json','gt.json',
        'latest_identity_memory.json','final_identity_memory.json','resume_memory.json'}
    model={'tokenizer.json','tokenizer_config.json','config.json','preprocessor_config.json',
        'processor_config.json','generation_config.json','special_tokens_map.json','chat_template.json'}
    for base,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d!='__pycache__']
        for name in files:
            path=Path(base)/name
            if path.is_symlink():continue
            stat=path.stat();key=(stat.st_dev,stat.st_ino)
            if key in seen:continue
            seen.add(key);size=stat.st_size
            if name in protected or 'identity_memory' in name:kind='protected_scientific_predictions_and_identity_states'
            elif name in model or name.endswith(('.safetensors','.pt','.pth','.bin')) and not any(s in name for s in ('feature','bank','regions')):kind='protected_model_and_optimizer_artifacts'
            elif name.endswith(('.npy','.npz','.pt','.pth','.bin')):kind='temporary_feature_arrays'
            elif name.endswith(('.json','.jsonl','.gz','.log','.txt','.csv','.png','.jpg','.jpeg')):kind='conservative_other_metadata_media_and_logs'
            else:kind='other'
            totals[kind]+=size;counts[kind]+=1
    return dict(bytes=dict(totals),files=dict(counts),conservative_temporary_total=
        totals['temporary_feature_arrays']+totals['conservative_other_metadata_media_and_logs']+totals['other'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=inventory(args.source/'outputs/groundjev')
    result.update(scope='All source GroundJEV outputs counted once per inode, no symlink traversal. ALL non-protected files count conservatively, including inline candidate features, curricula, logs, images, videos and unclassified files. Scientific predictions/identity states and model/optimizer artifacts are enumerated separately, never deleted.',
        existing_registered_upper_bytes=26840605662,new_stage_reservation_bytes=2097152,
        combined_registered_upper_bytes=26842702814,limit_bytes=26843545600,
        old_files_deleted_or_modified=False,new_GPU_processes=0)
    result['actual_conservative_headroom_bytes']=result['limit_bytes']-result['conservative_temporary_total']
    assert result['conservative_temporary_total']+result['new_stage_reservation_bytes']<result['limit_bytes'],result
    assert result['combined_registered_upper_bytes']<result['limit_bytes']
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
