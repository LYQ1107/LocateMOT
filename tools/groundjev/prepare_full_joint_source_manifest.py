"""One CPU owner waits for all native RMOT units, then freezes full exposure."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.run_history_refresh_controls import wait_dependency
    from tools.groundjev.run_source_manifest_sft import PROTOCOL
    parser = argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out=args.out.resolve(); args.out.mkdir(parents=True,exist_ok=False)
    state=dict(status='WAITING_ALL18_ACTUAL_RMOT_FULL_GROUP_EXPORTS', owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        new_GPU_or_optimizer_updates=0, full_goal_complete=False)
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    dependency=dict(directory='outputs/groundjev/training_data/rmot_allfit1814_query18_full_v3_groups_v2_shared_gpu7',
        identity=[3320470,'48408216'],script='tools/groundjev/export_rmot_full_fit_groups.py',
        required_terminal_status='ALL18_NATIVE_RMOT_QUERY_ROLLOUTS_FULL_V3_GROUPS_AND_CPU_PLANS_VALIDATED')
    registration=dict(project_root='/data2/user/LocateMOT', dependency=dependency,
        protocol=PROTOCOL, current6000_changed=False,
        policy='One pass over all retained MOT43-clips rows, original RMOT4350-expression rows and Base175 category rows; new RMOT18/class-proposal components contribute query-membership/presence only.',
        task_mixture='Natural full-source row counts, explicitly differs from current balanced24k; no implicit row weighting or oversampling.',
        training_launch='not launched by this preparation; requires valid distributed protocol, complete6000 checkpoint and registered larger-stage recipe',
        storage='only source-list manifest and summaries; no new row, image, ROI or history archive',
        reserved_existing_future_curriculum_bytes=16777216, full_goal_complete=False)
    (args.out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n');save()
    try:
        rmot=wait_dependency(dependency)
        assert len(rmot['completed'])==18 and rmot['unique_physical_frames']==1814
        mot=json.loads((ROOT/'outputs/groundjev/training_data/broader_native_MOT1056_full_v3_groups_v1/owner_state.json').read_text())
        assert mot['status']=='ALL43_NATIVE_TRAIN_CLIPS_FULL_V3_GROUP_REFS_AND_CPU_PLANS_VALIDATED'
        sources=[];counts=Counter();ids=set()
        def add(path,bucket,families=None,expected=None):
            path=Path(path);path=path if path.is_absolute() else ROOT/path
            selected=0;local=Counter()
            with gzip.open(path,'rt') as stream:
                for line in stream:
                    row=json.loads(line)
                    if families is not None and row['task'] not in families:continue
                    assert row.get('split','train')=='train'
                    key=(str(path),row['id']);assert key not in ids;ids.add(key)
                    selected+=1;local[row['task']]+=1
                    if row['task'] in ('query_membership','empty_query_presence'):
                        local['semantic_target:'+next(iter(row['target']))]+=1
            if expected is not None:assert selected==expected
            sources.append(dict(path=str(path.relative_to(ROOT)),bucket=bucket,families=families,rows=selected,counts=dict(local)))
            counts[bucket]+=selected
        for item in mot['completed'].values():
            assert item['dropped_rows']==0 and item['all_actor_outputs_states_exact']
            add(item['row_references'],'mot',expected=item['counts']['rows'])
        add('outputs/groundjev/training_data/rmot_query_diversity_component_v3_official_box_projection/train.jsonl.gz','rmot',expected=87721)
        add('outputs/groundjev/training_data/base175_streamed_component_v1/train.jsonl.gz','ovmot',expected=105956)
        semantic=['query_membership','empty_query_presence']
        for item in rmot['completed'].values():
            assert item['dropped_rows']==0 and item['all_actor_outputs_states_exact']
            add(item['row_references'],'rmot',semantic)
        class_root=ROOT/'outputs/groundjev/training_data/rmot_class_proposals_original_query154_v1'
        class_done=json.loads((class_root/'owner_state.json').read_text())
        assert class_done['status']=='NATIVE_CLASS_PROPOSALS_ORIGINAL_QUERY_154_COMPONENT_ALL_INPUTS_AND_CPU_ROWS_VALIDATED'
        add(class_done['export_summary']['row_references'],'rmot',semantic,expected=2182)
        manifest=dict(protocol=PROTOCOL,sources=sources,rows=sum(counts.values()),rows_per_task=dict(counts),
            coverage=dict(MOT_clips=43,MOT_physical_frames=1056,RMOT_unique_physical_frames=1814,
                RMOT_expression_files=4350,OVMOT_Base_classes=175,OVMOT_category_annotation_rows=105956),
            all_source_files_immutable_completed=True, GT_only_fit_and_Base_training=True,
            physical_frames_not_summed_across_query_or_proposal_repeats=True,
            current6000_inputs_unchanged=True,new_optimizer_updates=0,full_goal_complete=False)
        assert counts['mot']==252258 and counts['ovmot']==105956
        (args.out/'train.sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
        state.update(status='FULL_NATIVE_THREE_TASK_SOURCE_EXPOSURE_MANIFEST_PREPARED_NOT_TRAINED',
            rows=manifest['rows'],rows_per_task=dict(counts),source_files=len(sources));save()
    except Exception as error:
        state.update(status='FULL_JOINT_SOURCE_MANIFEST_PREPARATION_FAILED',error=repr(error));save();raise


if __name__=='__main__':
    main()
