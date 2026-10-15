"""One blocking official CPU plan for all streamed-media category component rows."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def main():
    from tools.groundjev.run_streamed_media_sft import configure_streamed_sft
    stateful=configure_streamed_sft()
    from train import sft
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--component',type=Path,required=True)
    parser.add_argument('--media-validation',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    proof=json.loads(args.media_validation.read_text())
    assert proof['status']=='REAL_STREAMED_CATEGORY_PIXELS_QUESTIONS_AND_PROCESSOR_TENSORS_EXACT'
    assert proof['all_rows_validated']==105956
    args.out.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/groundjev/onejev_unified600_native_v2.json').read_text())
    config.update(model=str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600'),
        train=str((args.component/'train.jsonl.gz').resolve()),name='official_component_CPU_plan',
        runs_dir=str(args.out.resolve()),plan_only=True,device='cpu',max_steps=0)
    path=args.out/'official_plan_config.json';path.write_text(json.dumps(config,indent=2)+'\n')
    state={'status':'RUNNING_FULL105956_NATIVE_CPU_TOKEN_PLAN','owner_pid':os.getpid(),
        'component':str(args.component),'new_model_or_GPU_loaded':False,'training_started':False,
        'text_rows_loaded_in_RAM_by_native_trainer':True,'images_generated_only_when_requested':True,
        'max_steps0_role':'plan-only full component; no optimizer budget or training launch registered',
        'native_reporter_timing_scope':'cached-plan reporter only; not full tokenization/planning throughput',
        'full_goal_complete':False}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    native_plan,native_epoch=sft.plan_only,sft.plan_epoch
    def capture(bundle,rows,settings,world):
        cached=native_epoch(bundle,rows,settings,0,0,1)
        batches,dropped,lengths=cached
        assert len(rows)==105956 and dropped==0 and len(lengths)==len(rows)
        families=Counter(row['task'] for row in rows)
        assert families=={'category_group':52978,'category_leaf':52978}
        classes=Counter(row['supervision']['Base_vocabulary_index'] for row in rows)
        assert len(classes)==175 and all(n%2==0 for n in classes.values())
        protocol=stateful.protocol(settings)
        assert protocol['input_io_protocol']=='gzip-jsonl-and-native-category-crop-recipes-v1'
        steps=sft.plan_steps(batches,lengths,1,settings.accum_rows)
        assert len(steps)==26489
        sft.plan_epoch=lambda *unused,**kwargs:cached
        try:
            code=native_plan(bundle,rows,settings,world)
        finally:
            sft.plan_epoch=native_epoch
        assert code==0 and not torch.cuda.is_initialized()
        state.update(status='FULL_BASE175_OFFICIAL_CPU_TOKEN_PLAN_COMPLETE_NOT_TRAINED',
            all_rows=len(rows),overlength_dropped_rows=dropped,max_actual_tokens=max(lengths),
            total_actual_tokens=sum(lengths),question_families=dict(families),Base_classes=len(classes),
            hypothetical_full_component_epoch_updates=len(steps),actual_new_training_updates=0,
            input_io_resume_protocol=protocol['input_io_protocol'],
            native_rendering_step_planning_and_optimizer_math_changed=False,
            process_CPU_RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save()
        return code
    sft.plan_only=capture;save()
    try:
        code=sft.main(['--config',str(path),'--plan-only'])
        assert code==0
    except Exception as error:
        state.update(status='BASE175_OFFICIAL_CPU_PLAN_FAILED',error=repr(error));save()
        raise
    finally:
        sft.plan_only=native_plan


if __name__=='__main__':
    main()
