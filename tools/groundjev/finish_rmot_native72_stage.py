"""Resume only the unfinished full-video unit, retaining the original failure."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    root = ROOT/'outputs/groundjev/compact/rmot_native72_stage_v1'
    old = json.loads((root/'owner_state.json').read_text())
    assert old['status'] == 'BROAD_FIT_NATIVE72_STAGE_FAILED'
    probe = json.loads((root/'failed_frame799_regression.json').read_text())
    assert probe['status'] == 'ACTUAL_FAILED_FRAME_FRESH_CUDA_CONTEXT_PASSED'
    source = root/'broad_native_full1203/joint1200'
    failure = json.loads((source/'failure.json').read_text())
    assert failure['frame'] == 799 and failure['completed_frames'] == 798
    out = root/'broad_native_full1203_v2_cuda_resume'
    state = {'status': 'RESUMING_ONLY_INTERRUPTED_FULL_VIDEO_UNIT', 'original_failure': str(source/'failure.json'),
        'completed_prefix': 798, 'first_unfinished_frame': 799, 'same_GPU': 7,
        'same_model_parameters_GT_candidates_metrics': True, 'checkpoint_or_driver_reset': False,
        'external_CUDA_root_cause': 'not established; accessible NVML ECC counts0/recovery actionNone',
        'completed': old['completed']}
    def save():
        (root/'recovery_owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)
    save()
    try:
        with (root/'broad_native_FULL_VIDEO_RESUME.log').open('w') as log:
            subprocess.run(['python3', 'tools/groundjev/run_joint_full_video.py', '--out', str(out),
                '--checkpoint', str(root/'broad_native_training/checkpoint.pt'), '--resume-from', str(source)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((out/'owner_state.json').read_text())
        assert result['status'] == 'JOINT_CORE_FULL_VIDEO_RETENTION_COMPLETE'
        inference = json.loads((out/'joint1200/result.json').read_text())
        assert inference['frames'] == 1203 and inference['resumed_completed_frames'] == 798
        state['completed']['broad_native_full1203'] = result
        state['status'] = 'BROAD_FIT_STAGE_COMPLETE_WITH_RETAINED_CUDA_FAILURE_AND_CAUSAL_RESUME'; save()
    except Exception as error:
        state.update(status='NATIVE72_FULL_VIDEO_RECOVERY_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
