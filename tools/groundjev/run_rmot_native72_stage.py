"""One finite owner: Ground replica, native own-history data, paired adaptation."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = ROOT/'outputs/groundjev/compact/rmot_native72_stage_v1'
    out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'STARTING_FINITE_BROAD_FIT_RMOT_STAGE', 'owner_pid': os.getpid(),
        'plan': 'configs/groundjev/rmot_native72_20261014.json', 'registered_frames': 72,
        'owned_GPU_roles': {'existing_OneJev500': 4, 'existing_native_ROI': 5,
            'temporary_native_Ground_service': 6, 'actor_training_and_controls': 7},
        'maximum_project_physical_GPUs': 4, 'feature_budget_bytes': 25*2**30,
        'old_U2_queue_started': False, 'completed': {}}
    def save():
        (out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)
    def run(stage, command):
        state['status'] = stage; save()
        with (out/(stage+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    def gpu(command):
        return ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '7', '--',
                str(ROOT/'.venv-onejev-py310/bin/python'), *command]
    server = None
    def stop_server():
        nonlocal server
        if server is None:
            return
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL); server.wait()
        state['owned_Ground_replica_stopped'] = True
        server = None; save()
    save()
    try:
        # A fresh isolated replica owns this port and its process group.
        # An existing listener is an error, never an invitation to kill it.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 8101))
        with (out/'Ground_service.log').open('w') as log:
            server = subprocess.Popen(['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '6', '--',
                str(ROOT/'external/GroundAnything/.venv-serve/bin/python'), 'tools/groundjev/serve_groundanything.py',
                '--port', '8101', '--mem-fraction-static', '.4', '--out', str(out/'Ground_service')],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='WAITING_FOR_OWNED_NATIVE_GROUND_READY', owned_Ground_process_group=server.pid); save()
            deadline = time.monotonic()+600
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Owned Ground service exited before readiness; preserve first server log error')
                try:
                    with urlopen('http://127.0.0.1:8101/v1/models', timeout=5) as response:
                        ready = json.load(response)
                    if any(m['id'] == 'groundinganything' for m in ready['data']):
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Owned Ground service readiness deadline exceeded')
                time.sleep(2)
            state.update(status='OWNED_NATIVE_GROUND_READY'); save()
            run('BUILD_NATIVE72_AND_BROAD_TEACHER_PACKS', gpu(['tools/groundjev/prepare_rmot_native72.py']))
            pack = ROOT/'outputs/groundjev/compact/rmot_native72_v1'
            receipt = json.loads((pack/'owner_state.json').read_text())
            assert receipt['status'] == 'BROAD_FIT_NATIVE_OWN_HISTORY_PACK_COMPLETE' and receipt['frames'] == 72
            state['completed']['pack'] = receipt; save()
            # No detector is needed by the subsequent identical-candidate
            # controls. Release its physical GPU before training the heads.
            stop_server()
        for variant in ('broad_teacher', 'broad_native'):
            training = out/(variant+'_training')
            run(variant+'_TRAINING', gpu(['tools/groundjev/train_compact_joint.py', '--pack', str(pack/variant),
                '--out', str(training), '--checkpoint', str(ROOT/'outputs/groundjev/compact/joint1200_v1/checkpoint.pt')]))
            trained = json.loads((training/'owner_state.json').read_text())
            assert trained['status'] == 'SHARED_COMPACT_JOINT1200_TRAINING_COMPLETE' and trained['steps'] == 1200
            state['completed'][variant+'_training'] = trained; save()
            controls = out/(variant+'_controls')
            run(variant+'_OFFICIAL_CONTROLS', ['python3', 'tools/groundjev/run_compact_joint_controls.py',
                '--training', str(training), '--out', str(controls), '--reuse-before',
                str(ROOT/'outputs/groundjev/compact/joint1200_controls_v1')])
            run(variant+'_VALIDATION', ['python3', 'tools/groundjev/diagnose_joint_controls.py', '--root', str(controls)])
            state['completed'][variant+'_controls'] = json.loads((controls/'validation_and_diagnostic.json').read_text()); save()
            full = out/(variant+'_full1203')
            run(variant+'_FULL_VIDEO', ['python3', 'tools/groundjev/run_joint_full_video.py',
                '--out', str(full), '--checkpoint', str(training/'checkpoint.pt')])
            state['completed'][variant+'_full1203'] = json.loads((full/'owner_state.json').read_text()); save()
        state['status'] = 'BROAD_FIT_TEACHER_VERSUS_NATIVE_OWN_HISTORY_STAGE_COMPLETE'; save()
    except Exception as error:
        state.update(status='BROAD_FIT_NATIVE72_STAGE_FAILED', error=repr(error)); save()
        raise
    finally:
        stop_server()


if __name__ == '__main__':
    main()
