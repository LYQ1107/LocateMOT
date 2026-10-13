"""Single blocking owner: native full video, identical-candidate control, metrics."""
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, default=ROOT/'outputs/groundjev/pilot_data/dancetrack.json')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/groundjev/pilot_runs/dancetrack0004_pretrained_native_v1')
    parser.add_argument('--decoder-policy', choices=['generic_greedy', 'tracking_dense'], default='generic_greedy')
    parser.add_argument('--perception-failure-policy', choices=['stop','empty_and_record'], default='stop')
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('preserve an existing pilot owner and its results')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.video.read_text())
    state = {'status':'STARTING_FULL_REGISTERED_VIDEO', 'video':str(args.video),
             'frames':len(manifest['images']), 'scope':manifest['scope'], 'physical_gpus':[6],
             'checkpoint':str(ROOT/'models/OneJev-0.8B'), 'parser_policy':'official_scoring',
             'decoder_policy':args.decoder_policy,
             'perception_failure_policy':args.perception_failure_policy,
             'resume_from':str(args.resume_from) if args.resume_from else None,
             'candidate_controls':'exact full-run candidates replayed by Simple',
             'training_gt_used_by_inference':False, 'pilot_completion':False}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    save()
    peaks, device_peaks = {}, defaultdict(int)
    stop = threading.Event()
    monitor_errors = []
    def monitor():
        # Monitoring is internal to this sole blocking owner, never agent polling.
        while not stop.is_set():
            try:
                raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                                '--format=csv,noheader,nounits'],text=True)
                current = defaultdict(int)
                for uuid, pid, memory in csv.reader(io.StringIO(raw)):
                    proc = Path('/proc')/pid.strip()
                    try:
                        cwd = proc.joinpath('cwd').resolve()
                        if not cwd.is_relative_to(ROOT):
                            continue
                        amount = int(memory)*2**20
                        key = uuid.strip()+':'+pid.strip()
                        peaks[key] = max(peaks.get(key,0),amount)
                        current[uuid.strip()] += amount
                    except (OSError, ValueError):
                        continue
                for uuid, amount in current.items():
                    device_peaks[uuid] = max(device_peaks[uuid],amount)
            except Exception as error:
                monitor_errors.append(str(error))
                return
            stop.wait(5)
    thread = threading.Thread(target=monitor,daemon=True)
    thread.start()
    def run(stage, command):
        state['status'] = stage
        save()
        with (args.out/(stage.lower()+'.log')).open('w') as log:
            process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                log.write(line); log.flush()
                print(line,end='',flush=True)
            code = process.wait()
        if code:
            state.update(status='FULL_VIDEO_PILOT_FAILED',failed_stage=stage,exit_code=code)
            save()
            raise RuntimeError(f'{stage} failed; retain completed frames and inspect its first failure')
    try:
        start = time.perf_counter()
        full, simple = args.out/'full', args.out/'simple'
        command = ['python3',str(ROOT/'tools/groundjev/run_with_resources.py'),'--gpus','6','--',
            str(ROOT/'.venv-onejev-py310/bin/python'),str(ROOT/'tools/groundjev/run_video.py'),
            '--task','mot','--video',str(args.video),'--backend','full','--grounding-parser','official_scoring',
            '--grounding-decoder-policy',args.decoder_policy,'--perception-failure-policy',args.perception_failure_policy,'--out',str(full)]
        if args.resume_from:
            command += ['--resume-from',str(args.resume_from)]
        run('RUNNING_NATIVE_FULL_VIDEO',command)
        state['full_invocation_wall_seconds'] = time.perf_counter()-start
        run('RUNNING_IDENTICAL_CANDIDATE_SIMPLE_CONTROL', [str(ROOT/'.venv-onejev-py310/bin/python'),
            str(ROOT/'tools/groundjev/run_video.py'),'--task','mot','--video',str(args.video),'--backend','simple',
            '--grounding-parser','official_scoring','--grounding-decoder-policy',args.decoder_policy,
            '--perception-failure-policy',args.perception_failure_policy,
            '--candidate-replay',str(full/'frame_decisions.jsonl'),'--out',str(simple)])
        for name, directory in [('FULL',full),('SIMPLE',simple)]:
            run('EVALUATING_'+name, ['python3',str(ROOT/'tools/groundjev/evaluate_video.py'),
                                    '--video',str(args.video),'--run',str(directory)])
        state.update(status='FULL_VIDEO_AND_FIXED_CANDIDATE_CONTROL_COMPLETE',
            full=json.loads((full/'official_metrics.json').read_text()),
            simple=json.loads((simple/'official_metrics.json').read_text()))
        save()
    finally:
        stop.set(); thread.join(timeout=10)
        (args.out/'gpu_process_peaks.json').write_text(json.dumps({
            'sampling_interval_seconds':5,'process_peaks_bytes':peaks,
            'simultaneous_project_device_peaks_bytes':dict(device_peaks),
            'foreign_processes_excluded':True,'scope':'all_groundjev_processes_observed_during_full_video_including_concurrent_training',
            'monitor_errors':monitor_errors},indent=2)+'\n')


if __name__=='__main__':
    main()
