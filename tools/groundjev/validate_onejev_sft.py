"""Block for the actual SFT owner, reload its weights, then measure fixed inputs."""
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import torch
    from PIL import Image
    from safetensors import safe_open
    from locatemot.groundjev.decision import OneJevDecision, SimpleDecision, overlap
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.run_video import restore_completed_prefix
    from qev.mm_engine import MMDecisionEngine
    training = ROOT/'outputs/groundjev/training/onejev_teacher_check_100'
    state_path = training/'owner_state.json'
    state = json.loads(state_path.read_text())
    owner = state['owner_pid']
    print(json.dumps({'status':'WAITING_EXISTING_100_STEP_OWNER','pid':owner}),flush=True)
    # A single blocking command owns this wait, then validates a terminal run.
    while state['status'] != 'TRAINING_CHECK_TERMINAL':
        proc = Path('/proc')/str(owner)/'stat'
        if not proc.exists() or proc.read_text().rsplit(')',1)[1].split()[0]=='Z':
            state = json.loads(state_path.read_text())
            if state['status'] != 'TRAINING_CHECK_TERMINAL':
                raise RuntimeError('Training owner ended without a terminal receipt')
            break
        time.sleep(10)
        state = json.loads(state_path.read_text())
    if state['exit_code'] != 0:
        raise RuntimeError('Training failed; inspect its first failure instead of loading partial weights')
    logs = [json.loads(line) for line in (training/'log.jsonl').read_text().splitlines()]
    done = next(row for row in logs if row.get('event') == 'done')
    if done['steps'] != 100:
        raise RuntimeError('The registered100-step stage did not complete')
    metrics = [row for row in logs if 'loss' in row]
    if not all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) and row['grad_norm']>0 for row in metrics):
        raise RuntimeError('Loss/gradient correctness check failed')
    checkpoint = training/'final'
    with safe_open(ROOT/'models/OneJev-0.8B/model.safetensors',framework='pt') as base, \
         safe_open(checkpoint/'model.safetensors',framework='pt') as trained:
        if set(base.keys()) != set(trained.keys()):
            raise RuntimeError('Full SFT checkpoint has missing/different weight names')
        names = [name for name in base.keys() if '.language_model.layers.' in name and name.endswith('mlp.gate_proj.weight')]
        probes = []
        for name in names[::max(1,len(names)//3)][:3]:
            a = base.get_tensor(name).to(torch.bfloat16).float()
            b = trained.get_tensor(name).to(torch.bfloat16).float()
            delta = (a-b).abs()
            probes.append({'name':name,'changed_elements':int((delta>0).sum()),
                           'mean_absolute_delta':float(delta.mean()),'max_absolute_delta':float(delta.max())})
        if not probes or not any(p['changed_elements'] for p in probes):
            raise RuntimeError('No trained language-weight change at the same BF16 precision')
        entries = len(trained.keys())
    rows = list(map(json.loads,(ROOT/'outputs/groundjev/runs/mot17_first_native_clip_repair2/frame_decisions.jsonl').read_text().splitlines()))
    manifest = json.loads((ROOT/'outputs/groundjev/pilot_data/mot17_first4.json').read_text())
    tracker = OnlineTracker(SimpleDecision())
    specification = {'task':'mot','vocabulary':['person'],'expression':None}
    restore_completed_prefix(tracker,rows[:1],list(zip(manifest['images'],manifest['frame_numbers'])),manifest['fps'],specification)
    with Image.open(rows[1]['image_path']) as source:
        image = source.convert('RGB')
    candidate = rows[1]['candidates'][:1]
    ranked = sorted(tracker.tracks.values(),key=lambda t:(-overlap(candidate[0]['box'],t.predicted_box(303/30)),t.track_id))[:3]
    builder = OneJevDecision.__new__(OneJevDecision)
    request, media = builder.build_request(image,candidate,ranked,303/30,specification)
    # Three native types plus one lifecycle decision, all actual tracking evidence.
    request.questions = {key:value for key,value in request.questions.items()
                         if key.startswith(('match_','update_','reliability_')) or key==f'life_{ranked[0].track_id}'}
    responses, infos = [], []
    for repeat in range(2):
        engine = MMDecisionEngine(str(checkpoint),device='cuda:0',dtype='bfloat16',cuda_graphs=False)
        response, metadata = engine.decide(request,media=media)
        responses.append(response.model_dump())
        infos.append(vars(engine.info))
        del engine
        gc.collect(); torch.cuda.empty_cache()
    maximum = 0.
    for key, answer in responses[0]['answers'].items():
        other = responses[1]['answers'][key]
        values = answer.get('probabilities') or {'noul':answer['noul']}
        repeated = other.get('probabilities') or {'noul':other['noul']}
        for label, value in values.items():
            if not math.isfinite(value):
                raise RuntimeError('Checkpoint produced non-finite native output')
            maximum = max(maximum,abs(value-repeated[label]))
    if maximum > 1e-6:
        raise RuntimeError(f'Repeated reload disagrees by{maximum}')
    receipt = {'status':'REAL_100_STEP_SFT_CHECKPOINT_RELOAD_VALIDATED','checkpoint':str(checkpoint),
        'steps':done['steps'],'examples':done['examples'],'elapsed_seconds':done['elapsed_s'],
        'weight_entries':entries,'same_bf16_precision_weight_probes':probes,
        'logged_loss_and_gradient_finite_nonzero':True,'reload_probability_max_absolute_difference':maximum,
        'reload_tolerance':1e-6,'responses':responses,'engine_info':infos,
        'native_fork_auto_check_unchanged':True,'evaluation_gt_read_by_inference':False,
        'scope':'teacher_curriculum_correctness_check_not_formal500_to1000_step_completion'}
    (training/'checkpoint_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('responses','engine_info')}),flush=True)
    out = ROOT/'outputs/groundjev/runs/mot17_first4_onejev_sft100_fixed_candidates'
    subprocess.run([sys.executable,str(ROOT/'tools/groundjev/run_video.py'),'--task','mot',
        '--video',str(ROOT/'outputs/groundjev/pilot_data/mot17_first4.json'),'--backend','full',
        '--checkpoint',str(checkpoint),'--grounding-parser','official_scoring',
        '--candidate-replay',str(ROOT/'outputs/groundjev/runs/mot17_first_native_clip_repair2/frame_decisions.jsonl'),
        '--out',str(out)],cwd=ROOT,check=True)
    subprocess.run(['python3',str(ROOT/'tools/groundjev/evaluate_video.py'),
        '--video',str(ROOT/'outputs/groundjev/pilot_data/mot17_first4.json'),'--run',str(out)],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
