"""Explicit bindings for future controls; every effective command is printed."""
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT/'configs/groundjev/shared6000_registered_inference_reuse_v1.json'


def option(command, name):
    return command[command.index(name)+1] if name in command else None


def prepare_command(command, registration=REGISTRATION):
    if not registration.exists() or not any(Path(a).resolve() == ROOT/'tools/groundjev/run_video.py'
            for a in command if a.endswith('.py')):
        return command, None
    plan = json.loads(registration.read_text())
    if not plan['enabled']:
        return command, None
    output = option(command, '--out')
    for binding in plan['bindings']:
        if output is None or Path(output).resolve() != ROOT/binding['consumer']:
            continue
        for flag, expected in binding['required_options'].items():
            actual = option(command, flag)
            if flag in ('--video', '--checkpoint', '--candidate-replay'):
                assert actual is not None and Path(actual).resolve() == ROOT/expected, (flag, actual)
            else:
                assert actual == expected, (flag, actual)
        assert all(flag in command for flag in binding['required_flags'])
        assert ('--assigned-identity-updates' in command) == binding['assigned_identity_updates']
        assert '--category-replay' not in command
        evidence = json.loads((ROOT/plan['validation']/'result.json').read_text())
        assert evidence['status'] == 'NATIVE_CATEGORY_AND_CONTINUOUS_FULL_FRAME_MEMORY_REPLAY_VALIDATED'
        receipt = dict(registration=str(registration), binding=binding,
            timing_comparable_to_fresh_inference=False)
        if binding['mode'] == 'same_checkpoint_current_frame_categories':
            command = command+['--category-replay', str(ROOT/binding['source'])]
        else:
            assert binding['mode'] == 'completed_identical_baseline'
        return command, receipt
    return command, None


def materialize_completed_run(source, target, manifest, checkpoint, candidate_source):
    """Link immutable large outputs; mutable scorer/result files stay independent."""
    result = json.loads((source/'result.json').read_text())
    official = json.loads((source/'official_metrics.json').read_text())
    assert result['all_input_frames_processed'] and result['frames'] == official['frames'] == len(manifest['images'])
    assert result['backend'] == 'full' and result['evidence_format'] == 'v3_crop_motion'
    assert result['adaptive_context_groups'] and result['memory_policy'] == 'gated' and result['lost_seconds'] == 3
    assert not result.get('assigned_identity_updates', False)
    assert not result['gt_read_by_runtime'] and not result['future_frames_read_by_runtime']
    assert Path(result['checkpoint']).resolve() == checkpoint.resolve()
    assert Path(result['shared_input_features_source']).resolve() == candidate_source.resolve()
    assert result['specification'] == dict(task=manifest['task'], vocabulary=manifest['vocabulary'], expression=manifest.get('expression'))
    if manifest['task'] == 'ovmot':
        assert result['category_policy'] == 'native_hierarchical'
    seen = []
    with (source/'frame_decisions.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line); seen.append((row['frame'], row['image_path']))
    assert seen == list(zip(manifest['frame_numbers'], manifest['images']))
    memory = Path(result['final_identity_memory_path'])
    assert memory.is_file()
    target.mkdir(parents=True, exist_ok=False)
    for name in ('frame_decisions.jsonl', 'predictions.txt', 'tracks.mp4'):
        os.link(source/name, target/name)
    os.link(memory, target/memory.name)
    result['final_identity_memory_path'] = str(target/memory.name)
    result['official_metrics'] = 'PENDING_INDEPENDENT_OFFICIAL_REEVALUATION'
    result['completed_run_reuse'] = dict(source=str(source), native_model_calls_this_invocation=0,
        immutable_large_artifacts_hardlinked=True, result_and_scorer_outputs_independent=True,
        source_timing_is_provenance_only=True)
    result['timing_limitations'] = 'Completed identical baseline reused; retained timings describe source execution, not this invocation.'
    (target/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result['completed_run_reuse']


def reuse_registered_baseline(receipt):
    binding = receipt['binding']; source = ROOT/binding['source']
    print(json.dumps(dict(status='WAITING_REGISTERED_ORIGINAL_BASELINE_COMPLETION', **receipt)), flush=True)
    while not ((source/'result.json').exists() and (source/'official_metrics.json').exists()):
        pid, tick = binding['original_owner_identity']
        proc = Path('/proc')/str(pid)
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        assert fields[0] != 'Z' and fields[19] == tick, 'Original baseline owner stopped before completion'
        assert (proc/'cwd').resolve() == ROOT
        assert binding['original_owner_script'] in (proc/'cmdline').read_bytes().decode().split('\0')
        time.sleep(20)
    required = binding['required_options']
    manifest = json.loads((ROOT/required['--video']).read_text())
    report = materialize_completed_run(source, ROOT/binding['consumer'], manifest,
        ROOT/required['--checkpoint'], ROOT/required['--candidate-replay'])
    (ROOT/binding['consumer']/'registered_reuse.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETED_BASELINE_REUSED_WITHOUT_NEW_MODEL_CALLS', **report)), flush=True)
