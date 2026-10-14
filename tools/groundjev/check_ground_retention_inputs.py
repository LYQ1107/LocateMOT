"""Validate all nine native benchmark inputs using the unmodified evaluator."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    code = ROOT/'external/GroundAnything'
    destination = code/'weights/groundjev_original_dlm'
    destination.mkdir(parents=True, exist_ok=True)
    for source in (ROOT/'models/GroundAnything-DLM').iterdir():
        if source.is_file():
            target = destination/source.name
            if not target.exists():
                os.link(source, target)
            assert source.samefile(target)
    # The original frontend resolves recipe paths against its own checkout.
    # Keep the tracked recipe and copy it into a distinct local recipe name.
    recipe = code/'configs/eval/groundjev_original_dense_ref9.local.yaml'
    shutil.copyfile(ROOT/'configs/groundjev/ground_retention_dense_ref9_original.yaml', recipe)
    command = [str(code/'.venv-serve/bin/python'), 'scripts/evaluate.py',
               'configs/eval/groundjev_original_dense_ref9.local.yaml', '--dry-run']
    result = subprocess.run(command, cwd=code, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = ROOT/'outputs/groundjev/ground_retention/dense_ref9_data'
    (out/'original_evaluator_dry_run_stdout.json').write_text(result.stdout)
    (out/'original_evaluator_dry_run_stderr.log').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    data = json.loads(result.stdout)
    assert not data['missing_inputs'] and data['evaluation_scope'] == 'full' and len(data['tasks']) == 9
    receipt = {'status': 'UNMODIFIED_ORIGINAL_DENSE_REF9_INPUT_DRY_RUN_PASSED',
        'tasks': data['tasks'], 'missing_inputs': data['missing_inputs'], 'evaluation_scope': data['evaluation_scope'],
        'native_evaluator_source_modified': False, 'model_files_hardlinks_to_original_release': True,
        'model_evaluation_performed': False,
        'benchmark_data_revision': '772e34bb1505053ed4857bba4405ee7b308bbfb6'}
    (out/'input_validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
