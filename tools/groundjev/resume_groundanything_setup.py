"""Resume the official serve dependency stages using verified local wheel transport."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'external/GroundAnything'
PYTHON = SOURCE/'.venv-serve/bin/python'
UV = ROOT/'.venv-bootstrap/bin/uv'


def main():
    sys.path.insert(0,str(SOURCE))
    from models.dependency_contract import verify_dependency
    verify_dependency('sglang')
    env = os.environ.copy()
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','PIP_CONSTRAINT'):
        env.pop(key,None)
    env['UV_CACHE_DIR'] = str(ROOT/'.download-cache/ground-serve')
    common = [str(UV),'pip','install','--python',str(PYTHON),'--find-links',str(ROOT/'.download-cache/wheelhouse')]
    commands = [common+['-c',str(SOURCE/'requirements/serve.txt'),'-r',str(SOURCE/'requirements/serve.txt'),'-e',str(SOURCE/'vendor/sglang')],
                common+['--no-deps','nvidia-cudnn-cu12==9.16.0.29'],
                common+['--no-deps','-e',str(SOURCE)],
                [str(PYTHON),str(SOURCE/'scripts/check_serve_environment.py')]]
    report = ROOT/'outputs/groundjev/model_setup/ground_setup_resume.json'
    state = {'status':'RESUMING_OFFICIAL_SERVE_STAGES','commands':commands,'vendor_archive_verified':True,
             'prior_interruption':'controlled replacement of slow single-connection package transport',
             'bootstrap_and_vendor_preparation_reused':True,'runtime_constraints_changed':False}
    report.write_text(json.dumps(state,indent=2)+'\n')
    for index,command in enumerate(commands):
        print(json.dumps({'state':'SERVE_SETUP_STAGE','index':index,'command':command}),flush=True)
        subprocess.run(command,cwd=SOURCE,env=env,check=True)
    freeze = subprocess.check_output([str(PYTHON),'-m','pip','freeze','--all'],text=True)
    (PYTHON.parent.parent/'installed.freeze.txt').write_text(freeze)
    state['status'] = 'OFFICIAL_SERVE_ENVIRONMENT_CHECKS_COMPLETE_NOT_MODEL_LOAD'
    report.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({'status':state['status']}),flush=True)


if __name__=='__main__':
    main()
