import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.cache_store import audit
from tools.groundjev.eval_no_mot20.replay_simple import audit_input


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    result={'detector':audit(args.job),'arms':{}}
    for arm in ('simple','j1','coupled'):
        path=args.job/arm
        if (path/'result.json').exists():result['arms'][arm]=audit_input(args.job,path)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
