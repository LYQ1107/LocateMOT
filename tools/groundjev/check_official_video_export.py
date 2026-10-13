"""Validate the complete legal-scope exporter using a labeled GT oracle fixture."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest_path = ROOT/'outputs/groundjev/pilot_data/mot17.json'
    manifest = json.loads(manifest_path.read_text())
    allowed = set(manifest['frame_numbers'])
    with tempfile.TemporaryDirectory(prefix='groundjev_trackeval_fixture_') as temp:
        run = Path(temp)
        lines = []
        for line in Path(manifest['gt_path_for_evaluation_only']).read_text().splitlines():
            fields = list(map(float,line.split(',')))
            if int(fields[0]) in allowed and fields[6]==1 and fields[7]==1:
                lines.append(','.join(map(str,[int(fields[0]),int(fields[1]),*fields[2:6],1,-1,-1,-1])))
        (run/'predictions.txt').write_text('\n'.join(lines)+'\n')
        (run/'frame_decisions.jsonl').write_text(''.join(json.dumps({'frame':f,'image_path':p})+'\n'
            for f,p in zip(manifest['frame_numbers'],manifest['images'])))
        (run/'result.json').write_text(json.dumps({'status':'VIDEO_INFERENCE_COMPLETE','task':'mot',
            'scope':manifest['scope'],'frames':len(manifest['images']),'fixture_only':True}))
        subprocess.run([sys.executable,str(ROOT/'tools/groundjev/evaluate_video.py'),'--video',str(manifest_path),
                        '--run',str(run),'--fixture-only'],check=True)
        report = json.loads((run/'official_metrics.json').read_text())
        assert report['fixture_only'] and not report['model_score_claim']
        assert report['frames']==299 and report['metrics']['IDSW']==0
        report.pop('gt_view')
        report['status'] = 'OFFICIAL_TRACK_EVAL_EXPORT_FIXTURE_PASS_NOT_MODEL_BENCHMARK'
        report['input_kind'] = 'legal development GT oracle fixture; no model inference'
        out = ROOT/'outputs/groundjev/validation/official_export_fixture.json'
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
