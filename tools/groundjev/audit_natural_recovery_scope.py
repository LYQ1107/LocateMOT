"""Find annotation-defined visibility challenges, without reading model predictions."""
from collections import defaultdict
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=Path('/data2/user/LocateMOT/outputs/unified/u2/protocol')


def main():
    out=ROOT/'outputs/groundjev/recovery_scope_20261015';out.mkdir(exist_ok=False)
    sequences=[];cases=[]
    for dataset in ('mot17_half','mot20_half','dancetrack'):
        protocol=json.loads((PROTOCOL/(dataset+'.json')).read_text())
        for sequence in protocol['sequences']:
            frames=sequence['validation_frames']
            if not frames:continue
            allowed={int(Path(p).stem) for p in frames};tracks=defaultdict(list)
            with Path(sequence['gt_path']).open() as source:
                for fields in csv.reader(source):
                    number=int(fields[0])
                    if number in allowed and int(float(fields[6]))==1 and int(float(fields[7]))==1:
                        tracks[int(fields[1])].append((number,float(fields[8])))
            missing=0;low=0
            for identity,observations in tracks.items():
                observations.sort();fps=sequence['fps']
                for (a,_),(b,_) in zip(observations,observations[1:]):
                    gap=(b-a)/fps
                    if 3<gap<=30:
                        missing+=1;cases.append({'dataset':dataset,'sequence':sequence['sequence'],'identity':identity,
                            'before':a,'after':b,'gap_seconds':gap,'kind':'GT_annotation_gap','GT_absent_frames':b-a-1})
                reliable=[r for r in observations if r[1]>=.6]
                for (a,_),(b,_) in zip(reliable,reliable[1:]):
                    gap=(b-a)/fps
                    if 3<gap<=30:
                        between=[v for f,v in observations if a<f<b]
                        if between:
                            low+=1;cases.append({'dataset':dataset,'sequence':sequence['sequence'],'identity':identity,
                                'before':a,'after':b,'gap_seconds':gap,'kind':'gap_between_GT_visibility_ge06',
                                'annotated_intermediate_frames':len(between),'intermediate_max_visibility':max(between),
                                'not_a_prediction_or_candidate_missing_gap':True})
            sequences.append({'dataset':dataset,'sequence':sequence['sequence'],'validation_frames':len(frames),
                'GT_tracks':len(tracks),'annotation_gaps_gt3_le30s':missing,'visibility_challenges_gt3_le30s':low})
    cases.sort(key=lambda x:(-x['gap_seconds'],x['dataset'],x['sequence'],x['identity'],x['before']))
    report={'status':'REGISTERED_VALIDATION_ANNOTATION_CHALLENGES_AUDITED','sequences':sequences,'cases':cases,
        'GT_used_for_challenge_registration_only':True,'model_outputs_read':False,
        'artificial_occlusion_or_proposal_removal':False,'actual_candidate_gap_support_still_requires_inference':True,
        'official_GT_visibility_filter_changed':False,'challenge_visibility_threshold':.6}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if cases:
        selected=cases[0]
        protocol=json.loads((PROTOCOL/(selected['dataset']+'.json')).read_text())
        s=next(s for s in protocol['sequences'] if s['sequence']==selected['sequence'])
        paths=[str(Path(s['image_dir'])/p) for p in s['validation_frames']]
        manifest={'dataset':selected['dataset'],'sequence':s['sequence'],'task':'mot','images':paths,
            'protocol':str(PROTOCOL/(selected['dataset']+'.json')),
            'frame_numbers':[int(Path(p).stem) for p in paths],'fps':s['fps'],'vocabulary':['person'],
            'gt_path_for_evaluation_only':s['gt_path'],'scope':'complete_registered_validation_scope_natural_visibility_challenge',
            'training_on_this_scope':False,'runtime_GT_fields':False,'official_test_labels_read':False,
            'image_size':[s['width'],s['height']],'challenge_registration':selected,
            'scope_note':'complete registeredvalidation frames; halfvideo protocols are not complete physical videos'}
        assert all(Path(p).is_file() for p in paths)
        (out/'registered_mot_challenge.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'sequences':len(sequences),'cases':len(cases),'first_cases':cases[:5]},indent=2))


if __name__=='__main__':main()
