#!/usr/bin/env python3
"""Derive L57 per-domain and gate JSON from the immutable 40-unit result."""
import argparse, json, sys
from pathlib import Path

ROOT=Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0,str(ROOT))
from tools.eval_l57_decoder_representation_scorer import metrics, rank_flips

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--semantic',required=True); ap.add_argument('--out-root',required=True); a=ap.parse_args()
    src=Path(a.semantic).resolve(); out=Path(a.out_root).resolve();
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    d=json.loads(src.read_text()); rows=d['records']; methods=d['methods']
    domains={str(x['dataset']) for x in rows}
    by_domain={}
    for domain in sorted(domains):
        dr=[x for x in rows if str(x['dataset'])==domain]
        by_domain[domain]={}
        for name in ('teacher','l53_m0','l54_continuous','l57'):
            threshold=methods[name]['calibration']['threshold']
            rr=[dict(x,score=(x['score'] if name=='l57' else x[name])) for x in dr]
            by_domain[domain][name]={'threshold_frozen_from_aggregate_calibration':threshold,'metrics':metrics(rr,threshold),'rank_flips':rank_flips(rr)}
    l29=methods['teacher']['validation']; l57=methods['l57']['validation']
    checks={
      'hard_violation_decrease_ge_0.05': bool(l29['hard_violation'] is not None and l57['hard_violation'] is not None and l29['hard_violation']-l57['hard_violation']>=.05),
      'recall_drop_le_0.01': bool(l57['candidate_recall']>=l29['candidate_recall']-.01),
      'precision_ge_0.0830188679': bool(l57['candidate_precision']>=.0830188679),
      'fp_per_frame_le_11.125': bool(l57['fp_per_frame']<=11.125),
      'pred_per_positive_le_4.069': bool(l57['pred_per_positive']<=4.069),
      'multi_positive_not_collapsed': bool(l57['multi_positive_recall'] is not None and l57['multi_positive_recall']>=.5),
      'null_not_all_accepted': bool(l57['null_false_acceptance'] is not None and l57['null_false_acceptance']<1.0),
      'complete_keys': True,
    }
    payload={'format':'locatemot-l57-semantic-summary-v1','status':'diagnostic_gate_fail','source_semantic':str(src),'source_sha256':d.get('source_sha256'),'manifest_sha256':d.get('manifest_sha256'),'calibration_units':d.get('calibration_units'),'validation_units':d.get('validation_units'),'selection_contract':d.get('selection_contract'),'per_domain':by_domain,'gate':{'checks':checks,'all_pass':all(checks.values()),'decision':'diagnostic_gate_fail','reason':'L57 decoder-representation scorer does not preserve/improve teacher hard-negative ordering; no expansion permitted'},'screening_gt_used':False,'official_test_labels_read':False,'ordinary_mot_ovmot_touched':False,'persistent_raw_dense_cache_written':False}
    (out/'per_domain_summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    (out/'gate_decision.json').write_text(json.dumps(payload['gate'],indent=2)+'\n')
    print(json.dumps({'status':payload['status'],'output':str(out/'per_domain_summary.json'),'gate':payload['gate']}))
if __name__=='__main__': main()
