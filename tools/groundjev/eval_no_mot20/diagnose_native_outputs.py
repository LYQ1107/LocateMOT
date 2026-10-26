"""Explain persisted native parser failures without changing any prediction."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,load,save


def parser_module():
    path=ROOT/'external/GroundAnything/eval/utils/prompt_mode.py'
    spec=importlib.util.spec_from_file_location('_native_format_diagnostic',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module);return module


def explain(text,official):
    candidate=official._native_answer_slice(text)
    matches=list(official._GAM_SCORING_ENTRY_RE.finditer(candidate))
    if not matches:return 'missing_complete_native_wrappers'
    if ', '.join(m.group(0) for m in matches)!=candidate:
        return 'malformed_wrappers_or_entry_separators'
    for match in matches:
        try:label=official.canonicalize_phrase(match.group(1))
        except Exception:return 'invalid_native_label'
        if label!=match.group(1):return 'noncanonical_native_label'
        payload=match.group(2)
        if payload=='None':
            # A strict-invalid response falls back to the official deduplication
            # parser, which does not accept a None component alongside boxes.
            return 'strict_failure_then_reference_fallback_rejects_None_entry'
        for component in payload.split(','):
            values=official._GAM_SCORING_COORD_RE.findall(component)
            if not values or ''.join(f'<{v}>' for v in values)!=component:
                return 'malformed_coordinate_payload'
            if not official._validate_foreign_tuple(tuple(map(int,values)),official.TaskType.BBOX,official.GRID_MAX):
                return 'invalid_native_box_geometry'
    return 'reference_parser_should_accept_complete_deduplicated_boxes'


def diagnose(run=RUN,tokenizer=None):
    official=parser_module();results=[]
    for entry in load(run/'jobs.json')['jobs']:
        if entry['dataset']!='tao_amodal':continue
        job=Path(entry['path']);path=job/'detector.sqlite'
        if not path.exists():continue
        db=sqlite3.connect(f'file:{path}?mode=ro',uri=True,timeout=30)
        statuses=Counter();failures=Counter();strict=Counter();seconds=characters=tokens=0
        usage_tokens=0;usage_requests=0;examples=[]
        for frame,chunk,blob in db.execute('SELECT frame,chunk,data FROM requests ORDER BY frame,chunk'):
            row=json.loads(gzip.decompress(blob));meta=row['metadata'];result=meta['result'];text=result['raw_output']
            statuses[row['status']]+=1;seconds+=row['seconds'];characters+=len(text)
            if tokenizer is not None:tokens+=len(tokenizer.encode(text,add_special_tokens=False))
            usage=meta.get('response_usage') or {}
            if usage.get('completion_tokens') is not None:
                usage_tokens+=usage['completion_tokens'];usage_requests+=1
            strict[str(result.get('parse_error'))]+=1
            if row['status']=='REJECTED':
                reason=explain(text,official);failures[reason]+=1
                if len(examples)<4:examples.append({'frame':frame,'chunk':chunk,'reason':reason,
                    'strict_error':result.get('parse_error'),'response_prefix':text[:240]})
        db.close();count=sum(statuses.values())
        if not count:continue
        results.append({'job':entry['id'],'requests':count,'status_counts':dict(statuses),
            'reference_parser_failure_categories':dict(failures),'strict_parse_errors':dict(strict),
            'request_seconds':seconds,'mean_request_seconds':seconds/count,'mean_output_characters':characters/count,
            'retokenized_output_tokens':tokens if tokenizer is not None else None,
            'retokenized_tokens_per_wall_second':tokens/seconds if tokenizer is not None and seconds else None,
            'server_completion_tokens':usage_tokens,'requests_with_server_usage':usage_requests,'examples':examples})
    report={'status':'PERSISTED_NATIVE_OUTPUT_DIAGNOSIS','jobs':results,'GT_read':False,
        'predictions_changed':False,'protocol_changed':False,
        'retokenized_speed_scope':'response retokenization divided by full request wall time, including prefill/transport; not server decode-only TPS'}
    save(run/'native_output_diagnosis.json',report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RUN)
    parser.add_argument('--tokenizer',type=Path);args=parser.parse_args();tokenizer=None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,trust_remote_code=True,local_files_only=True)
    report=diagnose(args.root,tokenizer)
    print(json.dumps({'status':report['status'],'jobs':[{k:v for k,v in row.items() if k not in ('examples','strict_parse_errors')}
          for row in report['jobs']]}),flush=True)


if __name__=='__main__':main()
