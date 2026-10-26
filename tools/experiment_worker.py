"""Run a submitted experiment group with one model load. No LLM calls."""
import json
import sys
import time
from pathlib import Path
from common import Runtime, now, read, score_positions, write
from model_tools import sweep, attribute
from paper_sections import run as paper_sections
from followups import run as followup_measurements


def resolve_root(request, runtime):
    root=dict(request['root'])
    if 'token_ids' in request:
        root['position']=request.get('position',root['position'])
    elif root.get('marker') is not None:
        ids=runtime.tokenizer(request['text'],add_special_tokens=False,return_offsets_mapping=True)
        root['position']=score_positions(request['text'],ids['offset_mapping'],(0,0),{
            'location':'marker','text':root['marker'],'occurrence':root.get('occurrence',0),
            'token_offset':root.get('token_offset',-1),'reduction':'single'})[0]
    if root.get('token_text') is not None:
        target = runtime.tokenizer.encode(root['token_text'], add_special_tokens=False)
        root['token_id'] = target[0]
        request['output_token_selection'] = {
            'text':root['token_text'], 'token_ids':target, 'scored_token_index':0}
    request['root'] = root
    return request


def run(group, runtime_loader=Runtime.load, operations=None):
    operations = operations or {'replacement_sweep':sweep,'attribute':attribute,'paper_sections':paper_sections,'followup_measurements':followup_measurements}
    folder = Path(group['output'])
    started = time.monotonic()
    rows = []
    for job in group['experiments']:
        out = Path(job['request']['output'])
        out.mkdir(parents=True, exist_ok=True)
        write(out/'tool-request.json', job['request'])
        rows.append({'id':job['request']['id'],'tool':job['tool'],
                     'output':str(out),'status':'not_run'})
    write(folder/'batch-result.json', {'experiments':rows,'model_loads':0})
    runtime = None
    try:
        pending=[j for j in group['experiments'] if not
                 (group.get('resume') and (Path(j['request']['output'])/'result.json').exists()
                  and read(Path(j['request']['output'])/'result.json').get('status')=='complete')]
        needs_model=[j for j in pending if j['tool']!='paper_sections' or
                     any(j['request'].get('plan',{}).get(k) for k in ['search','atlas','family'])]
        if needs_model:
            runtime = runtime_loader(needs_model[0]['request'])
        for job,row in zip(group['experiments'],rows):
            out = Path(row['output']);request = dict(job['request'])
            if group.get('resume') and (out/'result.json').exists() and read(out/'result.json').get('status')=='complete':
                row.update(status='complete',reused=True,forwards=0,wall_seconds=0)
                continue
            if group.get('resume'):
                import shutil
                prior=[out/n for n in ['result.json','failure.json'] if (out/n).exists()]
                if prior:
                    history=out/'attempt-history'/str(time.time_ns());history.mkdir(parents=True)
                    for p in prior:shutil.copy2(p,history/p.name)
            before = runtime.forwards if runtime else 0
            began = time.monotonic()
            row.update(status='running',started_at=now())
            write(folder/'batch-result.json', {'experiments':rows,'model_loads':int(runtime is not None)})
            try:
                if job['tool']=='attribute':request=resolve_root(request,runtime)
                write(out/'tool-request.json',request)
                result=operations[job['tool']](request,out,runtime=runtime)
                if result.get('telemetry'):
                    result['telemetry']=dict(result['telemetry'])
                    result['telemetry']['forwards']=(runtime.forwards-before) if runtime else 0
                result['wall_seconds']=time.monotonic()-began
                write(out/'result.json',result)
                if result.get('status')=='complete':(out/'failure.json').unlink(missing_ok=True)
                row.update(status=result['status'],forwards=(runtime.forwards-before) if runtime else 0,
                           wall_seconds=result['wall_seconds'])
            except Exception as exc:
                failure={'status':'failed','error_type':type(exc).__name__,
                         'error':str(exc),'automatic_retry':False,
                         'forwards':(runtime.forwards-before) if runtime else 0}
                write(out/'failure.json',failure)
                row.update(failure)
                if any(t in str(exc).lower() for t in ['misaligned address','device-side assert','illegal memory access']):
                    raise
            write(folder/'batch-result.json', {'experiments':rows,'model_loads':int(runtime is not None)})
    except Exception as exc:
        write(folder/'failure.json',{'error_type':type(exc).__name__,'error':str(exc),
                                    'automatic_retry':False})
    result={'experiments':rows,'model_loads':int(runtime is not None),
            'wall_seconds':time.monotonic()-started,'finished_at':now()}
    write(folder/'batch-result.json',result)
    return result

if __name__=='__main__':
    run(read(sys.argv[1]))
