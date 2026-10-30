"""Blocking CLI for host tools; the model continues after the tool completes."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

WORKDIR=Path(os.environ.get('NEURON_PAPER_WORKSPACE','/workspace'))
SOCKET=os.environ.get('NEURON_PAPER_SOCKET','/run/neuron-paper/tool.sock')

LONG={'run_experiments','label_spectrum','run_tests','prepare_tests','complete_experiments','finish'}

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False)+'\n');tmp.replace(p)

def call(tool,arguments,path=None):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as c:
        c.connect(path or SOCKET)
        c.sendall((json.dumps({'tool':tool,'arguments':arguments})+'\n').encode())
        with c.makefile('rb') as stream:return json.loads(stream.readline())

def recover(tool,args):
    root=WORKDIR
    if tool=='label_spectrum' and args.get('groups') and not args.get('resume_from'):
        for d in sorted(root.glob('spectrum-*')):
            if (d/'groups.json').exists() and (d/'result.json').exists():
                saved=read(d/'groups.json')
                keys=set().union(*(g.keys() for g in args['groups']))
                compact=lambda gs:[{k:g[k] for k in keys if k in g} for g in gs]
                if compact(saved)==compact(args['groups']):
                    return d/'result.json'
    if tool=='run_experiments':
        for d in sorted((root/'experiments').glob('group-*')):
            p=d/'agent-request.json'
            if p.exists() and read(p)==args:return d/'agent-results.json'
    return None

def response(job):
    if (job/'response.json').exists():return read(job/'response.json')
    state=read(job/'state.json') if (job/'state.json').exists() else {}
    result=Path(state['response_path']) if state.get('response_path') else None
    if result and result.exists():return read(result)
    return {'status':'running','job':str(job),
        'wait_command':'python '+str(Path(__file__).resolve())+' --wait '+str(job)}

def submit(tool,args):
    key=hashlib.sha256(json.dumps([tool,args],ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:24]
    root=WORKDIR/'.tool-jobs';root.mkdir(exist_ok=True)
    job=root/(tool+'-'+key)
    try:job.mkdir()
    except FileExistsError:return response(job)
    write(job/'request.json',{'tool':tool,'arguments':args})
    recovered=recover(tool,args)
    if recovered:
        write(job/'state.json',{'response_path':str(recovered),'recovered_existing_operation':True})
        return response(job)
    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(job)],
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
        start_new_session=True)
    write(job/'state.json',{'pid':p.pid,'submitted_at':time.time()})
    return response(job)

TOOL_DESCRIPTIONS={
    'evidence':'Read an evidence brief, complete sections or saved cases for the target neuron.',
    'submit_interpretation':'Save the initial explanation and render its HTML.',
    'label_spectrum':'Label saved activation samples with semantic groups.',
    'run_experiments':'Run an experiment group and wait for its recorded result.',
    'run_tests':'Run the independent writer tests and wait for the result.',
    'prepare_tests':'Save test proposals without model inference.',
    'complete_experiments':'Complete the experiment handoff and render its HTML.',
    'finish':'Submit the final explanation, run its independent review and response predictions, and render its HTML.',
    'render_report':'Update saved report captions, section introductions and figure selections without API or model calls.'
}

def _arguments(tool):
    import argparse
    parser=argparse.ArgumentParser(description=TOOL_DESCRIPTIONS.get(tool,'Call a neuron-report host tool.'),
        epilog='Pass a JSON object through stdin, or use --input FILE. Tools wait for completion.')
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--input',metavar='FILE',help='Read JSON from FILE; use - for redirected stdin.')
    source.add_argument('--worker',metavar='JOB_DIR',help=argparse.SUPPRESS)
    source.add_argument('--wait',metavar='JOB_DIR',help=argparse.SUPPRESS)
    if tool=='evidence':
        sections=parser.add_mutually_exclusive_group()
        sections.add_argument('--section',help='Evidence section, for example brief, index or contexts.')
        sections.add_argument('--sections',nargs='+',help='Read several complete evidence sections in one call.')
        parser.add_argument('--group',help='Context sample group.')
        parser.add_argument('--ids',nargs='+',help='Saved case IDs.')
        parser.add_argument('--offset',type=int,help='First sample offset.')
        parser.add_argument('--limit',type=int,help='Number of samples.')
    args=parser.parse_args()
    if args.worker or args.wait:
        if tool=='evidence' and any(getattr(args,k) is not None for k in ['section','sections','group','ids','offset','limit']):
            parser.error('Evidence flags cannot be combined with a legacy job command.')
        return args,None
    fields={k:getattr(args,k) for k in ['section','sections','group','ids','offset','limit']
            if tool=='evidence' and getattr(args,k) is not None}
    if fields:
        if args.input is not None:parser.error('Use evidence flags or JSON input, not both.')
        if 'sections' not in fields:fields.setdefault('section','index')
        return args,fields
    if args.input is None and sys.stdin.isatty() and tool=='evidence':
        return args,{'section':'index'}
    if args.input in [None,'-'] and sys.stdin.isatty():
        parser.error('JSON input is required. Redirect stdin or use --input FILE.')
    try:
        if args.input not in [None,'-']:
            with Path(args.input).open() as stream:payload=json.load(stream)
        else:
            payload=json.load(sys.stdin)
    except (OSError,ValueError) as exc:
        parser.error('Cannot read JSON input: '+str(exc))
    if not isinstance(payload,dict):parser.error('JSON input must be an object.')
    return args,payload

def main(tool=None):
    options,args=_arguments(tool)
    if options.worker:
        job=Path(options.worker);request=read(job/'request.json')
        try:answer=call(request['tool'],request['arguments'])
        except Exception as e:answer={'error_type':type(e).__name__,'error':str(e)}
        write(job/'response.json',answer)
        return
    if options.wait:
        job=Path(options.wait)
        while True:
            answer=response(job)
            if answer.get('status')!='running':break
            time.sleep(.25)
        print(json.dumps(answer,ensure_ascii=False))
        return
    prior=recover(tool,args) if tool in LONG else None
    answer=read(prior) if prior and prior.exists() else call(tool,args)
    print(json.dumps(answer,ensure_ascii=False))

if __name__=='__main__':main()
