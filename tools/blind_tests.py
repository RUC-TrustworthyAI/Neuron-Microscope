"""Isolated explanation-only writers; route prediction is paused."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from common import read,write,now
from settings import load

def call(messages,folder,route,*,max_tokens=393216):
    folder=Path(folder);folder.mkdir(parents=True)
    llm_config=load()['llm']
    request={'model':llm_config['model'],'reasoning_effort':llm_config['reasoning_effort'],
             'max_tokens':llm_config['max_output_tokens'],
             'stream':True,'messages':messages}
    write(folder/'request.json',request)
    # Attribute auxiliary calls to their neuron and stage in parallel batches.
    for parent in folder.resolve().parents:
        config_path=parent/'host-session.json'
        if config_path.is_file():
            config=read(config_path);n=config['neuron']
            route='L'+str(n['layer'])+'_N'+str(n['index'])+'/'+config['stage']+'/'+route
            break
    port=os.environ['NEURON_PAPER_METER_PORT']
    token=os.environ['DEEPSEEK_API_KEY']
    req=urllib.request.Request('http://127.0.0.1:'+port+'/'+route+'/v1/chat/completions',
            data=json.dumps(request).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    content=[];reasoning=[];finish=[];usage=None;api_error=None
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req,timeout=900) as response,(folder/'response.sse').open('wb') as log:
        for line in response:
            log.write(line)
            if not line.startswith(b'data:'):continue
            text=line[5:].strip()
            if text==b'[DONE]':continue
            event=json.loads(text)
            if event.get('usage'):usage=event['usage']
            if event.get('error'):api_error=event['error']
            for choice in event.get('choices',[]):
                if choice.get('finish_reason'):finish.append(choice['finish_reason'])
                delta=choice.get('delta',{})
                content.append(delta.get('content') or '')
                reasoning.append(delta.get('reasoning_content') or '')
    answer=''.join(content);(folder/'output.txt').write_text(answer)
    (folder/'reasoning.txt').write_text(''.join(reasoning))
    write(folder/'completion.json',{'finish_reasons':finish,'usage':usage,
        'content_characters':len(answer),'reasoning_characters':sum(map(len,reasoning)),
        'api_error':api_error})
    if not answer.strip():
        reason='; '.join(finish) or 'no finish reason'
        raise RuntimeError('LLM returned no final answer ('+reason+'). See completion.json and reasoning.txt.')
    stripped=answer.strip()
    if stripped.startswith('```'):stripped=stripped.split('\n',1)[1].rsplit('```',1)[0]
    result=json.loads(stripped)
    write(folder/'output.json',result)
    return result

def route_result(prediction,routes,targets,n_heldout,train_counts):
    by_id={str(x['id']):x.get('route') for x in prediction['predictions']}
    ids=[r['id'] for r in routes];unknown=len(ids)
    matrix=[[0]*(len(ids)+1) for _ in range(len(ids)+1)]
    guesses=[by_id.get(str(i)) for i in range(len(targets))]
    for target,guess in zip(targets,guesses):
        matrix[ids.index(target)][ids.index(guess) if guess in ids else unknown]+=1
    majority=max(train_counts,key=train_counts.get)
    return {'n':len(targets),'n_heldout':n_heldout,'n_unassigned':n_heldout-len(targets),
        'n_resolved':sum(g in ids for g in guesses),
        'accuracy':sum(t==g for t,g in zip(targets,guesses))/len(targets),
        'baseline':sum(t==majority for t in targets)/len(targets),
        'baseline_route':majority,'confusion':matrix,'route_ids':ids+['unknown'],
        'route_names':[x.get('name',x['id']) for x in routes]+['Unresolved'],
        'judge':load()['llm']['model'],'k':len(ids),'descriptions':routes}

def retry_route_prediction(source,factors,output,*,tests=None,llm=call):
    """Repeat only the saved text-to-route request, reusing all measured route targets."""
    source,output=Path(source),Path(output)
    output.mkdir(parents=True)
    request=read(source/'route-predictor/request.json')
    targets=read(source/'route-targets.json')
    packet=json.loads(request['messages'][1]['content'])
    tests=json.loads(json.dumps(tests if tests is not None else read(source/'tests.json')))
    prediction=llm(request['messages'],output/'route-predictor','route-predictor')
    factors=read(factors)
    tests['route_prediction']=route_result(prediction,packet['routes'],targets['targets'],
        targets['n_heldout'],{r['id']:r['n_graphs'] for r in factors['routes']})
    tests.setdefault('execution',{})['route_prediction']={'status':'complete'}
    write(output/'route-targets.json',targets)
    write(output/'tests.json',tests)
    write(output/'sections.json',{'tests':tests})
    write(output/'tool-request.json',{'id':output.name,'output':str(output),
        'source':str(source),'only':'route_prediction'})
    write(output/'result.json',{'tool':'paper_sections','status':'complete',
        'sections':['tests'],'new_model_forwards':0,'llm_requests':1})
    return {'tests':tests,'records':str(output)}


def record_tests(session,folder,tests):
    # Keep Tests independent of the GPU section summary, which can be regenerated.
    identity=folder.name
    write(folder/'sections.json',{'tests':tests})
    write(folder/'tool-request.json',{'id':identity,'output':str(folder)})
    write(folder/'result.json',{'tool':'paper_sections',
        'status':'complete' if tests.get('blind_max') and tests.get('blind_min') else 'incomplete',
        'sections':['tests']})
    entry={'tool':'paper_sections','request':{'id':identity,'output':str(folder)}}
    if not any(j['request']['id']==identity for j in session.jobs):
        session.jobs.append(entry)
        write(session.root/'experiment-index.json',session.jobs)

def run(session,args,llm=call):
    if args.get('only')=='route_prediction':
        return {'status':'not_run','tests':{'route_prediction':None},
                'reason':'Route prediction is paused.',
                'new_model_forwards':0,'llm_requests':0}
    # This source is the incoming handoff, never caller-supplied examples or the official answer.
    claim=session.prior['interpretation']['claim']
    explanation={k:claim.get(k) for k in ['lede','one_paragraph','negative_pole']}
    frozen=json.dumps(explanation,ensure_ascii=False,sort_keys=True)
    version=hashlib.sha256(frozen.encode()).hexdigest()
    cfg=session.config['settings']
    count=cfg['blind']['candidates_per_pole']
    # A normal stage completion and an explicit tool call share these saved measurements.
    for previous in sorted(session.root.glob('blind-tests-*'),reverse=True):
        path=previous/'tests.json'
        if not path.is_file():continue
        saved=read(path)
        complete=all(saved.get(key,{}).get('completed_candidates',0)>=count
                     for key in ['blind_max','blind_min'])
        if saved.get('explanation_version')==version and (complete or args.get('reuse_failed')):
            record_tests(session,previous,saved)
            return {'tests':saved,'records':str(previous.relative_to(session.root)),
                    'reused':True,'new_model_forwards':0,'llm_requests':0}
    serial=max([int(p.name.rsplit('-',1)[-1]) for p in session.root.glob('blind-tests-*')
                if p.name.rsplit('-',1)[-1].isdigit()],default=0)+1
    folder=session.root/('blind-tests-%03d'%serial);folder.mkdir()
    write(folder/'frozen-explanation.json',{'claim':explanation,'sha256':version})
    messages=[{'role':'system','content':'Write texts from a neuron explanation. Return JSON with positive and negative arrays of texts.'},
        {'role':'user','content':json.dumps({'explanation':explanation,'task':
           'Write '+str(count)+' texts for the strongest positive activation and '+str(count)+
           ' for the strongest negative activation. Each text must fit within '+str(cfg['blind']['max_written_tokens'])+' tokens.'},ensure_ascii=False)}]
    statuses={}
    measured_folder=None
    tests={'explanation_version':version,'tested_explanation':explanation}
    try:
        generated=llm(messages,folder/'writer','blind-writer',max_tokens=393216)
        def texts(pole):
            values=generated[pole]
            if isinstance(values,str):values=[values]
            return [x if isinstance(x,str) else x['text'] for x in values[:count]]
        plan={'search':[{'id':'blind-'+p,'name':'Blind '+p,'pole':p,
                         'candidates':texts(p),'controls':[],
                         'score':{'location':'input','reduction':'max' if p=='positive' else 'min'}} for p in ['positive','negative']]}
        answer=session.run_experiments({'paper':plan,'analysis':'Measure the frozen explanation-only writer batch.'})
        job=next(j for j in reversed(session.jobs) if j['tool']=='paper_sections'
                 and any(r.get('id','').startswith('blind-') for r in j['request'].get('plan',{}).get('search',[])))
        measured_folder=Path(job['request']['output'])
        raw=read(measured_folder/'raw.json')
        corpus=job['request']['plan']['_corpus']
        from paper_sections import exemplar
        for region in raw.get('search',[]):
            rows=[x for x in region['candidates'] if x['status']=='complete']
            key='blind_max' if region['pole']=='positive' else 'blind_min'
            if not rows:statuses[key]={'status':'unmeasured','expected':len(region['candidates'])};continue
            chooser=max if key=='blind_max' else min
            best=chooser(rows,key=lambda x:x['selected']['product'])
            achieved=best['selected']['product'];target=corpus['max' if key=='blind_max' else 'min']
            tests[key]={**exemplar(best),'achieved':achieved,'target':target,
                'beat':achieved>target if key=='blind_max' else achieved<target,
                'writer':load()['llm']['model'],'explanation_version':version,
                'all':[{'text':x['text'],'activation':x['selected']['product']} for x in rows],
                'expected_candidates':len(region['candidates']),'completed_candidates':len(rows)}
        statuses['writer']={'status':'complete','requested_candidates_per_pole':count,
            'generated_candidates':{p:len(generated[p]) if isinstance(generated[p],list) else 1 for p in ['positive','negative']}}
    except Exception as e:statuses['writer']={'status':'failed','error':str(e)}

    # Paused independently of atlas/factorization measurements.
    tests['route_prediction']=None
    statuses['route_prediction']={'status':'not_run','reason':'Route prediction is paused.'}
    tests['execution']=statuses
    write(folder/'tests.json',tests)
    if measured_folder:
        # Blind candidates are separate from feedback-driven Activation search.
        write(measured_folder/'sections.json',{'tests':tests})
    record_tests(session,folder,tests)
    return {'tests':tests,'records':str(folder.relative_to(session.root))}
