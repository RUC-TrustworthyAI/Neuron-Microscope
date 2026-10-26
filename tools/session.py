"""Session tools: own all paths, queueing, experiment records and report rendering."""
import copy
import contextlib
import collections
import json
import os
import signal
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from common import now, read, write
from gpu_queue import GPUPool
from settings import effective


def default_runner(group, config):
    output=Path(group['output']);pool=GPUPool(config['gpu_pairs'],config['gpu_lock_dir'])
    deadline=config.get('_deadline',time.monotonic()+config.get('queue_seconds',1800)+config.get('experiment_seconds',900))
    needs_gpu=any(j['tool']!='paper_sections' or any(j['request'].get('plan',{}).get(k)
                  for k in ['search','atlas','family']) for j in group['experiments'])
    allocation=pool.acquire(min(config.get('queue_seconds',1800),max(0,deadline-time.monotonic()))) if needs_gpu else contextlib.nullcontext(())
    with allocation as pair:
        write(output/'queue.json',{'stage':'assigned','gpu_pair':pair,'at':now()})
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,pair)),
                   OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        command=[sys.executable,str(Path(__file__).with_name('experiment_worker.py')),
                 str(output/'batch-request.json')]
        with (output/'worker.log').open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True)
            write(output/'worker-receipt.json',{'pid':process.pid,'argv':command,
                                               'gpu_pair':pair,'started_at':now()})
            try:code=process.wait(timeout=min(config.get('experiment_seconds',900),max(1,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                code=process.returncode
                write(output/'failure.json',{'error':'Experiment runtime budget reached',
                                             'automatic_retry':False})
        result=read(output/'batch-result.json') if (output/'batch-result.json').exists() else {'experiments':[]}
        if code:result['worker_exit_code']=code
        return result


def measured_summary(job):
    request=job['request'];out=Path(request['output'])
    row={'id':request['id'],'tool':job['tool'],'title':request.get('title',request['id']),
         'prediction':request.get('prediction','')}
    if not (out/'result.json').exists():
        row.update(status='not_measured',failure=read(out/'failure.json') if (out/'failure.json').exists()
                   else {'reason':'No completed measurement was recorded'})
        return row
    result=read(out/'result.json');row['status']=result.get('status')
    row['wall_seconds']=result.get('wall_seconds')
    if job['tool'] in ['paper_sections','followup_measurements']:
        row.update(sections=result.get('sections',[]),atlas_completed=result.get('atlas_completed'),
                   atlas_expected=result.get('atlas_expected'))
        if job['tool']=='followup_measurements':
            row.update(conditions=result.get('conditions'),new_conditions=result.get('new_conditions'),
                       completed_conditions=result.get('completed_conditions'),registry=result.get('registry'))
        if (out/'sections.json').exists():
            data=read(out/'sections.json')
            row['measured_sections']={k:v for k,v in data.items() if k not in ['routes_matrix','anatomy','family']}
            if job['tool']=='followup_measurements':
                row['measured_sections']={
                    'ablation':{'summary':data.get('ablation',{}).get('summary',{})},
                    'routes':data.get('routes',[]),
                    'circuit':{'n_atlas_graphs':data.get('circuit',{}).get('n_atlas_graphs'),
                               'method':data.get('circuit',{}).get('method')}}
                row['records']=str(out/'registry.json')
            row['anatomy_cases']=data.get('anatomy',{}).get('n')
            row['family_units']=data.get('family',{}).get('family',[])
        return row
    if job['tool']=='replacement_sweep':
        row['conditions']=[]
        for condition in result.get('conditions',[]):
            item={k:v for k,v in condition.items() if k in
                  ['condition','label','text','status','error','error_type','reason']}
            if condition.get('selected'):
                item['selected']={k:v for k,v in condition['selected'].items() if k in
                                  ['token','token_id','position','gate','value','product']}
            row['conditions'].append(item)
    else:
        graph=result.get('graph',{});ablation=result.get('ablation') or {}
        row.update(text=request['text'],method=result.get('method'),edge_receiver_count=graph.get('meta',{}).get('edge_receiver_count'),root=result.get('root'),root_value=result.get('root_value'),
                   root_token=(graph.get('tokens') or [''])[(graph.get('position',-1))] if graph.get('tokens') else None,
                   source_nodes=[{k:v for k,v in node.items() if k in
                       ['layer','feature','ctx_idx','feature_type','influence','activation','gradient','token']}
                       for node in graph.get('nodes',[])[:6]],
                   ablation={k:ablation[k] for k in ['baseline_target_activation','edit','kl',
                       'reference_token','reference_token_id','delta_target_logit','delta_target_prob',
                       'baseline_top','zeroed_top'] if k in ablation})
        for key in ['baseline_top','zeroed_top']:
            if key in row['ablation']:row['ablation'][key]=row['ablation'][key][:5]
    return row


class Session:
    def __init__(self,config,runner=default_runner):
        self.config=dict(config);self.config['settings']=effective(config)
        cfg=self.config['settings']
        self.config.update(seed=cfg['seed'],torch_seed=cfg['torch_seed'],gpu_pairs=cfg['gpu_groups'],
                           runtime={**cfg['runtime'],'devices':list(range(len(cfg['gpu_groups'][0])))},
                           queue_seconds=cfg['queue_seconds'],experiment_seconds=cfg['experiment_seconds'])
        self.root=Path(config['session_dir'])
        self.root.mkdir(parents=True,exist_ok=True);self.runner=runner
        self.lock=threading.Lock()
        self.jobs=read(self.root/'experiment-index.json') if (self.root/'experiment-index.json').exists() else []
        self.serial=len(list((self.root/'experiments').glob('group-*'))) if (self.root/'experiments').exists() else 0

    def prepare(self):
        cache=self.root/'agent-evidence.json'
        if cache.exists():return read(cache)
        from evidence import prepare
        out=self.root/'evidence';out.mkdir(exist_ok=True)
        packet=prepare(Path(self.config['report']),out,self.config.get('tokenizer'))
        write(cache,packet)
        return packet

    def evidence(self,arguments):
        with self.lock:
            packet=self.prepare()
            sections={'distribution':'distribution_json','contexts':'contexts_json',
                      'token_selectivity':'token_selectivity_json','peak_census':'peak_census_json',
                      'gate_value':'gate_value_json','output_effects':'output_effects_json'}
            section=arguments.get('section','index')
            if section=='index':
                cases=packet['contexts_json']['cases']
                answer={'neuron':self.config['neuron'],'sections':list(sections),
                        'context_count':len(cases),
                        'context_groups':dict(collections.Counter(g for c in cases for g in c['groups']))}
            elif section=='contexts':
                data=packet['contexts_json'];cases=data['cases']
                if arguments.get('group'):cases=[c for c in cases if arguments['group'] in c['groups']]
                if arguments.get('ids'):
                    ids=set(map(str,arguments['ids']));cases=[c for c in cases if c['id'] in ids]
                offset=arguments.get('offset',0);limit=arguments.get('limit',4)
                page=cases[offset:offset+limit]
                answer={'section':section,'trace_columns':data['trace_columns'],
                        'total':len(cases),'offset':offset,
                        'next_offset':offset+len(page) if offset+len(page)<len(cases) else None,
                        'cases':page}
            elif section=='all':answer=packet
            else:answer={'section':section,'data':packet[sections[section]]}
            log=self.root/'evidence-reads';log.mkdir(exist_ok=True)
            number=len(list(log.glob('*.json')))+1
            write(log/('%03d.json'%number),{'arguments':arguments,'response':answer,'at':now()})
            return answer

    def run_experiments(self,arguments):
        with self.lock:
            self.serial+=1;folder=self.root/'experiments'/('group-%03d'%self.serial)
            folder.mkdir(parents=True)
            write(folder/'agent-request.json',arguments)
            group={'output':str(folder),'experiments':[]};cfg=self.config['settings']
            for key,name,short in [('sweeps','replacement_sweep','sweep'),('graphs','attribute','graph')]:
                for item in arguments.get(key,[]):
                    number=1+sum(j['tool']==name for j in self.jobs+group['experiments'])
                    identity=short+'-%03d'%number
                    req={k:copy.deepcopy(v) for k,v in item.items() if k not in
                         ['id','model','neuron','output','seed','runtime','limits']}
                    req.update(id=identity,model=self.config['model'],neuron=self.config['neuron'],
                               output=str(folder/identity),seed=cfg['seed'],torch_seed=cfg['torch_seed'],settings=cfg,
                               runtime=self.config['runtime'],
                               limits={'max_input_tokens':cfg['inputs']['max_written_tokens']})
                    if key=='graphs':
                        if req.get('case_id') is not None:
                            from paper_sections import prepare
                            case=prepare({'atlas':{'cases':[req['case_id']]}},Path(self.config['report']))['atlas']['cases'][0]
                            req.update(case)
                            req['text']=req.get('text','')
                            req['root']={**req.get('root',{'kind':'neuron_product'}),'position':case['position']}
                        req['zero_target_neuron']=False
                        req['nodes']={'mlp':cfg['graphs']['mlp_nodes'],'embedding':cfg['graphs']['embedding_nodes']}
                    group['experiments'].append({'tool':name,'request':req})
            plan=copy.deepcopy(arguments.get('paper') or {})
            from experiment_defaults import measured_plan,semantic_cases
            additional=semantic_cases(self.jobs,cfg['followups']['semantic_cases_per_group'])
            if additional:
                plan.setdefault('atlas',{}).setdefault('cases',[]).extend(additional)
            first=not any(j['tool']=='paper_sections' and j['request'].get('plan',{}).get('atlas') and j['request'].get('settings')==cfg for j in self.jobs)
            if self.config.get('full_paper') and first:
                plan=measured_plan(plan,Path(self.config['report']),cfg['seed'],cfg)
            elif plan.get('atlas'):
                known=set()
                for j in self.jobs:
                    path=Path(j['request']['output'])/'atlas-raw.json'
                    if j['tool']=='paper_sections' and j['request'].get('settings')==cfg and path.exists():
                        known.update(str(r['case_id']) for r in read(path)
                                     if r.get('status')=='complete' and r.get('source_start',0)==0 and r.get('case_id') is not None)
                cases=plan['atlas'].get('cases',[])
                fresh=[c for c in cases if str(c if isinstance(c,(str,int)) else c['case_id']) not in known]
                plan['atlas']={**plan['atlas'],**cfg['atlas'],'cases':fresh} if fresh else None
            if plan:
                from paper_sections import prepare
                number=1+sum(j['tool']=='paper_sections' for j in self.jobs)
                identity='sections-%03d'%number
                plan=prepare(plan,Path(self.config['report']))
                req={'id':identity,'plan':plan,'model':self.config['model'],'neuron':self.config['neuron'],
                     'seed':cfg['seed'],'torch_seed':cfg['torch_seed'],'settings':cfg,
                     'output':str(folder/identity),'runtime':self.config['runtime'],
                     'limits':{'max_input_tokens':cfg['inputs']['max_written_tokens']}}
                group['experiments'].append({'tool':'paper_sections','request':req})
            current=group['experiments'][:]
            sources=[{'id':j['request']['id'],'tool':j['tool'],'output':j['request']['output'],'current':j in current,'settings_match':j['request'].get('settings')==cfg}
                     for j in self.jobs+current if j['tool'] in ['replacement_sweep','attribute','paper_sections']]
            if current:
                identity='followups-%03d'%(1+sum(j['tool']=='followup_measurements' for j in self.jobs))
                req={'id':identity,'model':self.config['model'],'neuron':self.config['neuron'],
                     'seed':cfg['seed'],'torch_seed':cfg['torch_seed'],'settings':cfg,
                     'runtime':self.config['runtime'],'output':str(folder/identity),
                     'report':self.config['report'],'sources':sources,
                     'prior_registries':[str(Path(j['request']['output'])/'registry.json')
                                         for j in self.jobs if j['tool']=='followup_measurements']}
                group['experiments'].append({'tool':'followup_measurements','request':req})
            self.jobs.extend(group['experiments'])
            write(folder/'batch-request.json',group)
            write(self.root/'experiment-index.json',self.jobs)
            write(folder/'queue.json',{'stage':'waiting','at':now()})
            try:run=self.runner(group,self.config)
            except Exception as exc:
                run={'error_type':type(exc).__name__,'error':str(exc),'automatic_retry':False}
                write(folder/'failure.json',run)
            result={'experiments':[measured_summary(j) for j in group['experiments']],
                    'execution':{k:run[k] for k in ['wall_seconds','model_loads','worker_exit_code','error'] if k in run}}
            write(folder/'agent-results.json',result)
            return result

    def finish(self,arguments):
        with self.lock:
            from report import run as render
            self.prepare()
            number=1
            while (self.root/('paper' if number==1 else 'paper-%03d'%number)).exists():number+=1
            suffix='' if number==1 else '-%03d'%number
            interpretation=self.root/('interpretation'+suffix);interpretation.mkdir()
            write(interpretation/'output.json',{k:arguments[k] for k in ['claim','criticism','agent_read'] if k in arguments})
            for name in ['input.json','manifest.json']:
                source=self.root/'evidence'/name
                if source.exists():write(interpretation/name,read(source))
            annotations=interpretation/'annotations.json';write(annotations,arguments.get('annotations',{}))
            paper=self.root/('paper'+suffix);paper.mkdir()
            request={'report':self.config['report'],'model':self.config['model'],'settings':self.config['settings'],
                     'stage':self.config.get('stage'),'activations':self.config.get('activations'),
                     'interpretation':str(interpretation),'annotations':str(annotations),
                     'sweeps':[j['request']['output'] for j in self.jobs if j['tool']=='replacement_sweep'
                               and (Path(j['request']['output'])/'figure.json').exists()],
                     'graphs':[j['request']['output'] for j in self.jobs if j['tool']=='attribute'
                               and (Path(j['request']['output'])/'graph.json').exists()]}
            request['sections']=[j['request']['output'] for j in self.jobs
                                 if j['tool'] in ['paper_sections','followup_measurements'] and (Path(j['request']['output'])/'sections.json').exists()]
            if self.config.get('stage')=='02-experiments':
                self.finalize_anatomy(request)
            write(interpretation/'agent-output.json',arguments)
            write(interpretation/'publication-request.json',request)
            result=render(request,paper)
            write(self.root/'latest-paper.json',{'path':str(paper/'index.html')})
            return {'status':result['status'],'report':str(paper.relative_to(self.root)/'index.html'),
                    'experiments':len(request['sweeps'])+len(request['graphs'])}

    def call(self,name,arguments):
        if name=='run_experiments':return self.run_experiments(arguments)
        if name=='finish':return self.finish(arguments)
        if name=='evidence':return self.evidence(arguments)
        raise KeyError(name)


class ToolServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads=True
    def __init__(self,path,session):
        self.session=session
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                call=json.loads(self.rfile.readline())
                try:response=session.call(call['tool'],call.get('arguments',{}))
                except Exception as exc:
                    response={'error_type':type(exc).__name__,'error':str(exc),'automatic_retry':False}
                self.wfile.write((json.dumps(response,ensure_ascii=False)+'\n').encode())
        super().__init__(str(path),Handler)
        os.chmod(path,0o666)
