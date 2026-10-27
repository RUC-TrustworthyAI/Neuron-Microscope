"""Write launch configuration, records and JSON/HTML handoffs for run.sh."""
import json
import os
import shutil
import sys
from pathlib import Path
from common import FRAMEWORK, now, read, write
from settings import load, paths, agent_settings, effective

ROOT=Path(os.environ.get('NEURON_MICROSCOPE_ROOT',FRAMEWORK))


def make_batch(output,stages,units):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'framework-source';snapshot.mkdir()
    # Never copy .git, local settings, credentials, datasets, models or previous runs.
    for name in ('run.sh','settings.json','tools','skills','analysis'):
        source=FRAMEWORK/name
        if source.is_dir():
            shutil.copytree(source,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:shutil.copy2(source,snapshot/name)
    config=load();resolved=paths(config)
    write(snapshot/'settings.json',config)
    (output/'logs/api').mkdir(parents=True)
    (output/'units.txt').write_text(''.join(u+'\n' for u in units))
    budget=os.environ.get('NEURON_PAPER_BUDGET_USD','2')
    budget=None if budget.lower()=='unlimited' else float(budget)
    write(output/'manifest.json',{'created_at':now(),'units':units,
          'stages':[s.split(':') for s in stages.split()],
          'settings':config,'paths':resolved,'gpu_pairs':config['gpu_groups'],
          'start_next_after':os.environ.get('NEURON_PAPER_START_NEXT_AFTER',config.get('execution',{}).get('start_next_after','02-experiments')),'budget_usd':budget,'existing_only':bool(os.environ.get('NEURON_PAPER_EXISTING_MEASUREMENTS')),
          'reuse_measurements':os.environ.get('NEURON_PAPER_EXISTING_MEASUREMENTS') or None,
          'reuse_completed_experiments':bool(os.environ.get('NEURON_PAPER_REUSE_COMPLETED')),
          'image':config.get('execution',{}).get('image','neuron-microscope:local'),
          'agent_wall_seconds':config.get('execution',{}).get('agent_wall_seconds',10800)})


def make_stage(batch,unit,stage,skill):
    batch=Path(batch);step=batch/'units'/unit/stage
    layer,index=map(int,unit.replace('L','').replace('N','').split('_'))
    manifest=read(batch/'manifest.json');settings=manifest['settings'];host=manifest['paths']
    for name in ['control','workspace','logs/agent']:(step/name).mkdir(parents=True)
    reuse=manifest.get('reuse_measurements') or os.environ.get('NEURON_PAPER_EXISTING_MEASUREMENTS')
    if reuse and (Path(reuse)/unit/'handoff.json').is_file():
        reuse=str(Path(reuse)/unit)
    cfg={'session_dir':str(step/'workspace'),'stage':stage,'full_paper':True,
         'existing_only':bool(reuse),
         'reuse_completed_experiments':manifest.get('reuse_completed_experiments',False),
         'reuse_measurements':reuse if stage=='02-experiments' else None,
         'report':str(Path(host['reports'])/unit),
         'model':host['model'],'tokenizer':host['tokenizer'],
         'activations':host['activations'],'state_dir':host['state'],
         'neuron':{'layer':layer,'index':index},'settings':effective({'settings':settings}),
         'seed':settings['seed'],'torch_seed':settings['torch_seed'],
         'runtime':{**settings['runtime'],'devices':list(range(len(settings['gpu_groups'][0])))},
         'gpu_pairs':settings['gpu_groups'],'gpu_lock_dir':str(Path(host['state'])/'gpu-locks'),
         'experiment_seconds':settings['experiment_seconds'],'queue_seconds':settings['queue_seconds']}
    cfg['private_scan']=str(batch/'units'/unit/'01-interpretation/preparation/initial-scan')
    if stage=='01-interpretation':
        cfg['source_report']=cfg['report']
        cfg['report']=str(step/'input/report')
        cfg['preparation_activations']=cfg['activations']
        cfg.pop('activations')
    write(step/'host-session.json',cfg)
    write(step/'control/settings.json',agent_settings(settings))
    write(step/'control/llm.json',{k:v for k,v in settings.get('llm',{}).items() if k in ('provider','model','reasoning_effort','max_output_tokens')})
    shutil.copytree(batch/'framework-source/skills'/skill,step/'control/skills'/skill)
    task={'01-interpretation':'Form an initial explanation of neuron '+unit+' from /inputs/report/index.html and its structured measurement data.',
          '02-experiments':'Design and run experiments for neuron '+unit+' using /inputs/handoff/handoff.json and its HTML.',
          '03-revision':'Revise the explanation, Criticism and figure annotations for neuron '+unit+' using /inputs/handoff/handoff.json and its HTML.'}[stage]
    if cfg['existing_only']:
        if stage=='02-experiments':
            task='Use saved activations, CPU coactivation and saved measurements to complete semantic groups, propose text tests and write the report for neuron '+unit+'. No new model inference is available. Read the existing-only workflow in the skill.'
        elif stage=='03-revision':
            task+=' Use only the supplied measurements and newly labelled spectrum; no new experiments.'
            if cfg['reuse_completed_experiments']:
                task+=' The handoff includes reused measured GPU experiments and blind predictions from the current initial explanation. Use each experiment\'s recorded method and scoring position. Independent text tests retain the explanation version they originally tested.'
    (step/'control/task.txt').write_text(task+' Use the skill at /skills/'+skill+'/SKILL.md.\n')
    entry='reuse_saved.py' if stage=='02-experiments' and cfg['reuse_completed_experiments'] else 'agent.py'
    shutil.copy2(batch/'framework-source/tools'/entry,step/'control/agent.py')
    shutil.copy2(batch/'framework-source/tools/output-policy.yml',step/'control/output-policy.yml')
    shutil.copy2(batch/'framework-source/tools/rpc_client.py',step/'control/rpc_client.py')
    (step/'control/bin').mkdir(exist_ok=True)
    shutil.copy2(batch/'framework-source/tools/grep.sh',step/'control/bin/grep')
    write(step/'status.json',{'stage':'prepared','at':now()})


INITIAL_DATA_FILES=('report.json','dataset-index.json','token-rankings.json','effects.json',
                    'peak-census.json','peak-census-negative.json','peak-census-absolute.json')


def initial_html(source,data=None):
    from report import source_bundle,merge
    bundle,_=source_bundle(source)
    neuron={**bundle['neuron'],'autointerp_label':None,'autointerp_description':None}
    tokens=bundle['distribution']['n_tokens']
    documents=(bundle.get('bigscan') or {}).get('n_docs')
    if documents is None:documents=(bundle.get('local_metadata') or {}).get('population_documents')
    claim={'title':f"{neuron['id']} · {tokens} saved tokens",
           'lede':f'{tokens:,} tokens · {documents:,} documents · saved native gate/value/product.' if documents is not None else 'Saved native gate/value/product.',
           'one_paragraph':'—','negative_pole':None,'what_would_change_my_mind':None,'confidence':None}
    from spectrum import scan_fields
    scan=scan_fields(bundle,data) if data else {}
    anatomy=__import__('copy').deepcopy(bundle.get('anatomy') or {})
    if anatomy.get('strip'):
        for key in ['case_id','source_start','anchor_index']:anatomy['strip'].pop(key,None)
        anatomy['strip']['caption']='Saved native gate, value and product.'
    anatomy['points']=[p for p in anatomy.get('points',[]) if p.get('source_family')!='probe' and p.get('source')!='corpus:scan']
    anatomy['n']=len(anatomy['points'])
    anatomy.pop('selection',None)
    anatomy.pop('groups',None)
    for key in ['clusters','semantic_groups','semantic_labeler']:anatomy.pop(key,None)
    anatomy.pop('quadrants',None)
    anatomy['headline']=[]
    anatomy.pop('pole_strips',None)
    anatomy.pop('headline_source',None)
    anatomy['group_names']={}
    for point in anatomy.get('points',[]):
        point['cluster']='unassigned'
        for key in ['semantic_group','label_source','label_reason','predicted_pole','prediction_reason','explanation_version','evaluation_split']:point.pop(key,None)
    for key in ['prediction_summary','explanation_version','colour_mode']:anatomy.pop(key,None)
    page,_=merge(source,{**scan,'neuron':neuron,'claim':claim,'criticism':[],
        'agent_read':None,'agent_read_full':None,'interpretation_costs':None,'spectrum':None,'clusters':[],'routes':[],'routes_matrix':None,'routes_meta':None,
        'cluster_source':None,'cluster_definitions':[],'section_prose':{},'family':None,'circuit':None,'tests':None,
        'custom':None,'graphs_curated':None,'anatomy':anatomy,
        'headline':{'strip':bundle.get('headline',{}).get('strip')},
        'version':{'n':None,'ts':None,'history':[]},'appendix':{'notes':[],'provenance':[]}})
    return page


def stage_input(folder):
    cfg=read(Path(folder)/'host-session.json')
    source=Path(cfg['source_report']);target=Path(cfg['report'])
    target.mkdir(parents=True,exist_ok=True)
    for name in INITIAL_DATA_FILES:
        if (source/name).is_file():shutil.copyfile(source/name,target/name)
    (target/'index.html').write_text(initial_html((source/'index.html').read_text(),read(source/'report.json')))
    from initial_scan import prepare
    artifact=prepare(target,cfg,Path(cfg['preparation_activations']),
                     Path(folder)/'preparation/initial-scan')
    cfg.pop('initial_scan',None)
    write(Path(folder)/'host-session.json',cfg)


def record(path,status,items):
    obj={'status':status,'at':now()}
    for item in items:
        key,value=item.split('=',1)
        try:value=json.loads(value)
        except ValueError:pass
        obj[key]=value
    write(path,obj)


def main(args):
    op,*a=args
    if op=='make-batch':make_batch(a[0],a[1],a[2:])
    elif op=='make-stage':make_stage(*a)
    elif op=='stage-input':stage_input(a[0])
    elif op=='set-handoff':
        path=Path(a[0])/'manifest.json';cfg=read(path)
        cfg['initial_handoffs']={a[1]:a[2]};write(path,cfg)
    elif op=='initial-handoff':
        print(read(Path(a[0])/'manifest.json').get('initial_handoffs',{}).get(a[1],''))
    elif op=='start-next-after':
        print(read(Path(a[0])/'manifest.json').get('start_next_after',''))
    elif op=='state-dir':
        print(read(Path(a[0])/'manifest.json')['paths']['state'])
    elif op=='stages':
        for stage,skill in read(a[0])['stages']:print(stage+':'+skill)
    elif op=='field':
        value=read(a[0])[a[1]]
        print(value if isinstance(value,str) else json.dumps(value))
    elif op=='record':record(a[0],a[1],a[2:])
    elif op=='attach':
        path=Path(a[0])/'host-session.json';cfg=read(path)
        if a[1]:cfg.update(handoff=a[1],report=str(Path(a[1])/'report'))
        write(path,cfg)
    elif op=='launch':
        write(Path(a[0])/'launch.json',{'container':a[1],'argv':a[2:],'started_at':now()})
    elif op=='handoff':
        from stage_session import make_handoff
        result=make_handoff(a[0]);record(Path(a[0])/'status.json','complete',['handoff='+str(result)])
    elif op=='publish':
        batch=Path(a[0]);unit=a[1];destination=Path(a[2])
        terminal=read(batch/'units'/unit/'PIPELINE_TERMINAL.json')
        source=Path(terminal['handoff'])/'report'
        from comparison import publish as publish_comparison
        references=read(batch/'manifest.json')['paths'].get('references')
        publish_comparison(source,destination,unit,references)
        write(batch/'units'/unit/'published.json',{'source':str(source),'destination':str(destination),
                                     'method':'Scientific content unchanged; display and comparison controls added at publication','at':now()})
        print(destination/'index.html')
    elif op=='finalize-costs':
        from report_costs import finalize
        print(json.dumps(finalize(a[0],a[1],a[2])))
    elif op=='finish-unit':
        folder=Path(a[0]);handoff=Path(a[1]) if len(a)>1 else folder/'03-revision/handoff'
        record(folder/'PIPELINE_TERMINAL.json','complete',[
            'handoff='+str(handoff),'html='+str(handoff/'report/index.html')])
    elif op=='finish-batch':
        batch=Path(a[0]);units=read(batch/'manifest.json')['units']
        states={u:read(batch/'units'/u/'PIPELINE_TERMINAL.json') for u in units}
        write(batch/'BATCH_TERMINAL.json',{'finished_at':now(),'units':states,
              'usage':read(batch/'logs/usage.json')})
    else:raise ValueError(op)

if __name__=='__main__':main(sys.argv[1:])
