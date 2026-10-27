"""Stage-specific tools and JSON/HTML handoff; isolated tests use the shared API meter."""
import os
import shutil
from pathlib import Path
from common import read,write,now
from session import Session,measured_summary


def link_file(source,target):
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)
    return target


def brief_measurement(row,directory):
    """Handoff overview; full texts, traces and plot arrays stay in their files."""
    keys=('id','tool','title','prediction','status','wall_seconds','failure',
          'method','root','root_value','atlas_completed','atlas_expected',
          'anatomy_cases','new_conditions','completed_conditions')
    brief={k:row[k] for k in keys if k in row}
    brief['path']=directory
    conditions=row.get('conditions')
    if isinstance(conditions,list):
        brief['conditions']=[{k:c[k] for k in ('condition','label','status','selected')
                              if k in c} for c in conditions]
    elif conditions is not None:
        brief['conditions']=conditions
    statistics={}
    for name,value in row.get('measured_sections',{}).items():
        if isinstance(value,dict):
            stats={k:v for k,v in value.items()
                   if v is None or isinstance(v,(str,int,float,bool))}
            if isinstance(value.get('summary'),dict):
                stats['summary']={k:v for k,v in value['summary'].items()
                                  if v is None or isinstance(v,(str,int,float,bool))}
            if stats:statistics[name]=stats
        elif isinstance(value,list):
            statistics[name]={'count':len(value)}
    if statistics:brief['statistics']=statistics
    return brief


class StageSession(Session):
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        self.stage=config['stage']
        self.prior=read(Path(config['handoff'])/'handoff.json') if config.get('handoff') else {}
        if not self.jobs:
            for item in self.prior.get('measurements',[]):
                folder=Path(config['handoff'])/item['path']
                request=read(folder/'tool-request.json')
                request['output']=str(folder)
                self.jobs.append({'tool':item['tool'],'request':request})
            if config.get('reuse_measurements'):
                self.import_measurements()
        if config.get('initial_scan'):
            folder=Path(config['initial_scan']);request=read(folder/'tool-request.json')
            request['output']=str(folder)
            if not any(j['request']['id']==request['id'] for j in self.jobs):
                self.jobs.append({'tool':'paper_sections','request':request})
                write(self.root/'experiment-index.json',self.jobs)

    def import_measurements(self):
        source=Path(self.config['reuse_measurements'])
        entries=read(source/'handoff.json')['measurements']
        dest=self.root/'reused-measurements';dest.mkdir(exist_ok=True)
        for entry in entries:
            original=source/entry['path']
            if self.config.get('reuse_completed_experiments') and any((original/name).exists()
                    for name in ('anatomy-scan.json','anatomy-semantics.json','response-predictions.json')):
                continue
            if any(j['request']['id']==entry['id'] for j in self.jobs):continue
            folder=dest/entry['id']
            if not folder.exists():
                ignore=(shutil.ignore_patterns('frozen-explanation.json','messages.json','request.json',
                    'response.sse','reasoning.txt','output.txt','call','calls','host-session.json')
                    if self.config.get('reuse_completed_experiments') else None)
                shutil.copytree(original,folder,ignore=ignore)
                for name in ['sections.json','result.json']:
                    f=folder/name
                    if f.exists():
                        data=read(f)
                        if isinstance(data.get('tests'),dict):
                            data['tests'].pop('tested_explanation',None)
                            data['tests']['reused_measurement']=True
                        write(f,data)
            request=read(folder/'tool-request.json');request['output']=str(folder)
            self.jobs.append({'tool':entry['tool'],'request':request,'reused':True})
        write(self.root/'experiment-index.json',self.jobs)
        write(dest/'source.json',{'source_handoff':str(source),'new_model_forwards':0,
              'note':'Measured artifacts reused. Prior report and prior claim are not supplied to the agents. Tests retain their original explanation version.'})

    def prepare(self):
        packet=super().prepare()
        if self.stage=='02-experiments' and self.config.get('existing_only') and not self.config.get('reuse_completed_experiments') and not getattr(self,'cpu_ready',False):
            from saved_analysis import prepare
            prepare(self)
            folder=self.root/'saved-analysis'
            request=read(folder/'tool-request.json')
            if not any(j['request']['id']==request['id'] for j in self.jobs):
                self.jobs.append({'tool':'paper_sections','request':request})
                write(self.root/'experiment-index.json',self.jobs)
            self.cpu_ready=True
        return packet

    def prepare_tests(self,arguments):
        from saved_analysis import draft_tests
        return draft_tests(self,arguments)

    def evidence(self,arguments):
        if arguments.get('sections'):
            selected={k:v for k,v in arguments.items() if k not in ('section','sections')}
            return {'sections':{name:self.evidence({**selected,'section':name})
                                for name in arguments['sections']}}
        if arguments.get('section')=='brief':
            from stage_brief import build
            return build(self,arguments)
        if arguments.get('section')=='response_predictions':
            from blind_anatomy import evidence
            return evidence({'report':self.config['report'],
                'sections':[j['request']['output'] for j in self.jobs]},arguments)
        if arguments.get('section')=='routes':
            from semantic_clusters import evidence
            from report import source_bundle
            prior, _ = source_bundle((Path(self.config['report'])/'index.html').read_text())
            answer = evidence([j['request']['output'] for j in self.jobs], arguments, prior)
            folder=self.root/'evidence-reads';folder.mkdir(exist_ok=True)
            write(folder/('%03d.json'%(len(list(folder.glob('*.json')))+1)),
                  {'arguments':arguments,'response':answer,'at':now()})
            return answer
        if arguments.get('section')=='anatomy_scan':
            from initial_scan import evidence
            answer=evidence(self.config['report'],arguments)
            folder=self.root/'evidence-reads';folder.mkdir(exist_ok=True)
            write(folder/('%03d.json'%(len(list(folder.glob('*.json')))+1)),
                  {'arguments':arguments,'response':answer,'at':now()})
            return answer
        if arguments.get('section')=='saved_analysis':
            self.prepare()
            folder=self.root/'saved-analysis'
            return {'analysis':read(folder/'result.json'),
                    'directory':'saved-analysis','raw_correlations':'saved-analysis/correlations.json'}
        if arguments.get('section')=='measurements':
            selected=set(map(str,arguments.get('ids',[])))
            rows=[]
            for job in self.jobs:
                if selected and job['request']['id'] not in selected:continue
                folder=Path(job['request']['output'])
                directory=str(folder)
                mounts=[(self.root,Path('/workspace'))]
                if self.config.get('handoff'):
                    mounts.append((Path(self.config['handoff']),Path('/inputs/handoff')))
                mounts.append((Path(self.config['report']),Path('/inputs/report')
                               if self.stage=='01-interpretation' else Path('/inputs/handoff/report')))
                for source,target in mounts:
                    if folder.is_relative_to(source):
                        directory=str(target/folder.relative_to(source));break
                row=brief_measurement(measured_summary(job),directory)
                row['files']=sorted(p.name for p in folder.glob('*.json'))
                rows.append(row)
            return {'existing_only':self.config.get('existing_only',False),'measurements':rows}
        result=super().evidence(arguments)
        if arguments.get('section','index')=='index':
            result['sections'].append('brief')
            if (Path(self.config['report'])/'anatomy-scan-evidence.json').exists():
                result['sections'].append('anatomy_scan')
            result.update(existing_only=self.config.get('existing_only',False),
                          additional_sections=['measurements','saved_analysis','response_predictions'] if self.stage=='02-experiments' and self.config.get('existing_only',False) else ['measurements','routes','response_predictions'])
        return result

    def label_spectrum(self,arguments):
        from blind_anatomy import prior,apply_spectrum
        predictions=prior({'report':self.config['report'],
                           'sections':[j['request']['output'] for j in self.jobs]})
        if predictions:
            from report import source_bundle
            bundle,_=source_bundle((Path(self.config['report'])/'index.html').read_text())
            apply_spectrum(bundle,predictions[-1]);spectrum=bundle.get('spectrum') or {}
            return {'status':'complete','groups':spectrum.get('groups',[]),
                    'n_requested':len(spectrum.get('points',[])),
                    'explanation_version':predictions[-1]['version'],'reused':True,
                    'llm_requests':0,'new_model_forwards':0}
        from spectrum import run
        return run(self,arguments)

    def finalize_anatomy(self,request):
        from blind_anatomy import finalize
        finalize(self,request)

    def refresh_measured_report(self):
        """Rebuild a completed experiment handoff after resuming its measurements."""
        previous=read(self.root/'stage-result.json')
        payload={**self.prior['interpretation'],
                 'annotations':previous['agent_output'].get('annotations',{})}
        result=super().finish(payload)
        write(self.root/'stage-result.json',{
            'stage':self.stage,
            'interpretation':{k:payload[k] for k in ['claim','criticism','response_rules'] if k in payload},
            'agent_output':previous['agent_output'],
            'report':str(self.root/result['report']),
            'measurements':[measured_summary(j) for j in self.jobs],
            'finished_at':now()})
        return result

    def complete(self,arguments):
        if self.stage=='02-experiments':
            if not self.config.get('existing_only'):
                from experiment_defaults import semantic_cases
                candidates=semantic_cases(self.jobs,self.config['settings']['followups']['semantic_cases_per_group'])
                known=set()
                for job in self.jobs:
                    path=Path(job['request']['output'])/'atlas-raw.json'
                    if job['tool']=='paper_sections' and job['request'].get('settings')==self.config['settings'] and path.exists():
                        known.update(str(r['case_id']) for r in read(path)
                                     if r.get('status')=='complete' and r.get('source_start',0)==0 and r.get('case_id') is not None)
                fresh=[c for c in candidates if c['case_id'] not in known]
                if fresh:self.run_experiments({'analysis':'Measure representative cases from all named semantic groups.',
                                              'paper':{'atlas':{'cases':fresh}}})
            if self.config.get('full_paper') and not self.config.get('existing_only'):
                # Fixed end-of-stage step; completed tests for this frozen explanation are reused.
                self.run_tests({'reuse_failed':True})
            interpretation=self.prior['interpretation']
            annotations=arguments.get('annotations',{})
            if self.config.get('reuse_completed_experiments'):
                saved=read(Path(self.config['reuse_measurements'])/'handoff.json')
                annotations=saved.get('agent_output',{}).get('annotations',{})
                arguments={'annotations':annotations,'reused_measurements':self.config['reuse_measurements']}
            payload={**interpretation,'annotations':annotations}
        else:payload=dict(arguments)
        if self.stage=='03-revision' and self.config.get('full_paper'):
            from paper_reading import run
            try:
                payload['agent_read']=run(self,{**arguments,'_measurements':[measured_summary(j) for j in self.jobs]})
            except Exception as exc:
                write(self.root/'paper-reading-failure.json',{'error':str(exc)})
        result=super().finish(payload)
        write(self.root/'stage-result.json',{
            'stage':self.stage,'interpretation':{k:payload[k] for k in ['claim','criticism','response_rules'] if k in payload},
            'agent_output':arguments,'report':str(self.root/result['report']),
            'measurements':[measured_summary(j) for j in self.jobs], 'finished_at':now()})
        return result

    def run_tests(self,arguments):
        from blind_tests import run
        return run(self,arguments)

    def render_report(self,arguments):
        from report_annotations import render
        with self.lock:return render(self,arguments)

    def call(self,name,arguments):
        routes={
            '01-interpretation':{'evidence':self.evidence,'submit_interpretation':self.complete},
            '02-experiments':{'evidence':self.evidence,'run_experiments':self.run_experiments,
                              'complete_experiments':self.complete,'run_tests':self.run_tests,
                              'label_spectrum':self.label_spectrum,'prepare_tests':self.prepare_tests},
            '03-revision':{'evidence':self.evidence,'finish':self.complete,
                           'render_report':self.render_report}}
        if self.config.get('existing_only') and self.stage=='02-experiments':
            routes[self.stage].pop('run_experiments')
            routes[self.stage].pop('run_tests')
        return routes[self.stage][name](arguments)


def make_handoff(stage_folder):
    stage_folder=Path(stage_folder);workspace=stage_folder/'workspace'
    result=read(workspace/'stage-result.json')
    destination=stage_folder/'handoff';destination.mkdir()
    paper=Path(result['report']).parent
    shutil.copytree(paper,destination/'report',copy_function=link_file)
    jobs=read(workspace/'experiment-index.json') if (workspace/'experiment-index.json').exists() else []
    cfg=read(stage_folder/'host-session.json')
    if cfg.get('handoff'):
        prior=read(Path(cfg['handoff'])/'handoff.json')
        previous=[]
        for entry in prior.get('measurements',[]):
            folder=Path(cfg['handoff'])/entry['path'];request=read(folder/'tool-request.json')
            request['output']=str(folder);previous.append({'tool':entry['tool'],'request':request})
        by_id={j['request']['id']:j for j in previous+jobs};jobs=list(by_id.values())
    measurements=[]
    for job in jobs:
        source=Path(job['request']['output']);identity=job['request']['id']
        relative='measurements/'+identity
        if source.exists():
            shutil.copytree(source,destination/relative,copy_function=link_file)
        else:
            target=destination/relative;target.mkdir(parents=True)
            write(target/'tool-request.json',job['request'])
            write(target/'failure.json',measured_summary(job).get('failure',{}))
        measurements.append({'id':identity,'tool':job['tool'],'path':relative})
    handoff={'neuron':cfg['neuron'],'stage':result['stage'],
             'interpretation':result['interpretation'],'agent_output':result['agent_output'],
             'report':'report/index.html','measurements':measurements,
             'measurement_summary':[brief_measurement(row,'measurements/'+row['id'])
                                    for row in result['measurements']],
             'settings':__import__('settings').agent_settings(cfg.get('settings') or {}),'created_at':now()}
    write(destination/'handoff.json',handoff)
    return destination
