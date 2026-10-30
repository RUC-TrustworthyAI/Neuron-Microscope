"""CPU-only session tests. No model inference, API, Docker or nvidia-smi calls."""
import copy
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from common import read,write
from session import Session,ToolServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'skills/interpret/scripts'))
from read_evidence import call
from gpu_queue import GPUPool
from experiment_worker import run as run_group
import subprocess

BASE=Path(__file__).resolve().parents[1]
STAGES=[('01-interpretation','interpret'),('02-experiments','experiment'),('03-revision','revise')]

def prepare_batch(root):
    source=Path(root)/'data/reports/L3_N7';source.mkdir(parents=True)
    bundle={'neuron':{'id':'L3_N7','layer':3,'neuron':7,'autointerp_label':'OLD_ANSWER'},
            'distribution':{'n_tokens':100},'bigscan':{'n_docs':2},
            'claim':{'title':'OLD_ANSWER','one_paragraph':'OLD_ANSWER'},
            'criticism':[{'result':'OLD_ANSWER'}],'version':{'history':['OLD_ANSWER']},
            'appendix':{'notes':['OLD_ANSWER']}}
    page='<title>OLD_ANSWER</title><script id="bundle" type="application/json">'+json.dumps(bundle)+'</script>'
    page+="<script>const B = JSON.parse(document.getElementById('bundle').textContent);\nconst POS = 1;</script>"
    page+='<script id="saved-interpretation" type="application/json">{"answer":"OLD_ANSWER"}</script>'
    (source/'index.html').write_text(page)
    from launch_files import INITIAL_DATA_FILES
    for name in INITIAL_DATA_FILES:write(source/name,{'identity':{'id':'L3_N7'},'measurement':'TARGET_ONLY'})
    write(source/'interpretation.json',{'answer':'OLD_ANSWER'})
    other=source.parent/'L9_N9';other.mkdir();write(other/'report.json',{'measurement':'OTHER_NEURON'})
    batch=Path(root)/'batch'
    env={**os.environ,'NEURON_MICROSCOPE_ROOT':str(root),'NEURON_PAPER_PYTHON':sys.executable}
    subprocess.run([str(BASE/'run.sh'),'prepare','--output',str(batch),'--neurons','L3_N7'],env=env,check=True,capture_output=True)
    manifest=read(batch/'manifest.json');manifest['settings']['spectrum_sampling']['enabled']=False
    write(batch/'manifest.json',manifest)
    for stage,skill in STAGES:
        path=batch/'units/L3_N7'/stage/'host-session.json'
        cfg=read(path);cfg['settings']['spectrum_sampling']['enabled']=False;write(path,cfg)
    return batch


def config(root):
    return {'session_dir':str(root),'report':str(root/'source'),'model':str(root/'model'),
            'neuron':{'layer':3,'index':7},'seed':42,'gpu_pairs':[[1,2]],
            'gpu_lock_dir':str(root/'locks')}


def fake_measure(group,cfg):
    for job in group['experiments']:
        q=job['request'];p=Path(q['output']);p.mkdir(parents=True,exist_ok=True)
        write(p/'tool-request.json',q)
        if job['tool']=='replacement_sweep':
            rows=[{'status':'complete','condition':i,'label':v,'selected':{'token':v,
                   'position':0,'gate':1.,'value':2.,'product':2.}} for i,v in enumerate(q['replacements'])]
            write(p/'result.json',{'status':'complete','conditions':rows})
            write(p/'figure.json',{'id':q['id']})
        elif job['tool']=='followup_measurements':
            write(p/'result.json',{'status':'complete','conditions':0,'new_conditions':0,'completed_conditions':0})
            write(p/'sections.json',{})
            write(p/'registry.json',{'conditions':[]})
        else:
            write(p/'result.json',{'status':'complete','root_value':2.,'root':q['root'],
                  'graph':{'tokens':['a'],'position':0,'nodes':[]},'ablation':{'kl':.01}})
            write(p/'graph.json',{'id':q['id']})
    return {'model_loads':1,'wall_seconds':.01}


class SessionTests(unittest.TestCase):
    def test_group_supplies_context_and_returns_direct_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            session=Session(config(Path(tmp)),runner=fake_measure)
            result=session.run_experiments({'analysis':'A hypothesis','sweeps':[
                {'template':'{slot}','replacements':['a','b']},
                {'template':'{slot}','replacements':['c']}],
                'graphs':[{'text':'a','root':{'kind':'neuron_product','position':-1}}]})
            self.assertEqual([x['id'] for x in result['experiments']],['sweep-001','sweep-002','graph-001','followups-001'])
            self.assertEqual(result['experiments'][0]['conditions'][0]['selected']['product'],2.)
            self.assertEqual(result['experiments'][2]['ablation']['kl'],.01)
            for job in session.jobs:
                self.assertEqual(job['request']['neuron'],{'layer':3,'index':7})
                self.assertEqual(job['request']['seed'],42)
                self.assertNotIn('budget_ledger',job['request'])
            self.assertEqual(read(Path(tmp)/'experiments/group-001/agent-request.json')['analysis'],'A hypothesis')

    def test_failure_does_not_create_a_count_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempts=[]
            def failed(group,cfg):
                attempts.append(group)
                raise RuntimeError('A worker could not start')
            session=Session(config(Path(tmp)),runner=failed)
            for i in range(5):
                result=session.run_experiments({'graphs':[{'text':'a'}]})
                self.assertEqual(result['experiments'][0]['status'],'not_measured')
            self.assertEqual(len(attempts),5)

    def test_unix_client_works_without_environment_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tool.sock';session=Session(config(Path(tmp)),runner=fake_measure)
            server=ToolServer(path,session)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                with patch.dict(os.environ,{},clear=True):
                    response=call('run_experiments',{'sweeps':[{'template':'{slot}',
                                  'replacements':['x']}]},str(path))
                self.assertEqual(response['experiments'][0]['status'],'complete')
            finally:server.shutdown();server.server_close()

    def test_finish_uses_measured_artifacts_and_keeps_raw_prose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=Session(config(root),runner=fake_measure)
            session.run_experiments({'sweeps':[{'template':'{slot}','replacements':['a']}],
                                     'graphs':[{'text':'a','root':{'kind':'neuron_product','position':-1}}]})
            seen=[]
            def renderer(request,out):
                seen.append(request)
                (out/'index.html').write_text('report')
                return {'status':'complete'}
            obj={'claim':{'title':'A claim','one_paragraph':'Prose without citation IDs'},
                 'criticism':[{'attack':'A concern','result':'Raw result prose'}],
                 'annotations':{'figures':{'sweep-001':{'caption':'A caption'}}}}
            with patch('report.run',renderer), patch.object(session,'prepare',return_value={}):
                first=session.finish(obj);second=session.finish(obj)
            self.assertEqual(first['report'],'paper/index.html')
            self.assertEqual(second['report'],'paper-002/index.html')
            self.assertEqual(read(root/'interpretation/output.json')['claim'],obj['claim'])
            self.assertEqual(len(seen[0]['sweeps']),1);self.assertEqual(len(seen[0]['graphs']),1)
            self.assertEqual(read(root/'interpretation/annotations.json'),obj['annotations'])

    def test_gpu_queue_waits_and_releases_its_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            states=iter([True,False]);pool=GPUPool([[1,2]],tmp,poll_seconds=0,
                                                active=lambda pair:next(states,False))
            with pool.acquire(timeout=1) as pair:
                self.assertEqual(pair,(1,2))
                second=GPUPool([[1,2]],tmp,poll_seconds=0,active=lambda pair:False)
                with self.assertRaises(TimeoutError):
                    with second.acquire(timeout=0):pass
            with second.acquire(timeout=0) as pair:self.assertEqual(pair,(1,2))

    def test_group_loads_one_model_and_counts_each_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);loads=[];runtime=SimpleNamespace(forwards=0)
            def load(request):loads.append(request);return runtime
            def measure(request,out,runtime):
                runtime.forwards+=len(request['replacements'])
                return {'status':'complete','telemetry':{'forwards':runtime.forwards}}
            experiments=[{'tool':'replacement_sweep','request':{'id':str(i),'output':str(root/str(i)),
                           'replacements':['a','b']}} for i in range(3)]
            result=run_group({'output':str(root),'experiments':experiments},runtime_loader=load,
                             operations={'replacement_sweep':measure})
            self.assertEqual(len(loads),1);self.assertEqual(result['model_loads'],1)
            self.assertEqual([x['forwards'] for x in result['experiments']],[2,2,2])
            self.assertEqual([read(root/str(i)/'result.json')['telemetry']['forwards'] for i in range(3)],[2,2,2])


    def test_preparation_exports_skill_without_reading_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            batch=prepare_batch(root)
            unit=batch/'units/L3_N7'
            for stage,route in STAGES:
                step=unit/stage;task=(step/'control/task.txt').read_text()
                self.assertIn('Use the skill at /skills/'+route+'/SKILL.md.',task)
                self.assertNotIn('Measured evidence',task)
                self.assertFalse((step/'workspace/agent-evidence.json').exists())
                skills=list((step/'control/skills').iterdir())
                self.assertEqual([p.name for p in skills],[route])
                self.assertTrue((skills[0]/'SKILL.md').exists())

    def test_evidence_is_lazy_cached_and_keeps_full_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=Session(config(root));calls=[]
            prefix='A complete prefix. '*500
            cases=[{'id':str(i),'groups':['top_tokens'],'observed_prefix':prefix,
                    'anchor':{'product':i},'prediction_target':'next','trace':[]} for i in range(3)]
            packet={'contexts_json':{'trace_columns':['offset','token','gate','value','product'],'cases':cases},
                    'distribution_json':{'mean':1.},'token_selectivity_json':{},'peak_census_json':{},
                    'gate_value_json':{},'output_effects_json':{}}
            def assembly(report,out,tokenizer=None):
                calls.append(True);write(out/'input.json',packet);write(out/'manifest.json',{});return packet
            with patch('evidence.prepare',side_effect=assembly):
                index=session.call('evidence',{})
                page=session.call('evidence',{'section':'contexts','offset':0,'limit':1})
                chosen=session.call('evidence',{'section':'contexts','ids':['2']})
                distribution=session.call('evidence',{'section':'distribution'})
            self.assertEqual(calls,[True]);self.assertEqual(index['context_count'],3)
            self.assertEqual(page['cases'][0]['observed_prefix'],prefix)
            self.assertEqual(page['next_offset'],1);self.assertEqual(page['total'],3)
            self.assertEqual(chosen['cases'][0]['id'],'2')
            self.assertEqual(distribution['data'],{'mean':1.})
            self.assertEqual(len(list((root/'evidence-reads').glob('*.json'))),4)

    def test_exported_route_scripts_are_standalone_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch=prepare_batch(Path(tmp))
            for script in (batch/'units/L3_N7').glob('*/control/skills/*/scripts/*.py'):
                compile(script.read_text(),str(script),'exec')

    def test_harness_program_is_valid_python_without_executing_it(self):
        compile((BASE/'tools/agent.py').read_text(),'agent.py','exec')

if __name__=='__main__':unittest.main()
