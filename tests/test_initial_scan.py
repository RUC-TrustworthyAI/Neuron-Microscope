"""Preparation and handoff checks with synthetic channels and a fake labeler only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from common import read, write
from initial_scan import prepare, evidence
from report import source_bundle, merge
from spectrum import build as spectrum
from stage_session import StageSession
from anatomy_semantics import label_request


class InitialScanTests(unittest.TestCase):
    def fixture(self,root):
        report=root/'report';report.mkdir()
        data={'identity':{'id':'L1_N2','layer':1,'neuron':2},'points':{'0':{'document':'d','position':0,'values':[2,2,4]}},
              'documents':{'d':{'tokens':['native'],'product':[4]}},
              'examples':{'top_tokens':['0']},'bands':[]}
        write(report/'report.json',data)
        bundle={'neuron':{'id':'L1_N2','layer':1,'index':2},
                'distribution':{'n_tokens':1000},'claim':{'one_paragraph':'—'},
                'spectrum':None,'anatomy':{'points':[]}}
        page='<script id="bundle" type="application/json">'+json.dumps(bundle)+'</script>'
        page+="<script>const B = JSON.parse(document.getElementById('bundle').textContent);\\nconst POS = 1;</script>"
        (report/'index.html').write_text(page)
        from settings import load
        settings=load()
        config={'stage':'01-interpretation','neuron':{'layer':1,'index':2},
                'session_dir':str(root/'workspace'),'report':str(report),'model':str(root/'model'),
                'settings':settings}
        return report,data,config

    def scan(self,request,bundle,folder):
        folder=Path(folder);path=folder/'anatomy-scan.json'
        if not path.exists():
            edges=bundle['spectrum']['edges'];points=[]
            for band,(a,b) in enumerate(zip(edges,edges[1:])):
                for i in range(40):
                    n=band*40+i;value=(a+b)/2
                    points.append({'id':'scan-'+str(n),'dataset':'synthetic','neuron':'L1_N2',
                        'dataset_token_index':n,'document':'scan-doc','position':n,
                        'token_ids':list(range(n+1)),'gate':1,'up':value,'activation':value,
                        'source':'corpus:scan','source_family':'corpus','cluster':'unlabelled-scan',
                        'peak_token':'anchor','observed_prefix':'FULL PREFIX '+str(n),
                        'text':'local prefix '+str(n)})
            write(path,{'points':points,'edges':edges,'seed':42,'candidate_points_per_bin':40,
                        'scan_seconds':1.25,'tokens_scanned':1000})
        request['sections'].append(str(folder))
        return folder

    def labeler(self,calls):
        def llm(messages,*args,**kwargs):
            obj=json.loads(messages[1]['content']);calls.append(obj)
            self.assertNotIn('OLD_ANSWER',json.dumps(obj))
            self.assertNotIn('OFFICIAL_GROUP',json.dumps(obj))
            groups=obj['semantic_groups']
            return {'new_groups':[] if groups else [
                        {'id':'meaning','name':'Observed meaning','pole':'positive','description':'Synthetic context'}],
                    'labels':[{'id':p['id'],'group':'meaning','reason':'Read the full prefix.'}
                              for p in obj['experimental_points']],
                    'clusters':[{'id':'cluster','name':'Observed contexts','member_groups':['meaning']}]}
        return llm

    def test_scan_is_labelled_before_initial_explanation_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,data,cfg=self.fixture(root);folder=root/'preparation/initial-scan'
            calls=[]
            with patch('anatomy_scan.ensure',self.scan):
                prepare(report,cfg,root/'archive',folder,llm=self.labeler(calls))
            self.assertEqual(sum(len(c['experimental_points']) for c in calls),256)
            self.assertTrue(all(not c['semantic_groups'] for c in calls[:1]))
            saved=read(report/'anatomy-scan-evidence.json')
            self.assertEqual(len(saved['points']),256)
            self.assertEqual(sum(b['selected'] for b in saved['sampling']['bins']),256)
            self.assertEqual({p['semantic_group'] for p in saved['points']},{'meaning'})
            bundle,_=source_bundle((report/'index.html').read_text())
            self.assertEqual(bundle['claim']['one_paragraph'],'—')
            self.assertIsNone(bundle['spectrum'])
            self.assertEqual(bundle['anatomy']['selection']['scan_anchors'],256)
            before=len(calls)
            with patch('anatomy_scan.ensure',self.scan):
                prepare(report,cfg,root/'archive',folder,llm=self.labeler(calls))
            self.assertEqual(len(calls),before)
            cfg['initial_scan']=str(folder)
            session=StageSession(cfg)
            self.assertNotIn('activations',session.config)
            self.assertEqual(len(session.jobs),1)
            with patch.object(session,'prepare',return_value={'contexts_json':{'cases':[]}}):
                index=session.evidence({'section':'index'})
            self.assertIn('anatomy_scan',index['sections'])
            page=session.evidence({'section':'anatomy_scan','offset':7,'limit':2})
            self.assertEqual(page['total'],256)
            self.assertEqual(page['next_offset'],9)
            self.assertTrue(page['cases'][0]['observed_prefix'].startswith('FULL PREFIX'))
            self.assertEqual(page['cases'][0]['semantic_group'],'meaning')
            summary=evidence(report,{'limit':0})
            self.assertEqual(summary['cases'],[])
            self.assertEqual(summary['groups'][0]['id'],'meaning')
            self.assertFalse((report/'host-session.json').exists())
            self.assertFalse((report/'anatomy-semantics.json').exists())

    def test_stage_two_only_labels_new_probe_and_never_rescans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,data,cfg=self.fixture(root);folder=root/'preparation/initial-scan'
            calls=[]
            with patch('anatomy_scan.ensure',self.scan):
                prepare(report,cfg,root/'archive',folder,llm=self.labeler(calls))
            numeric=spectrum(data,[],{})
            numeric['groups']=read(folder/'anatomy-semantics.json')['groups']
            page,_=merge((report/'index.html').read_text(),{'spectrum':numeric})
            (report/'index.html').write_text(page)
            experiment=root/'experiment';experiment.mkdir()
            write(experiment/'sections.json',{})
            write(experiment/'figure.json',{'id':'sweep','title':'Measured probe'})
            write(experiment/'conditions.json',[{'status':'complete','text':'new prefix',
                'token_ids':[9],'selected':{'position':0,'token':'new prefix','offset':[0,10],
                                          'gate':2,'value':3,'product':6}}])
            request={'report':str(report),'sections':[str(folder),str(experiment)],'settings':cfg['settings'],
                     'activations':str(root/'archive')}
            fresh=[]
            with patch('anatomy_scan.ensure',side_effect=AssertionError('Stage two must not scan')):
                result=label_request(request,root/'new-labels',llm=self.labeler(fresh),include_scan=False)
            self.assertIsNotNone(result)
            self.assertEqual(len(fresh),1)
            self.assertEqual(len(fresh[0]['experimental_points']),1)
            self.assertTrue(fresh[0]['experimental_points'][0]['id'].startswith('probe-'))
            stats=read(root/'new-labels/result.json')
            self.assertEqual(stats['n_retained'],256)
            self.assertEqual(stats['n_requested'],1)
            self.assertEqual(stats['n_scan_points'],256)

    def test_initial_report_render_preserves_scan_data_and_labels(self):
        from report import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,data,cfg=self.fixture(root);folder=root/'preparation/initial-scan'
            with patch('anatomy_scan.ensure',self.scan):
                prepare(report,cfg,root/'archive',folder,llm=self.labeler([]))
            output=root/'rendered';output.mkdir()
            request={'report':str(report),'model':cfg['model'],'settings':cfg['settings'],
                     'stage':'01-interpretation','sections':[str(folder)]}
            with patch('report.model_config',return_value=(root/'model',{})):
                run(request,output)
            bundle,_=source_bundle((output/'index.html').read_text())
            points=[p for p in bundle['anatomy']['points'] if p.get('source')=='corpus:scan']
            self.assertEqual(len(points),256)
            self.assertEqual({p['semantic_group'] for p in points},{'meaning'})
            self.assertEqual(read(output/'anatomy-scan-evidence.json'),
                             read(report/'anatomy-scan-evidence.json'))
            self.assertTrue((output/'artifacts/sections-1/anatomy-scan.json').exists())
            self.assertTrue((output/'artifacts/sections-1/anatomy-semantics.json').exists())

    def test_preparation_cost_is_charged_to_preparation_row(self):
        from report_costs import collect
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);call=root/'logs/api/0001';call.mkdir(parents=True)
            write(root/'manifest.json',{'units':['L1_N2']})
            write(call/'attempt.json',{'client_route':'L1_N2/prepare-evidence/anatomy-semantics'})
            write(call/'billing.json',{'estimated_usd':0.123})
            costs=collect(root,'L1_N2')
            self.assertEqual(costs['steps'][0]['api_usd'],0.123)
            self.assertEqual(costs['steps'][1]['api_usd'],0)
            self.assertEqual(costs['total_api_requests'],1)


if __name__=='__main__':unittest.main()
