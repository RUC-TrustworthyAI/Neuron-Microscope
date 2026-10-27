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

    def test_scan_remains_private_until_explanation_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,data,cfg=self.fixture(root);folder=root/'preparation/initial-scan'
            before=(report/'index.html').read_bytes();calls=[]
            with patch('anatomy_scan.ensure',self.scan):
                prepare(report,cfg,root/'archive',folder,llm=self.labeler(calls))
            self.assertEqual(calls,[])
            self.assertEqual((report/'index.html').read_bytes(),before)
            self.assertFalse((report/'anatomy-scan-evidence.json').exists())
            saved=read(folder/'selected-scan.json')
            self.assertEqual(len(saved['points']),256)
            self.assertEqual(read(folder/'preparation.json')['llm_requests'],0)
            self.assertTrue(all('semantic_group' not in p for p in saved['points']))

    def test_split_keeps_documents_disjoint_and_seen_documents_in_feedback(self):
        from initial_scan import split_documents
        points=[{'id':str(i),'document':str(i//2)} for i in range(256)]
        splits=split_documents(points,{'0','1'})
        docs={name:{p['document'] for p in points if splits[p['id']]==name}
              for name in ('feedback','final_holdout')}
        self.assertEqual(len([x for x in splits.values() if x=='final_holdout']),128)
        self.assertFalse(docs['feedback'] & docs['final_holdout'])
        self.assertFalse({'0','1'} & docs['final_holdout'])

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
