"""Preparation and handoff checks with synthetic channels and a fake labeler only."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from common import read, write
from initial_scan import prepare


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

    def test_scan_remains_private_until_explanation_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,data,cfg=self.fixture(root);folder=root/'preparation/initial-scan'
            before=(report/'index.html').read_bytes();calls=[]
            sampled=root/'preparation/spectrum-sample';sampled.mkdir(parents=True)
            write(sampled/'sections.json',{'spectrum':{'points':[{'id':str(i)} for i in range(500)]}})
            write(sampled/'sampling.json',{'started_at':'2026-01-01T00:00:00+00:00',
                'finished_at':'2026-01-01T00:00:01+00:00','wall_seconds':1,'sampled_points':500,
                'llm_requests':0,'new_model_forwards':0,'gpu_seconds':0})
            with patch('spectrum_sampling.prepare',return_value=sampled):
                prepare(report,cfg,root/'archive',folder,llm=lambda *args:calls.append(args))
            self.assertEqual(calls,[])
            self.assertEqual((report/'index.html').read_bytes(),before)
            self.assertFalse((report/'anatomy-scan-evidence.json').exists())
            saved=read(sampled/'sections.json')['spectrum']
            self.assertEqual(len(saved['points']),500)
            self.assertFalse((folder/'selected-scan.json').exists())
            self.assertEqual(read(folder/'preparation.json')['llm_requests'],0)
            self.assertTrue(all('semantic_group' not in p for p in saved['points']))

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
