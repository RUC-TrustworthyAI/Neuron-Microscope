import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from blind_tests import call, retry_route_prediction

class BlindTests(unittest.TestCase):
    def test_reasoning_only_length_is_saved_and_explained(self):
        events=[{'choices':[{'delta':{'reasoning_content':'thinking'},'finish_reason':None}]},
                {'choices':[{'delta':{},'finish_reason':'length'}],
                 'usage':{'completion_tokens':16384}}]
        stream=io.BytesIO((''.join('data: '+json.dumps(e)+'\n' for e in events)+'data: [DONE]\n').encode())
        class Opener:
            def open(self,request,timeout):return stream
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'NEURON_PAPER_METER_PORT':'1','DEEPSEEK_API_KEY':'test'}),patch('urllib.request.build_opener',return_value=Opener()):
            folder=Path(d)/'request'
            with self.assertRaisesRegex(RuntimeError,'no final answer.*length'):
                call([],folder,'route-predictor')
            self.assertEqual((folder/'reasoning.txt').read_text(),'thinking')
            self.assertEqual(json.loads((folder/'completion.json').read_text())['finish_reasons'],['length'])
            self.assertEqual(json.loads((folder/'request.json').read_text())['max_tokens'],393216)

    def test_route_only_reuses_saved_targets_without_writer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';(source/'route-predictor').mkdir(parents=True)
            messages=[{'role':'system','content':'route task'}, {'role':'user','content':json.dumps({'routes':[{'id':'p1','name':'positive'},{'id':'n1','name':'negative'}],'texts':[{'id':'0','text':'a'},{'id':'1','text':'b'}]})}]
            def put(path,value):path.write_text(json.dumps(value))
            put(source/'route-predictor/request.json',{'messages':messages,'max_tokens':16384})
            put(source/'route-targets.json',{'targets':['p1','n1'],'n_heldout':3,'examples':[]})
            put(source/'tests.json',{'blind_max':{'achieved':7},'execution':{'writer':{'status':'complete'},'route_prediction':{'status':'failed'}}})
            factors=root/'factors.json';put(factors,{'routes':[{'id':'p1','n_graphs':5},{'id':'n1','n_graphs':2}]})
            calls=[]
            def llm(actual,folder,route):
                calls.append(route);self.assertEqual(actual,messages)
                return {'predictions':[{'id':'0','route':'p1'},{'id':'1','route':'n1'}]}
            result=retry_route_prediction(source,factors,root/'result',llm=llm)
            self.assertEqual(calls,['route-predictor'])
            self.assertEqual(result['tests']['blind_max'],{'achieved':7})
            self.assertEqual(result['tests']['route_prediction']['accuracy'],1.)
            self.assertEqual(result['tests']['route_prediction']['n_unassigned'],1)
            self.assertEqual(result['tests']['execution']['writer']['status'],'complete')

if __name__=='__main__':unittest.main()
