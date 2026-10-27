import json
import math
import tempfile
import unittest
from pathlib import Path
from measured_response import response_pole, describe_group, mixed_batches
from anatomy_semantics import annotate, apply, probe_key


class MeasuredResponseTests(unittest.TestCase):
    def test_native_activation_sign_in_all_four_quadrants(self):
        values = [g/(1+math.exp(-g))*u for g,u in [(2,3),(2,-3),(-2,3),(-2,-3)]]
        self.assertEqual(list(map(response_pole,values)),['positive','negative','negative','positive'])
        self.assertEqual(response_pole(0),'zero')
        self.assertEqual(response_pole(None),'unknown')
        group=describe_group({'id':'x','pole':'negative'},[0.1,300,-50])
        self.assertEqual(group['pole'],'either')
        self.assertEqual(group['response_summary']['positive'],2)

    def test_each_scan_batch_spans_response_range_without_losing_points(self):
        points=[{'id':str(i),'activation':i-128} for i in range(256)]
        batches=mixed_batches(points,32)
        self.assertEqual(len(batches),8)
        self.assertEqual(sorted(p['id'] for b in batches for p in b),sorted(p['id'] for p in points))
        for batch in batches:
            self.assertEqual(len(batch),32)
            self.assertLess(min(p['activation'] for p in batch),-100)
            self.assertGreater(max(p['activation'] for p in batch),100)

    def test_later_batch_can_revise_definition_and_previous_assignment(self):
        points=[{'id':str(i),'activation':v,'source':'corpus:scan','dataset':'test',
                 'neuron':'L1_N2','dataset_token_index':i,'gate':2,'up':v/2,
                 'position':0,'text':'context '+str(i),'peak_token':'anchor','cluster':'unlabelled'}
                for i,v in enumerate([-20,-1,1,300])]
        bundle={'spectrum':{'groups':[]},'anatomy':{'selection':{},'points':points}}
        calls=[]
        def llm(messages,*args,**kwargs):
            evidence=json.loads(messages[1]['content']);calls.append(evidence)
            if len(calls)==1:
                return {'new_groups':[{'id':'broad','name':'Broad','description':'Topic','pole':'negative'}],
                    'labels':[{'id':p['id'],'group':'broad'} for p in evidence['experimental_points']]}
            return {'group_updates':[{'id':'broad','name':'Revised','description':'Narrowed context'}],
                    'new_groups':[{'id':'different','name':'Different context'}],
                    'labels':[{'id':p['id'],'group':'broad'} for p in evidence['experimental_points']]+
                             [{'id':'2','group':'different'}],
                    'clusters':[{'id':'c','name':'Contexts','member_groups':['broad','different']}]}
        with tempfile.TemporaryDirectory() as folder:
            result=annotate(bundle,Path(folder),llm=llm,batch_size=2)
        self.assertNotIn('pole',result['groups'][0])
        self.assertEqual(result['groups'][0]['name'],'Revised')
        self.assertEqual(result['assignments'][probe_key(points[2])]['group'],'different')
        self.assertIn('negative', {p['anchor']['response_pole'] for p in calls[0]['experimental_points']})
        apply(bundle,result)
        self.assertEqual([p['activation'] for p in points],[-20,-1,1,300])
        self.assertEqual([p['response_pole'] for p in points],['negative','negative','positive','positive'])
        self.assertEqual(bundle['anatomy']['clusters'][0]['response_summary']['positive'],2)


if __name__ == '__main__': unittest.main()
