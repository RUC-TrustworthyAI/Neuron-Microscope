"""Saved-data pole annotation and propagation; no API calls."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from group_poles import annotate, packet, cluster_poles, SYSTEM

class GroupPoleTests(unittest.TestCase):
    def example(self):
        return {'groups':[{'id':'mixed','name':'Mixed context','description':'A background group'},
                          {'id':'empty','name':'No saved examples'}],
                'points':[{'id':'1','group':'mixed','activation':2,
                           'peak_index':1,'cut_left':True,
                           'token_activations':[{'token':'before','activation':-1},
                                                {'token':' anchor','activation':2},
                                                {'token':' later','activation':-3}]}]}
    def test_prefix_and_future_are_separate(self):
        evidence=packet(self.example())
        self.assertEqual(evidence['groups'][0]['samples'][0]['observed_prefix'],'before anchor')
        self.assertEqual(evidence['groups'][0]['samples'][0]['following_prediction_target'],' later')
        self.assertEqual(evidence['groups'][1]['samples'],[])
    def test_llm_pole_overrides_positive_median_without_reassignment(self):
        data=self.example(); original=copy.deepcopy(data); calls=[]
        def llm(messages, folder, route, max_tokens):
            calls.append((messages,route))
            return {'groups':[{'id':'mixed','pole':'either','reason':'No directional pattern'},
                              {'id':'empty','pole':'either','reason':'No samples'}]}
        with tempfile.TemporaryDirectory() as folder:
            result=annotate(data,folder,llm)
        self.assertEqual(data,original)
        self.assertEqual(result['groups'][0]['pole'],'either')
        self.assertEqual(result['points'],data['points'])
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0][0]['content'],SYSTEM)
        clusters=[{'id':'mixed','member_groups':['mixed'],'pole':'negative','examples':[]}]
        labelled=cluster_poles(clusters,result)
        self.assertEqual(labelled[0]['pole'],'either')
        self.assertEqual(clusters[0]['pole'],'negative')
    def test_partial_output_keeps_missing_label_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            result=annotate(self.example(),folder,lambda *a,**k:{'groups':[]})
            status=json.loads((Path(folder)/'result.json').read_text())
        self.assertEqual(status['status'],'partial')
        self.assertNotIn('pole',result['groups'][0])

if __name__=='__main__': unittest.main()
