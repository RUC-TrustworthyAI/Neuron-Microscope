"""Synthetic fixtures: semantic regrouping must retain measured members and numbers."""
import copy
from pathlib import Path
import tempfile
import unittest
from common import write
from semantic_clusters import apply, evidence


class SemanticClusterTests(unittest.TestCase):
    def fixture(self, folder):
        rows=[]
        for identity, value in [('a',10.),('b',.2),('c',-3.),('d',1.)]:
            rows.append({'id':identity,'status':'complete','case_id':identity,
                         'text':'Synthetic context '+identity,
                         'observed_prefix':'Synthetic context '+identity,
                         'selected':{'position':0,'token':identity,'product':value},
                         'trace':[{'token':identity,'product':value}],
                         'upstream':[{'layer':0,'neuron':1,'weight':value}]})
        write(folder/'atlas-raw.json',rows)
        write(folder/'route-factors.json',{'routes':[
            {'id':'p1','pole':1,'graph_ids':['a','b'],'n_graphs':2},
            {'id':'n1','pole':-1,'graph_ids':['c','d'],'n_graphs':2}],
            'routes_matrix':{}})
        return rows

    def test_merge_split_and_unassigned_keep_every_measured_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            raw=(folder/'route-factors.json').read_bytes()
            definitions=[{'id':'mixed','name':'Mixed contexts','pole':'either',
                          'graph_ids':['a','c'],'example_graph_ids':['c','a']},
                         {'id':'weak','name':'Weak context','pole':'either',
                          'graph_ids':['b'],'example_graph_ids':['b']}]
            bundle={};apply(bundle,definitions,[folder])
            self.assertEqual([g['graph_ids'] for g in bundle['clusters']],
                             [['a','c'],['b'],['d']])
            self.assertEqual(bundle['clusters'][0]['member_routes'],['p1','n1'])
            self.assertEqual(bundle['clusters'][0]['examples'][0]['activation'],-3.)
            self.assertEqual(bundle['clusters'][1]['examples'][0]['activation'],.2)
            self.assertAlmostEqual(sum(g['share'] for g in bundle['clusters']),1.)
            self.assertEqual((folder/'route-factors.json').read_bytes(),raw)
            original=copy.deepcopy(bundle)
            apply(bundle,bundle['cluster_definitions'],[folder])
            self.assertEqual(bundle,original)

    def test_evidence_resolves_graph_ids_to_full_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rows=self.fixture(folder)
            index=evidence([folder],{})
            self.assertEqual(index['n_members'],4)
            self.assertEqual([r['graph_id'] for r in evidence([folder],{'group':'p1'})['members']],['a','b'])
            member=evidence([folder],{'ids':['b']})['members'][0]
            self.assertEqual(member['observed_prefix'],rows[1]['observed_prefix'])
            self.assertEqual(member['selected']['product'],.2)


if __name__=='__main__':unittest.main()
