import json
import tempfile
import unittest
from pathlib import Path

from spectrum_sampling import allocate, save_contexts, register_contexts
from blind_anatomy import apply_spectrum, frozen_explanation, point_identity, prediction_points, predict


class SpectrumSamplingTests(unittest.TestCase):
    def test_sampled_context_enters_experiments_with_exact_ids_and_anchor(self):
        import numpy as np
        from common import write, read
        from paper_sections import prepare
        from experiment_defaults import measured_plan
        from settings import load
        class Dataset:
            def document_activations(self, document, neuron):
                self.request = (document, neuron)
                return {'token_ids': np.array([12, 13, 14, 15]),
                        'activations': np.array([[1,2,2],[2,3,6],[3,4,12],[4,5,20]])}
        class Tokenizer:
            def decode(self, ids, **kwargs):
                return ''.join(map(str, ids))
        point={'id':'spectrum-424573007','document':'77','position':2,
               'gate':3,'up':4,'activation':12}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sample=root/'sample';sample.mkdir()
            api=Dataset();save_contexts([point],api,'L1_N2',Tokenizer(),sample)
            data={'points':{},'documents':{},'examples':{},'bands':[],
                  'statistics':{'product':{'max':20,'min':-3,'count':4}}}
            register_contexts(data,[sample]);write(root/'report.json',data)
            plan=measured_plan({'atlas':{'cases':[point['id']]}},root,settings=load())
            resolved=prepare(plan,root)['atlas']['cases'][0]
        self.assertEqual(api.request,(77,'L1_N2'))
        self.assertEqual(resolved['case_id'],point['id'])
        self.assertEqual(resolved['token_ids'],[12,13,14])
        self.assertEqual(resolved['position'],2)
        self.assertEqual(resolved['following_token_id'],15)
        self.assertEqual(resolved['original_product'],12)
        self.assertEqual(resolved['score'],{'location':'token_position','position':2,'reduction':'single'})

    def test_allocates_by_log_height_and_keeps_exact_total(self):
        self.assertEqual(allocate([100, 10000, 1000000]), [83, 167, 250])
        self.assertEqual(allocate([100, 100, 100], 5), [2, 2, 1])

    def test_respects_population_capacity_and_zero_height(self):
        self.assertEqual(allocate([0, 1, 100, 100]), [0, 0, 100, 100])
        self.assertEqual(allocate([2, 1000000]), [2, 498])
        self.assertEqual(allocate([0, 1]), [0, 0])

    def test_corpus_context_shared_by_two_views_keeps_one_prediction(self):
        anchor = {'id': 'saved', 'document': 'd', 'position': 7,
                  'source_family': 'corpus', 'source': 'corpus:selected', 'activation': -6}
        sample = {**anchor, 'id': 'spectrum-77', 'source': 'saved natural-corpus token',
                  'observed_prefix': 'Complete earlier context', 'bin': 0, 'group': 'unlabelled'}
        bundle = {'anatomy': {'points': [anchor]}, 'spectrum': {
            'points': [sample], 'bins': [{'n_labeled': 1, 'groups': {}}]}}
        points = prediction_points(bundle, {'points': {}})
        self.assertEqual([p['id'] for p in points], ['saved'])
        rule = {'id': 'g', 'name': 'Context', 'predicted_pole': 'positive'}
        saved = {'version': 'v', 'response_rules': [rule], 'records': [
            {'id': 'saved', 'identity': point_identity(anchor), 'group': 'g',
             'predicted_pole': 'positive', 'reason': 'Earlier semantic cue.'}]}
        apply_spectrum(bundle, saved)
        self.assertEqual(sample['group'], 'g')
        self.assertEqual(sample['predicted_pole'], 'positive')
        self.assertEqual(sample['activation'], -6)
        self.assertEqual(bundle['spectrum']['n_assigned'], 1)

    def test_new_spectrum_prefix_enters_the_fixed_predictor(self):
        sample = {'id': 'spectrum-77', 'document': 'new-document', 'position': 7,
                  'source_family': 'corpus', 'source': 'saved natural-corpus token',
                  'observed_prefix': 'Complete earlier context', 'activation': 4}
        bundle = {'spectrum': {'points': [sample]}}
        points = prediction_points(bundle, {'points': {}})
        self.assertEqual(points, [sample])
        self.assertIsNot(points[0], sample)

    def test_shared_500_points_use_two_batches_once_for_both_views(self):
        def point(i):
            return {'id':str(i),'document':str(i),'position':0,'activation':1,
                    'source_family':'corpus','source':'saved natural-corpus token',
                    'observed_prefix':'Context '+str(i),'peak_token':'anchor'}
        inside=[point(i) for i in range(500)]
        spectrum=[{**p,'id':'spectrum-'+p['id']} for p in inside]
        bundle={'anatomy':{'points':inside},'spectrum':{'points':spectrum}}
        points=prediction_points(bundle,{'points':{}})
        frozen=frozen_explanation({'claim':{'one_paragraph':'A context neuron.'},
            'response_rules':[{'id':'g','name':'Context','predicted_pole':'positive'}]})
        calls=[]
        def llm(messages,*args):
            rows=json.loads(messages[1]['content'])['points'];calls.append(rows)
            return {'predictions':[{'id':p['id'],'group':'g','predicted_pole':'positive'} for p in rows]}
        with tempfile.TemporaryDirectory() as tmp:
            result=predict(points,{},frozen,Path(tmp),llm=llm)
        self.assertEqual(sorted(map(len,calls)),[244,256])
        self.assertEqual(len({p['id'] for rows in calls for p in rows}),500)
        self.assertEqual(len(result['assignments']),500)
        self.assertEqual(result['llm_requests'],2)


if __name__ == '__main__':
    unittest.main()
