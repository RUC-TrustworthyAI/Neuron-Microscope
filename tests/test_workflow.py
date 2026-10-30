import copy
import tempfile
import unittest
from pathlib import Path
from workflow import contracts
from workflow.labels import plot_fields, predict
from workflow.state import union_seconds
from workflow.client import Client, price
from workflow.state import write


class WorkflowTests(unittest.TestCase):
    def test_rule_limit_applies_to_each_pole_and_both_explanation_prompts(self):
        from workflow import prompts
        rules = [dict(id=f'{pole}-{i}', name='Context', description='Condition', predicted_pole=pole)
                 for pole in ('positive', 'negative') for i in range(5)]
        contracts.validate_rules({'response_rules': rules}, 5)
        with self.assertRaises(ValueError):
            contracts.validate_rules({'response_rules': rules})
        rules[-1]['predicted_pole'] = 'positive'
        with self.assertRaises(ValueError):
            contracts.validate_rules({'response_rules': rules}, 5)
        self.assertEqual(contracts.interpretation_schema(5)['properties']['response_rules']['maxItems'], 10)
        self.assertEqual(contracts.INTERPRETATION['properties']['response_rules']['maxItems'], 6)
        for prompt in (prompts.INTERPRET, prompts.REFINE):
            self.assertIn('at most 5 positive and 5 negative', prompt.format(max_rules_per_pole=5))

    def test_axis_label_is_separate_from_full_input(self):
        plan = {'template': '{slot} Results follow.', 'score': 'replacement_span', 'reduction': 'max',
                'replacements': [{'label': 'Arithmetic', 'text': 'A long mathematical passage. ' * 20 + ' Results follow.'}]}
        self.assertEqual(contracts.control_request(plan)['replacements'][0]['label'], 'Arithmetic')
        plan['replacements'][0]['label'] = plan['replacements'][0]['text']
        with self.assertRaises(ValueError):
            contracts.control_request(plan)

    def test_complete_condition_is_inserted_only_once(self):
        plan = {'template': 'Visit {slot} today.', 'score': 'replacement_span', 'reduction': 'max',
                'replacements': [{'label': 'City', 'text': 'Visit the city today.'}]}
        request = contracts.control_request(plan)
        prefix, suffix = request['template'].split('{slot}')
        self.assertEqual(request['replacements'][0]['text'], 'the city')
        self.assertEqual(prefix + request['replacements'][0]['text'] + suffix, 'Visit the city today.')
        plan['replacements'][0]['text'] = 'the city'
        with self.assertRaises(ValueError):
            contracts.control_request(plan)

    def test_suffix_scoring_cannot_target_an_absent_suffix(self):
        plan = {'template': 'Prefix {slot}', 'score': 'after_replacement', 'reduction': 'max', 'replacements': []}
        with self.assertRaises(ValueError):
            contracts.control_request(plan)

    def test_blind_labels_exclude_truth_and_share_only_labelled_points(self):
        class FakeClient:
            calls = []
            def request(self, name, prompt, packet, schema, stage):
                self.calls.append(packet)
                self.assertions(packet)
                return {'predictions': [{'id': p['id'], 'group': 'positive-rule',
                    'predicted_pole': 'positive', 'reason': 'Context supports the rule.'}
                    for p in packet['points'][:-1]]}
            @staticmethod
            def assertions(packet):
                for point in packet['points']:
                    assert 'activation' not in point and 'true_pole' not in point
                    assert 'gate' not in point and 'up' not in point
        initial = {'claim': {}, 'response_rules': [{'id': 'positive-rule', 'name': 'Local context',
                    'description': 'A context rule', 'predicted_pole': 'positive'}]}
        sample = {'points': [{'id': str(i), 'document': str(i), 'position': 0, 'observed_prefix': f'word{i}',
                  'activation': (-1.) ** i, 'gate': 1., 'up': (-1.) ** i, 'bin': 0} for i in range(20)],
                  'bins': [{'i': 0}]}
        contexts = {'documents': {str(i): {'tokens': [f'word{i}']} for i in range(20)}}
        with tempfile.TemporaryDirectory() as folder:
            result = predict(FakeClient(), initial, sample, contexts, Path(folder))
        self.assertEqual(len(result['records']), 19)
        fields = plot_fields(sample, result, {})
        self.assertEqual(fields['spectrum']['points'], fields['anatomy']['points'])
        self.assertEqual(len(fields['anatomy']['points']), 19)
        self.assertTrue(any(p['activation'] < 0 and p['predicted_pole'] == 'positive'
                            for p in fields['anatomy']['points']))

    def test_time_counts_overlap_once(self):
        self.assertEqual(union_seconds([(0, 10), (3, 5), (7, 15), (20, 22)]), 17)

    def test_bipolar_score_retains_negative_peak_and_common_frame(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        from common import select_peak
        from workflow.render import headline_figure
        trace = [{'position': 0, 'product': 12.}, {'position': 1, 'product': -25.}]
        self.assertEqual(select_peak(trace, 'absmax')['product'], -25.)
        figures = [{'id': 'positive-only', 'results': [{'activation': 300.}, {'activation': -.5}]},
                   {'id': 'both', 'results': [{'activation': 200.}, {'activation': -25.}]}]
        self.assertEqual(headline_figure(figures, {'max': 330., 'min': -37.5}), 'both')

    def test_measured_explanation_refines_once_and_labels_stay_blind(self):
        from workflow.interpretation import explain
        from workflow.evidence import spectrum_packet
        from workflow.state import Timeline, read
        class FakeClient:
            calls = []
            def request(self, name, prompt, packet, schema, stage, check=None):
                self.calls.append((name, packet))
                if name in ('initial', 'refinement'):
                    self.assert_measurements(packet)
                    if name == 'refinement':
                        assert len(packet['prediction_feedback']['assignments']) == 4
                    rules = [dict(id=pole, name=pole, description='Contextual rule',
                                  predicted_pole=pole) for pole in ('positive', 'negative')]
                    result = dict(claim={'title': name}, response_rules=rules, hypotheses=[])
                    check(result)
                    return result
                assert 'prediction_feedback' not in packet and 'documents' not in packet
                for point in packet['points']:
                    assert not {'activation', 'true_pole', 'gate', 'up'} & point.keys()
                return {'predictions': [{'id': point['id'], 'group': 'positive',
                    'predicted_pole': 'positive', 'reason': 'Contextual prediction'} for point in packet['points']]}
            @staticmethod
            def assert_measurements(packet):
                positions = [p for d in packet['documents'] for p in d['positions']]
                assert len(positions) == 4
                assert all({'activation', 'gate', 'up'} <= p.keys() for p in positions)
                assert any(p['activation'] < 0 for p in positions)
        sample = {'points': [dict(id=str(i), document='one', position=i, observed_prefix=f'word{i}',
                   activation=(-1.) ** i, gate=1., up=(-1.) ** i, bin=0) for i in range(4)],
                  'bins': [{}], 'edges': [-2, 2], 'sample_hist': {'counts': [4]}, 'corpus_hist': {'counts': [100]}}
        contexts = {'documents': {'one': {'tokens': ['first ', 'second ', 'third ', 'fourth']}}}
        client = FakeClient()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / 'report.json', {'statistics': {}})
            packet = spectrum_packet(root, sample, contexts)
            self.assertEqual(len(packet['documents']), 1)
            for point in packet['documents'][0]['positions']:
                self.assertEqual(packet['documents'][0]['text'][point['character_start']:point['character_end']],
                                 point['token'])
            first, refined, before, after = explain(client, packet, sample, contexts, root, Timeline(root))
            self.assertEqual(len(read(root / 'interpretation-history.json')), 2)
        self.assertEqual(after['claim'], refined['claim'])
        self.assertEqual(refined['claim']['title'], 'refinement')
        self.assertEqual([n for n, _ in client.calls if n in ('initial', 'refinement')], ['initial', 'refinement'])
        self.assertTrue(any(r['true_pole'] == 'negative' and r['predicted_pole'] == 'positive'
                            for r in after['records']))

    def test_report_analysis_cannot_replace_frozen_rules(self):
        from workflow.revision import finalize
        from workflow.state import Timeline
        class FakeClient:
            calls = []
            def request(self, name, prompt, packet, schema, stage, check=None):
                self.calls.append(name)
                if name == 'report-analysis':
                    assert 'response_rules' not in schema['properties']
                    result = dict(criticism=[], clusters=[], units=[], section_prose={}, graph_captions=[])
                    check(result)
                    return result
                assert name == 'final-review'
                return {}
        frozen = {'claim': {'title': 'Frozen'}, 'response_rules': []}
        predictions = dict(response_rules=[], records=[], summary={})
        client = FakeClient()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            revised, labels, review = finalize(client, frozen, predictions,
                dict(atlas=[], search_progress=[], controls=[], blind_tests=[]), {}, {}, {}, root, Timeline(root))
        self.assertEqual(revised['claim'], frozen['claim'])
        self.assertEqual(revised['response_rules'], frozen['response_rules'])
        self.assertIs(labels, predictions)
        self.assertEqual(client.calls, ['report-analysis', 'final-review'])

    def test_price_accounts_for_cached_tokens(self):
        usage = {'prompt_tokens': 1000, 'prompt_cache_hit_tokens': 500, 'completion_tokens': 100}
        pricing = {'cached_input_per_million': .003, 'input_per_million': .15, 'output_per_million': .6}
        self.assertAlmostEqual(price(usage, pricing, 0), .0001365)

    def test_completed_response_can_resume_after_contract_fix(self):
        import json
        schema = {'type': 'object', 'properties': {'id': {'type': 'string'}}, 'required': ['id']}
        messages = [{'role': 'system', 'content': 'Prompt\nJSON schema:\n' + json.dumps(schema)},
                    {'role': 'user', 'content': '{}'}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempt = root / 'api/revision/attempt-001'
            write(attempt / 'request.json', {'messages': messages})
            write(attempt / 'completion.json', {'finish_reason': 'stop'})
            (attempt / 'output.txt').write_text('{"id":"graph-1"}')
            client = Client({}, root / 'api', root / 'slots')
            client._stream = lambda *args: self.fail('Completed response should be reused')
            self.assertEqual(client.request('revision', 'Prompt', {}, schema, 4), {'id': 'graph-1'})

    def test_resume_corrects_saved_answer_without_regenerating_it(self):
        import json
        schema = {'type': 'object', 'properties': {'label': {'type': 'string'}}, 'required': ['label']}
        messages = [{'role': 'system', 'content': 'Prompt\nJSON schema:\n' + json.dumps(schema)},
                    {'role': 'user', 'content': '{}'}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempt = root / 'api/revision/attempt-001'
            write(attempt / 'request.json', {'messages': messages})
            write(attempt / 'completion.json', {'finish_reason': 'stop'})
            (attempt / 'output.txt').write_text('{"label":"A long structural label"}')
            client = Client({}, root / 'api', root / 'slots')
            def check(output):
                if len(output['label'].split()) > 2:
                    raise ValueError('Shorten label to two words')
            def stream(sent, destination, stage):
                self.assertEqual(sent[:2], messages)
                self.assertEqual(json.loads(sent[2]['content'])['label'], 'A long structural label')
                self.assertIn('Shorten label to two words', sent[3]['content'])
                return '{"label":"Structural unit"}'
            client._stream = stream
            self.assertEqual(client.request('revision', 'Prompt', {}, schema, 4, check),
                             {'label': 'Structural unit'})


if __name__ == '__main__':
    unittest.main()
