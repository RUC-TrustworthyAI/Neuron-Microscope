import copy
import tempfile
import unittest
from pathlib import Path
from language.workflow import contracts
from language.workflow.labels import plot_fields, predict
from language.workflow.state import union_seconds
from language.workflow.client import Client, price
from language.workflow.state import write


class WorkflowTests(unittest.TestCase):
    def test_rule_limit_applies_to_each_pole_and_both_explanation_prompts(self):
        from language.workflow import prompts
        rules = [dict(id=f'{pole}-{i}', name='Context', description='Condition', predicted_pole=pole)
                 for pole in ('positive', 'negative') for i in range(5)]
        contracts.validate_rules({'response_rules': rules}, 5)
        with self.assertRaises(ValueError):
            contracts.validate_rules({'response_rules': rules})
        rules[-1]['predicted_pole'] = 'positive'
        with self.assertRaises(ValueError):
            contracts.validate_rules({'response_rules': rules}, 5)
        self.assertEqual(contracts.interpretation_schema(5)['properties']['positive']['maxItems'], 5)
        self.assertEqual(contracts.interpretation_schema(5)['properties']['negative']['maxItems'], 5)
        self.assertEqual(contracts.interpretation_schema(5)['properties']['positive']['minItems'], 0)
        self.assertEqual(contracts.interpretation_schema(5)['properties']['negative']['minItems'], 0)
        self.assertEqual(contracts.INTERPRETATION['properties']['positive']['maxItems'], 3)
        for prompt in (prompts.INTERPRET, prompts.REFINE):
            formatted = prompt.format(max_rules_per_pole=5)
            self.assertIn('at most 5', formatted)
            self.assertIn('positive', formatted)
            self.assertIn('negative', formatted)

    def test_pole_explanations_and_blind_rules_have_the_same_source(self):
        output = dict(title='Context', lede='Overview', positive=[
            dict(id='p', name='Positive condition', description='Positive explanation.')], negative=[
            dict(id='n', name='Negative condition', description='Negative explanation.')],
            what_would_change_my_mind='Counterexample', confidence='low', hypotheses=[
                dict(id='h1', description='Context', rival='Token'),
                dict(id='h2', description='Position', rival='Topic')])
        contracts.validate(output, contracts.interpretation_schema())
        result = contracts.interpretation_result(output)
        self.assertEqual(result['claim']['one_paragraph'], result['response_rules'][0]['description'])
        self.assertEqual(result['claim']['negative_pole'], result['response_rules'][1]['description'])
        self.assertEqual([r['predicted_pole'] for r in result['response_rules']], ['positive', 'negative'])
        for pole in ('positive', 'negative'):
            empty = copy.deepcopy(output)
            empty[pole] = []
            contracts.validate(empty, contracts.interpretation_schema())
            contracts.validate_rules(contracts.interpretation_result(empty))
            empty[pole] = [dict(id=pole, name='Condition', description='   ')]
            with self.assertRaises(ValueError):
                contracts.validate_rules(contracts.interpretation_result(empty))
        with self.assertRaises(ValueError):
            contracts.interpretation_schema(0)

    def test_no_supported_poles_can_leave_all_contexts_unexplained(self):
        output = dict(title='Unresolved', lede='Neither pole has a supported semantic rule.',
                      positive=[], negative=[], what_would_change_my_mind='Consistent contextual evidence.',
                      confidence='low', hypotheses=[dict(id='h1', description='Topic', rival='Form'),
                                                    dict(id='h2', description='Form', rival='Topic')])
        contracts.validate(output, contracts.interpretation_schema(5))
        explanation = contracts.interpretation_result(output)
        contracts.validate_rules(explanation, 5)
        self.assertEqual(explanation['response_rules'], [])
        class FakeClient:
            def request(self, name, prompt, packet, schema, stage, check=None):
                self.assert_no_directional_rules(packet)
                return {'predictions': [dict(id=p['id'], group='no-visible-cue', predicted_pole='uncertain',
                                            reason='No supported rule.') for p in packet['points']]}
            @staticmethod
            def assert_no_directional_rules(packet):
                assert [r['id'] for r in packet['response_rules']] == ['no-visible-cue']
        sample = {'points': [dict(id='a', document='d', position=0, observed_prefix='Context', activation=3.)]}
        contexts = {'documents': {'d': {'tokens': ['Context']}}}
        with tempfile.TemporaryDirectory() as folder:
            prediction = predict(FakeClient(), explanation, sample, contexts, Path(folder))
        self.assertEqual(prediction['records'][0]['predicted_pole'], 'uncertain')
        self.assertEqual(prediction['summary']['directional_coverage'], 0.)

    def test_prediction_scores_weight_strong_errors_and_retain_abstentions(self):
        from language.workflow.interpretation import prediction_scores
        sample = {'points': [dict(id='strong', activation=100.), dict(id='weak', activation=1.),
                             dict(id='negative', activation=-10.), dict(id='zero', activation=0.)]}
        before = {'records': [dict(id='strong', predicted_pole='positive'),
                              dict(id='weak', predicted_pole='negative'),
                              dict(id='negative', predicted_pole='negative')]}
        after = {'records': [dict(id='strong', predicted_pole='negative'),
                             dict(id='weak', predicted_pole='positive'),
                             dict(id='negative', predicted_pole='uncertain')]}
        a, b = prediction_scores(before, sample), prediction_scores(after, sample)
        self.assertGreater(a['positive'], b['positive'])
        self.assertEqual(a['negative'], 1.)
        self.assertEqual(b['negative'], 0.)
        self.assertEqual(len(after['records']), 3)

    def test_refinement_feedback_selects_reversals_and_strong_uncertain_cases(self):
        from language.workflow.evidence import refinement_feedback
        cases = [('wrong-negative', .001, 'negative'), ('wrong-positive', -.001, 'positive'),
                 ('strong-positive', 51., 'uncertain'), ('strong-negative', -11., 'uncertain'),
                 ('positive-boundary', 50., 'uncertain'), ('negative-boundary', -10., 'uncertain'),
                 ('weak-positive', 1., 'uncertain'), ('weak-negative', -1., 'uncertain'),
                 ('correct-positive', 100., 'positive'), ('correct-negative', -20., 'negative'),
                 ('zero-directional', 0., 'negative'), ('zero-uncertain', 0., 'uncertain')]
        rules = [dict(id=pole, predicted_pole=pole) for pole in ('positive', 'negative')]
        rules.append(dict(id='no-visible-cue', predicted_pole='uncertain'))
        records = [dict(id=identity, group='no-visible-cue' if predicted == 'uncertain' else predicted,
                        predicted_pole=predicted, reason='Contextual prediction', activation=value,
                        true_pole='positive' if value > 0 else 'negative' if value < 0 else 'zero')
                   for identity, value, predicted in cases]
        predictions = dict(records=records, response_rules=rules, summary={'labelled': len(records)})
        original = copy.deepcopy(predictions)
        feedback = refinement_feedback(predictions, {'min': -20., 'max': 100.})
        self.assertEqual([r['id'] for r in feedback['wrong_direction']],
                         ['wrong-negative', 'wrong-positive'])
        self.assertEqual([r['id'] for r in feedback['strong_no_visible_cue']],
                         ['strong-positive', 'strong-negative'])
        self.assertNotIn('assignments', feedback)
        self.assertTrue(all('examples' not in g for g in feedback['groups']))
        self.assertEqual(sum(g['count'] for g in feedback['groups']), len(records))
        self.assertEqual(predictions, original)

    def test_sampling_shares_500_points_equally_and_redistributes_shortfalls(self):
        import sys
        from unittest.mock import patch
        with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'tools'), *sys.path]):
            from language.workflow.sampling import allocate
        quotas = allocate([10_000] * 17)
        self.assertEqual(sum(quotas), 500)
        self.assertLessEqual(max(quotas) - min(quotas), 1)
        self.assertEqual(allocate([10_000, 1_000_000_000] * 8), allocate([10_000] * 16))
        self.assertEqual(allocate([0, 1, 2, 1000, 1_000_000_000]), [0, 1, 2, 249, 248])
        self.assertEqual(allocate([0, 1, 1000, 1000]), [0, 1, 250, 249])
        self.assertEqual(allocate([1] + [10_000] * 14), [1] + [36] * 9 + [35] * 5)
        self.assertEqual(allocate([17] + [10_000] * 16), [17] + [31] * 3 + [30] * 13)
        self.assertEqual(allocate([10_000] * 11 + [26]), [44] + [43] * 10 + [26])
        self.assertEqual(allocate([0, 1, 2, 3]), [0, 1, 2, 3])
        self.assertEqual(allocate([0, 0]), [0, 0])
        self.assertEqual(allocate([]), [])

    def test_bin_positions_cover_the_complete_population_without_duplicates(self):
        from language.workflow.sampling import spread_ranks
        self.assertEqual(spread_ranks(100, 4).tolist(), [12, 37, 62, 87])
        self.assertEqual(spread_ranks(6, 4).tolist(), [0, 2, 3, 5])
        self.assertEqual(spread_ranks(17, 17).tolist(), list(range(17)))
        self.assertEqual(spread_ranks(0, 0).tolist(), [])
        self.assertEqual(spread_ranks(100_000_000, 0).tolist(), [])
        selected = spread_ranks(100_000_000, 500).tolist()
        self.assertEqual(len(set(selected)), 500)
        self.assertGreater(selected[-1], 99_000_000)
        self.assertEqual(selected, spread_ranks(100_000_000, 500).tolist())
        with self.assertRaises(ValueError):
            spread_ranks(1, 2)

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
            def request(self, name, prompt, packet, schema, stage, check=None):
                self.calls.append(packet)
                self.assertions(packet)
                output = {'predictions': [{'id': p['id'], 'group': 'positive-rule',
                    'predicted_pole': 'positive', 'reason': 'Context supports the rule.'}
                    for p in packet['points'][:-1]]}
                output['predictions'].append(dict(id=packet['points'][-1]['id'], group='unknown',
                                                  predicted_pole='negative', reason='Unusable group.'))
                if check:
                    check(output)
                return output
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

    def test_spectrum_reference_resolves_with_or_without_its_prefix(self):
        from language.workflow.evidence import evidence_catalog
        point = dict(id='spectrum-19299029', text='Context', activation=2.0625,
                     token_activations=[dict(token='Context', activation=2.0625)],
                     peak_index=0, cut_left=False, cut_right=False)
        catalog = evidence_catalog([], [], [], {'points': [point]}, {})
        self.assertIs(catalog['19299029'], catalog['spectrum-19299029'])
        self.assertEqual(catalog['19299029']['activation'], 2.0625)

    def test_five_blind_batches_run_concurrently_and_retry_only_missing_points(self):
        import threading
        class FakeClient:
            def __init__(self):
                self.batches = []
                self.lock = threading.Lock()
                self.barrier = threading.Barrier(5, timeout=5)

            def request(self, name, prompt, packet, schema, stage, check=None):
                ids = [point['id'] for point in packet['points']]
                with self.lock:
                    index = len(self.batches)
                    self.batches.append(ids)
                if index < 5:
                    self.barrier.wait()
                    ids = ids[:-4]
                return {'predictions': [dict(id=identity, group='positive-rule',
                        predicted_pole='positive', reason='Context supports the rule.') for identity in ids]}

        initial = {'claim': {}, 'response_rules': [dict(id='positive-rule', name='Context',
                   description='A contextual pattern.', predicted_pole='positive')]}
        sample = {'points': [dict(id=str(i), document=str(i), position=0, observed_prefix=f'word{i}',
                  activation=(-1.) ** i, gate=1., up=(-1.) ** i, bin=0) for i in range(500)], 'bins': [{}]}
        contexts = {'documents': {str(i): {'tokens': [f'word{i}']} for i in range(500)}}
        client = FakeClient()
        with tempfile.TemporaryDirectory() as folder:
            result = predict(client, initial, sample, contexts, Path(folder))
        self.assertEqual([len(batch) for batch in client.batches], [100] * 5 + [20])
        self.assertEqual(len({identity for batch in client.batches[:5] for identity in batch}), 500)
        self.assertEqual(set(client.batches[5]),
                         {identity for batch in client.batches[:5] for identity in batch[-4:]})
        self.assertEqual(len(result['records']), 500)
        fields = plot_fields(sample, result, {})
        self.assertEqual(fields['spectrum']['points'], fields['anatomy']['points'])

    def test_bipolar_score_retains_negative_peak_and_common_frame(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        from common import select_peak
        from language.workflow.render import headline_figure
        trace = [{'position': 0, 'product': 12.}, {'position': 1, 'product': -25.}]
        self.assertEqual(select_peak(trace, 'absmax')['product'], -25.)
        figures = [{'id': 'positive-only', 'results': [{'activation': 300.}, {'activation': -.5}]},
                   {'id': 'both', 'results': [{'activation': 200.}, {'activation': -25.}]}]
        self.assertEqual(headline_figure(figures, {'max': 330., 'min': -37.5}), 'positive-only')
        negative = {'response_rules': [dict(predicted_pole='negative')]}
        self.assertEqual(headline_figure(figures, {'max': 330., 'min': -37.5}, negative), 'both')

    def test_control_revision_retains_the_stronger_measured_version(self):
        from language.workflow.controls import choose_versions
        def figure(identity, values, pole):
            return dict(id=identity, results=[dict(activation=value, expected=pole) for value in values])
        original = [figure('contact', [422., 0.], 'positive'), figure('notice', [-10., -2.], 'negative')]
        revised = [figure('notice', [-30., -1.], 'negative'), figure('contact', [130., 0.], 'positive')]
        explanation = {'response_rules': [dict(predicted_pole=pole) for pole in ('positive', 'negative')]}
        chosen = choose_versions(original, revised, {'min': -70., 'max': 322.}, explanation)
        self.assertEqual([f['results'][0]['activation'] for f in chosen], [422., -30.])

    def test_control_peak_follows_the_actual_difference_inside_the_slot(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        from language.workflow.measure import Measurements
        class SavedMeasurements(Measurements):
            def texts(self, items, name):
                return [dict(id=item['id'], label=item['label'], text=item['text'],
                             token_ids=[1, 2, 3 + i], scored_positions=[1, 2],
                             trace=[dict(position=0, token='same', product=100.),
                                    dict(position=1, token='of', product=34.5),
                                    dict(position=2, token=str(i), product=float(i + 1))])
                        for i, item in enumerate(items)]
        plan = dict(id='factors', title='Factors', rationale='Change the number',
                    template='Question: {slot}?', score='replacement_span', reduction='absmax',
                    replacements=[dict(label=str(i), text=f'Question: factor of {i}?', expected='positive')
                                  for i in range(2)])
        with tempfile.TemporaryDirectory() as folder:
            measured = SavedMeasurements(None, Path(folder), None)
            rows, figures = measured.controls([plan])
        self.assertEqual([row['selected']['product'] for row in rows], [1., 2.])
        self.assertEqual([row['scored_positions'] for row in rows], [[2], [2]])
        self.assertEqual([row['activation'] for row in figures[0]['results']], [1., 2.])

    def test_measured_explanation_respects_single_revision_and_labels_stay_blind(self):
        from language.workflow.interpretation import explain
        from language.workflow.evidence import spectrum_packet
        from language.workflow.state import Timeline, read
        class FakeClient:
            calls = []
            def request(self, name, prompt, packet, schema, stage, check=None):
                self.calls.append((name, packet))
                if name in ('initial', 'refinement'):
                    self.assert_measurements(packet)
                    if name == 'refinement':
                        feedback = packet['prediction_feedback']
                        assert {r['id'] for r in feedback['wrong_direction']} == {'1', '3'}
                        assert feedback['strong_no_visible_cue'] == []
                        assert 'assignments' not in feedback
                    result = dict(title=name, lede='Overview',
                                  what_would_change_my_mind='Counterexample', confidence='low', hypotheses=[],

                                  **{pole: [dict(id=pole, name=pole, description='Contextual rule')]
                                     for pole in ('positive', 'negative')})
                    check(result)
                    return result
                assert not {'prediction_feedback', 'documents'} & packet.keys()
                for point in packet['points']:
                    assert not {'activation', 'true_pole', 'gate', 'up'} & point.keys()
                return {'predictions': [{'id': point['id'], 'group': 'positive',
                    'predicted_pole': 'positive', 'reason': 'Contextual prediction'} for point in packet['points']]}
            @staticmethod
            def assert_measurements(packet):
                positions = [p for d in packet['documents'] for p in d['positions']]
                assert len(positions) == 4
                assert all('activation' in p and not {'gate', 'up', 'value'} & p.keys() for p in positions)
                assert any(p['activation'] < 0 for p in positions)
        sample = {'points': [dict(id=str(i), document='one', position=i, observed_prefix=f'word{i}',
                   activation=(-1.) ** i, gate=1., up=(-1.) ** i, bin=0) for i in range(4)],
                  'bins': [{}], 'edges': [-2, 2], 'sample_hist': {'counts': [4]}, 'corpus_hist': {'counts': [100]}}
        contexts = {'documents': {'one': {'tokens': ['first ', 'second ', 'third ', 'fourth']}}}
        client = FakeClient()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / 'report.json', {'statistics': {'product': {'min': -1., 'max': 1.}},
                                        'examples': {}, 'bands': [],
                                        'points': {}, 'documents': {}})
            packet = spectrum_packet(root, sample, contexts)
            self.assertEqual(len(packet['documents']), 1)
            for point in packet['documents'][0]['positions']:
                self.assertEqual(packet['documents'][0]['text'][point['character_start']:point['character_end']],
                                 point['token'])
            first, refined, before, after = explain(client, packet, sample, contexts, root, Timeline(root),
                                                   max_refinements=1)
            self.assertEqual(len(read(root / 'interpretation-history.json')), 2)
        self.assertEqual(after['claim'], refined['claim'])
        self.assertEqual(refined['claim']['title'], 'refinement')
        self.assertEqual([n for n, _ in client.calls if n in ('initial', 'refinement')], ['initial', 'refinement'])
        self.assertTrue(any(r['true_pole'] == 'negative' and r['predicted_pole'] == 'positive'
                            for r in after['records']))

    def test_explanation_keeps_extremes_quantile_traces_and_token_statistics(self):
        from language.workflow.evidence import spectrum_packet
        doc = dict(tokens=['A ', 'B ', 'C'], gate=[1., 2., 3.], value=[-1., 1., 2.],
                   product=[-1., 2., 6.])
        data = dict(statistics={'product': {'min': -1., 'max': 6.},
                                'gate': {'min': 1., 'max': 3.}, 'value': {'min': -1., 'max': 2.}},
                    examples={'top_documents': ['top'], 'bottom_documents': ['bottom']},
                    bands=[dict(low=50, high=75, lower=0., upper=3., points=['middle'])],
                    points={'top': dict(document='one', position=2, values=[3., 2., 6.]),
                            'bottom': dict(document='two', position=0, values=[1., -1., -1.]),
                            'middle': dict(document='one', position=1, values=[2., 1., 2.])},
                    documents={'one': doc, 'two': doc})
        sample = dict(points=[dict(id='sample', document='one', position=1, gate=2., up=1.,
                                  activation=2., bin=0)], edges=[-2, 8], corpus_hist={'counts': [3]},
                      sample_hist={'counts': [1]})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / 'report.json', data)
            write(root / 'token-rankings.json', [dict(view=dict(mode='positive', fraction=.1,
                  threshold=.6, column=4), minimum_count=50, rows=[[1, 'A', 100, .5, 40]])])
            write(root / 'peak-census.json', {'n': 1, 'peaks': [{'token': 'C', 'product': 6.}]})
            write(root / 'peak-census-negative.json', {'n': 1, 'peaks': [{'token': 'A', 'product': -1.}]})
            packet = spectrum_packet(root, sample, {'documents': {'one': doc}})
        self.assertEqual(len(packet['documents']), 2)
        self.assertEqual(sum(len(d['positions']) for d in packet['documents']), 1)
        documents = {d['id']: d for d in packet['documents']}
        for case in packet['samples']:
            self.assertNotIn('observed_prefix', case)
            self.assertEqual(documents[case['document']]['text'][case['character_start']:case['character_end']],
                             case['anchor']['token'])
            self.assertEqual(case['anchor']['product'], data['points'][case['id']]['values'][2])
            self.assertEqual(set(case['anchor']), {'position', 'token', 'product'})
            self.assertTrue(all(set(row) == {'position', 'token', 'product'} for row in case['local_trace']))
        self.assertEqual(packet['statistics'], {'product': data['statistics']['product']})
        def assert_no_channels(value):
            if isinstance(value, dict):
                self.assertFalse({'gate', 'up', 'value'} & value.keys())
                for item in value.values():
                    assert_no_channels(item)
            elif isinstance(value, list):
                for item in value:
                    assert_no_channels(item)
        assert_no_channels(packet)
        self.assertEqual(packet['quantile_bands'][0]['case_ids'], ['middle'])
        self.assertEqual(packet['token_selectivity']['tables'][0]['rows'][0]['hit_fraction'], .4)
        self.assertEqual(set(packet['peak_census']), {'positive', 'negative'})

    def test_selection_keeps_matching_explanation_and_labels(self):
        from language.workflow.interpretation import explain
        from language.workflow.state import Timeline, read
        class FakeClient:
            def __init__(self, improve):
                self.improve, self.calls = improve, []
            def request(self, name, prompt, packet, schema, stage, check=None):
                self.calls.append(name)
                if name in ('initial', 'refinement'):
                    return dict(title=name, lede='Overview', what_would_change_my_mind='Counterexample',
                                confidence='low', hypotheses=[],
                                **{pole: [dict(id=pole, name=pole,
                                description=f'{name} {pole} condition')] for pole in ('positive', 'negative')})
                correct = name.startswith('refined-labels') == self.improve
                predictions = []
                for point in packet['points']:
                    pole = 'negative' if correct and point['anchor']['position'] % 2 else 'positive'
                    predictions.append(dict(id=point['id'], group=pole, predicted_pole=pole, reason='Fixture'))
                return {'predictions': predictions}
        sample = {'points': [dict(id=str(i), document='one', position=i, observed_prefix=f'word{i}',
                   activation=(-1.) ** i, gate=1., up=(-1.) ** i, bin=0) for i in range(4)]}
        contexts = {'documents': {'one': {'tokens': ['first ', 'second ', 'third ', 'fourth']}}}
        for improve in (True, False):
            with self.subTest(improve=improve), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                client = FakeClient(improve)
                packet = {'statistics': {'product': {'min': -1., 'max': 1.}}}
                _, selected, _, predictions = explain(client, packet, sample, contexts, root, Timeline(root),
                                                      max_refinements=1)
                self.assertEqual(selected['claim']['title'], 'refinement' if improve else 'initial')
                self.assertEqual(predictions['summary']['directional_accuracy'], 1.)
                history = read(root / 'interpretation-history.json')
                self.assertEqual([h['selected'] for h in history], [not improve, improve])
                self.assertEqual(len(history), 2)
                self.assertEqual(len(client.calls), 4)
                self.assertEqual(read(root / 'interpretation.json'), selected)
                self.assertEqual(read(root / 'predictions.json'), predictions)

    def test_refinements_use_previous_feedback_even_when_an_earlier_pair_is_selected(self):
        from language.workflow.interpretation import explain
        from language.workflow.state import Timeline, read
        class FakeClient:
            def __init__(self):
                self.explanations, self.blind_calls = [], []
            def request(self, name, prompt, packet, schema, stage, check=None):
                if name == 'initial' or name.startswith('refinement'):
                    if name != 'initial':
                        assert packet['current_explanation']['claim']['title'] == self.explanations[-1]
                        expected = [
                            {str(i): 'uncertain' for i in range(4)},
                            {'1': 'uncertain', '3': 'uncertain'},
                            {'0': 'negative', '2': 'negative'},
                        ][len(self.explanations) - 1]
                        feedback = packet['prediction_feedback']
                        observed = {r['id']: r['predicted_pole']
                                    for r in feedback['wrong_direction'] + feedback['strong_no_visible_cue']}
                        assert observed == expected
                        assert 'assignments' not in feedback
                    self.explanations.append(name)
                    return dict(title=name, lede='Overview', what_would_change_my_mind='Counterexample',
                                confidence='low', hypotheses=[],
                                **{pole: [dict(id=pole, name=pole,
                                description=f'{name} {pole} condition')] for pole in ('positive', 'negative')})
                self.blind_calls.append(name)
                assert not {'prediction_feedback', 'documents'} & packet.keys()
                rows = []
                for point in packet['points']:
                    assert not {'activation', 'true_pole', 'gate', 'up', 'value'} & point.keys()
                    i = point['anchor']['position']
                    pole = (['uncertain'] * 4,
                            ['positive', 'uncertain', 'positive', 'uncertain'],
                            ['negative'] * 4,
                            ['uncertain'] * 4)[len(self.explanations) - 1][i]
                    rows.append(dict(id=point['id'], group='no-visible-cue' if pole == 'uncertain' else pole,
                                     predicted_pole=pole, reason='Contextual prediction'))
                return {'predictions': rows}
        sample = {'points': [dict(id=str(i), document='one', position=i, observed_prefix=f'word{i}',
                                  activation=2. * (-1.) ** i, bin=0) for i in range(4)]}
        contexts = {'documents': {'one': {'tokens': ['first ', 'second ', 'third ', 'fourth']}}}
        with tempfile.TemporaryDirectory() as folder:
            root, client = Path(folder), FakeClient()
            packet = {'statistics': {'product': {'min': -2., 'max': 2.}}}
            _, selected, _, predictions = explain(client, packet, sample, contexts, root, Timeline(root), max_refinements=3)
            history = read(root / 'interpretation-history.json')
            self.assertEqual(client.explanations, ['initial', 'refinement', 'refinement-2', 'refinement-3'])
            self.assertEqual(len(client.blind_calls), 4)
            self.assertEqual([h['round'] for h in history], [0, 1, 2, 3])
            self.assertEqual([h['selected'] for h in history], [False, True, False, False])
            self.assertEqual(history[1]['pole_scores'], {'positive': 1., 'negative': 0.})
            self.assertEqual(history[2]['pole_scores'], {'positive': -1., 'negative': 1.})
            for h in history:
                self.assertTrue((root / h['predictions_path']).is_file())
            self.assertEqual(selected['claim']['title'], 'refinement')
            self.assertEqual(predictions['summary']['directional_coverage'], .5)
            self.assertEqual(history[-1]['pole_scores'], {'positive': 0., 'negative': 0.})
            self.assertEqual(read(root / 'interpretation.json'), selected)
            self.assertEqual(read(root / 'predictions.json'), predictions)

    def test_selection_uses_strict_accuracy_threshold_then_coverage(self):
        from language.workflow.interpretation import select_round
        def entry(number, accuracy, coverage):
            return dict(round=number, summary=dict(directional_accuracy=accuracy,
                                                  directional_coverage=coverage))
        rounds = [entry(0, .9, 1.), entry(1, .91, .6), entry(2, .99, .4)]
        self.assertEqual(select_round(rounds)['round'], 1)
        self.assertEqual(select_round([entry(0, .8, .9), entry(1, .9, .3)])['round'], 1)
        self.assertEqual(select_round([entry(0, .8, .9), entry(1, .8, .3)])['round'], 0)
        self.assertEqual(select_round([entry(0, .95, .4), entry(1, .95, .4)])['round'], 1)


    def test_report_analysis_cannot_replace_frozen_rules(self):
        from language.workflow.revision import finalize
        from language.workflow.state import Timeline
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

    def test_saved_prediction_response_reuses_content_with_response_format_metadata(self):
        import json
        schema = contracts.PREDICTIONS
        expected = {'predictions': [dict(id='sample', group='N1', predicted_pole='negative',
                                          reason='Context matches the negative rule.')]}
        raw = json.dumps({'type': 'json_object', **expected})
        messages = [{'role': 'system', 'content': 'Prompt\nJSON schema:\n' + json.dumps(schema)},
                    {'role': 'user', 'content': '{}'}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempt = root / 'api/labels/attempt-001'
            write(attempt / 'request.json', {'messages': messages})
            write(attempt / 'completion.json', {'finish_reason': 'stop'})
            (attempt / 'output.txt').write_text(raw)
            client = Client({}, root / 'api', root / 'slots')
            client._stream = lambda *args: self.fail('Complete predictions must not be requested again')
            self.assertEqual(client.request('labels', 'Prompt', {}, schema, 2), expected)
            self.assertEqual((attempt / 'output.txt').read_text(), raw)

    def test_response_format_cleanup_preserves_declared_fields_and_rejects_other_extras(self):
        from language.workflow.client import parse_output
        schema = contracts.obj(id=contracts.S)
        cleaned = parse_output('{"type":"json_object","id":"sample"}', schema)
        contracts.validate(cleaned, schema)
        self.assertEqual(cleaned, {'id': 'sample'})
        for raw in ('{"id":"sample","extra":"value"}', '{"type":"different","id":"sample"}'):
            with self.assertRaises(ValueError):
                contracts.validate(parse_output(raw, schema), schema)
        declared = contracts.obj(type=contracts.S, id=contracts.S)
        output = parse_output('{"type":"json_object","id":"sample"}', declared)
        contracts.validate(output, declared)
        self.assertEqual(output['type'], 'json_object')

    def test_context_capacity_uses_exact_remaining_window_without_changing_messages(self):
        import urllib.error
        schema = contracts.obj(id=contracts.S)
        with tempfile.TemporaryDirectory() as folder:
            client = Client({'max_output_tokens': 393216}, Path(folder) / 'api', Path(folder) / 'slots')
            sent = []
            def stream(messages, destination, stage, max_tokens=None):
                destination.mkdir()
                sent.append((copy.deepcopy(messages), max_tokens))
                if max_tokens is None:
                    error = urllib.error.HTTPError('https://example.invalid', 400, 'Context capacity', {}, None)
                    error.context_output_tokens = 352808
                    raise error
                self.assertEqual(max_tokens, 352808)
                return '{"id":"result"}'
            client._stream = stream
            self.assertEqual(client.request('initial', 'Prompt', {'evidence': 'Full text'}, schema, 2),
                             {'id': 'result'})
            self.assertEqual(sent[0][0], sent[1][0])
            self.assertEqual(len(sent), 2)

    def test_resume_corrects_saved_answer_without_regenerating_it(self):
        import hashlib
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
            write(root / 'api/revision/result.json', {
                'input_id': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                'output': {'label': 'A long structural label'}})
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
