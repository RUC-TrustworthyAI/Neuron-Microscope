"""Measured readouts and archive operations used by the Criticism cycle."""
import copy
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from language.workflow.criticism_operations import prepare, execute
from language.workflow.measure import Measurements
from language.workflow.state import Timeline


OPTIONS = {'max_conditions': 36, 'max_input_tokens': 256}


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': list(map(ord, text)),
                'offset_mapping': [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class Runtime:
    tokenizer = CharacterTokenizer()

    def __init__(self):
        self.expected, self.forwards = {}, []

    def span(self, name):
        return nullcontext()

    def batch_forward(self, token_lists):
        traces = []
        for ids in token_lists:
            text = self.tokenizer.decode(ids)
            self.forwards.append(text)
            product = self.expected[text]
            traces.append({'gate': [1.] * len(product), 'value': product, 'product': product})
        return traces


def readout(id='peak', location='input', reduction='max', metric='product', **fields):
    return {'id': id, 'location': location, 'reduction': reduction, 'metric': metric, **fields}


def probe(id, text=None, *, case_id=None, readouts=None):
    return {'id': id, 'label': id, 'source': 'archive' if case_id else 'new',
            **({'case_id': case_id} if case_id else {'text': text}),
            'readouts': readouts or [readout()]}


class OperationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        self.runtime = Runtime()
        self.measure = Measurements(self.runtime, root / 'measurements', Timeline(root), 1, 256)
        self.archive = {'traces': {}, 'graphs': {}, 'cohorts': {}, 'distributions': {}}

    def saved(self, id, text, products, **extra):
        trace = [{'position': i, 'token': c, 'token_id': ord(c), 'offset': [i, i + 1],
                  'gate': 1., 'value': products[i], 'product': products[i]} for i, c in enumerate(text)]
        self.archive['traces'][id] = {'id': id, 'text': text, 'token_ids': list(map(ord, text)),
                                      'trace': trace, 'selected': max(trace, key=lambda t: t['product']), **extra}

    def register(self, **operations):
        return prepare(operations, self.runtime.tokenizer, self.archive, OPTIONS, 2, 3)

    def test_multiple_signed_extrema_and_token_readouts_share_one_forward(self):
        text = 'Cue (303)'
        self.runtime.expected[text] = [0., -8., 0., 0., 17., 12., 3., 1., 0.]
        registration = self.register(probes=[probe('trajectory', text, readouts=[
            readout('maximum'), readout('minimum', reduction='min'),
            readout('position', metric='peak_position'),
            readout('paren', location='marker', reduction='single', marker='(', occurrence=0),
            readout('digit', location='token_position', reduction='single', start=5),
            readout('local.max', location='token_range', reduction='max', start=5, end=8)]),
            probe('trajectory.local', text, readouts=[readout('max', location='token_range',
                                                           reduction='max', start=5, end=8)]),
            probe('same-input', text, readouts=[readout('absolute', reduction='absmax')])])
        self.assertEqual(len(registration['model_inputs']), 1)
        result = execute(registration, self.measure, self.archive)
        self.assertEqual(self.runtime.forwards, [text])
        self.assertEqual(result['metrics'], {'trajectory.maximum': 17., 'trajectory.minimum': -8.,
            'trajectory.position': 4, 'trajectory.paren': 17., 'trajectory.digit': 12.,
            'trajectory.local%2Emax': 12., 'trajectory%2Elocal.max': 12., 'same-input.absolute': 17.})
        self.assertEqual(len(result['measurements']), 1)
        self.assertEqual(len(result['conditions']), 3)
        # The same preregistration can resume from the measurement cache without another forward.
        self.assertEqual(execute(registration, self.measure, self.archive), result)
        self.assertEqual(self.runtime.forwards, [text])

    def test_sparse_saved_trace_needs_explicit_token_positions(self):
        self.saved('sparse', 'abcd', [0., -3., 2., 1.])
        self.archive['traces']['sparse'].update(trace_complete=False,
            trace=[self.archive['traces']['sparse']['trace'][1], self.archive['traces']['sparse']['trace'][3]])
        with self.assertRaisesRegex(ValueError, 'complete saved trace'):
            self.register(probes=[probe('saved', case_id='sparse')])
        registration = self.register(probes=[probe('saved', case_id='sparse',
            readouts=[readout(location='token_position', reduction='single', start=1)])])
        self.assertEqual(execute(registration, self.measure, self.archive)['metrics']['saved.peak'], -3.)

    def test_invalid_model_trace_cannot_be_reported_as_evidence(self):
        self.runtime.expected['abc'] = [0., 2., 1.]
        registration = self.register(probes=[probe('new', 'abc')])
        result = execute(registration, self.measure, self.archive)
        bad_row = copy.deepcopy(result['measurements'][0])
        bad_row['trace'][0]['token_id'] = 999
        fake = type('BadMeasurement', (), {'texts': lambda *_: [bad_row]})()
        with self.assertRaisesRegex(ValueError, 'mismatched native trace'):
            execute(registration, fake, self.archive)

    def test_graph_missing_values_use_available_observations_as_denominators(self):
        node = {'layer': 31, 'neuron': 17}
        self.archive['graphs'] = {
            'present': {'nodes': [{**node, 'rank': 2, 'influence': -7., 'activation': -9.},
                                 {**node, 'rank': 4, 'influence': 2., 'activation': 4.}]},
            'missing-fields': {'nodes': [dict(node)]},
            'absent': {'nodes': []}}
        self.archive['cohorts']['all'] = ['present', 'missing-fields', 'absent']
        registration = self.register(graph_queries=[{'id': 'carrier', 'layer': 31, 'neuron': 17,
            'groups': [{'id': 'mixed', 'cohort_id': 'all'}, {'id': 'none', 'case_ids': ['absent']}]}])
        result = execute(registration, self.measure, self.archive)
        metrics = result['metrics']
        self.assertEqual(metrics['carrier.mixed.graph_count'], 3)
        self.assertEqual(metrics['carrier.mixed.presence_count'], 2)
        self.assertAlmostEqual(metrics['carrier.mixed.presence_rate'], 2 / 3)
        self.assertEqual(metrics['carrier.mixed.rank_count'], 1)
        self.assertEqual(metrics['carrier.mixed.rank_median'], 2)
        self.assertEqual(metrics['carrier.mixed.influence_count'], 1)
        self.assertEqual(metrics['carrier.mixed.influence_median'], -7.)
        self.assertEqual(metrics['carrier.mixed.activation_count'], 2)
        self.assertEqual(metrics['carrier.mixed.negative_share'], .5)
        self.assertEqual(metrics['carrier.none.presence_rate'], 0.)
        for field in ('rank_median', 'influence_median', 'activation_median', 'negative_share'):
            self.assertNotIn('carrier.none.' + field, metrics)
        self.assertEqual(result['measurements'], [])
        self.archive['graphs']['absent']['nodes'].append(node)
        with self.assertRaisesRegex(ValueError, 'graph archive changed'):
            execute(registration, self.measure, self.archive)

    def test_seeded_replay_filters_cohort_and_preserves_paired_originals(self):
        for i in range(20):
            text = f'Case {i:02d}'
            value = 5. + i / 4
            self.saved(f'case-{i:02d}', f'Source {i}', [value] * len(f'Source {i}'),
                       isolation_text=text, group='no-visible-cue')
            self.runtime.expected[text] = [0.] * (len(text) - 1) + [value + (3. if i == 1 else .25)]
        self.saved('other', 'other', [9.] * 5, group='other')
        ids = list(self.archive['traces'])
        self.archive['cohorts']['weak'] = ids
        query = {'id': 'replay', 'cohort_id': 'weak', 'group': 'no-visible-cue',
                 'activation_band': [4., 16.], 'sample_size': 14, 'seed': 1, 'tolerance': 2.,
                 'quantile': .999, 'readout': {k: v for k, v in readout().items() if k != 'id'}}
        registration = self.register(resamples=[query])
        self.archive['cohorts']['weak'].reverse()
        self.assertEqual(registration, self.register(resamples=[query]))
        self.assertNotIn('other', registration['operations'][0]['selected_case_ids'])
        result = execute(registration, self.measure, self.archive)
        self.assertEqual(result['metrics']['replay.candidate_count'], 21)
        self.assertEqual(result['metrics']['replay.population_count'], 20)
        self.assertEqual(result['metrics']['replay.sample_count'], 14)
        self.assertEqual(result['metrics']['replay.reproduced_count'], 13)
        self.assertEqual(len(self.runtime.forwards), 14)
        pairs = result['operations'][0]['pairs']
        self.assertEqual(sum(not p['reproduced'] for p in pairs), 1)
        self.assertEqual(len({p['case_id'] for p in pairs}), 14)
        for pair in pairs:
            self.assertEqual(pair['original'], self.archive['traces'][pair['case_id']]['selected']['product'])
        saved = self.archive['traces'][pairs[0]['case_id']]
        saved.update(isolation_available=False, isolation_unavailable_reason='Undecodable isolation span')
        with self.assertRaisesRegex(ValueError, 'Undecodable isolation span'):
            self.register(resamples=[query])
        self.archive['traces'][pairs[0]['case_id']]['selected']['product'] += 1
        with self.assertRaisesRegex(ValueError, 'population changed'):
            execute(registration, self.measure, self.archive)

    def test_quantile_uses_exact_weighted_counts_and_preserves_scope(self):
        self.archive['distributions']['corpus'] = {'values': [8., 0., 4.], 'counts': [1, 2, 1],
                                                   'scope': 'All four saved tokens'}
        registration = self.register(quantiles=[{'id': 'floor', 'distribution_id': 'corpus', 'quantile': .5}])
        result = execute(registration, self.measure, self.archive)
        self.assertEqual(result['metrics'], {'floor.count': 4, 'floor.value': 2.})
        self.assertEqual(result['operations'][0]['scope'], 'All four saved tokens')
        self.archive['distributions']['corpus']['counts'][0] = 3
        with self.assertRaisesRegex(ValueError, 'distribution changed'):
            execute(registration, self.measure, self.archive)


if __name__ == '__main__':
    unittest.main()
