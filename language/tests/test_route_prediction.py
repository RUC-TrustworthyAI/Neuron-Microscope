"""Saved attribution isolation, scoring, and automatic API-only continuation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from language.workflow.route_prediction import run as predict_routes
from language.workflow.state import read, write


def fixture(root):
    rows = []
    for identity, split, route in [('a', 'train', 'p1'), ('b', 'train', 'p1'),
                                   ('c', 'train', 'n1'), ('d', 'test', 'p1'),
                                   ('e', 'test', 'n1'), ('f', 'test', None)]:
        rows.append({'id': identity, 'split': split, 'status': 'complete',
                     'source_document': identity, 'text': identity + ' FUTURE',
                     'observed_prefix': identity, 'selected': {'token': identity, 'position': 0,
                     'product': 99}, 'upstream': [{'weight': 77}]})
    factors = {'routes': [{'id': route, 'name': route, 'upstream': []} for route in ('p1', 'n1')],
               'assignments': dict(zip('abcdef', ['p1', 'p1', 'n1', 'p1', 'n1', None])),
               'routes_matrix': {'all': {'features': [[0, 1, 1]], 'components': [[1], [1]],
                   'route_ids': ['p1', 'n1'], 'graph_ids': list('abcdef'),
                   'route_weights': [[1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 0]]}}}
    write(root / 'atlas/atlas-raw.json', rows)
    write(root / 'atlas/route-factors.json', factors)
    return rows, factors


class SavedClient:
    config = {'model': 'fixture'}

    def __init__(self, *args):
        self.packets = []

    def request(self, name, prompt, packet, schema, stage, check):
        self.packets.append((name, copy.deepcopy(packet)))
        if name == 'route-descriptions':
            result = {'routes': [{'id': r['id'], 'name': 'Name ' + r['id'],
                                  'description': 'Training description'} for r in packet['routes']]}
        else:
            guesses = {'d': 'n1', 'e': 'n1', 'f': 'p1'}
            result = {'predictions': [{'id': row['id'], 'predicted_route': guesses[row['id']]}
                                     for row in packet['samples']]}
        check(result)
        return result


class RoutePredictionTests(unittest.TestCase):
    def test_saved_split_blind_payload_and_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas, factors = fixture(root)
            client = SavedClient()
            result = predict_routes(client, root, SimpleNamespace(status=Mock()))
            naming, prediction = [packet for _, packet in client.packets]
            self.assertEqual({r['id'] for route in naming['routes'] for r in route['examples']}, set('abc'))
            self.assertEqual(set(prediction), {'routes', 'samples'})
            for row in prediction['samples']:
                self.assertEqual(set(row), {'id', 'observed_prefix', 'token'})
                self.assertEqual(row['observed_prefix'], row['id'])
            self.assertNotIn('FUTURE', json.dumps(prediction))
            self.assertEqual((result['accuracy'], result['baseline'], result['baseline_route_id']), (.5, .5, 'p1'))
            self.assertEqual(result['confusion'], [[0, 1], [0, 1]])
            self.assertEqual((result['n'], result['test_n'], result['unassigned_n']), (2, 3, 1))
            self.assertEqual(read(root / 'atlas/atlas-raw.json'), atlas)
            self.assertEqual(read(root / 'atlas/route-factors.json'), factors)
            with patch.object(client, 'request', side_effect=AssertionError('No repeated API call')), \
                 patch('language.workflow.route_prediction.write', side_effect=AssertionError('No repeated output')), \
                 patch('language.workflow.route_prediction.read', wraps=read) as cached_read:
                self.assertEqual(predict_routes(client, root, SimpleNamespace(status=Mock())), result)
            cached_read.assert_called_once_with(root / 'route-prediction/result.json')

    def test_completed_measurements_resume_without_model_and_preserve_followups(self):
        from language.workflow.run import _run_foundation
        from language.workflow.followups import integrate_report
        from report import BUNDLE, encode_bundle
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            config = {'model': 'unused', 'settings': {}, 'report': str(root / 'source'),
                      'neuron': 'L1_N2', 'gpus': [0], 'llm': {}, 'api_slots': str(root / 'slots')}
            write(root / 'source/report.json', {'identity': {'layer': 1, 'neuron': 2}})
            write(root / 'artifacts.json', {'frozen_followup_source': True})
            write(root / 'status.json', {'phase': 'failed'})
            original = {'neuron': {'id': 'L1_N2'}, 'routes': [{'id': 'p1'}, {'id': 'n1'}],
                        'criticism': [{'investigation_id': 'R1_Q1', 'attack': 'Keep this finding'}],
                        'clusters': [{'id': 'semantic', 'name': 'Keep this group'}],
                        'anatomy': {'pair': {'measured': True}}, 'ablation': {'rows': [1]},
                        'tests': {'blind_max': {'achieved': 4}}}
            source = ('<html><script id="bundle">' + encode_bundle(original) + '</script><script>'
                      'const B = JSON.parse(document.getElementById("bundle").textContent);'
                      'const POS = {};\n// 7. Tests.\n// old renderer\n// 8. Criticism.\n</script></html>')
            (root / 'report').mkdir()
            preview = source.replace('</html>', '<script id="interpretation-view">preview</script></html>')
            (root / 'report/index.html').write_text(preview)

            def assemble_saved(*args):
                (root / 'report/index.html').write_text(source)
                return root / 'report'

            with patch('language.workflow.run.Client', SavedClient), \
                 patch('common.Runtime.load', side_effect=AssertionError('No target model')), \
                 patch('language.workflow.run.run_attribution', side_effect=AssertionError('No attribution')), \
                 patch('language.workflow.render.assemble', side_effect=assemble_saved) as assemble:
                with patch('language.workflow.run.predict_routes', side_effect=RuntimeError('API interruption')):
                    with self.assertRaisesRegex(RuntimeError, 'API interruption'):
                        _run_foundation(config, root)
                self.assertNotIn('interpretation-view', (root / 'report/index.html').read_text())
                _run_foundation(config, root)
            assemble.assert_called_once()
            updated = read(root / 'report/paper-data.json')['bundle']
            for key in ('criticism', 'clusters', 'anatomy', 'ablation'):
                self.assertEqual(updated[key], original[key])
            self.assertEqual(updated['tests']['blind_max'], original['tests']['blind_max'])
            self.assertEqual(updated['tests']['route_prediction']['accuracy'], .5)
            self.assertEqual(updated['routes'][0]['name'], 'Name p1')
            self.assertEqual(read(root / 'artifacts.json'), {'frozen_followup_source': True})
            self.assertEqual(read(root / 'status.json')['phase'], 'foundation_complete')
            (root / 'followup/report').mkdir(parents=True)
            (root / 'followup/report/index.html').write_text(source)
            integrate_report(root / 'followup/report', root)
            overlaid = json.loads(BUNDLE.search((root / 'report/index.html').read_text())[2])
            self.assertEqual(overlaid, updated)


if __name__ == '__main__':
    unittest.main()
