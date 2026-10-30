"""Optional renderer integration test using an activation-only report fixture."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from workflow.state import write


@unittest.skipUnless(os.environ.get('WORKFLOW_TEST_REPORT'), 'Set an activation-only report fixture')
class RendererTest(unittest.TestCase):
    def test_interpretation_preview_shares_the_same_blind_labels(self):
        from workflow.render import assemble_interpretation
        from spectrum import build
        report = Path(os.environ['WORKFLOW_TEST_REPORT'])
        data = json.loads((report / 'report.json').read_text())
        sample = build(data, [], {})
        sample['points'] = sample['points'][:2]
        for point in sample['points']:
            gate, up, activation = data['points'][point['id']]['values']
            point.update(gate=gate, up=up, activation=activation)
        rule = dict(id='n', name='Negative hypothesis', description='Fixture', predicted_pole='negative')
        predictions = dict(version='fixture', response_rules=[rule], records=[
            dict(id=p['id'], group='n', predicted_pole='negative', reason='Fixture') for p in sample['points']])
        with tempfile.TemporaryDirectory() as folder:
            output = assemble_interpretation(Path(folder), report, {'claim': {'title': 'Fixture'}}, sample, predictions)
            bundle = json.loads((output / 'paper-data.json').read_text())['bundle']
            self.assertEqual(bundle['spectrum']['points'], bundle['anatomy']['points'])
            self.assertTrue(all(p['predicted_pole'] == 'negative' for p in bundle['spectrum']['points']))
            self.assertIsNone(bundle['custom'])
            self.assertIsNone(bundle['graphs_curated'])
            self.assertIn("new Set(['claim', 'spectrum', 'anatomy'])", (output / 'index.html').read_text())

    def test_numeric_fields_labels_and_shared_points(self):
        from workflow.evidence import initial_packet, evidence_catalog
        from workflow.render import assemble
        from spectrum import build
        report = Path(os.environ['WORKFLOW_TEST_REPORT'])
        data = json.loads((report / 'report.json').read_text())
        packet = initial_packet(report)
        sample = build(data, [], {})
        sample['points'] = sample['points'][:2]
        for point in sample['points']:
            gate, up, activation = data['points'][point['id']]['values']
            point.update(gate=gate, up=up, activation=activation)
        rule = {'id': 'p', 'name': 'Context', 'description': 'Fixture', 'predicted_pole': 'positive'}
        pred = {'version': 'fixture', 'records': [dict(id=x['id'], group='p', predicted_pole='positive',
                reason='Fixture') for x in sample['points']], 'response_rules': [rule]}
        trace = [{'position': 0, 'token': 'Condition', 'product': 1., 'gate': 1., 'value': 1.,
                  'token_id': 1, 'offset': [0, 9]}]
        rows = [{'id': f'control:{i}', 'label': f'Case{i}', 'text': f'Long test text {i}.',
                 'status': 'complete', 'trace': trace, 'selected': trace[0], 'scored_positions': [0]}
                for i in range(10)]
        figure = {'id': 'control', 'kind': 'sweep', 'title': 'Fixture', 'template': '{slot}',
                  'results': [{'tag': x['label'], 'text': x['text'], 'activation': 1., 'filler': x['label']}
                              for x in rows]}
        claim = dict(title='Test fixture', lede='Fixture', one_paragraph='Fixture', negative_pole='Fixture',
                     what_would_change_my_mind='Fixture', confidence='low')
        artifacts = {'initial': {}, 'revision': {'claim': claim, 'section_prose': {}, 'clusters': [],
            'units': [], 'graph_captions': [], 'criticism': []}, 'review': {'group': 'unresolved',
            'organized': 1, 'understood': 1, 'one_line': 'Fixture', 'concerning': [], 'fine': []},
            'logit_effect': packet['logit_effect'], 'figures': [figure], 'search_regions': [],
            'corpus': {'max': 1., 'min': -1.}, 'atlas_sections': {}, 'sample': sample,
            'predictions': pred, 'atlas': [], 'graphs': [], 'all_measured': rows, 'tests': {}}
        graph = {'graph': {'id': 'graph-1', 'text': 'Condition', 'position': 0}, 'trace': trace}
        catalog = evidence_catalog(rows, [], [graph], sample, packet)
        self.assertIn(sample['points'][0]['id'], catalog)
        self.assertEqual(catalog['graph-1']['activation'], 1.)
        artifacts['evidence'] = catalog
        artifacts['revision']['criticism'] = [{'attack': 'Fixture', 'prediction': 'Fixture',
            'result': 'Fixture', 'verdict': 'narrowed', 'evidence_ids': ['graph-1']}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / 'timeline.json', [])
            output = assemble(root, report, 'L57_N2065', artifacts, 4)
            bundle = json.loads((output / 'paper-data.json').read_text())['bundle']
            self.assertEqual(len(bundle['custom']['figures'][0]['results']), 10)
            self.assertTrue(bundle['logit_effect']['promote'])
            self.assertTrue(bundle['logit_effect']['suppress'])
            self.assertEqual(bundle['anatomy']['points'], bundle['spectrum']['points'])
            self.assertTrue(bundle['anatomy']['strip']['case_id'])
            self.assertTrue(bundle['peak_census']['kinds'])
            self.assertTrue(bundle['peak_census_negative']['kinds'])
            self.assertEqual(bundle['criticism'][0]['evidence'][0]['activation'], 1.)
            self.assertEqual([x['tag'] for x in bundle['headline']['comparison']['fillers']],
                             [f'Case{i}' for i in range(10)])


if __name__ == '__main__':
    unittest.main()
