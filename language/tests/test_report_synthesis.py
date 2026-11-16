import unittest
from unittest.mock import Mock

from language.workflow.report_synthesis import synthesize


class ReportSynthesisEvidenceTests(unittest.TestCase):
    def test_request_preserves_scored_text_sign_regimes_and_direct_relp(self):
        points = [
            {'id': str(i), 'bin': 1, 'activation': i / 100,
             'gate': -1 if i == 1 else 1, 'up': -1 if i == 1 else 1}
            for i in range(5)
        ]
        nodes = [
            {'node_id': name, 'layer': layer, 'feature': unit, 'ctx_idx': pos}
            for name, layer, unit, pos in (('scored', 2, 6, 4), ('earlier', 2, 6, 1), ('upstream', 1, 3, 4))
        ]
        links = [
            {'source': 'upstream', 'target': 'earlier', 'weight': 100},
            {'source': 'upstream', 'target': 'scored', 'weight': -5},
        ]
        search = {'activation': 342.0, 'text': 'Cover | Table of Contents', 'peak_index': 2,
                  'token_activations': [{'token': token} for token in ('Cover', ' |', ' Table', ' of Contents')]}
        bundle = {
            'neuron': {'layer': 2, 'neuron': 6},
            'spectrum': {'points': points},
            'search': {'regions': [{'id': 'best', 'name': 'Search', 'frontier': [search]}]},
            'graphs_curated': {'picks': [{'id': 'graph-1', 'graph': {
                'position': 4, 'activation': 8, 'nodes': nodes, 'links': links,
            }}]},
            'criticism': [{'investigation_id': 'r1-i1', 'evidence': [
                {'id': 'contrast', 'text': 'Cover | Table', 'activation': 342.0},
            ]}],
        }
        client = Mock()
        client.request.return_value = {'claim': {'findings': {}}, 'section_prose': {}}
        synthesize(client, bundle, 4)

        packet = client.request.call_args.args[2]
        cases = packet['cases']
        self.assertIn('spectrum:1', cases)
        self.assertEqual(packet['measurements']['spectrum']['sampled_gate_up_sign_counts'][1]['-/-'], 1)
        self.assertEqual(cases['circuit:graph-1']['details']['incoming_relp_flow'], 'upstream: -5')
        self.assertEqual(cases['search:best']['observed_prefix_excerpt'], 'Cover | Table')
        self.assertEqual(cases['search:best']['following_text'], ' of Contents')
        self.assertEqual(cases['criticism:r1-i1:contrast']['details']['activation'], 342.0)

    def test_summary_cannot_cite_only_another_sections_evidence(self):
        bundle = {
            'spectrum': {'points': [{'id': 'weak', 'activation': 0.01}]},
            'search': {'regions': [{'id': 'best', 'name': 'Search',
                                    'frontier': [{'activation': 1.0}]}]},
        }
        answer = {
            'claim': {'findings': {}},
            'section_prose': {
                'search': {'summary': 'Search summary', 'evidence_ids': ['search:best']},
                'spectrum': {'summary': 'Spectrum summary', 'evidence_ids': ['search:best']},
            },
        }

        client = Mock()

        def respond(name, prompt, packet, schema, stage, check):
            check(answer)
            return answer

        client.request.side_effect = respond
        with self.assertRaisesRegex(ValueError, 'Section spectrum must cite'):
            synthesize(client, bundle, 4)
        answer['section_prose']['spectrum']['evidence_ids'].append('spectrum:weak')
        self.assertEqual(set(synthesize(client, bundle, 4)['evidence']), {'spectrum:weak', 'search:best'})
