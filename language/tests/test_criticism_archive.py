"""The production archive preserves native evidence and scopes unavailable fields."""
import copy
import json
import sys
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from language.workflow.criticism_archive import (
    build_archive, load_archive, _corpus_distribution)


class ArchiveTests(unittest.TestCase):
    def unicode_fixture(self):
        point = {'id': 'unicode', 'document': '1', 'position': 2,
                 'observed_prefix': 'A\ufffd\ufffd', 'peak_index': 1,
                 'token_activations': [{'token': '\ufffd'}, {'token': '\ufffd'}, {'token': 'Z'}]}
        contexts = {'documents': {'1': {
            'tokens': ['A', '\ufffd', '\ufffd', 'Z'], 'token_ids': [10, 20, 21, 30],
            'gate': [1., 2., 3., 4.], 'value': [1., 1., 1., 1.], 'product': [1., 2., 3., 4.]}}}

        class Decoder:
            def decode(self, ids, **options):
                assert options == {'skip_special_tokens': False, 'clean_up_tokenization_spaces': False}
                return {(10, 20, 21): 'A\U0001f9a6', (20, 21): '\U0001f9a6',
                        (10, 20): 'A\ufffd', (20,): '\ufffd', (10, 22): 'A\ufffd', (22,): '\ufffd'}[tuple(ids)]

            def encode(self, text, **options):
                assert options == {'add_special_tokens': False}
                return {'\U0001f9a6': [20, 21], '\ufffd': [22]}[text]

        return point, contexts, Decoder()

    def test_whole_unicode_id_sequence_restores_prefix_and_isolation(self):
        point, contexts, decoder = self.unicode_fixture()
        archive = build_archive([], {'points': [point]}, contexts, {'version': 'v', 'records': []}, [],
                                tokenizer=decoder)
        row = archive['traces']['unicode']
        self.assertEqual(row['text'], 'A\U0001f9a6')
        self.assertEqual(row['token_ids'], [10, 20, 21])
        self.assertEqual(row['selected']['position'], 2)
        self.assertEqual(row['isolation_text'], '\U0001f9a6')
        self.assertTrue(row['isolation_available'])
        self.assertEqual(row['isolation_token_ids'], [20, 21])
        self.assertEqual(row['isolation_token_span'], [1, 3])
        self.assertTrue(all(token['offset'] is None for token in row['trace']))

    def test_partial_unicode_byte_span_is_not_replayed_as_replacement_text(self):
        point, contexts, decoder = self.unicode_fixture()
        point.update(position=1, peak_index=0)
        row = build_archive([], {'points': [point]}, contexts, {'version': 'v'}, [],
                            tokenizer=decoder)['traces']['unicode']
        self.assertFalse(row['isolation_available'])
        self.assertNotIn('isolation_text', row)
        self.assertEqual(row['isolation_token_ids'], [20])
        self.assertIn('does not round-trip', row['isolation_unavailable_reason'])

    def test_full_corpus_distribution_uses_exact_bf16_values_and_counts(self):
        import numpy as np
        counts = np.zeros(65536, dtype=np.uint64)
        counts[0x3f80], counts[0xbf80], counts[0x7fc0] = 4, 2, 1
        api = SimpleNamespace(histogram=lambda neuron, channel: counts, close=lambda: None)
        with patch('dataset_reader.open_dataset', return_value=api):
            distribution = _corpus_distribution('fixture', 'L2_N5')
        self.assertEqual(distribution['values'], [-1., 1.])
        self.assertEqual(distribution['counts'], [2, 4])
        self.assertEqual(distribution['finite_token_count'], 6)
        self.assertEqual(distribution['excluded_nonfinite_count'], 1)


class ImportedArchiveTests(unittest.TestCase):
    identity = {'model': '/project-model/Qwen3', 'neuron': {'layer': 2, 'index': 5},
                'settings': {'precision': 'bfloat16', 'batch_size': 1}}

    def fixture(self):
        return {'identity': copy.deepcopy(self.identity), 'archive': {
            'graphs': {'g': {'nodes': [{'layer': 0, 'neuron': 9, 'rank': 1,
                                       'influence': .5, 'activation': -3.}]}},
            'cohorts': {'imported-contexts': ['g']},
            'cohort_descriptions': {'imported-contexts': 'Actual project-owned stored graph cohort.'}}}

    def load(self, envelope, base=None):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'archive.json'
            path.write_text(json.dumps(envelope), encoding='utf-8')
            return load_archive(path, self.identity, base or {})

    def test_imported_graph_cohort_runs_through_operation_engine(self):
        from language.workflow.criticism_operations import prepare, execute
        envelope = self.fixture()
        envelope['archive']['graphs'] = {
            f'g-{i}': {'nodes': [{'layer': 0, 'neuron': 9, 'rank': 1,
                                 'influence': .5, 'activation': -3.}] if i < 2 else []}
            for i in range(3)}
        envelope['archive']['cohorts']['imported-contexts'] = list(envelope['archive']['graphs'])
        archive = self.load(envelope)
        plan = {'probes': [], 'resamples': [], 'quantiles': [], 'graph_queries': [
            {'id': 'carrier', 'layer': 0, 'neuron': 9,
             'groups': [{'id': 'all', 'cohort_id': 'imported-contexts'}]}]}
        registered = prepare(plan, None, archive, {'max_conditions': 36, 'max_input_tokens': 1024}, 1, 1)
        self.assertEqual(registered['model_inputs'], [])
        result = execute(registered, None, archive)
        self.assertEqual(result['metrics']['carrier.all.graph_count'], 3)
        self.assertEqual(result['metrics']['carrier.all.presence_count'], 2)
        self.assertEqual(result['metrics']['carrier.all.negative_share'], 1.)
        self.assertEqual(archive['availability']['graph_count'], 3)

    def test_model_neuron_settings_and_numeric_types_must_match(self):
        for field, replacement in (('model', '/different-model'), ('neuron', {'layer': 2, 'index': 6}),
                                   ('settings', {'precision': 'float32', 'batch_size': 1}),
                                   ('settings', {'precision': 'bfloat16', 'batch_size': True})):
            envelope = self.fixture()
            envelope['identity'][field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'exactly match'):
                self.load(envelope)

    def test_imported_native_trace_cannot_change_its_original_anchor(self):
        envelope = self.fixture()
        envelope['archive']['traces'] = {'t': {'text': 'x', 'trace': [
            {'position': 0, 'token': 'x', 'token_id': 8, 'product': 1.}], 'token_ids': [8],
            'selected': {'position': 0, 'product': 2.}}}
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            self.load(envelope)

    def test_complete_phase_checks_imported_source_before_returning_cached_report(self):
        from language.workflow.run import run
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            report = Path(directory) / 'initial-report'
            report.mkdir()
            (root / 'followups').mkdir(parents=True)
            source = Path(directory) / 'external-archive.json'
            source.write_text(json.dumps(self.fixture()), encoding='utf-8')
            archive = load_archive(source, self.identity, {})
            (root / 'followups/archive.json').write_text(json.dumps(archive), encoding='utf-8')
            (root / 'status.json').write_text(json.dumps({'phase': 'complete'}), encoding='utf-8')
            (report / 'report.json').write_text(json.dumps({'identity': {'layer': 2, 'neuron': 5}}), encoding='utf-8')
            config = {'report': str(report), 'followups': {'max_rounds': 1}, 'llm': {}, 'api_slots': 1,
                      'model': self.identity['model'], 'settings': self.identity['settings'],
                      'followup_archive': str(source)}
            with patch('language.workflow.run.Client'):
                self.assertEqual(run(config, root), root / 'report')
                source.write_text(json.dumps(self.fixture()) + ' ', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'contents changed'):
                    run(config, root)


if __name__ == '__main__':
    unittest.main()
