"""Offline follow-up contracts: no API calls, real neuron inputs, or model forwards."""
import copy
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from language.workflow import followup_protocol as protocol
from language.workflow.followup_corpus import _lookup
from language.workflow.followup_execute import Executor, contrast_items
from language.workflow.followup_report import render
from language.workflow.followups import DEFAULTS, SOURCE_FILES, rounds, run
from language.tools.report import BUNDLE
from language.workflow.contracts import validate
from language.workflow.state import Timeline, read, write


def question(name, action='contrast'):
    return {'question': name, 'criticism_ids': ['C1'], 'action': action,
            'evidence_ids': ['saved-1'], 'hypothesis': 'relation', 'rival': 'lexical',
            'prediction_if_hypothesis': 'changes with relation',
            'prediction_if_rival': 'unchanged with relation',
            'distinguishing_observation': 'paired change', 'why_new': name}


def plan(items):
    return {'decision': 'continue' if items else 'stop', 'reason': 'fixed test plan',
            'investigations': items}


def specification():
    return {'score_location': 'replacement_span', 'frames': [{'prefix': 'repeatA ', 'suffix': ' repeatA'}],
            'conditions': [{'label': text, 'replacement': text,
                            'prediction_if_hypothesis': 'different', 'prediction_if_rival': 'same'}
                           for text in ('repeatA', 'repeatB')]}


def measured(identity):
    token = {'position': 0, 'token_id': 97, 'token': 'a', 'gate': 1., 'value': 2., 'product': 2.}
    return {'id': identity, 'label': 'synthetic', 'text': 'a', 'token_ids': [97], 'trace': [token],
            'selected': token, 'scored_positions': [0], 'status': 'complete',
            'score': {'location': 'replacement_span', 'reduction': 'absmax'}}


class FakeClient:
    def __init__(self, plans):
        self.plans, self.calls = plans, []

    def request(self, name, prompt, packet, schema, stage, check=None):
        self.calls.append((name, copy.deepcopy(packet)))
        if name.endswith('/plan'):
            value = self.plans[packet['round'] - 1]
        elif name.endswith('/specification'):
            value = specification()
        else:
            value = {'result': 'Synthetic comparison only', 'verdict': 'narrowed',
                     'suggested_revision': 'Restrict the proposed relation to this frame',
                     'evidence_ids': [row['id'] for row in packet['evidence']['rows']]}
        value = copy.deepcopy(value)
        validate(value, schema)
        if check:
            check(value)
        return value


class FakeExecutor:
    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []

    def __enter__(self):
        return self

    def __exit__(self, *error):
        pass

    def validate_spec(self, action, spec, identity):
        validate(spec, protocol.SPECIFICATIONS[action])

    def execute(self, action, spec, identity):
        self.calls.append(identity)
        if identity in self.fail:
            raise RuntimeError('synthetic measurement failure')
        return {'kind': action, 'rows': [measured(identity + '-case')]}


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': list(map(ord, text)), 'offset_mapping': [(i, i + 1) for i in range(len(text))]}

    def encode(self, text, **kwargs):
        value = self(text)
        return SimpleNamespace(ids=value['input_ids'], offsets=value['offset_mapping'])

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class FakeDataset:
    def __init__(self, texts, multiplier=1):
        self.scanned, self.reads, self.multiplier = False, [], multiplier
        self.token_ids, self.rows = [], []
        for identity, text in enumerate(texts):
            self.rows.append({'document_id': identity, 'token_start': len(self.token_ids), 'token_count': len(text)})
            self.token_ids.extend(map(ord, text))
        self.documents = self
        self.manifest = {'channels': ['gate', 'value', 'product'], 'tokens': len(self.token_ids),
                         'documents': len(texts)}

    def to_batches(self, **kwargs):
        yield SimpleNamespace(to_pylist=lambda: self.rows)
        self.scanned = True

    def document_activations(self, identity, neuron):
        if not self.scanned:
            raise AssertionError('Activations read before completing document selection')
        self.reads.append(identity)
        row = self.rows[identity]
        ids = self.token_ids[row['token_start']:row['token_start'] + row['token_count']]
        values = np.arange(1, len(ids) + 1, dtype=float) * self.multiplier
        return {'token_ids': ids, 'activations': np.stack([values, values, values], axis=1)}


def base_packet():
    return {'initial': {'claim': {'title': 'Frozen'}, 'response_rules': []},
            'original_criticism': [{'id': 'C1', 'attack': 'Is it a relation?'}],
            'catalog': {'saved-1': {'id': 'saved-1', 'text': 'saved context', 'activation': 1.}}}


class FollowupTests(unittest.TestCase):
    def test_two_round_limit_preserves_original_and_all_first_round_outcomes(self):
        base = base_packet()
        client = FakeClient([plan([question(f'round {r} question {i}') for i in range(5)]) for r in (1, 2)])
        executor = FakeExecutor(fail=['r1-i2'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = rounds(client, executor, base, root, DEFAULTS, Timeline(root))
        self.assertEqual([len(row['investigations']) for row in result['rounds']], [5, 5])
        second = dict(client.calls)['round-2/plan']
        self.assertEqual(second['original_criticism'], base['original_criticism'])
        first = result['rounds'][0]['investigations']
        inherited = second['previous_rounds'][0]['investigations']
        self.assertEqual([row['id'] for row in inherited], [row['id'] for row in first])
        for original, received in zip(first, inherited):
            self.assertEqual(received['question'], original['question'])
            self.assertEqual(received['status'], original['status'])
            if original['status'] == 'failed':
                self.assertEqual(received['error'], original['error'])
            else:
                self.assertEqual(received['assessment'], original['assessment'])
                self.assertEqual([(row['id'], row['selected']) for row in received['evidence']['rows']],
                                 [(row['id'], row['selected']) for row in original['evidence']['rows']])
        self.assertEqual(len(second['working_interpretation']['proposed_revisions']), 4)
        self.assertEqual(second['frozen_interpretation'], base['initial'])
        self.assertNotIn('round-3/plan', dict(client.calls))
        with self.assertRaises(ValueError):
            validate(plan([question(str(i)) for i in range(6)]), protocol.plan_schema(['C1'], ['saved-1']))

    def test_empty_plan_stops_before_or_after_first_round(self):
        for plans, expected_calls in (([plan([])], 0), ([plan([question('first')]), plan([])], 1)):
            with self.subTest(rounds=len(plans)), tempfile.TemporaryDirectory() as temporary:
                executor, client = FakeExecutor(), FakeClient(plans)
                root = Path(temporary)
                result = rounds(client, executor, base_packet(), root, DEFAULTS, Timeline(root))
                self.assertEqual(len(executor.calls), expected_calls)
                self.assertEqual(result['stop_reason'], 'no_new_distinguishable_questions')
                self.assertEqual(len(result['rounds']), len(plans))

    def test_structured_fragments_resolve_repeated_text_and_token_budget(self):
        spec, tokenizer = specification(), CharacterTokenizer()
        items = contrast_items(spec, tokenizer, 'r1-i1', 100)
        for item, condition in zip(items, spec['conditions']):
            self.assertEqual(item['text'], 'repeatA ' + condition['replacement'] + ' repeatA')
            self.assertEqual(item['span'], [8, 15])
            self.assertEqual(item['positions'], [14])
        spec['score_location'] = 'after_replacement'
        self.assertEqual(contrast_items(spec, tokenizer, 'r1-i1', 100)[0]['positions'], list(range(15, 23)))
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            contrast_items(spec, tokenizer, 'r1-i1', 8)

    def test_failed_model_load_releases_gpu_lock_before_next_investigation(self):
        active = 0

        @contextmanager
        def gpu_lock(path):
            nonlocal active
            self.assertEqual(active, 0, 'Previous failed load retained the GPU lock')
            active += 1
            try:
                yield
            finally:
                active -= 1

        with tempfile.TemporaryDirectory() as temporary, \
             patch('language.workflow.followup_execute.lock', gpu_lock), \
             patch('common.Runtime.load', side_effect=RuntimeError('synthetic allocation failure')) as load:
            root = Path(temporary)
            config = {'neuron': 'L1_N2', 'model': '/synthetic/model', 'settings': {}, 'gpu_lock': 'synthetic.lock'}
            with Executor(config, root, Timeline(root), DEFAULTS, []) as executor:
                executor.tokenizer = CharacterTokenizer()
                for identity in ('r1-i1', 'r1-i2'):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic allocation failure'):
                        executor.execute('contrast', specification(), identity)
                    self.assertEqual(active, 0)
            self.assertEqual(load.call_count, 2)

    def test_corpus_scans_all_documents_and_samples_without_activations(self):
        texts = [f'prefix CAT {i} tail dog' for i in range(20)]
        spec = {'groups': [{'label': 'cat', 'terms': ['cat']}, {'label': 'dog', 'terms': ['DOG']}]}
        datasets = [FakeDataset(texts, multiplier=value) for value in (1, -100)]
        results = [_lookup(dataset, CharacterTokenizer(), 'L1_N2', spec, seed=7, limit=3,
                           exclude_document_ids=[0, 1]) for dataset in datasets]
        self.assertEqual([row['document_id'] for row in results[0]['rows']],
                         [row['document_id'] for row in results[1]['rows']])
        for dataset, result in zip(datasets, results):
            self.assertEqual(result['scanned_documents'], 20)
            self.assertEqual(result['scanned_tokens'], sum(map(len, texts)))
            self.assertEqual(len(dataset.reads), 6)
            self.assertFalse({0, 1} & set(dataset.reads))
            self.assertEqual(result['new_model_forwards'], 0)
            for group in result['groups']:
                self.assertEqual((group['matched_documents'], group['excluded_documents'], group['eligible_documents']), (20, 2, 18))
            for row in result['rows']:
                self.assertEqual(row['causal_prefix_token_ids'], row['token_ids'][:row['selected']['position'] + 1])
                self.assertEqual(row['text'], CharacterTokenizer().decode(row['token_ids']))

    def test_corpus_literal_unicode_matching_and_empty_group(self):
        dataset = FakeDataset(['Straße [cat] trailing', 'cat without brackets'])
        result = _lookup(dataset, CharacterTokenizer(), 'L1_N2',
                         {'groups': [{'label': 'literal', 'terms': ['[CAT]', 'STRASSE']},
                                     {'label': 'empty', 'terms': ['never found']}]})
        self.assertEqual(result['rows'][0]['match']['text'], 'Straße')
        self.assertEqual(result['groups'][0]['matched_documents'], 1)
        self.assertTrue(result['groups'][1]['no_match'])
        self.assertEqual(result['groups'][1]['selected_documents'], 0)
        with Executor({}, Path('.'), None, DEFAULTS, []) as executor:
            for invalid in ([{'label': 'same', 'terms': ['cat']}] * 2,
                            [{'label': 'cat', 'terms': ['cat']}, {'label': 'blank', 'terms': [' ']}],
                            [{'label': 'one', 'terms': ['cat', 'DOG']},
                             {'label': 'two', 'terms': ['dog', 'CAT', 'cat']}]):
                with self.subTest(groups=invalid), self.assertRaises(ValueError):
                    executor.validate_spec('corpus', {'groups': invalid}, 'r1-i1')

    def test_resume_preserves_source_and_rejects_changed_execution_identity(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            root, source = Path(temporary) / 'followup', Path(temporary) / 'foundation'
            base = base_packet()
            config = {'neuron': 'L1_N2', 'model': '/synthetic/model', 'activations': '/synthetic/archive',
                      'gpus': [], 'settings': {}, 'llm': {}, 'api_slots': str(Path(temporary) / 'slots')}
            bundle = {'neuron': {'id': 'L1_N2'}, 'criticism': [{'attack': 'original'}],
                      'anatomy': {'saved': [1, 2]}, 'ablation': {'saved': [3, 4]}}
            values = {'config.json': config, 'interpretation.json': base['initial'], 'predictions.json': {'records': []},
                      'evidence.json': {'documents': [{'id': '0'}]},
                      'artifacts.json': {'initial': base['initial'], 'predictions': {'records': []},
                                         'revision': {'criticism': [{'attack': 'original'}]},
                                         'evidence': base['catalog']}, 'status.json': {'phase': 'complete'}}
            for name, value in values.items():
                write(source / name, value)
            (source / 'report').mkdir()
            (source / 'report/index.html').write_text('<script id="bundle">' + json.dumps(bundle) + '</script>')
            before = {name: (source / name).read_bytes() for name in SOURCE_FILES}
            client, executor = FakeClient([plan([question('first')]), plan([])]), FakeExecutor()
            kwargs = {'client_factory': lambda *args: client, 'executor_factory': lambda *args: executor}
            with patch('language.workflow.followups.input_identity', return_value={'synthetic': 1}), \
                 patch('language.workflow.followups.render', side_effect=RuntimeError('synthetic render interruption')):
                with self.assertRaises(RuntimeError):
                    run(source, root, **kwargs)
            self.assertEqual(read(root / 'status.json')['phase'], 'failed')
            (root / 'frozen.json').unlink()
            calls_before = len(executor.calls)
            with patch('language.workflow.followups.input_identity', return_value={'synthetic': 1}):
                output = run(source, root, **kwargs)
                self.assertTrue((output / 'index.html').is_file())
                self.assertEqual(read(root / 'status.json')['phase'], 'complete')
                self.assertEqual(read(root / 'frozen.json')['initial'], base['initial'])
                self.assertEqual(len(executor.calls), calls_before)
                with self.assertRaises(ValueError):
                    run(source, root, {'max_tokens': 512}, **kwargs)
                calls_after = len(client.calls)
                run(source, root, **kwargs)
                self.assertEqual(len(client.calls), calls_after)
            with patch('language.workflow.followups.input_identity', return_value={'synthetic': 2}):
                with self.assertRaises(ValueError):
                    run(source, root, **kwargs)
            self.assertEqual(before, {name: (source / name).read_bytes() for name in SOURCE_FILES})
            changed = read(source / 'evidence.json')
            changed['documents'].append({'id': '1'})
            write(source / 'evidence.json', changed)
            with patch('language.workflow.followups.input_identity', return_value={'synthetic': 1}):
                with self.assertRaises(ValueError):
                    run(source, root, **kwargs)

    def test_report_only_appends_criticism_and_preserves_failed_status(self):
        bundle = {'claim': {'title': 'Frozen'}, 'criticism': [{'attack': 'Original', 'round': 1}],
                  'anatomy': {'data': [1, 2]}, 'ablation': {'effects': [3, 4]}, 'search': ['saved']}
        source = '<script id="bundle">' + json.dumps(bundle) + '</script><p>unchanged layout</p>'
        record = {'id': 'r1-i1', 'question': question('new question'), 'status': 'complete',
                  'assessment': {'result': 'synthetic result', 'verdict': 'narrowed',
                                 'suggested_revision': 'synthetic revision', 'evidence_ids': ['case', 'graph:1']},
                  'evidence': {'rows': [measured('case'), {'id': 'graph:1', 'kind': 'saved_graph', 'data': {}}]}}
        failed = {'id': 'r1-i2', 'question': question('failed question'), 'status': 'failed', 'error': 'synthetic error'}
        with tempfile.TemporaryDirectory() as temporary:
            output = render(source, [{'round': 1, 'investigations': [record, failed]}], Path(temporary))
            page = (output / 'index.html').read_text()
        rendered = json.loads(BUNDLE.search(page)[2])
        self.assertEqual({key: value for key, value in rendered.items() if key != 'criticism'},
                         {key: value for key, value in bundle.items() if key != 'criticism'})
        self.assertEqual(rendered['criticism'][0], bundle['criticism'][0])
        self.assertEqual(rendered['criticism'][1]['evidence_ids'], ['case', 'graph:1'])
        self.assertEqual([row['activation'] for row in rendered['criticism'][1]['evidence']], [2.])
        self.assertEqual(rendered['criticism'][-1]['verdict'], 'inconclusive')
        self.assertEqual(rendered['criticism'][-1]['evidence'], [])
        self.assertEqual(page.split('</script>', 1)[1], '<p>unchanged layout</p>')


if __name__ == '__main__':
    unittest.main()
