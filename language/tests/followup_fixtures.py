"""Synthetic follow-up inputs shared by offline execution and resume tests."""
import copy
from types import SimpleNamespace

import numpy as np

from language.workflow import followup_protocol as protocol
from language.workflow.contracts import validate


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
