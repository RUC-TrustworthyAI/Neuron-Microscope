"""Followup provenance, registration, and recovery through the real workflow."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from language.tests.test_criticism_operations import Runtime, probe, readout
from language.workflow import contracts, followups
from language.workflow.run import investigate_criticism
from language.workflow.state import Timeline, read


INITIAL = {'claim': {'title': 'Cue raises activation'}, 'response_rules': []}
PREDICTIONS = {'version': 'fixture', 'records': [], 'response_rules': [], 'summary': {}}
CUE, CONTROL = 'has cue!', 'no cue!'


def study(identity='contrast', probes=None):
    return {'id': identity, 'source_ids': ['criticism-1', 'criticism-2', 'criticism-3'], 'parent_ids': [],
            'claim_id': 'claim-0', 'evidence_ids': [], 'attack': 'Does the earlier cue matter?',
            'prediction': 'The cue raises the response at the shared target.',
            'probes': probes if probes is not None else [probe('cue', CUE), probe('control', CONTROL)],
            'graph_queries': [], 'resamples': [], 'quantiles': []}


def round_plan(*studies):
    return {'action': 'experiment' if studies else 'stop', 'reason': 'No further open question.',
            'investigations': list(studies), 'deferred': []}


class ScriptedClient:
    def __init__(self):
        sections = contracts.REPORT_ANALYSIS['properties']['section_prose']['properties']
        self.responses = {
            'report-analysis': {'section_prose': dict.fromkeys(sections, 'Fixture.'),
                'criticism': [{'attack': attack, 'prediction': 'A matched control may split.',
                              'result': 'Untested contrast.', 'verdict': 'inconclusive', 'evidence_ids': ['saved']}
                             for attack in ('The cue may be incidental.', 'The target might matter.',
                                            'The effect may depend on the context.')],
                'clusters': [], 'units': [], 'graph_captions': []},
            'inspect': {'reason': 'No saved read needed.', 'probes': [], 'graph_queries': [], 'quantiles': []},
            'followups/round-1/plan': round_plan(study()),
            'plan': round_plan(),
            'review': {'verdict': 'survived', 'result': 'Measured the comparison.',
                       'suggested_revision': '', 'next_question': '',
                       'evidence_metrics': []},
        }
        self.calls = {}

    def request(self, name, prompt, packet, schema, stage, check=None):
        self.calls[name] = copy.deepcopy(packet)
        kind = name.rsplit('/', 1)[-1].split('-')[0]
        response = self.responses[name] if name in self.responses else self.responses[kind]
        value = copy.deepcopy(response(packet, schema) if callable(response) else response)
        contracts.validate(value, schema)
        if check:
            check(value)
        return value


class FollowupWorkflowTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.runtime, self.client = Runtime(), ScriptedClient()
        self.runtime.expected = {text: [0.] * (len(text) - 1) + [value]
                                 for text, value in [(CUE, 2.), (CONTROL, 1.)]}
        self.case = {'id': 'saved', 'text': 'x!', 'trace': [
            {'position': i, 'token': token, 'token_id': ord(token), 'offset': [i, i + 1],
             'gate': 1., 'value': float(i), 'product': float(i)} for i, token in enumerate('x!')]}
        self.case['selected'] = self.case['trace'][1]
        self.archive = {'traces': {'saved': self.case}}

    def invoke(self, options=None):
        return investigate_criticism(self.client, type('Handle', (), {'runtime': self.runtime})(),
            INITIAL, PREDICTIONS, [], {'atlas': []}, {'saved': {'text': 'x!', 'activation': 1.}},
            self.root, Timeline(self.root), followups.settings({'max_rounds': 1, **(options or {})}),
            {'model': 'fixture'}, {'min': 0., 'max': 2.}, archive=self.archive)

    def test_resume_reuses_measurements_and_rejects_changed_archive(self):
        forward = self.runtime.batch_forward

        def interrupt(inputs):
            if self.runtime.tokenizer.decode(inputs[0]) == CUE:
                raise RuntimeError('Simulated GPU interruption')
            return forward(inputs)

        with patch.object(self.runtime, 'batch_forward', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'GPU interruption'):
                self.invoke()
        self.assertEqual(self.runtime.forwards, [CONTROL])  # Shortest input is measured first.
        _, summary, rows = self.invoke()
        self.assertEqual(len(summary['rounds'][0]['investigations']), 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.runtime.forwards, [CONTROL, CUE])
        self.case['trace'][0]['product'] = 2.
        with self.assertRaisesRegex(ValueError, 'archived evidence changed'):
            self.invoke()
        self.assertEqual(self.runtime.forwards, [CONTROL, CUE])

    def test_unexecutable_design_does_not_cancel_valid_sibling(self):
        self.client.responses['followups/round-1/plan'] = round_plan(
            study('too-long', probes=[probe('long', 'x' * 100)]), study())
        _, summary, rows = self.invoke({'max_input_tokens': 32})
        step, = summary['rounds']
        self.assertEqual([item['plan']['id'] for item in step['investigations']], ['contrast'])
        unavailable, = step['unavailable']
        self.assertEqual(unavailable['id'], 'too-long')
        self.assertEqual(len(rows), 2)
        registered = read(self.root / 'followups/round-1/registered.json')['registration']
        self.assertIsNone(registered['designs'][0]['operations'])
        self.assertTrue(registered['designs'][0]['error'])

    def test_disabled_followups_keep_upstream_analysis(self):
        revision, summary, rows = self.invoke({'max_rounds': 0})
        self.assertEqual(revision['criticism'], self.client.responses['report-analysis']['criticism'])
        self.assertEqual(revision['claim'], INITIAL['claim'])
        self.assertEqual(summary['status'], 'disabled')
        self.assertEqual(rows, [])
        self.assertEqual(list(self.client.calls), ['report-analysis'])

    def test_saved_inspection_informs_design_and_next_round_uses_local_revision(self):
        case_id = 'followup-1-1-forward-1'
        self.archive['traces'][case_id] = {**self.case, 'id': case_id}

        def inspect(packet, schema):
            return {'reason': 'Locate the stored peak.', 'graph_queries': [], 'quantiles': [],
                    'probes': [probe('stored', case_id='saved', readouts=[readout('position', metric='peak_position')])]}

        def plan(packet, _):
            self.assertEqual(self.runtime.forwards, [])
            position = packet['saved_inspection']['metrics']['stored.position']
            text = 'a' * 140 + f'Read position {position}!' + 'b' * 140
            self.runtime.expected[text] = [1.] * len(text)
            return round_plan(study(probes=[probe('adapted', text)]),
                              study('reread', probes=[probe('saved-collision', case_id=case_id)]))

        def next_plan(packet, _):
            self.assertEqual(packet['previous_rounds'][0]['investigations'][0]['revision']['id'],
                             'claim-contrast')
            self.assertEqual(packet['working_claims'][-1]['investigation_id'], 'contrast')
            return round_plan(study('revised') | {'claim_id': 'claim-contrast', 'parent_ids': []})

        review = self.client.responses['review'] | {
            'suggested_revision': 'Cue effect is local to the tested context.'}
        self.client.responses.update({'followups/round-1/inspect': inspect,
            'followups/round-1/plan': plan, 'followups/round-1/review-1': review,
            'followups/round-2/plan': next_plan})
        _, summary, rows = self.invoke({'max_rounds': 2, 'max_input_tokens': 512})
        self.assertEqual([row['text'] for row in rows], ['a' * 140 + 'Read position 1!' + 'b' * 140, CUE, CONTROL])
        self.assertEqual(len(summary['rounds']), 2)
        reread = summary['rounds'][0]['investigations'][1]
        self.assertEqual(reread['metrics']['saved-collision.peak'], 1.)
        review = self.client.calls['followups/round-1/review-1']
        self.assertEqual(review['conditions'][0]['text'], rows[0]['text'])
        self.assertEqual(review['saved_inspection']['metrics']['stored.position'], 1)
        self.assertEqual(followups.history_for_planning(summary['rounds'])[0]['saved_inspection'],
                         review['saved_inspection'])
        self.assertEqual(self.client.calls['followups/round-2/review-1']['parent_results'][0]['id'], 'contrast')

    def test_round_limits_allow_early_stop(self):
        options = followups.settings({})
        self.assertEqual(options['max_rounds'], 2)
        with self.assertRaises(ValueError):
            followups.settings({'max_rounds': 3})
        self.assertEqual([followups.round_schema(options, r)['properties']['investigations']['maxItems']
                          for r in (1, 2)], [6, 5])
        _, summary, _ = self.invoke({'max_rounds': 2})
        self.assertEqual(summary['status'], 'stopped')
        self.assertEqual(len(summary['rounds']), 1)
