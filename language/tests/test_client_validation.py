"""The correction request must identify all invalid references in a response."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from language.examples.export_walkthrough import recorded_request
from language.workflow import contracts, prompts
from language.workflow.client import Client
from language.workflow.followup_protocol import plan_schema
from language.workflow.state import write


class ClientValidationTests(unittest.TestCase):
    def test_resume_repairs_unambiguous_json_and_preserves_the_accepted_attempt(self):
        schema = contracts.obj(id=contracts.S)
        expected = {'id': 'saved'}
        answer = json.dumps(expected)
        messages = [{'role': 'system', 'content': 'prompt' + prompts.SCHEMA_HEADER + json.dumps(schema)},
                    {'role': 'user', 'content': '{}'}]
        cases = ((answer + '}', False), (answer + answer, False),
                 ('```json\n' + answer + '\n```', False),
                 (answer + '{"id":"different"}', True), ('```json', True))
        for raw, needs_retry in cases:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as folder:
                client = Client({}, Path(folder) / 'api', Path(folder) / 'slots')
                call = client.root / 'report-synthesis'
                first = call / 'attempt-001'
                request = {'messages': messages}
                write(first / 'request.json', request)
                write(first / 'completion.json', {'finish_reason': 'stop'})
                (first / 'output.txt').write_text(raw)
                write(first / 'format-error.json', {'error': 'Extra data'})

                def stream(retry, destination, stage):
                    write(destination / 'request.json', {'messages': retry})
                    return answer

                with patch.object(client, '_stream', side_effect=stream) as api:
                    self.assertEqual(client.request('report-synthesis', 'prompt', {}, schema, 1), expected)
                self.assertEqual(api.call_count, int(needs_retry))
                saved = json.loads((call / 'result.json').read_text())
                self.assertEqual(saved['attempt'], 'attempt-002' if needs_retry else first.name)
                if needs_retry:
                    self.assertIn('Return exactly one JSON value', recorded_request(call, saved)['messages'][-1]['content'])
                else:
                    self.assertEqual(recorded_request(call, saved), request)
                self.assertTrue((first / 'format-error.json').is_file())

    def test_correction_reports_all_unknown_evidence_ids(self):
        plan = {'decision': 'continue', 'reason': 'Check the result', 'investigations': [{
            'question': 'Does the response generalize?', 'criticism_ids': ['C1'],
            'action': 'review', 'evidence_ids': ['wrong-1', 'valid', 'wrong-2'],
            'hypothesis': 'A', 'rival': 'B', 'prediction_if_hypothesis': 'A',
            'prediction_if_rival': 'B', 'distinguishing_observation': 'C', 'why_new': 'D'}]}
        corrected = json.loads(json.dumps(plan))
        corrected['investigations'][0]['evidence_ids'] = ['valid']
        schema = plan_schema(['C1'], ['valid'])
        for resume in (False, True):
            with self.subTest(resume=resume), tempfile.TemporaryDirectory() as folder:
                client = Client({}, Path(folder) / 'api', Path(folder) / 'slots')
                messages = []

                def stream(request, destination, stage, max_tokens=None):
                    messages.append(json.loads(json.dumps(request)))
                    return json.dumps(corrected if resume or len(messages) == 2 else plan)

                if resume:
                    attempt = client.root / 'plan/attempt-001'
                    write(attempt / 'request.json', {'messages': [
                        {'role': 'system', 'content': 'prompt\nJSON schema:\n' + json.dumps(schema)},
                        {'role': 'user', 'content': '{}'}]})
                    write(attempt / 'completion.json', {'finish_reason': 'stop'})
                    (attempt / 'output.txt').write_text(json.dumps(plan))
                    # Old logs reported only the first error. Resume must recompute all errors.
                    write(attempt / 'format-error.json', {'error': 'wrong-1'})
                with patch.object(client, '_stream', side_effect=stream):
                    output = client.request('plan', 'prompt', {}, schema, 1)
                self.assertEqual(output, corrected)
                self.assertEqual(len(messages), 1 if resume else 2)
                self.assertIn('wrong-1', messages[-1][-1]['content'])
                self.assertIn('wrong-2', messages[-1][-1]['content'])

    def test_invalid_correction_is_rejected_and_not_cached(self):
        schema = {'type': 'string', 'enum': ['saved-id']}
        with tempfile.TemporaryDirectory() as folder:
            client = Client({}, Path(folder) / 'api', Path(folder) / 'slots')
            with patch.object(client, '_stream', return_value='"invented-id"') as stream:
                with self.assertRaisesRegex(ValueError, 'invented-id'):
                    client.request('invalid', 'prompt', {}, schema, 1)
            self.assertEqual(stream.call_count, 2)
            self.assertFalse((client.root / 'invalid/result.json').exists())


if __name__ == '__main__':
    unittest.main()
