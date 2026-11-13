import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from language.workflow.ablation_run import compatible_plan
from language.workflow.interventions import measure_ablations, measure_case
from language.workflow.revision import ablation_analysis_packet, resume_graph_ablation_packet
from language.workflow.state import write


class AblationMeasurementTests(unittest.TestCase):
    def test_plan_upgrade_only_accepts_new_controlled_next_token(self):
        case = {'kind': 'controlled', 'next_token_id': 0, 'token_ids': [2]}
        plan = {'cases': [case], 'control_neurons': [1]}
        saved = {**plan, 'cases': [{**case, 'next_token_id': None}]}
        self.assertTrue(compatible_plan(saved, plan))
        self.assertFalse(compatible_plan({**saved, 'control_neurons': [2]}, plan))
        self.assertFalse(compatible_plan({**saved, 'cases': [{**case, 'next_token_id': 1}]}, plan))
        natural = {**case, 'kind': 'weak'}
        self.assertFalse(compatible_plan({**plan, 'cases': [{**natural, 'next_token_id': None}]},
                                         {**plan, 'cases': [natural]}))

    def test_completed_checkpoint_adds_controlled_readout_without_new_forwards(self):
        runtime = SimpleNamespace(torch=torch, tokenizer=SimpleNamespace(decode=lambda ids: str(ids)),
                                  forwards=0, telemetry=lambda: {})
        case = {'case_id': 'controlled-one', 'kind': 'controlled', 'position': 0,
                'token_ids': [2], 'next_token_id': 0}
        plan = {'cases': [case], 'control_neurons': [1], 'requested': {'controlled': 1},
                'realized': {'controlled': 1}}
        saved_row = dict(case_id=case['case_id'], kind='controlled', peak_index=0,
                         next_token_id=None, next_token=None, next_logprob_base=None,
                         next_logprob_ablated=None, control_neurons=[1], kl=.03, kl_controls=[.01])
        baseline, target = np.array([0., 2.]), np.array([1., 2.])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint = folder / 'cases/001/result.json'
            write(checkpoint, {'schema_version': 1, 'status': 'complete', 'ablation': saved_row})
            raw = checkpoint.with_name('logits.npz')
            np.savez_compressed(raw, baseline=baseline, target=target, controls=np.array([[0., 2.]]))
            before = raw.read_bytes()
            with patch('language.workflow.interventions.measure_case') as forward:
                result = measure_ablations(runtime, plan, folder)
                self.assertEqual(measure_ablations(runtime, plan, folder), result)
                forward.assert_not_called()
            self.assertEqual(raw.read_bytes(), before)
        row = result['rows'][0]
        self.assertEqual(row['next_token'], '[0]')
        self.assertAlmostEqual(row['next_logprob_base'], -np.logaddexp(0., 2.))
        self.assertAlmostEqual(row['next_logprob_ablated'], 1. - np.logaddexp(1., 2.))
        self.assertEqual(result['summary']['median_kl'], .03)

    def test_analysis_examples_expose_only_citable_source_ids(self):
        common = dict(kind='controlled', text='Example', peak_token='.', live_product=1.,
                      kl_controls=[0.01], gains=[], losses=[])
        rows = [dict(common, case_id='case-one', source_id='source-one', kl=0.1),
                dict(common, case_id='case-two', source_id='source-two', kl=0.2)]
        examples = ablation_analysis_packet({'rows': rows, 'summary': {}}, {'source-one': {}})['examples']
        self.assertEqual([example['evidence_id'] for example in examples], [None, 'source-one'])
        self.assertTrue(all('case_id' not in example and 'source_id' not in example for example in examples))

    def test_baseline_tie_uses_the_displayed_token_for_ablated_probability(self):
        logits = [torch.tensor(values, dtype=torch.float32)
                  for values in [[3, 3, 3, 3, 1, 1, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7], [0] * 8]]
        runtime = SimpleNamespace(torch=torch, neuron={'index': 2},
                                  tokenizer=SimpleNamespace(decode=lambda ids, **_: str(ids)))
        case = {'case_id': 'controlled-one', 'token_ids': [6, 5], 'position': 1,
                'next_token_id': None, 'kind': 'controlled', 'saved_product': 2.0}
        baseline = {'product': [1., 2.]}
        edits = [{'edit': {'neuron': index, 'position': 1, 'after': 0., 'before': 2.,
                           'entries_targeted': 1}} for index in [2, 1]]
        with tempfile.TemporaryDirectory() as folder:
            with patch('model_tools.point_logits', side_effect=list(zip(logits, [baseline, *edits]))) as run:
                record = measure_case(runtime, case, [1], Path(folder))
            self.assertEqual([call.args for call in run.call_args_list],
                             [(runtime, [6, 5], 1), (runtime, [6, 5], 1, 2), (runtime, [6, 5], 1, 1)])
            with np.load(Path(folder) / record['logit_arrays']) as saved:
                np.testing.assert_array_equal(saved['target'], logits[1].numpy())
        row = record['ablation']
        displayed = row['top_base'][0]['token_id']
        self.assertAlmostEqual(row['baseline_top_logprob_ablated'],
                               logits[1].double().log_softmax(-1)[displayed].item())

    def test_resume_accepts_only_the_legacy_packet_used_by_cached_analysis(self):
        legacy = {'rows': [{'case_id': 'graph-1'}], 'replaces_case_ids': ['graph-1']}
        messages = [{'role': 'system', 'content': 'prompt'},
                    {'role': 'user', 'content': json.dumps({'matched_ablations': legacy})}]
        fingerprint = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            analysis = root / 'api/report-analysis'
            write(analysis / 'result.json', {'input_id': fingerprint})
            write(analysis / 'attempt-001/request.json', {'messages': messages[:1]})
            packet = {'matched_ablations': {'summary': {'n': 80}}}
            self.assertFalse(resume_graph_ablation_packet(root, packet))
            write(analysis / 'attempt-002/request.json', {'messages': messages})
            self.assertTrue(resume_graph_ablation_packet(root, packet))
            self.assertEqual(packet['matched_ablations'], legacy)
