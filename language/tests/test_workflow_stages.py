"""Offline orchestration checks for search and local/worker attribution stages."""
import copy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from language.workflow.attribution import run as attribute
from language.workflow.search import run as search
from language.workflow.state import read, write


def measured(identity, product):
    token = {'position': 0, 'token': identity, 'product': product}
    return {'id': identity, 'label': identity, 'text': identity,
            'selected': token, 'trace': [token]}


class WorkflowStageTests(unittest.TestCase):
    def test_search_preserves_round_feedback_and_measurement_paths(self):
        proposal = {'regions': [
            {'id': pole, 'name': pole, 'pole': pole, 'rationale': 'fixture',
             'candidates': [{'label': pole, 'text': pole}]}
            for pole in ('positive', 'negative')]}
        client = SimpleNamespace(request=Mock(return_value=copy.deepcopy(proposal)))
        runtime = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text, **kw: [1, 2]))
        names, spans = [], []

        def texts(items, name):
            names.append(name)
            return [{**item, **measured(item['id'], 2 if item['label'] == 'positive' else -3),
                     'text': item['text'], 'label': item['label']} for item in items]

        measure = SimpleNamespace(runtime=runtime, texts=texts)
        config = {'search_rounds': 2, 'search_regions': 2, 'candidates_per_region': 1}
        packet = {'statistics': {'product': {'min': -10, 'max': 10}}}
        with tempfile.TemporaryDirectory() as folder, \
             patch('language.workflow.search.batch_comparison', return_value={'difference': 0}) as compare:
            root = Path(folder)

            def span(name):
                spans.append(name)
                return nullcontext()

            rows, regions, progress = search(client, measure, {'claim': 'frozen'}, packet,
                copy.deepcopy(proposal), config, root, SimpleNamespace(status=Mock()), span)
            self.assertEqual(names, ['search-round-1', 'search-round-2'])
            self.assertEqual([row['id'] for row in rows], [
                'search-1-positive-0', 'search-1-negative-0', 'search-2-positive-0', 'search-2-negative-0'])
            self.assertEqual([len(region['candidates']) for region in regions.values()], [2, 2])
            self.assertEqual(read(root / 'search-progress.json'), progress)
            feedback = client.request.call_args.args[2]
            self.assertEqual([row['id'] for row in feedback['previous_round']], [row['id'] for row in rows[:2]])
            self.assertEqual(feedback['natural_extremes'], packet['statistics']['product'])
            search(client, measure, {'claim': 'frozen'}, packet, copy.deepcopy(proposal),
                   config, root, SimpleNamespace(status=Mock()), span)
            compare.assert_called_once_with(runtime, [[1, 2], [1, 2]])
            self.assertEqual(spans, ['batch-comparison'])

    def test_attribution_preserves_local_worker_calls_and_graph_checkpoints(self):
        for use_worker in (False, True):
            with self.subTest(worker=use_worker), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                runtime = object()
                config = {'atlas_natural': 2, 'atlas_written': 2,
                          'settings': {'inputs': {'max_written_tokens': 128}}}
                request = {'model': 'fixture', 'neuron': {'layer': 1, 'index': 2},
                           'settings': config['settings']}
                positives = [measured('positive', 5), measured('negative', -4)]
                controls = [measured('control', 3)]
                calls, spans = [], []

                def sections(payload, destination, actual_runtime):
                    self.assertIs(actual_runtime, runtime)
                    destination = Path(destination)
                    calls.append(('sections', payload))
                    write(destination / 'sections.json', {'circuit': {'fixture': True}})
                    write(destination / 'atlas-raw.json', [{'id': 'atlas-case'}])
                    return {'status': 'complete'}

                def graph(payload, destination, actual_runtime):
                    self.assertIs(actual_runtime, runtime)
                    destination = Path(destination)
                    calls.append(('attribute', payload))
                    write(destination / 'graph.json', {'id': payload['id'], 'text': payload['text']})
                    result = {'status': 'complete', 'steering': {}}
                    write(destination / 'result.json', result)
                    return result

                def remote_call(operation, **payload):
                    handler = sections if operation == 'sections' else graph
                    return handler(payload['request'], payload['folder'], runtime)

                worker = SimpleNamespace(call=remote_call) if use_worker else None
                timeline = SimpleNamespace(status=Mock())

                def span(name):
                    spans.append(name)
                    return nullcontext()

                with patch('language.workflow.attribution.natural_cases', return_value=['natural']), \
                     patch('paper_sections.prepare', side_effect=lambda plan, report: plan), \
                     patch('paper_sections.run', side_effect=sections) as local_sections, \
                     patch('model_tools.attribute', side_effect=graph) as local_graph:
                    first = attribute(config, root, root / 'source', {}, request,
                        {'family_sentences': ['family']}, positives, controls, runtime, worker, timeline, span)
                    second = attribute(config, root, root / 'source', {}, request,
                        {'family_sentences': ['family']}, positives, controls, runtime, worker, timeline, span)
                self.assertEqual(first, second)
                self.assertEqual([kind for kind, _ in calls], ['sections', 'attribute', 'attribute', 'attribute', 'sections'])
                self.assertEqual([payload['text'] for kind, payload in calls if kind == 'attribute'],
                                 ['positive', 'negative', 'control'])
                self.assertEqual(spans, ['atlas-and-family', 'graph-1', 'graph-2', 'graph-3', 'atlas-and-family'])
                self.assertEqual([row['steering']['activation_context']['activation'] for row in first[3]], [5, -4, 3])
                self.assertEqual((local_sections.call_count, local_graph.call_count), (0, 0) if use_worker else (2, 3))
                self.assertEqual(read(root / 'atlas/tool-request.json')['plan']['atlas']['cases'], [
                    {'case_id': 'natural'}, {'text': 'positive', 'position': 0}, {'text': 'negative', 'position': 0}])


if __name__ == '__main__':
    unittest.main()
