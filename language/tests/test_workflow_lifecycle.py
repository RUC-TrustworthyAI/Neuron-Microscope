"""Offline checks for the complete workflow's status, resume and lock boundary."""
import os
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

from language.tests.followup_fixtures import FakeClient, FakeExecutor, base_packet, plan, question
from language.tools.report import encode_bundle, strip_report_synthesis, update_report_synthesis
from language.workflow.followups import FollowupTimeline, run as investigate_followups
from language.workflow.run import run
from language.workflow.state import Timeline, read, write


class WorkflowLifecycleTests(unittest.TestCase):
    @staticmethod
    def foundation(config, root, stop_after=None, *, checkpoint_attribution=False):
        timeline = Timeline(root)
        with timeline.span('foundation', 'execution', 0):
            write(root / 'route-prediction/result.json', {'fixture': 'routes'})
        timeline.status('interpretation_complete' if stop_after else 'foundation_complete')
        return root / 'report'

    def test_direct_run_finishes_after_synthesizing_followups(self):
        for phase in ('complete', 'complete_with_failures'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as folder:
                root, config = Path(folder), {'neuron': 'L1_N2'}
                events = []

                def followups(source, output, *, in_place, workflow_timeline):
                    self.assertEqual((source, output, in_place), (root, root / 'followup', True))
                    self.assertEqual(read(root / 'status.json')['phase'], 'foundation_complete')
                    progress = FollowupTimeline(output, workflow_timeline)
                    for step in ('planning', 'investigating'):
                        progress.status(step, round=1)
                        status = read(root / 'status.json')
                        self.assertEqual((status['phase'], status['step'], status['round']),
                                         ('followup', step, 1))
                    progress.status(phase, failures=int(phase != 'complete'), stop_reason='fixture')
                    events.append('followup')
                    return root / 'report'

                def finalize(source, result, actual_config, *, followup):
                    self.assertEqual((source, result, actual_config),
                                     (root, {'fixture': 'routes'}, config))
                    self.assertTrue(followup)
                    self.assertEqual(read(root / 'status.json')['phase'], 'followup')
                    events.append('synthesis')
                    return root / 'report'

                with patch('language.workflow.run._run_foundation', side_effect=self.foundation) as base, \
                     patch('language.workflow.followups.run', side_effect=followups), \
                     patch('language.workflow.render.finalize_report', side_effect=finalize):
                    self.assertEqual(run(config, root, checkpoint_attribution=True), root / 'report')
                status = read(root / 'status.json')
                self.assertEqual(status['phase'], phase)
                self.assertEqual(status['failures'], int(phase != 'complete'))
                self.assertEqual(status['stop_reason'], 'fixture')
                self.assertEqual(events, ['followup', 'synthesis'])
                base.assert_called_once_with(config, root, stop_after=None, checkpoint_attribution=True)

    def test_saved_artifacts_resume_still_runs_default_followups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'run'
            report = Path(folder) / 'initial-report'
            config = {'report': str(report), 'llm': {}, 'api_slots': str(Path(folder) / 'slots'),
                      'model': 'fixture', 'settings': {}, 'neuron': 'L1_N2', 'gpus': []}
            write(report / 'report.json', {'identity': {'layer': 1, 'neuron': 2}})
            write(root / 'artifacts.json', {'atlas_sections': {}})
            write(root / 'route-prediction/result.json', {'fixture': 'routes'})
            (root / 'report').mkdir()
            (root / 'report/index.html').write_text('Completed foundation report')

            def followups(source, output, *, in_place, workflow_timeline):
                self.assertEqual(read(root / 'status.json')['phase'], 'foundation_complete')
                FollowupTimeline(output, workflow_timeline).status('complete', failures=0)
                return source / 'report'

            with patch('language.workflow.run.Client'), \
                 patch('language.workflow.run.predict_routes', return_value={'fixture': 'routes'}), \
                 patch('language.workflow.run.neuron_labels.complete'), \
                 patch('language.workflow.render.assemble') as assemble, \
                 patch('language.workflow.render.update_routes', return_value=root / 'report'), \
                 patch('language.workflow.render.finalize_report', return_value=root / 'report') as finalize, \
                 patch('language.workflow.followups.run', side_effect=followups) as investigate:
                self.assertEqual(run(config, root), root / 'report')
            assemble.assert_not_called()
            investigate.assert_called_once()
            finalize.assert_called_once_with(root, {'fixture': 'routes'}, config, followup=True)

    def test_partial_and_opt_out_runs_skip_followups(self):
        for stop_after, followup in (('interpretation', None), (None, False)):
            with self.subTest(stop_after=stop_after), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with patch('language.workflow.run._run_foundation', side_effect=self.foundation), \
                     patch('language.workflow.followups.run') as investigate, \
                     patch('language.workflow.render.finalize_report', return_value=root / 'report') as finalize:
                    self.assertEqual(run({}, root, stop_after=stop_after, followup=followup), root / 'report')
                investigate.assert_not_called()
                if stop_after is None:
                    finalize.assert_called_once_with(root, {'fixture': 'routes'}, {}, followup=False)
                else:
                    finalize.assert_not_called()
                self.assertEqual(read(root / 'status.json')['phase'],
                                 'interpretation_complete' if stop_after else 'complete')

    def test_failed_synthesis_preserves_previously_completed_report(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            root, report = Path(folder) / 'run', Path(folder) / 'initial-report'
            config = {'report': str(report), 'llm': {}, 'api_slots': str(Path(folder) / 'slots'),
                      'model': 'fixture', 'settings': {}, 'neuron': 'L1_N2', 'gpus': []}
            base = base_packet()
            predictions = {'records': []}
            criticism = {'attack': 'Original criticism'}
            write(report / 'report.json', {'identity': {'layer': 1, 'neuron': 2}})
            write(root / 'artifacts.json', {'initial': base['initial'], 'predictions': predictions,
                                           'revision': {'criticism': [criticism]}, 'evidence': base['catalog']})
            write(root / 'interpretation.json', base['initial'])
            write(root / 'predictions.json', predictions)
            write(root / 'evidence.json', {'documents': []})
            routes = {'rows': []}
            write(root / 'route-prediction/result.json', routes)
            html = ('<script id="bundle">' + encode_bundle({'neuron': {'id': 'L1_N2'}, 'criticism': [criticism]})
                    + '</script><script>const B = JSON.parse(document.getElementById("bundle").textContent);'
                    'const POS = {}\n// 7. Tests.\n// 8. Criticism.</script></body>')
            page, bundle = update_report_synthesis(html, {'claim': {'title': 'Previous synthesis'},
                                                          'section_prose': {}, 'evidence': {}})
            self.assertEqual(strip_report_synthesis(page), html)
            output = root / 'report'
            write(output / 'paper-data.json', {'bundle': bundle})
            (output / 'index.html').write_text(page, encoding='utf-8')
            (output / 'interpretation-costs.json').write_text('{}', encoding='utf-8')
            files = ('index.html', 'paper-data.json', 'interpretation-costs.json')
            before = [(output / name).read_bytes() for name in files]
            client, executor = FakeClient([plan([question('first')]), plan([])]), FakeExecutor()
            followups = partial(investigate_followups, client_factory=lambda *args: client,
                                executor_factory=lambda *args: executor)

            with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'tools'), *sys.path]), \
                 patch('language.workflow.run.predict_routes', return_value=routes), \
                 patch('language.workflow.run.neuron_labels.complete'), \
                 patch('language.workflow.followups.input_identity', return_value={'fixture': True}), \
                 patch('language.workflow.followups.run', side_effect=followups), \
                 patch('language.workflow.report_synthesis.synthesize',
                       side_effect=RuntimeError('synthetic synthesis failure')) as synthesize:
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic synthesis failure'):
                        run(config, root)
                    self.assertEqual([(output / name).read_bytes() for name in files], before)
                    self.assertEqual(read(root / 'followup/status.json')['phase'], 'complete')
                    self.assertEqual(read(root / 'status.json')['phase'], 'failed')
            self.assertEqual(len(executor.calls), 1)
            for call in synthesize.call_args_list:
                self.assertEqual(len(call.args[1]['criticism']), 2)

    def test_cached_partial_run_preserves_report_and_existing_failure_status(self):
        for phase in ('complete_with_failures', 'failed'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as folder:
                root, report = Path(folder) / 'run', Path(folder) / 'initial-report'
                config = {'report': str(report), 'llm': {}, 'api_slots': str(Path(folder) / 'slots'),
                          'model': 'fixture', 'settings': {}, 'neuron': 'L1_N2', 'gpus': []}
                write(report / 'report.json', {'identity': {'layer': 1, 'neuron': 2}})
                write(root / 'artifacts.json', {'atlas_sections': {}})
                write(root / 'status.json', {'phase': phase, 'error': 'saved follow-up failure'})
                (root / 'report').mkdir()
                html = root / 'report/index.html'
                original = b'Completed foundation and saved follow-up report'
                html.write_bytes(original)
                previous_status = (root / 'status.json').read_bytes()
                with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'tools'), *sys.path]), \
                     patch('language.workflow.run.Client'), \
                     patch('language.workflow.sampling.prepare') as sample, \
                     patch('language.workflow.followups.run') as investigate:
                    self.assertEqual(run(config, root, stop_after='interpretation'), root / 'report')
                sample.assert_not_called()
                investigate.assert_not_called()
                self.assertEqual(html.read_bytes(), original)
                self.assertEqual((root / 'status.json').read_bytes(), previous_status)

    def test_stage_failures_write_root_failure_without_premature_completion(self):
        for stage in ('foundation', 'followup', 'synthesis'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)

                def failure():
                    if (root / 'status.json').exists():
                        self.assertNotIn(read(root / 'status.json')['phase'],
                                         ('complete', 'complete_with_failures'))
                    raise RuntimeError('synthetic ' + stage + ' failure')

                def foundation(*args, **kwargs):
                    if stage == 'foundation':
                        failure()
                    return self.foundation(*args, **kwargs)

                def followups(source, output, *, in_place, workflow_timeline):
                    progress = FollowupTimeline(output, workflow_timeline)
                    progress.status('investigating', round=1)
                    if stage == 'followup':
                        failure()
                    progress.status('complete')
                    return source / 'report'

                with patch('language.workflow.run._run_foundation', side_effect=foundation), \
                     patch('language.workflow.followups.run', side_effect=followups), \
                     patch('language.workflow.render.finalize_report', side_effect=lambda *args, **kw: failure()):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic ' + stage + ' failure'):
                        run({}, root)
                self.assertEqual(read(root / 'status.json')['phase'], 'failed')
                saved = read(root / 'failure.json')
                self.assertEqual(saved['error'], 'synthetic ' + stage + ' failure')
                self.assertIn('RuntimeError', saved['traceback'])

    def test_one_lock_excludes_overlapping_callers_through_synthesis(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stages = []

            def reject_overlap(stage):
                before = (root / 'status.json').read_bytes()
                with self.assertRaises(BlockingIOError):
                    run({}, root)
                self.assertEqual((root / 'status.json').read_bytes(), before)
                self.assertFalse((root / 'failure.json').exists())
                stages.append(stage)

            def foundation(*args, **kwargs):
                output = self.foundation(*args, **kwargs)
                reject_overlap('foundation')
                return output

            def followups(source, output, *, in_place, workflow_timeline):
                progress = FollowupTimeline(output, workflow_timeline)
                progress.status('investigating', round=1)
                reject_overlap('followup')
                progress.status('complete')
                return source / 'report'

            def finalize(*args, **kwargs):
                reject_overlap('synthesis')
                return root / 'report'

            with patch('language.workflow.run._run_foundation', side_effect=foundation), \
                 patch('language.workflow.followups.run', side_effect=followups), \
                 patch('language.workflow.render.finalize_report', side_effect=finalize):
                run({}, root)
            self.assertEqual(stages, ['foundation', 'followup', 'synthesis'])
            self.assertEqual(read(root / 'status.json')['phase'], 'complete')


if __name__ == '__main__':
    unittest.main()
