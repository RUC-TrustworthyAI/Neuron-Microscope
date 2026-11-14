"""Repository-root entry points and shared GPU worker startup."""
import io
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from language.examples.configure import configuration
from language.workflow.gpu_worker import GPUClient
from language.workflow.run import main
from language.workflow.state import read, write


ROOT = Path(__file__).resolve().parents[2]


class EntrypointTests(unittest.TestCase):
    def test_repository_root_commands_load_without_model_or_credentials(self):
        for module in ('language', 'language.collection', 'language.workflow.run',
                       'language.workflow.gpu_worker', 'language.workflow.followups'):
            result = subprocess.run([sys.executable, '-m', module, '--help'],
                                    cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertIn('usage:', result.stdout)

    def test_configuration_preserves_measurement_and_interpretation_settings(self):
        config = configuration('L1_N2', 'models/subject', 'data/report', 'data/activations', [4, 5, 6, 7])
        self.assertEqual(config['max_rules_per_pole'], 5)
        self.assertEqual(config['max_refinements'], 2)
        self.assertEqual(config['settings']['spectrum_sampling']['total_points'], 500)
        self.assertEqual(config['settings']['gpu_groups'], [[4, 5, 6, 7]])
        self.assertEqual(config['llm']['max_concurrent_requests'], None)
        self.assertEqual((config['search_rounds'], config['search_regions'], config['candidates_per_region']),
                         (3, 6, 16))
        self.assertEqual((config['batch_size'], config['batch_tokens']), (16, 8192))
        self.assertTrue(config['gpu_lock'].endswith('gpu-4-5-6-7.lock'))
        self.assertEqual(config['settings']['runtime']['dtype'], 'bfloat16')
        self.assertEqual(config['settings']['runtime']['attention'], 'sdpa')

    def test_followup_is_opt_in_and_publishes_the_final_report(self):
        for enabled in (False, True):
            with self.subTest(followup=enabled), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                foundation, followup = root / 'example', root / 'example-followup'
                destination = root / 'published' / 'L1_N2'
                config = {'gpus': [0], 'settings': {}, 'neuron': 'L1_N2',
                          'publish': str(destination.parent), 'references': str(root / 'references')}
                config_path = root / 'config.json'
                write(config_path, config)
                if enabled:
                    destination.mkdir(parents=True)
                    (destination / 'index.html').write_text('Previously exported foundation report')
                events = []

                def complete_foundation(*args, **kwargs):
                    events.append('foundation')
                    write(foundation / 'status.json', {'phase': 'complete'})
                    return foundation / 'report'

                def complete_followup(source, output):
                    self.assertEqual(read(source / 'status.json')['phase'], 'complete')
                    events.append('followup')
                    write(output / 'status.json', {'phase': 'complete'})
                    return output / 'report'

                argv = ['language', '--config', str(config_path), '--output', str(foundation)]
                if enabled:
                    argv += ['--followup', '--checkpoint-attribution']
                with patch.object(sys, 'argv', argv), patch.object(sys, 'path', sys.path.copy()), \
                     patch.dict(os.environ), redirect_stdout(io.StringIO()), \
                     patch('language.workflow.run.run', side_effect=complete_foundation) as base_run, \
                     patch('language.workflow.followups.run', side_effect=complete_followup) as followup_run, \
                     patch('comparison.publish', side_effect=lambda *a, **kw: events.append('publish')) as publish, \
                     patch('language.workflow.render.directory') as directory:
                    self.assertEqual(main(), 0)
                base_run.assert_called_once_with(config, foundation, stop_after=None,
                                                 checkpoint_attribution=enabled)
                if enabled:
                    followup_run.assert_called_once_with(foundation.resolve(), followup.resolve())
                else:
                    followup_run.assert_not_called()
                report = (followup if enabled else foundation) / 'report'
                publish.assert_called_once_with(report, destination, 'L1_N2', config['references'], replace=enabled)
                directory.assert_called_once()
                self.assertEqual(events, ['foundation', 'followup', 'publish'] if enabled else ['foundation', 'publish'])
                self.assertEqual(read(config_path), config)
                self.assertEqual(read(foundation / 'status.json')['phase'], 'complete')

    def test_followup_failure_preserves_foundation_and_exits_unsuccessfully(self):
        for fatal in (False, True):
            with self.subTest(fatal=fatal), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                foundation, followup = root / 'example', root / 'example-followup'
                config_path = root / 'config.json'
                write(config_path, {'gpus': [0], 'settings': {}, 'neuron': 'L1_N2',
                                    'publish': str(root / 'published'), 'references': str(root / 'references')})

                def complete_foundation(*args, **kwargs):
                    write(foundation / 'status.json', {'phase': 'complete'})
                    return foundation / 'report'

                def fail_followup(source, output):
                    write(output / 'status.json', {'phase': 'failed' if fatal else 'complete_with_failures'})
                    if fatal:
                        raise RuntimeError('synthetic follow-up failure')
                    return output / 'report'

                argv = ['language', '--config', str(config_path), '--output', str(foundation), '--followup']
                with patch.object(sys, 'argv', argv), patch.object(sys, 'path', sys.path.copy()), \
                     patch.dict(os.environ), redirect_stdout(io.StringIO()), \
                     patch('language.workflow.run.run', side_effect=complete_foundation), \
                     patch('language.workflow.followups.run', side_effect=fail_followup), \
                     patch('comparison.publish') as publish:
                    if fatal:
                        with self.assertRaisesRegex(RuntimeError, 'synthetic follow-up failure'):
                            runpy.run_module('language', run_name='__main__')
                    else:
                        with self.assertRaises(SystemExit) as stopped:
                            runpy.run_module('language', run_name='__main__')
                        self.assertEqual(stopped.exception.code, 1)
                publish.assert_not_called()
                self.assertEqual(read(foundation / 'status.json')['phase'], 'complete')
                self.assertFalse((foundation / 'failure.json').exists())
                self.assertEqual(read(followup / 'status.json')['phase'],
                                 'failed' if fatal else 'complete_with_failures')

    def test_followup_rejects_partial_and_replay_modes_before_writing(self):
        modes = (['--config', 'absent.json', '--stop-after', 'interpretation'],
                 ['--ablate-from', 'absent'], ['--anatomy-from', 'absent'])
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / 'must-not-exist'
                argv = ['language', *mode, '--output', str(output), '--followup']
                with patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()), \
                     patch('language.workflow.run.run') as base_run, \
                     patch('language.workflow.followups.run') as followup_run:
                    with self.assertRaises(SystemExit) as stopped:
                        main()
                self.assertEqual(stopped.exception.code, 2)
                base_run.assert_not_called()
                followup_run.assert_not_called()
                self.assertFalse(output.exists())

    def test_followup_publication_cannot_overlap_research_directories(self):
        for location in ('foundation', 'followup', 'ancestor'):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                foundation, followup = root / 'L1_N2' / 'example', root / 'L1_N2' / 'example-followup'
                for research in (foundation, followup):
                    write(research / 'status.json', {'phase': 'complete'})
                    write(research / 'backend-settings.json', {'preserve': str(research)})
                publications = {'foundation': foundation / 'report',
                                'followup': followup / 'report', 'ancestor': root}
                config_path = root / 'config.json'
                write(config_path, {'gpus': [0], 'settings': {}, 'neuron': 'L1_N2',
                                    'publish': str(publications[location]), 'references': str(root / 'references')})
                before = {path.relative_to(root): path.read_bytes() for path in root.rglob('*') if path.is_file()}
                argv = ['language', '--config', str(config_path), '--output', str(foundation), '--followup']
                with patch.object(sys, 'argv', argv), patch.object(sys, 'path', sys.path.copy()), \
                     patch.dict(os.environ), redirect_stderr(io.StringIO()), \
                     patch('language.workflow.run.run') as base_run, \
                     patch('language.workflow.followups.run') as followup_run, \
                     patch('comparison.publish') as publish:
                    with self.assertRaises(SystemExit) as stopped:
                        main()
                self.assertEqual(stopped.exception.code, 2)
                base_run.assert_not_called()
                followup_run.assert_not_called()
                publish.assert_not_called()
                after = {path.relative_to(root): path.read_bytes() for path in root.rglob('*') if path.is_file()}
                self.assertEqual(after, before)

    def test_shared_worker_starts_the_relocated_module(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = GPUClient.__new__(GPUClient)
            worker.queue = Path(folder)
            worker.config = {'gpus': [0, 1, 2, 3], 'gpu_lock': str(Path(folder) / 'gpu.lock')}
            with patch('language.workflow.gpu_worker.subprocess.Popen', return_value=SimpleNamespace(pid=7)) as launch, \
                 patch('language.workflow.gpu_worker.process_identity', return_value={'pid': 7}):
                worker.start()
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('-m') + 1], 'language.workflow.gpu_worker')
            self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
            self.assertEqual(read(worker.queue / 'identity.json'), {'pid': 7})


if __name__ == '__main__':
    unittest.main()
