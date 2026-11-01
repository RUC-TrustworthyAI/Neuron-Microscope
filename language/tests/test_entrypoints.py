"""Repository-root entry points and shared GPU worker startup."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from language.examples.configure import configuration
from language.workflow.gpu_worker import GPUClient
from language.workflow.state import read


ROOT = Path(__file__).resolve().parents[2]


class EntrypointTests(unittest.TestCase):
    def test_repository_root_commands_load_without_model_or_credentials(self):
        for module in ('language', 'language.collection', 'language.workflow.run', 'language.workflow.gpu_worker'):
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
