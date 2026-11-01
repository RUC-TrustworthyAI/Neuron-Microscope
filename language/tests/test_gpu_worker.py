import copy
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from language.workflow.gpu_worker import GPUClient, RuntimePool
from language.workflow.state import Timeline, read, write


class GPUWorkerTests(unittest.TestCase):
    def test_contexts_share_weights_and_isolate_neuron_caches_and_rng(self):
        class State(int):
            def clone(self):
                return State(self)
        class CUDA:
            @staticmethod
            def get_rng_state_all():
                return []
            @staticmethod
            def set_rng_state_all(_):
                pass
        class Torch:
            cuda, value = CUDA(), State(123)
            @classmethod
            def get_rng_state(cls):
                return cls.value
            @classmethod
            def set_rng_state(cls, value):
                cls.value = value
        class Runtime:
            loads = 0
            def __init__(self, model, tokenizer, settings, neuron):
                self.model, self.tokenizer, self.settings, self.neuron = model, tokenizer, settings, neuron
                self.torch, self.forwards = Torch, 0
            @classmethod
            def load(cls, request):
                cls.loads += 1
                result = cls(object(), object(), {}, request['neuron'])
                result.identity, result.versions = {}, {}
                return result
        cfg = {'runtime': {}, 'gpu_groups': [[0, 1, 2, 3]], 'torch_seed': 0}
        a = {'session': 'a', 'request': {'model': '/model', 'neuron': {'layer': 2, 'index': 3}}}
        b = copy.deepcopy(a)
        b['session'], b['request']['neuron']['index'] = 'b', 4
        with patch('common.Runtime', Runtime), patch('settings.request_settings', return_value=cfg):
            pool = RuntimePool()
            ra, _ = pool.context(a)
            ra.cache = {'a': True}
            Torch.value = State(456)
            pool.save('a')
            rb, loaded = pool.context(b)
            self.assertIs(ra.model, rb.model)
            self.assertNotEqual(ra.neuron, rb.neuron)
            self.assertFalse(hasattr(rb, 'cache'))
            self.assertEqual(Torch.value, 123)
            self.assertIsNone(loaded)
            again, _ = pool.context(a)
            self.assertIs(again, ra)
            self.assertEqual(Torch.value, 456)
            self.assertEqual(Runtime.loads, 1)

    def test_queue_preserves_request_and_excludes_waiting_from_gpu_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queue = root / 'queue'
            queue.mkdir()
            client = GPUClient.__new__(GPUClient)
            client.queue, client.root, client.timeline = queue, root, Timeline(root)
            client.session, client.request = 'a', {'neuron': {'layer': 2, 'index': 3}}
            client.config, client.name, client.queued_seconds = {'gpus': [0, 1, 2, 3]}, None, 0.
            client.start = lambda: None
            observed = []
            def worker():
                while not list(queue.glob('*.request.json')):
                    time.sleep(.005)
                path = next(queue.glob('*.request.json'))
                job = read(path)
                observed.append(job)
                time.sleep(.03)
                end = time.time()
                write(path.with_name(path.name.replace('.request.json', '.result.json')),
                      {'status': 'complete', 'output': [[1., 2.]], 'timings': [
                          {'name': job['name'], 'resource': 'gpu', 'stage': 3,
                           'start': end - .002, 'end': end, 'seconds': .002, 'status': 'complete'}]})
            thread = threading.Thread(target=worker)
            thread.start()
            with client.span('search-round-2'):
                output = client.batch_forward([[11, 12]])
            thread.join()
            self.assertEqual(output, [[1., 2.]])
            self.assertEqual(observed[0]['payload']['inputs'], [[11, 12]])
            self.assertEqual(observed[0]['request'], client.request)
            self.assertEqual(client.timeline.rows[0]['name'], 'search-round-2')
            self.assertAlmostEqual(client.timeline.rows[0]['seconds'], .002)
            self.assertGreater(client.queued_seconds, .01)


if __name__ == '__main__':
    unittest.main()
