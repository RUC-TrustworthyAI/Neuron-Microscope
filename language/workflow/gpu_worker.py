"""Serialize GPU operations on one resident model without holding up API calls."""
import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from .state import lock, read, write


def process_identity(pid):
    stat = Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()
    return {'pid': pid, 'startticks': int(stat[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def alive(identity):
    try:
        return (process_identity(identity['pid']) == identity and
                Path('/proc', str(identity['pid']), 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z')
    except (FileNotFoundError, ProcessLookupError):
        return False


class GPUClient:
    def __init__(self, config, root, timeline, request):
        from transformers import AutoTokenizer
        self.config, self.root, self.timeline, self.request = config, Path(root), timeline, request
        self.queue = Path(config['gpu_lock']).parent / 'workers' / ('gpu-' + '-'.join(map(str, config['gpus'])))
        self.queue.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True, use_fast=True)
        self.session = str(self.root.resolve())
        self.marker = self.queue / 'clients' / (uuid.uuid4().hex + '.json')
        self.marker.parent.mkdir(exist_ok=True)
        self.name = None
        self.queued_seconds = 0.

    def __enter__(self):
        write(self.marker, process_identity(os.getpid()))
        return self

    def __exit__(self, *_):
        # A live client remains registered while it waits for an LLM response.
        self.marker.unlink(missing_ok=True)

    @contextlib.contextmanager
    def span(self, name):
        previous, self.name = self.name, name
        try:
            yield
        finally:
            self.name = previous

    def start(self):
        with lock(self.queue / 'startup.lock'):
            receipt = self.queue / 'identity.json'
            if receipt.exists() and alive(read(receipt)):
                return
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.config['gpus']))
            tools = str(Path(__file__).resolve().parents[1] / 'tools')
            env['PYTHONPATH'] = tools + os.pathsep + env.get('PYTHONPATH', '')
            command = [sys.executable, '-u', '-m', f'{__package__}.gpu_worker', '--queue', str(self.queue),
                       '--gpu-lock', self.config['gpu_lock']]
            with (self.queue / 'worker.log').open('ab', buffering=0) as log:
                process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(receipt, process_identity(process.pid))

    def call(self, operation, **payload):
        self.start()
        identity = f'{time.time_ns():020d}-{uuid.uuid4().hex}'
        request_path = self.queue / (identity + '.request.json')
        result_path = self.queue / (identity + '.result.json')
        submitted = time.time()
        write(request_path, {'operation': operation, 'payload': payload, 'session': self.session,
              'request': self.request, 'settings_file': str(self.root / 'backend-settings.json'),
              'name': self.name or operation})
        while not result_path.exists():
            time.sleep(.1)
            if result_path.exists():
                break
            receipt = self.queue / 'identity.json'
            if not receipt.exists() or not alive(read(receipt)):
                raise RuntimeError(f'GPU worker stopped; inspect {self.queue / "worker.log"}')
        result = read(result_path)
        timings = result.get('timings', [])
        self.timeline.rows.extend(timings)
        write(self.root / 'timeline.json', self.timeline.rows)
        if timings:
            self.queued_seconds += max(0., timings[0]['start'] - submitted)
            write(self.root / 'gpu-queue.json', {'seconds': self.queued_seconds, 'devices': self.config['gpus']})
        if result['status'] != 'complete':
            raise RuntimeError(result['traceback'])
        return result['output']

    def batch_forward(self, inputs):
        return self.call('forward', inputs=inputs)


class RuntimePool:
    """Weights are shared; neuron caches, forward counters and RNG states are private."""
    def __init__(self):
        self.base, self.signature, self.contexts = None, None, {}

    def context(self, job):
        from common import Runtime
        from settings import request_settings
        request = job['request']
        cfg = request_settings(request)
        signature = (request['model'], json.dumps(cfg['runtime'], sort_keys=True),
                     tuple(cfg['gpu_groups'][0]), request.get('torch_seed', cfg['torch_seed']))
        loaded = None
        if self.base is None:
            started = time.time()
            self.base = Runtime.load(request)
            self.signature = signature
            self.initial_rng = self.rng()
            loaded = {'name': 'model-load', 'resource': 'gpu', 'stage': 3,
                      'start': started, 'end': time.time(), 'status': 'complete'}
            loaded['seconds'] = loaded['end'] - started
        elif signature != self.signature:
            raise ValueError('GPU worker received incompatible model or numerical settings')
        key = job['session']
        if key not in self.contexts:
            runtime = Runtime(self.base.model, self.base.tokenizer,
                              copy.deepcopy(self.base.settings), copy.deepcopy(request['neuron']))
            runtime.identity, runtime.versions = self.base.identity, self.base.versions
            self.contexts[key] = (runtime, copy.deepcopy(self.initial_rng))
        runtime, rng = self.contexts[key]
        if runtime.neuron != request['neuron']:
            raise ValueError('Neuron changed within a GPU session')
        self.restore(rng)
        return runtime, loaded

    def rng(self):
        torch = self.base.torch
        return torch.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()]

    def restore(self, state):
        torch = self.base.torch
        torch.set_rng_state(state[0])
        torch.cuda.set_rng_state_all(state[1])

    def save(self, key):
        runtime, _ = self.contexts[key]
        self.contexts[key] = runtime, self.rng()


def dispatch(runtime, job):
    payload, operation = job['payload'], job['operation']
    if operation == 'forward':
        from .measure import Measurements
        return Measurements._forward(type('Measurement', (), {'runtime': runtime})(), payload['inputs'])
    if operation == 'batch-comparison':
        from .measure import batch_comparison
        return batch_comparison(runtime, payload['inputs'])
    if operation == 'sections':
        from paper_sections import run
        return run(payload['request'], Path(payload['folder']), runtime)
    if operation == 'attribute':
        from model_tools import attribute
        return attribute(payload['request'], Path(payload['folder']), runtime)
    if operation == 'matched-ablation':
        from .interventions import matched_ablation
        return matched_ablation(runtime, payload['results'], Path(payload['folder']), payload['settings'])
    raise ValueError('Unknown GPU operation: ' + operation)


def serve(queue, gpu_lock):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    queue = Path(queue)
    pool = RuntimePool()
    with lock(queue / 'worker.lock', blocking=False), lock(gpu_lock):
        last_client = time.monotonic()
        while True:
            pending = sorted(queue.glob('*.request.json'))
            if not pending:
                living = False
                for marker in (queue / 'clients').glob('*.json'):
                    if alive(read(marker)):
                        living = True
                    else:
                        marker.unlink(missing_ok=True)
                if living:
                    last_client = time.monotonic()
                elif time.monotonic() - last_client > 120:
                    break
                time.sleep(.05)
                continue
            path = pending[0]
            running = path.with_name(path.name.replace('.request.json', '.running.json'))
            path.rename(running)
            job, timings = read(running), []
            try:
                os.environ['NEURON_MICROSCOPE_SETTINGS'] = job['settings_file']
                runtime, loaded = pool.context(job)
                if loaded:
                    timings.append(loaded)
                started = time.time()
                output = dispatch(runtime, job)
                pool.save(job['session'])
                ended = time.time()
                timings.append({'name': job['name'], 'resource': 'gpu', 'stage': 3,
                                'start': started, 'end': ended, 'seconds': ended - started, 'status': 'complete'})
                result = {'status': 'complete', 'output': output, 'timings': timings}
            except Exception:
                result = {'status': 'failed', 'traceback': traceback.format_exc(), 'timings': timings}
            write(path.with_name(path.name.replace('.request.json', '.result.json')), result)
            running.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True, type=Path)
    parser.add_argument('--gpu-lock', required=True)
    args = parser.parse_args()
    serve(args.queue, args.gpu_lock)
