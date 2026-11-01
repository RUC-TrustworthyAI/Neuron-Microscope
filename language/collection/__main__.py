"""Sample text, collect native activations and build neuron report inputs."""
import argparse
import fcntl
import gc
import glob
import os
from pathlib import Path
import re
import sys
import time

from .corpus import digest, prepare, read, write


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', type=Path, required=True)
    result.add_argument('--model-name', help='Display name; defaults to the model directory name')
    result.add_argument('--corpus', nargs='+', required=True, help='Local JSONL/Parquet files, globs or directories')
    result.add_argument('--text-column', default='text')
    result.add_argument('--neurons', nargs='+', required=True, help='Zero-based coordinates, for example L55_N11703')
    result.add_argument('--tokens', type=int, default=100_000_000)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--max-document-tokens', type=int, default=4096)
    result.add_argument('--shard-tokens', type=int, default=262144)
    result.add_argument('--batch-size', type=int, default=4)
    result.add_argument('--batch-tokens', type=int, default=8192)
    result.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    result.add_argument('--report-workers', type=int, default=2)
    result.add_argument('--output', type=Path, required=True)
    return result


def request_for(args):
    model = args.model.resolve()
    config = read(model / 'config.json')
    if config.get('model_type') != 'qwen3':
        raise ValueError('Activation collection supports dense Qwen3 models')
    neurons = []
    for identity in args.neurons:
        match = re.fullmatch(r'L(\d+)_N(\d+)', identity)
        if not match:
            raise ValueError(f'Expected L<layer>_N<neuron>: {identity}')
        layer, neuron = map(int, match.groups())
        if layer >= config['num_hidden_layers'] or neuron >= config['intermediate_size']:
            raise ValueError(f'Neuron coordinate is outside the model: {identity}')
        neurons.append({'id': f'L{layer}_N{neuron}', 'layer': layer, 'neuron': neuron})
    if len({unit['id'] for unit in neurons}) != len(neurons):
        raise ValueError('Neuron coordinates must be unique')
    paths = set()
    for pattern in args.corpus:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            raise FileNotFoundError(pattern)
        for match in matches:
            path = Path(match)
            if path.is_dir():
                paths.update(p.resolve() for p in path.rglob('*')
                             if p.suffix in ('.jsonl', '.ndjson', '.parquet')
                             and not p.resolve().is_relative_to(args.output.resolve()))
            else:
                paths.add(path.resolve())
    if not paths:
        raise ValueError('No JSONL or Parquet files found')
    if min(args.tokens, args.max_document_tokens, args.batch_size, args.batch_tokens, args.report_workers) < 1:
        raise ValueError('Token budgets, batch sizes and worker counts must be positive')
    if args.shard_tokens < args.max_document_tokens or args.batch_tokens < args.max_document_tokens:
        raise ValueError('Shard and batch token limits must fit a document prefix')
    if not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        raise ValueError('GPU indices must be distinct nonnegative integers')
    signatures = [{'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in sorted(paths)]
    request = {'model': str(model), 'model_name': args.model_name or model.name,
               'model_config_sha256': digest(model / 'config.json'),
               'tokenizer_sha256': digest(model / 'tokenizer.json'),
               'weights': [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                           for p in sorted(model.glob('*.safetensors'))],
               'corpus': [str(p) for p in sorted(paths)], 'sources': signatures,
               'text_column': args.text_column, 'neurons': neurons,
               **{name: getattr(args, name) for name in ('tokens', 'seed', 'max_document_tokens', 'shard_tokens',
                                                        'batch_size', 'batch_tokens')}}
    return request, neurons


def run(args, runtime=None):
    root = args.output.resolve()
    request, neurons = request_for(args)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.collect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'request.json').exists():
            if read(root / 'request.json') != request:
                raise ValueError('Collection inputs differ from this output directory')
        elif any(path.name != '.collect.lock' for path in root.iterdir()):
            raise FileExistsError('Collection output directory already contains files')
        else:
            write(root / 'request.json', request)
        write(root / 'status.json', {'stage': 'inputs', 'tokens': request['tokens']})
        manifest = prepare(root, request, neurons)
        from .capture import collect, index, logit_effects
        state = read(root / 'capture-state.json') if (root / 'capture-state.json').exists() else {'completed_shards': []}
        needed = (len(state['completed_shards']) != len(manifest['parts'][0]['shards']) or
                  any(not (root / 'effects' / (unit['id'] + '.json')).exists() for unit in neurons))
        if needed:
            if runtime is None:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
                from common import Runtime
                from settings import load
                settings = load()
                settings['gpu_groups'] = [args.gpus]
                began = time.perf_counter()
                runtime = Runtime.load({'model': str(args.model.resolve()),
                                        'neuron': {'layer': neurons[0]['layer'], 'index': neurons[0]['neuron']},
                                        'settings': settings})
                write(root / 'model.json', {'identity': runtime.identity, 'versions': runtime.versions,
                                           'load_seconds': time.perf_counter() - began})
            collect(root, runtime.model)
            logit_effects(root, runtime.model, runtime.tokenizer)
            del runtime
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write(root / 'status.json', {'stage': 'index'})
        index(root)
        from .reports import build
        build(root, args.report_workers)
        write(root / 'status.json', {'stage': 'complete', 'tokens': manifest['tokens'],
                                    'neurons': len(neurons), 'reports': str(root / 'reports')})
    print(f'Activations: {root}\nReports: {root / "reports"}')
    return root


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
