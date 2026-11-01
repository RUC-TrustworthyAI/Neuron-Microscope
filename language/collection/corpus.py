"""Sample local text records and write aligned model inputs."""
import bisect
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TextCorpus:
    """Random access to JSONL rows and a memory-mapped Parquet text cache."""
    def __init__(self, paths, column, cache):
        self.column = column
        self.sources, self.ends, self.resources = [], [], []
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        total = 0
        for number, path in enumerate(paths):
            path = Path(path)
            prefix = cache / str(number)
            marker = prefix.with_suffix('.json')
            signature = {'path': str(path.resolve()), 'bytes': path.stat().st_size,
                         'mtime_ns': path.stat().st_mtime_ns, 'column': column}
            if marker.exists() and read(marker)['source'] != signature:
                raise ValueError(f'Corpus file changed: {path}')
            if path.suffix in ('.jsonl', '.ndjson'):
                offsets_path = prefix.with_suffix('.offsets.npy')
                if not marker.exists():
                    from array import array
                    offsets = array('Q')
                    with path.open('rb') as stream:
                        while True:
                            offset = stream.tell()
                            line = stream.readline()
                            if not line:
                                break
                            if line.strip():
                                offsets.append(offset)
                    np.save(offsets_path, np.asarray(offsets, dtype=np.uint64))
                    write(marker, {'source': signature, 'rows': len(offsets)})
                offsets = np.load(offsets_path, mmap_mode='r', allow_pickle=False)
                stream = path.open('rb')
                source = {'path': path, 'rows': len(offsets), 'offsets': offsets, 'stream': stream}
                self.resources.extend([offsets, stream])
            elif path.suffix == '.parquet':
                arrow_path = prefix.with_suffix('.arrow')
                if not marker.exists():
                    parquet = pq.ParquetFile(path)
                    schema = pa.schema([parquet.schema_arrow.field(column)])
                    starts, count = [], 0
                    temporary = arrow_path.with_suffix('.arrow.tmp')
                    with pa.OSFile(str(temporary), 'wb') as sink, ipc.new_file(sink, schema) as writer:
                        for batch in parquet.iter_batches(batch_size=8192, columns=[column]):
                            starts.append(count)
                            writer.write_batch(batch)
                            count += batch.num_rows
                    temporary.replace(arrow_path)
                    write(marker, {'source': signature, 'rows': count, 'batch_starts': starts})
                metadata = read(marker)
                mapped = pa.memory_map(str(arrow_path), 'r')
                source = {'path': path, 'rows': metadata['rows'], 'starts': metadata['batch_starts'],
                          'reader': ipc.open_file(mapped)}
                self.resources.append(mapped)
            else:
                raise ValueError(f'Expected a JSONL or Parquet file: {path}')
            self.sources.append(source)
            total += source['rows']
            self.ends.append(total)
        self.total = total

    def document(self, index):
        file_index = bisect.bisect_right(self.ends, index)
        source = self.sources[file_index]
        row = index - (self.ends[file_index - 1] if file_index else 0)
        if 'stream' in source:
            source['stream'].seek(int(source['offsets'][row]))
            value = json.loads(source['stream'].readline()).get(self.column)
        else:
            batch = bisect.bisect_right(source['starts'], row) - 1
            value = source['reader'].get_batch(batch).column(0)[row - source['starts'][batch]].as_py()
        if value is not None and not isinstance(value, str):
            raise ValueError(f'Text column must contain strings: {source["path"].name}, row {row}')
        return value or '', {'source_file': source['path'].name, 'input_row': row}

    def close(self):
        for resource in self.resources:
            if isinstance(resource, np.memmap):
                resource._mmap.close()
            else:
                resource.close()


def prepare(root, request, neurons):
    root = Path(root)
    if (root / 'inputs-complete.json').exists():
        return read(root / 'dataset.json')
    tokenizer = Tokenizer.from_file(str(Path(request['model']) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    corpus = TextCorpus(request['corpus'], request['text_column'], root / 'source-index')
    budget, maximum = request['tokens'], request['max_document_tokens']
    ids = np.lib.format.open_memmap(root / 'token_ids.npy', mode='w+', dtype=np.uint32, shape=(budget,))
    inputs = root / 'inputs'
    inputs.mkdir(exist_ok=True)
    documents, shards, block_tokens, block_documents = [], [], [], []
    consumed = block_start = 0

    def flush():
        nonlocal block_start
        block = len(shards)
        prefix = inputs / f'block-{block:06d}'
        np.save(str(prefix) + '.ids.u32.npy', np.asarray(block_tokens, np.uint32))
        lengths = [doc['tokens'] for doc in block_documents]
        offsets = np.r_[np.uint64(0), np.cumsum(lengths, dtype=np.uint64)]
        np.save(str(prefix) + '.offsets.u64.npy', offsets)
        write(str(prefix) + '.documents.json', block_documents)
        shards.append({'id': block, 'path': f'shard-{block:06d}.npy',
                       'token_start': block_start, 'token_count': len(block_tokens)})
        for row, offset in zip(block_documents, offsets):
            documents.append({'document_id': row['document_order'], 'token_start': block_start + int(offset),
                              'token_count': row['tokens'], 'shard_id': block, 'shard_token_start': int(offset)})
        block_start += len(block_tokens)
        block_tokens.clear()
        block_documents.clear()
        write(root / 'status.json', {'stage': 'inputs', 'tokens': block_start, 'total_tokens': budget})

    try:
        for index in np.random.default_rng(request['seed']).permutation(corpus.total):
            text, source = corpus.document(int(index))
            tokens = tokenizer.encode(text, add_special_tokens=False).ids[:min(maximum, budget - consumed)]
            if not tokens:
                continue
            if block_tokens and len(block_tokens) + len(tokens) > request['shard_tokens']:
                flush()
            order = len(documents) + len(block_documents)
            block_documents.append({**source, 'document_order': order, 'tokens': len(tokens),
                                    'source_token_start': 0})
            block_tokens.extend(tokens)
            ids[consumed:consumed + len(tokens)] = tokens
            consumed += len(tokens)
            if consumed == budget:
                break
        if consumed != budget:
            raise ValueError(f'Corpus supplied {consumed:,} tokens after prefix limits; requested {budget:,}')
        if block_tokens:
            flush()
        ids.flush()
    finally:
        ids._mmap.close()
        corpus.close()
    pq.write_table(pa.Table.from_pylist(documents), root / 'documents.parquet')
    units = [{**unit, 'part': 'part-0', 'part_column': i} for i, unit in enumerate(neurons)]
    write(root / 'neurons.json', units)
    config = read(Path(request['model']) / 'config.json')
    manifest = {'schema_version': 1, 'complete': False, 'tokens': budget, 'documents': len(documents),
                'neuron_count': len(units), 'neurons': 'neurons.json', 'channels': ['gate', 'value', 'product'],
                'model': request['model'], 'model_name': request['model_name'],
                'model_snapshot': digest(Path(request['model']) / 'config.json'),
                'capture': {'max_document_tokens': maximum, 'dtype': 'bfloat16',
                            'attention': 'sdpa', 'batch_size': request['batch_size'],
                            'batch_tokens': request['batch_tokens'], 'add_special_tokens': False,
                            'hidden_size': config['hidden_size']},
                'selection': {'original_dataset_seed': request['seed'], 'method': 'uniform-document-permutation',
                              'source_documents': corpus.total, 'text_column': request['text_column'],
                              'context': 'document-prefix', 'last_document': 'trim-to-token-budget'},
                'corpus': {'documents': 'documents.parquet', 'token_ids': 'token_ids.npy'},
                'parts': [{'id': 'part-0', 'path': 'part-0', 'shards': shards,
                           'indices': {'histogram': 'histogram.npy'}}],
                'indices': {'k': min(30, budget), 'extremes': 'indices/extremes.npz',
                            'statistics': 'indices/statistics.parquet'}}
    write(root / 'dataset.json', manifest)
    write(root / 'inputs-complete.json', {'tokens': consumed, 'documents': len(documents), 'shards': len(shards)})
    shutil.rmtree(root / 'source-index')
    return manifest
