"""Sample spectrum contexts in proportion to displayed log-count bar heights."""
import html
import math
import time
from pathlib import Path

import numpy as np

from common import read, write, now


def save_contexts(points, api, neuron, tokenizer, output):
    """Keep exact saved inputs available to later experiment tools."""
    documents, anchors, tokens = {}, {}, {}
    for point in points:
        document = str(point['document'])
        if document not in documents:
            saved = api.document_activations(int(document), neuron)
            ids, channels = saved['token_ids'], saved['activations']
            ids = list(map(int, ids))
            for token in set(ids) - tokens.keys():
                tokens[token] = tokenizer.decode([token], skip_special_tokens=False)
            documents[document] = {'token_ids': ids, 'tokens': [tokens[t] for t in ids],
                'gate': channels[:, 0].astype(float).tolist(),
                'value': channels[:, 1].astype(float).tolist(),
                'product': channels[:, 2].astype(float).tolist()}
        anchors[str(point['id'])] = {'document': document, 'position': point['position'],
            'values': [point['gate'], point['up'], point['activation']]}
    write(Path(output) / 'saved-contexts.json', {'points': anchors, 'documents': documents})


def register_contexts(data, folders):
    for folder in folders:
        path = Path(folder) / 'saved-contexts.json'
        if not path.is_file():
            continue
        saved = read(path)
        for document, value in saved['documents'].items():
            data['documents'].setdefault(document, value)
        data['points'].update(saved['points'])


def allocate(counts, total=500):
    """Largest-remainder allocation with corpus capacities and the chart's log10 scale."""
    weights = [math.log10(n) if n > 1 else 0.0 for n in counts]
    available = sum(n for n, weight in zip(counts, weights) if weight)
    total = min(total, available)
    if not total:
        return [0] * len(counts)
    targets = [total * weight / sum(weights) for weight in weights]
    quotas = [min(n, math.floor(target)) for n, target in zip(counts, targets)]
    remaining = total - sum(quotas)
    while remaining:
        order = sorted((i for i, n in enumerate(counts) if weights[i] and quotas[i] < n),
                       key=lambda i: (-(targets[i] - quotas[i]), i))
        for i in order[:remaining]:
            quotas[i] += 1
            remaining -= 1
    return quotas


def prepare(report, config, activations, output):
    """Read saved activations and preserve exact contexts for explanation and prediction."""
    from dataset_reader import Dataset as open_dataset
    from spectrum import build
    from tokenizers import Tokenizer

    report, output = Path(report), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = read(report / 'report.json')
    edges = build(data, [], {})['edges']
    neuron = 'L%d_N%d' % (data['identity']['layer'], data['identity']['neuron'])
    settings = config.get('settings', {})
    total = settings.get('spectrum_sampling', {}).get('total_points', 500)
    seed = settings.get('seed', 42)
    request = {'neuron': neuron, 'dataset': str(Path(activations).resolve()),
               'edges': edges, 'total_points': total, 'seed': seed,
               'allocation': 'log10-corpus-bin-count-largest-remainder'}
    if (output / 'sampling.json').is_file() and (output / 'sections.json').is_file():
        if read(output / 'sampling.json')['request'] == request:
            if not (output / 'saved-contexts.json').is_file():
                api = open_dataset(activations)
                try:
                    tokenizer = Tokenizer.from_file(str(config.get('tokenizer') or Path(config['model']) / 'tokenizer.json'))
                    save_contexts(read(output / 'sections.json')['spectrum']['points'],
                                  api, neuron, tokenizer, output)
                finally:
                    api.close()
            return output
    started, clock = now(), time.perf_counter()
    api = open_dataset(activations)
    try:
        histogram = api.histogram(neuron, 'product')
        bf16 = (np.arange(65536, dtype=np.uint32) << np.uint32(16)).view(np.float32)
        finite = np.isfinite(bf16)
        lookup = np.zeros(len(bf16), dtype=np.int64)
        lookup[finite] = np.clip(np.searchsorted(edges, bf16[finite], side='right') - 1,
                                 0, len(edges) - 2)
        counts = np.zeros(len(edges) - 1, dtype=np.uint64)
        np.add.at(counts, lookup[finite], histogram[finite])
        population = list(map(int, counts))
        quotas = allocate(population, total)
        rng = np.random.default_rng(seed)
        ranks = [np.sort(rng.choice(n, quota, replace=False))
                 for n, quota in zip(population, quotas)]
        seen = np.zeros(len(population), dtype=np.uint64)
        rows, tokens, shards = [], 0, 0
        scan_clock = time.perf_counter()
        for start, values in api.iter_neuron(neuron):
            bits = (values[:, 2].copy().view(np.uint32) >> np.uint32(16)).astype(np.uint16)
            bins = lookup[bits]
            batch_counts = np.bincount(bins, minlength=len(population)).astype(np.uint64)
            for i in range(len(population)):
                lo = np.searchsorted(ranks[i], seen[i])
                hi = np.searchsorted(ranks[i], int(seen[i]) + int(batch_counts[i]))
                if hi <= lo:
                    continue
                positions = np.flatnonzero(bins == i)
                for pos in positions[ranks[i][lo:hi] - int(seen[i])]:
                    rows.append((int(start + pos), i))
            seen += batch_counts
            tokens += len(values)
            shards += 1
            write(output / 'progress.json', {'status': 'scanning', 'tokens': tokens,
                  'total_tokens': api.manifest['tokens'], 'sampled_points': len(rows)})
        scan_seconds = time.perf_counter() - scan_clock
        tokenizer = Tokenizer.from_file(str(config.get('tokenizer') or Path(config['model']) / 'tokenizer.json'))
        initial_documents = set(map(str, data['documents']))
        points = []
        for index, band in sorted(rows):
            doc_index = int(np.searchsorted(api.starts, index, side='right')) - 1
            document = api.documents.slice(doc_index, 1).to_pylist()[0]
            doc_id = str(document['document_id'])
            position = index - int(document['token_start'])
            saved = api.document_activations(int(doc_id), neuron)
            ids, channels = saved['token_ids'], saved['activations']
            lo, hi = max(0, position - 512), min(len(ids), position + 11)
            trace = [{'token': tokenizer.decode([int(ids[j])], skip_special_tokens=False),
                      'activation': float(channels[j, 2])} for j in range(lo, hi)]
            points.append({'id': 'spectrum-' + str(index), 'document': doc_id, 'position': position,
                'dataset_token_index': index, 'source': 'saved natural-corpus token', 'source_family': 'corpus',
                'activation': float(channels[position, 2]), 'gate': float(channels[position, 0]),
                'up': float(channels[position, 1]), 'bin': band, 'group': 'unlabelled',
                'observed_prefix': tokenizer.decode(list(map(int, ids[:position + 1])), skip_special_tokens=False),
                'peak_token': html.escape(tokenizer.decode([int(ids[position])], skip_special_tokens=False)),
                'text': ''.join(row['token'] for row in trace), 'token_activations': trace,
                'peak_index': position - lo, 'cut_left': lo > 0, 'cut_right': hi < len(ids),
                'evaluation_split': 'seen' if doc_id in initial_documents else 'feedback'})
        bins = [{'i': i, 'label': f'{a:g}…{b:g}', 'n_labeled': quota,
                 'groups': {'unlabelled': quota} if quota else {}}
                for i, (a, b, quota) in enumerate(zip(edges, edges[1:], quotas))]
        spectrum = {'edges': edges, 'groups': [{'id': 'unlabelled', 'name': 'Unlabelled',
                    'pole': 'either', 'n': len(points)}], 'bins': bins, 'points': points,
                    'sample_hist': {'counts': quotas},
                    'corpus_hist': {'counts': population, 'n_tokens': sum(population),
                                    'source': 'saved full-corpus product-value counts'},
                    'sampling': 'Uniform samples within each activation bin; bin quotas are proportional to log10 corpus counts.',
                    'n_requested': len(points), 'n_assigned': 0}
        save_contexts(points, api, neuron, tokenizer, output)
        write(output / 'sections.json', {'spectrum': spectrum})
        write(output / 'sampling.json', {'request': request, 'started_at': started, 'finished_at': now(),
              'population_bin_counts': population, 'sample_bin_counts': quotas,
              'sampled_points': len(points), 'scan_seconds': scan_seconds,
              'wall_seconds': time.perf_counter() - clock, 'tokens_scanned': tokens, 'shards_scanned': shards,
              'new_model_forwards': 0, 'gpu_seconds': 0, 'llm_requests': 0})
        write(output / 'tool-request.json', {'id': 'spectrum-sample', 'output': str(output)})
        write(output / 'result.json', {'status': 'complete', 'sections': ['spectrum'],
              'sampled_points': len(points), 'new_model_forwards': 0, 'llm_requests': 0})
        write(output / 'progress.json', {'status': 'complete', 'sampled_points': len(points)})
        return output
    finally:
        api.close()
