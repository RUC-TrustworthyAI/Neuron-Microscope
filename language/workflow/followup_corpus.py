"""Query the full saved corpus for follow-ups; sample before reading activations."""
import hashlib
import json
import random
from pathlib import Path

from language.tools.common import select_peak


def _first_match(text, terms):
    folded = text.casefold()
    matches = [(folded.find(term.casefold()), index, term) for index, term in enumerate(terms)]
    matches = [match for match in matches if match[0] >= 0]
    if not matches:
        return None
    start, _, term = min(matches)
    return {'term': term, 'folded_span': [start, start + len(term.casefold())]}


def _character_span(text, folded_span):
    # Casefold can expand one character (e.g. ß -> ss); retain original offsets.
    owners = [index for index, char in enumerate(text) for _ in char.casefold()]
    start, stop = folded_span
    return owners[start], owners[stop - 1] + 1


def _row(dataset, tokenizer, neuron, candidate, identity, label):
    document, match = candidate
    saved = dataset.document_activations(document['document_id'], neuron)
    ids = list(map(int, saved['token_ids']))
    text = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if encoded.ids != ids:
        raise ValueError(f"Cannot align decoded text to saved tokens for document {document['document_id']}")
    start, stop = _character_span(text, match['folded_span'])
    positions = [i for i, (a, b) in enumerate(encoded.offsets) if a < stop and b > start and b > a]
    if not positions:
        raise ValueError('Literal match has no aligned saved token')
    end = positions[-1] + 1
    prefix = tokenizer.decode(ids[:end], skip_special_tokens=False)
    if prefix != text[:encoded.offsets[end - 1][1]]:
        raise ValueError('Saved token prefix does not preserve the matched character offsets')
    values = saved['activations']
    if values.shape != (len(ids), 3):
        raise ValueError('Saved activation channels do not align with token IDs')
    trace = [{'position': i, 'token_id': token, 'token': tokenizer.decode([token], skip_special_tokens=False),
              'offset': list(encoded.offsets[i]),
              **{name: float(values[i, column]) for column, name in enumerate(('gate', 'value', 'product'))}}
             for i, token in enumerate(ids[:end])]
    selected = select_peak([trace[i] for i in positions], 'absmax')
    causal_ids = ids[:selected['position'] + 1]
    return {'id': identity, 'label': label, 'text': prefix, 'token_ids': ids[:end],
            'trace': trace, 'scored_positions': positions, 'selected': selected,
            'score': {'location': 'replacement_span', 'reduction': 'absmax'}, 'span': [start, stop],
            'status': 'complete', 'document_id': str(document['document_id']),
            'dataset_token_index': int(document['token_start']) + selected['position'],
            'match': {'term': match['term'], 'character_span': [start, stop], 'text': text[start:stop]},
            'causal_prefix_token_ids': causal_ids,
            'causal_prefix': tokenizer.decode(causal_ids, skip_special_tokens=False),
            'document_tokens': len(ids), 'cut_left': False, 'cut_right': end < len(ids)}


def validate_spec(spec):
    groups = spec['groups']
    if len({group['label'] for group in groups}) != len(groups):
        raise ValueError('Corpus groups must have distinct labels')
    if any(not group['label'].strip() or
           any(not term.strip() for term in group['terms']) for group in groups):
        raise ValueError('Each corpus group needs a label and nonempty literal terms')
    queries = [frozenset(term.casefold() for term in group['terms']) for group in groups]
    if len(set(queries)) != len(queries):
        raise ValueError('Comparison groups must not use identical literal queries')


def _lookup(dataset, tokenizer, neuron, spec, *, seed=42, limit=12, exclude_document_ids=()):
    """Inject a lightweight tokenizer and archive for offline contract tests."""
    groups = spec['groups']
    if dataset.manifest['channels'] != ['gate', 'value', 'product']:
        raise ValueError('Corpus lookup requires native gate, value and product channels')
    excluded = set(map(str, exclude_document_ids))
    summaries = [{**group, 'matched_documents': 0, 'excluded_documents': 0,
                  'eligible_documents': 0} for group in groups]
    samples = [[] for _ in groups]
    rngs = [random.Random(f'{seed}:{index}') for index in range(len(groups))]
    scanned_documents = scanned_tokens = excluded_documents = 0
    for batch in dataset.documents.to_batches(max_chunksize=256):
        for document in batch.to_pylist():
            start, count = int(document['token_start']), int(document['token_count'])
            ids = list(map(int, dataset.token_ids[start:start + count]))
            if len(ids) != count:
                raise ValueError('Corpus document exceeds its saved token array')
            text = tokenizer.decode(ids, skip_special_tokens=False)
            excluded_document = str(document['document_id']) in excluded
            scanned_documents += 1
            scanned_tokens += count
            excluded_documents += excluded_document
            for group, summary, sample, rng in zip(groups, summaries, samples, rngs):
                match = _first_match(text, group['terms'])
                if match is None:
                    continue
                summary['matched_documents'] += 1
                if excluded_document:
                    summary['excluded_documents'] += 1
                    continue
                summary['eligible_documents'] += 1
                index = rng.randrange(summary['eligible_documents'])
                candidate = (document, match)
                if len(sample) < limit:
                    sample.append(candidate)
                elif index < limit:
                    sample[index] = candidate
    if (scanned_documents, scanned_tokens) != (dataset.manifest['documents'], dataset.manifest['tokens']):
        raise ValueError('Corpus scan totals differ from its manifest')
    query_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    rows = []
    for index, (summary, sample) in enumerate(zip(summaries, samples), 1):
        summary.update(selected_documents=len(sample), no_match=summary['matched_documents'] == 0,
                       no_eligible_match=summary['eligible_documents'] == 0)
        for candidate in sorted(sample, key=lambda row: row[0]['document_id']):
            identity = f"corpus-{query_id}-g{index}-d{candidate[0]['document_id']}"
            rows.append(_row(dataset, tokenizer, neuron, candidate, identity, summary['label']))
    return {'status': 'complete', 'rows': rows, 'groups': summaries,
            'source': {'kind': 'saved_native_corpus', 'neuron': neuron,
                       'model': dataset.manifest.get('model'),
                       'model_snapshot': dataset.manifest.get('model_snapshot'),
                       'capture': dataset.manifest.get('capture', {})},
            'selection': {'method': 'uniform-document-reservoir-per-group', 'seed': seed,
                          'limit_per_group': limit, 'activation_used_for_document_selection': False,
                          'matching': 'literal case-insensitive OR; earliest occurrence, query-order tie break',
                          'anchor': 'absmax signed product over all tokens overlapping the first literal match',
                          'context': 'document start through match end; only causal_prefix precedes or reaches the anchor',
                          'groups_may_share_documents': True,
                          'excluded_document_ids': sorted(excluded)},
            'scanned_documents': scanned_documents, 'scanned_tokens': scanned_tokens,
            'excluded_documents': excluded_documents, 'new_model_forwards': 0}


def lookup(config, spec, *, seed=42, limit=12, exclude_document_ids=()):
    """Read the entire activation archive without loading a model or using a GPU."""
    from tokenizers import Tokenizer
    from language.tools.dataset_reader import Dataset

    tokenizer = Tokenizer.from_file(str(Path(config['model']) / 'tokenizer.json'))
    with Dataset(config['activations']) as dataset:
        result = _lookup(dataset, tokenizer, config['neuron'], spec, seed=seed, limit=limit,
                         exclude_document_ids=exclude_document_ids)
    result['source']['dataset'] = str(Path(config['activations']).resolve())
    return result
