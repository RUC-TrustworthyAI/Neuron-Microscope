"""Expose project-owned measurements with their original token and graph scopes."""
import copy
import hashlib
import json
from collections import defaultdict
from contextlib import closing
from itertools import accumulate
from pathlib import Path

from .criticism_operations import _distribution, _finite


MEASUREMENTS = ('traces', 'graphs', 'distributions')
ARCHIVE_DOMAINS = (*MEASUREMENTS, 'cohorts', 'cohort_descriptions')
CHANNELS = ('gate', 'value', 'product')


def _cohort(archive, identity, cases, description):
    cases = list(dict.fromkeys(cases))
    if cases:
        archive.setdefault('cohorts', {})[identity] = cases
        archive.setdefault('cohort_descriptions', {})[identity] = description


def _decode_tokens(tokenizer, ids):
    try:
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        # The low-level tokenizers.Tokenizer decoder has no cleanup option.
        return tokenizer.decode(ids, skip_special_tokens=False)


def _roundtrip_ids(tokenizer, text):
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except (AttributeError, TypeError, ValueError):
        return None
    return list(encoded.ids if hasattr(encoded, 'ids') else encoded)


def _isolation(point, tokens, token_ids, tokenizer):
    """End at the saved anchor so the isolated text has no future tokens."""
    peak = point.get('peak_index')
    if type(peak) is not int or not 0 <= peak < min(len(tokens), len(point.get('token_activations', []))):
        return {'isolation_available': False, 'isolation_unavailable_reason':
                'The saved visible fragment has no valid anchor-aligned token span.'}
    stop = len(tokens)
    start = stop - peak - 1
    ids = token_ids[start:]
    if tokenizer is not None:
        text = _decode_tokens(tokenizer, ids)
        method = 'Whole native token-ID span decoded without skipping special tokens or cleanup.'
        if '\ufffd' in text and _roundtrip_ids(tokenizer, text) != ids:
            text = None
            method = 'Unavailable: the isolated native span decodes with replacement characters and does not round-trip to its saved token IDs.'
    elif any('\ufffd' in token for token in tokens):
        text = None
        method = 'Unavailable: replacement characters in saved token pieces require the original tokenizer.'
    else:
        text = ''.join(tokens[start:])
        method = 'Saved decoded token pieces without replacement characters; no tokenizer supplied.'
    data = {'isolation_available': bool(text and text.strip()), 'isolation_token_ids': ids,
            'isolation_token_span': [start, stop], 'isolation_decode_method': method}
    if data['isolation_available']:
        data['isolation_text'] = text
    else:
        data['isolation_unavailable_reason'] = method if text is None else 'The isolated token span decodes to empty text.'
    return data


def _natural_trace(point, contexts, label, tokenizer=None):
    """Keep the complete causal prefix through this original corpus anchor."""
    document = contexts.get('documents', {}).get(str(point['document'])) or {}
    position = point['position']
    if (type(position) is not int or position < 0 or
            any(len(document.get(key, ())) <= position for key in ('tokens', 'token_ids', *CHANNELS))):
        return None
    tokens = document['tokens'][:position + 1]
    token_ids = list(map(int, document['token_ids'][:position + 1]))
    joined = ''.join(tokens)
    text = (_decode_tokens(tokenizer, token_ids) if tokenizer is not None
            else point.get('observed_prefix', joined))
    # Token pieces can disagree with whole-prefix decoding (e.g. split UTF-8 bytes).
    ends = list(accumulate(map(len, tokens), initial=0))
    trace = [{'position': i, 'token_id': token_id, 'token': token,
              'offset': ends[i:i + 2] if joined == text else None,
              **{key: document[key][i] for key in CHANNELS}}
             for i, (token, token_id) in enumerate(zip(tokens, token_ids))]
    if any(not _finite(row[key]) for row in trace for key in CHANNELS):
        return None
    selected = trace[-1]
    for public, native in (('activation', 'product'), ('gate', 'gate'), ('up', 'value')):
        if point.get(public, selected[native]) != selected[native]:
            raise ValueError(f"Saved spectrum anchor {point['id']} disagrees with its native {native} trace")
    return {'id': point['id'], 'text': text, 'token_ids': token_ids,
            'trace': trace, 'trace_complete': True, 'selected': selected,
            'scored_positions': [position], 'score': {'location': 'token_position', 'reduction': 'single',
                                                     'position': position},
            **_isolation(point, tokens, token_ids, tokenizer), 'group': label.get('group', 'unlabelled'),
            'prediction_version': label.get('version'), 'source': 'saved natural-corpus prefix',
            'document': str(point['document']), 'evaluation_split': point.get('evaluation_split'),
            'isolation_scope': 'Saved visible fragment through the original anchor; later tokens excluded.'}


def _saved_graph(row):
    """Normalize native RelP feature IDs without inventing missing activations."""
    graph = row.get('graph', row)
    if not isinstance(graph.get('nodes'), list):
        return None
    nodes = []
    for source in graph['nodes']:
        layer, neuron = source.get('layer'), source.get('neuron', source.get('feature'))
        if type(layer) is not int or type(neuron) is not int:
            continue
        node = {key: source[key] for key in ('node_id', 'ctx_idx', 'is_target_neuron') if key in source}
        node.update(layer=layer, neuron=neuron)
        node.update({key: source[key] for key in ('influence', 'activation', 'rank') if _finite(source.get(key))})
        nodes.append(node)
    # Rank only the retained MLP source nodes, excluding the target.
    ranked = sorted((node for node in nodes if not node.get('is_target_neuron') and
                     'influence' in node), key=lambda node: -abs(node['influence']))
    for rank, node in enumerate(ranked, 1):
        node.setdefault('rank', rank)
    meta = graph.get('meta', {})
    return {'id': row.get('id', graph.get('id')), 'text': graph.get('text', row.get('text', '')),
            'nodes': nodes, 'method': meta.get('method', row.get('method', 'saved attribution graph')),
            'rank_scope': 'Absolute influence order of saved non-target MLP source nodes.',
            'presence_scope': 'Presence in the retained saved graph; omitted nodes are not zero activations.',
            'selected_node_count': len(nodes), 'native_meta': meta}


def _corpus_distribution(activations, neuron_id):
    import numpy as np
    from dataset_reader import floats, open_dataset

    with closing(open_dataset(activations)) as api:
        counts = np.array(api.histogram(neuron_id, 'product'))
    if counts.shape != (65536,) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError('Exact corpus quantiles require the native 65,536-value BF16 histogram')
    values = floats(np.arange(65536, dtype=np.uint32))
    finite = np.isfinite(values)
    selected = finite & (counts > 0)
    order = np.argsort(values[selected])
    observed, weights = values[selected][order].tolist(), counts[selected][order].tolist()
    if not observed:
        raise ValueError('The corpus histogram contains no finite measured products')
    return {'values': observed, 'counts': weights,
            'scope': f'All finite native BF16 product values in the saved corpus for {neuron_id}.',
            'source': 'Exact native BF16 histogram; no report-bin interpolation.',
            'finite_token_count': sum(weights), 'excluded_nonfinite_count': sum(counts[~finite].tolist())}


def add_revision_cohorts(archive, revision):
    """Use existing saved analysis membership only where graph IDs truly exist."""
    if archive is None:
        return
    graph_ids = archive.get('graphs', {})
    for group in revision.get('clusters', []):
        _cohort(archive, 'cluster-' + group['id'],
                (case_id for case_id in group['graph_ids'] if case_id in graph_ids),
                'Saved report-analysis cluster: ' + group.get('name', group['id']))


def build_archive(measured, sample, contexts, predictions, graph_rows, *, activations=None, neuron_id=None,
                  tokenizer=None):
    """Build an executable archive from raw local evidence, before followup planning."""
    archive = {key: {} for key in (*ARCHIVE_DOMAINS, 'availability')}
    traces, graphs = archive['traces'], archive['graphs']
    traces.update({row['id']: row for row in measured
                   if row.get('status') == 'complete' and row.get('trace') and row.get('id')})
    _cohort(archive, 'measured-development', traces,
            'Measured Search, Control, Blind and atlas inputs; reused evidence, not an independent test.')
    labels = {row['id']: {**row, 'version': predictions['version']}
              for row in predictions.get('records', [])}
    groups, missing, natural_ids = defaultdict(list), [], []
    for point in sample.get('points', []):
        row = _natural_trace(point, contexts, labels.get(point['id'], {}), tokenizer)
        if row is None:
            missing.append(point['id'])
            continue
        traces[row['id']] = row
        natural_ids.append(row['id'])
        groups[row['group']].append(row['id'])
    _cohort(archive, 'natural-spectrum', natural_ids,
            'Saved spectrum anchors with complete original causal prefixes; spectrum sampling is stratified.')
    for name, ids in groups.items():
        _cohort(archive, 'natural-' + name, ids,
                'Frozen-explanation prediction group: ' + name + '; not relabelled after followup results.')
    _cohort(archive, 'no-visible-cue', groups.get('no-visible-cue', ()),
            'Saved natural-spectrum cases assigned no-visible-cue by the frozen explanation.')
    graph_groups = defaultdict(list)
    for row in graph_rows:
        graph = _saved_graph(row)
        if graph is None or not graph['id']:
            continue
        graphs[graph['id']] = graph
        if row.get('group'):
            graph_groups[row['group']].append(graph['id'])
        targets = [node for node in graph['nodes'] if node.get('is_target_neuron') and 'activation' in node]
        if len(targets) == 1:
            value = targets[0]['activation']
            graph_groups['positive' if value > 0 else 'negative' if value < 0 else 'zero'].append(graph['id'])
    _cohort(archive, 'all-graphs', graphs, 'All actual saved selected-node attribution graphs.')
    for name, ids in graph_groups.items():
        _cohort(archive, 'graphs-' + name, ids, 'Saved graph membership: ' + name)
    distribution = {'available': False, 'reason': 'No saved native activation dataset supplied.'}
    if activations is not None and neuron_id is not None:
        try:
            archive['distributions']['corpus-product'] = _corpus_distribution(activations, neuron_id)
        except (ImportError, OSError, KeyError, ValueError) as error:
            distribution['reason'] = str(error)
        else:
            distribution = {'available': True, 'id': 'corpus-product'}
    archive['availability'].update(
        natural_trace_count=len(natural_ids), missing_natural_trace_ids=missing,
        unavailable_isolation_ids=[case_id for case_id in natural_ids if not traces[case_id]['isolation_available']],
        graph_count=len(graphs), corpus_distribution=distribution,
        graph_scope='Only actual saved selected-node graphs; atlas weight summaries do not supply node activations.')
    return archive


def _nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def _natural(value):
    return type(value) is int and value >= 0


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_trace(case_id, row):
    _require(isinstance(row, dict) and _nonblank(row.get('text')) and
             isinstance(row.get('trace'), list) and row['trace'],
             f'Imported trace {case_id} needs text and saved token records')
    _require(row.get('id', case_id) == case_id and row.get('status', 'complete') == 'complete',
             f'Imported trace {case_id} has a mismatched ID or incomplete status')
    products, previous = {}, -1
    for token in row['trace']:
        _require(isinstance(token, dict) and _natural(token.get('position')) and
                 isinstance(token.get('token'), str) and _finite(token.get('product')) and
                 all(_finite(token[key]) for key in ('gate', 'value') if key in token) and
                 _natural(token.get('token_id', 0)), f'Imported trace {case_id} has an invalid token record')
        position = token['position']
        _require(position > previous, f'Imported trace {case_id} has repeated or unordered positions')
        products[position] = token['product']
        previous = position
        offset = token.get('offset')
        _require(offset is None or (isinstance(offset, list) and len(offset) == 2 and
                 all(_natural(x) for x in offset) and offset[0] <= offset[1] <= len(row['text'])),
                 f'Imported trace {case_id} has invalid character offsets')
    ids = row.get('token_ids')
    _require(ids is None or (isinstance(ids, list) and len(ids) == len(products) and
             all(_natural(value) and token.get('token_id', value) == value
                 for token, value in zip(row['trace'], ids))),
             f'Imported trace {case_id} has mismatched native token IDs')
    selected = row.get('selected')
    _require(isinstance(selected, dict) and type(selected.get('position')) is int and _finite(selected.get('product')),
             f'Imported trace {case_id} needs its original selected position and product')
    _require(products.get(selected['position']) == selected['product'],
             f'Imported trace {case_id} selected product disagrees with its native trace')
    _require('isolation_text' not in row or _nonblank(row['isolation_text']),
             f'Imported trace {case_id} has an empty isolation input')


def _validate_graph(case_id, row):
    _require(isinstance(row, dict) and isinstance(row.get('nodes'), list),
             f'Imported graph {case_id} needs canonical saved nodes')
    _require(row.get('id', case_id) == case_id and isinstance(row.get('text', ''), str),
             f'Imported graph {case_id} has a mismatched ID or invalid text')
    for node in row['nodes']:
        _require(isinstance(node, dict) and all(_natural(node.get(key)) for key in ('layer', 'neuron')) and
                 all(_finite(node[key]) for key in ('influence', 'activation') if key in node) and
                 _natural(node.get('rank', 1)) and node.get('rank', 1) >= 1,
                 f'Imported graph {case_id} has invalid canonical node fields')


def _same_identity(left, right):
    # JSON comparison preserves distinctions like true vs 1 that Python == does not.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def load_archive(path, identity, base_archive):
    """Merge a project-owned archive and record the identity and digest of its source."""
    path = Path(path)
    _require(path.suffix.lower() == '.json', 'followup_archive must be a canonical JSON evidence archive')
    raw = path.read_bytes()
    try:
        envelope = json.loads(raw.decode('utf-8-sig'))
    except (UnicodeError, ValueError) as error:
        raise ValueError('followup_archive is not a canonical JSON evidence envelope') from error
    _require(isinstance(envelope, dict) and set(envelope) - {'schema_version'} == {'identity', 'archive'},
             'followup_archive needs identity and archive fields')
    version = envelope.get('schema_version', 1)
    _require(type(version) is int and version == 1, 'Unsupported followup_archive envelope version')
    _require(isinstance(envelope['identity'], dict) and
             set(envelope['identity']) == {'model', 'neuron', 'settings'} and
             _same_identity(envelope['identity'], identity),
             'followup_archive model, neuron, and settings must exactly match this run')
    imported = envelope['archive']
    _require(isinstance(imported, dict) and not set(imported) - {*ARCHIVE_DOMAINS, 'availability'},
             'followup_archive has unknown archive fields')
    _require(any(imported.get(name) for name in MEASUREMENTS), 'followup_archive contains no saved measurement evidence')
    availability = imported.get('availability', {})
    _require(isinstance(availability, dict), 'followup_archive availability must be an object')
    result = copy.deepcopy(base_archive)
    for domain in ARCHIVE_DOMAINS:
        values = imported.get(domain, {})
        _require(isinstance(values, dict) and all(_nonblank(key) for key in values),
                 f'followup_archive {domain} must be a mapping of nonblank IDs')
        target = result.setdefault(domain, {})
        for case_id, row in values.items():
            if domain == 'traces':
                _validate_trace(case_id, row)
            elif domain == 'graphs':
                _validate_graph(case_id, row)
            elif domain == 'distributions':
                _require(isinstance(row, dict) and _nonblank(row.get('scope')),
                         f'Imported distribution {case_id} needs its explicit population scope')
                _distribution(row)
            _require(case_id not in target or target[case_id] == row,
                     f'followup_archive conflicts with existing {domain} ID {case_id}')
            target[case_id] = row
    for cohort_id, ids in imported.get('cohorts', {}).items():
        _require(isinstance(ids, list) and ids and all(_nonblank(case_id) for case_id in ids) and len(ids) == len(set(ids)),
                 f'Imported cohort {cohort_id} needs distinct nonblank case IDs')
        _require(any(set(ids) <= result[domain].keys() for domain in ('graphs', 'traces')),
                 f'Imported cohort {cohort_id} contains unknown or mixed graph and trace IDs')
        _require(_nonblank(result['cohort_descriptions'].get(cohort_id)),
                 f'Imported cohort {cohort_id} needs an explicit cohort description')
    _require(all(key in result['cohorts'] and _nonblank(value)
                 for key, value in imported.get('cohort_descriptions', {}).items()),
             'Imported cohort descriptions must identify existing cohorts')
    provenance = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'identity': envelope['identity'], 'availability': availability}
    result.setdefault('archive_sources', []).append(provenance)
    result.setdefault('availability', {}).update(
        trace_count=len(result['traces']), graph_count=len(result['graphs']),
        distribution_count=len(result['distributions']),
        available_distribution_ids=list(result['distributions']),
        imported_archive_count=len(result['archive_sources']))
    if 'corpus-product' in result['distributions']:
        result['availability']['corpus_distribution'] = {'available': True, 'id': 'corpus-product'}
    return result


def check_archive_source(path, identity, saved_archive):
    """Reject a changed imported source before returning a completed cached run."""
    from common import digest

    path = Path(path)
    resolved = str(path.resolve())
    sources = saved_archive.get('archive_sources', []) if isinstance(saved_archive, dict) else []
    matches = [source for source in sources if isinstance(source, dict) and
               source.get('path') == resolved]
    if len(matches) != 1 or not _same_identity(matches[0].get('identity'), identity):
        raise ValueError('Saved followup_archive source identity is missing or changed; choose a new run directory')
    try:
        observed = digest(path)
    except OSError as error:
        raise ValueError('The saved followup_archive source is missing or unreadable; choose a new run directory') from error
    if observed != matches[0].get('sha256'):
        raise ValueError('followup_archive contents changed; choose a new run directory')
