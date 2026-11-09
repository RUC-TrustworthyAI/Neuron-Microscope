"""Register and execute Colorado-style experiments against project-owned evidence."""

import hashlib
import json
import math
import random
from bisect import bisect_right
from collections import defaultdict
from itertools import accumulate
from statistics import median

from . import contracts


TEXT = {'type': 'string', 'minLength': 1}
ID = {'type': 'string', 'minLength': 1, 'maxLength': 48}
NATURAL = {'type': 'integer', 'minimum': 0}
POSITIVE = {'type': 'integer', 'minimum': 1}
NUMBER = {'type': 'number'}
PROPORTION = {'type': 'number', 'minimum': 0, 'maximum': 1}
FULL_SCORE = {'location': 'input', 'reduction': 'absmax'}
RESAMPLE_SETTINGS = ('seed', 'tolerance', 'quantile', 'group', 'activation_band')


def _object(required, **optional):
    schema = contracts.obj(**required, **optional)
    schema['required'] = list(required)
    return schema


def _readout(named=True):
    fields = {'location': contracts.string('input', 'marker', 'token_position', 'token_range'),
              'reduction': contracts.string('max', 'min', 'absmax', 'single'),
              'metric': contracts.string('product', 'peak_position')}
    if named:
        fields = {'id': ID, **fields}
    return _object(fields, marker=contracts.S, occurrence=NATURAL, token_offset={'type': 'integer'},
                   start=NATURAL, end=POSITIVE)


def operation_fields(options, *, archive_only=False):
    """Describe the four evidence operations available to the planner."""
    probe = {'id': ID, 'label': contracts.SHORT_LABEL, 'readouts': contracts.arr(_readout(), 1, 32)}
    probe = (contracts.obj(**probe, source=contracts.string('archive'), case_id=TEXT) if archive_only else
             _object({**probe, 'source': contracts.string('new', 'archive')}, text=TEXT, case_id=TEXT))
    selector = {'case_ids': contracts.arr(TEXT, 1), 'cohort_id': TEXT}
    group = _object({'id': ID}, **selector)
    graph_query = _object({'id': ID, 'layer': NATURAL, 'neuron': NATURAL,
                           'groups': contracts.arr(group, 1, 16)},
                          view=contracts.string('target', 'census'),
                          census_limit={'type': 'integer', 'minimum': 1, 'maximum': 128})
    resample = _object({'id': ID, 'sample_size': POSITIVE, 'seed': NATURAL,
                        'tolerance': {'type': 'number', 'minimum': 0},
                        'quantile': PROPORTION, 'readout': _readout(False)}, **selector,
                       group=TEXT, activation_band=contracts.arr(NUMBER, 2, 2))
    quantile = contracts.obj(id=ID, distribution_id=TEXT, quantile=PROPORTION)
    return {'probes': contracts.arr(probe, 0, options['max_conditions']),
            'graph_queries': contracts.arr(graph_query, 0, 6),
            'resamples': contracts.arr(resample, 0, 6),
            'quantiles': contracts.arr(quantile, 0, 6)}


def _metric_id(*parts):
    return '.'.join(part.replace('%', '%25').replace('.', '%2E') for part in parts)


def _finite(value):
    return type(value) in (float, int) and -math.inf < value < math.inf


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                     sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def _case_ids(selection, archive, domain):
    direct, cohort = selection.get('case_ids'), selection.get('cohort_id')
    if (direct is None) == (cohort is None):
        raise ValueError('Select either explicit case_ids or one cohort_id')
    ids = direct if cohort is None else archive.get('cohorts', {}).get(cohort)
    if not ids or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'Unknown, empty, or duplicated {domain} selection')
    missing = set(ids) - archive.get(domain, {}).keys()
    if missing:
        raise ValueError(f'Unknown saved {domain}: {sorted(missing)[:5]}')
    return sorted(ids)


def _trace_layout(row):
    trace = row.get('trace')
    if not trace:
        raise ValueError('The saved case has no token activation trace')
    positions = [item['position'] for item in trace]
    tokens = row.get('token_ids')
    full = (row.get('trace_complete', True) and positions == list(range(len(positions))) and
            (tokens is None or len(tokens) == len(positions)))
    return trace, positions, full


def _register_readout(readout, text, offsets, positions, complete, default_id=None):
    from common import score_positions

    location, reduction = readout['location'], readout['reduction']
    score = dict(location=location, reduction=reduction)
    if location != 'marker' and (readout.get('marker') or readout.get('occurrence', 0) or
                                 'token_offset' in readout):
        raise ValueError('Marker and occurrence belong only to marker readouts')
    start, end = readout.get('start'), readout.get('end')
    if location in ('input', 'marker'):
        if start is not None or end is not None:
            raise ValueError('Token positions belong only to token_position or token_range readouts')
        if not complete:
            raise ValueError(f'{location} scoring requires a complete saved trace')
        if location == 'marker':
            if any(offset is None for offset in offsets):
                raise ValueError('Marker scoring requires character offsets')
            score.update(text=readout.get('marker', ''), occurrence=readout.get('occurrence', 0))
            if 'token_offset' in readout:
                score['token_offset'] = readout['token_offset']
        selected = score_positions(text, offsets, (0, len(text)), score)
    elif location == 'token_position':
        if start is None or end is not None:
            raise ValueError('token_position needs one explicit start position')
        score['position'], selected = start, [start]
    else:
        if start is None or end is None or end <= start:
            raise ValueError('token_range needs a nonempty half-open [start, end) range')
        if end - start > len(positions):
            raise ValueError('Readout extends beyond measured token positions')
        score.update(start=start, end=end)
        selected = list(range(start, end))
    if not selected or not set(selected) <= set(positions):
        raise ValueError('Readout extends beyond measured token positions')
    if reduction == 'single' and len(selected) != 1:
        raise ValueError('Single-token scoring resolved to multiple tokens')
    return {'id': readout.get('id', default_id), 'score': score,
            'metric': readout['metric'], 'scored_positions': selected}


def _register_graph(query, archive):
    view = query.get('view', 'target')
    groups = {}
    for group in query['groups']:
        group_id = group['id']
        if group_id in groups:
            raise ValueError('Graph groups must have distinct IDs')
        ids = _case_ids(group, archive, 'graphs')
        nodes = [archive['graphs'][i]['nodes'] for i in ids]
        groups[group_id] = {'id': group_id, 'case_ids': ids, 'archive_digest': _digest(nodes)}
    return {'kind': 'graph_query', 'id': query['id'], 'layer': query['layer'], 'neuron': query['neuron'],
            'groups': list(groups.values()),
            **({'view': 'census', 'census_limit': query.get('census_limit', 32)} if view == 'census' else {})}


def _distribution(data):
    if isinstance(data, dict):
        values, counts = data.get('values'), data.get('counts')
        if (not isinstance(values, list) or not isinstance(counts, list) or
                not values or len(values) != len(counts) or
                any(type(n) is not int or n <= 0 for n in counts)):
            raise ValueError('Archive distribution needs finite values with positive integer counts')
    elif isinstance(data, list) and data:
        values, counts = data, [1] * len(data)
    else:
        raise ValueError('Archive distribution has no observed values')
    if any(not _finite(x) for x in values):
        raise ValueError('Archive distribution contains a nonfinite value')
    return sorted(zip(values, counts))


def _percentile(weighted, quantile):
    """Linear interpolation at (n - 1) * q on exact order statistics."""
    totals = list(accumulate(n for _, n in weighted))
    count = totals[-1]
    index = (count - 1) * quantile
    low, high = math.floor(index), math.ceil(index)
    start, stop = (weighted[bisect_right(totals, i)][0] for i in (low, high))
    result = start * (1 - (index - low)) + stop * (index - low)
    if not _finite(result):
        raise ValueError('Computed quantile is not finite')
    return count, result


def prepare(plan, tokenizer, archive, options, round_id, investigation_index):
    """Resolve exact model inputs, archive selections and scoring positions."""
    from common import tokenize

    operations, model_inputs = {}, {}
    prefix = f'followup-{round_id}-{investigation_index}'
    archive_traces = archive.get('traces', {})

    def input_for(text, label):
        ids, offsets = tokenize(tokenizer, text, options['max_input_tokens'])
        key = tuple(ids)
        if key not in model_inputs:
            model_inputs[key] = {'id': f'{prefix}-forward-{len(model_inputs) + 1}', 'label': label, 'text': text,
                                 'score': dict(FULL_SCORE), 'token_ids': ids}
        return model_inputs[key]['id'], offsets, list(range(len(ids)))

    def add_operation(operation):
        oid = operation['id']
        if oid in operations:
            raise ValueError('Operation IDs must be unique within an investigation')
        operations[oid] = operation

    for probe in plan.get('probes', []):
        oid = probe['id']
        source, label = probe['source'], probe['label']
        source_details = {}
        if source == 'new':
            if not probe.get('text', '').strip() or 'case_id' in probe:
                raise ValueError('A new probe needs nonblank text and no archived case ID')
            text = probe['text']
            measurement_id, offsets, positions = input_for(text, label)
            full = True
        else:
            case_id = probe.get('case_id')
            if 'text' in probe:
                raise ValueError(f'Archive probe {oid} must omit text; provide case_id only.')
            if case_id not in archive_traces:
                raise ValueError(f'Unknown archive case ID: {case_id!r}; copy an exact ID from archive.traces')
            saved = archive_traces[case_id]
            text = saved['text']
            trace, positions, full = _trace_layout(saved)
            offsets = [token.get('offset') for token in trace]
            measurement_id = case_id
            source_details = {'case_id': case_id,
                              'archive_digest': _digest({'trace': trace, 'text': text})}
        try:
            readouts = [_register_readout(r, text, offsets, positions, full) for r in probe['readouts']]
        except ValueError as error:
            raise ValueError(f'Probe {oid} ({measurement_id}, positions {positions[0]}..{positions[-1]}): '
                             f'{error}') from error
        if len({r['id'] for r in readouts}) != len(readouts):
            raise ValueError(f'Probe {oid} needs distinct readout IDs')
        add_operation({'id': oid, 'kind': 'probe', 'source': source, 'label': label, 'text': text,
                       'condition_id': f'{prefix}-probe-{oid}', 'measurement_id': measurement_id,
                       'offsets': offsets, 'readouts': readouts, **source_details})
    for query in plan.get('graph_queries', []):
        add_operation(_register_graph(query, archive))
    for query in plan.get('resamples', []):
        oid = query['id']
        if query['readout']['metric'] != 'product':
            raise ValueError('Isolation replay compares saved and new native products')
        candidates = _case_ids(query, archive, 'traces')
        band = query.get('activation_band')
        group = query.get('group')
        population = []
        for case_id in candidates:
            row = archive_traces[case_id]
            if group is not None and row.get('group') != group:
                continue
            original = row['selected']['product']
            if band is None or band[0] <= original <= band[1]:
                population.append((case_id, original))
        if query['sample_size'] > min(len(population), options['max_conditions']):
            raise ValueError('Isolation sample exceeds its eligible population or condition budget')
        selected = [case_id for case_id, _ in random.Random(query['seed']).sample(population, query['sample_size'])]
        pairs = []
        for n, case_id in enumerate(selected, 1):
            row = archive_traces[case_id]
            if row.get('isolation_available') is False:
                reason = row.get('isolation_unavailable_reason', 'The saved isolation input is unavailable.')
                raise ValueError(f'Isolation replay unavailable for {case_id}: {reason}')
            text = row.get('isolation_text', row['text'])
            measurement_id, offsets, positions = input_for(text, case_id)
            readout = _register_readout(query['readout'], text, offsets, positions, True, 'isolated')
            pairs.append({'case_id': case_id, 'text': text, 'condition_id': f'{prefix}-resample-{oid}-{n}',
                          'measurement_id': measurement_id, 'offsets': offsets, 'readout': readout})
        add_operation({'id': oid, 'kind': 'resample', 'candidate_count': len(candidates),
                       'population_ids': [case_id for case_id, _ in population], 'pairs': pairs,
                       'selected_case_ids': selected,
                       'archive_digest': _digest(population),
                       **{k: query[k] for k in RESAMPLE_SETTINGS if k in query}})
    for query in plan.get('quantiles', []):
        distribution = archive.get('distributions', {}).get(query['distribution_id'])
        _distribution(distribution)
        add_operation({'kind': 'quantile', **query, 'archive_digest': _digest(distribution)})
    if not operations:
        raise ValueError('An investigation needs at least one executable or archive operation')
    if len(model_inputs) > options['max_conditions']:
        raise ValueError('Distinct forward inputs exceed the condition budget')
    return {'operations': list(operations.values()), 'model_inputs': list(model_inputs.values()),
            'key': prefix}


def _check_forward(item, row):
    if (row.get('status') != 'complete' or
            any(row.get(key) != item[key] for key in ('id', 'text', 'score', 'token_ids'))):
        raise ValueError('Target-model result does not match its registered input')
    trace, positions, full = _trace_layout(row)
    if (not full or row.get('scored_positions') != positions or
            [t.get('token_id') for t in trace] != item['token_ids'] or
            any(not _finite(t.get(key)) for t in trace for key in ('gate', 'value', 'product'))):
        raise ValueError('Target-model result has a missing, nonfinite, or mismatched native trace')


def _read(row, readout, metric_id, offsets=None):
    from common import select_peak

    tokens = {t['position']: t for t in row['trace']}
    try:
        selected = dict(select_peak([tokens[i] for i in readout['scored_positions']],
                                    readout['score']['reduction']))
    except KeyError as error:
        raise ValueError('A registered token is absent from its measured trace') from error
    if offsets is not None:
        selected['offset'] = offsets[selected['position']]
    metric = readout['metric']
    value = selected['product'] if metric == 'product' else selected['position']
    return {**readout, 'metric_id': metric_id, 'selected': selected, 'value': value}


def _condition(operation, readouts, source=None):
    return {'id': operation['condition_id'], 'label': operation.get('label', operation.get('case_id')),
            'text': operation['text'], 'source': source or operation['source'],
            'measurement_id': operation['measurement_id'], 'readouts': readouts,
            'score': readouts[0]['score'], 'selected': readouts[0]['selected'],
            **({'case_id': operation['case_id']} if 'case_id' in operation else {})}


def _unit_statistics(nodes_by_case, graph_count):
    # Rank and influence contribute at most once per graph; activation is per node.
    ranks, influences, activations = [], [], []
    for nodes in nodes_by_case.values():
        rank = min((x for n in nodes if _finite(x := n.get('rank')) and x >= 1), default=None)
        influence = max((x for n in nodes if _finite(x := n.get('influence'))), key=abs, default=None)
        if rank is not None:
            ranks.append(rank)
        if influence is not None:
            influences.append(influence)
        activations.extend(x for n in nodes if _finite(x := n.get('activation')))
    present = len(nodes_by_case)
    statistics = {'graph_count': graph_count, 'presence_count': present, 'presence_rate': present / graph_count,
                  'negative_count': sum(x < 0 for x in activations)}
    for field, values in (('rank', ranks), ('influence', influences), ('activation', activations)):
        statistics[f'{field}_count'] = len(values)
        if values:
            statistics[f'{field}_median'] = median(values)
    if activations:
        statistics['negative_share'] = statistics['negative_count'] / len(activations)
    return statistics


def _graph_statistics(operation, group, archive):
    cases = [archive['graphs'][case_id]['nodes'] for case_id in group['case_ids']]
    if _digest(cases) != group['archive_digest']:
        raise ValueError('Registered graph archive changed before execution')
    by_unit = defaultdict(lambda: defaultdict(list))
    census = operation.get('view') == 'census'
    target_id = (operation['layer'], operation['neuron'])
    for case_id, nodes in zip(group['case_ids'], cases):
        for node in nodes:
            unit = (node.get('layer'), node.get('neuron'))
            if unit == target_id or (census and not node.get('is_target_neuron') and
                                     all(type(value) is int and value >= 0 for value in unit)):
                by_unit[unit][case_id].append(node)
    target = by_unit.pop(target_id, {})
    result = {'id': group['id'], 'case_ids': group['case_ids'],
              'statistics': _unit_statistics(target, len(cases)),
              'matches': [{'case_id': case_id, 'nodes': nodes} for case_id, nodes in target.items()]}
    if census:
        units = [{'layer': layer, 'neuron': neuron, **_unit_statistics(nodes, len(cases)),
                  'example_case_ids': sorted(nodes)[:5]} for (layer, neuron), nodes in by_unit.items()]
        units.sort(key=lambda row: (-row['presence_count'], row.get('rank_median', math.inf),
                                   -abs(row.get('influence_median', 0)), row['layer'], row['neuron']))
        limit = operation['census_limit']
        result['census'] = {'units': units[:limit], 'total_unit_count': len(units),
            'retained_unit_count': min(len(units), limit), 'limit': limit,
            'scope': 'Other retained upstream MLP units in the selected saved graphs. '
                     'Explicit root nodes and the separately queried unit are excluded. '
                     'Presence uses all selected graphs; rank and influence summarize one strongest observation '
                     'per graph, while activation and negative share summarize all observed nodes. '
                     'This census does not reconstruct omitted graph nodes.',
            'ordering': 'Presence count descending, best saved rank median ascending, absolute influence median '
                        'descending, then layer and neuron ascending.'}
    return result


def _probe_observation(operation, row, offsets=None):
    oid = operation['id']
    readouts = [_read(row, r, _metric_id(oid, r['id']), offsets) for r in operation['readouts']]
    condition = _condition(operation, readouts)
    result = {'id': oid, 'kind': 'probe', 'source': condition['source'],
              'condition_id': condition['id'], 'measurement_id': condition['measurement_id'],
              'readouts': readouts}
    if 'case_id' in condition:
        result['case_id'] = condition['case_id']
    return {'metrics': {r['metric_id']: r['value'] for r in readouts},
            'conditions': [condition], 'operation': result}


def _saved_observation(operation, archive):
    """Resolve saved evidence without initiating target-model work."""
    oid, kind = operation['id'], operation['kind']
    if kind == 'probe':
        row = archive['traces'][operation['case_id']]
        if _digest({'trace': row['trace'], 'text': row['text']}) != operation['archive_digest']:
            raise ValueError('Registered trace archive changed before execution')
        return _probe_observation(operation, row)
    result = {'id': oid, 'kind': kind}
    if kind == 'graph_query':
        groups = [_graph_statistics(operation, group, archive) for group in operation['groups']]
        metrics = {_metric_id(oid, g['id'], k): v for g in groups for k, v in g['statistics'].items()}
        result.update(layer=operation['layer'], neuron=operation['neuron'], groups=groups)
    else:
        did = operation['distribution_id']
        distribution = archive['distributions'][did]
        if _digest(distribution) != operation['archive_digest']:
            raise ValueError('Registered activation distribution changed before execution')
        n, value = _percentile(_distribution(distribution), operation['quantile'])
        statistics = {'count': n, 'value': value}
        metrics = {_metric_id(oid, k): v for k, v in statistics.items()}
        scope = distribution.get('scope', did) if isinstance(distribution, dict) else did
        result.update(distribution_id=did, quantile=operation['quantile'], scope=scope, statistics=statistics)
    return {'metrics': metrics, 'conditions': [], 'operation': result}


def execute(registration, measure, archive):
    """Inspect saved evidence before new forwards, retaining registered result order."""
    saved, populations = {}, {}
    traces = archive.get('traces', {})
    # Resolve saved observations and fingerprints before scheduling model work.
    operations = registration['operations']
    for operation in operations:
        oid, kind = operation['id'], operation['kind']
        try:
            if kind in ('graph_query', 'quantile') or (kind == 'probe' and operation['source'] == 'archive'):
                saved[oid] = _saved_observation(operation, archive)
            elif kind == 'resample':
                population = [(i, traces[i]['selected']['product']) for i in operation['population_ids']]
                if _digest(population) != operation['archive_digest']:
                    raise ValueError('Registered isolation population changed before execution')
                populations[oid] = population
        except KeyError as error:
            raise ValueError('Registered saved evidence is unavailable before execution') from error

    inputs = registration['model_inputs']
    measurements = measure.texts(inputs, registration['key']) if inputs else []
    for item, row in zip(inputs, measurements):
        _check_forward(item, row)
    new = {row['id']: row for row in measurements}
    metrics, conditions, results = {}, [], []
    for operation in operations:
        oid, kind = operation['id'], operation['kind']
        if oid in saved:
            observed = saved[oid]
        elif kind == 'probe':
            observed = _probe_observation(operation, new[operation['measurement_id']], operation['offsets'])
        elif kind == 'resample':
            population = populations[oid]
            pairs, replay_conditions, original = [], [], dict(population)
            for pair in operation['pairs']:
                case_id = pair['case_id']
                readout = _read(new[pair['measurement_id']], pair['readout'], _metric_id(oid, 'case', case_id),
                                pair['offsets'])
                isolated = readout['value']
                delta = abs(isolated - original[case_id])
                if not _finite(delta):
                    raise ValueError('Isolation replay produced a nonfinite difference')
                replay_conditions.append(_condition(pair, [readout], source='isolation'))
                pairs.append({'case_id': case_id, 'condition_id': pair['condition_id'],
                              'original': original[case_id], 'isolated': isolated,
                              'absolute_difference': delta, 'reproduced': delta <= operation['tolerance']})
            q = operation['quantile']
            reproduced = sum(p['reproduced'] for p in pairs)
            statistics = {'candidate_count': operation['candidate_count'], 'population_count': len(population),
                          'sample_count': len(pairs), 'reproduced_count': reproduced,
                          'reproduced_rate': reproduced / len(pairs),
                          'population_quantile': _percentile(sorted((v, 1) for _, v in population), q)[1],
                          'isolated_quantile': _percentile(sorted((p['isolated'], 1) for p in pairs), q)[1]}
            result = {'id': oid, 'kind': kind, 'statistics': statistics, 'pairs': pairs,
                      **{k: operation[k] for k in ('population_ids', 'selected_case_ids', *RESAMPLE_SETTINGS)
                         if k in operation}}
            observed = {'metrics': {_metric_id(oid, k): v for k, v in statistics.items()},
                        'conditions': replay_conditions, 'operation': result}
        else:
            raise ValueError(f'Unregistered operation kind: {kind}')
        metrics.update(observed['metrics'])
        conditions.extend(observed['conditions'])
        results.append(observed['operation'])
    return {'metrics': metrics, 'conditions': conditions, 'operations': results, 'measurements': measurements}
