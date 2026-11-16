"""Synthesize measured results without changing the explanation used in blind tests."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import prompts


def _fields(row, *names):
    return {name: row[name] for name in names if row.get(name) is not None}


def _excerpt(row):
    text = row.get('text') or ''
    selected = row.get('selected') or {}
    if selected.get('offset'):
        _, end = selected['offset']
        prefix, suffix, token = text[:end], text[end:], selected.get('token')
    else:
        index = row.get('peak_index', row.get('position'))
        tokens = row.get('tokens') or [t['token'] for t in row.get('token_activations') or []]
        if index is None or not 0 <= index < len(tokens):
            return ('…' + text[-320:] if len(text) > 320 else text), None, None
        prefix, suffix, token = ''.join(tokens[:index + 1]), ''.join(tokens[index + 1:]), tokens[index]
    if len(prefix) + len(suffix) > 320:
        prefix = ('…' if len(prefix) > 260 else '') + prefix[-260:]
        suffix = suffix[:60] + ('…' if len(suffix) > 60 else '')
    if row.get('cut_left') and not prefix.startswith('…'):
        prefix = '…' + prefix
    return prefix + suffix, len(prefix), token


def _group(rows, key):
    groups = defaultdict(list)
    group_key = key if callable(key) else lambda row: row.get(key)
    for row in rows:
        groups[group_key(row)].append(row)
    return groups.values()


def _representatives(rows, key):
    ranked = sorted(rows, key=lambda row: row[key])
    return [ranked[i] for i in sorted({0, len(ranked) // 2, len(ranked) - 1})]


def _projection_signs(point):
    def sign(value):
        if value is None:
            return '?'
        if value == 0:
            return '0'
        return '+' if value > 0 else '-'

    return sign(point.get('gate')) + '/' + sign(point.get('up'))


def _relp_readouts(graph, neuron):
    details = _fields(graph, 'position', 'activation')
    nodes = {node['node_id']: node for key in ('nodes', 'neuron_nodes', 'logit_nodes')
             for node in graph.get(key) or []}
    target = next((key for key, node in nodes.items()
                   if (node.get('layer'), node.get('feature'), node.get('ctx_idx')) ==
                   (neuron.get('layer'), neuron.get('neuron'), graph.get('position'))), None)
    if target is None:
        return details

    links = sorted(graph.get('links') or [], key=lambda edge: -abs(edge['weight']))
    for direction, center, neighbor in (('incoming', 'target', 'source'), ('outgoing', 'source', 'target')):
        strongest = [edge for edge in links if edge[center] == target and edge[neighbor] in nodes][:3]
        if strongest:
            details[f'{direction}_relp_flow'] = ', '.join(f"{edge[neighbor]}: {edge['weight']:+.4g}"
                                                          for edge in strongest)
    return details


def _evidence(bundle):
    cases = {}

    def add(section, identity, label, row, details, /, **locator):
        excerpt, prefix_end, token = _excerpt(row)
        if token is not None:
            details['scored_token'] = token
        cases[f'{section}:{identity}'] = {'section': section, 'label': label, 'locator': locator,
                                       'text': excerpt, 'prefix_end': prefix_end, 'details': details}

    spectrum = bundle.get('spectrum') or {}
    groups = {row['id']: row['name'] for row in spectrum.get('groups', [])}
    bins = {row['i']: row['label'] for row in spectrum.get('bins', [])}
    sign_counts = {}
    for points in _group(spectrum.get('points', []), 'bin'):
        band = points[0].get('bin')
        sign_counts[band] = dict(Counter(map(_projection_signs, points)))
        representatives = _representatives(points, 'activation')
        covered = {_projection_signs(point) for point in representatives}
        for point in points:
            regime = _projection_signs(point)
            if regime not in covered:
                representatives.append(point)
                covered.add(regime)
        for point in representatives:
            identity = point['id']
            details = _fields(point, 'activation', 'gate', 'up', 'peak_token', 'position', 'predicted_pole')
            details.update(frozen_group=groups.get(point.get('group')), activation_bin=bins.get(band))
            add('spectrum', identity, f'Spectrum · {identity}', point, details)

    search = bundle.get('search') or {}
    for region in search.get('regions') or []:
        frontier = region.get('frontier') or []
        if frontier:
            case = frontier[0]
            add('search', region['id'], region['name'], case,
                _fields(case, 'activation', 'peak_index'))

    custom = bundle.get('custom') or {}
    figures = custom if isinstance(custom, list) else custom.get('figures') or []
    for index, figure in enumerate(figures):
        for sample, case in enumerate(figure.get('results') or []):
            if case.get('activation') is None:
                continue
            identity = case.get('measurement_id') or f'{index + 1}-{sample + 1}'
            details = _fields(case, 'activation', 'peak_token', 'peak_index', 'max', 'min')
            if case.get('scored_tokens'):
                details['scored_positions'] = ', '.join(map(str, case['scored_tokens']))
            add('custom', identity, f"{figure['title']} · {case.get('tag') or case.get('filler') or identity}",
                case, details, figure=figure.get('id'))

    pair = (bundle.get('anatomy') or {}).get('pair') or {}
    for row in pair.get('rows', []):
        identity, anchor = row['case_id'], row['anchor_index']
        details = {'comparison': pair['comparison'], 'position': anchor, 'token': row['tokens'][anchor],
                   'gate': row['gate'][anchor], 'up': row['up'][anchor], 'product': row['act'][anchor],
                   'scoring_location': row['score']['location']}
        selected = {'offset': row['offsets'][anchor], 'token': row['tokens'][anchor]}
        add('anatomy', identity, f"Anatomy · {row['label']}",
            {'text': row['text'], 'selected': selected}, details, id=identity)

    ablation = bundle.get('ablation') or {}
    ablation_rows = ablation.get('rows') or []
    for rows in _group((r for r in ablation_rows if r.get('kl') is not None), 'kind'):
        for row in _representatives(rows, 'kl'):
            identity = row.get('case_id') or str(ablation_rows.index(row) + 1)
            details = _fields(row, 'kind', 'position', 'peak_token', 'saved_product', 'live_product',
                              'relative_product_magnitude', 'kl', 'next_token', 'next_logprob_base', 'next_logprob_ablated')
            details['same_layer_control_kl'] = ', '.join(map(str, row.get('kl_controls') or []))
            add('ablation', identity, f'Ablation · {identity}', row, details, id=identity)

    tests = bundle.get('tests') or {}
    for key, label in [('blind_max', 'Blind maximum'), ('blind_min', 'Blind minimum')]:
        if tests.get(key):
            add('tests', key, label, tests[key], _fields(tests[key], 'target', 'achieved', 'peak_index'), id=key)
    route = tests.get('route_prediction') or {}
    for rows in _group(route.get('rows') or [], 'correct'):
        row = rows[0]
        identity = row['id']
        add('tests', 'route:' + identity, f'Route prediction · {identity}', row,
            _fields(row, 'token', 'position', 'predicted_route', 'actual_route', 'correct'), id=identity)

    for index, row in enumerate(bundle.get('criticism') or []):
        identity = row.get('investigation_id') or f'original-{index + 1}'
        failed = (row.get('result') or '').startswith('Investigation failed:')
        details = {'verdict': row.get('verdict'), 'result': 'Investigation failed' if failed else row.get('result')}
        add('criticism', identity, f'Criticism · {identity}', {'text': row.get('attack')}, details, row=index)
        for sample, case in enumerate(row.get('evidence') or []):
            sample_id = case.get('id') or str(sample + 1)
            details = _fields(case, 'activation', 'peak_index')
            details.update(_fields(case.get('selected') or {}, 'position', 'token', 'gate', 'value', 'product'))
            details.update(_fields(case.get('score') or {}, 'location', 'reduction'))
            add('criticism', f'{identity}:{sample_id}', f'Criticism · {identity} · {sample_id}',
                case, details, row=index, sample=sample)

    neuron = bundle.get('neuron') or {}
    family = bundle.get('family') or {}
    for i, measurement in enumerate(family.get('measurements') or []):
        readouts = measurement.get('readouts') or []
        target = next((r for r in readouts
                       if (r['layer'], r['neuron']) == (neuron.get('layer'), neuron.get('neuron'))), None)
        if target and target.get('values'):
            values = target['values']
            anchor = max(range(len(values)), key=lambda index: abs(values[index]))
            related = ', '.join(f"L{row['layer']}_N{row['neuron']} {row['values'][anchor]}"
                                for row in readouts if row is not target and len(row['values']) > anchor)
            add('family', i + 1, f'Family prompt {i + 1}',
                {'text': measurement.get('prompt'), 'tokens': measurement['tokens'], 'position': anchor},
                {'position': anchor, 'token': measurement['tokens'][anchor],
                 'target_product': values[anchor], 'related_products_at_same_position': related})

    circuit = bundle.get('circuit') or {}
    labels = bundle.get('neuron_labels') or {}
    for pick in (bundle.get('graphs_curated') or {}).get('picks', []):
        graph = pick['graph']
        identity = pick['id']
        add('circuit', identity, f'RelP · {identity}', graph, _relp_readouts(graph, neuron), graph=identity)
    for direction in ('upstream', 'downstream'):
        for row in (circuit.get(direction) or [])[:3]:
            unit = f"L{row['layer']}_N{row['neuron']}"
            label = labels.get('labels', {}).get(unit) or {}
            identities = label.get('evidence_ids') or []
            if identities:
                identity = identities[0]
                case = labels['evidence'][identity]
                details = {'direction': direction, 'unit': unit, 'label': label['label'], 'atlas_weight': row['weight'],
                           **_fields(case, 'token', 'position', 'activation')}
                add('circuit', f'{direction}:{identity}', f'Circuit · {unit}', {'text': case.get('context')}, details)

    summary = {'corpus': _fields(bundle.get('distribution') or {}, 'n_tokens', 'corpus_min', 'corpus_max', 'quantiles'),
               'spectrum': {**_fields(spectrum, 'bins', 'corpus_hist', 'sampling'),
                            'sampled_gate_up_sign_counts': sign_counts},
               'controlled_sweeps': [_fields(figure, 'title', 'template') for figure in figures],
               'ablation': _fields(ablation.get('summary') or {}, 'n', 'coverage', 'median_kl', 'median_kl_controls'),
               'route_prediction': _fields(route, 'status', 'accuracy', 'baseline', 'n', 'train_n', 'test_n', 'unassigned_n',
                                           'route_ids', 'route_names', 'confusion'),
               'anatomy_pair': _fields(pair, 'template', 'anchor', 'explanation'),
               'circuit': _fields(circuit, 'n_atlas_graphs', 'method')}
    return cases, summary


def _prompt_case(case):
    fields = {key: value for key, value in case.items() if key not in ('locator', 'prefix_end')}
    end = case['prefix_end']
    if end is not None:
        text = fields.pop('text')
        fields['observed_prefix_excerpt'] = text[:end]
        if text[end:]:
            fields['following_text'] = text[end:]
    return fields


def _assemble(cases, claim, section_prose):
    passages = [*claim['findings'].values(), *section_prose.values()]
    cited = dict.fromkeys(identity for passage in passages for identity in passage['evidence_ids'])
    return {'claim': claim, 'section_prose': section_prose,
            'evidence': {identity: cases[identity] for identity in cited}}


def _response_value(case):
    details = case['details']
    if case['section'] == 'ablation':
        return details['kl']
    return next((details[key] for key in ('activation', 'product', 'target_product', 'live_product')
                 if type(details.get(key)) in (int, float)), None)


def _contrasts(cases):
    measured = [(key, value) for key, case in cases.items() if (value := _response_value(case)) is not None]
    if not measured:
        return dict(list(cases.items())[:1])
    chosen = {min(measured, key=lambda row: row[1])[0], max(measured, key=lambda row: row[1])[0],
              min(measured, key=lambda row: abs(row[1]))[0]}
    return {key: case for key, case in cases.items() if key in chosen}


def _section_cases(section, cases):
    if section == 'spectrum':
        groups = _group(cases.items(), lambda row: _projection_signs(row[1]['details']))
    elif section == 'custom':
        groups = _group(cases.items(), lambda row: row[1]['locator'].get('figure'))
    elif section == 'criticism':
        groups = _group(((key, case) for key, case in cases.items() if 'sample' in case['locator']),
                        lambda row: row[1]['locator']['row'])
        summaries = {key: case for key, case in cases.items() if 'sample' not in case['locator']}
        return {**summaries, **{key: case for group in groups for key, case in _contrasts(dict(group)).items()}}
    else:
        return cases
    return {key: case for group in groups for key, case in _contrasts(dict(group)).items()}


SECTION_MEASUREMENTS = {
    'spectrum': ('corpus', 'spectrum'), 'search': ('corpus',), 'custom': ('controlled_sweeps',),
    'anatomy': ('anatomy_pair',), 'ablation': ('ablation',), 'tests': ('route_prediction',),
    'circuit': ('circuit',),
}
FINDING_SECTIONS = {
    'pattern': ('spectrum', 'search', 'custom', 'tests', 'criticism'),
    'strength': ('spectrum', 'search', 'custom', 'anatomy', 'tests'),
    'mechanism': ('spectrum', 'anatomy', 'ablation', 'circuit', 'family'),
}


def synthesize(client, bundle, stage):
    cases, measurements = _evidence(bundle)
    if not cases:
        return None
    context = {'neuron': bundle.get('neuron'), 'frozen_claim': bundle.get('claim')}
    sections = sorted({case['section'] for case in cases.values()})
    references = {section: _section_cases(section, {key: case for key, case in cases.items()
                                                   if case['section'] == section}) for section in sections}

    def fill(section_names, slot, prompt, references, **content):
        metrics = dict.fromkeys(key for section in section_names for key in SECTION_MEASUREMENTS.get(section, ()))
        packet = {**context, **content, 'measurements': {key: measurements[key] for key in metrics},
                  'cases': {key: _prompt_case(case) for key, case in references.items()}}
        prose = client.request_text(f'report-synthesis/{slot}', prompt, packet, stage)
        return prose, list(references)

    def section_slot(section):
        prose, ids = fill((section,), f'sections/{section}', prompts.REPORT_SECTION_SLOT,
                          references[section], section=section)
        return {'summary': prose, 'evidence_ids': ids}

    def finding_slot(kind):
        scope = [section for section in FINDING_SECTIONS.get(kind, sections) if section in section_prose]
        selected = {key: case for section in scope for key, case in _contrasts(references[section]).items()}
        prose, ids = fill(scope, f'findings/{kind}', prompts.REPORT_FINDING_SLOT, selected,
                          focus=prompts.REPORT_FINDING_SLOTS[kind],
                          section_summaries={section: section_prose[section]['summary'] for section in scope})
        return {'text': prose, 'evidence_ids': ids}

    def overview_slot(key, prompt):
        return client.request_text(f'report-synthesis/overview/{key}', prompt, overview, stage)

    with ThreadPoolExecutor(max_workers=5) as pool:
        section_prose = dict(zip(sections, pool.map(section_slot, sections)))
        kinds = prompts.REPORT_FINDING_SLOTS
        findings = dict(zip(kinds, pool.map(finding_slot, kinds)))
        overview = {'neuron': context['neuron'], 'findings': {key: value['text'] for key, value in findings.items()}}
        title = pool.submit(overview_slot, 'title', prompts.REPORT_TITLE_SLOT)
        lede = pool.submit(overview_slot, 'lede', prompts.REPORT_LEDE_SLOT)
        claim = {'title': title.result(), 'lede': lede.result(), 'findings': findings}
    return _assemble(cases, claim, section_prose)
