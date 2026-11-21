"""Register matched ablation cases from saved corpus and controlled measurements."""
import math
import random
from collections import Counter


WEAK_MAX = 0.10
MEDIUM_MAX = 0.50


def _natural_case(identity, point, documents, group=None):
    document_id = str(point['document'])
    document = documents.get(document_id)
    position, product = point['position'], point['values'][2]
    if document is None or not math.isfinite(product):
        return None
    ids, tokens = document['token_ids'], document['tokens']
    if not 0 <= position < len(ids) or len(ids) != len(tokens):
        return None
    return {
        'case_id': f'corpus-{identity}', 'source_id': identity, 'document_id': document_id,
        'position': position, 'token_ids': ids[:position + 1],
        'next_token_id': ids[position + 1] if position + 1 < len(ids) else None,
        'saved_product': product, 'text': ''.join(tokens[:position + 1]),
        'peak_token': tokens[position], 'group': group,
    }


def _controlled_case(row, sweep_id):
    selected = row['selected']
    position, product = selected['position'], selected['product']
    ids, trace = row['token_ids'], row['trace']
    if not math.isfinite(product) or not 0 <= position < min(len(ids), len(trace)):
        return None
    return {
        'case_id': f"controlled-{row['id']}", 'source_id': row['id'], 'kind': 'controlled',
        'sweep_id': sweep_id, 'condition_id': row['id'], 'condition_label': row.get('label'),
        'position': position, 'token_ids': ids[:position + 1],
        'next_token_id': ids[position + 1] if position + 1 < len(ids) else None,
        'saved_product': product, 'text': ''.join(token['token'] for token in trace[:position + 1]),
        'peak_token': selected['token'],
    }


def _key(case):
    return tuple(case['token_ids']), case['position']


def _distinct(candidates, used):
    seen = set(used)
    for case in candidates:
        key = _key(case)
        if key not in seen:
            seen.add(key)
            yield case


def _controlled(rows, figures, preferred_id, limit, used):
    if not limit:
        return None, []
    by_id = {row['id']: row for row in rows if row.get('status') == 'complete'}
    sweeps = []
    for figure in figures:
        if figure.get('kind') != 'sweep':
            continue
        sweep_id = figure['id']
        candidates = [_controlled_case(by_id[result['measurement_id']], sweep_id)
                      for result in figure['results'] if result.get('measurement_id') in by_id]
        unique = list(_distinct((case for case in candidates if case is not None), used))
        if unique:
            sweeps.append((sweep_id, unique))
    if not sweeps:
        return None, []
    sweep_id, cases = max(sweeps, key=lambda item: (min(len(item[1]), limit), item[0] == preferred_id))
    if len(cases) > limit:
        # Keep the order of the retained sweep and span its whole list of conditions.
        cases = [cases[i * (len(cases) - 1) // max(limit - 1, 1)] for i in range(limit)]
    return sweep_id, cases


def _sign(case):
    product = case['saved_product']
    return (product > 0) - (product < 0)


def _diverse(candidates, limit, used, documents, rng):
    pool = list(_distinct(sorted(candidates, key=lambda case: case['case_id']), used))
    rng.shuffle(pool)
    chosen, local_documents, signs, groups = [], Counter(), Counter(), Counter()

    def rank(case):
        document, group = case['document_id'], case['group']
        return (local_documents[document], signs[_sign(case)], group is None, groups[group], documents[document])

    while pool and len(chosen) < limit:
        case = min(pool, key=rank)
        pool.remove(case)
        documents[case['document_id']] += 1
        local_documents[case['document_id']] += 1
        signs[_sign(case)] += 1
        groups[case['group']] += 1
        chosen.append(case)
    return chosen


def select_ablation_cases(sample, contexts, control_rows, figures, *, data=None, predictions=None,
                          preferred_figure_id=None, seed=42, natural_top=20, natural_bottom=10,
                          medium=20, weak=20, controlled=10):
    """Select signed corpus tails, diverse nonextreme anchors and one retained sweep."""
    requested = {'strong-positive': natural_top, 'strong-negative': natural_bottom,
                 'medium': medium, 'weak': weak, 'controlled': controlled}
    if any(type(count) is not int or count < 0 for count in requested.values()):
        raise ValueError('Ablation case counts must be nonnegative integers')
    data = data or {}
    sample_points = {point['id']: point for point in sample['points']}
    labels = {row['id']: row['group'] for row in (predictions or {}).get('records', [])}
    spectrum = []
    for identity, point in contexts['points'].items():
        metadata = sample_points.get(identity)
        if metadata is None:
            continue
        group = labels.get(identity, metadata.get('semantic_group', metadata.get('group')))
        case = _natural_case(identity, point, contexts['documents'], None if group == 'unlabelled' else group)
        if case is not None:
            case['dataset_token_index'] = metadata.get('dataset_token_index')
            spectrum.append(case)

    used, strata, tails = set(), {}, {}
    for kind, sign, field in [('strong-positive', 1, 'top_tokens'), ('strong-negative', -1, 'bottom_tokens')]:
        ranked = [str(identity) for identity in data.get('examples', {}).get(field, [])]
        if ranked:
            saved_points, saved_documents = data['points'], data['documents']
            candidates = [_natural_case(identity, saved_points[identity], saved_documents) for identity in ranked]
            for identity, case in zip(ranked, candidates):
                if case is not None and identity.isdecimal():
                    case['dataset_token_index'] = int(identity)
            tails[kind] = f'Saved corpus ranking ({field}).'
        else:
            candidates = sorted(spectrum, key=lambda case: (-sign * case['saved_product'], case['case_id']))
            tails[kind] = 'Saved natural spectrum candidate pool.'
        selected = list(_distinct((case for case in candidates if case is not None and _sign(case) == sign), used))
        strata[kind] = selected[:requested[kind]]
        used.update(map(_key, strata[kind]))

    sweep, strata['controlled'] = _controlled(control_rows, figures, preferred_figure_id, controlled, used)
    used.update(map(_key, strata['controlled']))
    statistics = data.get('statistics', {}).get('product', {})
    if 'min' in statistics and 'max' in statistics:
        low, high = statistics['min'], statistics['max']
        magnitude_scope = 'Saved corpus native product extrema.'
    else:
        values = [case['saved_product'] for case in spectrum]
        low, high = min(values, default=0.), max(values, default=0.)
        magnitude_scope = 'Native product extrema of the saved natural spectrum candidate pool.'

    def relative_magnitude(case):
        value = case['saved_product']
        extreme = high if value >= 0 else low
        if value == 0:
            return 0.
        return abs(value / extreme) if _sign(case) * extreme > 0 else None

    medium_pool, weak_pool = [], []
    for case in spectrum:
        fraction = relative_magnitude(case)
        if fraction is not None and fraction <= MEDIUM_MAX:
            (weak_pool if fraction <= WEAK_MAX else medium_pool).append(case)
    rng = random.Random(seed)
    documents = Counter(case['document_id'] for kind in tails for case in strata[kind])
    strata['medium'] = _diverse(medium_pool, medium, used, documents, rng)
    used.update(map(_key, strata['medium']))
    strata['weak'] = _diverse(weak_pool, weak, used, documents, rng)

    cases = [{**case, 'kind': kind, 'relative_product_magnitude': relative_magnitude(case)}
             for kind in requested for case in strata[kind]]
    return {'cases': cases, 'requested': requested, 'realized': {kind: len(strata[kind]) for kind in requested},
            'controlled_sweep': sweep,
            'selection': {'tails': tails, 'nonextreme_pool': 'Saved natural spectrum anchors.',
                          'magnitude_scope': magnitude_scope, 'min_product': low, 'max_product': high,
                          'weak_max_fraction': WEAK_MAX, 'medium_max_fraction': MEDIUM_MAX,
                          'seed': seed}}
