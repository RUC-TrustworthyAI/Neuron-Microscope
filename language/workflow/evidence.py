"""Build bounded research packets from saved measurements, without reference explanations."""
import collections
from .state import read


def natural_cases(data, per_pole=10):
    chosen = []
    for key in ('top_documents', 'bottom_documents'):
        chosen.extend(str(x) for x in data['examples'].get(key, [])[:per_pole])
    return list(dict.fromkeys(chosen))


def initial_packet(report, data=None):
    data = data if data is not None else read(report / 'report.json')
    chosen, memberships = [], collections.defaultdict(list)
    for group in ('top_documents', 'bottom_documents', 'top_tokens', 'bottom_tokens'):
        for identity in data['examples'].get(group, [])[:10]:
            chosen.append(str(identity))
            memberships[str(identity)].append(group)
    for band in data['bands']:
        chosen.extend(str(x) for x in band['points'][:1])
        for identity in band['points'][:1]:
            memberships[str(identity)].append(f"p{band['low']}-p{band['high']}")
    samples = []
    for identity in dict.fromkeys(chosen):
        point = data['points'][identity]
        document = data['documents'][str(point['document'])]
        pos = point['position']
        lo, hi = max(0, pos - 32), min(len(document['tokens']), pos + 11)
        samples.append({'id': identity, 'document': str(point['document']),
            'groups': memberships[identity], 'observed_prefix': ''.join(document['tokens'][:pos + 1]),
            'nearby_continuation': ''.join(document['tokens'][pos + 1:hi]),
            'anchor': {'position': pos, 'token': document['tokens'][pos], 'product': point['values'][2]},
            'local_trace': [{'position': j, 'token': document['tokens'][j], 'product': document['product'][j]}
                            for j in range(lo, hi)]})
    effects = read(report / 'effects.json').get('logit_effect') if (report / 'effects.json').exists() else None
    tables = []
    if (report / 'token-rankings.json').exists():
        for table in read(report / 'token-rankings.json'):
            view = table['view']
            rows = [{'token_id': row[0], 'token': row[1], 'occurrences': row[2],
                     'mean_product': row[3], 'hits': row[view['column']],
                     'hit_fraction': row[view['column']] / row[2]} for row in table['rows'][:20]]
            tables.append({'pole': view['mode'], 'extreme_fraction': view['fraction'],
                           'threshold': view['threshold'], 'minimum_occurrences': table['minimum_count'],
                           'comparison': '>' if view['mode'] == 'positive' else '<', 'rows': rows})
    census = {pole: read(report / filename) for pole, filename in
              [('positive', 'peak-census.json'), ('negative', 'peak-census-negative.json')]
              if (report / filename).exists()}
    return {'statistics': {key: value for key, value in data['statistics'].items() if key == 'product'},
            'samples': samples, 'logit_effect': effects,
            'quantile_bands': [{**{k: v for k, v in band.items() if k != 'points'},
                                'case_ids': [str(x) for x in band['points'][:1]]} for band in data['bands']],
            'token_selectivity': {'definition': 'Thresholds are fractions of the signed corpus extreme, not percentiles. '
                                  'hit_fraction is hits / occurrences of that token type. mean_product averages all its occurrences.',
                                  'tables': tables},
            'peak_census': census,
            'activation_definition': 'Product is the measured signed activation; observed_prefix ends at the measured anchor.'}


def spectrum_packet(report, sample, contexts):
    """Expose every sampled measurement, storing shared document text once."""
    data = read(report / 'report.json')
    evidence = initial_packet(report, data)
    documents, grouped = [], collections.defaultdict(list)
    for point in sample['points']:
        grouped[str(point['document'])].append(point)
    extra = collections.defaultdict(list)
    for case in evidence['samples']:
        extra[case['document']].append(case)
        grouped.setdefault(case['document'], [])
    for identity, points in grouped.items():
        document = contexts['documents'].get(identity, data['documents'].get(identity))
        tokens = document['tokens']
        positions = [p['position'] for p in points] + [p['anchor']['position'] for p in extra[identity]]
        stop = min(len(tokens), max(positions) + 11)
        offsets = [0]
        for token in tokens[:stop]:
            offsets.append(offsets[-1] + len(token))
        for case in extra[identity]:
            pos = case['anchor']['position']
            case.update(character_start=offsets[pos], character_end=offsets[pos + 1])
            case.pop('observed_prefix')
            case.pop('nearby_continuation')
        documents.append({'id': identity, 'text': ''.join(tokens[:stop]), 'positions': [
            {'id': p['id'], 'token_position': p['position'], 'token': tokens[p['position']],
             'character_start': offsets[p['position']], 'character_end': offsets[p['position'] + 1],
             'activation': p['activation'], 'bin': p['bin']}
            for p in points]})
    return {**evidence, 'spectrum': {
                'edges': sample['edges'], 'corpus_hist': sample['corpus_hist'],
                'sample_hist': sample['sample_hist'], 'n_points': len(sample['points'])},
            'documents': documents,
            'activation_definition': 'Product and activation denote the measured signed response.',
            'context_definition': 'Character offsets index the document text. At each marked position, the causal prefix ends at character_end. Later text provides linguistic context only.'}


def prediction_feedback(predictions):
    records = predictions['records']
    groups = []
    for rule in predictions['response_rules']:
        rows = [r for r in records if r['group'] == rule['id']]
        examples = []
        for pole in ('positive', 'negative', 'zero'):
            examples.extend(sorted((r for r in rows if r['true_pole'] == pole),
                                   key=lambda r: -abs(r['activation']))[:3])
        groups.append({'rule': rule, 'count': len(rows),
                       'signs': dict(collections.Counter(r['true_pole'] for r in rows)),
                       'examples': examples})
    return {'summary': predictions['summary'], 'groups': groups,
            'evaluation': 'Feedback on predictions from this frozen explanation version; not a held-out test of a revision.'}


def refinement_feedback(predictions, statistics):
    """Review sign reversals and strong unexplained responses against corpus extrema."""
    feedback = prediction_feedback(predictions)
    for group in feedback['groups']:
        group.pop('examples')
    feedback.update(wrong_direction=[], strong_no_visible_cue=[])
    lower, upper = statistics['min'], statistics['max']
    for row in predictions['records']:
        predicted, truth, value = row['predicted_pole'], row['true_pole'], row['activation']
        if predicted in ('positive', 'negative') and truth in ('positive', 'negative') and predicted != truth:
            cases = feedback['wrong_direction']
        elif row['group'] == 'no-visible-cue' and (
                upper > 0 and value > .5 * upper or lower < 0 and value < .5 * lower):
            cases = feedback['strong_no_visible_cue']
        else:
            continue
        cases.append({key: row[key] for key in
                      ('id', 'group', 'predicted_pole', 'reason', 'activation', 'true_pole')})
    return feedback


def evidence_catalog(measured, atlas, graphs, sample, packet):
    """Resolve references from every measured evidence source supplied to revision."""
    from paper_sections import exemplar
    catalog = {row['id']: exemplar(row) for row in [*measured, *atlas]}
    for result in graphs:
        position = result['graph']['position']
        catalog[result['graph']['id']] = exemplar({
            'text': result['graph']['text'], 'trace': result['trace'],
            'selected': result['trace'][position]})
    for point in sample['points']:
        catalog[point['id']] = {key: point[key] for key in
            ('text', 'activation', 'token_activations', 'peak_index', 'cut_left', 'cut_right')}
        if point['id'].startswith('spectrum-'):
            catalog.setdefault(point['id'][len('spectrum-'):], catalog[point['id']])
    for point in packet.get('samples', []):
        trace = point['local_trace']
        catalog[point['id']] = {
            'text': ''.join(row['token'] for row in trace),
            'activation': point['anchor']['product'],
            'token_activations': [{'token': row['token'], 'activation': row['product']} for row in trace],
            'peak_index': point['anchor']['position'] - trace[0]['position'],
            'cut_left': trace[0]['position'] > 0, 'cut_right': True}
        if point['id'].isdigit():
            catalog.setdefault('spectrum-' + point['id'], catalog[point['id']])
    return catalog
