"""Build bounded research packets from saved measurements, without reference explanations."""
import collections
from .state import read


def natural_cases(data, per_pole=10):
    chosen = []
    for key in ('top_documents', 'bottom_documents'):
        chosen.extend(str(x) for x in data['examples'].get(key, [])[:per_pole])
    return list(dict.fromkeys(chosen))


def initial_packet(report):
    data = read(report / 'report.json')
    chosen = natural_cases(data, 8)
    for band in data['bands']:
        chosen.extend(str(x) for x in band['points'][:1])
    samples = []
    for identity in dict.fromkeys(chosen):
        point = data['points'][identity]
        document = data['documents'][str(point['document'])]
        pos = point['position']
        lo, hi = max(0, pos - 32), min(len(document['tokens']), pos + 11)
        samples.append({'id': identity, 'observed_prefix': ''.join(document['tokens'][:pos + 1]),
            'nearby_continuation': ''.join(document['tokens'][pos + 1:hi]),
            'anchor': {'position': pos, 'token': document['tokens'][pos], 'gate': point['values'][0],
                       'up': point['values'][1], 'product': point['values'][2]},
            'local_trace': [{'position': j, 'token': document['tokens'][j], 'product': document['product'][j]}
                            for j in range(lo, hi)]})
    effects = read(report / 'effects.json').get('logit_effect') if (report / 'effects.json').exists() else None
    return {'statistics': data['statistics'], 'samples': samples, 'logit_effect': effects,
            'activation_definition': 'Native product = SiLU(raw gate) * up; observed_prefix ends at the measured anchor.'}


def spectrum_packet(report, sample, contexts):
    """Expose every sampled measurement, storing shared document text once."""
    data = read(report / 'report.json')
    documents, grouped = [], collections.defaultdict(list)
    for point in sample['points']:
        grouped[str(point['document'])].append(point)
    for identity, points in grouped.items():
        tokens = contexts['documents'][identity]['tokens']
        stop = min(len(tokens), max(p['position'] for p in points) + 11)
        offsets = [0]
        for token in tokens[:stop]:
            offsets.append(offsets[-1] + len(token))
        documents.append({'id': identity, 'text': ''.join(tokens[:stop]), 'positions': [
            {'id': p['id'], 'token_position': p['position'], 'token': tokens[p['position']],
             'character_start': offsets[p['position']], 'character_end': offsets[p['position'] + 1],
             'gate': p['gate'], 'up': p['up'], 'activation': p['activation'], 'bin': p['bin']}
            for p in points]})
    effects = read(report / 'effects.json').get('logit_effect') if (report / 'effects.json').exists() else None
    return {'statistics': data['statistics'], 'spectrum': {
                'edges': sample['edges'], 'corpus_hist': sample['corpus_hist'],
                'sample_hist': sample['sample_hist'], 'n_points': len(sample['points'])},
            'documents': documents, 'logit_effect': effects,
            'activation_definition': 'Native product = SiLU(raw gate) * up.',
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
    for point in packet.get('samples', []):
        trace = point['local_trace']
        catalog[point['id']] = {
            'text': ''.join(row['token'] for row in trace),
            'activation': point['anchor']['product'],
            'token_activations': [{'token': row['token'], 'activation': row['product']} for row in trace],
            'peak_index': point['anchor']['position'] - trace[0]['position'],
            'cut_left': trace[0]['position'] > 0, 'cut_right': True}
    return catalog
