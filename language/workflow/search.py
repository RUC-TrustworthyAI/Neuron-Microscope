"""Measure activation-search rounds with bounded feedback and stable checkpoints."""
from . import contracts, prompts
from .measure import batch_comparison, search_regions, summarized
from .state import write


def validate_candidates(output):
    ids = [r['id'] for r in output['regions']]
    if len(set(ids)) != len(ids):
        raise ValueError('Region IDs must be unique')
    for region in output['regions']:
        for candidate in region['candidates']:
            if len(candidate['label'].split()) > 4:
                raise ValueError('Candidate labels must be one word or a phrase of at most four words')


def run(client, measure, initial, packet, search_proposal, config, root, timeline, gpu_span):
    runtime = measure.runtime
    if not (root / 'batch-comparison.json').exists():
        candidates = [c['text'] for r in search_proposal['regions'] for c in r['candidates']]
        pair = sorted(candidates, key=len)
        token_lists = [runtime.tokenizer.encode(s, add_special_tokens=False) for s in (pair[0], pair[-1])]
        with gpu_span('batch-comparison'):
            differences = batch_comparison(runtime, token_lists)
        write(root / 'batch-comparison.json', differences)
    search_rows, region_records, round_summaries = [], {}, []
    for round_index in range(1, config['search_rounds'] + 1):
        timeline.status('activation-search', round=round_index)
        items = []
        for region in search_proposal['regions']:
            for i, candidate in enumerate(region['candidates']):
                identity = f'search-{round_index}-{region["id"]}-{i}'
                candidate['id'] = identity
                items.append({**candidate, 'score': {'location': 'input',
                    'reduction': 'max' if region['pole'] == 'positive' else 'min'}})
        rows = measure.texts(items, f'search-round-{round_index}')
        search_rows.extend(rows)
        for region in search_regions(search_proposal['regions'], rows):
            key = (region['id'], region['pole'])
            if key in region_records:
                region_records[key]['candidates'].extend(region['candidates'])
            else:
                region_records[key] = region
        round_summaries.append({'round': round_index, 'candidates': len(rows),
            'maximum': max(x['product'] for r in rows for x in r['trace']),
            'minimum': min(x['product'] for r in rows for x in r['trace'])})
        write(root / 'search-progress.json', round_summaries)
        if round_index < config['search_rounds']:
            # Feedback is bounded by the requested batch, not a growing tool conversation.
            search_proposal = client.request(f'search/round-{round_index + 1}', prompts.SEARCH,
                {'initial': initial, 'natural_extremes': packet['statistics']['product'],
                 'previous_round': [summarized(r) for r in rows], 'progress': round_summaries,
                 'regions_requested': config['search_regions'],
                 'candidates_per_region': config['candidates_per_region']},
                contracts.SEARCH, 3, check=validate_candidates)
    return search_rows, region_records, round_summaries
