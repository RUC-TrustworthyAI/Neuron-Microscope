"""Blind route prediction from saved attribution factors and their existing split."""
from collections import Counter
from pathlib import Path

from . import contracts, prompts
from .state import read, write


def representatives(route_id, training, assignments, weights, limit=3):
    """Prefer distinct documents among the strongest saved members of one route."""
    ranked = sorted((row for row in training if assignments.get(row['id']) == route_id),
                    key=lambda row: (-weights[route_id][row['id']], row['id']))
    selected, remaining, documents = [], [], set()
    for row in ranked:
        document = row.get('source_document')
        document = str(document) if document is not None else row.get('text', row['observed_prefix'])
        if document not in documents:
            selected.append(row)
            documents.add(document)
        else:
            remaining.append(row)
    return (selected + remaining)[:limit]


def not_evaluable(client, folder, routes, training, testing, assignments, reason):
    rows = [{'id': row['id'], 'text': row['observed_prefix'], 'token': row['selected']['token'],
             'position': row['selected']['position'], 'predicted_route': None,
             'actual_route': assignments.get(row['id']), 'correct': None} for row in testing]
    result = {'status': 'not_evaluable', 'reason': reason, 'accuracy': None, 'baseline': None,
              'baseline_route_id': None, 'test_majority_fraction': None,
              'confusion': [[0] * len(routes) for _ in routes],
              'route_ids': [route['id'] for route in routes],
              'route_names': [route.get('name', route['id']) for route in routes], 'descriptions': [],
              'n': 0, 'train_n': len(training), 'test_n': len(testing),
              'unassigned_n': sum(row['actual_route'] is None for row in rows),
              'k': len(routes), 'judge': client.config['model'], 'rows': rows}
    write(folder / 'descriptions.json', [])
    write(folder / 'predictions.json', [])
    write(folder / 'result.json', result)
    return result


def run(client, root, timeline):
    """Reuse saved NMF assignments; never load a target model or recompute attribution."""
    root = Path(root)
    folder = root / 'route-prediction'
    if (folder / 'result.json').is_file():
        return read(folder / 'result.json')
    paths = [root / 'atlas' / name for name in ('atlas-raw.json', 'route-factors.json')]
    if any(not path.is_file() for path in paths):
        raise ValueError('Route prediction requires saved atlas/atlas-raw.json and atlas/route-factors.json')
    atlas, factors = (read(path) for path in paths)
    routes = factors.get('routes', [])
    assignments = factors.get('assignments')
    matrices = factors.get('routes_matrix', {})
    if 'routes' not in factors or assignments is None:
        raise ValueError('Route prediction requires saved NMF routes and assignments')
    if any(row.get('status') != 'complete' or row.get('split') not in ('train', 'test') for row in atlas):
        raise ValueError('Route prediction requires complete saved attribution rows with existing train/test splits')
    route_ids = [route['id'] for route in routes]
    if len(set(route_ids)) != len(route_ids):
        raise ValueError('Saved attribution route IDs must be unique')
    if any(value is not None and value not in route_ids for value in assignments.values()):
        raise ValueError('Saved assignments refer to an unavailable attribution route')
    training = [row for row in atlas if row['split'] == 'train']
    testing = [row for row in atlas if row['split'] == 'test']
    train_counts = Counter(assignments.get(row['id']) for row in training
                           if assignments.get(row['id']) is not None)
    reason = ('The saved NMF has no routes.' if not routes else
              'The saved split has no assigned training cases.' if not train_counts else
              'The saved split has no test cases with an assigned route.'
              if not any(assignments.get(row['id']) is not None for row in testing) else None)
    if reason:
        return not_evaluable(client, folder, routes, training, testing, assignments, reason)
    if not matrices:
        raise ValueError('Route prediction requires complete saved NMF factor matrices')
    weights = {}
    try:
        for matrix in matrices.values():
            for column, route_id in enumerate(matrix['route_ids']):
                weights[route_id] = {identity: matrix['route_weights'][index][column]
                                     for index, identity in enumerate(matrix['graph_ids'])}
        naming_routes = [{'id': route['id'], 'upstream': route['upstream'],
                          'examples': [{'id': row['id'], 'observed_prefix': row['observed_prefix'],
                                        'token': row['selected']['token']}
                                       for row in representatives(route['id'], training, assignments, weights)]}
                         for route in routes]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('Route prediction requires complete saved factors, prefixes and selected tokens') from error

    def check_descriptions(output):
        items = output['routes']
        if len(items) != len(route_ids) or {item['id'] for item in items} != set(route_ids):
            raise ValueError('Return every supplied route ID exactly once')
        if any(not item[key].strip() for item in items for key in ('name', 'description')):
            raise ValueError('Every route needs a nonempty name and description')

    timeline.status('route-descriptions')
    descriptions = client.request('route-descriptions', prompts.ROUTE_DESCRIPTIONS,
                                  {'routes': naming_routes}, contracts.ROUTE_DESCRIPTIONS, 4,
                                  check_descriptions)['routes']
    description_map = {item['id']: item for item in descriptions}
    descriptions = [description_map[identity] for identity in route_ids]
    write(folder / 'descriptions.json', descriptions)

    timeline.status('route-predictions')
    predictions = []
    for start in range(0, len(testing), 25):
        batch = testing[start:start + 25]
        expected = {row['id'] for row in batch}

        def check_predictions(output):
            items = output['predictions']
            if len(items) != len(batch) or {item['id'] for item in items} != expected:
                raise ValueError('Return every supplied sample ID exactly once')
            if any(item['predicted_route'] not in route_ids for item in items):
                raise ValueError('Every prediction must use one of the supplied route IDs')

        packet = {'routes': descriptions,
                  'samples': [{'id': row['id'], 'observed_prefix': row['observed_prefix'],
                               'token': row['selected']['token']} for row in batch]}
        answer = client.request(f'route-predictions/batch-{start // 25 + 1:03d}',
                                prompts.ROUTE_PREDICTIONS, packet, contracts.ROUTE_PREDICTIONS,
                                4, check_predictions)
        predictions.extend(answer['predictions'])
    predicted = {item['id']: item['predicted_route'] for item in predictions}
    predictions = [{'id': row['id'], 'predicted_route': predicted[row['id']]} for row in testing]
    write(folder / 'predictions.json', predictions)

    rows = []
    confusion = [[0] * len(route_ids) for _ in route_ids]
    indices = {identity: index for index, identity in enumerate(route_ids)}
    for row in testing:
        actual, guess = assignments.get(row['id']), predicted[row['id']]
        rows.append({'id': row['id'], 'text': row['observed_prefix'], 'token': row['selected']['token'],
                     'position': row['selected']['position'], 'predicted_route': guess,
                     'actual_route': actual, 'correct': guess == actual if actual is not None else None})
        if actual is not None:
            confusion[indices[actual]][indices[guess]] += 1
    scored = [row for row in rows if row['actual_route'] is not None]
    n = len(scored)
    baseline_id = max(route_ids, key=lambda identity: train_counts[identity])
    test_counts = Counter(row['actual_route'] for row in scored)
    result = {'status': 'complete', 'accuracy': sum(row['correct'] for row in scored) / n if n else None,
              'baseline': test_counts[baseline_id] / n if n else None,
              'baseline_route_id': baseline_id,
              'test_majority_fraction': max(test_counts.values()) / n if n else None,
              'confusion': confusion, 'route_ids': route_ids,
              'route_names': [item['name'] for item in descriptions], 'descriptions': descriptions,
              'n': n, 'train_n': len(training), 'test_n': len(testing),
              'unassigned_n': len(testing) - n, 'k': len(route_ids),
              'judge': client.config['model'], 'rows': rows}
    write(folder / 'result.json', result)
    return result
