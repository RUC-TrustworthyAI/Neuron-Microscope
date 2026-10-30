"""One frozen-explanation prediction set shared by both plots."""
import collections
import concurrent.futures
import copy
import hashlib
import json
from . import contracts, prompts
from .state import read, write

NO_CUE = {'id': 'no-visible-cue', 'name': 'No visible cue', 'predicted_pole': 'uncertain',
          'description': 'The explanation does not clearly predict a directional response in this context.'}


def predict(client, initial, sample, contexts, root, name='labels', stage=2):
    rules = initial['response_rules'] + [NO_CUE]
    explanation = {'claim': initial['claim'], 'response_rules': rules}
    version = hashlib.sha256(json.dumps(explanation, sort_keys=True).encode()).hexdigest()
    output_path = root / 'predictions.json'
    if output_path.exists():
        saved = read(output_path)
        if saved['version'] != version:
            raise ValueError('Frozen explanation changed; use a separate run')
        return saved
    checkpoint = root / 'prediction-checkpoint.json'
    assignments = read(checkpoint) if checkpoint.exists() else {}
    inputs, keys = {}, {}
    for point in sample['points']:
        document = contexts['documents'][str(point['document'])]
        position = point['position']
        visible = {'observed_prefix': ''.join(document['tokens'][:position + 1]),
                   'anchor': {'position': position, 'token': document['tokens'][position]},
                   'nearby_continuation': ''.join(document['tokens'][position + 1:position + 11])}
        key = hashlib.sha256(json.dumps(visible, sort_keys=True).encode()).hexdigest()
        keys[point['id']] = key
        inputs[key] = visible
    pending = sorted(set(inputs) - assignments.keys())
    rule_map = {rule['id']: rule for rule in rules}
    round_index = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        while pending and (not assignments or sum(key not in assignments for key in keys.values()) >= 16):
            round_index += 1
            batches = [pending[i:i + 256] for i in range(0, len(pending), 256)]
            def run_batch(index, batch):
                batch_key = hashlib.sha256(json.dumps(batch).encode()).hexdigest()[:16]
                answer = client.request(f'{name}/batch-{batch_key}', prompts.LABEL,
                    {**explanation, 'points': [{'id': key, **inputs[key]} for key in batch]}, contracts.PREDICTIONS, stage)
                result = {}
                for row in answer['predictions']:
                    if row['id'] not in batch or row['group'] not in rule_map:
                        continue
                    if row['predicted_pole'] != rule_map[row['group']]['predicted_pole']:
                        continue
                    result[row['id']] = row
                return result
            futures = [pool.submit(run_batch, i, batch) for i, batch in enumerate(batches)]
            before = len(assignments)
            for future in concurrent.futures.as_completed(futures):
                assignments.update(future.result())
                write(checkpoint, assignments)
            pending = sorted(set(inputs) - assignments.keys())
            if len(assignments) == before and len(pending) >= 16:
                raise RuntimeError('Prediction round made no progress; preserved completed labels for resume')
    records = []
    for point in sample['points']:
        if keys[point['id']] not in assignments:
            continue
        row = assignments[keys[point['id']]]
        truth = 'positive' if point['activation'] > 0 else 'negative' if point['activation'] < 0 else 'zero'
        records.append({**row, 'id': point['id'], 'activation': point['activation'], 'true_pole': truth,
                        'correct': row['predicted_pole'] == truth,
                        'anchor': inputs[keys[point['id']]]['anchor'], 'observed_prefix': point['observed_prefix'],
                        'nearby_continuation': inputs[keys[point['id']]]['nearby_continuation']})
    confusion = collections.defaultdict(collections.Counter)
    for row in records:
        confusion[row['true_pole']][row['predicted_pole']] += 1
    result = {**explanation, 'version': version, 'records': records,
              'summary': {'requested': len(sample['points']), 'labelled': len(records),
                          'confusion': {key: dict(value) for key, value in confusion.items()},
                          'directional_coverage': sum(r['predicted_pole'] != 'uncertain' for r in records) /
                              max(1, len(records)),
                          'directional_accuracy': sum(r['correct'] for r in records) /
                              max(1, sum(r['predicted_pole'] != 'uncertain' for r in records))}}
    write(output_path, result)
    return result


def plot_fields(sample, predictions, initial_anatomy):
    spectrum = copy.deepcopy(sample)
    records = {row['id']: row for row in predictions['records']}
    spectrum['points'] = [point for point in spectrum['points'] if point['id'] in records]
    counts = collections.Counter()
    quadrants = collections.defaultdict(collections.Counter)
    for band in spectrum['bins']:
        band.update(n_labeled=0, groups={})
    for point in spectrum['points']:
        row = records[point['id']]
        point.update(group=row['group'], cluster=row['group'], semantic_group=row['group'],
                     predicted_pole=row['predicted_pole'], prediction_reason=row['reason'])
        counts[row['group']] += 1
        band = spectrum['bins'][point['bin']]
        band['n_labeled'] += 1
        band['groups'][row['group']] = band['groups'].get(row['group'], 0) + 1
        quadrants[row['group']][('g+' if point['gate'] > 0 else 'g-') + ('u+' if point['up'] > 0 else 'u-')] += 1
    groups = [{**rule, 'pole': rule['predicted_pole'] if rule['predicted_pole'] != 'uncertain' else 'either',
               'n': counts[rule['id']]} for rule in predictions['response_rules'] if counts[rule['id']]]
    spectrum.update(groups=groups, n_assigned=len(spectrum['points']),
                    labeler='deepseek-v4.1-flash-1017', explanation_version=predictions['version'],
                    label_source='frozen-explanation-blind-prediction',
                    sample_hist={'counts': [b['n_labeled'] for b in spectrum['bins']]})
    anatomy = copy.deepcopy(initial_anatomy or {})
    anatomy.update(points=copy.deepcopy(spectrum['points']), n=len(spectrum['points']), clusters=groups,
                   semantic_groups=groups, group_names={g['id']: g['name'] for g in groups},
                   quadrants={key: dict(value) for key, value in quadrants.items()},
                   selection={'corpus_anchors': 0, 'scan_anchors': len(spectrum['points']), 'probe_anchors': 0},
                   colour_mode='predicted_pole', explanation_version=predictions['version'])
    return {'spectrum': spectrum, 'anatomy': anatomy}
