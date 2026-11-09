"""Project measured evidence into API packets and the public Criticism rows."""
from .contracts import FOLLOWUP_SCHEMA_VERSION
from .criticism_operations import _finite, _trace_layout


PLAN_FIELDS = ('id', 'source_ids', 'parent_ids', 'claim_id', 'evidence_ids', 'attack', 'prediction')


def fields(row, keys):
    return {key: row[key] for key in keys if key in row}


def excerpt(text, limit=160):
    if len(text) <= limit:
        return text
    return text[:limit // 2] + ' … ' + text[-limit // 2:]


def condition_summary(condition, *, full_text=False):
    text = condition.get('text', '')
    readouts = []
    for row in condition.get('readouts', []):
        selected = row.get('selected') or {}
        summary = {**fields(row, ('metric_id', 'metric', 'value', 'score')),
                   'selected': fields(selected, ('position', 'token'))}
        offset = selected.get('offset')
        if not full_text and len(text) > 240 and offset and offset[0] < len(text) - 120 and offset[1] > 120:
            summary['scored_excerpt'] = text[max(0, offset[0] - 48):offset[1] + 48]
        readouts.append(summary)
    return {**fields(condition, ('id', 'label', 'source', 'case_id')),
            'text': text if full_text else excerpt(text, 240), 'readouts': readouts}


def operation_summary(operation):
    result = {k: v for k, v in operation.items() if k not in ('groups', 'readouts')}
    if 'groups' in operation:
        result['groups'] = [{k: v for k, v in group.items() if k != 'matches'}
                            for group in operation['groups']]
    return result


def history_for_planning(rounds, include_examples=False):
    return [{'round': step['round'], 'saved_inspection': step.get('saved_inspection', {}),
        'investigations': [
        {**fields(item['plan'], PLAN_FIELDS),
         'revision': item['revision'], **item['review'],
         'metrics': item.get('metrics', {}),
         'operations': [operation_summary(o) for o in item.get('operations', [])],
         **({'condition_examples': [condition_summary(c) for c in item['conditions']]}
             if include_examples else {})}
        for item in step['investigations']], 'deferred': step.get('deferred', []),
        'unavailable': step.get('unavailable', [])} for step in rounds]


def archive_inventory(archive):
    traces = []
    for identity, row in archive['traces'].items():
        if not row.get('trace'):
            continue
        trace, _, full = _trace_layout(row)
        traces.append({'id': identity, 'text': excerpt(row['text']),
                       'anchor': fields(row.get('selected') or {}, ('position', 'token', 'product')),
                       'marker_readouts_available': full and all(t.get('offset') is not None for t in trace),
                       **({'trace_complete': False} if not full else {}),
                       **fields(row, ('isolation_available', 'group'))})
    return {'traces': traces,
            'graphs': [{'id': i, 'text': excerpt(row.get('text', '')), 'nodes': len(row.get('nodes', []))}
                       for i, row in archive['graphs'].items()],
            'cohorts': [{'id': i, 'case_count': len(ids), 'examples': ids[:5],
                         'description': archive.get('cohort_descriptions', {}).get(i)}
                        for i, ids in archive['cohorts'].items()],
            'distributions': [{'id': i, 'scope': row.get('scope', i) if isinstance(row, dict) else i}
                              for i, row in archive['distributions'].items()],
            'availability': archive.get('availability', {})}


def evidence_links(ids, catalog):
    return [{'text': catalog[case_id]['text'], 'activation': catalog[case_id]['activation']}
            for case_id in dict.fromkeys(ids)
            if case_id in catalog and catalog[case_id].get('activation') is not None]


def _has_text_span(condition, readout):
    text = condition.get('text')
    offset = (readout.get('selected') or {}).get('offset')
    return (isinstance(text, str) and isinstance(offset, (list, tuple)) and len(offset) == 2
            and all(type(i) is int for i in offset) and 0 <= offset[0] < offset[1] <= len(text))


def publishable_product_readouts(conditions):
    """Link only finite products that can be displayed through their measured token."""
    return {row['metric_id']: (condition, row) for condition in conditions
            for row in condition.get('readouts', [])
            if row.get('metric') == 'product' and _finite(row.get('value')) and _has_text_span(condition, row)}


def _readout_evidence(condition, readout):
    text = condition['text'][:readout['selected']['offset'][1]]
    if len(text) > 300:
        text = text[:140].rstrip() + ' … ' + text[-140:].lstrip()
    return {'text': text, 'activation': readout['value']}


def publication_rows(rounds):
    """Keep one question, prediction, result and representative evidence per test."""
    rows = []
    for step in rounds:
        for item in step.get('investigations', []):
            review = item.get('review')
            if not review:
                continue
            plan = item['plan']
            candidates = publishable_product_readouts(item.get('conditions', []))
            selected = review['evidence_metrics']
            unknown = set(selected) - candidates.keys()
            if unknown:
                raise ValueError(f'Published evidence references unavailable product metrics: {sorted(unknown)}')
            rows.append({'round': step['round'], 'attack': plan['attack'],
                         'prediction': plan['prediction'], 'result': review['result'],
                         'verdict': review['verdict'],
                         'evidence': [_readout_evidence(*candidates[metric]) for metric in selected],
                         'suggested_revision': review.get('suggested_revision') or None})
    return rows


def criticism_patch(criticism, followups, evidence):
    """The reader sees measured Criticism, while workflow logs remain on disk."""
    followups = followups or {}
    enabled = followups.get('status', 'disabled') != 'disabled'
    version = followups.get('schema_version')
    if enabled and (type(version) is not int or version != FOLLOWUP_SCHEMA_VERSION):
        raise ValueError(f'Unsupported followup report schema: {version}; expected schema {FOLLOWUP_SCHEMA_VERSION}')
    status = None
    rows = []
    if enabled:
        rows = publication_rows(followups.get('rounds', []))
        status = {key: followups[key] for key in ('status', 'stop_reason', 'source_count', 'schema_version')
                  if key in followups}
        if not rows:
            status['showing_baseline'] = True
    if not rows:
        rows = [{'round': item.get('round', 1), 'attack': item['attack'],
                 'prediction': item['prediction'], 'result': item['result'],
                 'verdict': item['verdict'], 'evidence': evidence_links(item['evidence_ids'], evidence),
                 'suggested_revision': item.get('suggested_revision')}
                for item in criticism]
    return {'criticism': rows, 'followup_status': status}
