"""Plan and execute follow-up rounds while retaining all previous outcomes."""
import re

from . import followup_protocol as protocol, prompts
from .state import read, write


def evidence_packet(evidence, *, full_context=False):
    """Keep every case and measured outcome; full token traces remain in the archive."""
    def compact(row):
        if 'data' in row:
            return {'id': row['id'], 'kind': row['kind'], 'source': 'saved baseline evidence; see cited original'}
        if 'trace' not in row:
            return {key: row[key] for key in ('id', 'activation') if key in row} | {'input_excerpt': row['text'][:512]}
        values = [row['trace'][p]['product'] for p in row['scored_positions']]
        ranges = []
        for position in row['scored_positions']:
            if ranges and position == ranges[-1][1] + 1:
                ranges[-1][1] = position
            else:
                ranges.append([position, position])
        anchor_end = row['selected'].get('offset', [0, len(row['text'])])[1]
        start, end = max(0, anchor_end - 384), min(len(row['text']), anchor_end + 128)
        fields = ('id', 'label', 'selected', 'score', 'status', 'frame', 'condition',
                  'delta_from_reference', 'document_id')
        return {**{key: row[key] for key in fields if key in row},
                **({'text': row['text'], 'causal_prefix': row.get('causal_prefix', row['text'][:anchor_end])}
                   if full_context else {}),
                'input_excerpt': row['text'][start:end], 'excerpt_char_span': [start, end],
                'full_input_chars': len(row['text']), 'causal_end_char': anchor_end,
                'scored_position_ranges': ranges,
                'scored_summary': {'n': len(values), 'min': min(values), 'max': max(values),
                                   'mean': sum(values) / len(values)}}
    return {**{key: value for key, value in evidence.items() if key != 'rows'},
            'rows': [compact(row) for row in evidence['rows']]}


def investigate(client, executor, question, identity, folder, catalog, context, options):
    record_path = folder / 'record.json'
    if record_path.exists():
        record = read(record_path)
        if record['question'] != question:
            raise ValueError('Completed investigation question changed; use a new directory')
        return record
    record = {'id': identity, 'question': question}
    try:
        action = question['action']
        if action == 'review':
            spec = {'evidence_ids': question['evidence_ids']}
        else:
            spec = client.request(f'{identity}/specification', prompts.FOLLOWUP_SPECIFICATION,
                {**context, 'question': question, 'max_tokens': options['max_tokens'],
                 'referenced_evidence': [catalog[key] for key in question['evidence_ids']]},
                protocol.SPECIFICATIONS[action], 1,
                check=lambda output: executor.validate_spec(action, output, identity))
        record['specification'] = spec
        write(folder / 'preregistered.json', record)
        evidence_path = folder / 'evidence.json'
        if evidence_path.exists():
            evidence = read(evidence_path)
        else:
            evidence = ({'kind': 'review', 'rows': [catalog[key] for key in spec['evidence_ids']]}
                        if action == 'review' else executor.execute(action, spec, identity))
            write(evidence_path, evidence)
        record['evidence'] = evidence_packet(evidence)
        record['evidence_file'] = f'{identity}/evidence.json'
        valid_ids = {row['id'] for row in evidence['rows']}

        def check_assessment(output):
            if not valid_ids and (output['evidence_ids'] or output['verdict'] != 'inconclusive'):
                raise ValueError('No observations: cite no evidence and return inconclusive')
            if any(group['selected_documents'] == 0 for group in evidence.get('groups', [])) and output['verdict'] == 'survived':
                raise ValueError('An empty comparison group cannot establish survival')
            if valid_ids and not output['evidence_ids']:
                raise ValueError('Cite the evidence used for this assessment')

        record['assessment'] = client.request(f'{identity}/assessment', prompts.FOLLOWUP_ASSESSMENT,
            {**context, 'question': question, 'specification': spec,
             'evidence': evidence if action == 'review' else evidence_packet(evidence, full_context=True)},
            protocol.assessment_schema(sorted(valid_ids)), 2, check=check_assessment)
        record['status'] = 'complete'
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    write(record_path, record)
    return record


def rounds(client, executor, base, root, options, timeline):
    history, seen = [], set()
    working = {'frozen_interpretation': base['initial'], 'proposed_revisions': []}
    criticisms, catalog = base['original_criticism'], base['catalog']
    criticism_ids = [item['id'] for item in criticisms]
    stop_reason = 'round_limit'
    for number in range(1, protocol.ROUNDS + 1):
        timeline.status('planning', round=number)
        previews = [{'id': key, 'kind': row.get('kind', 'measured_case'),
                     'preview': row.get('text', '')[:160], 'activation': row.get('activation')}
                    for key, row in catalog.items()]
        context = {'original_criticism': criticisms, 'frozen_interpretation': base['initial'],
                   'previous_rounds': history, 'working_interpretation': working}
        plan = client.request(f'round-{number}/plan', prompts.FOLLOWUP_PLAN,
            {**context, 'round': number, 'maximum_investigations': protocol.MAX_INVESTIGATIONS,
             'evidence_catalog': previews}, protocol.plan_schema(criticism_ids, list(catalog)), 1,
            check=lambda output: protocol.check_plan(output, catalog))
        current = {'round': number, 'plan': plan, 'investigations': [], 'skipped_duplicates': []}
        for index, question in enumerate(plan['investigations'], 1):
            key = re.sub(r'\W+', ' ', question['question'].casefold()).strip()
            if key in seen:
                current['skipped_duplicates'].append(question)
                continue
            seen.add(key)
            identity = f'r{number}-i{index}'
            timeline.status('investigating', round=number, investigation=identity)
            record = investigate(client, executor, question, identity, root / identity,
                                 catalog, context, options)
            current['investigations'].append(record)
        history.append(current)
        for record in current['investigations']:
            assessment = record.get('assessment', {})
            if assessment.get('suggested_revision'):
                working['proposed_revisions'].append({'id': record['id'],
                    'verdict': assessment['verdict'], 'revision': assessment['suggested_revision']})
            path = root / record['id'] / 'evidence.json'
            if path.exists():
                catalog.update((row['id'], row) for row in read(path)['rows'])
        write(root / f'round-{number}.json', current)
        if not current['investigations']:
            stop_reason = 'no_new_distinguishable_questions'
            break
        if all(record['status'] == 'failed' for record in current['investigations']):
            stop_reason = 'round_failed'
            break
    return {'original_criticism': criticisms, 'rounds': history,
            'working_interpretation': working, 'stop_reason': stop_reason}
