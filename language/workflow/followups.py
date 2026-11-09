"""Run the public Criticism cycle over measured evidence and versioned working claims."""
from pathlib import Path

from . import contracts, prompts
from .criticism_operations import _digest as fingerprint
from .criticism_operations import execute, operation_fields, prepare
from .criticism_report import (PLAN_FIELDS, archive_inventory, condition_summary, excerpt, fields, history_for_planning,
                              operation_summary, publishable_product_readouts)
from .measure import Measurements
from .state import now, read, write


SCHEMA_VERSION = contracts.FOLLOWUP_SCHEMA_VERSION
DEFAULTS = {'max_rounds': 2, 'max_experiments': 6, 'max_conditions': 36, 'max_input_tokens': 1024}
TEXT = {'type': 'string', 'minLength': 1}
IDS = contracts.arr(TEXT, 0, 64)
REVIEW = contracts.obj(verdict=contracts.string('survived', 'narrowed', 'rejected', 'inconclusive'),
                       result=TEXT, suggested_revision=contracts.S,
                       next_question=contracts.S, evidence_metrics=contracts.arr(TEXT, 0, 7))
INSPECTION_OPERATIONS = ('probes', 'graph_queries', 'quantiles')


def settings(options):
    if not isinstance(options, dict):
        raise ValueError('followups must be an object')
    result = {**DEFAULTS, **options}
    contracts.validate(result, contracts.obj(**{
        name: {'type': 'integer', 'minimum': low, 'maximum': high}
        for name, low, high in [('max_rounds', 0, 2), ('max_experiments', 1, 6),
                               ('max_conditions', 2, 36), ('max_input_tokens', 8, 4096)]}), 'followups')
    return result


def check_run_directory(root):
    path = Path(root) / 'followups/context.json'
    if path.exists():
        version = read(path).get('schema_version')
        if type(version) is not int or version != SCHEMA_VERSION:
            raise ValueError('Followup schema mismatch; choose a new run directory')


def criticism_sources(criticism):
    return [{'id': f'criticism-{index}', **fields(entry,
             ('attack', 'prediction', 'result', 'verdict', 'evidence_ids'))}
            for index, entry in enumerate(criticism, 1)]


def round_schema(options, round_id=1):
    investigation = contracts.obj(id=TEXT, source_ids=IDS, parent_ids=IDS, claim_id=TEXT, evidence_ids=IDS,
        attack=TEXT, prediction=TEXT, **operation_fields(options))
    return contracts.obj(action=contracts.string('experiment', 'stop'), reason=TEXT,
        investigations=contracts.arr(investigation, 0,
                                    min(options['max_experiments'], 6 if round_id == 1 else 5)),
        deferred=contracts.arr(contracts.obj(source_id=TEXT, reason=TEXT)))


def inspect_saved(request, packet, archive, tokenizer, step):
    """Read saved evidence before the scientist commits to new input designs."""
    options, round_id = packet['limits'], packet['round']

    def register(selection):
        if any(selection[key] for key in INSPECTION_OPERATIONS):
            return prepare(selection, tokenizer, archive, options, round_id, 0)

    schema = contracts.obj(reason=TEXT,
        **fields(operation_fields(options, archive_only=True), INSPECTION_OPERATIONS))
    selection = request('inspect', prompts.FOLLOWUP_INSPECT, packet, schema, register)
    registration = register(selection)
    observed = execute(registration, None, archive) if registration else {
        'metrics': {}, 'conditions': [], 'operations': [], 'measurements': []}
    write(step / 'inspection.json', {'selection': selection, 'registration': registration, 'observed': observed})
    return {'reason': selection['reason'], 'metrics': observed['metrics'],
            'conditions': [condition_summary(c) for c in observed['conditions']],
            'operations': [operation_summary(o) for o in observed['operations']]}


def check_round(plan, sources, parents, claims, evidence_ids, round_id, used_ids):
    ids = [item['id'] for item in plan['investigations']]
    if len(set(ids)) != len(ids) or set(ids) & used_ids:
        raise ValueError('Investigation IDs must be unique across rounds')
    for item in plan['investigations']:
        for key, known in (('source_ids', sources), ('parent_ids', parents), ('evidence_ids', evidence_ids)):
            if set(item[key]) - known:
                raise ValueError(f'Investigation has unknown {key}')
        if item['claim_id'] not in claims:
            raise ValueError('Investigation must test a registered working claim')
        if not item['source_ids'] and (round_id == 1 or not (item['parent_ids'] or item['evidence_ids'])):
            raise ValueError('Cite upstream Criticism, or a prior investigation or observation in later rounds')
    deferred = [item['source_id'] for item in plan['deferred']]
    if set(deferred) - sources:
        raise ValueError('Deferrals must cite known upstream Criticism')
    if (plan['action'] == 'experiment') != bool(ids):
        raise ValueError('Only an experiment can contain investigations')


def design_context(plan, packet):
    sources = [s for s in packet['sources'] if s['id'] in plan['source_ids']]
    cited = dict.fromkeys(plan['evidence_ids'] + [i for s in sources for i in s['evidence_ids']])
    claim = next(c for c in packet['working_claims'] if c['id'] == plan['claim_id'])
    parents = set(plan['parent_ids']) | {claim['investigation_id']}
    return {'round': packet['round'], 'sources': sources,
            'working_claim': claim,
            'parent_results': [fields(item, ('id', 'result', 'verdict', 'next_question'))
                               for step in packet['previous_rounds'] for item in step['investigations']
                               if item['id'] in parents],
            'cited_evidence': fields(packet['cited_evidence'], cited)}


def register_round(request, packet, check_plan, tokenizer, archive, step):
    """Save exact executable operations before observing any new measurements."""
    path = step / 'registered.json'
    options, round_id = packet['limits'], packet['round']
    plan = request('plan', prompts.FOLLOWUP_PLAN, packet, round_schema(options, round_id), check_plan)
    designs = []
    for index, item in enumerate(plan['investigations'], 1):
        design = {'plan': item, 'operations': None}
        try:
            design['operations'] = prepare(item, tokenizer, archive, options, round_id, index)
        except (KeyError, TypeError, ValueError) as error:
            design['error'] = str(error)
        designs.append(design)
    registered = {'plan': plan, 'designs': designs}
    if path.exists():
        if read(path)['registration'] != registered:
            raise ValueError('Registered operations changed; choose a new run directory')
    else:
        write(path, {'at': now(), 'registration': registered})
    return registered


def review_one(request, entry, context, inspection, index):
    eligible = publishable_product_readouts(entry['conditions'])

    def check(review):
        selected = review['evidence_metrics']
        if len(set(selected)) != len(selected) or set(selected) - eligible.keys():
            raise ValueError('Copy distinct IDs from eligible_product_metrics; use [] if none apply')

    return request(f'review-{index}', prompts.FOLLOWUP_REVIEW,
        {**context, 'plan': fields(entry['plan'], PLAN_FIELDS),
         'saved_inspection': inspection,
         'conditions': [condition_summary(c, full_text=True) for c in entry['conditions']],
         'operations': [operation_summary(o) for o in entry['operations']],
         'metrics': entry['metrics'],
         'eligible_product_metrics': list(eligible)}, REVIEW, check)


def run(client, measure, initial, predictions, measured, root, timeline, options, identity,
        corpus, *, criticism, evidence, archive=None):
    options = settings(options)
    if options['max_rounds'] == 0:
        return {'status': 'disabled', 'stop_reason': 'max_rounds is zero', 'rounds': []}, []
    sources = criticism_sources(criticism)
    source_ids = {item['id'] for item in sources}
    archive = {'traces': {}, 'graphs': {}, 'cohorts': {}, 'distributions': {}, **(archive or {})}
    archive['traces'] = {**{r['id']: r for r in measured if 'trace' in r}, **archive['traces']}
    cited = {i: evidence[i] for s in sources for i in s['evidence_ids']}
    context = {'schema_version': SCHEMA_VERSION, 'identity': identity, 'options': options,
               'explanation': initial, 'explanation_version': predictions['version'],
               'sources': sources, 'evidence': cited, 'corpus': corpus, 'archive_digest': fingerprint(archive)}
    folder = Path(root) / 'followups'
    path = folder / 'context.json'
    if path.exists() and read(path) != context:
        raise ValueError('Criticism or the archived evidence changed; choose a new run directory')
    write(path, context)
    initial_claim = initial['claim']
    claim_text = ('\n'.join(f'{key}: {value}' for key, value in initial_claim.items())
                  if isinstance(initial_claim, dict) else initial_claim)
    claim = {'id': 'claim-0', 'parent_id': None, 'investigation_id': None,
             'text': claim_text}
    claims, completed, used_ids = {claim['id']: claim}, set(), set()
    result = {'schema_version': SCHEMA_VERSION, 'status': 'running', 'stop_reason': '',
              'source_count': len(sources), 'working_claims': [claim], 'rounds': []}
    if not sources:
        result.update(status='no_criticism', stop_reason='The upstream analysis returned no Criticism entries.')
        write(folder / 'summary.json', result)
        return result, []
    tokenizer = measure.runtime.tokenizer
    measure = Measurements(measure.runtime, folder / 'measurements', timeline, 1, options['max_input_tokens'])
    new_rows = []
    cited_summary = {i: {**fields(row, ('activation',)), 'text': excerpt(row.get('text', ''), 240)}
                     for i, row in cited.items()}
    for round_id in range(1, options['max_rounds'] + 1):
        step = folder / f'round-{round_id}'
        evidence_ids = set(evidence) | archive['traces'].keys() | archive['graphs'].keys()
        packet = {'sources': sources, 'cited_evidence': cited_summary,
                  'baseline_rules': initial.get('response_rules', []),
                  'previous_rounds': history_for_planning(result['rounds'], include_examples=True),
                  'working_claims': list(claims.values()), 'archive': archive_inventory(archive),
                  'corpus_extrema': corpus, 'round': round_id,
                  'limits': {**options, 'max_experiments': min(options['max_experiments'],
                                                            6 if round_id == 1 else 5)}}
        def request(name, prompt, data, schema, check):
            return client.request(f'followups/round-{round_id}/{name}', prompt, data, schema, 3, check=check)

        def check_plan(plan):
            check_round(plan, source_ids, completed, claims, evidence_ids, round_id, used_ids)

        timeline.status('followup-inspection', round=round_id)
        packet['saved_inspection'] = inspect_saved(request, packet, archive, tokenizer, step)
        timeline.status('followup-planning', round=round_id)
        registration = register_round(request, packet, check_plan, tokenizer, archive, step)
        round_plan = registration['plan']
        result['deferred'] = round_plan['deferred']
        if round_plan['action'] == 'stop':
            result.update(status='stopped', stop_reason=round_plan['reason'])
            break
        designs = registration['designs']
        unavailable = [{'id': d['plan']['id'], 'reason': d['error']} for d in designs if 'error' in d]
        executable = [(i, d) for i, d in enumerate(designs, 1) if d['operations'] is not None]
        used_ids.update(d['plan']['id'] for d in designs)
        if not executable:
            result.update(status='no_executable_designs', stop_reason='No executable investigation was supplied.',
                          unavailable=unavailable)
            break
        record = {'round': round_id, 'status': 'measuring', 'deferred': round_plan['deferred'],
                  'saved_inspection': packet['saved_inspection'],
                  'unavailable': unavailable, 'investigations': []}
        write(step / 'result.json', record)
        first_row = len(new_rows)
        for index, design in executable:
            plan = design['plan']
            timeline.status('followup-measurement', round=round_id, investigation=index)
            observed = execute(design['operations'], measure, archive)
            entry = {'plan': plan, 'conditions': observed['conditions'],
                     'operations': observed['operations'], 'metrics': observed['metrics']}
            timeline.status('followup-analysis', round=round_id, investigation=index)
            review = review_one(request, entry, design_context(plan, packet), packet['saved_inspection'], index)
            revision = None
            if review['suggested_revision'].strip():
                revision = {'id': 'claim-' + plan['id'], 'parent_id': plan['claim_id'],
                            'investigation_id': plan['id'], 'text': review['suggested_revision']}
                claims[revision['id']] = revision
            entry.update(review=review, revision=revision)
            record['investigations'].append(entry)
            completed.add(plan['id'])
            new_rows.extend(observed['measurements'])
            write(step / 'result.json', record)
        archive['traces'].update({row['id']: row for row in new_rows[first_row:]})
        result['working_claims'] = list(claims.values())
        record['status'] = 'complete'
        result['rounds'].append(record)
        write(step / 'result.json', record)
        write(folder / 'summary.json', result)
    else:
        result.update(status='budget_exhausted', stop_reason='Round budget reached.')
    write(folder / 'summary.json', result)
    return result, new_rows
