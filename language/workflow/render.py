"""Assemble the existing report template entirely from structured outputs."""
import collections
import copy
from pathlib import Path
from .state import Timeline, read, write, union_seconds
from .labels import plot_fields


def costs(root, neuron, cards):
    timeline = read(root / 'timeline.json')
    steps = [('Prepare saved evidence', 'CPU'), ('Form an initial explanation', 'LLM'),
             ('Design and measure experiments', 'LLM + target model'),
             ('Revise and independently review', 'LLM'), ('Render the report', 'CPU')]
    sources = [(root, None)]
    followup = root / 'followup'
    if (followup / 'status.json').is_file() or (followup / 'api').is_dir():
        steps.append(('Follow-up investigation', 'LLM + target model'))
        sources.append((followup, 6))
        if (followup / 'timeline.json').is_file():
            # Child stages retain their checkpoint format; all contribute to the
            # workflow's follow-up stage only when assembling the cost report.
            timeline += [{**row, 'stage': 6} for row in read(followup / 'timeline.json')]
    rows = [{'step': i, 'name': name, 'resource': resource,
             'detail': '', 'api_usd': 0., 'api_requests': 0, 'gpu_seconds': 0., 'gpu_card_hours': 0.,
             'api_stream_seconds': 0., 'api_stream_time_estimated': False}
            for i, (name, resource) in enumerate(steps, 1)]
    streams, api_intervals = [], []
    stage_streams = collections.defaultdict(list)
    for source, stage_override in sources:
        for path in (source / 'api').glob('**/timing.json'):
            timing = read(path)
            stage = timing['stage'] if stage_override is None else stage_override
            row = rows[stage - 1]
            row['api_requests'] += 1
            row['api_usd'] += read(path.with_name('billing.json'))['estimated_usd']
            if 'start' in timing:
                api_intervals.append((timing['start'], timing['end']))
            if 'first_chunk' in timing:
                interval = (timing['first_chunk'], timing['last_chunk'])
                streams.append(interval)
                stage_streams[stage].append(interval)
    gpu = [(r['start'], r['end']) for r in timeline if r['resource'] == 'gpu']
    for row in rows:
        stage = row['step']
        intervals = [(r['start'], r['end']) for r in timeline if r['resource'] == 'gpu' and r['stage'] == stage]
        row['gpu_seconds'] = union_seconds(intervals)
        row['gpu_card_hours'] = row['gpu_seconds'] * cards / 3600
        row['api_stream_seconds'] = union_seconds(stage_streams[stage])
    execution = [(r['start'], r['end']) for r in timeline if r['resource'] == 'execution']
    execution_seconds = union_seconds(execution)
    gpu_seconds = union_seconds(gpu)
    return {'title': 'Interpretation Costs', 'unit': neuron, 'completed_steps': len(rows), 'steps': rows,
            'total_api_usd': sum(r['api_usd'] for r in rows), 'total_api_requests': sum(r['api_requests'] for r in rows),
            'total_gpu_seconds': gpu_seconds, 'total_gpu_card_hours': gpu_seconds * cards / 3600,
            'total_api_stream_seconds': union_seconds(streams), 'total_api_request_seconds': union_seconds(api_intervals),
            'total_elapsed_seconds': execution_seconds, 'total_execution_seconds': execution_seconds,
            'api_stream_time_estimated': False,
            'timing_basis': 'GPU time records measured operations and model loading. API streaming counts overlapping response streams once. Execution time includes CPU work and API latency; overlapping resources are counted once. Activation collection is excluded.'}


def clusters(raw, definitions, routes):
    from paper_sections import exemplar, aggregate
    from measured_response import describe_group
    complete = {row['id']: row for row in raw if row['status'] == 'complete'}
    result, used = [], set()
    definitions = copy.deepcopy(definitions)
    assigned = {identity for d in definitions for identity in d['graph_ids']}
    if complete.keys() - assigned:
        definitions.append({'id': 'unresolved', 'name': 'Unresolved contexts', 'story': '',
                            'graph_ids': sorted(complete.keys() - assigned), 'example_graph_ids': []})
    route_membership = {identity: route['id'] for route in routes for identity in route.get('graph_ids', [])}
    for definition in definitions:
        ids = [i for i in definition['graph_ids'] if i in complete and i not in used]
        used.update(ids)
        rows = [complete[i] for i in ids]
        if not rows:
            continue
        units = aggregate(rows, 'upstream')[:8]
        norm = sum(abs(u['weight']) for u in units) or 1
        units = [{**u, 'weight': u['weight'] / norm} for u in units]
        examples = [complete[i] for i in definition['example_graph_ids'] if i in ids]
        if not examples:
            examples = sorted(rows, key=lambda r: -abs(r['selected']['product']))[:3]
        result.append(describe_group({
            'id': definition['id'], 'name': definition['name'], 'story': definition['story'],
            'graph_ids': ids, 'n_graphs': len(rows), 'share': len(rows) / len(complete),
            'member_routes': sorted({route_membership[i] for i in ids if i in route_membership}),
            'key_units': units, 'examples': [dict(exemplar(r), graph_id=r['id']) for r in examples[:3]]},
            [r['selected']['product'] for r in rows]))
    return result


def headline_figure(figures, corpus, explanation=None):
    poles = {r['predicted_pole'] for r in explanation['response_rules']} if explanation else {'positive'}
    sign = 1 if 'positive' in poles else -1
    def rank(figure):
        values = [row['activation'] for row in figure['results']]
        return max(sign * value for value in values), max(values) - min(values)
    return max(figures, key=rank)['id']


def assemble_interpretation(root, initial_report, interpretation, sample, predictions):
    """Render the explanation and its two shared-label plots."""
    from report import merge
    source = (initial_report / 'index.html').read_text()
    patch = {'claim': interpretation['claim'], 'agent_read': None, 'agent_read_full': None,
             'section_prose': {}, 'headline': None, 'search': None, 'clusters': [],
             'routes': [], 'custom': None, 'graphs_curated': None, 'family': None,
             'circuit': None, 'tests': None, 'criticism': [], 'interpretation_costs': None}
    patch.update(plot_fields(sample, predictions, None))
    data = initial_report / 'report.json'
    if data.is_file():
        patch.update(corpus_peak_fields(initial_report, read(data)))
    page, bundle = merge(source, patch)
    scope = '''<script id="interpretation-view">
(() => {
  const sections = new Set(['claim', 'spectrum', 'anatomy']);
  document.querySelectorAll('#main > section').forEach(section => {
    if (!sections.has(section.id)) section.remove();
  });
  document.querySelectorAll('#nav a[href^="#"]').forEach(link => {
    if (!sections.has(link.getAttribute('href').slice(1))) link.remove();
  });
})();
</script>'''
    page = page.replace('</body>', scope + '\n</body>')
    output = root / 'report'
    output.mkdir(exist_ok=True)
    (output / 'index.html').write_text(page)
    write(output / 'paper-data.json', {'bundle': bundle})
    return output


def corpus_peak_fields(report, data):
    from ..analysis.peak_census import CENSUS_FILES, load_peak_censuses

    return {CENSUS_FILES[pole][1]: census for pole, census in load_peak_censuses(report, data).items()}


def assemble(root, initial_report, neuron, artifacts, cards):
    from report import merge, source_bundle
    from paper_sections import exemplar, search_patch
    from report_layout import headline
    from saved_display import default_anatomy
    from .anatomy_pairs import select_anatomy_pair
    source = (initial_report / 'index.html').read_text()
    bundle, _ = source_bundle(source)
    initial, revised, review = artifacts['initial'], artifacts['revision'], artifacts['review']
    patch = {'claim': revised['claim'], 'agent_read': review,
             'section_prose': {key: {'summary': text} for key, text in revised['section_prose'].items()},
             'custom': {'figures': artifacts['figures']}, 'logit_effect': artifacts['logit_effect']}
    patch.update(search_patch(artifacts['search_regions'], artifacts['corpus']))
    patch.update(artifacts['atlas_sections'])
    patch.update(plot_fields(artifacts['sample'], artifacts['predictions'], bundle.get('anatomy')))
    data = read(initial_report / 'report.json')
    patch.update(corpus_peak_fields(initial_report, data))
    anatomy = default_anatomy({**bundle, **patch}, data)
    if anatomy is not None:
        patch['anatomy'] = anatomy
    patch['clusters'] = clusters(artifacts['atlas'], revised['clusters'], patch.get('routes', []))
    graph_captions = {g['id']: g for g in revised['graph_captions']}
    patch['graphs_curated'] = {'picks': [dict(g, **{k: v for k, v in graph_captions.get(g['id'], {}).items() if k != 'id'})
                                        for g in artifacts['graphs']]}
    measured = {row['id']: row for row in artifacts['all_measured']}
    evidence = artifacts.get('evidence') or {identity: exemplar(row) for identity, row in measured.items()}
    patch['criticism'] = []
    for i, criticism in enumerate(revised['criticism']):
        patch['criticism'].append({**{k: v for k, v in criticism.items() if k != 'evidence_ids'}, 'round': i + 1,
            'evidence': [evidence[identity] for identity in criticism['evidence_ids']]})
    patch['tests'] = artifacts['tests']
    patch['ablation'] = artifacts.get('ablation')
    # Restore measured token-level output effects independently of narrative choices.
    if not patch['logit_effect']:
        raise ValueError('Missing measured logit effects')
    sweeps = []
    for figure in artifacts['figures']:
        folder = root / 'display-inputs' / figure['id']
        write(folder / 'figure.json', figure)
        identities = {row['measurement_id'] for row in figure['results'] if 'measurement_id' in row}
        rows = [r for r in artifacts['all_measured'] if r['id'] in identities] if identities else [
            r for r in artifacts['all_measured'] if r['id'].startswith(figure['id'] + ':')]
        write(folder / 'conditions.json', [{**r, 'condition': i} for i, r in enumerate(rows)])
        write(folder / 'tool-request.json', {'id': figure['id'], 'template': figure['template']})
        sweeps.append(str(folder))
    preferred = headline_figure(artifacts['figures'], artifacts['corpus'], initial)
    patch['display_selection'] = {'headline_figure_id': preferred}
    pair = select_anatomy_pair(artifacts['figures'], artifacts['all_measured'], preferred_figure_id=preferred)
    if pair is not None:
        patch['anatomy']['pair'] = pair
    patch['headline'] = headline({**bundle, **patch}, initial_report, sweeps)
    patch['interpretation_costs'] = costs(root, neuron, cards)
    page, bundle = merge(source, patch)
    output = root / 'report'
    output.mkdir(exist_ok=True)
    (output / 'index.html').write_text(page)
    write(output / 'paper-data.json', {'bundle': bundle})
    write(output / 'interpretation-costs.json', patch['interpretation_costs'])
    return output


def finalize_report(root, result, config, *, followup):
    """Synthesize the completed evidence before saving the final report."""
    from report import strip_report_synthesis
    from .client import Client
    from .neuron_labels import report_overlay
    from .report_synthesis import synthesize

    followup_root = root / 'followup'
    foundation = followup_root / 'foundation.html'
    current = strip_report_synthesis((root / 'report/index.html').read_text(encoding='utf-8'))
    if followup:
        page = (followup_root / 'report/index.html').read_text(encoding='utf-8')
    else:
        page = foundation.read_text(encoding='utf-8') if foundation.is_file() else current
    page, bundle = report_overlay(page, root)
    bundle['tests'] = {**(bundle.get('tests') or {}), 'route_prediction': result}
    cache = followup_root if followup else root
    client = Client(config['llm'], cache / 'api', config['api_slots'])
    stage = 6 if followup else 4
    timeline = Timeline(root)
    timeline.status('report-synthesis')
    with timeline.span('report-synthesis', 'execution', stage):
        synthesis = synthesize(client, bundle, stage)
    # A completed run may still render from its frozen, pre-census report.
    signed_fields = ('peak_census', 'peak_census_negative')
    census = {key: bundle[key] for key in signed_fields if bundle.get(key)}
    if synthesis is not None and len(census) < len(signed_fields) and config.get('report'):
        initial = Path(config['report']) / 'report.json'
        if initial.is_file():
            recovered = corpus_peak_fields(initial.parent, read(initial))
            missing = {key: recovered[key] for key in signed_fields if key not in census and key in recovered}
            if missing:
                synthesis['peak_census'] = {**census, **missing}
    return update_routes(root, result, config, page=page, synthesis=synthesis)


def update_routes(root, result, config, *, page=None, synthesis=None):
    """Add saved path predictions without rebuilding an investigated report."""
    from report import strip_report_synthesis, update_report_synthesis, update_route_prediction, with_neuron_labels
    from .neuron_labels import report_overlay

    output = root / 'report'
    if page is None:
        page = strip_report_synthesis((output / 'index.html').read_text(encoding='utf-8'))
        page, _ = report_overlay(page, root)
    page, bundle = update_route_prediction(page, result,
        interpretation_costs=costs(root, config['neuron'], len(config['gpus'])))
    if synthesis is not None:
        page, bundle = update_report_synthesis(page, synthesis)
    else:
        bundle = with_neuron_labels(page, bundle)
    write(output / 'paper-data.json', {'bundle': bundle})
    write(output / 'interpretation-costs.json', bundle['interpretation_costs'])
    temporary = output / 'index.html.tmp'
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(output / 'index.html')
    return output


def directory(output, neurons, references):
    from report_directory import comparison_rows, write_directory
    rows = comparison_rows(output, neurons, references=references)
    write_directory(Path(output) / 'index.html', rows)
