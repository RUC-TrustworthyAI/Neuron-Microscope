"""Assemble the existing report template entirely from structured outputs."""
import collections
import copy
from pathlib import Path
from .state import read, write, union_seconds
from .labels import plot_fields


def costs(root, neuron, cards):
    timeline = read(root / 'timeline.json')
    names = ['Prepare saved evidence', 'Explain, refine and freeze', 'Design and measure experiments',
             'Analyze results and independently review', 'Render the report']
    rows = [{'step': i + 1, 'name': name, 'resource': ['CPU', 'LLM', 'LLM + target model', 'LLM', 'CPU'][i],
             'detail': '', 'api_usd': 0., 'api_requests': 0, 'gpu_seconds': 0., 'gpu_card_hours': 0.,
             'api_stream_seconds': 0., 'api_stream_time_estimated': False} for i, name in enumerate(names)]
    streams, api_intervals = [], []
    stage_streams = collections.defaultdict(list)
    for path in (root / 'api').glob('**/timing.json'):
        timing = read(path)
        row = rows[timing['stage'] - 1]
        row['api_requests'] += 1
        row['api_usd'] += read(path.with_name('billing.json'))['estimated_usd']
        if 'start' in timing:
            api_intervals.append((timing['start'], timing['end']))
        if 'first_chunk' in timing:
            interval = (timing['first_chunk'], timing['last_chunk'])
            streams.append(interval)
            stage_streams[timing['stage']].append(interval)
    gpu = [(r['start'], r['end']) for r in timeline if r['resource'] == 'gpu']
    for i, row in enumerate(rows):
        intervals = [(r['start'], r['end']) for r in timeline if r['resource'] == 'gpu' and r['stage'] == i + 1]
        row['gpu_seconds'] = union_seconds(intervals)
        row['gpu_card_hours'] = row['gpu_seconds'] * cards / 3600
        row['api_stream_seconds'] = union_seconds(stage_streams[i + 1])
    execution = [(r['start'], r['end']) for r in timeline if r['resource'] == 'execution']
    execution_seconds = union_seconds(execution)
    return {'title': 'Interpretation Costs', 'unit': neuron, 'completed_steps': 5, 'steps': rows,
            'total_api_usd': sum(r['api_usd'] for r in rows), 'total_api_requests': sum(r['api_requests'] for r in rows),
            'total_gpu_seconds': union_seconds(gpu), 'total_gpu_card_hours': union_seconds(gpu) * cards / 3600,
            'total_api_stream_seconds': union_seconds(streams), 'total_api_request_seconds': union_seconds(api_intervals),
            'total_elapsed_seconds': execution_seconds, 'total_execution_seconds': execution_seconds,
            'api_stream_time_estimated': False,
            'timing_basis': 'GPU time records measured operations and model loading. API streaming counts overlapping response streams once. Execution time includes CPU work, API latency and GPU-lock waits; overlapping resources are counted once. Activation collection is excluded.'}


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


def headline_figure(figures, corpus):
    def rank(figure):
        values = [max((row.get('max', row['activation']), row.get('min', row['activation'])), key=abs)
                  for row in figure['results']]
        positive = max(0., max(values)) / (abs(corpus['max']) or 1.)
        negative = max(0., -min(values)) / (abs(corpus['min']) or 1.)
        return min(positive, negative), positive + negative
    return max(figures, key=rank)['id']


def assemble_interpretation(root, initial_report, interpretation, sample, predictions):
    """Render the explanation and its two shared-label plots."""
    from report import merge, source_bundle
    source = (initial_report / 'index.html').read_text()
    bundle, _ = source_bundle(source)
    patch = {'claim': interpretation['claim'], 'agent_read': None, 'agent_read_full': None,
             'section_prose': {}, 'headline': None, 'search': None, 'clusters': [],
             'routes': [], 'custom': None, 'graphs_curated': None, 'family': None,
             'circuit': None, 'tests': None, 'criticism': [], 'interpretation_costs': None}
    patch.update(plot_fields(sample, predictions, None))
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


def assemble(root, initial_report, neuron, artifacts, cards):
    from report import merge, source_bundle
    from paper_sections import exemplar, search_patch, labelled
    from report_layout import headline
    from saved_display import peak_fields, default_anatomy
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
    census = {name: read(initial_report / filename) for name, filename in
              [('positive', 'peak-census.json'), ('negative', 'peak-census-negative.json'),
               ('absolute_appendix', 'peak-census-absolute.json')]
              if (initial_report / filename).is_file()}
    patch.update(peak_fields({'peak_census_json': census}))
    anatomy = default_anatomy({**bundle, **patch}, data)
    if anatomy is not None:
        patch['anatomy'] = anatomy
    patch['clusters'] = clusters(artifacts['atlas'], revised['clusters'], patch.get('routes', []))
    patch = labelled(patch, {'units': {u['id']: u['label'] for u in revised['units']}})
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
        rows = [r for r in artifacts['all_measured'] if r['id'].startswith(figure['id'] + ':')]
        write(folder / 'conditions.json', [{**r, 'condition': i} for i, r in enumerate(rows)])
        write(folder / 'tool-request.json', {'id': figure['id'], 'template': figure['template']})
        sweeps.append(str(folder))
    patch['display_selection'] = {'headline_figure_id': headline_figure(artifacts['figures'], artifacts['corpus'])}
    patch['headline'] = headline({**bundle, **patch}, initial_report, sweeps)
    patch['interpretation_costs'] = costs(root, neuron, cards)
    page, bundle = merge(source, patch)
    output = root / 'report'
    output.mkdir(exist_ok=True)
    (output / 'index.html').write_text(page)
    write(output / 'paper-data.json', {'bundle': bundle})
    write(output / 'interpretation-costs.json', patch['interpretation_costs'])
    return output


def directory(output, neurons, references):
    from report_directory import comparison_rows, write_directory
    rows = comparison_rows(output, neurons, references=references)
    write_directory(Path(output) / 'index.html', rows)
