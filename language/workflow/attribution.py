"""Build the measured atlas and three context-specific attribution graphs."""
from .evidence import natural_cases
from .state import read, write


def run(config, root, report, data, request_base, plan, search_rows, control_rows,
        runtime, remote, timeline, gpu_span):
    from paper_sections import exemplar, prepare as prepare_sections, run as run_sections
    from model_tools import attribute

    # Balanced natural and authored cases are retained even when they contradict a hypothesis.
    natural = natural_cases(data, config['atlas_natural'] // 2)
    synthetic, seen_texts = [], set()
    for sign in (1, -1):
        ordered = sorted(search_rows, key=lambda r: sign * r['selected']['product'], reverse=True)
        chosen = []
        for row in ordered:
            if row['text'] not in seen_texts:
                chosen.append(row)
                seen_texts.add(row['text'])
            if len(chosen) >= config['atlas_written'] // 2:
                break
        synthetic.extend(chosen)
    scientific_plan = prepare_sections({
        'atlas': {'cases': [{'case_id': case} for case in natural] +
                           [{'text': r['text'], 'position': r['selected']['position']} for r in synthetic]},
        'family': {'cases': [{'text': text} for text in plan['family_sentences']]}}, report)
    atlas_dir = root / 'atlas'
    atlas_dir.mkdir(exist_ok=True)
    atlas_request = {**request_base, 'id': 'atlas', 'plan': scientific_plan,
                     'limits': {'max_input_tokens': config['settings']['inputs']['max_written_tokens']}}
    write(atlas_dir / 'tool-request.json', atlas_request)
    timeline.status('attribution-and-family')
    with gpu_span('atlas-and-family'):
        atlas_result = (remote.call('sections', request=atlas_request, folder=str(atlas_dir))
                        if remote is not None else run_sections(atlas_request, atlas_dir, runtime))
        write(atlas_dir / 'result.json', atlas_result)
    if atlas_result['status'] != 'complete':
        raise RuntimeError('Attribution measurements incomplete; inspect atlas/result.json before resuming')
    atlas_sections = read(atlas_dir / 'sections.json')
    atlas = read(atlas_dir / 'atlas-raw.json')
    graph_rows = []
    # Different measured contexts: strongest positive, strongest negative, strongest control counterexample.
    graph_sources = [max(search_rows, key=lambda r: r['selected']['product']),
                     min(search_rows, key=lambda r: r['selected']['product']),
                     max(control_rows, key=lambda r: abs(r['selected']['product']))]
    graph_results = []
    for i, row in enumerate(graph_sources):
        timeline.status('three-graphs', graph=i + 1)
        folder = root / 'graphs' / str(i + 1)
        folder.mkdir(parents=True, exist_ok=True)
        graph_request = {**request_base, 'id': f'graph-{i + 1}', 'title': row['label'],
            'text': row['text'], 'root': {'kind': 'neuron_product', 'position': row['selected']['position']}}
        write(folder / 'request.json', graph_request)
        result = read(folder / 'result.json') if (folder / 'result.json').exists() else None
        if result is None or result['status'] != 'complete':
            with gpu_span(f'graph-{i + 1}'):
                result = (remote.call('attribute', request=graph_request, folder=str(folder))
                          if remote is not None else attribute(graph_request, folder, runtime))
        result['steering']['activation_context'] = exemplar(row)
        graph_rows.append(read(folder / 'graph.json'))
        graph_results.append(result)
    return atlas_sections, atlas, graph_rows, graph_results
