"""Run an independent, schema-directed investigation from saved native activations."""
import argparse
import concurrent.futures
import copy
import gc
import os
import shutil
import sys
import time
from pathlib import Path
from . import contracts, prompts
from .client import Client
from .state import read, write, lock, Timeline, now
from .evidence import initial_packet, spectrum_packet, natural_cases, prediction_feedback, evidence_catalog
from .interpretation import explain
from .measure import Measurements, summarized, search_regions
from .revision import finalize


def validate_candidates(output):
    ids = [r['id'] for r in output['regions']]
    if len(set(ids)) != len(ids):
        raise ValueError('Region IDs must be unique')
    for region in output['regions']:
        for candidate in region['candidates']:
            if len(candidate['label'].split()) > 4 or len(candidate['label']) > 48:
                raise ValueError('Candidate labels must be one word or a phrase of at most four words and 48 characters')


def run(config, root, stop_after=None):
    from .render import assemble, assemble_interpretation, costs
    root, report = Path(root), Path(config['report'])
    root.mkdir(parents=True, exist_ok=True)
    timeline = Timeline(root)
    client = Client(config['llm'], root / 'api', config['api_slots'])
    data = read(report / 'report.json')
    neuron = {'layer': data['identity']['layer'], 'index': data['identity']['neuron']}
    request_base = {'model': config['model'], 'neuron': neuron, 'settings': config['settings']}
    old_config = root / 'config.json'
    if old_config.exists() and read(old_config) != config:
        raise ValueError('Configuration changed; use a separate run directory')
    write(old_config, config)
    if (root / 'status.json').exists() and read(root / 'status.json')['phase'] in (
            ('complete', 'interpretation_complete') if stop_after == 'interpretation' else ('complete',)):
        return root / 'report'
    with lock(root / 'execution.lock', blocking=False), timeline.span('workflow', 'execution', 0):
        timeline.status('prepare')
        with timeline.span('saved-evidence', 'cpu', 1):
            if config.get('saved_sample'):
                for filename in ('sections.json', 'saved-contexts.json', 'sampling.json'):
                    destination = root / 'sample' / filename
                    destination.parent.mkdir(exist_ok=True)
                    if not destination.exists():
                        shutil.copy2(Path(config['saved_sample']) / filename, destination)
            else:
                from .sampling import prepare as sample
                sample(report, {'model': config['model'], 'settings': config['settings']},
                       config['activations'], root / 'sample')
            sample_data = read(root / 'sample/sections.json')['spectrum']
            contexts = read(root / 'sample/saved-contexts.json')
            packet = spectrum_packet(report, sample_data, contexts)
            write(root / 'evidence.json', packet)
        _, initial, _, predictions = explain(client, packet, sample_data, contexts, root, timeline,
                                             config.get('max_rules_per_pole', 3))
        if stop_after == 'interpretation':
            with timeline.span('render', 'cpu', 5):
                output = assemble_interpretation(root, report, initial, sample_data, predictions)
    if stop_after == 'interpretation':
        write(root / 'interpretation-costs.json', costs(root, config['neuron'], 0))
        timeline.status('interpretation_complete', html=str(output / 'index.html'))
        return output
    from common import Runtime
    from paper_sections import prepare as prepare_sections, run as run_sections
    from model_tools import attribute
    from .interventions import matched_ablation
    with lock(root / 'execution.lock', blocking=False), timeline.span('workflow-experiments', 'execution', 0):
        packet = initial_packet(report)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            blind_future = pool.submit(client.request, 'blind-writer', prompts.BLIND,
                                      {'claim': initial['claim']}, contracts.BLIND, 3)
            def check_plan(plan):
                for control in plan['controls']:
                    contracts.control_request(control)
            timeline.status('search-design')
            search_proposal = client.request('search/round-1', prompts.SEARCH,
                {'initial': initial, 'evidence': packet, 'regions_requested': config['search_regions'],
                 'candidates_per_region': config['candidates_per_region'],
                 'instruction': 'Split directions evenly across positive and negative poles.'},
                contracts.SEARCH, 3, check=validate_candidates)
            timeline.status('gpu-queue')
            queued = time.time()
            with lock(config['gpu_lock']):
                write(root / 'gpu-queue.json', {'seconds': time.time() - queued, 'devices': config['gpus']})
                timeline.status('loading-model')
                with timeline.span('model-load', 'gpu', 3):
                    runtime = Runtime.load(request_base)
                measure = Measurements(runtime, root / 'measurements', timeline,
                                       config['batch_size'], config['batch_tokens'])
                if not (root / 'batch-comparison.json').exists():
                    candidates = [c['text'] for r in search_proposal['regions'] for c in r['candidates']]
                    pair = sorted(candidates, key=len)
                    token_lists = [runtime.tokenizer.encode(s, add_special_tokens=False) for s in (pair[0], pair[-1])]
                    with timeline.span('batch-comparison', 'gpu', 3):
                        batched = measure._forward(token_lists)
                        differences = []
                        for ids, batch in zip(token_lists, batched):
                            with runtime.torch.no_grad(), runtime.capture() as single:
                                runtime.forward(ids)
                            differences.append({key: {'maximum_absolute_difference': max(abs(a - b) for a, b in zip(batch[key], single[key].tolist())),
                                'maximum_absolute_value': max(map(abs, single[key].tolist()))}
                                for key in ('gate', 'value', 'product')})
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
                plan = client.request('control-plan', prompts.CONTROL,
                    {'initial': initial, 'evidence': packet,
                     'measured_search': [summarized(row) for sign in (1, -1) for row in
                         sorted(search_rows, key=lambda r: sign * r['selected']['product'], reverse=True)[:8]]},
                    contracts.PLAN, 3, check_plan)
                timeline.status('controlled-tests')
                control_rows, figures = measure.controls(plan['controls'])
                blind = blind_future.result()
                blind_items = [{'id': f'blind-{pole}-{i}', **candidate,
                                'score': {'location': 'input', 'reduction': 'max' if pole == 'positive' else 'min'}}
                               for pole in ('positive', 'negative') for i, candidate in enumerate(blind[pole])]
                blind_rows = measure.texts(blind_items, 'blind-tests')
                tests = {}
                for pole, field, sign in [('positive', 'blind_max', 1), ('negative', 'blind_min', -1)]:
                    winner = max((r for r in blind_rows if r['id'].startswith('blind-' + pole)),
                                 key=lambda r: sign * r['selected']['product'])
                    from paper_sections import exemplar
                    tests[field] = {'target': data['statistics']['product']['max' if sign == 1 else 'min'],
                                    'achieved': winner['selected']['product'], **exemplar(winner)}
                tests['explanation_version'] = predictions['version']
                tests['route_prediction'] = None
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
                with timeline.span('atlas-and-family', 'gpu', 3):
                    atlas_result = run_sections(atlas_request, atlas_dir, runtime)
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
                    if (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'complete':
                        result = read(folder / 'result.json')
                    else:
                        with timeline.span(f'graph-{i + 1}', 'gpu', 3):
                            result = attribute(graph_request, folder, runtime)
                    graph_rows.append(read(folder / 'graph.json'))
                    graph_results.append(result)
                with timeline.span('matched-ablations', 'gpu', 3):
                    ablation_section = matched_ablation(runtime, graph_results, root / 'graphs', config['settings'])
                for result, row in zip(graph_results, graph_sources):
                    result['steering']['activation_context'] = exemplar(row)
                torch = runtime.torch
                measure.runtime = None
                del runtime
                gc.collect()
                torch.cuda.empty_cache()
        timeline.status('revision')
        all_measured = search_rows + control_rows + blind_rows
        evidence = evidence_catalog(all_measured, atlas, graph_results, sample_data, packet)
        revision_packet = {'initial': initial, 'natural_evidence': packet,
            'search_progress': round_summaries,
            'search': [summarized(r) for r in search_rows], 'controls': [summarized(r) for r in control_rows],
            'blind_tests': [summarized(r) for r in blind_rows], 'prediction_feedback': prediction_feedback(predictions),
            'atlas': [{'id': r['id'], 'text': r['text'], 'selected': r['selected'],
                       'upstream': r['upstream'][:8], 'downstream': r['downstream'][:8]} for r in atlas],
            'circuit': atlas_sections['circuit'],
            'graphs': [{'id': g['id'], 'text': g['graph']['text'], 'meta': g['graph']['meta']} for g in graph_rows],
            'interventions': [{k: r.get(k) for k in ('ablation', 'steering')} for r in graph_results],
            'matched_ablations': ablation_section}
        revision, predictions, review = finalize(client, initial, predictions, revision_packet,
            evidence, sample_data, contexts, root, timeline)
        for result in graph_results:
            tests.setdefault('steering', []).append(result['steering'])
            ablation = result.get('ablation') or {}
            tests.setdefault('ablation', []).append({'graph_id': result['graph']['id'],
                'effect': ablation.get('delta_target_logit'),
                'note': 'Zero product at token ' + str(ablation.get('position')) +
                        '; change in next-token logit of ' + str(ablation.get('reference_token'))})
        artifacts = {'initial': initial, 'revision': revision, 'review': review,
            'sample': sample_data, 'predictions': predictions, 'figures': figures, 'tests': tests,
            'search_regions': list(region_records.values()), 'atlas': atlas, 'atlas_sections': atlas_sections,
            'graphs': graph_rows, 'all_measured': all_measured, 'logit_effect': packet['logit_effect'],
            'evidence': evidence,
            'ablation': ablation_section,
            'corpus': {key: data['statistics']['product'][key] for key in ('max', 'min')}}
        write(root / 'artifacts.json', artifacts)
        timeline.status('rendering')
        with timeline.span('render', 'cpu', 5):
            assemble(root, report, config['neuron'], artifacts, len(config['gpus']))
    # Final costs include the completed execution interval and render operation.
    output = assemble(root, report, config['neuron'], artifacts, len(config['gpus']))
    timeline.status('complete', html=str(output / 'index.html'))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backend-tools', type=Path, default=Path(__file__).resolve().parents[1] / 'tools')
    parser.add_argument('--stop-after', choices=['interpretation'])
    args = parser.parse_args()
    sys.path.insert(0, str(args.backend_tools.resolve()))
    config = read(args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.stop_after == 'interpretation' else ','.join(map(str, config['gpus']))
    os.environ['NEURON_MICROSCOPE_SETTINGS'] = str(args.output / 'backend-settings.json')
    write(Path(os.environ['NEURON_MICROSCOPE_SETTINGS']), config['settings'])
    try:
        output = run(config, args.output, stop_after=args.stop_after)
        if config.get('publish'):
            from comparison import publish
            destination = Path(config['publish']) / config['neuron']
            if not destination.exists():
                publish(output, destination, config['neuron'], config['references'])
            from .render import directory
            with lock(Path(config['publish']) / '.directory.lock'):
                directory(config['publish'], config.get('publish_neurons', [config['neuron']]), config['references'])
    except Exception as error:
        import traceback
        write(args.output / 'failure.json', {'at': now(), 'error': str(error), 'traceback': traceback.format_exc()})
        Timeline(args.output).status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
