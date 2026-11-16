"""Run an independent, schema-directed investigation from saved native activations."""
import concurrent.futures
import gc
import os
import shutil
import time
from pathlib import Path
from . import contracts, prompts
from .client import Client
from .state import read, write, lock, now, Timeline
from .evidence import initial_packet, spectrum_packet, prediction_feedback, evidence_catalog
from .interpretation import explain
from .measure import Measurements, summarized
from .revision import ablation_analysis_packet, finalize, resume_graph_ablation_packet
from .controls import run as run_controls
from .search import run as run_search, validate_candidates
from .attribution import run as run_attribution
from .route_prediction import run as predict_routes
from . import neuron_labels


def run(config, root, stop_after=None, *, checkpoint_attribution=False, followup=None):
    """Run all enabled stages under one lock and report their final status."""
    from .followups import run as investigate
    from .render import finalize_report

    if followup is None:
        followup = stop_after is None
    if followup and stop_after is not None:
        raise ValueError('Follow-ups require a full investigation')
    root = Path(root)
    with lock(root / 'execution.lock', blocking=False):
        try:
            output = _run_foundation(config, root, stop_after=stop_after,
                                     checkpoint_attribution=checkpoint_attribution)
            if stop_after is not None:
                return output
            timeline = Timeline(root)
            final_status = {'phase': 'complete'}
            if followup:
                with timeline.span('followup', 'execution', 6):
                    investigate(root, root / 'followup', in_place=True, workflow_timeline=timeline)
                saved = read(root / 'followup/status.json')
                final_status = {key: saved[key] for key in ('phase', 'failures', 'stop_reason') if key in saved}
            output = finalize_report(root, read(root / 'route-prediction/result.json'), config, followup=followup)
            timeline.status(**final_status, html=str(output / 'index.html'))
            return output
        except Exception as error:
            import traceback
            write(root / 'failure.json', {'at': now(), 'error': str(error), 'traceback': traceback.format_exc()})
            Timeline(root).status('failed', error=str(error))
            raise


def _run_foundation(config, root, stop_after=None, *, checkpoint_attribution=False):
    from .render import assemble, assemble_interpretation, costs, update_routes
    root, report = Path(root), Path(config['report'])
    root.mkdir(parents=True, exist_ok=True)
    timeline = Timeline(root)
    client = Client(config['llm'], root / 'api', config['api_slots'])
    data = read(report / 'report.json')
    neuron = {'layer': data['identity']['layer'], 'index': data['identity']['neuron']}
    request_base = {'model': config['model'], 'neuron': neuron, 'settings': config['settings']}
    runtime_request = ({**request_base, 'runtime': {'activation_offload': 'checkpoint'}}
                       if checkpoint_attribution else request_base)
    old_config = root / 'config.json'
    if old_config.exists() and read(old_config) != config:
        raise ValueError('Configuration changed; use a separate run directory')
    write(old_config, config)
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if stop_after == 'interpretation' else ','.join(map(str, config['gpus']))
    os.environ['NEURON_MICROSCOPE_SETTINGS'] = str(root / 'backend-settings.json')
    write(Path(os.environ['NEURON_MICROSCOPE_SETTINGS']), config['settings'])
    if stop_after is None and (root / 'artifacts.json').exists():
        # Resume report additions without repeating the completed investigation.
        from ..tools.report import REPORT_SYNTHESIS
        artifacts = read(root / 'artifacts.json')
        output = root / 'report'
        html = output / 'index.html'
        previous = html.read_text(encoding='utf-8') if html.exists() else ''
        with timeline.span('resume-report', 'execution', 0):
            if not previous or '<script id="interpretation-view">' in previous:
                with timeline.span('render', 'cpu', 5):
                    assemble(root, report, config['neuron'], artifacts, len(config['gpus']))
            result = predict_routes(client, root, timeline)
            neuron_labels.complete(config, root, artifacts.get('atlas_sections', {}), client, timeline)
        if not REPORT_SYNTHESIS.search(previous):
            output = update_routes(root, result, config)
        timeline.status('foundation_complete', html=str(output / 'index.html'))
        return output
    if stop_after == 'interpretation' and (root / 'report/index.html').exists() and (
            (root / 'artifacts.json').exists() or ((root / 'status.json').exists() and
            read(root / 'status.json')['phase'] in ('complete', 'interpretation_complete', 'complete_with_failures'))):
        return root / 'report'
    with timeline.span('workflow', 'execution', 0):
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
                                             config.get('max_rules_per_pole', 3),
                                             config.get('max_refinements', 2))
        if stop_after == 'interpretation':
            with timeline.span('render', 'cpu', 5):
                output = assemble_interpretation(root, report, initial, sample_data, predictions)
    if stop_after == 'interpretation':
        write(root / 'interpretation-costs.json', costs(root, config['neuron'], 0))
        timeline.status('interpretation_complete', html=str(output / 'index.html'))
        return output
    from common import Runtime
    from paper_sections import exemplar
    from .ablation_run import prepare as prepare_ablation, measure as measure_ablation
    with timeline.span('workflow-experiments', 'execution', 0):
        packet = initial_packet(report)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            blind_future = pool.submit(client.request, 'blind-writer', prompts.BLIND,
                                      {'claim': initial['claim'], 'response_rules': initial['response_rules']},
                                      contracts.BLIND, 3)
            timeline.status('search-design')
            search_proposal = client.request('search/round-1', prompts.SEARCH,
                {'initial': initial, 'evidence': packet, 'regions_requested': config['search_regions'],
                 'candidates_per_region': config['candidates_per_region'],
                 'instruction': prompts.INITIAL_SEARCH_INSTRUCTION},
                contracts.SEARCH, 3, check=validate_candidates)
            timeline.status('gpu-queue')
            queued = time.time()
            from .gpu_worker import GPUClient
            remote = GPUClient(config, root, timeline, runtime_request) if os.environ.get('NEURON_MICROSCOPE_GPU_WORKER') == '1' else None
            with remote if remote is not None else lock(config['gpu_lock']):
                if remote is not None:
                    runtime = remote
                else:
                    write(root / 'gpu-queue.json', {'seconds': time.time() - queued, 'devices': config['gpus']})
                    timeline.status('loading-model')
                    with timeline.span('model-load', 'gpu', 3):
                        runtime = Runtime.load(runtime_request)
                gpu_span = remote.span if remote is not None else lambda name: timeline.span(name, 'gpu', 3)
                measure = Measurements(runtime, root / 'measurements', timeline,
                                       config['batch_size'], config['batch_tokens'])
                search_rows, region_records, round_summaries = run_search(
                    client, measure, initial, packet, search_proposal, config, root, timeline, gpu_span)
                plan, control_rows, figures = run_controls(client, measure, initial, packet, search_rows,
                                                          root, timeline)
                blind = blind_future.result()
                blind_items = [{'id': f'blind-{pole}-{i}', **candidate,
                                'score': {'location': 'input', 'reduction': 'max' if pole == 'positive' else 'min'}}
                               for pole in ('positive', 'negative') for i, candidate in enumerate(blind[pole])]
                blind_rows = measure.texts(blind_items, 'blind-tests')
                tests = {}
                for pole, field, sign in [('positive', 'blind_max', 1), ('negative', 'blind_min', -1)]:
                    candidates = [r for r in blind_rows if r['id'].startswith('blind-' + pole)]
                    if not candidates:
                        continue
                    winner = max(candidates, key=lambda r: sign * r['selected']['product'])
                    tests[field] = {'target': data['statistics']['product']['max' if sign == 1 else 'min'],
                                    'achieved': winner['selected']['product'], **exemplar(winner)}
                tests['explanation_version'] = predictions['version']
                atlas_sections, atlas, graph_rows, graph_results = run_attribution(
                    config, root, report, data, request_base, plan, search_rows, control_rows,
                    runtime, remote, timeline, gpu_span)
                timeline.status('point-ablations')
                ablation_folder = root / 'ablation'
                ablation_plan = prepare_ablation(config, request_base, sample_data, contexts, control_rows,
                    figures, data, predictions, initial, ablation_folder)
                with gpu_span('point-ablations'):
                    ablation_section = (remote.call('measure-ablations', plan=ablation_plan,
                        folder=str(ablation_folder.resolve())) if remote is not None else
                        measure_ablation(runtime, ablation_plan, ablation_folder))
                label_evidence = neuron_labels.prepare(config, root, atlas_sections, timeline, runtime,
                                                       gpu_span=gpu_span)
                measure.runtime = None
                if remote is None:
                    torch = runtime.torch
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
            'matched_ablations': ablation_analysis_packet(ablation_section, evidence)}
        legacy_analysis = resume_graph_ablation_packet(root, revision_packet)
        revision, predictions, review = finalize(client, initial, predictions, revision_packet,
            evidence, sample_data, contexts, root, timeline)
        if legacy_analysis:
            revision['section_prose'].pop('ablation', None)
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
        # Finish the base report before API-only prediction so interrupted runs
        # cannot retain an earlier interpretation-only preview.
        timeline.status('rendering')
        with timeline.span('render', 'cpu', 5):
            assemble(root, report, config['neuron'], artifacts, len(config['gpus']))
        tests['route_prediction'] = predict_routes(client, root, timeline)
        write(root / 'artifacts.json', artifacts)
        neuron_labels.complete(config, root, atlas_sections, client, timeline, label_evidence)
    output = update_routes(root, tests['route_prediction'], config)
    timeline.status('foundation_complete', html=str(output / 'index.html'))
    return output


def main():
    from .cli import main as cli_main
    return cli_main()


if __name__ == '__main__':
    raise SystemExit(main())
