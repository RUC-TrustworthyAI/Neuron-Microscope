"""Register, measure and replay the expanded Effect experiment."""
import os
import traceback
from pathlib import Path

from .ablation_cases import select_ablation_cases
from .interventions import measure_ablations, random_controls
from .state import Timeline, lock, now, read, write


def compatible_plan(saved, current):
    """Allow a saved plan to acquire authored next tokens for its existing cases."""
    cases = []
    for case in current['cases']:
        if case['kind'] == 'controlled':
            cases.append({**case, 'next_token_id': None})
        else:
            cases.append(case)
    return saved == current or saved == {**current, 'cases': cases}


def prepare(config, request, sample, contexts, control_rows, figures, data, predictions, initial,
            folder, *, source=None):
    from common import model_config, model_identity
    from .render import headline_figure

    settings, neuron = request['settings'], request['neuron']
    options = settings['followups']
    path, model = model_config(request['model'], neuron)
    identity = model_identity(path, model)
    archive_model = Path(config['activations']) / 'model.json'
    if archive_model.is_file():
        saved = read(archive_model)['identity']
        keys = ('model_type', 'config_sha256', 'small_files', 'shards')
        if any(identity[key] != saved[key] for key in keys):
            raise ValueError('Ablation model differs from the native activation archive')
    preferred = headline_figure(figures, data['statistics']['product'], initial) if figures else None
    plan = select_ablation_cases(sample, contexts, control_rows, figures, data=data,
                                 predictions=predictions, preferred_figure_id=preferred,
                                 seed=settings['seed'], natural_top=options.get('natural_top', 20),
                                 natural_bottom=options.get('natural_bottom', 10))
    plan.update(schema_version=1, neuron=neuron, model=identity, settings=settings,
                devices=config['gpus'], activation_archive=str(Path(config['activations']).resolve()),
                source_investigation=str(source) if source is not None else None,
                control_neurons=random_controls(settings, neuron, model['intermediate_size']))
    path = folder / 'plan.json'
    if path.exists() and not compatible_plan(read(path), plan):
        raise ValueError('Ablation plan changed; choose a separate run directory')
    write(path, plan)
    return plan


def measure(runtime, plan, folder):
    loaded = {key: runtime.identity[key] for key in plan['model']}
    if loaded != plan['model'] or runtime.neuron != plan['neuron']:
        raise ValueError('Loaded runtime differs from the registered ablation plan')
    with lock(folder / 'measure.lock'):
        return measure_ablations(runtime, plan, folder)


def replay(source, root):
    """Add a fresh Effect panel to a completed local investigation."""
    from common import Runtime, digest
    from report import replace_ablation

    source, root = source.resolve(), root.resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Ablation replay requires a separate output directory')
    if read(source / 'status.json')['phase'] != 'complete':
        raise ValueError('Ablation replay requires a completed source investigation')
    config = read(source / 'config.json')
    source_html = source / 'report/index.html'
    inputs = ['config.json', 'artifacts.json', 'sample/sections.json',
              'sample/saved-contexts.json', 'report/index.html']
    receipt = {'source_investigation': str(source), 'files': {name: digest(source / name) for name in inputs}}
    marker = root / 'ablation-source.json'
    if root.is_dir() and not marker.exists() and any(path.name != 'execution.lock' for path in root.iterdir()):
        raise ValueError('Ablation replay requires an empty or matching output directory')
    with lock(root / 'execution.lock', blocking=False):
        if marker.exists():
            if read(marker) != receipt:
                raise ValueError('Source investigation changed; choose a separate output directory')
        elif any(path.name != 'execution.lock' for path in root.iterdir()):
            raise ValueError('Ablation replay requires an empty or matching output directory')
        else:
            write(marker, receipt)
        timeline = Timeline(root)
        try:
            timeline.status('ablation-prepare')
            artifacts = read(source / 'artifacts.json')
            sample = read(source / 'sample/sections.json')['spectrum']
            contexts = read(source / 'sample/saved-contexts.json')
            data = read(Path(config['report']) / 'report.json')
            neuron = {'layer': data['identity']['layer'], 'index': data['identity']['neuron']}
            if config['neuron'] != f"L{neuron['layer']}_N{neuron['index']}":
                raise ValueError('Source report and investigation identify different neurons')
            request = {key: config[key] for key in ('model', 'settings')}
            request['neuron'] = neuron
            folder = root / 'ablation'
            plan = prepare(config, request, sample, contexts, artifacts['all_measured'], artifacts['figures'],
                           data, artifacts['predictions'], artifacts['initial'], folder, source=source)
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config['gpus']))
            timeline.status('ablation-measurements')
            with lock(config['gpu_lock']):
                with timeline.span('ablation-model-load', 'gpu', 3):
                    runtime = Runtime.load(request)
                with timeline.span('point-ablations', 'gpu', 3):
                    section = measure(runtime, plan, folder)
            timeline.status('rendering')
            output = root / 'report'
            with timeline.span('ablation-render', 'cpu', 5):
                page, bundle = replace_ablation(source_html.read_text(encoding='utf-8'), section)
                write(output / 'paper-data.json', {'bundle': bundle})
                temporary = output / 'index.html.tmp'
                temporary.write_text(page, encoding='utf-8')
                temporary.replace(output / 'index.html')
            (root / 'failure.json').unlink(missing_ok=True)
            timeline.status('complete', html=str(output / 'index.html'))
        except (Exception, KeyboardInterrupt) as error:
            write(root / 'failure.json', {'at': now(), 'error': str(error), 'traceback': traceback.format_exc()})
            timeline.status('failed', error=str(error))
            raise
    return output
