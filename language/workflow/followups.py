"""Run at most two autonomous follow-up rounds with reproducible checkpoints."""
import argparse
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from .client import Client
from .state import Timeline, lock, read, write
from .followup_execute import Executor
from .followup_investigation import rounds
from .followup_report import render

DEFAULTS = {'seed': 42, 'max_tokens': 1024, 'natural_per_group': 12}
SOURCE_FILES = ('config.json', 'artifacts.json', 'interpretation.json', 'predictions.json',
                'evidence.json', 'report/index.html')


def fingerprint(value):
    return hashlib.sha256(value).hexdigest()


def input_identity(config):
    """Pin small metadata; record large arrays/weights by size and modification time."""
    from common import model_identity
    model, archive = Path(config['model']), Path(config['activations'])
    dataset = read(archive / 'dataset.json')
    files = [dataset['corpus']['documents'], dataset['corpus']['token_ids']]
    files += [str(Path(part['path']) / shard['path']) for part in dataset['parts'] for shard in part['shards']]
    return {'model': model_identity(model, read(model / 'config.json')),
            'archive': {'metadata': {name: fingerprint((archive / name).read_bytes())
                        for name in ('dataset.json', dataset['neurons'])},
                        'arrays': {name: {'bytes': (archive / name).stat().st_size,
                                          'mtime_ns': (archive / name).stat().st_mtime_ns} for name in files}}}


def load_source(source, report_snapshot=None):
    if read(source / 'status.json')['phase'] != 'complete':
        raise ValueError('Follow-up investigation requires a completed foundation investigation')
    files = {name: (report_snapshot if name == 'report/index.html' and report_snapshot is not None
                   else source / name).read_bytes() for name in SOURCE_FILES}
    config, artifacts, initial, predictions, evidence = [json.loads(files[name]) for name in SOURCE_FILES[:-1]]
    if artifacts['initial'] != initial or artifacts['predictions'] != predictions:
        raise ValueError('Foundation artifacts do not match the frozen explanation and predictions')
    from .followup_report import BUNDLE
    from report import strip_neuron_labels
    html = strip_neuron_labels(files['report/index.html'].decode('utf-8'))
    files['report/index.html'] = html.encode('utf-8')
    bundles = list(BUNDLE.finditer(html))
    if len(bundles) != 1:
        raise ValueError('Expected exactly one foundation report bundle')
    bundle = json.loads(bundles[0][2])
    if bundle['neuron']['id'] != config['neuron']:
        raise ValueError('Foundation report and configuration must identify the same neuron')
    if any(row.get('investigation_id') for row in bundle.get('criticism', [])):
        raise ValueError('Source report already contains follow-ups; use the frozen foundation report')
    criticisms = artifacts['revision']['criticism']
    if not criticisms:
        raise ValueError('Foundation has no original Criticism')
    catalog = {identity: {'id': identity, **row} for identity, row in artifacts['evidence'].items()}
    for graph in artifacts.get('graphs', []):
        identity = 'graph:' + graph['id']
        catalog[identity] = {'id': identity, 'kind': 'saved_graph', 'data': graph}
    if artifacts.get('ablation'):
        catalog['baseline:ablation'] = {'id': 'baseline:ablation', 'kind': 'ablation_summary',
                                         'data': artifacts['ablation']['summary']}
    original = [{'id': f'C{i}', **row} for i, row in enumerate(criticisms, 1)]
    excluded = [str(row['id']) for row in evidence.get('documents', [])]
    return {'config': config, 'initial': initial, 'predictions': predictions,
            'original_criticism': original, 'catalog': catalog, 'excluded_documents': excluded,
            'html': html,
            'files': {name: fingerprint(blob) for name, blob in files.items()}}


def route_overlay(html, source):
    """Keep later blind predictions outside the frozen follow-up evidence."""
    result = source / 'route-prediction/result.json'
    if not result.exists():
        return html
    from report import update_route_prediction
    costs = source / 'report/interpretation-costs.json'
    page, _ = update_route_prediction(html, read(result),
        interpretation_costs=read(costs) if costs.exists() else None)
    return page


def integrate_report(report, source, *, in_place=True):
    """Replace report files atomically, without writing through publication hardlinks."""
    if not in_place and not (source / 'neuron-labels/result.json').is_file():
        return report
    destination = source / 'report' if in_place else report
    html = (report / 'index.html').read_text(encoding='utf-8')
    if in_place:
        html = route_overlay(html, source)
    from .neuron_labels import report_overlay
    html, bundle = report_overlay(html, source)
    contents = {'paper-data.json': (json.dumps({'bundle': bundle},
                                               ensure_ascii=False, indent=2) + '\n').encode('utf-8'),
                'index.html': html.encode('utf-8')}
    for name, content in contents.items():
        temporary = destination / (name + '.followup.tmp')
        temporary.write_bytes(content)
        temporary.replace(destination / name)
    return destination


def run(source, root, options=None, *, in_place=False, client_factory=Client, executor_factory=Executor):
    source, root = Path(source).resolve(), Path(root).resolve()
    if in_place:
        if root != source / 'followup':
            raise ValueError('In-place follow-ups must store records in <source>/followup')
    elif source == root or source in root.parents or root in source.parents:
        raise ValueError('Follow-up investigation needs a separate output directory outside the foundation')
    options = {**DEFAULTS, **(options or {})}
    if set(options) != set(DEFAULTS) or any(type(value) is not int or value < (0 if key == 'seed' else 1)
                                           for key, value in options.items()):
        raise ValueError('Use a nonnegative integer seed and positive integer token/sample limits')
    with (lock(source / 'execution.lock', blocking=False) if in_place else nullcontext()), \
         lock(root / 'execution.lock', blocking=False):
        foundation = root / 'foundation.html'
        base = load_source(source, foundation if in_place and foundation.exists() else None)
        manifest = {'source': str(source), 'files': base['files'], 'options': options,
                    'inputs': input_identity(base['config'])}
        marker = root / 'manifest.json'
        if marker.exists():
            saved = read(marker)
            if any(saved.get(key) != value for key, value in manifest.items()):
                raise ValueError('Source or settings changed; use a new output directory')
        elif any(path.name != 'execution.lock' for path in root.iterdir()):
            raise ValueError('Follow-up output must be empty or belong to this same investigation')
        else:
            write(marker, manifest)
        if in_place:
            from report import strip_neuron_labels
            current = strip_neuron_labels((source / 'report/index.html').read_bytes().decode('utf-8')).encode('utf-8')
            rendered = root / 'report/index.html'
            if (fingerprint(current) != base['files']['report/index.html'] and
                    current != route_overlay(base['html'], source).encode('utf-8')) and (
                    not rendered.exists() or (current != rendered.read_bytes() and
                    current != route_overlay(rendered.read_text(encoding='utf-8'), source).encode('utf-8'))):
                raise ValueError('Current report changed outside this follow-up investigation')
            if not foundation.exists():
                temporary = foundation.with_suffix('.tmp')
                temporary.write_bytes(current)
                temporary.replace(foundation)
        write(root / 'frozen.json', {key: base[key] for key in ('initial', 'predictions', 'original_criticism')})
        timeline = Timeline(root)
        config = base['config']
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config['gpus']))
        os.environ['NEURON_MICROSCOPE_SETTINGS'] = str(root / 'backend-settings.json')
        write(root / 'backend-settings.json', config['settings'])
        try:
            if (root / 'status.json').exists() and read(root / 'status.json')['phase'] in ('complete', 'complete_with_failures'):
                return integrate_report(root / 'report', source, in_place=in_place)
            client = client_factory(config['llm'], root / 'api', config['api_slots'])
            with executor_factory(config, root / 'measurements', timeline, options,
                                  base['excluded_documents']) as executor:
                result = rounds(client, executor, base, root, options, timeline)
            write(root / 'investigation.json', result)
            # Rendering alone receives full traces; planner history includes every case's outcomes.
            display_rounds = json.loads(json.dumps(result['rounds']))
            for current in display_rounds:
                for record in current['investigations']:
                    path = root / record['id'] / 'evidence.json'
                    if path.exists():
                        record['evidence'] = read(path)
            output = render(base['html'], display_rounds, root)
            output = integrate_report(output, source, in_place=in_place)
            failures = sum(record['status'] == 'failed' for current in result['rounds']
                           for record in current['investigations'])
            timeline.status('complete_with_failures' if failures else 'complete',
                            failures=failures, stop_reason=result['stop_reason'], html=str(output / 'index.html'))
            return output
        except Exception as error:
            write(root / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
            timeline.status('failed')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='completed foundation investigation')
    parser.add_argument('--output', required=True, type=Path, help='separate follow-up directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-tokens', type=int, default=1024, help='per contrast input; never truncated')
    parser.add_argument('--natural-per-group', type=int, default=12)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    output = run(args.source, args.output, {'seed': args.seed, 'max_tokens': args.max_tokens,
                                          'natural_per_group': args.natural_per_group})
    print(output / 'index.html')
    return 0 if read(args.output / 'status.json')['phase'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
