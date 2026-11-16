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
from .followup_report import BUNDLE, legacy_scoring_view, render

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


def load_source(source, report_snapshot=None, *, integrated=False):
    ready = ('complete', 'foundation_complete') if integrated else ('complete',)
    if read(source / 'status.json')['phase'] not in ready:
        raise ValueError('Follow-up investigation requires a completed foundation investigation')
    files = {name: (report_snapshot if name == 'report/index.html' and report_snapshot is not None
                   else source / name).read_bytes() for name in SOURCE_FILES}
    config, artifacts, initial, predictions, evidence = [json.loads(files[name]) for name in SOURCE_FILES[:-1]]
    if artifacts['initial'] != initial or artifacts['predictions'] != predictions:
        raise ValueError('Foundation artifacts do not match the frozen explanation and predictions')
    from report import strip_neuron_labels, strip_report_synthesis
    html = strip_neuron_labels(strip_report_synthesis(files['report/index.html'].decode('utf-8')))
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


def route_overlay(html, source, *, costs):
    """Keep later blind predictions outside the frozen follow-up evidence."""
    result = source / 'route-prediction/result.json'
    if not result.exists():
        return html
    from report import update_route_prediction
    page, _ = update_route_prediction(html, read(result), interpretation_costs=costs)
    return page


def integrate_report(report, source, *, in_place=True):
    """Replace report files atomically, without writing through publication hardlinks."""
    if not in_place and not (source / 'neuron-labels/result.json').is_file():
        return report
    destination = source / 'report' if in_place else report
    html = (report / 'index.html').read_text(encoding='utf-8')
    if in_place:
        costs = source / 'report/interpretation-costs.json'
        html = route_overlay(html, source, costs=read(costs) if costs.exists() else None)
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


class FollowupTimeline(Timeline):
    def __init__(self, root, workflow=None):
        super().__init__(root)
        self.workflow = workflow

    def status(self, phase, **extra):
        super().status(phase, **extra)
        if self.workflow is not None:
            self.workflow.status('followup', step=phase, **extra)


def _render_saved_report(base_html, result, root):
    """Use the archived case records to add full traces to the rendered report."""
    display_rounds = json.loads(json.dumps(result['rounds']))
    for current in display_rounds:
        for record in current['investigations']:
            path = root / record['id'] / 'evidence.json'
            if path.exists():
                record['evidence'] = read(path)
    return render(base_html, display_rounds, root)


def run(source, root, options=None, *, in_place=False, client_factory=Client, executor_factory=Executor,
        workflow_timeline=None):
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
    if workflow_timeline is not None and (not in_place or workflow_timeline.root.resolve() != source):
        raise ValueError('Integrated follow-ups require the source workflow timeline')
    with (lock(source / 'execution.lock', blocking=False) if in_place and workflow_timeline is None else nullcontext()), \
         lock(root / 'execution.lock', blocking=False):
        foundation = root / 'foundation.html'
        base = load_source(source, foundation if in_place and foundation.exists() else None,
                           integrated=workflow_timeline is not None)
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
        status = root / 'status.json'
        completed = status.exists() and read(status)['phase'] in ('complete', 'complete_with_failures')
        output = root / 'report'
        rendered = output / 'index.html'
        if completed:
            if not rendered.is_file():
                _render_saved_report(base['html'], read(root / 'investigation.json'), root)
            elif not (output / 'paper-data.json').is_file():
                write(output / 'paper-data.json', {'bundle': json.loads(BUNDLE.search(rendered.read_bytes().decode('utf-8'))[2])})
        if in_place:
            from report import strip_neuron_labels, strip_report_synthesis
            current_html = (source / 'report/index.html').read_bytes().decode('utf-8')
            current_html = strip_neuron_labels(strip_report_synthesis(current_html))
            current = current_html.encode('utf-8')
            current_bundle = BUNDLE.search(current_html)
            # A previous promotion may have replaced the costs sidecar before the HTML.
            costs = json.loads(current_bundle[2]).get('interpretation_costs') if current_bundle else None
            if (fingerprint(current) != base['files']['report/index.html'] and
                    current_html != route_overlay(base['html'], source, costs=costs)):
                page = rendered.read_bytes().decode('utf-8') if rendered.exists() else base['html']
                if current_html not in (page, route_overlay(page, source, costs=costs)):
                    legacy = legacy_scoring_view(page)
                    if current_html not in (legacy, route_overlay(legacy, source, costs=costs)):
                        raise ValueError('Current report changed outside this follow-up investigation')
            if not foundation.exists():
                temporary = foundation.with_suffix('.tmp')
                temporary.write_bytes(current)
                temporary.replace(foundation)
        write(root / 'frozen.json', {key: base[key] for key in ('initial', 'predictions', 'original_criticism')})
        timeline = FollowupTimeline(root, workflow_timeline)
        config = base['config']
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config['gpus']))
        os.environ['NEURON_MICROSCOPE_SETTINGS'] = str(root / 'backend-settings.json')
        write(root / 'backend-settings.json', config['settings'])
        try:
            if completed:
                if workflow_timeline is not None:
                    workflow_timeline.status('followup', step='restoring-report')
                    return output
                return integrate_report(output, source, in_place=in_place)
            client = client_factory(config['llm'], root / 'api', config['api_slots'])
            with executor_factory(config, root / 'measurements', timeline, options,
                                  base['excluded_documents']) as executor:
                result = rounds(client, executor, base, root, options, timeline)
            write(root / 'investigation.json', result)
            output = _render_saved_report(base['html'], result, root)
            if workflow_timeline is None:
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
