"""Run at most two autonomous follow-up rounds in a separate artifact directory."""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from .client import Client
from .state import Timeline, lock, read, write
from . import followup_protocol as protocol
from .followup_execute import Executor
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


def load_source(source):
    if read(source / 'status.json')['phase'] != 'complete':
        raise ValueError('Follow-up investigation requires a completed foundation investigation')
    files = {name: (source / name).read_bytes() for name in SOURCE_FILES}
    config, artifacts, initial, predictions, evidence = [json.loads(files[name]) for name in SOURCE_FILES[:-1]]
    if artifacts['initial'] != initial or artifacts['predictions'] != predictions:
        raise ValueError('Foundation artifacts do not match the frozen explanation and predictions')
    from .followup_report import BUNDLE
    html = files['report/index.html'].decode('utf-8')
    bundles = list(BUNDLE.finditer(html))
    if len(bundles) != 1 or json.loads(bundles[0][2])['neuron']['id'] != config['neuron']:
        raise ValueError('Foundation report and configuration must identify the same neuron')
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
            spec = client.request(f'{identity}/specification', protocol.SPEC_PROMPT,
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
            if set(output['evidence_ids']) - valid_ids:
                raise ValueError('Cite only supplied measured evidence IDs')
            if not valid_ids and output['verdict'] != 'inconclusive':
                raise ValueError('No observations: verdict must be inconclusive')
            if any(group['selected_documents'] == 0 for group in evidence.get('groups', [])) and output['verdict'] == 'survived':
                raise ValueError('An empty comparison group cannot establish survival')
            if valid_ids and not output['evidence_ids']:
                raise ValueError('Cite the evidence used for this assessment')

        record['assessment'] = client.request(f'{identity}/assessment', protocol.ASSESS_PROMPT,
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
        plan = client.request(f'round-{number}/plan', protocol.PLAN_PROMPT,
            {**context, 'round': number, 'maximum_investigations': protocol.MAX_INVESTIGATIONS,
             'evidence_catalog': previews}, protocol.plan_schema(criticism_ids, list(catalog)), 1,
            check=lambda output: protocol.check_plan(output, criticism_ids, catalog))
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


def run(source, root, options=None, *, client_factory=Client, executor_factory=Executor):
    source, root = Path(source).resolve(), Path(root).resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError('Follow-up investigation needs a separate output directory outside the foundation')
    options = {**DEFAULTS, **(options or {})}
    if set(options) != set(DEFAULTS) or any(type(value) is not int or value < (0 if key == 'seed' else 1)
                                           for key, value in options.items()):
        raise ValueError('Use a nonnegative integer seed and positive integer token/sample limits')
    base = load_source(source)
    framework = Path(__file__).resolve().parents[1]
    implementation = [framework / name for name in (
        'workflow/followups.py', 'workflow/followup_protocol.py', 'workflow/followup_execute.py',
        'workflow/followup_corpus.py', 'workflow/followup_report.py',
        'workflow/client.py', 'workflow/contracts.py', 'workflow/measure.py', 'workflow/state.py',
        'tools/common.py', 'tools/dataset_reader.py', 'tools/settings.py', 'tools/report.py')]
    manifest = {'source': str(source), 'files': base['files'], 'config': base['config'], 'options': options,
                'inputs': input_identity(base['config']),
                'rounds': protocol.ROUNDS, 'max_investigations': protocol.MAX_INVESTIGATIONS,
                'implementation': {str(path.relative_to(framework)): fingerprint(path.read_bytes())
                    for path in sorted(implementation)}}
    with lock(root / 'execution.lock', blocking=False):
        marker = root / 'manifest.json'
        if marker.exists():
            if read(marker) != manifest:
                raise ValueError('Source, settings or implementation changed; use a new output directory')
        elif any(path.name != 'execution.lock' for path in root.iterdir()):
            raise ValueError('Follow-up output must be empty or belong to this same investigation')
        else:
            write(marker, manifest)
        frozen = {key: base[key] for key in ('initial', 'predictions', 'original_criticism')}
        snapshot = root / 'frozen.json'
        if not snapshot.exists():
            write(snapshot, frozen)
        elif read(snapshot) != frozen:
            raise ValueError('Frozen baseline snapshot changed; use a new output directory')
        if (root / 'status.json').exists() and read(root / 'status.json')['phase'] in ('complete', 'complete_with_failures'):
            return root / 'report'
        timeline = Timeline(root)
        config = base['config']
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config['gpus']))
        os.environ['NEURON_MICROSCOPE_SETTINGS'] = str(root / 'backend-settings.json')
        write(root / 'backend-settings.json', config['settings'])
        try:
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
